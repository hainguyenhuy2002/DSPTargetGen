"""
Orchestrator for the Drug-Target LLM pipeline.

Execution order — exactly as specified:

  STEP 1  Generate refined descriptions for the 10 ground-truth drugs.
          (These drugs do NOT go through the target-prediction pipeline.)

  STEP 2  For every remaining drug:
              a) refined-description pipeline (batched with checkpoints)
              b) target-prediction pipeline   (few-shot = ground-truth pool)

  STEP 3  Retry the drugs that failed PubMed-by-name with the CID fallback.
          Drugs recovered at this stage then go through both pipelines.

Each stage writes checkpoint CSVs after every mini-batch, so the run can
be resumed cheaply after any interruption.

Usage
-----
    python main.py                            # full run
    python main.py --stage descriptions       # only refined descriptions
    python main.py --stage targets            # only target prediction
    python main.py --tensor-parallel-size 4   # override TP (default from config)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys

import pandas as pd

# vLLM is heavy — import lazily inside main() so `--help` etc stay snappy.
import config
from pipelines.description_pipeline import run_description_pipeline
from pipelines.target_pipeline import run_target_pipeline
from utils.checkpoint import append_checkpoint, load_df


# ==========================================================================
# Logging setup
# ==========================================================================
def _setup_logging() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(config.LOG_DIR / "pipeline.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ==========================================================================
# Data loading / normalisation
# ==========================================================================
def _standardise_drugs_info(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns of drugs_info.csv to the names the pipeline expects."""
    rename_map = {}
    for col in df.columns:
        c = col.strip().lower()
        if c in {"drug name", "drug_name", "drug"}:
            rename_map[col] = "drug_name"
        elif c in {"cid", "pubchem_cid", "pubchemcid"}:
            rename_map[col] = "cid"
        elif c in {"smiles"}:
            rename_map[col] = "smiles"
        elif c in {"inchikey", "inchi key", "inchi_key"}:
            rename_map[col] = "inchikey"
    df = df.rename(columns=rename_map)

    for need in ("drug_name", "cid", "smiles", "inchikey"):
        if need not in df.columns:
            df[need] = None

    # drop rows with no drug name
    df = df.dropna(subset=["drug_name"]).reset_index(drop=True)
    return df


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drugs_df = pd.read_csv(config.DRUGS_INFO_CSV)
    proteins_df = pd.read_csv(config.PROTEINS_INFO_CSV)
    gt_df = pd.read_csv(config.GROUND_TRUTH_CSV)

    drugs_df = _standardise_drugs_info(drugs_df)

    # make sure protein columns are the ones the target pipeline expects
    for need in ("Protein ID", "Protein Name", "Coded Gene"):
        if need not in proteins_df.columns:
            raise ValueError(f"proteins_info.csv must contain column {need!r}")
    for need in ("Drug Name", "Proteins"):
        if need not in gt_df.columns:
            raise ValueError(
                f"elite_drug_target_groundtruth.csv must contain column {need!r}"
            )

    return drugs_df, proteins_df, gt_df


# ==========================================================================
# Model
# ==========================================================================
def _build_llm(tensor_parallel_size: int):
    """Load the quantised Mixtral once, sharded across all 4 GPUs."""
    from vllm import LLM

    log = logging.getLogger(__name__)
    log.info(
        "Loading vLLM model from %s (TP=%d)", config.MODEL_PATH, tensor_parallel_size
    )
    return LLM(
        model=config.MODEL_PATH,
        revision=config.MODEL_REVISION,
        quantization=config.QUANTIZATION,
        dtype=config.DTYPE,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=config.MAX_MODEL_LEN,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        trust_remote_code=config.TRUST_REMOTE_CODE,
        enforce_eager=config.ENFORCE_EAGER,
        seed=config.SEED,
    )


# ==========================================================================
# Data-parallel sharding
# ==========================================================================
def _stable_hash(s: str) -> int:
    """Deterministic integer hash, stable across processes and Python runs."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _shard_rows(
    df: pd.DataFrame, worker_id: int, num_workers: int, key_col: str = "drug_name"
) -> pd.DataFrame:
    """Keep only the rows whose stable_hash(key) % num_workers == worker_id."""
    if num_workers <= 1:
        return df
    mask = (
        df[key_col]
        .astype(str)
        .map(lambda n: _stable_hash(n) % num_workers == worker_id)
    )
    return df[mask].reset_index(drop=True)


def _apply_worker_suffix(worker_id: int, num_workers: int) -> None:
    """
    Rewrite the output paths on the config module so each data-parallel worker
    writes to its own `*.w{id}.json` file. A no-op when running single-worker.
    """
    if num_workers <= 1:
        return
    suffix = f".w{worker_id}.json"
    config.REFINED_DESC_JSON = config.REFINED_DESC_JSON.with_suffix(suffix)
    config.PREDICTED_TARGETS_JSON = config.PREDICTED_TARGETS_JSON.with_suffix(suffix)
    config.FAILED_ABSTRACTS_JSON = config.FAILED_ABSTRACTS_JSON.with_suffix(suffix)


def _throttle_entrez(num_workers: int) -> None:
    """
    NCBI rate-limits per IP (3 req/s w/o API key, 10 w/ key). With N parallel
    workers, per-worker thread count must be divided by N so total concurrent
    requests stay bounded.
    """
    if num_workers <= 1:
        return
    shared = max(1, config.ENTREZ_MAX_WORKERS // num_workers)
    config.ENTREZ_MAX_WORKERS = shared


# ==========================================================================
# Orchestration
# ==========================================================================
def run_all(
    tensor_parallel_size: int,
    stage: str = "all",
    worker_id: int = 0,
    num_workers: int = 1,
) -> None:
    log = logging.getLogger(__name__)

    # Per-worker output paths + throttled NCBI concurrency (both no-ops if N=1)
    _apply_worker_suffix(worker_id, num_workers)
    _throttle_entrez(num_workers)

    drugs_df, proteins_df, gt_df = _load_inputs()

    gt_names = set(gt_df["Drug Name"].astype(str))
    gt_rows = drugs_df[drugs_df["drug_name"].isin(gt_names)].copy()
    rest_rows = drugs_df[~drugs_df["drug_name"].isin(gt_names)].copy()

    # GT drugs are cheap (10 of them) — every worker processes them into its
    # own per-worker file, merge at the end picks one copy. This keeps the
    # few-shot pool locally available to each worker without cross-worker IPC.
    #
    # Non-GT drugs are sharded so each drug is processed exactly once.
    rest_rows = _shard_rows(rest_rows, worker_id, num_workers)

    log.info(
        "Worker %d/%d - loaded: %d drugs | %d proteins | %d GT pairs",
        worker_id,
        num_workers,
        len(drugs_df),
        len(proteins_df),
        len(gt_df),
    )
    log.info("   -> %d GT drugs (all workers)", len(gt_rows))
    log.info("   -> %d non-GT drugs on this worker", len(rest_rows))
    log.info("   -> outputs: %s", config.REFINED_DESC_JSON.name)

    llm = _build_llm(tensor_parallel_size)

    # ------------------------------------------------------------------
    # STEP 1 — refined descriptions for ground-truth drugs
    # ------------------------------------------------------------------
    failed_gt: list[str] = []
    if stage in ("all", "descriptions"):
        log.info("=" * 70)
        log.info("STEP 1: refined descriptions for ground-truth drugs")
        log.info("=" * 70)
        failed_gt = run_description_pipeline(llm, gt_rows, use_cid_fallback=False)

    # ------------------------------------------------------------------
    # STEP 2 — descriptions + targets for the remaining drugs
    # ------------------------------------------------------------------
    failed_rest: list[str] = []
    if stage in ("all", "descriptions"):
        log.info("=" * 70)
        log.info("STEP 2a: refined descriptions for remaining drugs")
        log.info("=" * 70)
        failed_rest = run_description_pipeline(llm, rest_rows, use_cid_fallback=False)

    if stage in ("all", "targets"):
        log.info("=" * 70)
        log.info("STEP 2b: target prediction for remaining drugs")
        log.info("=" * 70)
        refined_desc_df = load_df(config.REFINED_DESC_JSON)
        if refined_desc_df.empty:
            log.warning(
                "No refined descriptions on disk - run --stage descriptions first"
            )
        else:
            run_target_pipeline(llm, refined_desc_df, gt_df, proteins_df)

    # ------------------------------------------------------------------
    # STEP 3 — CID-fallback retry for everything that failed PubMed-by-name
    # ------------------------------------------------------------------
    failed = sorted(set(failed_gt + failed_rest))
    if failed and stage in ("all", "descriptions"):
        log.info("=" * 70)
        log.info("STEP 3: CID-fallback retry for %d failed drugs", len(failed))
        log.info("=" * 70)

        # persist the failed list so it survives restarts
        append_checkpoint(
            [{"drug_name": n, "reason": "pubmed_name_miss"} for n in failed],
            config.FAILED_ABSTRACTS_JSON,
            key_col="drug_name",
        )

        retry_df = drugs_df[drugs_df["drug_name"].isin(failed)].copy()
        still_failed = run_description_pipeline(llm, retry_df, use_cid_fallback=True)

        # drugs recovered by CID fallback also need target prediction
        if stage in ("all", "targets") and len(still_failed) < len(failed):
            refined_desc_df = load_df(config.REFINED_DESC_JSON)
            run_target_pipeline(llm, refined_desc_df, gt_df, proteins_df)

        if still_failed:
            log.warning(
                "%d drugs could not be recovered even via CID: %s",
                len(still_failed),
                still_failed[:10],
            )
            append_checkpoint(
                [{"drug_name": n, "reason": "pubmed_cid_miss"} for n in still_failed],
                config.FAILED_ABSTRACTS_JSON,
                key_col="drug_name",
            )

    log.info("All stages complete. Outputs in %s", config.OUTPUT_DIR)


# ==========================================================================
# CLI
# ==========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage",
        choices=["all", "descriptions", "targets"],
        default="all",
        help="Which stage(s) to run.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=config.TENSOR_PARALLEL_SIZE,
        help="vLLM tensor_parallel_size.",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Data-parallel worker index (0-based). Ignored when --num-workers=1.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of data-parallel workers running concurrently. "
        "Each worker processes a stable hash-sharded subset of the drugs and "
        "writes its outputs to `<name>.w{id}.json`. "
        "Use `merge_outputs.py` at the end to combine them.",
    )
    args = parser.parse_args()

    if not (0 <= args.worker_id < args.num_workers):
        parser.error(
            f"--worker-id {args.worker_id} out of range [0, {args.num_workers})"
        )

    _setup_logging()
    run_all(
        tensor_parallel_size=args.tensor_parallel_size,
        stage=args.stage,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
