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
    python main.py --batch-size 16             # override batch size (default=1)
    python main.py --model-path /path/to/model  # override model path (default from config)
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import pandas as pd
from tqdm.auto import tqdm

# vLLM is heavy — import lazily inside main() so `--help` etc stay snappy.
import config
from pipelines.description_pipeline import run_description_pipeline
from pipelines.target_pipeline import run_target_pipeline
from utils.list import get_batches
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
def _build_llm(tensor_parallel_size: int, model_path: str):
    """
    Build the LLM backend configured in config.LLM_BACKEND.

    Returns an object that exposes `.generate(prompts, sampling_params,
    use_tqdm=True)` with the same output shape as vLLM. The pipelines below
    don't need to know which backend is in use.
    """
    from utils.llm_backend import build_backend

    log = logging.getLogger(__name__)
    log.info("Building LLM backend: %s", config.LLM_BACKEND)

    return build_backend(
        kind=config.LLM_BACKEND,
        # vLLM-only kwargs
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        model_revision=config.MODEL_REVISION,
        quantization=config.QUANTIZATION,
        dtype=config.DTYPE,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        enforce_eager=config.ENFORCE_EAGER,
        # Together-only kwargs
        together_model=config.TOGETHER_MODEL,
        together_api_key=config.TOGETHER_API_KEY,
        together_max_workers=config.TOGETHER_MAX_WORKERS,
        together_request_timeout=config.TOGETHER_REQUEST_TIMEOUT,
        together_max_retries=config.TOGETHER_MAX_RETRIES,
        together_initial_backoff=config.TOGETHER_INITIAL_BACKOFF,
        together_max_backoff=config.TOGETHER_MAX_BACKOFF,
    )


# ==========================================================================
# Orchestration
# ==========================================================================
def run_all(tensor_parallel_size: int, stage: str = "all", batch_size: int = 1, model_path: str = config.MODEL_PATH) -> None:
    log = logging.getLogger(__name__)
    drugs_df, proteins_df, gt_df = _load_inputs()

    gt_names = set(gt_df["Drug Name"].astype(str))
    gt_rows = drugs_df[drugs_df["drug_name"].isin(gt_names)].copy()
    all_rest_rows = drugs_df[~drugs_df["drug_name"].isin(gt_names)].copy()
    all_rest_drugs = set(all_rest_rows["drug_name"])


    log.info(
        "Loaded: %d drugs | %d proteins | %d ground-truth pairs",
        len(drugs_df),
        len(proteins_df),
        len(gt_df),
    )
    log.info("   -> %d GT drugs will go through description only", len(gt_rows))
    log.info("   -> %d remaining drugs will go through both stages", len(all_rest_rows))

    llm = _build_llm(tensor_parallel_size, model_path)

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
    n_batches = math.ceil(len(all_rest_drugs) / batch_size) if batch_size else 0
    batch_pbar = tqdm(
        get_batches(all_rest_drugs, batch_size),
        total=n_batches,
        desc="STEP 2 batches",
        unit="batch",
    )
    for batch_idx, sub_drugs in enumerate(batch_pbar, start=1):
        batch_pbar.set_postfix(drugs=len(sub_drugs), stage=stage)
        log.info(
            "STEP 2: batch %d/%d (%d drugs, stage=%s)",
            batch_idx,
            n_batches,
            len(sub_drugs),
            stage,
        )
        rest_rows = all_rest_rows[all_rest_rows["drug_name"].isin(sub_drugs)].copy()

        sub_failed_rest: list[str] = []
        if stage in ("all", "descriptions"):
            log.info("=" * 70)
            log.info("STEP 2a: refined descriptions for remaining drugs")
            log.info("=" * 70)
            sub_failed_rest = run_description_pipeline(llm, rest_rows, use_cid_fallback=False)

        if stage in ("all", "targets"):
            log.info("=" * 70)
            log.info("STEP 2b: target prediction for remaining drugs")
            log.info("=" * 70)
            refined_desc_df = load_df(config.REFINED_DESC_JSON)

            current_drugs = rest_rows["drug_name"].unique().tolist()
            gt_drugs = gt_df["Drug Name"].unique().tolist()
            all_drugs  = current_drugs + gt_drugs
            refined_desc_df = refined_desc_df[refined_desc_df["drug_name"].isin(all_drugs)].reset_index(drop=True)
            if refined_desc_df.empty:
                log.warning(
                    "No refined descriptions on disk - run --stage descriptions first"
                )
            else:
                run_target_pipeline(llm, refined_desc_df, gt_df, proteins_df)
        failed_rest.extend(sub_failed_rest)
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
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=config.MODEL_PATH,
        help="Path to the vLLM model (only used when LLM_BACKEND=='vllm').",
    )
    parser.add_argument(
        "--backend",
        choices=["together", "vllm"],
        default=config.LLM_BACKEND,
        help="Which LLM backend to use. Overrides config.LLM_BACKEND.",
    )
    args = parser.parse_args()

    # Let the CLI flag override the config default.
    config.LLM_BACKEND = args.backend

    _setup_logging()
    run_all(
        tensor_parallel_size=args.tensor_parallel_size,
        stage=args.stage,
        batch_size=args.batch_size,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
