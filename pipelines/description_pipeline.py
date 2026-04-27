"""
Sub-pipeline 1 — Refined drug description.

Flow for each drug:

    PubMed abstracts (concurrent I/O)
           │
           ▼
    LLM call 1 : initial description from {name, SMILES, InChIKey}
           │
           ▼
    LLM call 2 : refine description using the abstracts as evidence
           │
           ▼
    append (drug_name, refined_description) to the checkpoint CSV

LLM calls are issued *per mini-batch* but each mini-batch is sent to vLLM
as a single list of prompts, so vLLM's continuous batching fully utilises
the 4×A100 tensor-parallel group.
"""

from __future__ import annotations

import logging

import pandas as pd

import config
from utils.abstracts import batch_fetch_by_cid, batch_fetch_by_name
from utils.checkpoint import append_checkpoint, load_processed
from utils.llm_backend import SamplingParams
from utils.prompts import render_prompt

log = logging.getLogger(__name__)


# ==========================================================================
# Helpers
# ==========================================================================
def _format_abstracts(abstracts: list[str], limit_chars: int = 6000) -> str:
    """Join abstracts, truncating so the full prompt stays within context."""
    joined = "\n\n---\n\n".join(f"[{i + 1}] {a}" for i, a in enumerate(abstracts))
    if len(joined) > limit_chars:
        joined = joined[:limit_chars] + " …[truncated]"
    return joined


def _vllm_generate(
    llm, prompts: list[str], sampling_params: SamplingParams
) -> list[str]:
    """Run generation and return first-sample text for each prompt.

    Works for both the vLLM engine and the TogetherBackend — both return a
    list of objects whose `.outputs[0].text` is the first sample.
    """
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    return [o.outputs[0].text for o in outputs]


# ==========================================================================
# Core stages
# ==========================================================================
def _run_llm_two_stage(
    llm,
    rows: list[dict],  # dicts with drug_name, smiles, inchikey, cid, abstracts
) -> list[dict]:
    """Run initial description + refinement as two batched vLLM calls."""
    if not rows:
        return []

    # --- Stage 1: initial description -------------------------------------
    desc_prompts = [
        render_prompt(
            config.DESCRIPTION_PROMPT,
            drug_name=r["drug_name"],
            cid=r.get("cid", "unknown"),
            smiles=r.get("smiles", "unknown"),
            inchikey=r.get("inchikey", "unknown"),
        )
        for r in rows
    ]
    desc_sp = SamplingParams(
        temperature=config.DESC_TEMPERATURE,
        top_p=config.DESC_TOP_P,
        max_tokens=config.DESC_MAX_TOKENS,
        seed=config.SEED,
    )
    initial_descs = _vllm_generate(llm, desc_prompts, desc_sp)

    # --- Stage 2: refine with abstracts -----------------------------------
    refine_prompts = [
        render_prompt(
            config.REFINER_PROMPT,
            drug_name=r["drug_name"],
            initial_description=init.strip(),
            abstracts=_format_abstracts(r["abstracts"]),
        )
        for r, init in zip(rows, initial_descs)
    ]
    refine_sp = SamplingParams(
        temperature=config.REFINE_TEMPERATURE,
        top_p=config.REFINE_TOP_P,
        max_tokens=config.REFINE_MAX_TOKENS,
        seed=config.SEED,
    )
    refined = _vllm_generate(llm, refine_prompts, refine_sp)

    return [
        {
            "drug_name": r["drug_name"],
            "cid": r.get("cid"),
            "initial_description": init.strip(),
            "refined_description": ref.strip(),
            "n_abstracts": len(r["abstracts"]),
            "abstracts": r["abstracts"],  # native list[str]
        }
        for r, init, ref in zip(rows, initial_descs, refined)
    ]


# ==========================================================================
# Public entry points
# ==========================================================================
def run_description_pipeline(
    llm,
    drugs_df: pd.DataFrame,
    *,
    batch_size: int | None = None,
    use_cid_fallback: bool = False,
) -> list[str]:
    """
    Produce refined descriptions for every row of `drugs_df`.

    Parameters
    ----------
    llm : vllm.LLM | utils.llm_backend.TogetherBackend
        Pre-loaded LLM backend exposing `.generate(prompts, SamplingParams)`.
    drugs_df : DataFrame with columns [drug_name, cid, smiles, inchikey]
    use_cid_fallback : bool
        If True, use `batch_fetch_by_cid` for abstract retrieval (used on
        the second pass, for drugs that failed the name-based lookup).

    Returns
    -------
    failed : list[str]
        Drug names whose abstracts could not be retrieved in this pass.
    """
    batch_size = batch_size or config.BATCH_SIZE
    fetcher = batch_fetch_by_cid if use_cid_fallback else batch_fetch_by_name

    # Resume: skip drugs we already have descriptions for
    done_desc = load_processed(config.REFINED_DESC_JSON) if config.RESUME else set()
    todo = drugs_df[~drugs_df["drug_name"].isin(done_desc)].reset_index(drop=True)
    if todo.empty:
        log.info("No drugs left for description pipeline (all cached).")
        return []

    log.info(
        "Description pipeline: %d drugs to process (use_cid_fallback=%s)",
        len(todo),
        use_cid_fallback,
    )

    all_failed: list[str] = []

    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start : start + batch_size]
        log.info("  -> batch %d-%d / %d", start, start + len(batch), len(todo))

        # 1. Fetch abstracts (I/O bound, parallel)
        abstracts_dict, failed = fetcher(
            drugs=list(zip(batch["drug_name"], batch.get("cid", [None] * len(batch)))),
            k=config.ABSTRACTS_PER_DRUG,
            email=config.PUBMED_EMAIL,
            api_key=config.PUBMED_API_KEY,
            max_workers=config.ENTREZ_MAX_WORKERS,
        )
        all_failed.extend(failed)
        log.info("     abstracts: %d ok, %d failed", len(abstracts_dict), len(failed))

        # 2. Build per-drug rows for the drugs that came back with abstracts
        rows = []
        for _, r in batch.iterrows():
            if r["drug_name"] not in abstracts_dict:
                continue
            rows.append(
                {
                    "drug_name": r["drug_name"],
                    "cid": r.get("cid", None),
                    "smiles": r.get("smiles", "unknown"),
                    "inchikey": r.get("inchikey", "unknown"),
                    "abstracts": abstracts_dict[r["drug_name"]],
                }
            )

        if not rows:
            continue

        # 3. Two-stage batched LLM call
        out_records = _run_llm_two_stage(llm, rows)

        # 4. Checkpoint
        append_checkpoint(out_records, config.REFINED_DESC_JSON, key_col="drug_name")
        log.info(
            "     wrote %d refined descriptions to %s",
            len(out_records),
            config.REFINED_DESC_JSON.name,
        )

    return all_failed
