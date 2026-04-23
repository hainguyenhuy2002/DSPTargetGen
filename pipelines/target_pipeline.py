"""
Sub-pipeline 2 — Target prediction with few-shot learning and self-consistency.

For every drug that (a) has a refined description and (b) is NOT in the
ground-truth set:

  1. Sample N ground-truth drugs as few-shot examples (with their refined
     descriptions and their known protein targets).
  2. Inject the full list of active proteins (gene symbol + protein name)
     as the candidate pool the model must choose from.
  3. Call vLLM with `n = SELF_CONSISTENCY_K` to get K samples in a single
     forward pass (vLLM reuses the prefix cache across the K samples, so
     this is dramatically cheaper than K independent `generate()` calls).
  4. Vote across the K samples, filter out targets not in the active
     protein set, map gene symbols back to ProteinIDs, and checkpoint.
"""

from __future__ import annotations

import ast
import json
import logging
import random
from typing import Iterable

import pandas as pd

import config
from utils.aggregator import filter_by_votes, map_genes_to_protein_ids, vote_targets
from utils.checkpoint import append_checkpoint, load_processed
from utils.llm_backend import SamplingParams
from utils.parser import extract_targets, parse_json
from utils.prompts import render_prompt

log = logging.getLogger(__name__)


# ==========================================================================
# Protein list & gene-symbol mappings
# ==========================================================================
def build_protein_context(
    proteins_df: pd.DataFrame,
) -> tuple[str, dict[str, str], set[str]]:
    """
    Compose the 'Available proteins' block injected into every target prompt,
    plus the lookup structures used to validate LLM output.

    Returns
    -------
    proteins_block : str
        Newline-separated 'GENE | Protein Name' entries.
    gene_to_pid : dict
        Upper-cased gene symbol -> protein ID.
    allowed_genes : set
        Upper-cased gene symbols that are valid LLM outputs.
    """
    df = proteins_df.dropna(subset=["Coded Gene"]).copy()
    df["Coded Gene"] = df["Coded Gene"].astype(str).str.strip()
    df = df[df["Coded Gene"] != ""]

    # gene -> first ProteinID seen
    gene_to_pid: dict[str, str] = {}
    for _, r in df.iterrows():
        g = str(r["Coded Gene"]).upper()
        if g not in gene_to_pid:
            gene_to_pid[g] = str(r["Protein ID"])

    # compact block for the prompt
    lines = [
        f"{r['Coded Gene']} | {r['Protein Name']}"
        for _, r in df.drop_duplicates(subset=["Coded Gene"]).iterrows()
    ]
    proteins_block = "\n".join(lines)
    allowed_genes = set(gene_to_pid.keys())

    log.info("Active-protein pool: %d unique gene symbols", len(allowed_genes))
    return proteins_block, gene_to_pid, allowed_genes


# ==========================================================================
# Few-shot pool construction
# ==========================================================================
def _protein_ids_to_genes(
    protein_ids: Iterable[str], pid_to_gene: dict[str, str]
) -> list[str]:
    out = []
    for pid in protein_ids:
        g = pid_to_gene.get(str(pid).strip())
        if g:
            out.append(g)
    return out


def _parse_protein_list(raw) -> list[str]:
    """Ground-truth 'Proteins' column may be a python-literal list string."""
    if isinstance(raw, list):
        return [str(x).strip() for x in raw]
    if pd.isna(raw):
        return []
    s = str(raw).strip()
    try:
        parsed = ast.literal_eval(s)
        return [str(x).strip() for x in parsed]
    except (ValueError, SyntaxError):
        # fall back to naive comma split
        return [t.strip(" []'\"") for t in s.split(",")]


def build_fewshot_pool(
    ground_truth_df: pd.DataFrame,
    refined_desc_df: pd.DataFrame,
    proteins_df: pd.DataFrame,
) -> list[dict]:
    """
    Build the pool of few-shot examples. Each entry contains the drug name,
    its refined description, and the list of gene-symbol targets.
    """
    pid_to_gene = {
        str(r["Protein ID"]): str(r["Coded Gene"]).upper()
        for _, r in proteins_df.dropna(subset=["Coded Gene"]).iterrows()
    }

    desc_by_drug = {
        r["drug_name"]: r["refined_description"] for _, r in refined_desc_df.iterrows()
    }

    pool: list[dict] = []
    for _, r in ground_truth_df.iterrows():
        name = r["Drug Name"]
        if name not in desc_by_drug:
            continue  # no refined description yet — skip
        gene_targets = _protein_ids_to_genes(
            _parse_protein_list(r["Proteins"]), pid_to_gene
        )
        if not gene_targets:
            continue
        pool.append(
            {
                "drug_name": name,
                "description": desc_by_drug[name],
                "targets": gene_targets,
            }
        )
    log.info("Few-shot pool: %d usable ground-truth drugs", len(pool))
    return pool


def format_fewshot(examples: list[dict], max_desc_chars: int = 1200) -> str:
    """Render few-shot examples as a single block for the target prompt."""
    blocks = []
    for i, ex in enumerate(examples, 1):
        desc = ex["description"]
        if len(desc) > max_desc_chars:
            desc = desc[:max_desc_chars] + " …[truncated]"
        # We render the example output in the same JSON schema the model
        # must follow so it can imitate.
        json_targets = {f"target_{k + 1}": g for k, g in enumerate(ex["targets"])}
        for k, g in enumerate(ex["targets"]):
            json_targets[f"rationale_{k + 1}"] = (
                f"{g} is a known target based on curated drug-target annotations."
            )
        blocks.append(
            f"--- Example {i} ---\n"
            f"Drug Name : {ex['drug_name']}\n"
            f"Refined description :\n{desc}\n"
            f"Expected output :\n```json\n{json.dumps(json_targets, indent=2)}\n```"
        )
    return "\n\n".join(blocks)


# ==========================================================================
# Main entry point
# ==========================================================================
def run_target_pipeline(
    llm,
    refined_desc_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    proteins_df: pd.DataFrame,
    *,
    batch_size: int | None = None,
    k_samples: int | None = None,
    n_fewshot: int | None = None,
) -> None:
    """
    Predict targets for every drug that has a refined description but is not
    in the ground-truth set.
    """
    batch_size = batch_size or config.BATCH_SIZE
    k_samples = k_samples or config.SELF_CONSISTENCY_K
    n_fewshot = n_fewshot or config.NUM_FEWSHOT_EXAMPLES

    proteins_block, gene_to_pid, allowed_genes = build_protein_context(proteins_df)
    fewshot_pool = build_fewshot_pool(ground_truth_df, refined_desc_df, proteins_df)
    if not fewshot_pool:
        raise RuntimeError(
            "Few-shot pool is empty - make sure the description pipeline has "
            "already run on the ground-truth drugs."
        )
    if n_fewshot > len(fewshot_pool):
        log.warning(
            "n_fewshot=%d > pool size %d; clamping.", n_fewshot, len(fewshot_pool)
        )
        n_fewshot = len(fewshot_pool)

    # Resume support
    done_targets = (
        load_processed(config.PREDICTED_TARGETS_JSON) if config.RESUME else set()
    )
    gt_names = set(ground_truth_df["Drug Name"].astype(str))
    todo = refined_desc_df[
        ~refined_desc_df["drug_name"].isin(gt_names | done_targets)
    ].reset_index(drop=True)

    if todo.empty:
        log.info("No drugs left for target pipeline.")
        return

    log.info(
        "Target pipeline: %d drugs to predict, K=%d samples each", len(todo), k_samples
    )

    sp = SamplingParams(
        temperature=config.TARGET_TEMPERATURE,
        top_p=config.TARGET_TOP_P,
        max_tokens=config.TARGET_MAX_TOKENS,
        n=k_samples,  # <-- key optimisation: K samples / prompt
        seed=config.SEED,
    )

    rng = random.Random(config.SEED)

    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start : start + batch_size]
        log.info("  -> batch %d-%d / %d", start, start + len(batch), len(todo))

        prompts, names = [], []
        for _, r in batch.iterrows():
            fs = rng.sample(fewshot_pool, n_fewshot)
            prompts.append(
                render_prompt(
                    config.TARGET_PROMPT,
                    drug_name=r["drug_name"],
                    refined_description=r["refined_description"],
                    fewshot_examples=format_fewshot(fs),
                    proteins_block=proteins_block,
                )
            )
            names.append(r["drug_name"])

        outputs = llm.generate(prompts, sp, use_tqdm=True)

        records = []
        for name, req_out in zip(names, outputs):
            # req_out.outputs is a list of length K (one per sample)
            runs = [
                extract_targets(parse_json(sample.text)) for sample in req_out.outputs
            ]

            votes = vote_targets(runs)
            filtered = filter_by_votes(
                votes, min_votes=config.MIN_VOTES, allowed=allowed_genes
            )
            ranked = map_genes_to_protein_ids(filtered, gene_to_pid)

            records.append(
                {
                    "drug_name": name,
                    "targets": ranked,  # list[{"gene_symbol","protein_id","votes","rationales"}]
                    "top_genes": [r["gene_symbol"] for r in ranked],  # list[str]
                    "top_pids": [r["protein_id"] for r in ranked if r["protein_id"]],
                    "n_runs": k_samples,
                    "raw_votes": {
                        g: v["count"] for g, v in votes.items()
                    },  # dict[str, int]
                }
            )

        append_checkpoint(records, config.PREDICTED_TARGETS_JSON, key_col="drug_name")
        log.info("     wrote %d target predictions", len(records))
