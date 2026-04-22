"""
Self-consistency voting across the K samples produced per drug.

Mirrors the logic of `CellHit.LLMs.utils.self_consistency` from the original
notebook but uses the (target, rationale) tuple schema returned by
`parser.extract_targets`.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


def vote_targets(runs: Iterable[list[tuple[str, str]]]) -> dict[str, dict]:
    """
    Combine multiple sampled predictions into a vote-count dict.

    Parameters
    ----------
    runs : iterable of lists of (target, rationale)
        One list per sampled generation.

    Returns
    -------
    dict[target] -> {"count": int, "rationales": list[str]}
    """
    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "rationales": []})
    for run in runs:
        if not run:
            continue
        seen_in_run: set[str] = set()
        for target, rationale in run:
            key = target.upper()          # case-insensitive deduping
            if key in seen_in_run:
                continue                   # a single sample cannot vote twice
            seen_in_run.add(key)
            agg[key]["count"] += 1
            if rationale:
                agg[key]["rationales"].append(rationale)
    return dict(agg)


def filter_by_votes(
    votes: Mapping[str, dict],
    min_votes: int,
    allowed: set[str] | None = None,
) -> dict[str, dict]:
    """
    Keep only targets with >= min_votes and, optionally, that appear in the
    `allowed` gene-symbol set (the active STRING proteins).
    """
    allowed_upper = {a.upper() for a in allowed} if allowed else None
    out = {}
    for t, v in votes.items():
        if v["count"] < min_votes:
            continue
        if allowed_upper is not None and t not in allowed_upper:
            continue  # hallucination — LLM proposed a gene not in the STRING list
        out[t] = v
    return out


def map_genes_to_protein_ids(
    gene_votes: Mapping[str, dict],
    gene_to_pid: Mapping[str, str],
) -> list[dict]:
    """
    Flatten the vote dict into a list of records suitable for a CSV row.
    Returns entries even if a gene has no known ProteinID (pid=None).
    """
    records = []
    for gene, v in sorted(gene_votes.items(), key=lambda kv: -kv[1]["count"]):
        records.append({
            "gene_symbol": gene,
            "protein_id": gene_to_pid.get(gene.upper()),
            "votes": v["count"],
            "rationales": " | ".join(v["rationales"][:3]),  # cap for CSV readability
        })
    return records
