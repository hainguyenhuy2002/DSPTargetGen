"""
Merge per-worker checkpoint files into the final single-file outputs.

Each data-parallel worker writes to `<name>.w{id}.json`. After all workers
exit successfully, run this script to combine them:

    python merge_outputs.py --num-workers 5 [--cleanup]

The merge reuses `append_checkpoint` from `utils.checkpoint`, so dedup-by-
drug_name is handled for free and nested JSON structures stay native.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import config
from utils.checkpoint import append_checkpoint

log = logging.getLogger(__name__)


def _read_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _worker_paths(base: Path, num_workers: int) -> list[Path]:
    """
    `refined_descriptions.json` -> [refined_descriptions.w0.json, ...].
    Uses the same convention as `main._apply_worker_suffix`.
    """
    parent, stem = base.parent, base.stem   # stem drops the .json
    return [parent / f"{stem}.w{i}.json" for i in range(num_workers)]


def merge_one(base: Path, num_workers: int, cleanup: bool = False) -> int:
    """
    Merge `base.w0.json … base.w{N-1}.json` into `base`.
    Returns the number of records in the merged file.
    """
    shards = [p for p in _worker_paths(base, num_workers) if p.exists()]
    if not shards:
        log.warning("No worker shards found for %s - skipping", base.name)
        return 0

    # Collect everything, then write via append_checkpoint to get dedup-by-drug
    all_records: list[dict] = []
    for shard in shards:
        recs = _read_json_list(shard)
        log.info("  shard %-40s %6d records", shard.name, len(recs))
        all_records.extend(recs)

    # Start fresh — remove any stale merged file so append_checkpoint writes
    # a clean, de-duplicated result.
    if base.exists():
        base.unlink()
    append_checkpoint(all_records, base, key_col="drug_name")

    merged = _read_json_list(base)
    log.info("  -> merged %d records into %s", len(merged), base.name)

    if cleanup:
        for shard in shards:
            shard.unlink()
            log.info("     deleted %s", shard.name)

    return len(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-workers", type=int, required=True,
                        help="Number of data-parallel workers that were launched.")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete the per-worker shard files after merging.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    for base in (config.REFINED_DESC_JSON, config.PREDICTED_TARGETS_JSON, config.FAILED_ABSTRACTS_JSON):
        log.info("Merging %s", base.name)
        merge_one(base, args.num_workers, cleanup=args.cleanup)

    log.info("Done. Outputs in %s", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
