"""
Crash-safe JSON checkpointing.

Pipelines call `append_checkpoint(records, path, key_col)` after every
mini-batch. Writes are atomic (write to `.tmp`, then `os.replace`) so an
interrupted run never corrupts the file. `load_processed(path, key_col)`
lets the orchestrator skip drugs already on disk.

On-disk format
--------------
A single JSON **array of record objects** — the same shape you get from
`pandas.DataFrame.to_dict(orient="records")`. Example:

    [
      {"drug_name": "Docetaxel", "refined_description": "...", ...},
      {"drug_name": "Cisplatin", "refined_description": "...", ...}
    ]

Fields that are themselves structured (lists, dicts) are stored as native
JSON — no `json.dumps()` wrapping required — so you can `jq` into them
directly.
"""
from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)


# ==========================================================================
# Low-level I/O
# ==========================================================================
def _atomic_write_json(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning("Could not read checkpoint %s: %s - treating as empty", path, e)
        return []
    if not isinstance(data, list):
        log.warning("Checkpoint %s is not a JSON array - ignoring", path)
        return []
    return data


# ==========================================================================
# Public API
# ==========================================================================
def append_checkpoint(
    records: Iterable[dict],
    path: Path,
    key_col: str = "drug_name",
) -> None:
    """
    Merge `records` into `path` (JSON array). De-duplicates on `key_col`,
    keeping the last write, so re-running a drug overwrites its previous row.
    """
    records = list(records)
    if not records:
        return

    existing = _read_json_list(path)

    # Dedupe-by-key, last write wins
    by_key: dict = {}
    for rec in existing:
        k = rec.get(key_col)
        if k is not None:
            by_key[k] = rec
    for rec in records:
        k = rec.get(key_col)
        if k is not None:
            by_key[k] = rec

    _atomic_write_json(list(by_key.values()), path)


def load_processed(path: Path, key_col: str = "drug_name") -> set[str]:
    """Return the set of values already present under `key_col` in `path`."""
    return {
        str(r[key_col])
        for r in _read_json_list(path)
        if key_col in r and r[key_col] is not None
    }


def load_df(path: Path) -> pd.DataFrame:
    """
    Read a checkpoint file into a DataFrame (or an empty one if missing).

    Nested JSON values (lists / dicts) are preserved as native Python
    objects inside the cells - consumers can use them directly without
    a `json.loads()` step.
    """
    data = _read_json_list(path)
    return pd.DataFrame(data) if data else pd.DataFrame()
