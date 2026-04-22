"""
Crash-safe checkpointing.

Pipelines call `append_checkpoint(records, path, key_col)` after every
mini-batch. Writes are atomic (write to `.tmp`, then `os.replace`) so an
interrupted run never corrupts the CSV. `load_processed(path, key_col)`
lets the orchestrator skip drugs already on disk.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def append_checkpoint(
    records: Iterable[dict],
    path: Path,
    key_col: str = "drug_name",
) -> None:
    """
    Append `records` to `path` (CSV). De-duplicates on `key_col`, keeping the
    last write (so re-running a drug overwrites the old row).
    """
    records = list(records)
    if not records:
        return

    new_df = pd.DataFrame(records)
    if path.exists():
        try:
            old_df = pd.read_csv(path)
            merged = pd.concat([old_df, new_df], ignore_index=True, sort=False)
            merged = merged.drop_duplicates(subset=[key_col], keep="last")
        except Exception as e:
            log.warning("Could not read existing checkpoint %s (%s) - overwriting", path, e)
            merged = new_df
    else:
        merged = new_df

    _atomic_write(merged, path)


def load_processed(path: Path, key_col: str = "drug_name") -> set[str]:
    """Return the set of values already present in `key_col` of `path`."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=[key_col])
    except Exception as e:
        log.warning("Could not read checkpoint %s: %s", path, e)
        return set()
    return set(df[key_col].astype(str).tolist())


def load_df(path: Path) -> pd.DataFrame:
    """Read a checkpoint file, returning an empty frame if it does not exist."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
