"""
Robust JSON extraction from LLM output.

LLMs routinely wrap JSON in ```json fences, prepend chat-template noise,
produce trailing commas, or use single quotes. We try several cheap
recovery strategies before giving up.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCE_RE  = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _try_loads(s: str) -> Optional[dict]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _sanitize(s: str) -> str:
    # drop trailing commas before } or ]
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    # swap single quotes for double, but only when they're clearly JSON-like
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')
    return s


def parse_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from `text`."""
    if not text:
        return None

    # 1. explicit ```json ... ``` fence
    m = _FENCE_RE.search(text)
    if m:
        obj = _try_loads(m.group(1)) or _try_loads(_sanitize(m.group(1)))
        if obj is not None:
            return obj

    # 2. the largest {...} span
    candidates = _OBJECT_RE.findall(text)
    candidates.sort(key=len, reverse=True)
    for c in candidates:
        obj = _try_loads(c) or _try_loads(_sanitize(c))
        if obj is not None:
            return obj

    # 3. last resort: the raw text
    return _try_loads(text) or _try_loads(_sanitize(text))


def extract_targets(parsed: dict | None) -> list[tuple[str, str]]:
    """
    Pull `(target, rationale)` pairs out of a parsed target-prediction JSON.

    Tolerates either the {"target_1": ..., "rationale_1": ...} flat schema
    or a {"targets": [{"target": ..., "rationale": ...}]} list schema.
    """
    if not parsed:
        return []

    pairs: list[tuple[str, str]] = []

    # flat schema
    target_keys = sorted(
        (k for k in parsed if k.startswith("target_")),
        key=lambda k: int(re.sub(r"\D", "", k) or "0"),
    )
    for tk in target_keys:
        idx = re.sub(r"\D", "", tk)
        target = parsed.get(tk)
        rationale = parsed.get(f"rationale_{idx}", "")
        if isinstance(target, str) and target.strip():
            pairs.append((target.strip(), str(rationale).strip()))

    # list schema
    if not pairs and isinstance(parsed.get("targets"), list):
        for item in parsed["targets"]:
            if not isinstance(item, dict):
                continue
            t = item.get("target") or item.get("gene") or item.get("name")
            r = item.get("rationale") or item.get("reason") or ""
            if isinstance(t, str) and t.strip():
                pairs.append((t.strip(), str(r).strip()))

    # drop obvious nulls
    return [(t, r) for t, r in pairs if t.lower() not in {"none", "null", "n/a", ""}]
