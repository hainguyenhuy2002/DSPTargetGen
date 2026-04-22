"""
Concurrent PubMed abstract retrieval.

Policy implemented (as required by the spec):
  1. On the first pass, try to fetch abstracts using the drug *name* only.
     If that returns nothing, mark the drug as FAILED and skip it entirely
     for this pass (no description, no target prediction).
  2. After the two main pipelines have finished, the orchestrator calls
     `fetch_with_cid_fallback` for every failed drug: we look up synonyms
     via PubChem (by CID) and re-query PubMed with those alternative names.

Network I/O is the bottleneck here, not compute, so we fan out with a
ThreadPoolExecutor. NCBI rate-limits at 3 req/s without an API key and
10 req/s with one — keep ENTREZ_MAX_WORKERS consistent with that.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Optional, Tuple

from Bio import Entrez

log = logging.getLogger(__name__)


# ==========================================================================
# Low-level single-request helpers
# ==========================================================================
def _entrez_config(email: str, api_key: Optional[str] = None) -> None:
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key


def _fetch_by_term(
    term: str, k: int, retries: int = 3, backoff: float = 0.5
) -> List[str]:
    """Query PubMed for `term`, return up to `k` abstract strings. [] on failure."""
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            handle = Entrez.esearch(db="pubmed", term=term, retmax=k, sort="relevance")
            search_results = Entrez.read(handle)
            handle.close()

            ids = search_results.get("IdList", [])
            if not ids:
                return []

            handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml")
            papers = Entrez.read(handle).get("PubmedArticle", [])
            handle.close()

            abstracts: List[str] = []
            for paper in papers:
                try:
                    chunks = paper["MedlineCitation"]["Article"]["Abstract"][
                        "AbstractText"
                    ]
                    abstracts.append(" ".join(str(c) for c in chunks))
                except (KeyError, IndexError):
                    continue
            return abstracts
        except Exception as e:  # network / XML errors are expected
            last_err = e
            time.sleep(backoff * (2**attempt))
    log.debug("PubMed failed for term=%r after %d retries: %s", term, retries, last_err)
    return []


def _cid_to_synonyms(cid: int | str, max_syn: int = 5) -> List[str]:
    """Use PubChem to turn a CID into a handful of searchable names."""
    try:
        import pubchempy as pcp
    except ImportError:
        log.warning("pubchempy not installed - cannot do CID fallback")
        return []

    try:
        compound = pcp.Compound.from_cid(int(cid))
    except Exception as e:
        log.debug("PubChem lookup failed for CID=%s: %s", cid, e)
        return []

    names: List[str] = []
    if getattr(compound, "iupac_name", None):
        names.append(compound.iupac_name)
    if getattr(compound, "synonyms", None):
        names.extend(compound.synonyms[:max_syn])
    # dedupe while preserving order
    return list(dict.fromkeys(n for n in names if n))


# ==========================================================================
# Public API
# ==========================================================================
def fetch_abstracts_by_name(
    drug_name: str,
    k: int,
    email: str,
    api_key: Optional[str] = None,
) -> List[str]:
    """Single-drug fetch using the drug name. Empty list on failure."""
    _entrez_config(email, api_key)
    return _fetch_by_term(drug_name, k=k)


def fetch_with_cid_fallback(
    drug_name: str,
    cid: int | str | None,
    k: int,
    email: str,
    api_key: Optional[str] = None,
) -> List[str]:
    """Retry using names looked up from PubChem by CID. Empty list if still no luck."""
    if cid is None or str(cid).lower() == "nan":
        return []
    _entrez_config(email, api_key)
    for alt in _cid_to_synonyms(cid):
        abstracts = _fetch_by_term(alt, k=k)
        if abstracts:
            log.info(
                "CID-fallback recovered %d abstracts for %r via %r",
                len(abstracts),
                drug_name,
                alt,
            )
            return abstracts
        # small courtesy sleep between PubChem + PubMed hops
        time.sleep(0.2)
    return []


def batch_fetch_by_name(
    drugs: Iterable[Tuple[str, int | str | None]],
    k: int,
    email: str,
    api_key: Optional[str] = None,
    max_workers: int = 8,
) -> Tuple[dict, List[str]]:
    """
    Fan out `fetch_abstracts_by_name` over many drugs in parallel.

    Parameters
    ----------
    drugs : iterable of (drug_name, cid)
    Returns
    -------
    results : dict[drug_name] -> list[str]
        Only drugs for which at least one abstract was found are present.
    failed  : list[str]
        Drug names for which the name-based lookup returned nothing.
    """
    _entrez_config(email, api_key)
    drugs = list(drugs)
    results: dict = {}
    failed: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_name = {
            pool.submit(_fetch_by_term, name, k): name for name, _cid in drugs
        }
        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            try:
                abstracts = fut.result()
            except Exception as e:
                log.warning("Unexpected error fetching %r: %s", name, e)
                abstracts = []
            if abstracts:
                results[name] = abstracts
            else:
                failed.append(name)

    return results, failed


def batch_fetch_by_cid(
    drugs: Iterable[Tuple[str, int | str | None]],
    k: int,
    email: str,
    api_key: Optional[str] = None,
    max_workers: int = 5,  # PubChem is slower; fewer workers to stay polite
) -> Tuple[dict, List[str]]:
    """
    Same shape as `batch_fetch_by_name` but uses the CID fallback strategy.
    """
    _entrez_config(email, api_key)
    drugs = list(drugs)
    results: dict = {}
    still_failed: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_name = {
            pool.submit(fetch_with_cid_fallback, name, cid, k, email, api_key): name
            for name, cid in drugs
        }
        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            try:
                abstracts = fut.result()
            except Exception as e:
                log.warning("CID fallback errored for %r: %s", name, e)
                abstracts = []
            if abstracts:
                results[name] = abstracts
            else:
                still_failed.append(name)

    return results, still_failed
