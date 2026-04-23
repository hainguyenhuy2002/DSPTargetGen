"""
LLM backend abstraction.

The original pipeline was written against vLLM's `llm.generate(prompts,
SamplingParams)` API, which returns a list of `RequestOutput` objects whose
`.outputs` attribute is a list of K sampled completions (each with a `.text`
attribute). The target pipeline, in particular, relies on `n=K` self-consistency
sampling in a single forward pass.

To keep the pipeline files untouched, this module provides:

    * A backend-agnostic `SamplingParams` dataclass with the same field names.
    * `_Sample` / `_RequestOutput` shims that mimic the vLLM output shape.
    * `TogetherBackend` — a drop-in replacement for a `vllm.LLM` instance that
      talks to the Together AI REST API via the llama-index wrapper.
    * `build_backend(kind, ...)` — small factory.

The Together backend fans out prompts (and per-prompt K samples) across a
thread pool, since each `complete()` call is an independent HTTPS request.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared sampling schema (matches the subset of vLLM SamplingParams the
# pipelines actually use).
# ---------------------------------------------------------------------------
@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    n: int = 1
    seed: Optional[int] = None
    # Anything you want the backend to forward verbatim to the underlying SDK.
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output shims — match `vllm.outputs.RequestOutput.outputs[i].text`.
# ---------------------------------------------------------------------------
@dataclass
class _Sample:
    text: str
    finish_reason: Optional[str] = None


@dataclass
class _RequestOutput:
    outputs: list  # list[_Sample]
    prompt: Optional[str] = None


# ---------------------------------------------------------------------------
# Together AI backend
# ---------------------------------------------------------------------------
class TogetherBackend:
    """
    Drop-in replacement for a vLLM engine object that uses Together AI.

    Exposes `.generate(prompts, sampling_params, use_tqdm=True)` and returns
    a list of `_RequestOutput` objects that the existing pipelines already
    know how to consume.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        max_workers: int = 8,
        request_timeout: float = 120.0,
    ):
        try:
            from llama_index.llms.together import TogetherLLM
        except ImportError as e:
            raise ImportError(
                "TogetherBackend requires `llama-index-llms-together`. "
                "Install with: pip install llama-index-llms-together"
            ) from e

        if not api_key or api_key == "YOUR_TOGETHER_API_KEY":
            # Also check env as a convenience.
            api_key = os.environ.get("TOGETHER_API_KEY", api_key)
        if not api_key or api_key == "YOUR_TOGETHER_API_KEY":
            raise ValueError(
                "No Together AI API key configured. Set TOGETHER_API_KEY in "
                "your environment, or edit config.TOGETHER_API_KEY."
            )

        self._TogetherLLM = TogetherLLM
        self._model = model
        self._api_key = api_key
        self._request_timeout = request_timeout
        self.max_workers = max_workers

        log.info(
            "TogetherBackend ready (model=%s, max_workers=%d)",
            model,
            max_workers,
        )

    # ---- internal helpers -------------------------------------------------
    def _make_llm(self, sp: SamplingParams):
        """Instantiate a TogetherLLM with sampling params baked in."""
        kwargs: dict[str, Any] = dict(
            model=self._model,
            api_key=self._api_key,
            temperature=sp.temperature,
            max_tokens=sp.max_tokens,
        )
        # Some TogetherLLM versions expose top_p / timeout as separate kwargs,
        # others only via `additional_kwargs`. Try both paths.
        additional: dict[str, Any] = {"top_p": sp.top_p}
        additional.update(sp.extra or {})
        try:
            return self._TogetherLLM(
                **kwargs,
                additional_kwargs=additional,
                timeout=self._request_timeout,
            )
        except TypeError:
            try:
                return self._TogetherLLM(**kwargs, additional_kwargs=additional)
            except TypeError:
                return self._TogetherLLM(**kwargs)

    @staticmethod
    def _complete(llm, prompt: str) -> str:
        resp = llm.complete(prompt)
        # llama_index's CompletionResponse exposes `.text`
        return getattr(resp, "text", str(resp))

    # ---- public API --------------------------------------------------------
    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
        use_tqdm: bool = True,
    ) -> list[_RequestOutput]:
        if not prompts:
            return []

        llm = self._make_llm(sampling_params)
        n = max(1, int(sampling_params.n))

        # Pre-allocate the per-prompt sample buckets so we can write to them
        # out of order as the thread pool completes.
        per_prompt: list[list[_Sample]] = [[] for _ in prompts]

        tasks: list[tuple[int, cf.Future]] = []
        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for i, prompt in enumerate(prompts):
                for _ in range(n):
                    fut = pool.submit(self._complete, llm, prompt)
                    tasks.append((i, fut))

            iterator: Any = cf.as_completed([f for _, f in tasks])
            if use_tqdm:
                try:
                    from tqdm import tqdm

                    iterator = tqdm(
                        iterator,
                        total=len(tasks),
                        desc=f"Together AI (n={n})",
                    )
                except ImportError:
                    pass

            # Map futures back to their prompt index.
            fut_to_idx = {f: i for i, f in tasks}
            for fut in iterator:
                i = fut_to_idx[fut]
                try:
                    text = fut.result()
                except Exception as e:
                    log.warning("Together AI call failed for prompt %d: %s", i, e)
                    text = ""
                per_prompt[i].append(_Sample(text=text, finish_reason="stop"))

        return [
            _RequestOutput(outputs=samples, prompt=prompts[i])
            for i, samples in enumerate(per_prompt)
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_backend(
    kind: str,
    *,
    # vLLM kwargs
    model_path: Optional[str] = None,
    tensor_parallel_size: int = 1,
    model_revision: Optional[str] = None,
    quantization: Optional[str] = None,
    dtype: Optional[str] = None,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = True,
    # Together kwargs
    together_model: Optional[str] = None,
    together_api_key: Optional[str] = None,
    together_max_workers: int = 8,
    together_request_timeout: float = 120.0,
):
    """Build an LLM backend exposing vLLM's `.generate()` shape."""
    kind = (kind or "together").lower()

    if kind == "together":
        if not together_model:
            raise ValueError("together backend requires `together_model`.")
        return TogetherBackend(
            model=together_model,
            api_key=together_api_key or "",
            max_workers=together_max_workers,
            request_timeout=together_request_timeout,
        )

    if kind == "vllm":
        from vllm import LLM

        log.info(
            "Loading vLLM model from %s (TP=%d)",
            model_path,
            tensor_parallel_size,
        )
        return LLM(
            model=model_path,
            revision=model_revision,
            quantization=quantization,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )

    raise ValueError(f"Unknown LLM backend: {kind!r}")
