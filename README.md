# Drug-Target LLM Pipeline (DrugComb + STRING)

End-to-end LLM pipeline that predicts protein targets for thousands of drugs
using few-shot learning + self-consistency, built on top of the description-
refinement pattern from the original CellHit notebook.

```
 ┌──────────────────────── Stage 1: Refined Description ────────────────────────┐
 │                                                                              │
 │   drug_name, SMILES, InChI                 PubMed abstracts (I/O pool)       │
 │           │                                         │                        │
 │           ▼                                         ▼                        │
 │     LLM (initial description)   ◀────── abstracts as evidence ──────▶ LLM    │
 │                                                                      (refine)│
 │                                        │                                     │
 │                                        ▼                                     │
 │                            refined_descriptions.csv                          │
 └──────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
 ┌──────────────────────── Stage 2: Target Prediction ──────────────────────────┐
 │                                                                              │
 │  few-shot pool = refined desc.  ⊕  STRING active proteins list  ⊕  drug       │
 │                                          │                                    │
 │                                          ▼                                    │
 │               vLLM.generate(prompts, SamplingParams(n=K))                     │
 │                  (K samples per prompt via prefix-cache reuse)                │
 │                                          │                                    │
 │                                          ▼                                    │
 │                    vote → filter(min_votes, allowed) → gene→PID               │
 │                                          │                                    │
 │                                          ▼                                    │
 │                              predicted_targets.csv                            │
 └──────────────────────────────────────────────────────────────────────────────┘
```

## Run order (as specified)

1. **GT drugs first** – refined descriptions for the 10 ground-truth drugs.
   They are *not* run through the target-prediction pipeline; their refined
   descriptions + known targets become the few-shot pool.
2. **All other drugs** – refined description, then target prediction with
   K-sample self-consistency.
3. **CID fallback retry** – drugs whose name-based PubMed lookup came back
   empty are retried via PubChem-synonym lookup. Recovered drugs are then
   run through both pipelines.

Every mini-batch writes to its checkpoint CSV atomically, so the run resumes
cheaply after any crash. Re-running `python main.py` never re-processes a
drug that is already in a checkpoint.

## Project layout

```
cellhit_targets/
├── config.py                     # all paths & hyper-parameters
├── main.py                       # orchestrator / CLI
├── launch.sh                     # SLURM-friendly launcher
├── requirements.txt
├── prompts/
│   ├── description_prompt.txt    # stage 1
│   ├── refiner_prompt.txt        # stage 2
│   └── target_prompt.txt         # stage 3 (few-shot + proteins list)
├── utils/
│   ├── abstracts.py              # concurrent PubMed + CID fallback
│   ├── prompts.py                # template renderer
│   ├── parser.py                 # tolerant JSON extractor
│   ├── aggregator.py             # self-consistency voting + gene→PID
│   └── checkpoint.py             # atomic CSV append + resume
└── pipelines/
    ├── description_pipeline.py   # I/O + 2-stage batched LLM call
    └── target_pipeline.py        # few-shot + self-consistency
```

## Inputs expected in `data/`

| File | Columns |
|------|---------|
| `drugs_info.csv` | Drug Name, InChiKey, CID, Smiles |
| `proteins_info.csv` | Protein ID, Protein Name, Coded Gene |
| `elite_drug_target_groundtruth.csv` | Drug Name, Proteins (list of Protein IDs) |

## Outputs written to `output/`

All pipeline outputs are written as **JSON arrays of records** (atomic
writes, last-write-wins on `drug_name`). Nested fields are native JSON —
no `json.loads()` gymnastics needed when consuming them.

| File | Contents |
|------|----------|
| `refined_descriptions.json` | `drug_name, cid, initial_description, refined_description, n_abstracts, abstracts (list[str])` |
| `predicted_targets.json`    | `drug_name, targets (list[{gene_symbol, protein_id, votes, rationales}]), top_genes (list[str]), top_pids (list[str]), n_runs, raw_votes (dict[str,int])` |
| `failed_abstracts.json`     | drugs for which PubMed returned nothing even after CID fallback (`drug_name, reason`) |
| `logs/pipeline.log`         | full run log |

Quick inspection:

```bash
jq '.[0]'                    output/refined_descriptions.json   # first drug
jq '.[] | .drug_name'        output/predicted_targets.json      # all drug names
jq '.[0].targets[0]'         output/predicted_targets.json      # top target record
jq '.[] | select(.raw_votes.EGFR >= 3)' output/predicted_targets.json
```

## Why this runs fast on 4 × A100 40 GB

| Optimisation | Where | Why it matters |
|--------------|-------|----------------|
| `tensor_parallel_size=4` | `_build_llm` | Mixtral-8×7B GPTQ 4-bit shards across the 4 GPUs; all forward passes are one collective. |
| **Single LLM instance for all 3 stages** | `main.py` | Model is loaded once (~60 s); every subsequent request reuses the engine, so per-drug overhead is dominated by the forward pass itself. |
| **Batch-level LLM calls** | both pipelines | Entire mini-batch of prompts is handed to `llm.generate(...)` at once — vLLM's continuous batching + paged KV cache fill the GPUs without Python-side synchronisation. |
| **`SamplingParams(n=K)` for self-consistency** | `target_pipeline` | K samples per prompt share the prefix KV-cache, so 5× self-consistency costs ~1.5–2× of a single sample instead of 5×. |
| **Concurrent PubMed I/O** | `abstracts.py` | `ThreadPoolExecutor` — the GIL is released during network I/O, so 8 workers saturate NCBI's 10 req/s limit. |
| **Atomic checkpointing** | `checkpoint.py` | Resumable runs = no wasted compute after a crash. |
| **Compact protein block** | `target_pipeline` | `GENE | Protein Name` one per line — ~15 tokens per protein keeps even 5 000-protein pools well under the 32 k context. |

### Throughput ballpark

For Mixtral-8×7B-GPTQ on 4 × A100 40 GB with `tensor_parallel_size=4`,
continuous batching delivers ~3–5 k decoding tokens / s aggregate. Per drug:

- Stage 1 (description) ≈ 500 tokens output
- Stage 2 (refine)      ≈ 700 tokens output
- Stage 3 (target, K=5) ≈ 5 × 400 tokens output ≈ 2 000 tokens

Total ≈ 3 200 tokens / drug → **~45 min for 5 000 drugs** assuming average
throughput. PubMed I/O overlaps with generation, so you rarely see it on
the critical path.

On 5 × A100 in **data-parallel** mode, expect roughly **5 × single-GPU
throughput minus merge cost (~a few seconds)** — in practice about
1.3–1.6× what 4-way tensor parallelism gives, and using every GPU
including the 5th.

## Running

```bash
# Fill in your email in config.PUBMED_EMAIL first!
pip install -r requirements.txt
```

### On 4 × A100 (tensor-parallel)

```bash
./launch.sh                       # default: TP=4, full run
./launch.sh --stage descriptions  # only stage 1/2a/3
./launch.sh --stage targets       # only stage 2b (needs refined_descriptions.json)
```

### On 5 × A100 (data-parallel)

`tensor_parallel_size=5` is **not legal** for Mixtral-8x7B (its 8 KV heads
don't divide by 5), so on 5 GPUs the right pattern is data parallelism —
5 independent vLLM instances, one per GPU, each running `TP=1`. Drugs are
stable-hash-sharded across workers and the per-worker JSONs are merged at
the end. Each worker processes the ground-truth drugs into its own shard
so the few-shot pool is locally available without cross-worker IPC;
duplicates are deduped during merge.

```bash
./launch_data_parallel.sh                      # 5 workers, 1 GPU each
NUM_WORKERS=4 ./launch_data_parallel.sh        # override count
./launch_data_parallel.sh --stage descriptions # forwarded flags work too
```

Per-worker logs land in `output/logs/worker_<N>.log`. When every worker
exits cleanly, the launcher runs `merge_outputs.py --cleanup` which
combines `refined_descriptions.w*.json` → `refined_descriptions.json`
(and likewise for the targets and failed-abstracts files), deleting the
shards.

If any worker fails, the launcher exits non-zero and skips the merge so
you can inspect the per-worker shards directly. Fix the problem, re-run,
then either rerun the full launcher (it resumes — every worker skips
drugs already in its own shard) or merge manually:

```bash
python merge_outputs.py --num-workers 5 --cleanup
```

Why data-parallel wins on 5 GPUs:

- No TP all-reduce between forward passes -> ~1.3-1.6× throughput vs TP.
- Mixtral-GPTQ-4bit fits in ~25 GB, so each A100 40 GB comfortably hosts
  its own instance plus a generous KV-cache.
- The NCBI rate limit is automatically divided by the number of workers
  (`_throttle_entrez` in `main.py`) so we stay polite to PubMed.

## Tuning knobs in `config.py`

- `SELF_CONSISTENCY_K` — more samples → more stable predictions, slower
- `MIN_VOTES` — stricter (higher) → fewer false positives, more false negatives
- `NUM_FEWSHOT_EXAMPLES` — 3 is a good default with 10 ground-truth drugs;
  with a larger pool, 5–8 usually helps.
- `BATCH_SIZE` — size of the checkpoint mini-batch. Bigger = fewer disk
  writes and better vLLM scheduling, but also more work lost on crash.
- `TARGET_TEMPERATURE` — keep in `[0.5, 0.9]`; below that, the K samples
  collapse to the same answer and self-consistency gives no signal.
