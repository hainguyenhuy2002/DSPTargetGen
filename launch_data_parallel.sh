#!/usr/bin/env bash
# --------------------------------------------------------------------------
# Data-parallel launcher for the Drug-Target LLM pipeline.
#
# Spawns N worker processes, each pinned to one GPU with tensor_parallel_size=1.
# Each worker handles a stable hash-sharded subset of the drugs and writes to
# its own `*.wID.json`. When all workers exit 0, merge_outputs.py combines
# them into the final single-file outputs.
#
# Usage:
#   ./launch_data_parallel.sh                        # default: 5 workers
#   NUM_WORKERS=4 ./launch_data_parallel.sh          # override count
#   ./launch_data_parallel.sh --stage descriptions   # pass through flags
# --------------------------------------------------------------------------
set -euo pipefail

NUM_WORKERS=${NUM_WORKERS:-5}

# HF + NCCL environment shared by all workers
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/hub}
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "$(dirname "$0")"
mkdir -p output/logs

echo "Launching ${NUM_WORKERS} data-parallel workers (1 GPU each, TP=1)"

PIDS=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    LOGFILE="output/logs/worker_${i}.log"
    echo "  worker ${i} -> GPU ${i}  (log: ${LOGFILE})"
    CUDA_VISIBLE_DEVICES=${i} python main.py \
        --worker-id ${i} \
        --num-workers ${NUM_WORKERS} \
        --tensor-parallel-size 1 \
        "$@" \
        > "${LOGFILE}" 2>&1 &
    PIDS+=($!)
done

# Wait for every worker. Record the first non-zero exit code but keep waiting
# for the rest so we can attribute failures cleanly.
EXIT_CODE=0
for idx in "${!PIDS[@]}"; do
    pid=${PIDS[$idx]}
    if ! wait "${pid}"; then
        rc=$?
        echo "worker ${idx} failed with exit ${rc} (see output/logs/worker_${idx}.log)"
        EXIT_CODE=${rc}
    fi
done

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "At least one worker failed - skipping merge."
    exit ${EXIT_CODE}
fi

echo "All workers finished. Merging per-worker JSONs..."
python merge_outputs.py --num-workers ${NUM_WORKERS} --cleanup

echo "Done. See output/ for the merged files."
