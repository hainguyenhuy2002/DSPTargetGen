#!/usr/bin/env bash
# --------------------------------------------------------------------------
# Launcher for the Drug-Target LLM pipeline on 4 x A100 40GB.
#
# Usage:
#   ./launch.sh                    # full run
#   ./launch.sh --stage targets    # only target prediction
# --------------------------------------------------------------------------
set -euo pipefail

# Pin the 4 GPUs we want vLLM to use
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}

# vLLM + HF caching
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/hub}

# NCCL tuning for A100 tensor parallelism
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export TOKENIZERS_PARALLELISM=false

# vLLM multiproc backend — spawn is safer inside notebooks / shells
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "$(dirname "$0")"

python main.py --tensor-parallel-size 5 "$@"
