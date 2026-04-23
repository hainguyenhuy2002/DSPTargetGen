"""
Central configuration for the Drug-Target LLM pipeline.

Every path, hyperparameter, and prompt template is defined here so the
pipelines themselves stay pure logic and can be unit-tested in isolation.
"""
import os

from pathlib import Path

# ==========================================================================
# Paths
# ==========================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = OUTPUT_DIR / "logs"

# --- Load secrets from .env --------------------------------------------
# Values in the actual shell environment take precedence over .env values
# (standard python-dotenv behavior with override=False). If python-dotenv
# is not installed, environment variables still work.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

# --- Inputs --------------------------------------------------------------
DRUGS_INFO_CSV = "/DATA/DATANAS2/rhh25/Cellhit/data/drugs/drugs_info.csv"
PROTEINS_INFO_CSV = "/DATA/DATANAS2/rhh25/dti_dataset/drugcomb_target/protein_info.csv"
GROUND_TRUTH_CSV = (
    "/DATA/DATANAS2/rhh25/dti_dataset/drugcomb_target/elite_drug_target_groundtruth.csv"
)


# --- Outputs (checkpointed, JSON) ---------------------------------------
REFINED_DESC_JSON = OUTPUT_DIR / "refined_descriptions.json"
PREDICTED_TARGETS_JSON = OUTPUT_DIR / "predicted_targets.json"
FAILED_ABSTRACTS_JSON = OUTPUT_DIR / "failed_abstracts.json"

# --- Prompt templates ---------------------------------------------------
DESCRIPTION_PROMPT = PROMPTS_DIR / "description_prompt.txt"
REFINER_PROMPT = PROMPTS_DIR / "refiner_prompt.txt"
TARGET_PROMPT = PROMPTS_DIR / "target_prompt.txt"

# ==========================================================================
# Model backend
# ==========================================================================
# Which backend to use for LLM inference.
#   "together" — Together AI REST API (via llama-index TogetherLLM)
#   "vllm"     — local vLLM engine (legacy path; requires GPUs)

LLM_BACKEND = os.environ.get("LLM_BACKEND", "together").lower()

# --- Together AI settings ------------------------------------------------
# Paste your key directly here, or export TOGETHER_API_KEY in the shell.
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "YOUR_TOGETHER_API_KEY")
TOGETHER_MODEL = "deepseek-ai/DeepSeek-V3"  # must match Together's exact model ID
TOGETHER_MAX_WORKERS = 8  # concurrent HTTP requests to Together
TOGETHER_REQUEST_TIMEOUT = 120.0  # seconds

# Retry-with-backoff on 429 / 5xx / transient network errors.
# `MAX_RETRIES` is the max number of *retries* after the first attempt
# (so total attempts = MAX_RETRIES + 1). Set to 0 to disable retries.
# If the server returns a `Retry-After` header, it is honored verbatim
# (plus a small random jitter). Otherwise a decorrelated-jitter exponential
# backoff is used, bounded by INITIAL_BACKOFF and MAX_BACKOFF.
TOGETHER_MAX_RETRIES = 8
TOGETHER_INITIAL_BACKOFF = 2.0  # seconds
TOGETHER_MAX_BACKOFF = 60.0  # seconds

# --- vLLM settings (only used when LLM_BACKEND == "vllm") ---------------
# Path can be a local snapshot or a HF repo id. Default is the GPTQ 4-bit
# Mixtral used in the original CellHit notebook.
MODEL_PATH = "/villa/rhh25/Cellhit/mixtral"
MODEL_REVISION = "gptq-4bit-32g-actorder_True"
QUANTIZATION = "gptq"  # set to None for fp16 models
DTYPE = "float16"
TENSOR_PARALLEL_SIZE = 4  # 4 x A100 40GB
MAX_MODEL_LEN = 32_768
GPU_MEMORY_UTILIZATION = 0.9
TRUST_REMOTE_CODE = True
ENFORCE_EAGER = True  # keep CUDA graphs for throughput
SEED = 17

# ==========================================================================
# Generation hyper-parameters
# ==========================================================================
# Stage 1 — initial drug description from name/SMILES/InChI
DESC_TEMPERATURE = 0.2
DESC_TOP_P = 0.9
DESC_MAX_TOKENS = 768

# Stage 2 — refinement with PubMed abstracts
REFINE_TEMPERATURE = 0.2
REFINE_TOP_P = 0.9
REFINE_MAX_TOKENS = 1024

# Stage 3 — target prediction (few-shot + proteins list)
TARGET_TEMPERATURE = 0.7  # higher -> diversity for self-consistency
TARGET_TOP_P = 0.95
TARGET_MAX_TOKENS = 1024
SELF_CONSISTENCY_K = 5  # samples drawn per drug via `n=K`
MIN_VOTES = 2  # keep targets with >= this many votes

# ==========================================================================
# Pipeline behavior
# ==========================================================================
BATCH_SIZE = 128  # drugs per checkpointed mini-batch
ABSTRACTS_PER_DRUG = 5
NUM_FEWSHOT_EXAMPLES = 3  # ground-truth drugs shown per target-prediction prompt
ENTREZ_MAX_WORKERS = (
    8  # concurrent PubMed fetches (NCBI allows ~3 req/s w/o API key, 10 w/ key)
)
PUBMED_EMAIL = "your_email@example.com"  # REQUIRED by NCBI
PUBMED_API_KEY = None  # optional, unlocks 10 req/s

# Resume behaviour
RESUME = True  # skip drugs already written to checkpoint files
