"""
Central configuration for the Drug-Target LLM pipeline.

Every path, hyperparameter, and prompt template is defined here so the
pipelines themselves stay pure logic and can be unit-tested in isolation.
"""
from pathlib import Path

# ==========================================================================
# Paths
# ==========================================================================
PROJECT_ROOT     = Path(__file__).resolve().parent
DATA_DIR         = PROJECT_ROOT / "data"
PROMPTS_DIR      = PROJECT_ROOT / "prompts"
OUTPUT_DIR       = PROJECT_ROOT / "output"
LOG_DIR          = OUTPUT_DIR / "logs"

# --- Inputs --------------------------------------------------------------
DRUGS_INFO_CSV        = DATA_DIR / "drugs_info.csv"
PROTEINS_INFO_CSV     = DATA_DIR / "proteins_info.csv"
GROUND_TRUTH_CSV      = DATA_DIR / "elite_drug_target_groundtruth.csv"

# --- Outputs (checkpointed) ---------------------------------------------
REFINED_DESC_CSV      = OUTPUT_DIR / "refined_descriptions.csv"
PREDICTED_TARGETS_CSV = OUTPUT_DIR / "predicted_targets.csv"
FAILED_ABSTRACTS_CSV  = OUTPUT_DIR / "failed_abstracts.csv"

# --- Prompt templates ---------------------------------------------------
DESCRIPTION_PROMPT = PROMPTS_DIR / "description_prompt.txt"
REFINER_PROMPT     = PROMPTS_DIR / "refiner_prompt.txt"
TARGET_PROMPT      = PROMPTS_DIR / "target_prompt.txt"

# ==========================================================================
# Model (vLLM)
# ==========================================================================
# Path can be a local snapshot or a HF repo id. Default is the GPTQ 4-bit
# Mixtral used in the original CellHit notebook.
MODEL_PATH             = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"
MODEL_REVISION         = "gptq-4bit-32g-actorder_True"
QUANTIZATION           = "gptq"             # set to None for fp16 models
DTYPE                  = "float16"
TENSOR_PARALLEL_SIZE   = 4                  # 4 x A100 40GB
MAX_MODEL_LEN          = 32_768
GPU_MEMORY_UTILIZATION = 0.90
TRUST_REMOTE_CODE      = True
ENFORCE_EAGER          = False              # keep CUDA graphs for throughput
SEED                   = 17

# ==========================================================================
# Generation hyper-parameters
# ==========================================================================
# Stage 1 — initial drug description from name/SMILES/InChI
DESC_TEMPERATURE = 0.2
DESC_TOP_P       = 0.9
DESC_MAX_TOKENS  = 768

# Stage 2 — refinement with PubMed abstracts
REFINE_TEMPERATURE = 0.2
REFINE_TOP_P       = 0.9
REFINE_MAX_TOKENS  = 1024

# Stage 3 — target prediction (few-shot + proteins list)
TARGET_TEMPERATURE = 0.7      # higher -> diversity for self-consistency
TARGET_TOP_P       = 0.95
TARGET_MAX_TOKENS  = 1024
SELF_CONSISTENCY_K = 5        # samples drawn per drug via `n=K`
MIN_VOTES          = 2        # keep targets with >= this many votes

# ==========================================================================
# Pipeline behavior
# ==========================================================================
BATCH_SIZE            = 128   # drugs per checkpointed mini-batch
ABSTRACTS_PER_DRUG    = 5
NUM_FEWSHOT_EXAMPLES  = 3     # ground-truth drugs shown per target-prediction prompt
ENTREZ_MAX_WORKERS    = 8     # concurrent PubMed fetches (NCBI allows ~3 req/s w/o API key, 10 w/ key)
PUBMED_EMAIL          = "your_email@example.com"      # REQUIRED by NCBI
PUBMED_API_KEY        = None                           # optional, unlocks 10 req/s

# Resume behaviour
RESUME = True         # skip drugs already written to checkpoint files
