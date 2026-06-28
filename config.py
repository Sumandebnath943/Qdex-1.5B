"""Central configuration for the QLoRA coding-model fine-tune project.

All scripts import from here so the model, dataset, and hyperparameters live in
one place. Stage 0 decisions are locked in below.
"""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# --- Stage 0 Decision 1: Base model ---
# The BASE (pretrained) model, NOT the -Instruct version.
# We are doing the instruction-tuning ourselves (that is the whole project).
# License: Apache 2.0 (verified) - permits fine-tuning + portfolio use.
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"

# --- Stage 0 Decision 2: Dataset ---
DATASET_NAME = "ise-uiuc/Magicoder-OSS-Instruct-75K"
DATASET_SLICE = 20000  # examples for the first training run (Stage 3)

# --- Instruction formatting ---
# System prompt shown to the model on every example during fine-tuning.
# Kept short and code-focused so the model learns a clear "coding assistant" persona.
SYSTEM_PROMPT = (
    "You are a helpful coding assistant. "
    "Respond with clean, correct, well-structured code and brief explanations."
)

# Maximum sequence length (in tokens) for training examples.
# 2048 covers ~98% of Magicoder examples comfortably; longer ones get truncated.
MAX_SEQ_LEN = 2048

# --- Stage 3 training hyperparameters (defined early for visibility; used later) ---
# Conservative defaults for a first QLoRA run on a single ~16GB GPU.
LORA_RANK = 16          # size of the adapter matrices - 16 is a good default
LORA_ALPHA = 32         # scaling factor - conventionally 2x rank
LORA_DROPOUT = 0.05     # light regularization
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]  # all linear layers
NUM_EPOCHS = 3          # conservative; Magicoder examples are short
BATCH_SIZE = 2          # per-device; small for memory safety on T4
GRAD_ACCUM = 4          # effective batch size = 2 x 4 = 8
LEARNING_RATE = 2e-4    # standard QLoRA rate for small models
WARMUP_RATIO = 0.03     # gentle warmup
SAVE_STEPS = 200        # checkpoint often so a dropped session loses little work
