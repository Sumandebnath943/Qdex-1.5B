"""Stage 4: Merge the trained LoRA adapter back into the base model.

QLoRA training only updates small "adapter" weights. To use the fine-tuned model
standalone (for benchmarking and GGUF export), we merge those adapters into the
base model's weights, producing one complete model.

Run this on the SAME GPU where training happened (the model is still in memory
if you run train + merge in one session — see kaggle_notebook.py). If run
separately, it reloads the adapter via Unsloth (which auto-loads the base too).

Output: outputs/merged/ — a full fp16 model that can be loaded by any
HuggingFace-compatible tool without needing the adapter separately.

REQUIRES: a GPU (same as training). Output size: ~3GB for a 1.5B fp16 model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402

from config import BASE_MODEL, MAX_SEQ_LEN, OUTPUTS_DIR  # noqa: E402
from src.chat_template import QWEN_CHATML_TEMPLATE  # noqa: E402

ADAPTER_PATH = str(OUTPUTS_DIR / "lora_adapter")
MERGED_PATH = str(OUTPUTS_DIR / "merged")


def main():
    print("=" * 60)
    print("STAGE 4: MERGE LORA ADAPTER INTO BASE MODEL")
    print("=" * 60)
    print(f"Adapter:  {ADAPTER_PATH}")
    print(f"Base:     {BASE_MODEL}")
    print(f"Output:   {MERGED_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU. Merging requires a GPU. Run on Kaggle.")

    # Unsloth loads the adapter + base model together in 4-bit.
    print("\nLoading adapter + base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = QWEN_CHATML_TEMPLATE

    # Merge and save as fp16 (full precision merge; needed for clean GGUF export).
    print("Merging adapter into base weights (saving as merged_16bit)...")
    model.save_pretrained_merged(
        MERGED_PATH,
        tokenizer,
        save_method="merged_16bit",
    )

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"Merged model: {MERGED_PATH}")
    print("Next: run scripts/export_gguf.py to convert to GGUF for local use,")
    print("      and scripts/run_benchmark.py --model-path outputs/merged "
          "--mode instruction")
    print("REMINDER: Download results, then SHUT DOWN the GPU instance.")


if __name__ == "__main__":
    main()
