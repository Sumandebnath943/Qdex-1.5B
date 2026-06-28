"""Stage 6: Export the merged model to GGUF for local CPU inference.

GGUF is the format used by llama.cpp / Ollama. It quantizes the model to 4-bit
(q4_k_m by default), shrinking a 1.5B model from ~3GB (fp16) to ~1GB — small
enough to run on a 16GB no-GPU laptop at usable speed (~15-30 tokens/sec).

This is what Project 3's CLI agent will load.

Run this on the SAME GPU where merge happened (uses Unsloth's GGUF exporter,
which shells out to llama.cpp's convert script). Output: outputs/gguf/*.gguf.

REQUIRES: a GPU (Kaggle). The Unsloth GGUF export also clones llama.cpp, so it
needs internet access on the instance (Kaggle has this).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402

from config import MAX_SEQ_LEN, OUTPUTS_DIR  # noqa: E402
from src.chat_template import QWEN_CHATML_TEMPLATE  # noqa: E402

MERGED_PATH = str(OUTPUTS_DIR / "merged")
GGUF_PATH = str(OUTPUTS_DIR / "gguf")

# Quantization methods to produce. q4_k_m is the recommended default (best
# quality/size tradeoff). q5_k_m is slightly better + slightly bigger. q8_0 is
# near-lossless but bigger. We produce q4_k_m + q5_k_m so the owner can choose.
QUANT_METHODS = ["q4_k_m", "q5_k_m"]


def main():
    print("=" * 60)
    print("STAGE 6: GGUF EXPORT")
    print("=" * 60)
    print(f"Source (merged): {MERGED_PATH}")
    print(f"Output dir:      {GGUF_PATH}")
    print(f"Quantization:    {', '.join(QUANT_METHODS)}")

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU. GGUF export requires a GPU (run on Kaggle).")

    print("\nLoading merged model via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MERGED_PATH,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = QWEN_CHATML_TEMPLATE

    for method in QUANT_METHODS:
        print(f"\nExporting {method}...")
        model.save_pretrained_gguf(
            GGUF_PATH,
            tokenizer,
            quantization_method=method,
        )

    print("\n" + "=" * 60)
    print("GGUF EXPORT COMPLETE")
    print("=" * 60)
    gguf_dir = Path(GGUF_PATH)
    files = list(gguf_dir.glob("*.gguf"))
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.0f} MB")
    print("\nNext: download the .gguf files to your laptop and test with Ollama")
    print("(see scripts/local_gguf_test.md).")
    print("REMINDER: SHUT DOWN the GPU instance after downloading.")


if __name__ == "__main__":
    main()
