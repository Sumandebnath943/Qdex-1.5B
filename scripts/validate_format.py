"""Stage 1 validation script.

Purpose: prove the data-formatting code works by loading a few REAL Magicoder
examples, formatting them with the Qwen2.5 ChatML template, and printing them
in full for manual inspection.

This needs NO GPU and downloads NO model weights (only the ~2MB tokenizer).
It is the "small testable increment" for Stage 1.

Run from the project root:
    python scripts/validate_format.py
"""
import sys
from pathlib import Path

# Make the project root importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from config import (  # noqa: E402
    BASE_MODEL, DATASET_NAME, DATASET_SLICE, SYSTEM_PROMPT, MAX_SEQ_LEN,
)
from src.format_data import get_tokenizer, format_example, load_raw_dataset  # noqa: E402

NUM_TO_SHOW = 3          # how many full examples to print
HIST_BINS = [0, 256, 512, 1024, 2048, 4096, 8192]  # for token-length histogram


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    section("STAGE 1 - FORMAT VALIDATION")
    print(f"Base model:    {BASE_MODEL}  (base/pretrained, Apache 2.0)")
    print(f"Dataset:       {DATASET_NAME}")
    print(f"Slice size:    {DATASET_SLICE} examples (for the Stage 3 run)")
    print(f"System prompt: {SYSTEM_PROMPT}")
    print(f"Max seq len:   {MAX_SEQ_LEN} tokens")

    # ---- 1. Load tokenizer ----
    section("1. TOKENIZER")
    print("Loading base model tokenizer (downloads ~2MB, NO model weights)...")
    tokenizer = get_tokenizer()
    print(f"  Vocab size:        {len(tokenizer)}")
    print(f"  Has chat_template: {tokenizer.chat_template is not None}")
    print(f"  Pad token:         {tokenizer.pad_token!r}")
    print(f"  EOS token:         {tokenizer.eos_token!r}")

    # ---- 2. Inspect raw dataset structure ----
    section("2. RAW DATASET STRUCTURE")
    print("Loading dataset (downloads ~80MB, cached after first run)...")
    ds = load_raw_dataset()
    print(f"  Total examples:  {len(ds):,}")
    print(f"  Column names:    {ds.column_names}")
    raw0 = ds[0]
    print(f"  First example raw fields:")
    for k, v in raw0.items():
        preview = str(v).replace("\n", " ")
        preview = (preview[:180] + "...") if len(preview) > 180 else preview
        print(f"    {k}: {preview}")

    # ---- 3. Format a few examples and print in FULL ----
    section(f"3. FORMATTED EXAMPLES (showing {NUM_TO_SHOW} in full)")
    for i in range(NUM_TO_SHOW):
        example = ds[i]
        formatted = format_example(example, tokenizer)
        print(f"\n--- Example {i} | {formatted['token_len']} tokens ---")
        print(formatted["text"])
        print(f"--- end example {i} ---")

    # ---- 4. Token-length sanity check on a larger sample ----
    section("4. TOKEN-LENGTH SANITY CHECK (1000 examples)")
    sample = ds.select(range(min(1000, len(ds))))
    lengths = []
    for ex in sample:
        lengths.append(format_example(ex, tokenizer)["token_len"])
    over_limit = sum(1 for L in lengths if L > MAX_SEQ_LEN)
    print(f"  Sampled:            {len(lengths)} examples")
    print(f"  Min / Max / Mean:   {min(lengths)} / {max(lengths)} / "
          f"{sum(lengths)//len(lengths)} tokens")
    print(f"  Over MAX_SEQ_LEN ({MAX_SEQ_LEN}):  {over_limit} "
          f"({100*over_limit/len(lengths):.1f}%) - these will be truncated in training")
    print("  Length distribution:")
    for lo, hi in zip(HIST_BINS, HIST_BINS[1:]):
        count = sum(1 for L in lengths if lo <= L < hi)
        bar = "#" * (count // 5)
        print(f"    {lo:>5}-{hi:<5}: {count:>4}  {bar}")

    section("DONE")
    print("If the formatted examples above look like clean system/user/assistant")
    print("blocks, Stage 1 is validated. Review them, then we move to Stage 2.")


if __name__ == "__main__":
    main()
