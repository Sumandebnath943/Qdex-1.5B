"""Stage 3: QLoRA fine-tuning of Qwen2.5-Coder-1.5B on Magicoder-OSS-Instruct.

This is the canonical training script, run from the cloned repo on a GPU
(see scripts/kaggle_run.md). It:
  1. Loads the base model in 4-bit (QLoRA quantization) via Unsloth.
  2. Attaches LoRA adapters to all linear layers (only these get trained).
  3. Loads + formats Magicoder examples into the Qwen ChatML template.
  4. Trains with conservative hyperparameters, checkpointing every SAVE_STEPS
     steps so a dropped session loses little work.
  5. Saves the LoRA adapter (small, ~50MB) to outputs/lora_adapter/.

Targets the library versions Kaggle ships (verified 2026-06): transformers 5.x,
trl 0.24 (SFTConfig API), peft 0.18, unsloth 2026.6. Unsloth is imported FIRST
so its optimizations patch transformers/trl/peft.

Quick dry run (proves the pipeline end-to-end in ~2 min before the real run):
    python scripts/train_qlora.py --limit 50 --max-steps 2
Full run:
    python scripts/train_qlora.py

REQUIRES: a GPU with >= 8GB VRAM (Kaggle T4). Run time on T4: ~2-3 hours.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Unsloth MUST be imported before transformers/trl/peft so its patches apply.
from unsloth import FastLanguageModel  # noqa: E402  (import-order is intentional)

import torch  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from config import (  # noqa: E402
    BASE_MODEL, DATASET_NAME, DATASET_SLICE, MAX_SEQ_LEN,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
    NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE,
    WARMUP_RATIO, SAVE_STEPS, OUTPUTS_DIR,
)
from src.chat_template import QWEN_CHATML_TEMPLATE  # noqa: E402
from src.format_data import format_example, load_raw_dataset  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="QLoRA fine-tune (Unsloth).")
    ap.add_argument("--limit", type=int, default=DATASET_SLICE,
                    help=f"How many examples to train on (default {DATASET_SLICE}). "
                         "Use a small number (e.g. 50) for a dry run.")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Cap total optimizer steps (default -1 = use epochs). "
                         "Set 2 for a quick dry run.")
    return ap.parse_args()


def main():
    args = parse_args()
    is_dry_run = args.max_steps and args.max_steps > 0

    print("=" * 60)
    print("STAGE 3: QLoRA FINE-TUNING" + ("  [DRY RUN]" if is_dry_run else ""))
    print("=" * 60)
    print(f"Base model:  {BASE_MODEL}")
    print(f"Dataset:     {DATASET_NAME} ({args.limit} examples)")
    print(f"LoRA:        rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"Training:    {NUM_EPOCHS} epochs, lr={LEARNING_RATE}, "
          f"batch={BATCH_SIZE}x{GRAD_ACCUM}"
          + (f", max_steps={args.max_steps}" if is_dry_run else ""))
    print(f"GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE!'}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. QLoRA training requires a GPU (>= 8GB VRAM). "
            "Run this on Kaggle (T4). See scripts/kaggle_run.md."
        )

    # ---- 1. Load base model in 4-bit ----
    print("\n[1/5] Loading base model in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,            # auto-select fp16/bf16 based on GPU
        load_in_4bit=True,
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = QWEN_CHATML_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. Add LoRA adapters ----
    print("[2/5] Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves memory
        random_state=42,
    )

    # ---- 3. Load + format dataset ----
    print(f"[3/5] Loading {args.limit} examples from {DATASET_NAME}...")
    ds_full = load_raw_dataset()
    ds = ds_full.select(range(min(args.limit, len(ds_full))))
    # Keep ONLY the "text" column (drop token_len etc.) for the SFT trainer.
    formatted = ds.map(
        lambda ex: {"text": format_example(ex, tokenizer)["text"]},
        remove_columns=ds.column_names,
        desc="Formatting",
    )
    print(f"  Formatted {len(formatted)} examples. "
          f"First example starts: {formatted[0]['text'][:60]!r}")

    # ---- 4. Train (trl 0.24 SFTConfig API) ----
    print("[4/5] Starting training...")
    use_bf16 = torch.cuda.is_bf16_supported()
    sft_config = SFTConfig(
        output_dir=str(OUTPUTS_DIR / "checkpoints"),
        # SFT-specific settings (these moved into SFTConfig in newer trl):
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=False,
        # Standard training settings:
        num_train_epochs=NUM_EPOCHS,
        max_steps=args.max_steps,           # -1 = ignore, use epochs
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=1 if is_dry_run else 10,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        save_strategy="no" if is_dry_run else "steps",
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="adamw_8bit",                 # 8-bit optimizer saves memory
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        report_to="none",
        dataloader_num_workers=2,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,         # was `tokenizer=` in old trl
        train_dataset=formatted,
        args=sft_config,
    )
    trainer.train()
    if trainer.state.log_history:
        last = trainer.state.log_history[-1]
        print(f"\n  Final loss: {last.get('loss', last.get('train_loss', 'N/A'))}")

    if is_dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN COMPLETE — pipeline works. Re-run WITHOUT --limit/--max-steps")
        print("for the full training run.")
        print("=" * 60)
        return

    # ---- 5. Save adapter ----
    print(f"[5/5] Saving LoRA adapter to {OUTPUTS_DIR / 'lora_adapter'}...")
    model.save_pretrained(str(OUTPUTS_DIR / "lora_adapter"))
    tokenizer.save_pretrained(str(OUTPUTS_DIR / "lora_adapter"))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Adapter saved: {OUTPUTS_DIR / 'lora_adapter'}")
    print("Next: scripts/merge_lora.py, then scripts/export_gguf.py.")
    print("REMINDER: download results, then SHUT DOWN the GPU instance.")


if __name__ == "__main__":
    main()
