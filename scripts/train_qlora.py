"""Stage 3: QLoRA fine-tuning of Qwen2.5-Coder-1.5B on Magicoder-OSS-Instruct.

This is the canonical, modular training script. For the actual Kaggle run, use
scripts/kaggle_notebook.py (self-contained, paste into one cell). This file is
kept for reference, local iteration, and code review.

What this script does (in order):
  1. Load the base model in 4-bit (QLoRA quantization) via Unsloth.
  2. Attach LoRA adapters to all linear layers (only these get trained).
  3. Load + format 20k Magicoder examples into the Qwen ChatML template.
  4. Train for 3 epochs with conservative hyperparameters, checkpointing every
     200 steps so a dropped session loses at most ~10 minutes of work.
  5. Save the LoRA adapter (small, ~50MB) to outputs/lora_adapter/.

REQUIRES: a GPU with >= 8GB VRAM. Kaggle T4 (16GB) works. No GPU = no run.
Run time on T4: ~2-3 hours for 20k examples x 3 epochs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import TrainingArguments  # noqa: E402
from trl import SFTTrainer  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402

from config import (  # noqa: E402
    BASE_MODEL, DATASET_NAME, DATASET_SLICE, MAX_SEQ_LEN,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
    NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE,
    WARMUP_RATIO, SAVE_STEPS, OUTPUTS_DIR,
)
from src.chat_template import QWEN_CHATML_TEMPLATE  # noqa: E402
from src.format_data import format_example, load_raw_dataset  # noqa: E402


def main():
    print("=" * 60)
    print("STAGE 3: QLoRA FINE-TUNING")
    print("=" * 60)
    print(f"Base model:  {BASE_MODEL}")
    print(f"Dataset:     {DATASET_NAME} ({DATASET_SLICE} examples)")
    print(f"LoRA:        rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"Training:    {NUM_EPOCHS} epochs, lr={LEARNING_RATE}, "
          f"batch={BATCH_SIZE}x{GRAD_ACCUM}")
    print(f"Output:      {OUTPUTS_DIR / 'lora_adapter'}")
    print(f"GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE!'}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. QLoRA training requires a GPU (>= 8GB VRAM). "
            "Run this on Kaggle (T4) or another GPU instance. "
            "See scripts/kaggle_run.md for steps."
        )

    # ---- 1. Load base model in 4-bit ----
    print("\n[1/5] Loading base model in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,            # auto-select fp16/bf16 based on GPU
        load_in_4bit=True,
    )
    # Attach ChatML template (base model may lack one).
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
    print(f"[3/5] Loading {DATASET_SLICE} examples from {DATASET_NAME}...")
    ds = load_raw_dataset().select(range(DATASET_SLICE))
    formatted = ds.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=ds.column_names,
        desc="Formatting",
    )
    print(f"  Formatted {len(formatted)} examples. Sample token_len: "
          f"{formatted[0]['token_len']}")

    # ---- 4. Train ----
    print("[4/5] Starting training...")
    use_bf16 = torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,         # Magicoder examples vary in length; no packing
        args=TrainingArguments(
            output_dir=str(OUTPUTS_DIR / "checkpoints"),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=SAVE_STEPS,
            save_total_limit=3,           # keep only 3 newest checkpoints
            save_strategy="steps",
            fp16=not use_bf16,
            bf16=use_bf16,
            optim="adamw_8bit",           # 8-bit optimizer saves memory
            weight_decay=0.01,
            max_grad_norm=1.0,
            seed=42,
            report_to="none",             # no wandb/tensorboard for simplicity
            dataloader_num_workers=2,
        ),
    )
    trainer.train()
    # Print final training stats from the log history.
    if trainer.state.log_history:
        last = trainer.state.log_history[-1]
        print(f"\n  Final loss: {last.get('train_loss', 'N/A')}")

    # ---- 5. Save adapter ----
    print(f"[5/5] Saving LoRA adapter to {OUTPUTS_DIR / 'lora_adapter'}...")
    model.save_pretrained(str(OUTPUTS_DIR / "lora_adapter"))
    tokenizer.save_pretrained(str(OUTPUTS_DIR / "lora_adapter"))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Adapter saved: {OUTPUTS_DIR / 'lora_adapter'}")
    print("Next: run scripts/merge_lora.py to merge the adapter into the base model.")
    print("REMINDER: If on a rented GPU, download results then SHUT DOWN the instance.")


if __name__ == "__main__":
    main()
