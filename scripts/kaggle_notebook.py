"""KAGGLE NOTEBOOK — Full pipeline in one cell (train + merge + GGUF export).

HOW TO USE (detailed steps in scripts/kaggle_run.md):
  1. Create a Kaggle Notebook with GPU = "GPU T4 x1" enabled.
  2. Copy this ENTIRE file's contents into a single notebook cell.
  3. Run the cell. It installs deps, trains, merges, and exports GGUF.
  4. When done, download outputs/lora_adapter/, outputs/merged/, outputs/gguf/
     from the notebook's /kaggle/working/ directory.
  5. SHUT DOWN the notebook session immediately after downloading.

ESTIMATED RUN TIME on T4: ~3-4 hours total (train ~2.5h + merge ~5m + export ~20m).
ESTIMATED COST: ₹0 (Kaggle free tier).

This file is SELF-CONTAINED — it does not import from the project's src/ because
Kaggle notebooks don't have the project files uploaded. All logic is inlined.
The modular scripts/ versions exist for reference and local iteration.
"""
# ============================================================================
# CELL 1 — Install dependencies (runs once, ~3 min)
# ============================================================================
# Unsloth must be installed BEFORE torch/trl in a specific order.
# These commands are for Kaggle's Python 3.10 + Tesla T4 environment.
import subprocess
subprocess.run(["pip", "install", "-q", "unsloth"], check=True)
subprocess.run(["pip", "install", "-q", "--no-deps", "trl", "peft", "accelerate"],
               check=True)
subprocess.run(["pip", "install", "-q", "datasets", "transformers>=4.44",
                "bitsandbytes"], check=True)
print("Dependencies installed.")

# ============================================================================
# CONFIGURATION (mirrors config.py)
# ============================================================================
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DATASET_NAME = "ise-uiuc/Magicoder-OSS-Instruct-75K"
DATASET_SLICE = 20000
SYSTEM_PROMPT = (
    "You are a helpful coding assistant. "
    "Respond with clean, correct, well-structured code and brief explanations."
)
MAX_SEQ_LEN = 2048
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
SAVE_STEPS = 200

# Qwen ChatML template (same as src/chat_template.py)
QWEN_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\\n'}}"
    "{% endif %}"
)

# Kaggle working directory (outputs here get saved with the notebook)
import os
OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# STEP 1 — Load model in 4-bit + attach LoRA
# ============================================================================
import torch
from unsloth import FastLanguageModel

print(f"\n{'='*60}\nLoading {BASE_MODEL} in 4-bit...\n{'='*60}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)
if tokenizer.chat_template is None:
    tokenizer.chat_template = QWEN_CHATML_TEMPLATE
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print("Model loaded + LoRA attached.")

# ============================================================================
# STEP 2 — Load + format dataset
# ============================================================================
from datasets import load_dataset

print(f"\n{'='*60}\nLoading {DATASET_SLICE} examples from {DATASET_NAME}...\n{'='*60}")
ds = load_dataset(DATASET_NAME, split="train").select(range(DATASET_SLICE))

def format_example(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]},
        {"role": "assistant", "content": example["solution"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

formatted = ds.map(format_example, remove_columns=ds.column_names, desc="Formatting")
print(f"Formatted {len(formatted)} examples.")
print(f"Sample (first 200 chars):\n{formatted[0]['text'][:200]}...")

# ============================================================================
# STEP 3 — Train
# ============================================================================
from transformers import TrainingArguments
from trl import SFTTrainer

print(f"\n{'='*60}\nTraining {NUM_EPOCHS} epochs...\n{'='*60}")
use_bf16 = torch.cuda.is_bf16_supported()
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
    args=TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        save_strategy="steps",
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        report_to="none",
        dataloader_num_workers=2,
    ),
)
trainer_stats = trainer.train()

# Save adapter
adapter_path = f"{OUTPUT_DIR}/lora_adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"\nAdapter saved to {adapter_path}")
print(f"Training loss: {trainer_stats.training_loss:.4f}")

# ============================================================================
# STEP 4 — Merge adapter into base (fp16)
# ============================================================================
print(f"\n{'='*60}\nMerging adapter into base model...\n{'='*60}")
merged_path = f"{OUTPUT_DIR}/merged"
model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
print(f"Merged model saved to {merged_path}")

# ============================================================================
# STEP 5 — Export to GGUF (q4_k_m + q5_k_m)
# ============================================================================
print(f"\n{'='*60}\nExporting to GGUF...\n{'='*60}")
gguf_path = f"{OUTPUT_DIR}/gguf"
for method in ["q4_k_m", "q5_k_m"]:
    print(f"  Exporting {method}...")
    model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method=method)

# ============================================================================
# STEP 6 — Summary
# ============================================================================
print(f"\n{'='*60}\nPIPELINE COMPLETE\n{'='*60}")
print(f"Adapter:  {adapter_path}")
print(f"Merged:   {merged_path}")
print(f"GGUF dir: {gguf_path}")
import glob
for f in sorted(glob.glob(f"{gguf_path}/*.gguf")):
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"  {os.path.basename(f)}: {size_mb:.0f} MB")

print("\n>>> ACTION REQUIRED <<<")
print("1. Download the outputs/ folder from /kaggle/working/")
print("   (especially the .gguf files and lora_adapter/)")
print("2. SHUT DOWN this notebook session NOW to free the GPU.")
print("3. Locally: test the GGUF with Ollama (see scripts/local_gguf_test.md)")
print("4. Benchmark: re-run scripts/run_benchmark.py with --model-path outputs/merged")
