"""KAGGLE NOTEBOOK CELLS — Qdex-1.5B full pipeline (clone-and-run, one session).

This file is a REFERENCE for the cells you paste into a Kaggle notebook. Each
block marked "# ===== CELL n =====" goes into its OWN Kaggle cell, run in order.
The step-by-step screenshot guide is in scripts/kaggle_run.md.

Why clone-and-run (not one giant pasted blob):
  The notebook clones THIS repo and runs the real, verified scripts
  (scripts/train_qlora.py, run_benchmark.py, merge_lora.py, export_gguf.py).
  That means the code on Kaggle is identical to the code we tested locally —
  nothing to keep in sync by hand.

Everything runs in ONE Kaggle session (~3.5 hours total, INR 0 on the free T4):
  CELL 1  setup (clone + install)                      ~5 min
  CELL 2  benchmark BASE model (the honest "before")   ~20 min
  CELL 3  train + merge + export to GGUF               ~3 hours
  CELL 4  benchmark FINE-TUNED model (the "after")     ~10 min

After CELL 4: download outputs/ (the .gguf files + the bench_*.json numbers),
then SHUT DOWN the session. A forgotten GPU is the only money risk.
"""

# ============================================================================
# ===== CELL 1 ===== Setup: clone the repo + install GPU libraries (~5 min)
# ============================================================================
# (In Kaggle these lines with '!' and '%' are notebook magics — paste as-is.)

# !git clone https://github.com/Sumandebnath943/Qdex-1.5B.git
# %cd Qdex-1.5B

# Kaggle's GPU options are "GPU T4 x2" or "GPU P100". Choose **GPU T4 x2**:
# the T4 is the chip our stack (Unsloth + 4-bit) is built for; P100 is older and
# unsupported by Unsloth. We pin to ONE of the two T4s so the trainer doesn't try
# to split across both (which breaks with quantized models). This env var is set
# before importing torch and is inherited by every later `!python ...` step.
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# OPTIONAL — avoid Hugging Face download rate limits. The model + dataset are
# public (no token strictly required), but a token gives faster, reliable
# downloads. SAFEST way: add it as a Kaggle Secret named HF_TOKEN
#   (Notebook -> Add-ons -> Secrets -> add HF_TOKEN), then:
# from kaggle_secrets import UserSecretsClient
# os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

# Install order matters: Unsloth first (it patches torch), then the rest.
# Kaggle already ships a CUDA-enabled torch, so we don't reinstall it.
# !pip install -q unsloth
# !pip install -q --no-deps trl peft accelerate
# !pip install -q datasets "transformers>=4.44" bitsandbytes tqdm

# import torch
# print("Visible GPUs:", torch.cuda.device_count(), "(should be 1 after the pin)")
# print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
#       else "NONE — set Accelerator to 'GPU T4 x2'!")


# ============================================================================
# ===== CELL 2 ===== Benchmark the BASE model — the honest "before" (~20 min)
# ============================================================================
# We measure the base model TWICE, on purpose:
#   - completion mode : how the research paper reports it (~41% expected).
#   - instruction mode: the SAME mode the fine-tuned model is judged in, so the
#                       before/after is a fair, apples-to-apples comparison.

# !python scripts/run_benchmark.py \
#     --model-path Qwen/Qwen2.5-Coder-1.5B --mode completion \
#     --n-samples 1 --temperature 0.0 \
#     --output outputs/bench_base_completion.json

# !python scripts/run_benchmark.py \
#     --model-path Qwen/Qwen2.5-Coder-1.5B --mode instruction \
#     --n-samples 1 --temperature 0.0 \
#     --output outputs/bench_base_instruction.json


# ============================================================================
# ===== CELL 3 ===== Train + merge + export to GGUF (~3 hours)
# ============================================================================
# Each script reloads from disk and is independently re-runnable, so if one step
# fails you re-run just that step (checkpoints are saved every 200 training
# steps in outputs/checkpoints/).

# !python scripts/train_qlora.py     # QLoRA fine-tune -> outputs/lora_adapter/
# !python scripts/merge_lora.py      # merge adapter into base -> outputs/merged/
# !python scripts/export_gguf.py     # quantize to GGUF -> outputs/gguf/*.gguf


# ============================================================================
# ===== CELL 4 ===== Benchmark the FINE-TUNED model — the "after" (~10 min)
# ============================================================================
# !python scripts/run_benchmark.py \
#     --model-path outputs/merged --mode instruction \
#     --n-samples 1 --temperature 0.0 \
#     --output outputs/bench_finetuned_instruction.json

# Print the before/after table:
# import json
# rows = [
#     ("BASE  (completion)",  "outputs/bench_base_completion.json"),
#     ("BASE  (instruction)", "outputs/bench_base_instruction.json"),
#     ("Qdex  (instruction)", "outputs/bench_finetuned_instruction.json"),
# ]
# print(f"{'model':22} {'passed':>8}   pass@1")
# for label, path in rows:
#     d = json.load(open(path))
#     print(f"{label:22} {d['passed_problems']:>3}/{d['total_problems']:<3}   "
#           f"{d['pass_at_1']*100:5.1f}%")

# >>> THEN: download the outputs/ folder (.gguf files + bench_*.json),
# >>> and SHUT DOWN the session (Session options -> Stop session).
