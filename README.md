# QLoRA Fine-Tune of a Small Coding Model

**Portfolio Project 2** — take an existing pretrained ~1.5B coding model and
fine-tune it on instruction-style coding data using QLoRA, producing a more
capable coding model that will later power a CLI coding agent (Project 3).

> **Status:** All code built and locally validated. Awaiting GPU runs on Kaggle.
> See [HANDOFF.md](./HANDOFF.md) for the full state and next steps.

---

## The goal (plain English)

Take **Qwen2.5-Coder-1.5B** — a small open-weight coding model that knows how to
*continue* code but cannot *follow instructions* — and teach it to answer coding
questions like an assistant. We do this with **QLoRA**, a technique that lets us
fine-tune on a cheap GPU. We measure the model's coding ability **before and after**
with a standard benchmark (HumanEval), export the result to a format that runs on a
16GB no-GPU laptop, and package it for Project 3's CLI agent to call.

This is a **~1.5B parameter model.** It is capable of scoped, well-defined coding
tasks. It is **not** a Claude/GPT replacement and never will be — that is an honest
limitation, not a failure.

---

## Stage 0 decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| **Base model** | `Qwen/Qwen2.5-Coder-1.5B` (base, not Instruct) | Strongest small open coder in its class; Apache 2.0 (verified) permits fine-tuning + portfolio use. We instruction-tune the base ourselves for an honest before/after. |
| **Dataset** | `ise-uiuc/Magicoder-OSS-Instruct-75K`, 20k slice | Highest-quality free instruction-coding set; grounded in real OSS code; proven by the Magicoder paper. 20k is enough to see improvement, small enough to train cheaply. |
| **Tooling** | Unsloth | 2× faster / ~40% less memory QLoRA on single GPU; beginner-friendly; first-class Qwen2.5 + GGUF support. |
| **Platform** | Kaggle free tier (T4 16GB) primary; Vast.ai fallback | 1.5B QLoRA needs only ~6GB VRAM, fits Kaggle's free T4. Total project cost: ~₹0 (Kaggle) or ~₹170 (Vast.ai fallback). Well under the ₹1,500 budget. |

---

## Pipeline (stages)

- [x] **Stage 1 — Local prep.** Project scaffold, git repo, data-loading + Qwen ChatML formatting. Validated on real examples, no GPU.
- [x] **Stage 2 (code) — Benchmark harness.** HumanEval pass@1 harness built + CPU-smoke-tested. **Run needs GPU.**
- [x] **Stage 3 (code) — QLoRA training script.** Complete Unsloth script + self-contained Kaggle notebook. **Run needs GPU.**
- [x] **Stage 4 (code) — Merge script.** LoRA→base merge. **Runs inside the Kaggle notebook.**
- [x] **Stage 5 (code) — Fine-tuned benchmark.** Reuses Stage 2 harness with `--mode instruction`. **Run needs GPU.**
- [x] **Stage 6 (code) — GGUF export.** Unsloth GGUF export (q4_k_m + q5_k_m). **Runs inside the Kaggle notebook.**
- [ ] **Stage 7 — Finalize.** Fill in real numbers after GPU runs; write honest limitations.

> **What's done vs. what needs a GPU:** All code is written, syntax-checked, and
> the testable logic (data formatting + benchmark harness) is validated on real
> data. The only things remaining are the actual GPU runs (training + benchmarks),
> which the owner executes on Kaggle following `scripts/kaggle_run.md`.

---

## Project structure

```
qlora-coder-finetune/
├── README.md                       ← this file
├── HANDOFF.md                      ← state + next steps for Claude Code
├── PROJECT_REPORT.md               ← portfolio report (numbers filled after runs)
├── config.py                       ← all settings in one place
├── requirements-stage1.txt         ← lightweight deps (laptop, no GPU)
├── requirements-kaggle.txt         ← heavy deps (GPU, Stage 3+)
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── chat_template.py            ← Qwen2.5 ChatML template
│   ├── format_data.py              ← Magicoder → chat template formatting
│   └── benchmark.py                ← HumanEval pass@1 harness (extract/exec/score)
├── scripts/
│   ├── validate_format.py          ← Stage 1 test (CPU, validated ✓)
│   ├── smoke_test_cpu.py           ← Stage 2 harness test (CPU, validated ✓)
│   ├── run_benchmark.py            ← Stage 2/5 CLI: run HumanEval on a model
│   ├── train_qlora.py              ← Stage 3 modular training script
│   ├── kaggle_notebook.py          ← Stage 3-6 self-contained Kaggle pipeline
│   ├── merge_lora.py               ← Stage 4 merge script
│   ├── export_gguf.py              ← Stage 6 GGUF export script
│   ├── kaggle_run.md               ← click-by-click Kaggle guide
│   ├── local_gguf_test.md          ← how to test GGUF in Ollama
│   └── full_pipeline.md            ← master checklist
├── data/                           ← (downloaded at runtime, gitignored)
└── outputs/                        ← (checkpoints + models, gitignored)
```

---

## How to run (quick reference)

### On your laptop (free, no GPU) — already validated

```bash
cd qlora-coder-finetune
pip install -r requirements-stage1.txt
python scripts/validate_format.py    # Stage 1: data formatting
python scripts/smoke_test_cpu.py     # Stage 2: benchmark harness logic
```

### On Kaggle (free T4 GPU) — the actual runs

1. Follow `scripts/kaggle_run.md` to run training + merge + GGUF export.
2. Run the base benchmark (Stage 2) and fine-tuned benchmark (Stage 5) on Kaggle.
3. Download results, test GGUF locally via `scripts/local_gguf_test.md`.
4. Fill in real numbers in `PROJECT_REPORT.md`.

See `scripts/full_pipeline.md` for the complete checklist.

---

## Benchmark numbers

> **These will be filled in after the GPU runs. No numbers are invented.**

| Model | Mode | pass@1 | Notes |
|---|---|---|---|
| Qwen2.5-Coder-1.5B (base) | completion | _TBD_ | Stage 2 result |
| Qwen2.5-Coder-1.5B (fine-tuned) | instruction | _TBD_ | Stage 5 result |

For reference, the Qwen2.5-Coder paper reports ~41.1% for the 1.5B base model on
HumanEval. Our measured number may differ slightly (different eval setup).

---

## Honest limitations

- A 1.5B model is **small.** Good at scoped tasks (write this function, fix this bug),
  poor at multi-file reasoning or large refactors.
- We fine-tune on 20k examples for ~3 epochs — enough to learn the assistant format
  and nudge coding ability, not enough to rival frontier models.
- HumanEval pass@1 is one benchmark. Real-world coding ability is broader than it measures.
- The base and fine-tuned models are benchmarked in *different modes* (completion vs
  instruction), because each is tested in the mode it was designed for. This is a
  defensible choice but means the comparison is not purely apples-to-apples. See
  `src/benchmark.py` docstring for the full rationale.
- All benchmark numbers in this README are **real and measured**, never invented or rounded up.
- If the fine-tune barely moves the score, we will say so and diagnose why (data quality,
  too few steps, format mismatch) rather than overclaiming.

---

## License

- Base model (Qwen2.5-Coder-1.5B): **Apache 2.0** — permits fine-tuning and commercial/portfolio use.
- Dataset (Magicoder-OSS-Instruct-75K): MIT-licensed for use.
- Code in this repo: MIT.
