# HANDOFF — Project 2: QLoRA Fine-Tune of a Small Coding Model

**Date:** _(handoff date)_
**From:** Z.ai Code (engineer role)
**To:** Claude Code (next engineer role)
**Owner:** Non-technical; architect + decision-maker. Needs plain-English explanations, one concept at a time, rationale before accepting defaults.

---

## 1. What this project is

Portfolio Project 2 of 3. Take an existing pretrained ~1.5B coding model
(Qwen2.5-Coder-1.5B, the *base* version) and fine-tune it on instruction-style
coding data (Magicoder-OSS-Instruct, 20k slice) using QLoRA, producing a more
capable coding model. Measure before/after with HumanEval pass@1. Export to GGUF
so it runs locally on the owner's 16GB no-GPU laptop. The result will power
Project 3's CLI coding agent.

Project 1 (a from-scratch 47M command model + CLI) is done. Project 3 (the CLI
agent that calls this fine-tuned model) comes next.

---

## 2. Current state — what's DONE

All code is written, syntax-checked, and the testable logic is validated on real
data. **The only things remaining are the actual GPU runs**, which the owner
executes on Kaggle (free T4).

### Validated locally (CPU, no GPU needed):

| Component | File | Status |
|---|---|---|
| Data formatting | `src/format_data.py` | ✅ Validated on real Magicoder examples (75,197 total; fields are `problem`/`solution`, NOT `instruction`/`response`) |
| ChatML template | `src/chat_template.py` | ✅ Tokenizer ships with a ChatML template; our fallback is a safety net |
| Token-length check | `scripts/validate_format.py` | ✅ 0% of 1000 sampled examples exceed 2048 tokens (mean 523, max 1600) |
| Benchmark harness logic | `src/benchmark.py` + `scripts/smoke_test_cpu.py` | ✅ Code extraction, sandboxed execution, pass@k math all tested |
| HumanEval dataset schema | — | ✅ Verified: 164 problems, fields `task_id`/`prompt`/`canonical_solution`/`test`/`entry_point`, `test` defines `def check(candidate):` |

### Built but NOT yet run (need GPU):

| Component | File | Needs |
|---|---|---|
| Base model benchmark (Stage 2) | `scripts/run_benchmark.py --mode completion` | Kaggle T4, ~10 min |
| QLoRA training (Stage 3) | `scripts/kaggle_notebook.py` | Kaggle T4, ~3-4 hours |
| LoRA merge (Stage 4) | runs inside kaggle_notebook.py | (same session as training) |
| GGUF export (Stage 6) | runs inside kaggle_notebook.py | (same session as training) |
| Fine-tuned benchmark (Stage 5) | `scripts/run_benchmark.py --mode instruction --model-path outputs/merged` | Kaggle T4, ~10 min |
| Local GGUF test | `scripts/local_gguf_test.md` | Owner's laptop (no GPU) |

---

## 3. Stage 0 decisions (locked by owner)

1. **Base model:** `Qwen/Qwen2.5-Coder-1.5B` (BASE, not Instruct). Apache 2.0.
   Rationale: we do the instruction-tuning ourselves for an honest before/after.
2. **Dataset:** `ise-uiuc/Magicoder-OSS-Instruct-75K`, 20k slice.
   Rationale: highest-quality free instruction-coding set.
3. **Tooling:** Unsloth. Rationale: 2× faster QLoRA, beginner-friendly.
4. **Platform:** Kaggle free tier (T4 16GB) primary; Vast.ai fallback.
   Rationale: 1.5B QLoRA needs only ~6GB VRAM; Kaggle is free.

---

## 4. Critical boundaries (owner-controlled, NOT yours)

- **You cannot rent GPUs, enter payment details, or set spending caps.** When a
  GPU is needed, prepare everything and give the owner click-by-click steps.
  `scripts/kaggle_run.md` is already written for this.
- **Always recommend the cheapest GPU.** Kaggle free is cheapest (₹0). Vast.ai
  RTX 3060 (~₹170 total) is the fallback.
- **Always remind the owner to shut down the GPU when a run finishes.** A
  forgotten running GPU is the main financial risk.
- **Budget: under ₹1,500 total.** Currently at ₹0 (Kaggle free). Flag immediately
  if any plan would exceed.
- **Report real numbers only.** Never invent or round up benchmark scores.
- **State plainly that this is a ~1.5B model with real limits.**

---

## 5. Next steps (in order)

### Step 1 — Owner runs the Kaggle pipeline (Stage 3 + 4 + 6)
The owner follows `scripts/kaggle_run.md` to paste `scripts/kaggle_notebook.py`
into a Kaggle T4 notebook. This trains, merges, and exports GGUF in one session
(~3-4 hours, ₹0). Owner downloads `outputs/gguf/*.gguf` + `outputs/lora_adapter/`
and shuts down the session.

**Your role:** Be ready to debug via pasted logs. Common issues:
- `CUDA out of memory` → reduce `BATCH_SIZE` from 2 to 1 in the notebook.
- Session disconnects → checkpoints are in `/kaggle/working/outputs/checkpoints/`;
  add resume logic if needed.
- Loss not decreasing → check the data formatting printout.

### Step 2 — Run the base model benchmark (Stage 2)
On a fresh Kaggle T4 session (~10 min):
```bash
pip install -q transformers datasets torch tqdm
python scripts/run_benchmark.py \
    --model-path Qwen/Qwen2.5-Coder-1.5B \
    --mode completion \
    --n-samples 1 \
    --temperature 0.0 \
    --output outputs/bench_base.json
```
Record the `pass_at_1` number. Shut down the session.

### Step 3 — Run the fine-tuned benchmark (Stage 5)
Upload `outputs/merged/` to Kaggle, then on a T4 session (~10 min):
```bash
python scripts/run_benchmark.py \
    --model-path outputs/merged \
    --mode instruction \
    --n-samples 1 \
    --temperature 0.0 \
    --output outputs/bench_finetuned.json
```
Record the `pass_at_1` number. Shut down the session.

### Step 4 — Test GGUF locally (Stage 6 cont.)
Owner follows `scripts/local_gguf_test.md` to load the .gguf in Ollama and test
generation. Record speed (tokens/sec) and qualitative quality.

### Step 5 — Finalize (Stage 7)
- Fill in real numbers in `PROJECT_REPORT.md` and `README.md`.
- Write the honest limitations section (a 1.5B model is small; good at scoped
  tasks, not frontier-level).
- If the fine-tune barely moved the score, diagnose honestly (data quality, too
  few steps, format mismatch) rather than overclaiming.
- Final git commit.

---

## 6. Key design decisions (rationale for Claude Code)

### Why benchmark in two different modes (completion vs instruction)
- **Base model** is tested in `completion` mode (raw HumanEval prompt → continuation).
  This matches how the Qwen2.5-Coder paper reports scores (~41.1%), giving an
  honest, comparable baseline.
- **Fine-tuned model** is tested in `instruction` mode (chat-wrapped prompt →
  code-block response). This matches how Project 3's agent will actually use it.
- This is defensible (each model tested in its designed-for mode) but NOT purely
  apples-to-apples. Documented in `src/benchmark.py` and the README.
- If the owner wants a stricter comparison, add `--mode instruction` for the base
  model too (it'll score low because it can't follow chat format — expected).

### Why greedy pass@1 (n=1, temp=0) as the default
- Deterministic, reproducible, fast (~10 min vs ~3.5 hours for n=20 sampling).
- Standard for quick model comparisons. The HumanEval paper's pass@1 uses n=20,
  but greedy pass@1 is widely reported and sufficient for a portfolio before/after.
- `--n-samples 20 --temperature 0.2` is available if a more rigorous measurement
  is wanted later (but it's ~20× slower).

### Why 20k examples and 3 epochs
- 20k is enough to learn the assistant format and nudge coding ability.
- 3 epochs is conservative; over-fitting is a risk with more epochs on a small model.
- Training time on T4: ~2.5 hours for 20k × 3 epochs. Fits comfortably in one
  Kaggle session.

### Why q4_k_m and q5_k_m GGUF quantizations
- q4_k_m: ~1 GB, best quality/size tradeoff. This is what Project 3 should use.
- q5_k_m: ~1.3 GB, slightly better quality. Optional alternative.
- q8_0 (near-lossless, ~1.6 GB) is available via `quantization_method="q8_0"` if needed.

---

## 7. Known gotchas

1. **Magicoder field names are `problem`/`solution`**, NOT `instruction`/`response`.
   This was caught by Stage 1 validation. Don't "fix" them back.

2. **The base model tokenizer DOES ship a chat_template** (contrary to some docs).
   Our fallback in `src/chat_template.py` is a safety net, not the primary path.

3. **Unsloth install order matters.** The Kaggle notebook installs unsloth first,
   then trl/peft/accelerate with `--no-deps` to avoid version conflicts. Don't
   reorder the pip installs.

4. **Kaggle's `/kaggle/working/` is the only persistent directory.** Outputs there
   get saved with the notebook. Outputs in `/tmp/` are lost when the session stops.

5. **Kaggle sessions can disconnect after ~12 hours or when idle.** Checkpoints
   every 200 steps (SAVE_STEPS in config.py) limit loss to ~10 min of work.

6. **The benchmark runs generated code in a subprocess with a 10s timeout.** This
   is standard but not Docker-level isolation. Don't run the benchmark on
   untrusted model outputs. (Our model is fine-tuned by us, so this is safe.)

7. **Ollama Modelfile needs the exact GGUF filename.** The owner must edit
   `scripts/local_gguf_test.md`'s Modelfile example to match the actual downloaded
   filename.

---

## 8. Vast.ai fallback (if Kaggle quota is exhausted)

If Kaggle's 30 hrs/week GPU quota runs out:
1. Go to https://vast.ai, create an account, add $5 credit (₹~420).
2. Search for "RTX 3060 12GB" instances (~$0.15/hr).
3. **Set a spending cap** (e.g., $3) before launching.
4. Launch an instance with the PyTorch template.
5. `git clone` the project (or scp it), then run the same scripts.
6. **Shut down the instance immediately after the run.**
7. Total cost: ~₹170 for 4 hours. Still well under ₹1,500.

---

## 9. File-by-file index

| File | Purpose | Tested? |
|---|---|---|
| `config.py` | All hyperparameters + paths in one place | ✅ syntax |
| `src/chat_template.py` | Qwen ChatML Jinja template + rationale | ✅ syntax |
| `src/format_data.py` | Magicoder → chat-templated text | ✅ validated on real data |
| `src/benchmark.py` | HumanEval pass@1 harness (extract/exec/score) | ✅ CPU smoke test |
| `scripts/validate_format.py` | Stage 1: print formatted examples + token stats | ✅ runs |
| `scripts/smoke_test_cpu.py` | Stage 2: test harness logic without a model | ✅ runs, all pass |
| `scripts/run_benchmark.py` | Stage 2/5 CLI: benchmark any model | ⬜ needs GPU |
| `scripts/train_qlora.py` | Stage 3 modular training script | ⬜ needs GPU |
| `scripts/kaggle_notebook.py` | Stage 3-6 self-contained Kaggle pipeline | ⬜ needs GPU |
| `scripts/merge_lora.py` | Stage 4 merge (modular version) | ⬜ needs GPU |
| `scripts/export_gguf.py` | Stage 6 GGUF export (modular version) | ⬜ needs GPU |
| `scripts/kaggle_run.md` | Click-by-click Kaggle guide for owner | n/a |
| `scripts/local_gguf_test.md` | Ollama local test guide | n/a |
| `scripts/full_pipeline.md` | Master checklist | n/a |
| `requirements-stage1.txt` | Laptop deps (datasets, transformers) | ✅ installed |
| `requirements-kaggle.txt` | GPU deps (torch, unsloth, trl, peft, bitsandbytes) | ⬜ install on Kaggle |

---

## 10. How to talk to the owner

- Plain English, one concept at a time.
- Explain the "why" before accepting any default.
- Wait for confirmation between stages.
- Teach concepts as they come up (pretraining vs fine-tuning, QLoRA, quantization,
  what a benchmark measures).
- When a GPU is needed: prepare everything, then give exact click-by-click steps.
  Never assume a GPU is running. Always remind to shut it down.
- If any result is disappointing, say so honestly and diagnose — never overclaim.
- Budget is ₹1,500 total; currently at ₹0. Flag any plan that would exceed.

---

## 11. Git state

```
Stage 1: project scaffold + Qwen ChatML data formatting (validated on real Magicoder examples)
Stage 2-7: benchmark harness + QLoRA training + merge + GGUF export + docs (all built, awaiting GPU runs)
```

The repo is local (not yet pushed to GitHub). If the owner wants to push it for
Kaggle access or portfolio display, `git remote add origin <url>` + `git push`.
