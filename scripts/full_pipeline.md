# Full Pipeline — Master Checklist

This is the complete end-to-end flow. Each stage links to its script/guide.
Check off each step. **Never skip the "shut down GPU" steps.**

---

## Stage 1 — Local prep ✅ (DONE)

- [x] Project folder + git repo created
- [x] Data formatting code written (`src/format_data.py`)
- [x] Validated on real Magicoder examples (`scripts/validate_format.py`)
- [x] Token-length sanity check passed (0% over 2048 limit)

## Stage 2 — Benchmark the BASE model ⬜ (needs GPU)

- [x] Benchmark harness built (`src/benchmark.py`)
- [x] CLI runner built (`scripts/run_benchmark.py`)
- [x] CPU smoke test passed (`scripts/smoke_test_cpu.py`)
- [ ] **Run on Kaggle:** base model, completion mode, greedy pass@1
      ```bash
      # In a Kaggle notebook (T4):
      pip install -q transformers datasets torch tqdm
      python scripts/run_benchmark.py \
          --model-path Qwen/Qwen2.5-Coder-1.5B \
          --mode completion \
          --n-samples 1 \
          --temperature 0.0 \
          --output outputs/bench_base.json
      ```
      (~10 min on T4)
- [ ] Record the base pass@1 number in `PROJECT_REPORT.md`
- [ ] Shut down the Kaggle session

## Stage 3 — QLoRA training ⬜ (needs GPU)

- [x] Training script built (`scripts/train_qlora.py`)
- [x] Self-contained Kaggle notebook built (`scripts/kaggle_notebook.py`)
- [x] Kaggle guide written (`scripts/kaggle_run.md`)
- [ ] **Follow `scripts/kaggle_run.md`** to run training + merge + GGUF export
      in one Kaggle session (~3-4 hours)
- [ ] Watch loss decrease during training (should drop from ~1.0 to ~0.5-0.8)
- [ ] Download outputs (gguf files + lora_adapter) to laptop
- [ ] **Shut down the Kaggle session immediately**

## Stage 4 — Merge ⬜ (done inside the Kaggle notebook)

- [x] Merge script built (`scripts/merge_lora.py`)
- [ ] Merge runs as Step 4 of `scripts/kaggle_notebook.py`
- [ ] Merged model saved to `outputs/merged/`

## Stage 5 — Benchmark the FINE-TUNED model ⬜ (needs GPU)

- [x] Reuses `scripts/run_benchmark.py` from Stage 2
- [ ] **Run on Kaggle:** merged model, instruction mode, greedy pass@1
      ```bash
      # In a Kaggle notebook (T4), after uploading outputs/merged/:
      python scripts/run_benchmark.py \
          --model-path outputs/merged \
          --mode instruction \
          --n-samples 1 \
          --temperature 0.0 \
          --output outputs/bench_finetuned.json
      ```
      (~10 min on T4)
- [ ] Compare before → after in `PROJECT_REPORT.md`
- [ ] **Shut down the Kaggle session**

## Stage 6 — Export + test GGUF locally ⬜

- [x] GGUF export script built (`scripts/export_gguf.py`)
- [ ] Export runs as Step 5 of `scripts/kaggle_notebook.py` (q4_k_m + q5_k_m)
- [ ] Download .gguf files to laptop
- [ ] **Follow `scripts/local_gguf_test.md`** to test with Ollama
- [ ] Record speed (tokens/sec) + qualitative quality in `PROJECT_REPORT.md`

## Stage 7 — Finalize README + report ⬜

- [ ] Fill in real benchmark numbers in `PROJECT_REPORT.md`
- [ ] Fill in real numbers in `README.md`
- [ ] Write honest limitations section
- [ ] Final git commit

---

## Cost tracker

| Stage | GPU hours | Cost |
|---|---|---|
| Stage 2 (base benchmark) | ~0.2 | ₹0 (Kaggle) |
| Stage 3 (training) | ~4 | ₹0 (Kaggle) |
| Stage 5 (finetuned benchmark) | ~0.2 | ₹0 (Kaggle) |
| **Total** | **~4.4** | **₹0** |
| Budget (₹1,500) | — | ₹1,500 remaining |

If Kaggle quota is exhausted, Vast.ai fallback: ~₹170 total (see HANDOFF.md).
