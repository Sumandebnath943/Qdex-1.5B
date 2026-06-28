# Stage 3 — Kaggle: Click-by-Click Guide

This guide walks the owner through renting a free GPU on Kaggle, running the
QLoRA training + merge + GGUF export in one session, downloading the results,
and shutting down. **Total cost: ₹0. Total time: ~3-4 hours.**

> **Read this fully before starting.** The main financial risk in this project
> is a forgotten running GPU. Kaggle's free tier enforces a weekly quota
> (30 hrs GPU), so sessions auto-stop — but you should still explicitly stop
> the session when done, as a habit for when we use paid Vast.ai later.

---

## Before you start (on your laptop, ~5 min)

1. Create a free Kaggle account at https://kaggle.com (if you don't have one).
2. Open the file `scripts/kaggle_notebook.py` from this project in a text editor.
   You'll copy its contents in Step 6 below.
3. Make sure you have ~5 GB free disk space for the downloads.

---

## Step 1 — Create a new notebook

1. Go to https://kaggle.com and sign in.
2. Click **Create** (top nav) → **New Notebook**.
3. Name it: `qlora-coder-finetune` (top-left title field).

## Step 2 — Enable the GPU (critical)

1. In the right sidebar, find **Session options** → **Accelerator**.
2. Click the dropdown and select **GPU T4 x1**.
3. A popup may ask you to confirm your phone number for GPU access — follow it.
   (Kaggle requires phone verification once to prevent abuse. It's free.)
4. Confirm the sidebar now shows "GPU T4 x1" with a green dot.

## Step 3 — Set the environment to Python 3.10

1. In the right sidebar, **Environment** → select **Python 3.10**.
   (Unsloth is tested on 3.10; newer versions may have issues.)

## Step 4 — (Optional) Set a notebook timeout

1. In the right sidebar, **Session options** → there's no explicit spending cap
   on Kaggle (it's free), but there IS a 12-hour session limit. Our run needs
   ~4 hours, so this is fine.
2. Note: Kaggle gives **30 hours of GPU per week**. One full run uses ~4 hours.
   If you need to re-run, you have ~7 tries per week.

## Step 5 — Open a code cell

1. The notebook starts with one empty code cell. Click inside it.

## Step 6 — Paste the pipeline script

1. Open `scripts/kaggle_notebook.py` from this project on your laptop.
2. Select ALL the text (Ctrl+A / Cmd+A) and copy it.
3. Paste it into the Kaggle code cell.
4. **Do not edit it** — it's self-contained and configured for Kaggle.

## Step 7 — Run it

1. Click the **▶ Run** button (or Shift+Enter).
2. The first ~3 minutes install dependencies. You'll see pip output. This is normal.
3. Then it loads the model (~2 min), formats data (~1 min), and starts training.
4. Training prints loss every 10 steps. **Watch the loss go down** — it should
   start around 1.0-1.5 and drop to ~0.5-0.8 over the run.
5. If you see `CUDA out of memory`, stop the session, reduce `BATCH_SIZE` to 1
   in the pasted code, and re-run.

## Step 8 — While it trains (~3 hours)

You can close the browser tab — Kaggle keeps running. But:
- Check back every ~30 min to confirm it's still running (no errors).
- If the session disconnects (Kaggle can drop idle sessions), your checkpoints
  are saved every 200 steps in `/kaggle/working/outputs/checkpoints/`. You can
  resume from the latest checkpoint if needed (ask Claude Code for resume steps).

## Step 9 — When training finishes

The cell will print `PIPELINE COMPLETE` and list the output files. Then:

1. In the right sidebar, click **Data** (or look at the `/kaggle/working/` tree).
2. You'll see:
   - `outputs/lora_adapter/` — the trained adapter (~50 MB)
   - `outputs/merged/` — the full merged model (~3 GB)
   - `outputs/gguf/*.gguf` — quantized models for your laptop (~1 GB each)
3. Download the **gguf** files (at least the `q4_k_m` one) — this is what your
   laptop will run. Right-click each file → **Download**.
4. (Optional) Also download `lora_adapter/` if you want to iterate later without
   retraining from scratch.

## Step 10 — SHUT DOWN the session (do this immediately after downloading)

1. In the right sidebar, click **Stop Session** (or the power icon).
2. Confirm. The GPU is now released.
3. **This is the most important step.** A forgotten running session wastes your
   weekly 30-hour GPU quota.

## Step 11 — Verify downloads on your laptop

Move the downloaded `.gguf` file to your project's `outputs/gguf/` folder:
```
qlora-coder-finetune/outputs/gguf/qwen25-coder-1.5b-finetuned-q4_k_m.gguf
```
Then follow `scripts/local_gguf_test.md` to test it with Ollama.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | In the pasted code, change `BATCH_SIZE = 2` to `1`. Re-run. |
| Session disconnected mid-run | Re-open the notebook, re-paste the cell, but add resume logic (ask Claude Code). Checkpoints are in `/kaggle/working/outputs/checkpoints/`. |
| `ModuleNotFoundError: unsloth` | The install cell didn't finish. Re-run the cell from the top. |
| Loss not decreasing | Check the data formatting printout looks right (system/user/assistant blocks). If wrong, stop and ask Claude Code. |
| Run hits 12-hour limit | Shouldn't happen (we need ~4h). If it does, the checkpoints let you resume. |

## Cost tracker

| Item | Cost |
|---|---|
| Kaggle GPU (T4, ~4 hours) | ₹0 (free tier) |
| Internet / downloads | ₹0 |
| **Total Stage 3-6 GPU cost** | **₹0** |
| Budget remaining (of ₹1,500) | ₹1,500 |

If Kaggle fails (e.g., quota exhausted), the Vast.ai fallback costs ~₹170. See
HANDOFF.md for the Vast.ai steps.
