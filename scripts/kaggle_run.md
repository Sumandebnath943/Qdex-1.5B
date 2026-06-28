# Kaggle Run — Click-by-Click Guide (Qdex-1.5B)

This walks you through running the whole pipeline — the **before** benchmark,
the training, and the **after** benchmark — on Kaggle's **free** GPU, in **one
session**, then downloading the results and shutting down.

> **Total cost: ₹0. Total time: ~3.5 hours (mostly hands-off).**
>
> The only money risk in this whole project is a forgotten running GPU. Kaggle's
> free tier auto-stops and is capped, so you can't actually be charged — but we
> still always stop the session by hand as a good habit.

The cells you'll paste are in [`scripts/kaggle_notebook.py`](kaggle_notebook.py).
Each `===== CELL n =====` block goes in its own Kaggle cell.

---

## Before you start (5 min, on your laptop)

1. The project must be pushed to GitHub first (so Kaggle can clone it). If you
   haven't, tell Claude Code "let's push to GitHub" and follow those steps.
2. Have your Hugging Face token handy (the `hf_...` string) — optional but
   recommended. We'll add it as a Kaggle "Secret," never typed in plain text.
3. Make sure your phone is verified on Kaggle (kaggle.com/settings) — this is
   what unlocks the GPU.

---

## Step 1 — Create the notebook

1. Go to **kaggle.com** → sign in.
2. Top nav: **Create** → **New Notebook**.
3. Rename it (top-left): `qdex-1.5b-finetune`.

## Step 2 — Turn on the GPU (critical)

1. Right sidebar → **Session options** → **Accelerator**.
2. Choose **GPU T4 x1** (just one — we don't need two; the 1.5B model uses only
   ~6 GB of the 16 GB).
3. If prompted, confirm your phone number (one-time, free).
4. Confirm the sidebar shows **GPU T4 x1** with a green dot.

## Step 3 — (Optional) Add your Hugging Face token as a Secret

1. Top menu → **Add-ons** → **Secrets**.
2. **Add a new secret**: Label = `HF_TOKEN`, Value = your `hf_...` token. Save.
3. Make sure its toggle is **on** (attached to the notebook).

   *(Skip this if you like — downloads still work without it, just a little
   slower and occasionally rate-limited.)*

## Step 4 — Paste and run CELL 1 (setup, ~5 min)

1. Open [`scripts/kaggle_notebook.py`](kaggle_notebook.py) on your laptop.
2. Copy the lines under **`===== CELL 1 =====`**. **Remove the leading `# `**
   from each command line so they actually run (they're commented in the file
   only so it stays a valid Python file). If you added the HF Secret, also
   un-comment the two `kaggle_secrets` lines.
3. Paste into the first Kaggle cell and click **▶ Run**.
4. You'll see git clone + pip output. It ends by printing your GPU name (e.g.
   `Tesla T4`). If it prints `NONE`, go back to Step 2.

> **This is the step most likely to need a small tweak**, because Kaggle updates
> its Python/library versions over time and Unsloth is version-sensitive. If you
> see a red error here, **copy the full red text and paste it to Claude Code** —
> we'll adjust one line and re-run. This is expected, not a failure.

## Step 5 — CELL 2: the "before" benchmark (~20 min)

1. New cell → paste the **`===== CELL 2 =====`** commands (un-commented) → Run.
2. It tests the *un-modified* base model two ways and saves two score files.
   Note the two `pass@1` percentages it prints — these are your **before**
   numbers.

## Step 6 — CELL 3: train + merge + export (~3 hours)

1. New cell → paste the **`===== CELL 3 =====`** commands (un-commented) → Run.
2. **Watch the loss.** During training it logs a `loss` number every few steps.
   It should start around **1.0–1.5** and drift **down** toward ~0.5–0.8.
   Falling loss = it's learning. (Send Claude Code a screenshot anytime.)
3. You can close the tab — Kaggle keeps running. Check back every ~30 min.
4. When it finishes it will have written:
   - `outputs/lora_adapter/` (~50 MB) — the trained adapter
   - `outputs/merged/` (~3 GB) — the full fine-tuned model
   - `outputs/gguf/*.gguf` (~1 GB each) — the laptop-ready models

## Step 7 — CELL 4: the "after" benchmark + the table (~10 min)

1. New cell → paste the **`===== CELL 4 =====`** block (un-commented) → Run.
2. It tests the fine-tuned model and prints a **before/after table**. This is
   your headline result. Screenshot it and send it to Claude Code.

## Step 8 — Download your results

In the right panel (the `/kaggle/working/Qdex-1.5B/outputs/` file tree), download:

1. **The GGUF file(s)** — at least `outputs/gguf/...q4_k_m.gguf` (this is what
   your laptop runs). Right-click → **Download**.
2. **The three `bench_*.json`** files (small — they hold the real numbers).
3. *(Optional)* `outputs/lora_adapter/` if you want to re-train later without
   starting over.

## Step 9 — SHUT DOWN the session (do this now)

1. Right sidebar → **Session options** → **Stop session** (or the power icon).
2. Confirm. The GPU is released. **This is the most important habit in the
   project.**

---

## Troubleshooting

| What you see | What to do |
|---|---|
| GPU shows `NONE` | Session options → Accelerator → **GPU T4 x1**. Re-run CELL 1. |
| Red error in CELL 1 (install) | Expected occasionally. Paste the full error to Claude Code — it's a version tweak. |
| `CUDA out of memory` during training | Open `config.py` in the repo... actually easiest: tell Claude Code; we lower `BATCH_SIZE` from 2 to 1 and push, then re-clone. |
| Session disconnected mid-training | Re-run CELL 1, then re-run CELL 3 — training **resumes from the last checkpoint** (saved every 200 steps). |
| Loss is flat / not dropping | Stop and send Claude Code the log — likely a data-format issue, worth catching early. |
| GGUF export fails | It builds llama.cpp and can be flaky. Send Claude Code the error; the merged model is already safe on disk. |

## Cost tracker

| Item | Cost |
|---|---|
| Kaggle GPU (T4 ×1, ~3.5 h) | ₹0 (free tier) |
| Downloads / Hugging Face | ₹0 |
| **Total** | **₹0** |
| Budget remaining (of ₹1,500) | ₹1,500 |

If Kaggle's weekly 30-hour quota is ever exhausted, the Vast.ai fallback
(~₹170) is documented in `HANDOFF.md`. We're nowhere near that.
