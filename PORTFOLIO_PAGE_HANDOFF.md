# HANDOFF — Build the "Qdex-1.5B" model page on my portfolio website

**You (Claude Code) are connected to my portfolio website's project folder.** Your job is to add a new section to my site and build a detailed landing page for an AI model I trained, called **Qdex-1.5B**. This page is about the **model** (not a generic "project") — it should read like a model's home page (think a polished Hugging Face model card crossed with a product landing page).

I am non-technical on implementation — explain what you're doing in plain English, work in small steps, and confirm with me before large changes. All the model facts and draft copy you need are in this document; you do **not** need to ask me about the model itself.

---

## PART A — What to do on the website

1. **Inspect the project first.** Detect the framework (it's probably **Next.js / React**, but confirm by looking). Find how the existing **Portfolio** menu and existing portfolio/project pages are built (routing, components, styling, theme). **Match those conventions exactly** — same components, same styling system, same layout patterns. Do not introduce a new design language.

2. **Add a new submenu.** Under the existing **Portfolio** menu item, add a child item called **LLMS**. Under **LLMS**, this Qdex-1.5B page is the first entry (build it so more models can be added later — e.g. an LLMS index/listing page plus individual model pages).

3. **Create the model page.** A dedicated, detailed landing page for **Qdex-1.5B** (suggested route like `/portfolio/llms/qdex-1.5b`, but follow the existing structure). Use the content in PART B. It must be responsive and match the site theme.

4. **Calls-to-action (buttons), prominently placed:**
   - **Download the model** → Hugging Face GGUF repo: `https://huggingface.co/SumanDebnath943/Qdex-1.5B-GGUF`
   - **View code** → GitHub: `https://github.com/Sumandebnath943/Qdex-1.5B`
   - (Optional) **Model card on Hugging Face** → same HF repo.

5. **Suggested visual:** a simple **before/after bar chart** showing instruction-mode HumanEval pass@1: **Base 1.2% → Qdex 42.1%** (and optionally the 40.2% base-completion reference line). Plus a clean **spec table**. Use whatever charting/style approach already exists in the repo; if none, a styled HTML/CSS bar is fine — keep it lightweight.

6. **Show me a local preview** before we consider it done, and let me confirm copy/layout.

---

## PART B — Page content (draft copy — use and adapt to the site's voice)

### Hero
- **Title:** Qdex-1.5B
- **Subtitle:** A 1.5B-parameter coding model, instruction-tuned with QLoRA to follow coding requests — small enough to run locally on a 16GB laptop with no GPU.
- **Badges/chips:** `1.5B params` · `QLoRA fine-tune` · `Base: Qwen2.5-Coder-1.5B (Apache-2.0)` · `GGUF (runs on CPU)` · `English`
- **Buttons:** Download (GGUF) · View on Hugging Face · View code on GitHub

### TL;DR (headline result)
Fine-tuning took this model from **1.2% → 42.1%** on HumanEval (pass@1) in instruction-following mode — a ~35× improvement. In doing so it **matched and slightly edged the base model's raw coding ability** (42.1% vs the base model's 40.2% in raw-completion mode), while making that ability actually usable through natural instructions. The fine-tune didn't just add knowledge; it taught the model to *answer when asked* — and kept all of its coding skill in the process.

### What it is
Qdex-1.5B is an instruction-tuned coding assistant built on Alibaba's **Qwen2.5-Coder-1.5B** (the *base*, pre-trained version — Apache-2.0 licensed). The base model already "knew" a lot of code but only knew how to *continue text*; it couldn't reliably respond to a request like "write a function that…". I instruction-tuned it so it behaves like a proper coding assistant, and exported it to **GGUF** so it runs on an ordinary laptop (no GPU) via Ollama or llama.cpp.

### Before & after (the honest numbers)
Benchmark: **HumanEval**, `pass@1`, greedy decoding (temperature 0.0), 1 sample/problem, 164 problems. The base model was measured **two ways** on purpose, so the before/after is a fair, apples-to-apples comparison.

| Model | How it's prompted | HumanEval pass@1 |
|---|---|---|
| Qwen2.5-Coder-1.5B (base) | raw completion | 40.2% (66/164) |
| Qwen2.5-Coder-1.5B (base) | **instruction** | **1.2% (2/164)** |
| **Qdex-1.5B (this model)** | **instruction** | **42.1% (69/164)** |

**What this means:** the base model could code (40.2% in completion mode — which also matches the published paper, confirming the test harness is sound), but was nearly useless when actually *asked* a question (1.2%). After fine-tuning, Qdex answers coding requests at **42.1%** — matching (and slightly beating) the model's full raw ability, now accessible through instructions.

### How it was built
**Method: QLoRA** (Quantized Low-Rank Adaptation).
- **Q (4-bit quantization):** the base model is loaded in 4-bit, shrinking its memory footprint ~4× so it fits on a small free GPU.
- **LoRA (adapters):** the entire base model is frozen; only tiny "adapter" matrices are trained — **18.5M of 1.56B parameters (1.18%)**. This is what makes fine-tuning cheap and fast.
- **Tooling:** Unsloth (fast single-GPU QLoRA) + Hugging Face TRL (supervised fine-tuning) + llama.cpp (GGUF export).
- **Pipeline:** benchmark the base model → instruction-tune with QLoRA → merge the adapter into the base → benchmark the fine-tuned model → export to GGUF for local use.

### Training details
| | |
|---|---|
| **Base model** | Qwen/Qwen2.5-Coder-1.5B (Apache-2.0) |
| **Dataset** | `ise-uiuc/Magicoder-OSS-Instruct-75K` — a 20,000-example slice |
| **Decontamination** | Used the HumanEval-decontaminated split — no benchmark answers leak into training, so the score is real |
| **Prompt format** | Qwen ChatML (system / user / assistant) |
| **Trainable params** | 18.5M / 1.56B (1.18%) |
| **LoRA rank / alpha / dropout** | 16 / 32 / 0.0 |
| **Target modules** | all 7 linear layers (q, k, v, o, gate, up, down) |
| **Epochs / steps** | 2 / 5,000 |
| **Effective batch size** | 8 (batch 2 × grad-accum 4) |
| **Learning rate / schedule** | 2e-4 / cosine, 3% warmup |
| **Max sequence length** | 2,048 tokens |
| **Optimizer / precision** | adamw_8bit / fp16 |
| **Hardware** | 1× NVIDIA Tesla T4 (16GB), Kaggle free tier |
| **Training time** | ~4 hours 55 minutes |
| **Cost** | ₹0 / $0 (free GPU) |
| **Final training loss** | ~0.53 (from ~1.06 at start) |

### Run it locally (Ollama)
Download `merged.Q4_K_M.gguf` (~1 GB; the 4-bit build recommended for a 16GB laptop — there's also a slightly higher-quality `merged.Q5_K_M.gguf`, ~1.2 GB). Create a `Modelfile`:

```
FROM ./merged.Q4_K_M.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
SYSTEM "You are a helpful coding assistant. Respond with clean, correct, well-structured code and brief explanations."
```

Then:
```
ollama create qdex-1.5b -f Modelfile
ollama run qdex-1.5b "Write a Python function that checks if a string is a palindrome."
```
*(Builder note: this usage block reflects the Qwen ChatML format the model was trained on; treat it as the intended example.)*

### Engineering challenges & how I solved them
Training ran on a free, time-limited cloud GPU — and the **first two runs were lost**. Each time, training actually *finished*, but the session ended (a power cut at home) before the model was saved, and the platform's default "No Persistence" setting wiped the working directory. I rebuilt the pipeline to be fault-tolerant:
- **Checkpoints every 200 steps**, so a dropped session loses minutes, not hours.
- **Persistence set to "Files only"**, so saved files survive a session ending.
- **One-command resume** from the latest checkpoint.
- **Immediate off-platform backup** to Hugging Face the moment training finished.

The third run completed and every artifact survived. The lesson — *assume the session can die at any moment* — is what separates a notebook that works once from a pipeline you can trust.

### Limitations (stated honestly)
Qdex-1.5B is a **small** model. It's genuinely useful on focused, well-specified coding tasks, but it is **not** a frontier model and won't replace GPT/Claude on complex, multi-step problems. The reported score is from a single benchmark (HumanEval), greedy decoding, one sample per problem — an honest, reproducible measurement, not a leaderboard-tuned number.

### What's next (Phase 2)
- **v2 — a data experiment:** retrain with an added, decontaminated slice of **NVIDIA's OpenCodeInstruct** (whose solutions ship with test cases and quality scores), then re-run the same HumanEval benchmark to measure whether more/better data moves the score. A controlled, data-driven comparison — not guesswork.
- **Project 3 — the agent:** Qdex-1.5B is the engine for a **local CLI coding agent** that runs entirely on a 16GB no-GPU laptop via Ollama. This model was built specifically to make that agent possible.

### Links & license
- **Download / model:** https://huggingface.co/SumanDebnath943/Qdex-1.5B-GGUF
- **Code:** https://github.com/Sumandebnath943/Qdex-1.5B
- **Base model license:** Apache-2.0 (permits fine-tuning + portfolio/commercial use). Fine-tuned weights released under Apache-2.0.
- **Author:** Suman Debnath

---

## PART C — Hugging Face model card (give this to the HF repo as its README.md)

> Paste this into the `Qdex-1.5B-GGUF` repo on huggingface.co (the repo's README / model card). It can be edited directly in the HF web UI.

```markdown
---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-1.5B
language:
  - en
tags:
  - code
  - coding-assistant
  - qlora
  - unsloth
  - gguf
pipeline_tag: text-generation
---

# Qdex-1.5B (GGUF)

Instruction-tuned coding model built on **Qwen2.5-Coder-1.5B** using **QLoRA**, exported to **GGUF** to run locally on CPU (16GB RAM laptops) via Ollama / llama.cpp.

## Result (HumanEval, pass@1, greedy, n=1)
| Model | Prompting | pass@1 |
|---|---|---|
| Qwen2.5-Coder-1.5B (base) | completion | 40.2% (66/164) |
| Qwen2.5-Coder-1.5B (base) | instruction | 1.2% (2/164) |
| **Qdex-1.5B** | instruction | **42.1% (69/164)** |

Fine-tuning lifted instruction-following coding from **1.2% → 42.1%**, matching and slightly exceeding the base model's raw coding ability (40.2%) in a usable, instruction-following form.

## Files
- `merged.Q4_K_M.gguf` (~1 GB) — recommended for 16GB laptops.
- `merged.Q5_K_M.gguf` (~1.2 GB) — slightly higher quality.

## Training
- Dataset: `ise-uiuc/Magicoder-OSS-Instruct-75K` (20k slice, HumanEval-decontaminated), ChatML format.
- QLoRA: rank 16, alpha 32, dropout 0, all 7 linear modules; 2 epochs / 5,000 steps; effective batch 8; lr 2e-4 cosine; max_seq 2,048.
- Hardware: 1× NVIDIA T4 (Kaggle free). Time: ~4h55m. Final loss ~0.53.

## Use (Ollama)
See the Modelfile example in the repo / portfolio page.

## Limitations
A 1.5B model: strong on focused coding tasks, not a frontier model. Single-benchmark, greedy, n=1 measurement.

Author: Suman Debnath · Code: https://github.com/Sumandebnath943/Qdex-1.5B
```

---

## PART D — Action items for me (Suman, the human) — do these so the links work
1. **Make the Hugging Face GGUF repo public:** `Qdex-1.5B-GGUF` → Settings → change visibility from Private to **Public**. (Required, or the download link 404s for visitors.)
2. **Confirm the GitHub repo is public** (`Sumandebnath943/Qdex-1.5B`).
3. **Add the model card** (PART C) to the HF repo's README.
4. *(Optional, later)* decide whether to also publish the LoRA adapter repo (`Qdex-1.5B-lora-adapter`).

## Canonical facts (single source of truth — don't invent numbers)
- Headline: instruction-mode HumanEval pass@1 **1.2% → 42.1%** (69/164); base completion **40.2%**. Scored with the hardened eval harness (GitHub commit 0b7a660).
- Base: Qwen2.5-Coder-1.5B (Apache-2.0). Dataset: Magicoder-OSS-Instruct-75K, 20k decontaminated slice.
- Training: 2 epochs / 5,000 steps, ~4h55m, 1× T4, ₹0, final loss ~0.53, 1.18% params trained (QLoRA r16/a32/dropout0).
- Links: GGUF → huggingface.co/SumanDebnath943/Qdex-1.5B-GGUF · Code → github.com/Sumandebnath943/Qdex-1.5B
