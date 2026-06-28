# Stage 6 — Test the GGUF model locally with Ollama

After downloading the `.gguf` file from Kaggle, test that it loads and generates
code on your 16GB no-GPU laptop. This verifies the export worked and that
Project 3's CLI agent will be able to use the model.

## Step 1 — Install Ollama (one-time, ~2 min)

1. Go to https://ollama.com/download
2. Download the version for your OS (macOS / Windows / Linux).
3. Run the installer. It installs a background service.
4. Open a terminal and verify: `ollama --version` should print a version number.

## Step 2 — Place the GGUF file

1. Move the downloaded `.gguf` file into the project:
   ```
   qlora-coder-finetune/outputs/gguf/<filename>.gguf
   ```
   (e.g. `qwen25-coder-1.5b-finetuned-q4_k_m.gguf`)

## Step 3 — Create an Ollama model from the GGUF

Ollama needs a "Modelfile" (like a Dockerfile for models) to register the GGUF.

1. Create a file at `qlora-coder-finetune/outputs/gguf/Modelfile` with this
   content (replace `<filename>` with your actual .gguf filename):
   ```
   FROM ./<filename>.gguf

   TEMPLATE """{{ if .System }}<|im_start|>system
   {{ .System }}<|im_end|>
   {{ end }}{{ if .Prompt }}<|im_start|>user
   {{ .Prompt }}<|im_end|>
   {{ end }}<|im_start|>assistant
   {{ .Response }}<|im_end|>
   """

   SYSTEM """You are a helpful coding assistant. Respond with clean, correct, well-structured code and brief explanations."""

   PARAMETER stop "<|im_start|>"
   PARAMETER stop "<|im_end|>"
   PARAMETER temperature 0.2
   PARAMETER num_ctx 2048
   ```

2. Register it with Ollama:
   ```bash
   cd qlora-coder-finetune/outputs/gguf
   ollama create qwen-coder-ft -f Modelfile
   ```
   This takes ~30 seconds (it copies the GGUF into Ollama's storage).

## Step 4 — Test it

Run an interactive chat:
```bash
ollama run qwen-coder-ft
```
Then type a coding question:
```
>>> Write a Python function to check if a string is a palindrome.
```
You should get a code response in a few seconds. Try 2-3 more prompts to sanity
check quality. Type `/bye` to exit.

## Step 5 — (Optional) Test via API

Ollama exposes a local API that Project 3's agent will use:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen-coder-ft",
  "prompt": "Write a Python function to reverse a linked list.",
  "stream": false
}'
```

## Step 6 — Record the result

Note in `PROJECT_REPORT.md`:
- Which GGUF you tested (q4_k_m vs q5_k_m)
- Generation speed (tokens/sec — Ollama prints this)
- Whether outputs look reasonable for scoped coding tasks
- File size on disk

## Honest expectations

A 1.5B model quantized to 4-bit is good at:
- Writing small, well-specified functions
- Explaining short code snippets
- Simple bug fixes

It will struggle with:
- Multi-file reasoning
- Complex algorithms
- Long contexts (>2048 tokens)
- Following many-step instructions

This is an honest limitation of a small model, not a bug. Project 3's agent
should scope tasks accordingly.
