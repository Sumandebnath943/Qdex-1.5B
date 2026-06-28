"""Stage 2 / Stage 5: Run the HumanEval benchmark on a model.

Usage:
    # Base model (completion mode, greedy):
    python scripts/run_benchmark.py \
        --model-path Qwen/Qwen2.5-Coder-1.5B \
        --mode completion \
        --n-samples 1 \
        --temperature 0.0 \
        --output outputs/bench_base.json

    # Fine-tuned model (instruction mode, greedy):
    python scripts/run_benchmark.py \
        --model-path outputs/merged \
        --mode instruction \
        --n-samples 1 \
        --temperature 0.0 \
        --output outputs/bench_finetuned.json

This script requires a GPU (or a patient CPU — see notes). On a Kaggle T4,
greedy pass@1 (n=1) over 164 problems takes ~10 minutes. On CPU it takes hours.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from tqdm import tqdm  # noqa: E402

from config import BASE_MODEL, SYSTEM_PROMPT, OUTPUTS_DIR  # noqa: E402
from src.benchmark import (  # noqa: E402
    HUMANEVAL_DATASET, aggregate_results, build_instruction_messages,
    extract_code_completion, extract_code_instruction, pass_at_k, run_solution,
)
from src.chat_template import QWEN_CHATML_TEMPLATE  # noqa: E402


def load_model_and_tokenizer(model_path: str):
    """Load a model for inference in fp16 (fits in ~3GB VRAM for 1.5B)."""
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Ensure a chat template exists (base models may lack one).
    if tokenizer.chat_template is None:
        tokenizer.chat_template = QWEN_CHATML_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded on device: {model.device}")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int,
             temperature: float, n_samples: int) -> list[str]:
    """Generate n_samples completions for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    completions = []
    for _ in range(n_samples):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                top_p=0.95 if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decode only the newly generated tokens (skip the prompt).
        new_tokens = output[0][inputs.input_ids.shape[1]:]
        completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return completions


def benchmark_model(model_path: str, mode: str, n_samples: int,
                    temperature: float, max_new_tokens: int,
                    max_problems: int | None = None) -> "BenchmarkResult":
    """Run the full HumanEval benchmark on one model."""
    from src.benchmark import BenchmarkResult  # for type hint

    model, tokenizer = load_model_and_tokenizer(model_path)
    problems = load_dataset(HUMANEVAL_DATASET, split="test")
    if max_problems:
        problems = problems.select(range(min(max_problems, len(problems))))
    print(f"Benchmarking {len(problems)} problems | mode={mode} "
          f"| n_samples={n_samples} | temp={temperature}")

    per_problem = []
    for problem in tqdm(problems, desc="Problems"):
        # Build the input prompt.
        if mode == "completion":
            prompt = problem["prompt"]
        else:  # instruction
            messages = build_instruction_messages(problem, SYSTEM_PROMPT)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Generate samples.
        try:
            completions = generate(
                model, tokenizer, prompt, max_new_tokens, temperature, n_samples
            )
        except Exception as e:
            per_problem.append({
                "task_id": problem["task_id"],
                "passed": False, "n_passed": 0, "n_total": n_samples,
                "pass_at_1": 0.0, "error": f"generation failed: {e}",
            })
            continue

        # Extract + test each sample.
        n_passed = 0
        errors = []
        for comp in completions:
            if mode == "completion":
                code = extract_code_completion(problem["prompt"], comp)
            else:
                code = extract_code_instruction(comp)
            passed, msg = run_solution(code, problem["test"], problem["entry_point"])
            if passed:
                n_passed += 1
            else:
                errors.append(msg)

        p1 = pass_at_k(n_passed, n_samples, k=1)
        per_problem.append({
            "task_id": problem["task_id"],
            "passed": p1 >= 1.0,
            "n_passed": n_passed,
            "n_total": n_samples,
            "pass_at_1": p1,
            "errors": errors[:3],  # keep small
        })

    return aggregate_results(model_path, mode, n_samples, temperature, per_problem)


def main():
    ap = argparse.ArgumentParser(description="Run HumanEval pass@1 benchmark.")
    ap.add_argument("--model-path", required=True,
                    help="HuggingFace model id or local path (e.g. outputs/merged).")
    ap.add_argument("--mode", choices=["completion", "instruction"], required=True,
                    help="completion = raw prompt (base model); "
                         "instruction = chat-wrapped (fine-tuned model).")
    ap.add_argument("--n-samples", type=int, default=1,
                    help="Samples per problem. 1 = greedy pass@1 (fast, deterministic).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy. >0 = sampling (use with n-samples > 1).")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="Max generation length per sample. 512 avoids truncating "
                         "longer valid solutions; applied equally to base + "
                         "fine-tuned so the before/after stays comparable.")
    ap.add_argument("--max-problems", type=int, default=None,
                    help="Limit to N problems (for quick tests). Default: all 164.")
    ap.add_argument("--output", required=True,
                    help="Path to write JSON results (e.g. outputs/bench_base.json).")
    args = ap.parse_args()

    result = benchmark_model(
        model_path=args.model_path,
        mode=args.mode,
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_problems=args.max_problems,
    )

    print("\n" + "=" * 60)
    print("BENCHMARK RESULT")
    print("=" * 60)
    print(result.summary())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model_path": result.model_path,
        "mode": result.mode,
        "n_samples": result.n_samples,
        "temperature": result.temperature,
        "total_problems": result.total_problems,
        "passed_problems": result.passed_problems,
        "pass_at_1": result.pass_at_1,
        "per_problem": result.per_problem,
    }, indent=2))
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
