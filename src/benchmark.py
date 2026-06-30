"""HumanEval benchmark harness for measuring coding model quality.

What this measures:
    pass@1 — the fraction of HumanEval problems (164 total) where the model's
    generated code passes the official test cases.

Two generation modes are supported (each model is tested in the mode it was
designed for — see scripts/full_pipeline.md for the rationale):

    - "completion":  feed the raw HumanEval prompt (a function signature +
                     docstring) and let the model continue the code. Used for
                     the BASE model (it was pretrained for next-token prediction).
                     Matches how the Qwen2.5-Coder paper reports scores.

    - "instruction": wrap the prompt as a user message ("Complete this function:
                     ...") and take the assistant's code-block response. Used for
                     the FINE-TUNED model (it was instruction-tuned for this).
                     Matches how the model will actually be used in Project 3.

Safety:
    Generated code is executed in a subprocess with a hard timeout. This is the
    standard approach used by most HumanEval runners. For production-grade
    isolation you would use Docker; for a portfolio project subprocess+timeout
    is sufficient and honest about its limitations.

This module contains NO model-loading code (that lives in run_benchmark.py) so
the extraction/execution/scoring logic can be unit-tested on CPU without a GPU.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from math import comb
from typing import List, Optional

# HumanEval dataset: 164 hand-written Python programming problems, each with a
# function signature + docstring prompt, a canonical solution, and a test
# function `check(candidate)` that asserts correct behavior on several inputs.
HUMANEVAL_DATASET = "openai/openai_humaneval"

# Subprocess execution timeout (seconds). 10s is generous; canonical solutions
# run in <1s. Infinite loops / hangs get killed at 10s and counted as failures.
EXEC_TIMEOUT = 10


@dataclass
class BenchmarkResult:
    """Result of benchmarking one model on HumanEval."""
    model_path: str
    mode: str                          # "completion" or "instruction"
    n_samples: int                     # samples per problem
    temperature: float
    total_problems: int
    passed_problems: int
    pass_at_1: float                   # 0.0 - 1.0
    per_problem: List[dict] = field(default_factory=list)
    # Each per_problem dict: {task_id, passed (bool), n_passed, n_total, error?}

    def summary(self) -> str:
        pct = self.pass_at_1 * 100
        return (
            f"Model: {self.model_path}\n"
            f"Mode:  {self.mode} | samples/problem: {self.n_samples} | temp: {self.temperature}\n"
            f"pass@1: {self.passed_problems}/{self.total_problems} = {pct:.1f}%"
        )


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code_completion(prompt: str, completion: str) -> str:
    """Extract the completed function from a completion-mode generation.

    The prompt ends mid-function (e.g. after the docstring). The model continues
    with the function body. We concatenate prompt + completion, then cut at the
    first top-level boundary (a new def/class/main-guard) so trailing garbage
    doesn't break execution.
    """
    full = prompt + completion
    # Only look for stop patterns in the completion portion (after the prompt).
    search_start = len(prompt)
    for stop in ["\nclass ", "\ndef ", "\nif __name__", "\nprint(", "\n\n\n"]:
        idx = full.find(stop, search_start)
        if idx != -1:
            full = full[:idx]
    return full


def extract_code_instruction(text: str) -> str:
    """Extract Python code from an instruction-mode assistant response.

    Robust to the shapes instruction-tuned models actually produce:
      - a ```python / ```py / bare ``` fenced block (preferred),
      - an UNCLOSED fence (generation truncated before the closing ```),
      - no fence at all (raw code, possibly with prose around it).
    """
    # 1. A properly closed fenced block (prefer a python-tagged one).
    for pat in (r"```(?:python|py)[ \t]*\n(.*?)```", r"```[ \t]*\n(.*?)```"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    # 2. Unclosed fence: take everything after the opening fence.
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 3. No fence: use the raw text as-is.
    return text.strip()


def candidate_programs(generation: str, prompt: str, entry_point: str) -> List[str]:
    """Runnable program variants to try (in order) for instruction mode.

    Instruction-tuned models answer in different shapes: a full self-contained
    function, just the body, or code that drops the prompt's imports/signature.
    We extract the code, then offer up to two candidates:
      1. the extracted code as-is (self-contained answers pass here);
      2. the original prompt prepended to the extracted code, which recovers
         answers that gave only the body or omitted the prompt's imports.

    Honesty note: prepending the prompt can only ever turn a FAIL into a PASS
    when the model's code is *actually correct* — the official `check()` tests
    behavior, so wrong logic still fails. This fixes lossy extraction; it does
    not inflate the score. The same two-candidate logic, if anything, makes the
    instruction score MORE comparable to completion mode (which always keeps the
    prompt).
    """
    code = extract_code_instruction(generation)
    variants = [code]
    prepended = prompt.rstrip() + "\n" + code
    if prepended != code:
        variants.append(prepended)
    # Body-only answers (no `def` at all): re-indent the lines and graft them
    # onto the prompt's signature so the function is actually defined.
    if "def " not in code:
        body = "\n".join(
            ("    " + ln if ln.strip() else ln) for ln in code.splitlines())
        variants.append(prompt.rstrip() + "\n" + body)
    return variants


# ---------------------------------------------------------------------------
# Code execution (sandboxed subprocess)
# ---------------------------------------------------------------------------

def run_solution(code: str, test: str, entry_point: str,
                 timeout: int = EXEC_TIMEOUT) -> tuple[bool, str]:
    """Execute generated code + official test in a subprocess.

    Args:
        code:        the extracted Python code (prompt + completion, or
                     instruction-mode code block). Must define `entry_point`.
        test:        the HumanEval `test` field (defines `check(candidate)`).
        entry_point: the function name to test.
        timeout:     seconds before killing the process.

    Returns:
        (passed, message) — passed is True iff returncode == 0.
    """
    # HumanEval tests define `def check(candidate): ...` with assertions.
    # We call check(entry_point) to run them.
    full_script = f"{code}\n\n{test}\n\ncheck({entry_point})\n"
    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="humaneval_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(full_script)
        result = subprocess.run(
            [sys.executable, tmp_path],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, "OK"
        return False, (result.stderr or result.stdout or "non-zero exit")[:300]
    except subprocess.TimeoutExpired:
        return False, f"timeout ({timeout}s)"
    except Exception as e:
        return False, f"exec error: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# pass@k computation (standard HumanEval formula)
# ---------------------------------------------------------------------------

def pass_at_k(n_passed: int, n_total: int, k: int = 1) -> float:
    """Standard pass@k for a single problem.

    pass@k = 1 - C(n - c, k) / C(n, k), where n = samples generated, c = passed.
    For pass@1 with n=1: returns 1.0 if the single sample passed, else 0.0.
    """
    if n_total - n_passed < k:
        return 1.0
    return 1.0 - comb(n_total - n_passed, k) / comb(n_total, k)


# ---------------------------------------------------------------------------
# Prompt construction (used by run_benchmark.py)
# ---------------------------------------------------------------------------

def build_prompt(problem: dict, mode: str, system_prompt: str) -> str:
    """Build the input string to feed the model for one problem.

    - completion mode: return the raw HumanEval prompt (function sig + docstring).
    - instruction mode: wrap the prompt as a user message through the chat template.
    """
    if mode == "completion":
        return problem["prompt"]
    elif mode == "instruction":
        # The tokenizer's apply_chat_template is called by the caller; here we
        # just return the message list as a JSON-serializable structure. The
        # caller (run_benchmark.py) applies the chat template.
        raise NotImplementedError(
            "instruction-mode prompt building is handled in run_benchmark.py "
            "where the tokenizer is available for apply_chat_template."
        )
    raise ValueError(f"Unknown mode: {mode!r}. Use 'completion' or 'instruction'.")


def build_instruction_messages(problem: dict, system_prompt: str) -> list:
    """Build the chat messages for instruction-mode generation."""
    user_content = (
        f"Complete the following Python function. "
        f"Respond with ONLY the code, no explanation.\n\n"
        f"{problem['prompt']}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(model_path: str, mode: str, n_samples: int,
                      temperature: float,
                      per_problem: List[dict]) -> BenchmarkResult:
    """Compute overall pass@1 from per-problem results."""
    total = len(per_problem)
    passed = sum(1 for p in per_problem if p["pass_at_1"] >= 1.0)
    overall = passed / total if total > 0 else 0.0
    return BenchmarkResult(
        model_path=model_path,
        mode=mode,
        n_samples=n_samples,
        temperature=temperature,
        total_problems=total,
        passed_problems=passed,
        pass_at_1=overall,
        per_problem=per_problem,
    )
