"""CPU smoke test for the benchmark harness — NO model, NO GPU needed.

Validates that the code extraction + execution + pass@1 logic works correctly,
using a few hand-crafted examples with KNOWN answers. If this passes, the only
thing untested is the model generation (which needs a GPU).

Run from project root:
    python scripts/smoke_test_cpu.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import (  # noqa: E402
    extract_code_completion, extract_code_instruction, pass_at_k, run_solution,
)


# --- A real HumanEval-style problem (simplified, with known answer) ---
PROMPT = (
    'def add(a, b):\n'
    '    """Return the sum of a and b."""\n'
)
TEST = (
    'def check(candidate):\n'
    '    assert candidate(1, 2) == 3\n'
    '    assert candidate(0, 0) == 0\n'
    '    assert candidate(-1, 1) == 0\n'
)
ENTRY_POINT = "add"


def test_extraction_completion():
    """Completion-mode extraction should cut at top-level boundaries."""
    # Model generates the function body + trailing garbage.
    completion = "    return a + b\n\n\ndef unrelated():\n    pass\n"
    code = extract_code_completion(PROMPT, completion)
    assert "return a + b" in code
    assert "unrelated" not in code, "Should cut before the trailing def"
    print("[PASS] extract_code_completion cuts at top-level def")


def test_extraction_instruction():
    """Instruction-mode extraction should pull code from a fenced block."""
    response = "Here's the code:\n```python\n    return a + b\n```\nDone."
    code = extract_code_instruction(response)
    assert "return a + b" in code
    assert "Here's" not in code
    print("[PASS] extract_code_instruction pulls code from ```python fence")

    # Fallback when no fence.
    code2 = extract_code_instruction("    return a + b")
    assert "return a + b" in code2
    print("[PASS] extract_code_instruction falls back to raw text")


def test_execution_correct():
    """A correct solution should pass."""
    code = PROMPT + "    return a + b\n"
    passed, msg = run_solution(code, TEST, ENTRY_POINT)
    assert passed, f"Expected pass, got: {msg}"
    print("[PASS] correct solution passes the test")


def test_execution_wrong():
    """A wrong solution should fail."""
    code = PROMPT + "    return a * b\n"  # multiply instead of add
    passed, msg = run_solution(code, TEST, ENTRY_POINT)
    assert not passed, "Wrong solution should fail"
    print(f"[PASS] wrong solution fails: {msg[:60]}")


def test_execution_syntax_error():
    """A syntax-error solution should fail gracefully (no crash)."""
    code = PROMPT + "    return a + b\n"  # wait this is fine, let me make it broken
    code = PROMPT + "    return  +\n"  # syntax error
    passed, msg = run_solution(code, TEST, ENTRY_POINT)
    assert not passed, "Syntax error should fail"
    print(f"[PASS] syntax error handled gracefully: {msg[:60]}")


def test_pass_at_k():
    """pass@1 with n=1: 1 pass -> 1.0, 0 pass -> 0.0."""
    assert pass_at_k(1, 1, k=1) == 1.0, "1/1 passed -> pass@1 = 1.0"
    assert pass_at_k(0, 1, k=1) == 0.0, "0/1 passed -> pass@1 = 0.0"
    # pass@1 with n=20, 10 passed: should be 1 - C(10,1)/C(20,1) = 1 - 10/20 = 0.5
    assert abs(pass_at_k(10, 20, k=1) - 0.5) < 1e-9
    # All pass -> 1.0
    assert pass_at_k(20, 20, k=1) == 1.0
    print("[PASS] pass_at_k math is correct")


def test_full_flow_completion_mode():
    """End-to-end: a correct completion passes, a wrong one fails."""
    correct = extract_code_completion(PROMPT, "    return a + b\n")
    p1, _ = run_solution(correct, TEST, ENTRY_POINT)
    assert p1

    wrong = extract_code_completion(PROMPT, "    return a - b\n")
    p2, _ = run_solution(wrong, TEST, ENTRY_POINT)
    assert not p2
    print("[PASS] end-to-end completion flow works")


def main():
    print("=" * 60)
    print("BENCHMARK HARNESS SMOKE TEST (no model, no GPU)")
    print("=" * 60)
    test_extraction_completion()
    test_extraction_instruction()
    test_execution_correct()
    test_execution_wrong()
    test_execution_syntax_error()
    test_pass_at_k()
    test_full_flow_completion_mode()
    print("\nAll smoke tests passed. The harness logic is verified.")
    print("Only model generation remains untested (needs a GPU).")


if __name__ == "__main__":
    main()
