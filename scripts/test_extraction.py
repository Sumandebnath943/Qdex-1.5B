"""Regression test for instruction-mode code extraction (CPU, no GPU).

Instruction-tuned models return code in several shapes. An earlier version of
the benchmark used a single strict regex and ran ONLY the extracted snippet,
which silently failed correct solutions that (a) dropped the prompt's imports,
(b) returned just the function body, or (c) had an unclosed code fence. That
undercounts the model's true pass@1.

This test feeds four answers that are ALL correct solutions (shaped differently)
through the hardened `candidate_programs` + `run_solution` path and asserts they
all pass. Run it any time the extraction logic changes:

    python scripts/test_extraction.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import candidate_programs, run_solution  # noqa: E402

# A HumanEval-style problem that NEEDS an import (so a dropped import is fatal).
PROMPT = ("from typing import List\n\n\n"
          "def add_one(numbers: List[int]) -> List[int]:\n"
          '    """Add one to each number."""\n')
TEST = ("def check(candidate):\n"
        "    assert candidate([1, 2]) == [2, 3]\n"
        "    assert candidate([]) == []\n")
ENTRY = "add_one"

# The OLD extractor, kept here verbatim only so the test documents what regressed.
def _old_extract(text: str) -> str:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()

# Four CORRECT solutions, in the shapes real instruction models produce.
CASES = {
    "full fn + imports, fenced":
        "```python\nfrom typing import List\n"
        "def add_one(numbers: List[int]) -> List[int]:\n"
        "    return [n + 1 for n in numbers]\n```",
    "full fn, missing import (prose + fence)":
        "Here is the solution:\n```python\n"
        "def add_one(numbers: List[int]) -> List[int]:\n"
        "    return [n + 1 for n in numbers]\n```",
    "body only (no def line)":
        "```python\n    return [n + 1 for n in numbers]\n```",
    "unclosed fence (truncated output)":
        "Sure!\n```python\nfrom typing import List\n"
        "def add_one(numbers: List[int]) -> List[int]:\n"
        "    return [n + 1 for n in numbers]",
}


def _old_pass(gen: str) -> bool:
    return run_solution(_old_extract(gen), TEST, ENTRY)[0]


def _new_pass(gen: str) -> bool:
    return any(run_solution(c, TEST, ENTRY)[0]
               for c in candidate_programs(gen, PROMPT, ENTRY))


def main() -> int:
    print(f"{'case':42} {'OLD':>6} {'NEW':>6}")
    print("-" * 58)
    new_fails = []
    for name, gen in CASES.items():
        o, n = _old_pass(gen), _new_pass(gen)
        print(f"{name:42} {('PASS' if o else 'fail'):>6} {('PASS' if n else 'fail'):>6}")
        if not n:
            new_fails.append(name)
    print("-" * 58)
    old_n = sum(_old_pass(g) for g in CASES.values())
    print(f"OLD parser: {old_n}/{len(CASES)} | NEW parser: "
          f"{len(CASES) - len(new_fails)}/{len(CASES)}")

    assert not new_fails, f"hardened extraction should pass all cases; failed: {new_fails}"
    print("OK: hardened extraction recovers every correctly-solved shape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
