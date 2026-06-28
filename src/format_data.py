"""Load Magicoder-OSS-Instruct and format examples into the Qwen2.5 chat template.

This is the Stage 1 deliverable: a function that turns one raw dataset example
into the exact text string the base model must see during instruction
fine-tuning.

A raw Magicoder-OSS-Instruct example has these fields (verified live):
    {
      "lang": "python",            # source language of the seed snippet
      "problem":  "Write a ...",   # the instruction/question (GPT-4 generated)
      "solution": "def foo(): ...",# the answer (GPT-4 generated)
      "seed": "<code snippet>",    # the OSS code snippet that inspired it
      "index": ..., "raw_index": ..., "openai_fingerprint": "..."
    }

We use only `problem` (as the user turn) and `solution` (as the assistant turn),
wrap them as a 3-turn conversation (system / user / assistant), and render
through the Qwen ChatML template so the model learns the assistant format.
"""
from typing import Any, Dict

from datasets import load_dataset
from transformers import AutoTokenizer

from config import BASE_MODEL, DATASET_NAME, SYSTEM_PROMPT
from src.chat_template import QWEN_CHATML_TEMPLATE


def get_tokenizer():
    """Load the base model's tokenizer and attach the ChatML template if missing.

    Downloads only the tokenizer files (~2MB), NOT the model weights.
    The base model ships without a chat_template, so we set the standard
    Qwen ChatML one (see src/chat_template.py for why).
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = QWEN_CHATML_TEMPLATE
    # Some base tokenizers lack a pad token; set one so training doesn't choke.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_example(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Convert one Magicoder example into a chat-templated text string.

    Returns a dict with:
      - "text":      the full formatted ChatML string (used as the training target)
      - "token_len": length in tokens (for sanity-checking against MAX_SEQ_LEN)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]},
        {"role": "assistant", "content": example["solution"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    token_len = len(tokenizer.encode(text, add_special_tokens=False))
    return {"text": text, "token_len": token_len}


def load_raw_dataset(split: str = "train"):
    """Load the full Magicoder dataset (downloads ~80MB on first call, cached after)."""
    return load_dataset(DATASET_NAME, split=split)


def load_and_format(num_examples: int = None, tokenizer=None):
    """Load the dataset and return a formatted copy with 'text' and 'token_len' columns.

    Args:
        num_examples: if given, take only the first N examples (for quick tests).
        tokenizer:    if None, loads the base model tokenizer via get_tokenizer().
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    ds = load_raw_dataset()
    if num_examples is not None:
        ds = ds.select(range(min(num_examples, len(ds))))
    formatted = ds.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=ds.column_names,
        desc="Formatting examples",
    )
    return formatted


if __name__ == "__main__":
    # Quick smoke test: load 3 examples, print lengths.
    tok = get_tokenizer()
    ds = load_and_format(num_examples=3, tokenizer=tok)
    for i, row in enumerate(ds):
        print(f"--- Example {i} | {row['token_len']} tokens ---")
        print(row["text"][:300] + "...\n")
