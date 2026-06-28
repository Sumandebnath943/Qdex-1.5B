"""Qwen2.5 ChatML template.

Why this file exists:
The Qwen2.5-Coder-1.5B *base* model (which we chose) was trained to predict the
next token, but it does NOT ship with a chat_template in its tokenizer config.
The *Instruct* version of the same model DOES ship one, and it uses the ChatML
format (standard across all Qwen models).

We take that exact ChatML template and apply it to format our training data.
This is the standard, correct way to instruction-tune a Qwen base model: you
teach it the assistant conversation format by fine-tuning on examples already
wrapped in that format.

The template below is identical to Qwen2.5-Coder-1.5B-Instruct's, so our
fine-tuned model will be compatible with any Qwen-compatible tooling later.
"""

# Jinja2 template string, applied by tokenizer.apply_chat_template().
# Renders a list of {role, content} messages into ChatML text:
#   <|im_start|>system\n{...}<|im_end|>\n
#   <|im_start|>user\n{...}<|im_end|>\n
#   <|im_start|>assistant\n{...}<|im_end|>\n
QWEN_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)
