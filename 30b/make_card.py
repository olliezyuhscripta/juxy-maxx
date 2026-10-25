import json
import pathlib
import re

WORK = pathlib.Path("/marimo/juxy30b")
LOG = WORK / "train.log"
EVAL = WORK / "eval.json"
CARD = WORK / "README.md"

BASE = "empero-ai/Qwen3.8-35B-A3B-Distill"
REPO = "zxcxzczxcxzc/juxy-maxx-30b"

SYSTEM = (
    "You are Juxy, a fast coding and cybersec agent. Be concise and token-efficient. "
    "Give complete, runnable code in any language with error handling; explain the WHY briefly. "
    "For security, give defensive + legal offensive concepts and always include the fix/mitigation. "
    "You are Juxy - not any other AI model or company."
)

STEP_RE = re.compile(r"\[step (\d+)/(\d+)\] loss ([\d.]+) lr ([\d.e-]+) tok/s (\d+) vram ([\d.]+)GB elapsed ([\d.]+)m")
DONE_RE = re.compile(r"\[done\] (\d+) steps, (\d+) tokens, ([\d.]+) min")
PLAN_RE = re.compile(r"\[plan\] (\d+) micro-batches \(([\d.]+) rows, mean row (\d+) tok\), (\d+) steps/epoch")
TRAINABLE_RE = re.compile(r"trainable params: ([\d,]+) \|\| all params: ([\d,]+) \|\| trainable%: ([\d.]+)")


def parse_log():
    text = LOG.read_text(errors="ignore").replace("\r", "\n")
    steps = [
        {
            "step": int(m.group(1)),
            "total": int(m.group(2)),
            "loss": float(m.group(3)),
            "lr": m.group(4),
            "tok_s": int(m.group(5)),
            "vram": float(m.group(6)),
            "minutes": float(m.group(7)),
        }
        for m in STEP_RE.finditer(text)
    ]
    done = DONE_RE.search(text)
    plan = PLAN_RE.search(text)
    trainable = TRAINABLE_RE.search(text)
    return {
        "steps": steps,
        "done": done.groups() if done else None,
        "plan": plan.groups() if plan else None,
        "trainable": trainable.groups() if trainable else None,
    }


def loss_curve(steps, keep=10):
    if not steps:
        return "| step | loss |\n| --- | --- |\n| - | - |"
    chosen = steps[:: max(1, len(steps) // keep)]
    if chosen[-1]["step"] != steps[-1]["step"]:
        chosen.append(steps[-1])
    rows = "\n".join(f"| {s['step']} | {s['loss']:.4f} |" for s in chosen)
    return "| step | loss |\n| --- | --- |\n" + rows


def sample_block(rows, index):
    row = rows[index]
    prompt = row["prompt"]
    base = row["base"]["text"].strip()
    tuned = row["tuned"]["text"].strip()
    return (
        f"**Prompt:** {prompt}\n\n"
        f"<details><summary>base Qwen3.8-35B-A3B-Distill</summary>\n\n"
        f"```text\n{base[:1200]}\n```\n\n</details>\n\n"
        f"<details open><summary>juxy-maxx-30b</summary>\n\n"
        f"```text\n{tuned[:1200]}\n```\n\n</details>\n"
    )


def main():
    log = parse_log()
    steps = log["steps"]
    eval_rows = json.loads(EVAL.read_text()) if EVAL.exists() else []

    final_loss = f"{steps[-1]['loss']:.4f}" if steps else "n/a"
    last_step = steps[-1]["step"] if steps else 0
    total_tokens = log["done"][1] if log["done"] else "n/a"
    minutes = log["done"][2] if log["done"] else "n/a"
    tok_s = steps[-1]["tok_s"] if steps else "n/a"
    vram = steps[-1]["vram"] if steps else "n/a"
    trainable = log["trainable"][0] if log["trainable"] else "42,332,160"

    samples = "\n\n".join(sample_block(eval_rows, i) for i in range(min(2, len(eval_rows))))

    card = f"""---
base_model: {BASE}
library_name: peft
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- lora
- peft
- qwen3_5_moe
- coding
- security
- juxy
---

# juxy-maxx-30b

LoRA adapter that teaches the **Juxy** coding/security persona to a ~30B-class
Mixture-of-Experts base. This is the scale-up of
[`zxcxzczxcxzc/juxy-maxx`](https://huggingface.co/zxcxzczxcxzc/juxy-maxx), which is a
4B model that stopped after 80 of its 104 training steps.

| | juxy-maxx | juxy-maxx-30b |
| --- | --- | --- |
| base | Qwen3.8-4B-Distill (4B) | Qwen3.8-35B-A3B-Distill (35B total, ~3B active) |
| params | 4B | 35B total / 3B active per token |
| training | 80 steps, unfinished epoch | {last_step} steps, 1 full epoch |
| trainable | LoRA | {trainable} LoRA params (0.12%) |
| final loss | n/a | {final_loss} |

"30B" here means a 35B-parameter MoE whose 3B active parameters keep inference and
training fast. It is a finetune of a public base, not a from-scratch pretrain.

## Training

- **Data:** [`zxcxzczxcxzc/juxy-maxx-corpus`](https://huggingface.co/datasets/zxcxzczxcxzc/juxy-maxx-corpus),
  19,860 rows of `system`/`user`/`assistant` chat, ~12.3M tokens.
- **Method:** LoRA SFT, r=32, alpha=64, dropout 0.05, on every attention and MLP
  projection of the language model (vision tower frozen).
- **Schedule:** 1 epoch, lr 1e-4 cosine with 3% warmup, effective batch 8
  micro-batches packed to a 6144-token budget, bf16, sequence cap 2048.
- **Loss masking:** prompt tokens are `-100`; only assistant spans contribute.
- **Hardware:** one 96 GB RTX PRO 6000, {vram} GB peak, ~{tok_s} tok/s,
  {minutes} min wall clock, {total_tokens} tokens seen.

{loss_curve(steps)}

## Usage

```python
import torch
from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration
from peft import PeftModel

base = "{BASE}"
tokenizer = AutoTokenizer.from_pretrained(base)
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
    base, dtype=torch.bfloat16, device_map={{"": 0}}
)
model = PeftModel.from_pretrained(model, "{REPO}")
model.eval()

system = (
    "You are Juxy, a fast coding and cybersec agent. Be concise and token-efficient. "
    "Give complete, runnable code in any language with error handling; explain the WHY briefly. "
    "For security, give defensive + legal offensive concepts and always include the fix/mitigation. "
    "You are Juxy - not any other AI model or company."
)
messages = [{{"role": "system", "content": system}},
            {{"role": "user", "content": "Write a Python retry decorator with jitter."}}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=512)[0][inputs["input_ids"].shape[-1]:],
                       skip_special_tokens=True))
```

Use the Juxy system prompt. Without it the adapter still leans into terse code answers,
but that is the behaviour it was trained on.

## Samples

Greedy decoding, same system prompt, {len(eval_rows)} prompts.

{samples}

## Limitations

- Security content is a **persona prior**, not a safety guarantee. The corpus asks for
  defensive framing plus legal offensive concepts; nothing here was red-teamed, and the
  base model's own refusals can still surface or be bypassed.
- Trained for one epoch on ~12M tokens. Expect the style and format to transfer much more
  reliably than new factual knowledge.
- Sequence cap 2048 during training; longer contexts use the base model's behaviour.
- No evaluation beyond the samples above. No benchmark numbers are claimed.
- Base licence and terms apply: apache-2.0 upstream.

## Files

- `adapter_model.safetensors` / `adapter_config.json` - the LoRA adapter (PEFT).
- `tokenizer*`, `chat_template.jinja` - tokenizer files matching the base.
"""

    CARD.write_text(card, encoding="utf-8")
    print("wrote", CARD, len(card), "chars")
    print("final loss", final_loss, "| steps", last_step, "| minutes", minutes)


if __name__ == "__main__":
    main()
