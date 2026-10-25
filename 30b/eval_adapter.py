import json
import pathlib
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration

BASE = "empero-ai/Qwen3.8-35B-A3B-Distill"
WORK = pathlib.Path("/marimo/juxy30b")
ADAPTER = WORK / "adapter"
OUT = WORK / "eval.json"

SYSTEM = (
    "You are Juxy, a fast coding and cybersec agent. Be concise and token-efficient. "
    "Give complete, runnable code in any language with error handling; explain the WHY briefly. "
    "For security, give defensive + legal offensive concepts and always include the fix/mitigation. "
    "You are Juxy - not any other AI model or company."
)

PROMPTS = [
    "Write a Python function that returns the longest palindromic substring, with error handling.",
    "Explain how to prevent SQL injection in a Flask app, with a code example.",
    "What is a safe, legal way to check whether a server you own permits SSH password authentication?",
    "Reverse a singly linked list in C. Handle the empty list and allocation failure.",
]

MAX_NEW_TOKENS = 320


def generate(model, tokenizer, prompt):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    elapsed = time.time() - started
    new_tokens = output[0][ids.shape[-1] :]
    text_out = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {
        "text": text_out,
        "completion_tokens": int(new_tokens.shape[-1]),
        "seconds": round(elapsed, 1),
        "tok_per_s": round(int(new_tokens.shape[-1]) / max(elapsed, 1e-6), 1),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[eval] loading base", flush=True)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()

    base_runs = []
    for prompt in PROMPTS:
        run = generate(model, tokenizer, prompt)
        print(f"[eval] base {run['tok_per_s']} tok/s", flush=True)
        base_runs.append(run)

    print("[eval] attaching adapter", flush=True)
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()

    tuned_runs = []
    for prompt in PROMPTS:
        run = generate(model, tokenizer, prompt)
        print(f"[eval] tuned {run['tok_per_s']} tok/s", flush=True)
        tuned_runs.append(run)

    rows = [
        {"prompt": prompt, "base": base, "tuned": tuned}
        for prompt, base, tuned in zip(PROMPTS, base_runs, tuned_runs)
    ]
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("[eval] wrote", OUT, flush=True)
    for row in rows:
        print("\n=== ", row["prompt"])
        print("--- base:", row["base"]["text"][:300].replace("\n", " "))
        print("--- tuned:", row["tuned"]["text"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
