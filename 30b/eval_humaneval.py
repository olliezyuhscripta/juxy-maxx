import gzip
import io
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time

import requests
import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration

BASE = "empero-ai/Qwen3.8-35B-A3B-Distill"
WORK = pathlib.Path("/marimo/juxy30b")
CANDIDATES = [("ckpt150", WORK / "adapter-150"), ("final", WORK / "adapter")]
PROBLEMS = 24
TIMEOUT = 20

SYSTEM = (
    "You are Juxy, a fast coding and cybersec agent. Be concise and token-efficient. "
    "Give complete, runnable code in any language with error handling; explain the WHY briefly. "
    "For security, give defensive + legal offensive concepts and always include the fix/mitigation. "
    "You are Juxy - not any other AI model or company."
)

SAMPLE_PROMPTS = [
    "Write a Python function that returns the longest palindromic substring, with error handling.",
    "Explain how to prevent SQL injection in a Flask app, with a code example.",
    "What is a safe, legal way to check whether a server you own permits SSH password authentication?",
    "Reverse a singly linked list in C. Handle the empty list and allocation failure.",
]

FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def load_problems():
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    raw = requests.get(url, timeout=120)
    raw.raise_for_status()
    rows = []
    with gzip.GzipFile(fileobj=io.BytesIO(raw.content)) as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows[:PROBLEMS]


def extract_code(text, prompt):
    match = FENCE.search(text)
    body = match.group(1) if match else text
    if not match and "def " in body:
        body = body[body.index("def ") :]
    return body.strip()


def run_program(program):
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "solution.py"
        path.write_text(program, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        detail = (result.stdout + result.stderr).strip().splitlines()
        return result.returncode == 0, (detail[-1][:200] if detail else "")


def generate(model, tokenizer, messages, max_new=512):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    elapsed = time.time() - started
    new = output[0][ids.shape[-1] :]
    return tokenizer.decode(new, skip_special_tokens=True), int(new.shape[-1]), elapsed


def eval_candidate(model, tokenizer, problems, label):
    passed = 0
    results = []
    for index, problem in enumerate(problems):
        user = (
            "Complete the following Python function. Return only the implementation "
            "inside one python code block, no explanation.\n\n```python\n"
            + problem["prompt"]
            + "\n```"
        )
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        text, tokens, elapsed = generate(model, tokenizer, messages)
        code = extract_code(text, problem["prompt"])
        program = problem["prompt"] + code + "\n\n" + problem["test"] + f"\n\ncheck({problem['entry_point']})\n"
        ok, detail = run_program(program)
        passed += int(ok)
        results.append(
            {"task": problem["task_id"], "passed": ok, "detail": detail, "code": code[:2000],
             "tokens": tokens, "seconds": round(elapsed, 1)}
        )
        print(f"[{label}] {index + 1}/{len(problems)} {problem['task_id']} {'PASS' if ok else 'fail'} "
              f"{tokens} tok {elapsed:.1f}s", flush=True)
    rate = passed / len(problems)
    print(f"[{label}] pass@1 {passed}/{len(problems)} = {rate:.3f}", flush=True)
    return {"label": label, "passed": passed, "total": len(problems), "pass_at_1": round(rate, 3), "results": results}


def sample(model, tokenizer, prompt, max_new=384):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text, tokens, elapsed = generate(model, tokenizer, messages, max_new=max_new)
    return {"text": text, "completion_tokens": tokens, "seconds": round(elapsed, 1),
            "tok_per_s": round(tokens / max(elapsed, 1e-6), 1)}


def main():
    problems = load_problems()
    print(f"[humaneval] {len(problems)} problems", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[load] base", flush=True)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    report = {"base": eval_candidate(model, tokenizer, problems, "base")}
    base_samples = [sample(model, tokenizer, prompt) for prompt in SAMPLE_PROMPTS]
    print("[samples] base done", flush=True)

    for name, path in CANDIDATES:
        print(f"[load] adapter {name}", flush=True)
        if "peft" not in type(model).__module__.lower():
            model = PeftModel.from_pretrained(model, str(path), adapter_name=name)
        else:
            model.load_adapter(str(path), adapter_name=name)
        model.set_adapter(name)
        model.eval()
        report[name] = eval_candidate(model, tokenizer, problems, name)

    best = max(["base"] + [name for name, _ in CANDIDATES], key=lambda key: report[key]["pass_at_1"])
    print("[humaneval] winner by pass@1:", best, flush=True)

    tuned_samples = base_samples
    if best != "base":
        model.set_adapter(best)
        tuned_samples = [sample(model, tokenizer, prompt) for prompt in SAMPLE_PROMPTS]
        print(f"[samples] {best} done", flush=True)

    rows = [
        {"prompt": prompt, "base": base_run, "tuned": tuned_run}
        for prompt, base_run, tuned_run in zip(SAMPLE_PROMPTS, base_samples, tuned_samples)
    ]
    (WORK / "eval.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    report["winner"] = best
    (WORK / "humaneval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[humaneval] wrote humaneval.json and eval.json", flush=True)
    for label in ["base"] + [name for name, _ in CANDIDATES]:
        entry = report[label]
        print(f"  {label}: pass@1 {entry['passed']}/{entry['total']} = {entry['pass_at_1']}")


if __name__ == "__main__":
    main()
