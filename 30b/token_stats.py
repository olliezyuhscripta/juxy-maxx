import json
import pathlib
import random
import statistics

from transformers import AutoTokenizer

MODEL = "empero-ai/Qwen3.8-35B-A3B-Distill"
DATA = pathlib.Path("/marimo/juxy30b/normalized.jsonl")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print("vocab:", len(tokenizer), "pad:", tokenizer.pad_token, "eos:", tokenizer.eos_token)
    print("template:", (tokenizer.chat_template or "")[:160].replace("\n", " "))

    rows = [json.loads(line) for line in DATA.open(encoding="utf-8")]
    random.seed(0)
    sample = random.sample(rows, 400)

    round_trip = 0
    lengths, prompt_lengths, assistant_lengths = [], [], []
    for row in sample:
        messages = row["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        if text.endswith("<|im_end|>\n") or text.endswith("<|im_end|>"):
            round_trip += 1
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        pids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
        prompt_lengths.append(len(pids))
        assistant_lengths.append(len(ids) - len(pids))

    def report(name, values):
        values = sorted(values)
        n = len(values)
        print(
            f"{name}: p50={values[n // 2]} p90={values[int(n * 0.9)]} "
            f"p99={values[int(n * 0.99) - 1]} max={values[-1]} mean={round(statistics.mean(values))}"
        )

    report("full tokens", lengths)
    report("prompt tokens", prompt_lengths)
    report("assistant tokens", assistant_lengths)
    print("rows ending with im_end:", round_trip, "/", len(sample))
    print("rows over 2048 tokens:", sum(1 for n in lengths if n > 2048), "/", len(sample))
    print("rows over 3072 tokens:", sum(1 for n in lengths if n > 3072), "/", len(sample))


if __name__ == "__main__":
    main()
