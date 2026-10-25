import json
import pathlib
import re
import sys

import requests

WORK = pathlib.Path("/marimo/juxy30b")
RAW = WORK / "corpus.jsonl"
OUT = WORK / "normalized.jsonl"
URL = "https://huggingface.co/datasets/zxcxzczxcxzc/juxy-maxx-corpus/resolve/main/corpus.jsonl"

SEGMENT = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.DOTALL)


def download():
    if RAW.exists() and RAW.stat().st_size > 40_000_000:
        return
    with requests.get(URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with RAW.open("wb") as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)


def parse(text):
    messages = [{"role": role, "content": content} for role, content in SEGMENT.findall(text)]
    roles = [m["role"] for m in messages]
    if roles[:2] != ["system", "user"] or not roles or roles[-1] != "assistant":
        return None
    return messages


def main():
    download()
    stats = {"rows": 0, "skipped": 0, "chars": 0, "assistant_chars": 0, "turns": {}}
    with RAW.open(encoding="utf-8") as source, OUT.open("w", encoding="utf-8") as sink:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError):
                stats["skipped"] += 1
                continue
            messages = parse(text)
            if messages is None:
                stats["skipped"] += 1
                continue
            stats["rows"] += 1
            stats["chars"] += len(text)
            stats["assistant_chars"] += sum(len(m["content"]) for m in messages if m["role"] == "assistant")
            stats["turns"][len(messages)] = stats["turns"].get(len(messages), 0) + 1
            sink.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print("rows:", stats["rows"], "skipped:", stats["skipped"])
    print("corpus chars:", stats["chars"], "assistant chars:", stats["assistant_chars"])
    print("messages per row:", stats["turns"])
    print("approx tokens (chars/4):", round(stats["chars"] / 4 / 1e6, 2), "M")
    print("normalized:", OUT, OUT.stat().st_size // 1024, "KiB")


if __name__ == "__main__":
    main()
