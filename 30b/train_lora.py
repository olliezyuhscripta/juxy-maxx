import argparse
import json
import math
import os
import pathlib
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration

WORK = pathlib.Path("/marimo/juxy30b")
DATA = WORK / "normalized.jsonl"
ADAPTER_OUT = WORK / "adapter"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="empero-ai/Qwen3.8-35B-A3B-Distill")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def encode_row(tokenizer, messages, max_len):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    labels = list(ids)
    # Loss only on the assistant span: everything the prompt covers is masked.
    for index in range(min(len(prompt_ids), len(labels))):
        labels[index] = -100
    return ids, labels


def collate(batch, pad_id):
    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, attention = [], [], []
    for ids, row_labels in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(row_labels + [-100] * pad)
        attention.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attention, dtype=torch.long),
    )


def save_adapter(model, tokenizer, tag):
    target = ADAPTER_OUT if tag == "final" else WORK / f"adapter-{tag}"
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    print(f"[save] adapter -> {target}", flush=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()

    started = time.time()
    print(f"[load] tokenizer {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [json.loads(line)["messages"] for line in DATA.open(encoding="utf-8")]
    print(f"[data] {len(rows)} rows", flush=True)
    encoded = []
    skipped = 0
    for messages in rows:
        ids, labels = encode_row(tokenizer, messages, args.max_len)
        if sum(1 for label in labels if label != -100) < 8:
            skipped += 1
            continue
        encoded.append((ids, labels))
    print(f"[data] encoded {len(encoded)} rows, skipped {skipped}, {time.time() - started:.0f}s", flush=True)

    print(f"[load] model {args.model} (bf16, cuda:0)", flush=True)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        token=os.environ.get("HF_TOKEN"),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for parameter in model.model.visual.parameters():
        parameter.requires_grad = False

    linear_leaves = {
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and "language_model" in name
    }
    targets = sorted(linear_leaves - {"gate", "lm_head", "shared_expert_gate"})
    print(f"[lora] targeting {targets}", flush=True)

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    # Sort by length so every micro-batch pads to a similar width, then pack to a
    # token budget: this amortises the 256-expert routing cost instead of paying it
    # for two short rows at a time.
    lengths = [len(ids) for ids, _ in encoded]
    packed_order = sorted(range(len(encoded)), key=lambda index: lengths[index])
    batches, current, current_width = [], [], 0
    for index in packed_order:
        width = max(current_width, lengths[index])
        if current and (len(current) + 1) * width > args.token_budget:
            batches.append(current)
            current, current_width = [], 0
            width = lengths[index]
        current.append(index)
        current_width = width
    if current:
        batches.append(current)

    steps_per_epoch = math.ceil(len(batches) / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(10, int(total_steps * 0.03))
    mean_rows = round(len(encoded) / len(batches), 1)
    mean_width = round(sum(lengths) / len(encoded))
    print(
        f"[plan] {len(batches)} micro-batches ({mean_rows} rows, mean row {mean_width} tok), "
        f"{steps_per_epoch} steps/epoch, running {total_steps} steps, warmup {warmup}",
        flush=True,
    )

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    cursor = 0
    step = 0
    micro = 0
    tokens_seen = 0
    running_loss = 0.0
    running_count = 0
    epoch_tokens_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    while step < total_steps:
        if cursor >= len(batches):
            random.shuffle(batches)
            cursor = 0
        batch_rows = batches[cursor]
        cursor += 1

        input_ids, labels, attention = collate([encoded[i] for i in batch_rows], tokenizer.pad_token_id)
        input_ids = input_ids.to("cuda:0", non_blocking=True)
        labels = labels.to("cuda:0", non_blocking=True)
        attention = attention.to("cuda:0", non_blocking=True)

        output = model(input_ids=input_ids, attention_mask=attention, labels=labels)
        loss = output.loss
        (loss / args.grad_accum).backward()

        running_loss += loss.item()
        running_count += 1
        tokens_seen += int(attention.sum().item())
        micro += 1

        if micro % args.grad_accum == 0:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % 10 == 0:
                elapsed = time.time() - started
                rate = tokens_seen / max(1e-9, time.time() - epoch_tokens_start)
                vram = torch.cuda.max_memory_allocated() / 1e9
                print(
                    f"[step {step}/{total_steps}] loss {running_loss / running_count:.4f} "
                    f"lr {lr_at(step):.2e} tok/s {rate:.0f} vram {vram:.1f}GB elapsed {elapsed / 60:.1f}m",
                    flush=True,
                )
                running_loss = 0.0
                running_count = 0

            if args.save_every and step % args.save_every == 0:
                save_adapter(model, tokenizer, str(step))

    save_adapter(model, tokenizer, "final")
    print(f"[done] {step} steps, {tokens_seen} tokens, {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
