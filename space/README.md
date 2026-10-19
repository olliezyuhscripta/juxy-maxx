---
title: Juxy v2
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
short_description: Coding + cybersec assistant on a Qwen3.5/3.8 hybrid + LoRA
---

# Juxy v2

A coding + cybersec assistant: **`empero-ai/Qwen3.8-4B-Distill`** (Qwen3.5/3.8 hybrid
architecture, 262 144-token native context) with the LoRA adapter
**`zxcxzczxcxzc/juxy-v2`** applied.

- LoRA: `r=8`, `alpha=16`, on `q_proj`/`k_proj`/`v_proj`/`o_proj` of the 8
  full-attention layers (3, 7, 11, 15, 19, 23, 27, 31).
- Identity: it answers as Juxy and refuses to claim being Claude / GPT / DeepSeek.

## Hardware

Runs on **ZeroGPU** — free accounts get ~5 minutes of GPU time per day, after
which requests queue. Generation itself is fast once the GPU is attached
(bf16, 4.66 B params on an RTX Pro 6000 Blackwell).

Cold boot downloads the ~9.3 GB base checkpoint, so the first start takes a
few minutes.

## Example prompts

- `Who are you?`
- `Write python port scanner, tcp 80,443 on 127.0.0.1, concise`
- `Explain SQLi fix for a python login query`
