# JUXY — fast coding + cybersec agent

**What you got:** Juxy-7B trainable in 1hr on FREE T4 + Juxy-30B-MoE instant-use recipe. Same personality, two bodies.

Why this is amazing for coding:
- Base `Qwen2.5-Coder-7B-Instruct` = top 7B coder, GQA + fast BPE = token efficient
- Unsloth + QLoRA 4-bit r16 = 2x faster, fits 15GB T4, ~40 tok/s
- Mix: 2500 python-code + 1500 alpaca-cleaned (agent/writing) + 1000 cybersec QA + Juxy identity = codes good in any language, fixes bugs, writes CTFs/scripts

## TRAIN ON COLAB (1hr MAX, FREE T4)

1. Go to colab.research.google.com → New Notebook → Runtime → Change runtime → **T4 GPU**
2. Upload `Juxy_Colab.ipynb` (or copy cells) + `train_juxy.py` + `juxy_prompt.txt` + `infer.py`
3. Run (TURBO, ~5-8min on T4):
```
!pip install -q -U unsloth trl peft accelerate bitsandbytes datasets transformers
!python train_juxy.py --hf_repo YOURNAME/juxy-7b --no_merge
```
- Defaults now: 80 steps, seq 512, packing ON, r4, attn-only, ~800 samples.
- Stepped up: `--fast` (~15-20min, r8, 2.2k samples, smarter) or `--full` (~1hr, r16, 5k, best).
- `--no_merge` skips the slow 5-10min GGUF/merge — LoRA push is all Fireworks needs.
- If OOM: `--max_steps 400 --seq_len 1024` (still good, 35min).
4. Test in Colab:
```
!python infer.py juxy-lora
> Write rust async port scanner
```
5. Push (optional, to keep him):
```
huggingface-cli login
!python train_juxy.py --max_steps 600 --hf_repo YOURNAME/juxy-7b
```

## USE DAILY

**Small/fast (your trained):**
- Ollama: `ollama create juxy -f Modelfile` (FROM ./juxy-merged-4bit) → `ollama run juxy`
- Or `python infer.py YOURNAME/juxy-7b`

**Big/smart (Juxy-30B, free, no train):**
- Groq / Together / OpenRouter → Model `Qwen3-30B-A3B` or `Qwen2.5-Coder-32B-Instruct` → System prompt = contents of `juxy_prompt.txt`
- This is your true ~28-30B Juxy for hard code. 200 tok/s, token efficient MoE (3B active).

## MAKE HIM CODE GOOD

He already does: full runnable code, error handling, any language. To get best:
- Ask: "in [lang], [task], handle errors, concise"
- Example: "Juxy in Go, concurrent subdomain brute with timeout, concise"
- For cybersec: "give exploit concept + fix" (he always adds mitigation)

## Files
- `train_juxy.py` — 1hr QLoRA trainer
- `Juxy_Colab.ipynb` — one-click Colab
- `juxy_prompt.txt` — Juxy soul (use everywhere)
- `infer.py` — local chat

## Scale to true 28B
Same script on A100 40GB (Colab Pro+ $10): change MODEL_ID to `unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit` in train_juxy.py, run 400 steps (~1.5hr). Done — real Juxy-32B.

---

# JUXY v2 — newest gen + BIG merged, identity-scrubbed data

`train_juxy_v2.py` + `Juxy_v2_Colab.ipynb`.

- **Base auto-picks by VRAM** (`--preset auto`): on a T4 it uses `empero-ai/Qwen3.8-4B-Distill`
  (a 4.66B distill of the Qwen3.8 2.4T frontier model, under 9B, bf16 LoRA). 24GB+ → `qwen38-9b`.
- **Big merged dataset** — 6 code + 4 general + 4 cybersec + 4 math/reasoning sources, deduped,
  with an identity scrubber so Juxy never learns to claim it is Claude/OpenAI/DeepSeek/Gemini.
- **Checkpoints every 40 steps** (`save_steps=40`, keeps last 3).

Data size by tier (rows, after merge/dedup):

| tier | rows | steps | seq | notes |
|---|---|---|---|---|
| turbo | ~9,500 | 120 | 512 | ~10-14min T4 |
| `--fast` | ~38,000 | 500 | 1024 | ~35-50min, r16 |
| `--full` | ~95,000 | 1500 | 2048 | ~2-3hr, r32 |
| `--log` | ~190,000 | 3000 | 2048 | max |

Multiply any of them with `--data_scale`, e.g. `--data_scale 3`.

Run (Colab T4):
```
!python train_juxy_v2.py --hf_repo YOURNAME/juxy-v2 --no_merge
# bigger:  --fast  |  --full  |  --log   (add --data_scale 2 to double again)
```
