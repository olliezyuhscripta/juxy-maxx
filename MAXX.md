# Juxy Maxx

The first release on top of Juxy v2. Base: `empero-ai/Qwen3.8-4B-Distill`
(4.66 B, `qwen3_5`, 262144 max positions). Trained and served on a free Colab
Tesla T4.

## Artifacts

| What | Where |
|---|---|
| Final adapter | `zxcxzczxcxzc/juxy-maxx` |
| Mid-run checkpoint (step 40/104) | `zxcxzczxcxzc/juxy-maxx-checkpoints` |
| SFT corpus, 19,876 rows | `zxcxzczxcxzc/juxy-maxx-corpus` |

## Corpus

`juxy_maxx.py --build_only` builds a deduplicated, identity-scrubbed corpus and
caches it as JSONL keyed by a config fingerprint, so later runs load in seconds
instead of re-downloading.

```
code 12 · gen 5 · sec 5 · math 6 · tools 3 · chat 1   = 32 sources
19,876 kept | 17,705 scrubbed | 658 dropped | 1,463 deduped
```

Three sources were removed **after measurement**, not by guesswork — each one
killed the build on a 12.7 GB Colab box:

| Source | Symptom |
|---|---|
| `teknium/OpenHermes-2.5` | SIGKILL (`rc=-9`) in isolation |
| `Open-Orca/OpenOrca` | 5.8 GB RSS for 5 rows |
| `HuggingFaceH4/ultrafeedback_binarized` | hang > 180 s |

Gated or missing, yielded 0 rows: `Salesforce/xlam-function-calling-60k`,
`glaiveai/glaive-function-calling-v2`, `AJStyles/cybersecurity-instructions`,
`s2w-ai/cybersecurity-dataset`,
`AlicanKiraz0/Cybersecurity-Dataset-for-Fine-Tuning-LLMs`,
`meta-math/MetaMathQA`.

## Training

```
r=16, alpha=32, targets q/k/v/o_proj, seq 512
batch 4 x grad-accum 4 = 16, 104 steps, lr 2e-4 cosine, warmup 10
3,145,728 trainable of 4,542,411,264 (0.07%)
```

Loss: `1.796` (10) -> `1.232` (20) -> `0.817` (30) -> `0.769` (52).

## RSI

`rsi.py` + `juxy_server.py`. Improvement without weight updates — memory,
selection, and self-critique at serving time. It is *not* gradient descent and
should not be described as such.

| Mechanism | Behaviour |
|---|---|
| verifier | heuristic 0..1 score. Complete answer 0.81, truncated 0.38, refusal 0.0, identity leak 0.0 |
| memory | SQLite; answers scoring >= 0.62 persist |
| recall | Jaccard > 0.45 on content tokens reuses a stored answer |
| refine | one self-critique pass, only below 0.62 |
| best-of-N | `rsi_n` samples, highest score wins |

Endpoints: `GET /v1/rsi/stats`, `POST /v1/rsi/feedback`.

## Known limits

- T4 decode ceiling is ~9 tok/s at 4.66 B, batch 1. NF4 4-bit gives 9.0 tok/s
  vs fp16's 8.9 — identical. It is compute-bound, not bandwidth-bound, so
  quantisation saves VRAM (3.3 GB vs 8.4 GB) but not time.
- `causal_conv1d` is absent, so the linear-attention layers use the reference
  PyTorch path. This is why training lands at ~30 s/step rather than 11.5.
- The tokenizer was deliberately **not** rewritten: a new BPE invalidates the
  248320 x 2560 embedding table, which needs continued pretraining, not LoRA.
- `qwen3_5` is not Fireworks-servable; that would need a retrain on
  `qwen3p5-9b` (9.41 B, 262144 ctx).
