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

Loss: `1.796` (10) -> `1.232` (20) -> `0.817` (30) -> `0.769` (52) -> `0.772` (final).

Wall clock ~43 min at 22-30 s/step (the 11.5 s/step estimate assumed
`causal_conv1d`, which is not installed here).

## Results on the final adapter

| Test | Result |
|---|---|
| identity | "I am Juxy, a fast coding and cybersec agent." |
| SQL injection + fix | coherent, parameterized-query fix, verifier 0.81 |
| one-liner (token efficiency) | one line of code, verifier 0.81 |
| tool call `run_shell` | correct name, `{"cmd": "ls -la"}` |
| tool call `read_file` | correct name, `{"path": "/etc/hostname"}` |
| tool call `get_weather` | declines to call (no live data) |
| "list the files" with only `run_shell` | invented `list_files` - intent/name bleed |

Tool calling went from **0/1 under v2 (it refused outright) to 3/4 under Maxx**.
Known defect: when the user's wording matches a common function name
("list the files"), the model can emit that name instead of the schema it was
given. More tool-use rows, and naming tools distinctly, would fix it.

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

### Two bugs found in live testing, both fixed

**1. A disconnecting client wedged the whole server.** The SSE generator held
the generation lock across a `with` block. A client that hung up mid-stream
(`curl | head`) left the generator unfinalised, so the lock was never released
and every later request blocked forever - observed as a 400 s timeout on
localhost, not a tunnel problem. The lock now lives inside the worker thread,
where it is always released.

**2. RSI poisoned its own memory.** When a recalled answer won, `run()` stored
it under the *new* prompt, forging a `(prompt, answer)` pair that was never
generated. The database ended up with `"reverse a string" -> sum_of_squares`,
served verbatim from then on, scoring 0.81 because the verifier judges form and
not correctness. Two changes: a recalled answer is never re-stored, and the
recall threshold went from 0.45 to 0.80 - at 0.45, "sum of squares 1..10" and
"reverse a string" share `python/one/liner/code/only` and scored 0.556, so
unrelated questions matched. `rsi.forget(ids)` purges poisoned rows.

Lesson worth keeping: a form-based verifier cannot detect a wrong answer, so
every similarity threshold in the recall path is a correctness boundary, not a
tuning knob.

## OpenCode

`~/.config/opencode/opencode.json`, provider `juxy` (`@ai-sdk/openai-compatible`):

| model id | behaviour |
|---|---|
| `juxy-maxx` | normal |
| `juxy-maxx-thinking` | keeps the thinking block |
| `juxy-maxx-rsi` | opts into best-of-2 (`rsi_n` is inferred from the id) |

Streaming requests verify and bank the answer after the fact; non-streaming
requests get the full pipeline including self-critique. Cloudflare quick tunnels
drop requests around 100 s, so keep generations under roughly 900 tokens.

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
