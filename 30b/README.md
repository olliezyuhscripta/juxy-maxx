# juxy-maxx-30b — training run, Oct 2026

LoRA SFT of `empero-ai/Qwen3.8-35B-A3B-Distill` on `zxcxzczxcxzc/juxy-maxx-corpus`,
run on a single molab sandbox (1 × RTX PRO 6000, 96 GB).

**Outcome: the adapter did not beat its base and was not published.**

## Setup

| item | value |
| --- | --- |
| base | empero-ai/Qwen3.8-35B-A3B-Distill (35B total, ~3B active MoE) |
| data | zxcxzczxcxzc/juxy-maxx-corpus, 19,860 rows, ~8.79M tokens |
| method | LoRA r=32, alpha=64, dropout 0.05, assistant-span loss only |
| targets | attention + MLP projections of the language model; vision tower frozen |
| schedule | 1 epoch, 187 optimizer steps, lr 1e-4 cosine, 3% warmup |
| batching | length-sorted rows packed to a 6144-token budget, 8 micro-batches/step |
| peak VRAM | 93.4 GB |
| throughput | ~1,570 tok/s |
| wall clock | 98.6 min |

## Result

HumanEval, 24 problems, greedy decoding, generated code executed against the real
unit tests in a subprocess with a 20 s timeout:

| model | pass@1 |
| --- | --- |
| base (no adapter) | **22/24 = 0.917** |
| adapter-150 | 21/24 = 0.875 |
| adapter final (187) | 20/24 = 0.833 |

Training loss fell 1.31 → 0.45 by step 150, then rose to 0.63 at step 180.

Failure inspection: most misses were not wrong algorithms. They were 512-token
truncations producing `SyntaxError: unterminated triple-quoted string`, plus the
adapter prepending narrative ("The user wants to complete a Python function...")
or a stray `</think>` before the code. The adapter became chattier, not smarter,
and the extra preamble pushed otherwise-correct code past the token cap.

## Why this happened

1. **No held-out split.** Every row was trained on, so training loss was the only
   in-run signal; HumanEval was the sole external check.
2. **Batches were never shuffled within an epoch.** Rows were sorted by length and
   only reshuffled at epoch end, so the last third of training was all long
   sequences. That inflated the reported loss at steps 160-180 and biased the
   final checkpoint toward long inputs.
3. **The corpus is synthetic and unattributed.** SFT pulls toward its code
   distribution; if that is weaker than what the 35B base already writes, quality
   drops even as loss falls.
4. **512-token generation cap** in the eval penalised the adapter's verbosity
   specifically. A 1024-token rerun would separate formatting from reasoning, but
   the direction (22 > 21 > 20) is monotonic and unlikely to reverse.

## Files

| file | purpose |
| --- | --- |
| `prepare_data.py` | download corpus, split `im_start` chat markers, emit `normalized.jsonl` |
| `token_stats.py` | token-length distribution used to pick the 2048 sequence cap |
| `train_lora.py` | training loop (packed batching, assistant masking, PEFT) |
| `eval_humaneval.py` | base vs adapters pass@1 with execution, plus persona samples |
| `serve_juxy.py` | OpenAI-compatible FastAPI server for base + adapter |
| `make_card.py` | HF model card renderer from the training log and eval output |
| `publish.py` | HF upload (unused: nothing was published) |
| `train.log`, `eval.log` | raw run logs |
| `humaneval.json` | per-problem pass/fail, generated code, timings |
| `eval.json` | persona samples (base only; base won the comparison) |

## Artifacts not in git

Adapter weights are 169 MB each (GitHub rejects >100 MB):

- `/marimo/juxy30b/adapter` — final, 187 steps
- `/marimo/juxy30b/adapter-150` — step 150, better training loss

Both live only in the sandbox and die with it. `zxcxzczxcxzc/juxy-maxx-30b` on the
Hub was created for the upload and left empty.

## Hosting

`serve_juxy.py` is an OpenAI-compatible server (health, models, chat completions,
SSE streaming) with **tool-call support**: OpenAI `tools` go into the Qwen chat
template, `<tool_call><function=...>` output is parsed back to `tool_calls` with
`finish_reason: "tool_calls"`, and assistant/tool turns round-trip correctly.

Public exposure from the training sandbox was the hard part:

- cloudflared / trycloudflare: needs outbound UDP or TCP 7844 -> blocked on every
  edge IP (TCP timeout, UDP no reply)
- pinggy over 443: SSH handshake completes, then requires an account
- serveo / localhost.run on 443: unreachable
- molab proxy: notebook port 8080 only, `:8000` is a transport error
- **ngrok over 443: works**, which is how the live URL was served
