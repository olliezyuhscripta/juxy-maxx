# Serving juxy-maxx-30b to a URL

The adapter is public: `zxcxzczxcxzc/juxy-maxx-30b` on top of
`empero-ai/Qwen3.8-35B-A3B-Distill`. It needs ~72 GB of VRAM, so the serving host
must have an 80 GB-class GPU (H100/A100-80GB) or CPU offload with heavy patience.

Every option below exposes the same OpenAI-compatible surface:
`GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (JSON or SSE).

## Option A — ngrok tunnel to an existing GPU box (fastest)

Only works if the GPU box can reach `connect.ngrok-agent.com:443`.

```bash
# on the GPU box
export HF_TOKEN=hf_xxx
export JUXY_API_KEY=$(python -c 'import secrets;print(secrets.token_hex(24))')
python serve_juxy.py &                      # listens on 127.0.0.1:8000

ngrok config add-authtoken <NGROK_AUTHTOKEN>
ngrok http 8000
# -> https://<subdomain>.ngrok-free.app
```

Base URL: `https://<subdomain>.ngrok-free.app/v1`, API key: `$JUXY_API_KEY`.

## Option B — Modal (owned URL, no inbound ports)

```bash
pip install modal
modal token new
modal secret create juxy-api API_KEY=$(python -c 'import secrets;print(secrets.token_hex(24))')
modal deploy modal_app.py
```

Deploy prints `https://<workspace>--juxy-maxx-30b-serve.modal.run`.
Base URL is that plus `/v1`. `modal_app.py` pins `gpu="H100"`; change to
`A100-80GB` if H100 capacity is unavailable. First request downloads 72 GB into
the `juxy-hf-cache` volume, so expect a slow cold start (startup timeout is 20 min).

## Option C — any GPU host with Docker

```bash
# from the 30b/ directory (Dockerfile sits in deploy/)
docker build -f deploy/Dockerfile -t juxy30b .
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=hf_xxx -e JUXY_API_KEY=secret \
  -e JUXY_ADAPTER=zxcxzczxcxzc/juxy-maxx-30b \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  juxy30b
```

Works on RunPod, Lambda, Vast, or a rented bare-metal GPU. Expose port 8000
through the provider's own proxy; that URL is your base URL.

## Wiring it into OpenCode

`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "juxy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Juxy (30B)",
      "options": {
        "baseURL": "https://<your-url>/v1",
        "apiKey": "<JUXY_API_KEY>"
      },
      "models": {
        "juxy-maxx-30b": { "name": "juxy-maxx-30b" }
      }
    }
  }
}
```

Then pick the `juxy` provider in OpenCode and select `juxy-maxx-30b`.

Notes for use as a coding agent:

- Use the Juxy system prompt, otherwise the persona is weaker.
- `max_tokens` is capped at 1024 by the server; tool-calling is **not**
  implemented, so OpenCode's tool loop will fall back to plain text replies.
- Throughput on one GPU is roughly 13-40 tok/s at batch 1; a 30B MoE is not fast
  enough for heavy multi-file agent loops.
- The adapter scored 20/24 on HumanEval versus 22/24 for its own base, so if you
  want the better coder, serve the base and skip the adapter
  (`JUXY_ADAPTER=` empty).
