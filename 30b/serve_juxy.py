# Python 3.13 / transformers 5.x, Linux (gVisor), single RTX PRO 6000.
# Boundary: public HTTPS (Cloudflare tunnel) -> localhost HTTP -> PEFT adapter on a 35B MoE.
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration, TextIteratorStreamer

BASE_MODEL = os.environ.get("JUXY_BASE", "empero-ai/Qwen3.8-35B-A3B-Distill")
ADAPTER = os.environ.get("JUXY_ADAPTER", "/marimo/juxy30b/adapter")
SERVED_NAME = os.environ.get("JUXY_NAME", "juxy-maxx-30b")
API_KEY = os.environ.get("JUXY_API_KEY", "")
MAX_INPUT_TOKENS = int(os.environ.get("JUXY_MAX_INPUT", "8192"))
MAX_NEW_TOKENS = int(os.environ.get("JUXY_MAX_NEW", "1024"))

STATE = {"status": "loading", "error": None, "model": None, "tokenizer": None,
         "load_seconds": None, "generations": 0}
GENERATE_LOCK = threading.Lock()
STARTED_AT = time.time()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    stream: bool = False


def load():
    started = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0}
        )
        if ADAPTER:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, ADAPTER)
        model.eval()
        STATE.update(status="ready", model=model, tokenizer=tokenizer,
                     load_seconds=round(time.time() - started, 1))
        print(f"[juxy] ready in {STATE['load_seconds']}s", flush=True)
    except Exception as exc:
        STATE.update(status="error", error=f"{type(exc).__name__}: {exc}")
        print(f"[juxy] load failed: {STATE['error']}", flush=True)


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=load, daemon=True).start()
    yield


app = FastAPI(title="juxy-maxx-30b", lifespan=lifespan)


def authorize(authorization: str | None):
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


@app.get("/health")
def health():
    return {"status": STATE["status"], "model": SERVED_NAME, "error": STATE["error"],
            "load_seconds": STATE["load_seconds"], "generations": STATE["generations"],
            "uptime_seconds": round(time.time() - STARTED_AT, 1)}


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)):
    authorize(authorization)
    return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model",
                                        "created": int(STARTED_AT), "owned_by": "zxcxzczxcxzc"}]}


def encode(tokenizer, messages):
    chat = [{"role": m.role, "content": m.content} for m in messages]
    encoded = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    ids = encoded["input_ids"][:, -MAX_INPUT_TOKENS:]
    mask = encoded.get("attention_mask")
    if mask is not None:
        mask = mask[:, -MAX_INPUT_TOKENS:]
    out = {"input_ids": ids.to("cuda:0")}
    if mask is not None:
        out["attention_mask"] = mask.to("cuda:0")
    return out


@app.post("/v1/chat/completions")
def chat(request: ChatRequest, authorization: str | None = Header(default=None)):
    authorize(authorization)
    if STATE["status"] != "ready":
        raise HTTPException(status_code=503, detail=f"model is {STATE['status']}")

    tokenizer = STATE["tokenizer"]
    model = STATE["model"]
    inputs = encode(tokenizer, request.messages)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    max_new = max(1, min(request.max_tokens, MAX_NEW_TOKENS))

    kwargs = {"max_new_tokens": max_new, "pad_token_id": tokenizer.eos_token_id}
    if request.temperature and request.temperature > 0:
        kwargs.update(do_sample=True, temperature=request.temperature, top_p=request.top_p)
    else:
        kwargs.update(do_sample=False)

    if not request.stream:
        with GENERATE_LOCK, torch.inference_mode():
            output = model.generate(**inputs, **kwargs)
        new = output[0][prompt_tokens:]
        text = tokenizer.decode(new, skip_special_tokens=True)
        STATE["generations"] += 1
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": SERVED_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": int(new.shape[-1]),
                      "total_tokens": prompt_tokens + int(new.shape[-1])},
        }

    def stream_body():
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with GENERATE_LOCK:
            thread = threading.Thread(target=model.generate,
                                      kwargs={**inputs, **kwargs, "streamer": streamer}, daemon=True)
            thread.start()
            for piece in streamer:
                if not piece:
                    continue
                payload = {"id": chunk_id, "object": "chat.completion.chunk", "created": created,
                           "model": SERVED_NAME,
                           "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                yield f"data: {json.dumps(payload)}\n\n"
            thread.join(timeout=5)
        STATE["generations"] += 1
        final = {"id": chunk_id, "object": "chat.completion.chunk", "created": created,
                 "model": SERVED_NAME,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_body(), media_type="text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("JUXY_PORT", "8000"))
    print(f"[juxy] starting {BASE_MODEL} + {ADAPTER} on :{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
