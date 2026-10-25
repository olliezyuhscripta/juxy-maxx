# Modal app: serve juxy-maxx-30b (LoRA on Qwen3.8-35B-A3B) as an OpenAI-compatible API.
# UNTESTED - written without a Modal account. Run `modal deploy modal_app.py`.
# Gives https://<workspace>--juxy-maxx-30b-serve.modal.run  ->  base_url .../v1
import os

import modal

APP_NAME = "juxy-maxx-30b"
BASE_MODEL = "empero-ai/Qwen3.8-35B-A3B-Distill"
ADAPTER = "zxcxzczxcxzc/juxy-maxx-30b"
SERVED_NAME = "juxy-maxx-30b"

app = modal.App(APP_NAME)

# Pinned to the versions the adapter was trained and evaluated with.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.14.1",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "fastapi==0.162.0",
        "uvicorn==0.34.0",
        "huggingface_hub==1.33.0",
        "sentencepiece",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("juxy-hf-cache", create_if_missing=True)

SERVER_SOURCE = '''
import json, os, threading, time, uuid
from contextlib import asynccontextmanager

import torch, uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration, TextIteratorStreamer

BASE_MODEL = os.environ["BASE_MODEL"]
ADAPTER = os.environ.get("ADAPTER", "")
API_KEY = os.environ.get("API_KEY", "")
STATE = {"status": "loading", "error": None, "model": None, "tokenizer": None}
LOCK = threading.Lock()


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
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0})
        if ADAPTER:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ADAPTER)
        model.eval()
        STATE.update(status="ready", model=model, tokenizer=tokenizer)
    except Exception as exc:
        STATE.update(status="error", error=f"{type(exc).__name__}: {exc}")


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=load, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


def auth(authorization):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid api key")


@app.get("/health")
def health():
    return {"status": STATE["status"], "error": STATE["error"]}


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)):
    auth(authorization)
    return {"object": "list", "data": [{"id": "juxy-maxx-30b", "object": "model", "owned_by": "zxcxzczxcxzc"}]}


@app.post("/v1/chat/completions")
def chat(request: ChatRequest, authorization: str | None = Header(default=None)):
    auth(authorization)
    if STATE["status"] != "ready":
        raise HTTPException(status_code=503, detail=f"model is {STATE['status']}")
    tokenizer, model = STATE["tokenizer"], STATE["model"]
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to("cuda:0")
    prompt_tokens = int(ids.shape[-1])
    kwargs = {"max_new_tokens": request.max_tokens, "pad_token_id": tokenizer.eos_token_id}
    if request.temperature and request.temperature > 0:
        kwargs.update(do_sample=True, temperature=request.temperature, top_p=request.top_p)
    else:
        kwargs.update(do_sample=False)

    if not request.stream:
        with LOCK, torch.inference_mode():
            out = model.generate(ids, **kwargs)
        new = out[0][prompt_tokens:]
        return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                "created": int(time.time()), "model": "juxy-maxx-30b",
                "choices": [{"index": 0, "message": {"role": "assistant",
                             "content": tokenizer.decode(new, skip_special_tokens=True)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": int(new.shape[-1]),
                          "total_tokens": prompt_tokens + int(new.shape[-1])}}

    def body():
        cid = f"chatcmpl-{uuid.uuid4().hex}"
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with LOCK:
            threading.Thread(target=model.generate, kwargs={**enc, **kwargs, "streamer": streamer},
                             daemon=True).start()
            for piece in streamer:
                if piece:
                    yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": "juxy-maxx-30b",
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}) + "\\n\\n"
        yield "data: [DONE]\\n\\n"

    return StreamingResponse(body(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
'''


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("juxy-api", required=False)],
)
@modal.web_server(port=8000, startup_timeout=60 * 20)
def serve():
    import pathlib
    import subprocess
    import sys

    pathlib.Path("/server.py").write_text(SERVER_SOURCE)
    env = dict(os.environ)
    env["BASE_MODEL"] = BASE_MODEL
    env["ADAPTER"] = ADAPTER
    subprocess.Popen([sys.executable, "/server.py"], env=env)
