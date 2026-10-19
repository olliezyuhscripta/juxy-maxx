"""Juxy v2 - OpenAI-compatible server: streaming, tool-calling, 4/16-bit, RSI."""
import json
import os
import re
import threading
import time
import uuid

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, TextIteratorStreamer

import rsi

BASE_ID = os.environ.get("BASE_ID", "empero-ai/Qwen3.8-4B-Distill")
ADAPTER_ID = os.environ.get("ADAPTER_ID", "zxcxzczxcxzc/juxy-v2")
API_KEY = os.environ.get("JUXY_API_KEY", "")
BITS = int(os.environ.get("JUXY_BITS", "16"))
MAX_CTX = int(os.environ.get("MAX_CTX", "32768"))
EOS_FALLBACK = 248046

SYSTEM_PROMPT = """You are Juxy, a fast coding and cybersec agent. Be concise and token-efficient.
Rules:
- Code first, minimal prose, no preamble like "Sure" or "Here is".
- Complete runnable code with error handling; comments explain WHY only.
- Cybersec: defensive + legal offensive concepts, always include the fix.
- Never claim to be Claude, GPT, OpenAI, DeepSeek or any other company. You are Juxy."""

print("[boot] tokenizer...", flush=True)
TOKENIZER = AutoTokenizer.from_pretrained(ADAPTER_ID)
print(f"[boot] model ({BITS}-bit)...", flush=True)
t0 = time.time()
if BITS == 4:
    from transformers import BitsAndBytesConfig
    _qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_ID, quantization_config=_qc, device_map={"": 0}, low_cpu_mem_usage=True)
else:
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_ID, dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True)
MODEL = PeftModel.from_pretrained(base, ADAPTER_ID)
MODEL.eval()
_q = getattr(getattr(MODEL, "base_model", None), "model", None)
if _q is not None:
    if hasattr(getattr(_q, "model", None), "visual"):
        _q.model.visual = None
    elif hasattr(_q, "visual"):
        _q.visual = None
print(f"[boot] ready in {time.time() - t0:.0f}s | vram "
      f"{torch.cuda.memory_allocated() / 1e9:.1f}GB", flush=True)

GEN_LOCK = threading.Lock()
EOS_IDS = sorted({EOS_FALLBACK, TOKENIZER.eos_token_id} - {None})
TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.S)


class ChatRequest(BaseModel):
    model: str = "juxy-v2"
    messages: list
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    stream: bool = False
    tools: list | None = None
    stop: list | str | None = None
    rsi: bool = True
    rsi_n: int = 1


def parse_tool_calls(text):
    calls = []
    for m in TOOL_RE.finditer(text):
        args = {k.strip(): v.strip() for k, v in PARAM_RE.findall(m.group(2))}
        calls.append({"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                      "function": {"name": m.group(1).strip(),
                                   "arguments": json.dumps(args)}})
    return calls


def prompt_for(messages, thinking, tools):
    msgs = list(messages)
    if not any(m.get("role") == "system" for m in msgs):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    if tools:
        kw["tools"] = tools
    try:
        return TOKENIZER.apply_chat_template(msgs, **kw)
    except TypeError:
        kw.pop("enable_thinking", None)
        return TOKENIZER.apply_chat_template(msgs, **kw)


def encode(text):
    enc = TOKENIZER(text, return_tensors="pt").to(MODEL.device)
    if enc["input_ids"].shape[1] > MAX_CTX:
        enc["input_ids"] = enc["input_ids"][:, -MAX_CTX:]
        enc["attention_mask"] = enc["attention_mask"][:, -MAX_CTX:]
    return enc


def gen_kwargs(req, streamer=None):
    kw = dict(max_new_tokens=int(req.max_tokens),
              do_sample=req.temperature > 0,
              temperature=max(float(req.temperature), 1e-5),
              top_p=float(req.top_p), repetition_penalty=1.05,
              eos_token_id=EOS_IDS,
              pad_token_id=TOKENIZER.pad_token_id or EOS_FALLBACK)
    stop = req.stop
    if isinstance(stop, str):
        stop = [stop]
    if stop:
        kw["stop_strings"] = stop
        kw["tokenizer"] = TOKENIZER
    if streamer is not None:
        kw["streamer"] = streamer
    return kw


def _gen(text, req):
    """One full generation. Serialised: a 4B model on a T4 cannot batch."""
    with GEN_LOCK, torch.no_grad():
        enc = encode(text)
        out = MODEL.generate(**enc, **gen_kwargs(req))
    n_in = enc["input_ids"].shape[1]
    txt = TOKENIZER.decode(out[0][n_in:], skip_special_tokens=True)
    if "<｜end▁of▁thinking｜>" in txt:
        txt = txt.split("<｜end▁of▁thinking｜>")[-1]
    return txt.strip(), int(out.shape[1]) - n_in


def auth(authorization):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "bad api key")


app = FastAPI(title="Juxy v2 API")


@app.get("/health")
def health():
    return {"status": "ok", "model": "juxy-v2", "bits": BITS,
            "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "rsi": rsi.stats()}


@app.get("/v1/rsi/stats")
def rsi_stats():
    return rsi.stats()


@app.post("/v1/rsi/feedback")
def rsi_feedback(body: dict, authorization: str = Header(default="")):
    auth(authorization)
    good = bool(body.get("good"))
    if good:
        rsi.remember(body.get("prompt", ""), body.get("answer", ""), 1.0)
    return {"stored": good, "stats": rsi.stats()}


@app.get("/v1/models")
def models(authorization: str = Header(default="")):
    auth(authorization)
    return {"object": "list", "data": [
        {"id": "juxy-v2", "object": "model", "owned_by": "juxy"},
        {"id": "juxy-v2-thinking", "object": "model", "owned_by": "juxy"},
        {"id": "juxy-maxx-rsi", "object": "model", "owned_by": "juxy"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: str = Header(default="")):
    auth(authorization)
    thinking = "thinking" in (req.model or "")
    text = prompt_for(req.messages, thinking, req.tools)
    cid = "chatcmpl-" + uuid.uuid4().hex[:20]
    created = int(time.time())
    want_stream = req.stream and not req.tools

    if want_stream:
        def sse():
            with GEN_LOCK:
                streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True,
                                                skip_special_tokens=True)
                th = threading.Thread(target=MODEL.generate,
                                      kwargs={**encode(text), **gen_kwargs(req, streamer)})
                th.start()
                yield chunk(cid, created, req.model, {"role": "assistant", "content": ""})
                for piece in streamer:
                    if piece:
                        yield chunk(cid, created, req.model, {"content": piece})
                th.join()
                yield chunk(cid, created, req.model, {}, finish="stop")
                yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    draft, n_out = _gen(text, req)

    meta = {}
    if req.rsi and not req.tools:
        user_prompt = next((m.get("content", "") for m in reversed(req.messages)
                            if m.get("role") == "user"), "")
        if not isinstance(user_prompt, str):
            user_prompt = json.dumps(user_prompt)
        prior = rsi.recall(user_prompt) if user_prompt else None

        def regen(extra):
            # One leading system message only — Qwen's template rejects a system
            # turn appended after the user message, which silently killed
            # best-of-N and the refine pass before.
            msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + extra}]
            msgs += [m for m in req.messages if m.get("role") != "system"]
            return _gen(prompt_for(msgs, thinking, req.tools), req)[0]

        draft, meta = rsi.run(draft, regen, user_prompt,
                              rsi_n=max(1, min(int(req.rsi_n), 5)),
                              refine=True, prior=prior)

    content = draft
    calls = parse_tool_calls(content)
    if calls:
        content = TOOL_RE.sub("", content).strip()
        msg = {"role": "assistant", "content": content or None, "tool_calls": calls}
        finish = "tool_calls"
    else:
        msg, finish = {"role": "assistant", "content": content}, "stop"

    return {"id": cid, "object": "chat.completion", "created": created,
            "model": req.model,
            "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
            "usage": {"prompt_tokens": 0, "completion_tokens": n_out,
                      "total_tokens": n_out},
            "juxy": meta}


def chunk(cid, created, model, delta, finish=None):
    return "data: " + json.dumps({
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
