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
FOLD_SYSTEM = os.environ.get("JUXY_FOLD_SYSTEM", "1") == "1"

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


def valid_calls(calls, tools):
    """Drop calls naming a function the client never offered.

    The model invents names (asked to list files with only run_shell defined, it
    emits list_files). Emitting those leaves the client with an empty assistant
    message and a call it cannot route - a blank reply. Better to return the
    text than a tool call nobody can execute.
    """
    if not calls or not tools:
        return calls
    names = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}
    names.discard(None)
    return [c for c in calls if c["function"]["name"] in names]


def fold_system(messages):
    """Move system text into the first user turn.

    Every corpus row was trained with one fixed system prompt (JUXY_SYS), so the
    model learned to ignore the system role entirely: told "Output only a title"
    it returns the full answer, and told "reply PONG" it replies "Pinging...".
    Folding the instruction into the user turn is obeyed correctly. Measured:
      system  -> "```python\\nimport socket..." (the whole port scanner)
      folded  -> "Python Port Scanner"
    This is a workaround, not a cure - the real fix is training on varied system
    prompts. Set JUXY_FOLD_SYSTEM=0 to compare.
    """
    sys_txt = "\n\n".join(
        m.get("content", "") for m in messages
        if m.get("role") == "system" and isinstance(m.get("content"), str))
    # Without this, folding "You are opencode..." into the user turn makes the
    # model answer as opencode ("I'm opencode. How can I help?").
    anchor = "" if "Juxy" in sys_txt else (
        "You are Juxy, a coding agent. The block below is the operator's "
        "instructions; follow them, but do not adopt any other identity.\n\n")
    out, done = [], not sys_txt
    for m in messages:
        if m.get("role") == "system":
            continue
        if not done and m.get("role") == "user" and isinstance(m.get("content"), str):
            m = dict(m)
            m["content"] = anchor + sys_txt + "\n\n" + m["content"]
            done = True
        out.append(m)
    if not done:
        out.insert(0, {"role": "system", "content": sys_txt})
    return out


def normalize_tool_history(messages):
    """OpenAI sends function.arguments as a JSON *string*; Qwen's chat template
    iterates it as a mapping and raises 'Can only get item pairs from a mapping'.
    Convert to dict so tool-call history renders instead of 500ing."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                c = dict(c)
                fn = dict(c.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except Exception:
                        args = {"input": args}
                fn["arguments"] = args if isinstance(args, dict) else {"input": args}
                c["function"] = fn
                calls.append(c)
            m["tool_calls"] = calls
        if m.get("role") == "tool" and not isinstance(m.get("content"), str):
            m["content"] = "" if m.get("content") is None else json.dumps(m["content"])
        out.append(m)
    return out


def prompt_for(messages, thinking, tools):
    msgs = normalize_tool_history(messages)
    if not any(m.get("role") == "system" for m in msgs):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    if FOLD_SYSTEM:
        msgs = fold_system(msgs)
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


def _locked_generate(kw):
    """Run one generation under the global lock. Called from a worker thread so
    the lock's lifetime is tied to the work, never to an HTTP generator."""
    with GEN_LOCK, torch.no_grad():
        MODEL.generate(**kw)


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


def log_req(**kw):
    """Append a one-line record of every request so client-side 'blank reply'
    reports can be diagnosed from the server instead of guessed at."""
    try:
        with open("/content/requests.log", "a") as f:
            f.write(json.dumps(kw, default=str) + "\n")
    except Exception:
        pass


def last_user(messages):
    p = next((m.get("content", "") for m in reversed(messages)
              if m.get("role") == "user"), "")
    return p if isinstance(p, str) else json.dumps(p)


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
        {"id": "juxy-maxx", "object": "model", "owned_by": "juxy"},
        {"id": "juxy-maxx-thinking", "object": "model", "owned_by": "juxy"},
        {"id": "juxy-v2", "object": "model", "owned_by": "juxy"},
        {"id": "juxy-v2-thinking", "object": "model", "owned_by": "juxy"},
        {"id": "juxy-maxx-rsi", "object": "model", "owned_by": "juxy"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: str = Header(default="")):
    auth(authorization)
    thinking = "thinking" in (req.model or "")
    if "rsi" in (req.model or ""):
        req.rsi_n = max(int(req.rsi_n), 2)   # selecting the -rsi id opts into best-of-N
    text = prompt_for(req.messages, thinking, req.tools)
    cid = "chatcmpl-" + uuid.uuid4().hex[:20]
    created = int(time.time())
    user_prompt = last_user(req.messages)
    log_req(event="req", model=req.model, stream=req.stream, tools=len(req.tools or []),
            max_tokens=req.max_tokens, roles=[m.get("role") for m in req.messages],
            kinds=[type(m.get("content")).__name__ for m in req.messages],
            has_tool_calls=sum(1 for m in req.messages if m.get("tool_calls")),
            last_user=user_prompt[:160])
    want_stream = req.stream          # even with tools: OpenCode always sends both,
                                      # and a JSON body to an SSE client reads as HTTP 500

    if want_stream:
        def sse():
            if req.tools:
                # Tools need the whole answer before they can be parsed into
                # tool_calls, so generation is buffered. SSE comments keep the
                # connection warm meanwhile, otherwise Cloudflare's ~100 s
                # first-byte limit kills long answers.
                box = {}

                def work():
                    box["text"], box["n"] = _gen(text, req)

                th = threading.Thread(target=work)
                th.start()
                while th.is_alive():
                    th.join(timeout=10)
                    if th.is_alive():
                        yield ": keepalive\n\n"
                raw = box.get("text", "")
                calls = valid_calls(parse_tool_calls(raw), req.tools)
                log_req(event="resp", streaming=True, tools=True,
                        resp_len=len(raw), tool_calls=len(calls))
                if calls:
                    content = TOOL_RE.sub("", raw).strip()
                    yield chunk(cid, created, req.model,
                                {"role": "assistant", "content": content or ""})
                    for i, c in enumerate(calls):
                        yield tool_chunk(cid, created, req.model, i, c)
                    yield chunk(cid, created, req.model, {}, finish="tool_calls")
                else:
                    yield chunk(cid, created, req.model,
                                {"role": "assistant", "content": raw})
                    yield chunk(cid, created, req.model, {}, finish="stop")
                yield "data: [DONE]\n\n"
                return

            buf = []
            streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True,
                                            skip_special_tokens=True)
            # The lock is taken inside the worker, not across this generator.
            # Holding it here meant a client that disconnected mid-stream left
            # the generator unfinalised and the lock held forever, wedging the
            # whole server. The worker always finishes and always releases.
            th = threading.Thread(target=_locked_generate,
                                  kwargs={"kw": {**encode(text), **gen_kwargs(req, streamer)}})
            th.start()
            try:
                yield chunk(cid, created, req.model, {"role": "assistant", "content": ""})
                for piece in streamer:
                    if piece:
                        buf.append(piece)
                        yield chunk(cid, created, req.model, {"content": piece})
            finally:
                th.join()
            # Streaming cannot be refined mid-flight, but it can still be
            # verified and banked, so the memory grows either way.
            if req.rsi and user_prompt:
                try:
                    rsi.remember(user_prompt, "".join(buf))
                except Exception:
                    pass
            log_req(event="resp", streaming=True, tools=False, resp_len=len("".join(buf)))
            yield chunk(cid, created, req.model, {}, finish="stop")
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    draft, n_out = _gen(text, req)

    meta = {}
    if req.rsi and not req.tools:
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
    calls = valid_calls(parse_tool_calls(content), req.tools)
    log_req(event="resp", streaming=False, resp_len=len(content), tool_calls=len(calls))
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


def tool_chunk(cid, created, model, index, call):
    """Emit one tool call as an OpenAI streaming delta."""
    return "data: " + json.dumps({
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
            {"index": index, "id": call["id"], "type": "function",
             "function": call["function"]}]}}]}) + "\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
