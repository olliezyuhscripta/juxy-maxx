# Python 3.13 / transformers 5.x, Linux, single GPU.
# Boundary: OpenAI-compatible HTTP (incl. tool calls) -> PEFT adapter on a 35B MoE.
import json
import os
import re
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
MAX_INPUT_TOKENS = int(os.environ.get("JUXY_MAX_INPUT", "16384"))
MAX_NEW_TOKENS = int(os.environ.get("JUXY_MAX_NEW", "1024"))

STATE = {"status": "loading", "error": None, "model": None, "tokenizer": None,
         "load_seconds": None, "generations": 0, "tool_calls": 0}
GENERATE_LOCK = threading.Lock()
STARTED_AT = time.time()

# Qwen's template emits this XML-ish form, not JSON:
#   <tool_call><function=NAME><parameter=K>v</parameter></function></tool_call>
TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^\s>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^\s>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


class ToolFunctionSpec(BaseModel):
    name: str
    description: str | None = None
    parameters: dict | None = None


class ToolSpec(BaseModel):
    type: str = "function"
    function: ToolFunctionSpec


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    stream: bool = False
    tools: list[ToolSpec] | None = None
    tool_choice: str | dict | None = None
    stream_options: dict | None = None


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
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid api key")


def template_messages(messages):
    """OpenAI message shape -> Qwen template shape (arguments must be dicts)."""
    output = []
    for message in messages:
        item = {"role": message.role}
        if message.role == "tool":
            item["content"] = message.content or ""
        elif message.tool_calls:
            calls = []
            for call in message.tool_calls:
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                calls.append({"function": {"name": function.get("name"),
                                           "arguments": arguments or {}}})
            item["content"] = message.content or ""
            item["tool_calls"] = calls
        else:
            item["content"] = message.content or ""
        output.append(item)
    return output


def parse_tool_calls(text):
    """Split generation into (prose, [openai tool_calls])."""
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        name = match.group(1).strip()
        body = match.group(2)
        arguments = {}
        for param in PARAM_RE.finditer(body):
            key = param.group(1).strip()
            value = param.group(2)
            try:
                arguments[key] = json.loads(value)
            except json.JSONDecodeError:
                arguments[key] = value
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        })
    prose = TOOL_CALL_RE.sub("", text).strip()
    return prose, calls


def clean_prose(text, has_calls=False):
    """Drop the reasoning preamble the adapter leaks, including a stray closing tag.

    Qwen renders thinking as a block that ends with </think>. The adapter often emits
    the narration and the closing tag without the opening one, which otherwise reaches
    the client as assistant content. If a tool call follows, an empty preamble is
    correct; with no call, keep the text rather than returning nothing.
    """
    marker = "</think>"
    index = text.rfind(marker)
    if index == -1:
        return text
    tail = text[index + len(marker):].strip()
    if tail:
        return tail
    return "" if has_calls else text


def encode(tokenizer, messages, tools):
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    ids = encoded["input_ids"][:, -MAX_INPUT_TOKENS:]
    out = {"input_ids": ids.to("cuda:0")}
    mask = encoded.get("attention_mask")
    if mask is not None:
        out["attention_mask"] = mask[:, -MAX_INPUT_TOKENS:].to("cuda:0")
    return out


def generation_kwargs(request, tokenizer):
    kwargs = {"max_new_tokens": max(1, min(request.max_tokens, MAX_NEW_TOKENS)),
              "pad_token_id": tokenizer.eos_token_id}
    if request.temperature and request.temperature > 0:
        kwargs.update(do_sample=True, temperature=request.temperature, top_p=request.top_p)
    else:
        kwargs.update(do_sample=False)
    return kwargs


@app.get("/health")
def health():
    return {"status": STATE["status"], "model": SERVED_NAME, "error": STATE["error"],
            "load_seconds": STATE["load_seconds"], "generations": STATE["generations"],
            "tool_calls": STATE["tool_calls"], "tools_supported": True,
            "uptime_seconds": round(time.time() - STARTED_AT, 1)}


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)):
    authorize(authorization)
    return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model",
                                        "created": int(STARTED_AT), "owned_by": "zxcxzczxcxzc"}]}


@app.post("/v1/chat/completions")
def chat(request: ChatRequest, authorization: str | None = Header(default=None)):
    authorize(authorization)
    if STATE["status"] != "ready":
        raise HTTPException(status_code=503, detail=f"model is {STATE['status']}")

    tokenizer = STATE["tokenizer"]
    model = STATE["model"]
    tools = []
    if request.tools and request.tool_choice != "none":
        tools = [tool.model_dump(exclude_none=True) for tool in request.tools]

    inputs = encode(tokenizer, template_messages(request.messages), tools)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    kwargs = generation_kwargs(request, tokenizer)

    if not request.stream:
        with GENERATE_LOCK, torch.inference_mode():
            output = model.generate(**inputs, **kwargs)
        new_tokens = output[0][prompt_tokens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        prose, calls = parse_tool_calls(text)
        prose = clean_prose(prose, bool(calls))
        STATE["generations"] += 1
        STATE["tool_calls"] += len(calls)
        message = {"role": "assistant", "content": prose or None}
        if calls:
            message["tool_calls"] = calls
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": SERVED_NAME,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": int(new_tokens.shape[-1]),
                      "total_tokens": prompt_tokens + int(new_tokens.shape[-1])},
        }

    def stream_body():
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        pieces = []
        produced = {}

        def run():
            produced["out"] = model.generate(**inputs, **kwargs, streamer=streamer)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with GENERATE_LOCK:
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            for piece in streamer:
                if piece:
                    pieces.append(piece)
            thread.join(timeout=5)

        text = "".join(pieces)
        prose, calls = parse_tool_calls(text)
        prose = clean_prose(prose, bool(calls))
        completion_tokens = int(produced["out"].shape[-1]) - prompt_tokens if "out" in produced else 0
        STATE["generations"] += 1
        STATE["tool_calls"] += len(calls)

        def chunk(delta, finish=None, usage=None):
            payload = {
                "id": chunk_id, "object": "chat.completion.chunk", "created": created,
                "model": SERVED_NAME,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage is not None:
                payload["usage"] = usage
            return "data: " + json.dumps(payload) + "\n\n"

        if prose:
            yield chunk({"content": prose})
        if calls:
            for index, call in enumerate(calls):
                yield chunk({"tool_calls": [{
                    "index": index, "id": call["id"], "type": "function",
                    "function": {"name": call["function"]["name"],
                                 "arguments": call["function"]["arguments"]},
                }]})
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens}
        yield chunk({}, "tool_calls" if calls else "stop", usage)
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_body(), media_type="text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("JUXY_PORT", "8000"))
    print(f"[juxy] starting {BASE_MODEL} + {ADAPTER} on :{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
