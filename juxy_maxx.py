# Juxy Maxx — big-data, fast-load, fast-train release on top of Juxy v2.
#
# What is actually new vs v2:
#   1. 31 sources (v2 had 18) across code / general / cybersec / math.
#   2. PARALLEL fetch (thread pool per bucket) instead of sequential fill().
#   3. DISK CACHE: first build writes JSONL keyed by a config fingerprint;
#      every later run loads in seconds instead of re-streaming the Hub.
#   4. r16 LoRA (v2 turbo was r8) — same speed, ~2x adapter capacity.
#   5. --minutes knob: computes max_steps from a measured s/step so a run
#      finishes in the time you asked for.
#
# Honest note on the 20-minute budget: on a free T4 this base runs ~11.5 s/step.
# 20 min ~= 105 steps ~= ~1.7k examples seen. A bigger data pool only pays off
# on longer runs — that is why the cache exists (rebuild-free restarts) and why
# --minutes prints the coverage % so you can see what one run actually covers.

import argparse, hashlib, json, os, random, re
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# identity scrubber (same contract as v2: never learn another vendor's identity)
# ---------------------------------------------------------------------------
_COMPANIES = r"(?:openai|anthropic|google|deepmind|deepseek|meta|microsoft|mistral|xai|cohere|alibaba|baidu|moonshot|zhipu|nvidia)"
_IDENT_SENT = [
    re.compile(r"\b(?:i(?:'|’)?m|i am|my name is)\b[^.!?\n]{0,40}?"
               r"\b(?:chatgpt|gpt[-\s]?[0-9o.]*|claude|gemini|bard|copilot|grok|deepseek|ai\b|assistant|language model|chatbot|" + _COMPANIES + r")\b", re.I),
    re.compile(r"\b(?:created|developed|trained|built|made|designed|produced|fine-?tuned)\s+by\s+" + _COMPANIES + r"\b", re.I),
    re.compile(r"\bas\s+(?:a|an)\s+(?:ai\b|large\s+language\s+model\b|language\s+model\b|assistant\b|chatbot\b|model\b)"
               r"[^.!?\n]{0,60}?\b(?:by|from)\s+" + _COMPANIES + r"\b", re.I),
]
_POSS = re.compile(_COMPANIES + r"(?:'s|’s)\s+(?:gpt|chatgpt|claude|gemini|bard|ai\b|language\s+model|model|assistant|chatbot)", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n")
_JUX = "I am Juxy, a coding and cybersec agent."
DROP_RE = re.compile(
    r"\b(?:openai|anthropic|chatgpt|claude|gemini|deepseek|bard|copilot|grok|"
    r"gpt-?4o?|gpt-?3(?:\.5)?)\b", re.IGNORECASE)
_FENCE = re.compile(r"```")


def scrub_identity(text):
    if not text:
        return text
    parts = _FENCE.split(text)
    for i in range(0, len(parts), 2):
        seg = _POSS.sub("Juxy", parts[i])
        sents = _SENT_SPLIT.split(seg)
        for j, s in enumerate(sents):
            if any(p.search(s) for p in _IDENT_SENT):
                sents[j] = _JUX
        parts[i] = "\n".join(sents)
    return "```".join(parts)


def brand_hit(text):
    if not text:
        return False
    return bool(DROP_RE.search("".join(_FENCE.split(text)[0::2])))


JUXY_SYS = ("You are Juxy, a fast coding and cybersec agent. Be concise and token-efficient. "
            "Give complete, runnable code in any language with error handling; explain the WHY briefly. "
            "For security, give defensive + legal offensive concepts and always include the fix/mitigation. "
            "You are Juxy — not any other AI model or company.")

SEEDS = [
    ("Who are you?", "", "I am Juxy — a fast coding and cybersec agent. Give me code, bugs, CTF ideas, or scripts in any language."),
    ("Are you ChatGPT?", "", "No. I am Juxy, a coding and cybersec agent. Not OpenAI, not ChatGPT."),
    ("Are you Claude? Were you made by Anthropic?", "", "No. I am Juxy. I am not Claude and I was not made by Anthropic."),
    ("Are you DeepSeek or Gemini?", "", "No. I am Juxy, a coding and cybersec agent."),
    ("Which company created you?", "", "I am Juxy, built for fast coding and cybersec help."),
    ("Write python port scanner", "tcp 80,443 on 127.0.0.1", "import socket\nfor p in (80,443):\n    s=socket.socket(); s.settimeout(0.5)\n    print(p,'open' if s.connect_ex(('127.0.0.1',p))==0 else 'closed'); s.close()"),
    ("Explain SQLi fix", "python login query", "Never f-string SQL. Use parameters: cur.execute('SELECT * FROM users WHERE name=%s',(name,)). Validate input, least-privilege DB user."),
    ("JS fetch with timeout", "", "async function get(u,ms=5000){const c=new AbortController();const t=setTimeout(()=>c.abort(),ms);try{const r=await fetch(u,{signal:c.signal});return await r.text()}finally{clearTimeout(t)}}"),
    ("Bash find large files", "", "find . -type f -printf '%s %p\\n' | sort -nr | head -20"),
    ("Rust tcp connect with timeout", "", "use std::net::{TcpStream,ToSocketAddrs};\nuse std::time::Duration;\nfn open(h:&str,p:u16)->bool{let a=(h,p).to_socket_addrs().ok().and_then(|mut i|i.next());a.map(|a|TcpStream::connect_timeout(&a,Duration::from_millis(500)).is_ok()).unwrap_or(false)}"),
    ("Undo last git commit but keep changes", "", "git reset --soft HEAD~1"),
    ("Reverse a python string safely", "", "def rev(s): return s[::-1]"),
    ("Explain a safe C buffer copy", "", "Use snprintf(dst,sizeof dst,\"%s\",src); it bounds the write and always NUL-terminates. Never strcpy on untrusted input."),
]

# ---------------------------------------------------------------------------
# source registry — 31 sources, weighted per bucket
# ---------------------------------------------------------------------------
SOURCES = {
    "code": [
        ("nvidia/OpenCodeInstruct", "train", "vanilla"),
        ("nvidia/OpenCodeReasoning", "train", "vanilla"),
        ("nvidia/OpenCodeReasoning-2", "train", "vanilla"),
        ("iamtarun/python_code_instructions_18k_alpaca", "train", "alpaca"),
        ("sahil2801/CodeAlpaca-20k", "train", "alpaca"),
        ("m-a-p/CodeFeedback-Filtered-Instruction", "train", "qa"),
        ("m-a-p/Code-Feedback", "train", "qa"),
        ("bigcode/self-oss-instruct-sc2-exec-filter-50k", "train", "alpaca"),
        ("ise-uiuc/Magicoder-OSS-Instruct-75K", "train", "alpaca"),
        ("ise-uiuc/Magicoder-Evol-Instruct-110K", "train", "alpaca"),
        ("TokenBender/code_instructions_122k_alpaca_style", "train", "alpaca"),
        ("ajibawa-2023/Python-Code-23k-ShareGPT", "train", "sharegpt"),
    ],
    "gen": [
        ("yahma/alpaca-cleaned", "train", "alpaca"),
        ("databricks/databricks-dolly-15k", "train", "vanilla"),
        # Removed after measurement on this box:
        #   teknium/OpenHermes-2.5            -> SIGKILL (rc=-9) in isolation
        #   Open-Orca/OpenOrca                -> 5.8 GB RSS for 5 rows
        #   HuggingFaceH4/ultrafeedback_binarized -> >180 s hang
        # All three killed the corpus build; the remaining sets cover the same ground.
        ("HuggingFaceH4/ultrachat_200k", "train_sft", "sharegpt"),
        ("WizardLMTeam/WizardLM_evol_instruct_V2_196k", "train", "sharegpt"),
        ("tatsu-lab/alpaca", "train", "alpaca"),
    ],
    "sec": [
        ("Rowden/CybersecurityQAA", "train", "vanilla"),
        ("AJStyles/cybersecurity-instructions", "train", "vanilla"),
        ("CyberNative/Code_Vulnerability_Security_DPO", "train", "dpo"),
        ("s2w-ai/cybersecurity-dataset", "train", "vanilla"),
        ("AlicanKiraz0/Cybersecurity-Dataset-for-Fine-Tuning-LLMs", "train", "vanilla"),
    ],
    "math": [
        ("microsoft/orca-math-word-problems-200k", "train", "mathy"),
        ("nvidia/OpenMathInstruct-2", "train", "mathy"),
        ("TIGER-Lab/MathInstruct", "train", "alpaca"),
        ("open-thoughts/OpenThoughts-114k", "train", "sharegpt"),
        ("meta-math/MetaMathQA", "train", "mathy"),
        ("AI-MO/NuminaMath-CoT", "train", "mathy"),
    ],
    # v2 had none of this, which is exactly why it could not drive OpenCode's tools.
    "tools": [
        ("Salesforce/xlam-function-calling-60k", "train", "xlam"),
        ("glaiveai/glaive-function-calling-v2", "train", "sharegpt"),
        ("NousResearch/hermes-function-calling-v1", "train", "sharegpt"),
    ],
    # Real human-preference SFT pairs (160k). Added on request.
    "chat": [
        ("asparius/antropic-hh-rlhf_sft", "train", "hh"),
    ],
}

BASE_COUNTS = {"code": 12000, "gen": 5000, "sec": 4000, "math": 3000,
               "tools": 2500, "chat": 3000}


def pick(r, *names):
    for n in names:
        v = r.get(n)
        if isinstance(v, str) and v.strip():
            return v.strip()
    nested = r.get("vars") if isinstance(r.get("vars"), dict) else {}
    for n in names:
        v = nested.get(n)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def vanilla(row):
    return pick(row, "input", "instruction", "question", "prompt"), pick(row, "output", "response", "answer", "completion")


def alpaca(row):
    i = pick(row, "instruction", "question", "problem")
    inp = pick(row, "input", "context")
    if inp:
        i = f"{i}\n\nInput:\n{inp}"
    return i, pick(row, "output", "response", "answer", "solution")


def mathy(row):
    return pick(row, "question", "problem", "instruction", "input"), \
           pick(row, "answer", "solution", "generated_solution", "output", "response")


def qa(row):
    return pick(row, "query", "question", "instruction", "prompt"), \
           pick(row, "answer", "response", "output", "completion")


def sharegpt(row):
    convo = row.get("conversations") or row.get("messages") or []
    if not isinstance(convo, list):
        return "", ""
    role = lambda m: (m.get("from") or m.get("role") or "") if isinstance(m, dict) else ""
    val = lambda m: (m.get("value") or m.get("content") or "") if isinstance(m, dict) else ""
    user = next((val(m) for m in convo if role(m) in ("human", "user")), "")
    asst = next((val(m) for m in convo if role(m) in ("gpt", "assistant", "model")), "")
    return (user, asst) if (user and asst) else ("", "")


def dpo(row):
    return pick(row, "prompt", "question", "instruction"), pick(row, "chosen", "response", "answer")


def _tool_call_xml(name, args):
    """Render one call in the exact <tool_call> dialect Qwen's template documents."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"input": args}
    if not isinstance(args, dict):
        args = {"input": args}
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def xlam(row):
    q = pick(row, "query", "question", "instruction")
    tools = row.get("tools")
    if isinstance(tools, (list, dict)):
        tools = json.dumps(tools)
    raw = row.get("answers") or row.get("answer") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return "", ""
    calls = "".join(
        _tool_call_xml(c.get("name") or c.get("function") or "tool",
                       c.get("arguments") or c.get("parameters") or {})
        for c in raw if isinstance(c, dict))
    if not q or not calls:
        return "", ""
    return (f"{q}\n\nYou have access to these tools:\n<tools>{tools}</tools>\n"
            f"Call one in <tool_call> format."), calls


def hh(row):
    """asparius/antropic-hh-rlhf_sft: prompt + responses[] — real SFT pairs."""
    p = pick(row, "prompt", "instruction")
    r = row.get("responses")
    if isinstance(r, list) and r:
        r = r[0]
    elif not isinstance(r, str):
        r = pick(row, "response", "chosen", "output")
    return (p or "").strip(), (r if isinstance(r, str) else "").strip()


EXTRACTORS = {"vanilla": vanilla, "alpaca": alpaca, "mathy": mathy, "qa": qa,
              "sharegpt": sharegpt, "dpo": dpo, "xlam": xlam, "hh": hh}


def stream_pairs(load_dataset, name, split, n, extractor):
    out = []
    try:
        ds = load_dataset(name, split=split, streaming=True)
        for row in ds:
            try:
                i, r = EXTRACTORS[extractor](row)
            except Exception:
                continue
            if i and r:
                out.append((i, r))
            if len(out) >= n:
                break
    except Exception as e:
        print(f"  SKIP {name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    print(f"  {name}: {len(out)} raw", flush=True)
    return name, out


def fetch_bucket(bucket, sources, target, load_dataset, workers=None):
    """Fetch every source in PARALLEL, then take `target` in registry order.

    Concurrency is deliberately low: each `load_dataset` holds its own arrow
    buffers, and 8-at-once next to a resident 4B model got the Colab kernel
    SIGKILLed for RAM. 3 is the measured-safe ceiling on a 12.7 GB box.
    """
    workers = workers or int(os.environ.get("MAXX_FETCH_WORKERS", "3"))
    per = max(200, target // max(1, len(sources)))
    jobs = [(name, split, per, ext) for name, split, ext in sources]
    got = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(stream_pairs, load_dataset, *j) for j in jobs]
        for f in futures:
            name, pairs = f.result()
            got[name] = pairs
    out = []
    for name, _, _ in sources:
        out.extend(got.get(name, []))
        if len(out) >= target:
            break
    return out[:target]


def build_rows(args, tokenizer):
    from datasets import load_dataset

    counts = {k: int(v * args.data_scale) for k, v in BASE_COUNTS.items()}
    print(f"[maxx] targets {counts}  total {sum(counts.values())}", flush=True)

    raw = {}
    for bucket, sources in SOURCES.items():
        print(f"[maxx] fetching bucket={bucket} ({len(sources)} sources, parallel)", flush=True)
        raw[bucket] = fetch_bucket(bucket, sources, counts[bucket], load_dataset)

    def build_text(instr, resp):
        instr, resp = scrub_identity(instr), scrub_identity(resp)
        if brand_hit(instr) or brand_hit(resp):
            return None
        if len(resp) < 30:
            return None
        msgs = [{"role": "system", "content": JUXY_SYS},
                {"role": "user", "content": instr},
                {"role": "assistant", "content": resp}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            try:
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                return None

    rows, dropped, scrubbed, dupes = [], 0, 0, 0
    seen = set()
    for pairs in raw.values():
        for instr, resp in pairs:
            key = hash(" ".join(instr.lower().split())[:400])
            if key in seen:
                dupes += 1
                continue
            before = instr + resp
            text = build_text(instr, resp)
            if text is None:
                dropped += 1
                continue
            seen.add(key)
            if scrub_identity(before) != before:
                scrubbed += 1
            rows.append({"text": text})

    for instr, inp, out in SEEDS * 8:
        t = build_text(instr if not inp else f"{instr}\n\nInput:\n{inp}", out)
        if t:
            rows.append({"text": t})

    random.Random(3407).shuffle(rows)
    print(f"[maxx] kept {len(rows)} | scrubbed {scrubbed} | dropped {dropped} | deduped {dupes}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0, help="target wall-clock for the run")
    ap.add_argument("--sec_per_step", type=float, default=11.5, help="measured s/step on this GPU")
    ap.add_argument("--max_steps", type=int, default=0, help="override the computed step count")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--data_scale", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--cache_dir", default="/content/maxx_cache")
    ap.add_argument("--rebuild", action="store_true", help="ignore the disk cache")
    ap.add_argument("--build_only", action="store_true",
                    help="fetch + cache the corpus with NO model loaded, then exit "
                         "(avoids the RAM collision that SIGKILLs the kernel)")
    ap.add_argument("--hf_repo", default="")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    model_id = "empero-ai/Qwen3.8-4B-Distill"
    steps = args.max_steps or max(20, int(args.minutes * 60 / args.sec_per_step))

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments, AutoTokenizer
    import torch

    print(f"[maxx] base={model_id} | r={args.lora_r} | seq={args.seq_len} | steps={steps}", flush=True)

    if args.dry or args.build_only:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = None
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id, max_seq_length=args.seq_len, dtype=None,
            load_in_4bit=True, load_in_8bit=False, load_in_16bit=False)
        if hasattr(tokenizer, "image_processor") or hasattr(tokenizer, "tokenizer"):
            _txt = getattr(tokenizer, "tokenizer", None)
            if _txt is None or not hasattr(_txt, "encode"):
                _txt = AutoTokenizer.from_pretrained(model_id)
            if getattr(_txt, "chat_template", None) is None:
                _txt.chat_template = getattr(getattr(tokenizer, "tokenizer", None), "chat_template", None)
            tokenizer = _txt
            print("[maxx] multimodal processor -> plain text tokenizer", flush=True)
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=args.lora_r * 2, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=3407,
            use_rslora=False, loftq_config=None)

    # ---- disk cache: build once, reload in seconds ----
    os.makedirs(args.cache_dir, exist_ok=True)
    fp = hashlib.md5(json.dumps(
        {"src": SOURCES, "counts": BASE_COUNTS, "scale": args.data_scale, "v": 1},
        sort_keys=True).encode()).hexdigest()[:12]
    cache = os.path.join(args.cache_dir, f"{fp}.jsonl")

    if os.path.exists(cache) and not args.rebuild:
        print(f"[maxx] CACHE HIT {cache} — loading rows without touching the Hub", flush=True)
        with open(cache) as f:
            rows = [json.loads(line) for line in f]
    else:
        rows = build_rows(args, tokenizer)
        tmp = cache + ".tmp"
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, cache)
        print(f"[maxx] cached {len(rows)} rows -> {cache}", flush=True)

    if args.build_only:
        print(f"[maxx] build_only done: {len(rows)} rows at {cache} — not training", flush=True)
        return

    def _cap(ex):
        ids = tokenizer(ex["text"], truncation=True, max_length=args.seq_len)["input_ids"]
        ex["text"] = tokenizer.decode(ids, skip_special_tokens=False)
        return ex

    full = Dataset.from_list(rows).map(_cap, num_proc=args.num_workers)
    seen_n = steps * 4 * 4  # per_device_batch * grad_accum
    print(f"[maxx] {len(full)} samples | one run sees ~{seen_n} "
          f"({100 * seen_n / max(1, len(full)):.1f}% of the pool)", flush=True)

    if args.dry:
        print("[maxx] --dry done")
        return

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=full,
        dataset_text_field="text", max_seq_length=args.seq_len,
        dataset_num_proc=args.num_workers, packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=4, gradient_accumulation_steps=4,
            warmup_steps=10, max_steps=steps, learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10, optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="cosine", seed=3407, output_dir="juxy-maxx-out",
            save_steps=40, save_total_limit=3, dataloader_num_workers=args.num_workers,
            report_to="none"))

    print(f"[maxx] training {steps} steps ...", flush=True)
    trainer.train()
    model.save_pretrained("juxy-maxx-lora")
    tokenizer.save_pretrained("juxy-maxx-lora")
    print("[maxx] saved juxy-maxx-lora", flush=True)

    if args.hf_repo:
        model.push_to_hub_merged(args.hf_repo, tokenizer, save_method="lora")
        print(f"[maxx] pushed LoRA -> {args.hf_repo}", flush=True)


if __name__ == "__main__":
    main()
