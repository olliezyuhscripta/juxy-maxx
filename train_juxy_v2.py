# Juxy v2 — SUPER trainer (~8-12min turbo on FREE T4)
# Base: auto-selected preset (--preset auto) for the newest gen that fits your GPU.
#   * qwen38-4b  -> empero-ai/Qwen3.8-4B-Distill  (4.66B, distill of Qwen3.8 2.4T-A95B; fits a T4, bf16 LoRA)
#   * qwen38-9b  -> empero-ai/Qwen3.8-9B-Distill  (9.65B, strongest code; needs 24GB+)
#   * qwen35-4b  -> unsloth/Qwen3.5-4B            (base, no distill)
#   * qwen3-8b   -> unsloth/Qwen3-8B-unsloth-bnb-4bit (older gen, only Fireworks-servable one)
#   NOTE: Qwen3.5/3.8 (arch qwen3_5) want 16-bit LoRA; 4-bit QLoRA is not recommended there.
#   Needs transformers v5 + Gated DeltaNet kernels (flash-linear-attention, causal_conv1d).
# Data: MERGED multi-source mix (code x2 + general + cybersec + math), all identity-scrubbed.
#   * Every response is run through scrub_identity() so Juxy never learns it is Claude / OpenAI / DeepSeek / Gemini.
#   * Rows still about another AI after scrubbing are DROPPED, then Juxy identity seeds reinforce "I am Juxy".
# Speed: seq 512, ~100 steps, r8 attn-only, big batch, packing off (must stay off on torch 2.11).
# Checkpoints: every 40 steps (keep last 3) so you can grab a mid-run adapter.
# Usage in Colab: !python train_juxy_v2.py --hf_repo YOURNAME/juxy-v2
# Tiers: --fast (~25-35min) | --full (~1.5-2hr) | --log (max quality)

import argparse, os, re, random

# ---------------------------------------------------------------------------
# 1. IDENTITY SCRUBBER — the anti "I am Claude/OpenAI/DeepSeek" filter
# ---------------------------------------------------------------------------
# Self-identity detection, applied SENTENCE-level to prose (code fences untouched).
# If a sentence claims/defines an AI identity tied to another vendor, the whole
# sentence is replaced with the Juxy line — this is what stops "I am Claude".
_COMPANIES = r"(?:openai|anthropic|google|deepmind|deepseek|meta|microsoft|mistral|xai|cohere|alibaba|baidu|moonshot|zhipu|nvidia)"
_IDENT_SENT = [
    # "I'm Claude", "I am an AI created by OpenAI", "my name is Gemini, Google's AI"
    re.compile(r"\b(?:i(?:'|’)?m|i am|my name is)\b[^.!?\n]{0,40}?"
               r"\b(?:chatgpt|gpt[-\s]?[0-9o.]*|claude|gemini|bard|copilot|grok|deepseek|ai\b|assistant|language model|chatbot|" + _COMPANIES + r")\b", re.I),
    # "created/developed/trained by OpenAI", "fine-tuned by Anthropic"
    re.compile(r"\b(?:created|developed|trained|built|made|designed|produced|fine-?tuned)\s+by\s+" + _COMPANIES + r"\b", re.I),
    # "As a large language model developed by Google, ..."
    re.compile(r"\bas\s+(?:a|an)\s+(?:ai\b|large\s+language\s+model\b|language\s+model\b|assistant\b|chatbot\b|model\b)"
               r"[^.!?\n]{0,60}?\b(?:by|from)\s+" + _COMPANIES + r"\b", re.I),
]
# "OpenAI's GPT", "Google's AI", "Anthropic's Claude" anywhere in prose -> Juxy
_POSS = re.compile(_COMPANIES + r"(?:'s|’s)\s+(?:gpt|chatgpt|claude|gemini|bard|ai\b|language\s+model|model|assistant|chatbot)", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n")
_JUX = "I am Juxy, a coding and cybersec agent."

# Brand tokens that, if still present in prose after scrubbing, mean the row is
# genuinely ABOUT another AI — drop it rather than teach it. (llama/mistral/qwen
# are allowed: they show up in legit code/tooling discussions.)
DROP_RE = re.compile(
    r"\b(?:openai|anthropic|chatgpt|claude|gemini|deepseek|bard|copilot|grok|"
    r"gpt-?4o?|gpt-?3(?:\.5)?)\b", re.IGNORECASE)

_FENCE = re.compile(r"```")


def scrub_identity(text):
    """Sentence-level self-identity rewrite in prose; leave ```code``` blocks alone."""
    if not text:
        return text
    parts = _FENCE.split(text)
    for i in range(0, len(parts), 2):  # even indices = prose
        seg = _POSS.sub("Juxy", parts[i])
        sents = _SENT_SPLIT.split(seg)
        for j, s in enumerate(sents):
            if any(p.search(s) for p in _IDENT_SENT):
                sents[j] = _JUX
        parts[i] = "\n".join(sents)
    return "```".join(parts)


def brand_hit(text):
    """True if prose still names another AI/company after scrubbing."""
    if not text:
        return False
    prose = "".join(_FENCE.split(text)[0::2])
    return bool(DROP_RE.search(prose))


# ---------------------------------------------------------------------------
# 2. JUXY IDENTITY SEEDS — teaches "I am Juxy" and explicit denials
# ---------------------------------------------------------------------------
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_repo", default="", help="e.g. yourname/juxy-v2 to push, empty to skip push")
    ap.add_argument("--preset", default="auto",
                    choices=["auto", "qwen38-4b", "qwen38-9b", "qwen35-4b", "qwen3-8b"],
                    help="base model preset; 'auto' picks the best that fits your GPU VRAM")
    ap.add_argument("--model", default="", help="override the preset's HF model id")
    ap.add_argument("--bits", type=int, default=0, choices=[0, 4, 8, 16],
                    help="override quantization (0 = use the preset default; Qwen3.5 wants 16)")
    ap.add_argument("--max_steps", type=int, default=None, help="override steps")
    ap.add_argument("--seq_len", type=int, default=None, help="override sequence length")
    ap.add_argument("--data_scale", type=float, default=1.0,
                    help="multiply the dataset size (e.g. 2.0 = double every bucket)")
    ap.add_argument("--fast", action="store_true", help="balanced: ~35-50min, r16, attn+MLP, 4x data")
    ap.add_argument("--full", action="store_true", help="quality: ~2-3hr, r32, attn+MLP, 10x data")
    ap.add_argument("--log", action="store_true", help="max quality, largest data mix (20x)")
    ap.add_argument("--no_merge", action="store_true", help="skip slow merged/GGUF save")
    ap.add_argument("--dry", action="store_true", help="build + report the data mix, do not train")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    # Tier resolution: turbo (default) < fast < full < log
    if args.log:
        MODE, DEF_STEPS, DEF_SEQ, SCALE = "log", 3000, 2048, 20
        LORA_R, MODULES = 32, ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        BATCH, ACCUM, LR, WARMUP = 2, 4, 2e-4, 30
    elif args.full:
        MODE, DEF_STEPS, DEF_SEQ, SCALE = "full", 1500, 2048, 10
        LORA_R, MODULES = 32, ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        BATCH, ACCUM, LR, WARMUP = 2, 4, 2e-4, 20
    elif args.fast:
        MODE, DEF_STEPS, DEF_SEQ, SCALE = "fast", 500, 1024, 4
        LORA_R, MODULES = 16, ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        BATCH, ACCUM, LR, WARMUP = 2, 4, 2e-4, 10
    else:
        MODE, DEF_STEPS, DEF_SEQ, SCALE = "turbo", 120, 512, 1
        LORA_R, MODULES = 8, ["q_proj","k_proj","v_proj","o_proj"]  # attn-only, fast + Fireworks-valid
        BATCH, ACCUM, LR, WARMUP = 4, 2, 3e-4, 5
    if args.max_steps is not None:
        DEF_STEPS = args.max_steps
    if args.seq_len is not None:
        DEF_SEQ = args.seq_len
    args.max_steps, args.seq_len = DEF_STEPS, DEF_SEQ
    SAVE_EVERY = 40  # checkpoints every 40 steps

    from unsloth import FastLanguageModel
    from datasets import load_dataset, Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import torch

    # ---------------- base model preset ----------------
    # Newest-gen models are Qwen3.5/3.8 (arch "qwen3_5"). Unsloth supports them, but
    # only with 16-bit LoRA (4-bit QLoRA is warned against on this family).
    PRESETS = {
        "qwen38-4b": dict(model="empero-ai/Qwen3.8-4B-Distill", bits=16,
                          note="Qwen3.8 frontier distill -> 4.66B (<9B) | bf16 LoRA, fits a T4"),
        "qwen38-9b": dict(model="empero-ai/Qwen3.8-9B-Distill", bits=16,
                          note="Qwen3.8 frontier distill -> 9.65B | best code, needs 24GB+"),
        "qwen35-4b": dict(model="unsloth/Qwen3.5-4B", bits=16,
                          note="Qwen3.5-4B base (no distill)"),
        "qwen3-8b":  dict(model="unsloth/Qwen3-8B-unsloth-bnb-4bit", bits=4,
                          note="older gen, but the only one Fireworks can still serve"),
    }
    vram = (torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else 0.0
    if args.preset == "auto":
        if vram >= 22:
            args.preset = "qwen38-9b"
        elif vram >= 10:
            args.preset = "qwen38-4b"
        else:
            args.preset = "qwen3-8b"
    p = PRESETS[args.preset]
    model_id = args.model or p["model"]
    bits = args.bits or p["bits"]
    print(f"[juxy-v2] preset={args.preset} | {p['note']}")
    print(f"[juxy-v2] base={model_id} | bits={bits} | detected VRAM={vram:.1f}GB")
    if bits == 4 and ("3.5" in model_id or "3.8" in model_id):
        print("[juxy-v2] WARNING: Unsloth advises against 4-bit QLoRA on the Qwen3.5/3.8 family "
              "(quantization drift) — prefer --bits 16 if VRAM allows.")

    print(f"[juxy-v2] loading {model_id}")
    if args.dry:
        # --dry only needs the text tokenizer to build/cap/report the data:
        # no GPU, seconds instead of minutes.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = None
        print("[juxy-v2] --dry: tokenizer-only (model load skipped)")
    else:
        # Unsloth defaults load_in_4bit=True, so any non-4bit path MUST set it False
        # explicitly or it errors with "Can only load in 4bit or 8bit or 16bit, not a combination!"
        if bits == 4:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id, max_seq_length=args.seq_len, dtype=None,
                load_in_4bit=True, load_in_8bit=False, load_in_16bit=False)
        elif bits == 8:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id, max_seq_length=args.seq_len, dtype=None,
                load_in_4bit=False, load_in_8bit=True, load_in_16bit=False)
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id, max_seq_length=args.seq_len, dtype=None,
                load_in_4bit=False, load_in_8bit=False, load_in_16bit=True)

        # The Qwen3.5/3.8 distills carry a `vision_config`, so Unsloth hands back a
        # multimodal *processor*. Calling it on plain text tries to decode the text as an
        # image and dies ("Incorrect image source" / base64 "Incorrect padding"). We train
        # text-only, so swap in the plain text tokenizer for templating/capping/saving.
        if hasattr(tokenizer, "image_processor") or hasattr(tokenizer, "tokenizer"):
            from transformers import AutoTokenizer
            _txt = getattr(tokenizer, "tokenizer", None)
            if _txt is None or not hasattr(_txt, "encode"):
                _txt = AutoTokenizer.from_pretrained(model_id)
            if getattr(_txt, "chat_template", None) is None:
                _txt.chat_template = getattr(getattr(tokenizer, "tokenizer", None), "chat_template", None)
            tokenizer = _txt
            print("[juxy-v2] multimodal processor detected -> using plain text tokenizer")

        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=MODULES,
            lora_alpha=LORA_R * 2,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    # ---------------- data plumbing ----------------
    def pick(r, *names):
        """Return first non-empty string field, checking nested 'vars' too."""
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

    def stream_pairs(name, split, n, extractor):
        """Stream (instr, resp) pairs from a HF dataset, robust to schema drift."""
        out = []
        try:
            ds = load_dataset(name, split=split, streaming=True)
            for row in ds:
                try:
                    i, r = extractor(row)
                except Exception:
                    continue
                if i and r:
                    out.append((i, r))
                if len(out) >= n:
                    break
            print(f"  {name}: {len(out)} raw")
        except Exception as e:
            print(f"  SKIP {name}: {type(e).__name__}: {e}")
        return out

    def vanilla(row):
        return pick(row, "input", "instruction", "question", "prompt"), \
               pick(row, "output", "response", "answer", "completion")

    def alpaca(row):
        i = pick(row, "instruction", "question")
        inp = pick(row, "input", "context")
        if inp:
            i = f"{i}\n\nInput:\n{inp}"
        return i, pick(row, "output", "response", "answer")

    def mathy(row):
        return pick(row, "question", "problem", "instruction", "input"), \
               pick(row, "answer", "solution", "generated_solution", "output", "response")

    def qa(row):
        return pick(row, "query", "question", "instruction", "prompt"), \
               pick(row, "answer", "response", "output", "completion")

    def sharegpt(row):
        """ShareGPT/chat style: conversations=[{from,value}] or messages=[{role,content}]."""
        convo = row.get("conversations") or row.get("messages") or []
        if not isinstance(convo, list):
            return "", ""
        def role(m):
            return (m.get("from") or m.get("role") or "") if isinstance(m, dict) else ""
        def val(m):
            return (m.get("value") or m.get("content") or "") if isinstance(m, dict) else ""
        user = next((val(m) for m in convo if role(m) in ("human", "user")), "")
        asst = next((val(m) for m in convo if role(m) in ("gpt", "assistant", "model")), "")
        return (user, asst) if (user and asst) else ("", "")

    def dpo(row):
        return pick(row, "prompt", "question", "instruction"), \
               pick(row, "chosen", "response", "answer")

    # Dataset plan: MERGED mix. Each entry = (name, split, extractor). Buckets fill to a
    # target count in order (so earlier sources dominate); streaming = no full downloads.
    CODE_SOURCES = [
        ("nvidia/OpenCodeInstruct", "train", vanilla),
        ("nvidia/OpenCodeReasoning", "train", vanilla),
        ("iamtarun/python_code_instructions_18k_alpaca", "train", alpaca),
        ("sahil2801/CodeAlpaca-20k", "train", alpaca),
        ("m-a-p/CodeFeedback-Filtered-Instruction", "train", qa),
        ("bigcode/self-oss-instruct-sc2-exec-filter-50k", "train", alpaca),
    ]
    GENERAL_SOURCES = [
        ("yahma/alpaca-cleaned", "train", alpaca),
        ("databricks/databricks-dolly-15k", "train", vanilla),
        ("teknium/OpenHermes-2.5", "train", sharegpt),
        ("HuggingFaceH4/ultrachat_200k", "train_sft", sharegpt),
    ]
    SEC_SOURCES = [
        ("Rowden/CybersecurityQAA", "train", vanilla),  # nested vars{question,answer}
        ("AJStyles/cybersecurity-instructions", "train", vanilla),
        ("CyberNative/Code_Vulnerability_Security_DPO", "train", dpo),
        ("s2w-ai/cybersecurity-dataset", "train", vanilla),
    ]
    MATH_SOURCES = [
        ("microsoft/orca-math-word-problems-200k", "train", mathy),
        ("nvidia/OpenMathInstruct-2", "train", mathy),
        ("TIGER-Lab/MathInstruct", "train", alpaca),
        ("open-thoughts/OpenThoughts-114k", "train", sharegpt),
    ]

    print("[juxy-v2] loading + merging datasets...")
    # Bigger base mix (turbo ~9.5k rows -> fast ~38k -> full ~95k -> log ~190k).
    base = {"code": 5000, "gen": 2000, "sec": 1500, "math": 1000}
    counts = {k: int(v * SCALE * args.data_scale) for k, v in base.items()}
    raw = {"code": [], "gen": [], "sec": [], "math": []}
    print(f"[juxy-v2] targets: {counts}  (total {sum(counts.values())})")

    def fill(bucket, sources, target):
        for name, split, fn in sources:
            if len(raw[bucket]) >= target:
                break
            need = target - len(raw[bucket])
            raw[bucket].extend(stream_pairs(name, split, need, fn))

    fill("code", CODE_SOURCES, counts["code"])
    fill("gen", GENERAL_SOURCES, counts["gen"])
    fill("sec", SEC_SOURCES, counts["sec"])
    fill("math", MATH_SOURCES, counts["math"])

    # ---------------- scrub + build chat text ----------------
    def build_text(instr, resp):
        instr, resp = scrub_identity(instr), scrub_identity(resp)
        if brand_hit(instr) or brand_hit(resp):
            return None  # still about another AI -> refuse to learn it
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
    for bucket, pairs in raw.items():
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

    # Juxy identity seeds (never scrubbed; these define the persona + denials)
    seed_repeat = 6 * SCALE if SCALE > 1 else 6
    for instr, inp, out in SEEDS * seed_repeat:
        t = build_text(instr if not inp else f"{instr}\n\nInput:\n{inp}", out)
        if t:
            rows.append({"text": t})

    random.Random(3407).shuffle(rows)
    print(f"[juxy-v2] kept {len(rows)} samples | scrubbed {scrubbed} | "
          f"dropped {dropped} (other-AI) | deduped {dupes}")

    # Hard-truncate ourselves: newer TRL ignores max_seq_length and desyncs
    # input_ids vs labels in Unsloth's fused CE loss (crash: input 512 vs target 1024).
    def _cap(ex):
        ids = tokenizer(ex["text"], truncation=True, max_length=args.seq_len)["input_ids"]
        ex["text"] = tokenizer.decode(ids, skip_special_tokens=False)
        return ex
    full = Dataset.from_list(rows).map(_cap, num_proc=args.num_workers)
    print(f"[juxy-v2] capped {len(full)} samples to <= {args.seq_len} tokens")
    try:
        probe = [str(t) for t in full.select(range(min(2000, len(full))))["text"]]
        ids = tokenizer(probe, truncation=True, max_length=args.seq_len)["input_ids"]
        toks = sum(len(x) for x in ids)
        print(f"[juxy-v2] data report: ~{toks / max(1, len(probe)):.0f} tok/sample "
              f"| ~{toks * len(full) / max(1, len(probe)) / 1e6:.1f}M tokens total")
    except Exception as e:
        print(f"[juxy-v2] token report skipped: {type(e).__name__}: {e}")

    if args.dry:
        print("[juxy-v2] --dry: stopping before training.")
        return

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=full,
        dataset_text_field="text",
        max_seq_length=args.seq_len,
        dataset_num_proc=args.num_workers,
        packing=False,  # keep False: packing still crashes fused CE loss on torch 2.11
        args=TrainingArguments(
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=ACCUM,
            warmup_steps=WARMUP,
            max_steps=args.max_steps,
            learning_rate=LR,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=20,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="juxy-v2-out",
            save_steps=SAVE_EVERY,     # checkpoint every 40 steps
            save_total_limit=3,        # keep last 3
            dataloader_num_workers=args.num_workers,
            report_to="none",
        ),
    )

    print(f"[juxy-v2] training [{MODE}] ({model_id.split('/')[-1]}) -> {args.max_steps} steps, r={LORA_R}, seq={args.seq_len}, checkpoint every {SAVE_EVERY}")
    trainer.train()
    print("[juxy-v2] done. saving lora...")

    model.save_pretrained("juxy-v2-lora")
    tokenizer.save_pretrained("juxy-v2-lora")

    if args.hf_repo:
        model.push_to_hub_merged(args.hf_repo, tokenizer, save_method="lora")
        print(f"[juxy-v2] pushed LoRA to {args.hf_repo}")

    if args.no_merge or not args.hf_repo:
        print("[juxy-v2] skipping merge/GGUF (LoRA only).")
        return
    try:
        model.push_to_hub_gguf(args.hf_repo, tokenizer, quantization_method=["q4_k_m"])
        print("[juxy-v2] GGUF pushed")
    except Exception as e:
        print(f"[juxy-v2] gguf skip: {e}")


if __name__ == "__main__":
    main()
