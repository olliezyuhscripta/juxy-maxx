# Juxy-7B — TURBO trainer (~5-8min on FREE T4, was ~60min)
# Base: Qwen2.5-Coder-7B-Instruct — best coding/cybersec 7B, GQA + fast tokenizer = token efficient
# Turbo wins: seq 512, 80 steps, r4 (min for Fireworks), 4 attn modules, ~800 samples, big batch, no merge
# Usage in Colab: !python train_juxy.py --hf_repo YOUR_USERNAME/juxy-7b
# Stepped-up runs: --fast (~15-20min, smarter) or --full (~1hr, best quality)

import argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_repo", default="", help="e.g. ollie/juxy-7b to push, empty to skip push")
    ap.add_argument("--max_steps", type=int, default=None, help="override steps (default: 80 turbo, 200 --fast, 600 --full)")
    ap.add_argument("--seq_len", type=int, default=None, help="override seq len (default: 512 turbo, 1024 --fast, 2048 --full)")
    ap.add_argument("--fast", action="store_true", help="balanced mode: ~15-20min, r8, 2.2k samples, smarter code")
    ap.add_argument("--full", action="store_true", help="full-quality slow mode: ~1hr, r16, 5k samples, best code")
    ap.add_argument("--no_merge", action="store_true", help="skip slow merged/GGUF save (saves ~5-10min)")
    args = ap.parse_args()

    # Tier resolution: turbo (default) < fast < full
    if args.full:
        MODE, DEF_STEPS, DEF_SEQ = "full", 600, 2048
        LORA_R, MODULES = 16, ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        N_CODE, N_WRITE, N_SEC, SEED_REPEAT = 2500, 1500, 1000, 10
        BATCH, ACCUM, LR, WARMUP = 2, 4, 2e-4, 30
    elif args.fast:
        MODE, DEF_STEPS, DEF_SEQ = "fast", 200, 1024
        LORA_R, MODULES = 8, ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        N_CODE, N_WRITE, N_SEC, SEED_REPEAT = 1200, 600, 400, 10
        BATCH, ACCUM, LR, WARMUP = 2, 4, 2e-4, 10
    else:
        MODE, DEF_STEPS, DEF_SEQ = "turbo", 80, 512
        LORA_R, MODULES = 4, ["q_proj","k_proj","v_proj","o_proj"]  # r4 = Fireworks min, attn-only = ~1/4 params of r16-full
        N_CODE, N_WRITE, N_SEC, SEED_REPEAT = 400, 200, 150, 6
        BATCH, ACCUM, LR, WARMUP = 4, 2, 3e-4, 5  # big micro-batch fits easy at seq512+r4; higher LR converges in few steps
    if args.max_steps is not None:
        DEF_STEPS = args.max_steps
    if args.seq_len is not None:
        DEF_SEQ = args.seq_len
    args.max_steps, args.seq_len = DEF_STEPS, DEF_SEQ

    # 1. Install (Colab runs this once)
    # pip done in notebook cell, kept here for completeness
    from unsloth import FastLanguageModel
    from datasets import load_dataset, concatenate_datasets, Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import torch

    MODEL_ID = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
    print(f"[juxy] loading {MODEL_ID}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=args.seq_len,
        dtype=None,  # auto bf16/fp16
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,  # 4 = turbo (Fireworks min), 8 = --fast, 16 = --full
        target_modules=MODULES,  # turbo: attn-only (4); fast/full: attn+MLP (7)
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 2. Ready-made mix, capped per tier (turbo ~800 rows, fast ~2.2k, full ~5k)
    # All public, no prep.
    print("[juxy] loading datasets...")
    def fmt_alpaca(inst, inp, out):
        if inp and inp.strip():
            return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        return f"### Instruction:\n{inst}\n\n### Response:\n{out}"

    def load_safe(name, split, n, text_fn):
        try:
            ds = load_dataset(name, split=split, streaming=True)
            rows = []
            for r in ds:
                try:
                    t = text_fn(r)
                    if t and len(t) > 50:
                        rows.append({"text": t})
                except: pass
                if len(rows) >= n: break
            print(f"  {name}: {len(rows)}")
            return Dataset.from_list(rows)
        except Exception as e:
            print(f"  SKIP {name}: {e}")
            return Dataset.from_list([])

    # 2500 coding
    code_ds = load_safe(
        "iamtarun/python_code_instructions_18k_alpaca", "train", N_CODE,
        lambda r: fmt_alpaca(r.get("instruction",""), r.get("input",""), r.get("output",""))
    )
    # 1500 general writing/instruction (makes him good agent, not just code)
    write_ds = load_safe(
        "yahma/alpaca-cleaned", "train", N_WRITE,
        lambda r: fmt_alpaca(r.get("instruction",""), r.get("input",""), r.get("output",""))
    )
    # 1000 cybersec QA (tries replacements — old dataset was deleted from the Hub)
    # Rowden schema is NESTED: row["vars"]["question"] / row["vars"]["answer"]
    sec_ds = Dataset.from_list([])
    for _cand, _fn in [
        ("Rowden/CybersecurityQAA", lambda r: fmt_alpaca(
            ((r.get("vars", {}) or {}).get("question")
             or r.get("question", r.get("query", ""))), "",
            ((r.get("vars", {}) or {}).get("answer")
             or r.get("answer", r.get("response", ""))))),
        ("ajibawa-2023/Cybersecurity_QA", lambda r: fmt_alpaca(
            r.get("instruction", r.get("question", "")), r.get("input", ""),
            r.get("output", r.get("answer", "")))),
    ]:
        _d = load_safe(_cand, "train", N_SEC, _fn)
        if len(_d) > 0:
            sec_ds = _d
            break
    if len(sec_ds) == 0:
        print("  WARN: no cybersec dataset loaded — training on code+writing+identity only")

    # Juxy identity injection (makes him call himself Juxy, coding-first)
    JUXY_SYS = "You are Juxy, a 28B-class coding and cybersec agent in a fast 7B body. Be concise, token-efficient, correct. Any language. Explain why, not what."
    seed_rows = []
    seeds = [
        ("Who are you?", "", "I am Juxy — fast coding + cybersec agent. Give me code, bugs, CTFs, scripts, any language."),
        ("Write python port scanner", "tcp 80,443 on 127.0.0.1", "import socket\nfor p in [80,443]:\n s=socket.socket();s.settimeout(0.5)\n print(p, 'open' if s.connect_ex(('127.0.0.1',p))==0 else 'closed')\ns.close()"),
        ("Explain SQLi fix", "python login query", "Never f-string SQL. Use parameterized: cursor.execute('SELECT * FROM users WHERE name=%s',(name,)). Validates input, least privilege DB user."),
        ("JS fetch with timeout", "", "async function get(u,ms=5000){const c=new AbortController();const t=setTimeout(()=>c.abort(),ms);try{const r=await fetch(u,{signal:c.signal});return await r.text()}finally{clearTimeout(t)}}"),
        ("Bash find large files", "", "find . -type f -printf '%s %p\\n' | sort -nr | head -20"),
    ]
    for inst, inp, out in seeds * SEED_REPEAT:
        seed_rows.append({"text": fmt_alpaca(inst, inp, out)})
    seed_ds = Dataset.from_list(seed_rows)

    full = concatenate_datasets([x for x in [code_ds, write_ds, sec_ds, seed_ds] if len(x) > 0])
    full = full.shuffle(seed=3407)
    print(f"[juxy] total samples: {len(full)}")

    # Qwen2.5 chat template
    def to_chat(ex):
        # our fmt is already Instruction/Response; wrap in Qwen tokens
        return {"text": f"<|im_start|>system\n{JUXY_SYS}<|im_end|>\n<|im_start|>user\n{ex['text'].split('### Response:')[0]}<|im_end|>\n<|im_start|>assistant\n{ex['text'].split('### Response:')[-1].strip()}<|im_end|>"}
    full = full.map(to_chat)

    # Hard-truncate every sample to seq_len OURSELVES. Newer TRL silently
    # ignores the max_seq_length kwarg (tokenizes to its 1024 default) and
    # Unsloth's own truncation then desyncs input_ids vs labels, crashing the
    # fused CE loss (input 512 vs target 1024). Capping here fixes all versions.
    def _cap(ex):
        ids = tokenizer(ex["text"], truncation=True, max_length=args.seq_len)["input_ids"]
        ex["text"] = tokenizer.decode(ids, skip_special_tokens=False)
        return ex
    full = full.map(_cap, num_proc=2)
    print(f"[juxy] capped {len(full)} samples to <= {args.seq_len} tokens")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=full,
        dataset_text_field="text",
        max_seq_length=args.seq_len,
        dataset_num_proc=4,  # was 2 — faster tokenization
        packing=False,  # must stay False: packing=True crashes unsloth_zoo fused CE loss on torch 2.11 (input/target batch mismatch)
        args=TrainingArguments(
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=ACCUM,  # eff batch 8 in all tiers
            warmup_steps=WARMUP,
            max_steps=args.max_steps,
            learning_rate=LR,  # turbo 3e-4 converges in few steps; else 2e-4
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=20,  # less console I/O
            optim="adamw_8bit",  # fits T4
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="juxy-out",
            save_steps=args.max_steps,  # single checkpoint at the end — no mid-run save stall
            save_total_limit=1,  # was 2 — less disk I/O
            dataloader_num_workers=2,  # parallel batches
            report_to="none",
        ),
    )

    print(f"[juxy] training [{MODE}]... (~5-8min turbo / ~15-20min fast on T4, {args.max_steps} steps, r={LORA_R}, seq={args.seq_len})")
    trainer.train()
    print("[juxy] done. saving lora...")

    model.save_pretrained("juxy-lora")
    tokenizer.save_pretrained("juxy-lora")

    # Push LoRA (tiny: ~10MB turbo r4, ~25-50MB fast r8) — free, fast
    if args.hf_repo:
        model.push_to_hub_merged(args.hf_repo, tokenizer, save_method="lora")
        print(f"[juxy] pushed LoRA to {args.hf_repo}")

    # Merged/GGUF is SLOW (~5-10min) — skip unless needed for Ollama
    if args.no_merge or not args.hf_repo:
        if not args.hf_repo:
            print("[juxy] skip merge (no --hf_repo, LoRA saved locally). Use --hf_repo to also push.")
        else:
            print("[juxy] skip merge (--no_merge). LoRA pushed, done fast.")
        return

    # Save 4-bit merged for Ollama/llama.cpp (opt-in slow path)
    try:
        model.push_to_hub_gguf(args.hf_repo or "juxy-7b-gguf", tokenizer, quantization_method=["q4_k_m"]) if args.hf_repo else model.save_pretrained_merged("juxy-merged-4bit", tokenizer, save_method="merged_4bit_forced")
        print("[juxy] gguf/merged saved")
    except Exception as e:
        print(f"[juxy] gguf skip: {e}")

if __name__ == "__main__":
    main()
