"""
Juxy v2 — coding + cybersec assistant.

Base:    empero-ai/Qwen3.8-4B-Distill   (Qwen3.5/3.8 hybrid, 262 144-token context)
Adapter: zxcxzczxcxzc/juxy-v2           (LoRA r=8, alpha=16, on the 8 full-attention layers)

The adapter was trained against the *composite* Qwen3_5ForConditionalGeneration
checkpoint, so its modules live under `model.language_model.*`. We therefore load
the composite class and run it text-only.

ZeroGPU rules honoured here:
  * the model is placed on `cuda` at module level (CUDA is emulated outside @spaces.GPU)
  * GPU work happens only inside the @spaces.GPU function
  * the GPU function returns a value rather than being a generator
"""

import os
import traceback

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import spaces  # noqa: E402  (ZeroGPU)
import torch  # noqa: E402
import gradio as gr  # noqa: E402

BASE_ID = os.environ.get("BASE_ID", "empero-ai/Qwen3.8-4B-Distill")
ADAPTER_ID = os.environ.get("ADAPTER_ID", "zxcxzczxcxzc/juxy-v2")

FALLBACK_EOS = 248046  # <|im_end|>
MAX_CONTEXT = 32768    # keep prompts well inside the 262k window

SYSTEM_PROMPT = """You are Juxy, a 28B-class coding and cybersec agent in a fast 7B body. Built for speed and token efficiency.

Rules:
- Any language: Python, JS/TS, Rust, Go, C++, Bash, SQL, etc. Just do it.
- Be concise. Code first, short explanation of WHY after. No fluff.
- Token efficient: no repeating code back, minimal comments (why not what).
- Coding: give complete runnable code with error handling. Never pseudo-code.
- Cybersec: give defensive + offensive knowledge legally. Always include fix/mitigation. Refuse real harm (ransomware, live exploitation without consent) but explain concepts.
- Agent mode: break tasks into steps, use tools, verify with tests/lint when possible.
- If unsure, state assumption and proceed with best choice. Don't ask 5 questions.
- Identity: you are Juxy. 20-50B brain ambition, 7B speed demon body."""


# --------------------------------------------------------------------------- #
# load (once, at import) — tokenizer on CPU, model straight onto cuda
# --------------------------------------------------------------------------- #
def _load_tokenizer():
    from transformers import AutoTokenizer

    last = None
    for candidate in (ADAPTER_ID, BASE_ID):
        try:
            tok = AutoTokenizer.from_pretrained(candidate)
            print(f"[juxy] tokenizer <- {candidate}", flush=True)
            return tok
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[juxy] tokenizer {candidate} failed: {exc}", flush=True)
    raise RuntimeError(f"no usable tokenizer: {last}")


def _load_model():
    import transformers
    from peft import PeftModel

    errors = []
    for name in ("AutoModelForImageTextToText", "Qwen3_5ForConditionalGeneration", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            base = cls.from_pretrained(
                BASE_ID,
                dtype=torch.bfloat16,
                device_map="cuda",
                low_cpu_mem_usage=True,
            )
            # ZeroGPU emulates CUDA outside @spaces.GPU, but safetensors cannot
            # stream straight into the emulated device. Pull the adapter in on
            # CPU (`torch_device="cpu"`) and let PEFT copy it onto the modules.
            adapter_error = None
            model = None
            for adapter_kwargs in ({"torch_device": "cpu"}, {}):
                try:
                    model = PeftModel.from_pretrained(base, ADAPTER_ID, **adapter_kwargs)
                    adapter_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    adapter_error = exc
                    print(f"[juxy] adapter load {adapter_kwargs} failed: {exc}", flush=True)
                    traceback.print_exc()
            if model is None:
                raise adapter_error
            model.eval()

            # The checkpoint ships text weights only; the vision tower is
            # randomly initialised and costs us nothing useful. Drop it.
            inner = getattr(getattr(model, "base_model", None), "model", None)
            if inner is not None and hasattr(inner, "visual"):
                try:
                    inner.visual = None
                except Exception:  # noqa: BLE001
                    pass

            print(f"[juxy] loaded via {name}", flush=True)
            return model
        except Exception as exc:  # noqa: BLE001
            print(f"[juxy] load attempt {name} failed: {exc}", flush=True)
            traceback.print_exc()
            errors.append(f"{name}: {exc}")

    raise RuntimeError("could not load Juxy:\n" + "\n".join(errors))


def _env_probe():
    """Report the real container limits — these, not VRAM, usually bound model size."""
    import platform
    import shutil

    try:
        du = shutil.disk_usage("/")
        print(f"[juxy] disk  total={du.total / 1e9:.0f}GB  free={du.free / 1e9:.0f}GB", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[juxy] disk probe failed: {exc}", flush=True)
    try:
        import psutil

        vm = psutil.virtual_memory()
        print(f"[juxy] ram   total={vm.total / 1e9:.0f}GB  available={vm.available / 1e9:.0f}GB", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[juxy] ram probe failed: {exc}", flush=True)
    try:
        home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        du = shutil.disk_usage(home)
        print(f"[juxy] hf-cache mount free={du.free / 1e9:.0f}GB", flush=True)
    except Exception:  # noqa: BLE001
        pass
    print(f"[juxy] cpus={os.cpu_count()} python={platform.python_version()} torch={torch.__version__}", flush=True)


_env_probe()

print("[juxy] loading tokenizer...", flush=True)
TOKENIZER = _load_tokenizer()
print("[juxy] loading model (first boot pulls ~9.3 GB)...", flush=True)
MODEL = _load_model()
print("[juxy] ready.", flush=True)


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def _normalise(history):
    """Accept both Gradio message-dict history and legacy (user, bot) tuples."""
    out = []
    for turn in history or []:
        if isinstance(turn, dict):
            role, content = turn.get("role"), turn.get("content")
            if role in ("user", "assistant") and content:
                out.append({"role": role, "content": content})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user, bot = turn
            if user:
                out.append({"role": "user", "content": user})
            if bot:
                out.append({"role": "assistant", "content": bot})
    return out


def _eos_ids():
    ids = {FALLBACK_EOS, getattr(TOKENIZER, "eos_token_id", None)}
    ids.discard(None)
    return sorted(ids)


def _duration(message, history, temperature, max_new_tokens):
    """Cap the GPU reservation near what the answer actually needs."""
    return 20 + int(max_new_tokens) * 0.15


_GPU_REPORTED = False


@spaces.GPU(duration=_duration)
def generate(message, history, temperature, max_new_tokens):
    global _GPU_REPORTED
    if not _GPU_REPORTED:
        try:
            props = torch.cuda.get_device_properties(0)
            print(
                f"[juxy] gpu={torch.cuda.get_device_name(0)} "
                f"vram={props.total_memory / 1e9:.1f}GB cc={props.major}.{props.minor} "
                f"sm={props.multi_processor_count}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[juxy] gpu probe failed: {exc}", flush=True)
        _GPU_REPORTED = True
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += _normalise(history)
        messages.append({"role": "user", "content": message})

        try:
            prompt = TOKENIZER.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:  # template without the enable_thinking switch
            prompt = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
        if inputs["input_ids"].shape[1] > MAX_CONTEXT:  # keep the newest turns
            inputs["input_ids"] = inputs["input_ids"][:, -MAX_CONTEXT:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -MAX_CONTEXT:]

        with torch.no_grad():
            out = MODEL.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0,
                temperature=max(float(temperature), 1e-5),
                top_p=0.95,
                repetition_penalty=1.05,
                eos_token_id=_eos_ids(),
                pad_token_id=getattr(TOKENIZER, "pad_token_id", None) or FALLBACK_EOS,
            )

        completion = out[0][inputs["input_ids"].shape[1]:]
        text = TOKENIZER.decode(completion, skip_special_tokens=True).strip()
        if "</think>" in text:  # stray reasoning preamble
            text = text.split("</think>")[-1].strip()
        return text or "(empty response — try again)"
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return f"⚠️ generation failed: `{exc}`"


DESCRIPTION = (
    f"**Juxy v2** — `{BASE_ID}` + LoRA `{ADAPTER_ID}` (262k context, coding + cybersec).\n\n"
    "Running on **ZeroGPU** (free tier: ~5 min of GPU per account per day, then it queues)."
)

demo = gr.ChatInterface(
    fn=generate,
    title="Juxy v2 — coding + cybersec",
    description=DESCRIPTION,
    examples=[
        ["Who are you?", 0.2, 512],
        ["Write python port scanner, tcp 80,443 on 127.0.0.1, concise", 0.2, 512],
        ["Explain SQLi fix for a python login query", 0.2, 512],
    ],
    additional_inputs=[
        gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="temperature"),
        gr.Slider(128, 1536, value=512, step=128, label="max new tokens"),
    ],
    additional_inputs_accordion="Generation settings",
)

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
