# Juxy inference — Unsloth 4-bit fast (40 tok/s on T4, 20 tok/s on CPU offload)
# pip install unsloth transformers accelerate bitsandbytes
import sys
from unsloth import FastLanguageModel

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "juxy-lora"  # or YOUR_USERNAME/juxy-7b
BASE = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"

with open("juxy_prompt.txt") as f:
    SYS = f.read().strip()

model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=4096, dtype=None, load_in_4bit=True)
# if you pushed LoRA:
try:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER)
    print(f"[juxy] loaded adapter {ADAPTER}")
except Exception as e:
    print(f"[juxy] using base only ({e})")

FastLanguageModel.for_inference(model)

def ask(user: str):
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": user},
    ]
    inputs = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(inputs, max_new_tokens=1024, temperature=0.2, top_p=0.95, use_cache=True)
    print(tok.batch_decode(out[:, inputs.shape[1]:])[0])

if __name__ == "__main__":
    print("Juxy ready. Ctrl+C to exit.")
    while True:
        try: q = input("\n> ")
        except: break
        if q.strip(): ask(q)
