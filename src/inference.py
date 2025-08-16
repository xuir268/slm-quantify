# src/inference.py
import time, argparse, torch
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    # Optional: if you created these files
    from rag_compression import retrieve_topk, compress_to_bullets, build_prompt
except Exception:
    retrieve_topk = compress_to_bullets = build_prompt = None

def _make_prompt(tok, system: str, user: str) -> str:
    # Prefer chat template if the tokenizer provides one (Qwen, Gemma, Mistral, Phi, etc.)
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback to simple Q/A style
    return f"{system}\n\nQ: {user}\nA:"

def load_model(model_id:str, adapter_dir:Optional[str]=None, device:str="auto", quant: Optional[str]=None):
    # Device selection
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Quantized (4-bit) path: only valid on CUDA (bitsandbytes)
    if quant == "4bit":
        if device != "cuda":
            raise RuntimeError("4-bit quantization requires CUDA (bitsandbytes). Run on Colab/RTX, not MPS/CPU.")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=True,
            device_map="auto",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        torch_dtype = torch.float16 if device == "cuda" else torch.bfloat16
        # device_map='auto' lets HF place weights on the available accelerator
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    if adapter_dir and Path(adapter_dir).exists():
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"🔗 Loaded LoRA/QLoRA adapter: {adapter_dir}")

    # Return tokenizer, model, and detected device string
    return tok, model, device

def count_tokens(tok, text:str)->int:
    return len(tok(text, add_special_tokens=False)["input_ids"])

def generate_one(model_id:str, query:str,
                 adapter_dir:Optional[str]=None,
                 use_rag:bool=False,
                 gen_kwargs:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
    tok, model, device = load_model(model_id, adapter_dir, device="auto", quant=None)
    gen_kwargs = gen_kwargs or {"max_new_tokens":200, "temperature":0.7, "top_p":0.9}
    gen_kwargs.setdefault("do_sample", True)
    # Build prompt
    if use_rag and retrieve_topk and compress_to_bullets and build_prompt:
        ctx = retrieve_topk(query, k=3)
        bullets = compress_to_bullets(ctx, max_bullets=6, query=query)
        prompt = build_prompt(query, bullets)
    else:
        system = "You are a concise domain assistant."
        prompt = _make_prompt(tok, system, query)

    ptoks = count_tokens(tok, prompt)
    t0 = time.time()
    x = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        y = model.generate(**x, **gen_kwargs)
    dt_ms = (time.time()-t0)*1000.0
    out = tok.decode(y[0], skip_special_tokens=True)

    # Extract only the answer: if the model echoed the prompt, strip it
    if out.startswith(prompt):
        answer = out[len(prompt):].strip()
    else:
        # Try to split on common assistant tag if present in decoded text
        answer = out.split("<|assistant|>", 1)[-1].strip() if "<|assistant|>" in out else out

    otoks = count_tokens(tok, answer)
    return {
        "text": answer,
        "prompt_tokens": ptoks,
        "output_tokens": otoks,
        "latency_ms": round(dt_ms,1),
        "model_id": model_id,
        "adapter": adapter_dir or "",
        "rag_used": bool(use_rag and retrieve_topk)
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--adapter_dir", default=None)
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--quant", choices=["4bit"], default=None, help="Set 4bit when loading a QLoRA base on CUDA")
    args = ap.parse_args()
    res = generate_one(args.model_id, args.query, args.adapter_dir, args.rag)
    print(res)
