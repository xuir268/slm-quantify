# src/train_lora.py
import os
# Set MPS memory configuration before importing torch
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable upper limit

import json, math, argparse, yaml, random, inspect
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
    TrainingArguments, Trainer
)
import transformers
from peft import LoraConfig, get_peft_model

def set_seed(seed=42):
    import numpy as np, random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(path:str)->dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def format_example(ex, tok, max_len):
    # expected fields: "prompt", "response"
    text = f"<|user|>\n{ex['prompt']}\n<|assistant|>\n{ex['response']}"
    enc = tok(text, truncation=True, max_length=max_len, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc

def main(cfg_path:str):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))

    model_id   = cfg["model_id"]
    max_len    = int(cfg.get("seq_len", 1024))
    lora_cfg   = cfg.get("lora", {})
    train_path = cfg.get("data", {}).get("train", "data/processed/train.jsonl")
    dev_path   = cfg.get("data", {}).get("dev",   "data/processed/dev.jsonl")
    out_dir    = cfg.get("output_dir", f"results/models/{Path(model_id).name}_lora")
    os.makedirs(out_dir, exist_ok=True)

    is_cuda = torch.cuda.is_available()
    is_mps  = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    device  = "cuda" if is_cuda else ("mps" if is_mps else "cpu")

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    use_qlora = bool(cfg.get("use_qlora", False))
    if use_qlora and not is_cuda:
        raise RuntimeError("use_qlora=True requires CUDA (bitsandbytes). Run on Colab/RTX or set use_qlora=False.")

    if use_qlora:
        # 4-bit path (Colab/RTX)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=True,
            device_map="auto",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        # fp16 on CUDA, bf16 otherwise
        torch_dtype = torch.float16 if is_cuda else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    # LoRA config
    target_modules = lora_cfg.get("target_modules", ["q_proj","k_proj","v_proj","o_proj"])
    peft_config = LoraConfig(
        r            = lora_cfg.get("r", 16),
        lora_alpha   = lora_cfg.get("alpha", 32),
        lora_dropout = lora_cfg.get("dropout", 0.05),
        bias         = "none",
        task_type    = "CAUSAL_LM",
        target_modules = target_modules
    )
    model = get_peft_model(model, peft_config)

    try:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # Dataset
    ds = load_dataset("json", data_files={"train": train_path, "dev": dev_path})
    def _map(batch):
        out = {k:[] for k in ["input_ids","attention_mask","labels"]}
        for p,r in zip(batch["prompt"], batch["response"]):
            enc = tok(f"<|user|>\n{p}\n<|assistant|>\n{r}",
                      truncation=True, max_length=max_len, padding="max_length")
            input_ids = enc["input_ids"]
            attn      = enc["attention_mask"]
            # mask pads in labels with -100 so loss ignores them
            labels = [tid if m==1 else -100 for tid, m in zip(input_ids, attn)]
            out["input_ids"].append(input_ids)
            out["attention_mask"].append(attn)
            out["labels"].append(labels)
        return out

    cols = ds["train"].column_names
    ds_tok = ds.map(_map, batched=True, remove_columns=cols)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Training arguments with proper compatibility handling
    args_common = dict(
        output_dir = out_dir,
        per_device_train_batch_size = cfg.get("batch_size", 2),
        per_device_eval_batch_size  = cfg.get("eval_batch_size", 2),
        gradient_accumulation_steps = cfg.get("grad_accum", 8),
        num_train_epochs = cfg.get("epochs", 2),
        learning_rate    = float(cfg.get("lr", 2e-4)),
        lr_scheduler_type= cfg.get("lr_scheduler", "cosine"),
        warmup_ratio     = float(cfg.get("warmup_ratio", 0.03)),
        logging_steps    = cfg.get("logging_steps", 50),
        eval_steps       = cfg.get("eval_steps", 200),
        save_steps       = cfg.get("save_steps", 200),
        save_total_limit = cfg.get("save_total_limit", 2),
        report_to        = "none",
        dataloader_pin_memory = False if not torch.cuda.is_available() else True,
        dataloader_num_workers = 2 if torch.cuda.is_available() else 0,
        save_safetensors = True,
    )
    
    if is_cuda:
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            args_common["bf16"] = True
        else:
            args_common["fp16"] = True
    else:
        args_common["bf16"] = False
        args_common["fp16"] = False

    # Figure out which kw names exist in your installed transformers
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())

    # Start from common args
    ta_kwargs = dict(**args_common)

    # Add only the supported strategy parameters
    # eval strategy
    val = cfg.get("eval_strategy") or cfg.get("evaluation_strategy") or "no"
    if "eval_strategy" in params:
        ta_kwargs["eval_strategy"] = val
    elif "evaluation_strategy" in params:
        ta_kwargs["evaluation_strategy"] = val

    # save strategy
    val = cfg.get("save_strategy") or "epoch"
    if "save_strategy" in params:
        ta_kwargs["save_strategy"] = val

    # logging strategy
    val = cfg.get("logging_strategy") or "steps"
    if "logging_strategy" in params:
        ta_kwargs["logging_strategy"] = val

    if "max_steps" in params and cfg.get("max_steps", 0):
        ta_kwargs["max_steps"] = int(cfg["max_steps"])

    args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
    model = model,
    args  = args,
    train_dataset = ds_tok["train"],
    eval_dataset  = ds_tok["dev"],
    data_collator = collator
    )


    trainer.train()
    model.save_pretrained(Path(out_dir) / "adapter")
    with open(Path(out_dir) / "training_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    prec = "4bit" if use_qlora else ("fp16" if is_cuda else "bf16")
    print(f"ℹ️  Device: {device} | Precision: {prec}")
    print(f"✅ Saved LoRA adapter to: {Path(out_dir) / 'adapter'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()
    main(args.config)
