# scripts/debug_batch.py
import torch, json
from pathlib import Path
from datasets import load_dataset
from inference import load_model
from transformers import AutoTokenizer

tok, model, device = load_model("microsoft/phi-3-mini-4k-instruct", adapter_dir=None, device="auto")
model.train(False)

# load 1 item from your processed set
row = json.loads(Path("data/processed/train.jsonl").read_text().splitlines()[0])
prompt = row["prompt"]; answer = row["response"]

# build prompt+answer → labels mask: prompt=-100, answer tokens=labels
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

max_len = 512
enc_prompt = tok(prompt, add_special_tokens=False)
enc_answer = tok(answer, add_special_tokens=False)

# truncate from the left on prompt if needed, but keep answer intact
while len(enc_prompt["input_ids"]) + len(enc_answer["input_ids"]) + 1 > max_len:
    enc_prompt["input_ids"] = enc_prompt["input_ids"][1:]
    enc_prompt["attention_mask"] = enc_prompt["attention_mask"][1:]

ids = enc_prompt["input_ids"] + enc_answer["input_ids"] + [tok.eos_token_id]
att = [1]*len(ids)

labels = [-100]*len(enc_prompt["input_ids"]) + enc_answer["input_ids"] + [tok.eos_token_id]

x = torch.tensor([ids]).to(model.device)
m = torch.tensor([att]).to(model.device)
y = torch.tensor([labels]).to(model.device)

with torch.no_grad():
    out = model(input_ids=x, attention_mask=m, labels=y)
    loss = out.loss

print("len(ids)=", len(ids))
print("labels_non_ignored=", int((y!=-100).sum().item()))
print("loss=", float(loss))