# scripts/make_security_sft_from_dpo.py
import argparse, json
from pathlib import Path
from datasets import load_dataset

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main(out_dir: str, max_items: int):
    ds = load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split="train")
    rows=[]
    n = len(ds) if max_items<=0 else min(max_items, len(ds))
    for i in range(n):
        ex = ds[i]
        q = (ex.get("system") or "").strip()
        if ex.get("question"): q += ("\n\n" if q else "") + str(ex["question"]).strip()
        a = (ex.get("chosen") or "").strip()
        if not a or not (q.strip()): 
            continue
        rows.append({"prompt": q, "response": a})

    # 80/10/10 split
    n=len(rows); n_tr=int(0.8*n); n_dev=int(0.1*n)
    train, dev, test = rows[:n_tr], rows[n_tr:n_tr+n_dev], rows[n_tr+n_dev:]
    out = Path(out_dir)
    write_jsonl(out/"train.jsonl", train)
    write_jsonl(out/"dev.jsonl",   dev)
    write_jsonl(out/"test.jsonl",  test)
    print(f"✅ Wrote {len(train)}/{len(dev)}/{len(test)} to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()
    main(args.out_dir, args.max_items)
