# src/evaluate.py
import argparse, json, time, statistics, csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from evaluate import load as load_metric
from transformers import AutoTokenizer

from inference import generate_one

# Optional: lightweight exact match / F1 for QA-style answers
def normalize(s:str)->str:
    import re, string
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def f1_score(pred:str, gold:str)->float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not pred_tokens or not gold_tokens: return 0.0
    if not common: return 0.0
    prec = len(common)/len(pred_tokens)
    rec  = len(common)/len(gold_tokens)
    if prec+rec == 0: return 0.0
    return 2*prec*rec/(prec+rec)

def load_jsonl(path:str)->List[Dict[str,Any]]:
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def run_setting(name:str, model_id:str, test_path:str,
                adapter_dir:Optional[str]=None, use_rag:bool=False,
                max_items:int=0)->Dict[str,Any]:
    data = load_jsonl(test_path)
    if max_items>0: data = data[:max_items]
    results = []
    for ex in data:
        q = ex["prompt"]
        gold = ex.get("response","")
        res = generate_one(model_id=model_id, query=q, adapter_dir=adapter_dir, use_rag=use_rag)
        res_row = {
            "setting": name,
            "q": q,
            "gold": gold,
            "pred": res["text"],
            "prompt_tokens": res["prompt_tokens"],
            "output_tokens": res["output_tokens"],
            "latency_ms": res["latency_ms"],
            "model_id": res["model_id"],
            "adapter": res["adapter"],
            "rag_used": res["rag_used"],
            "f1": f1_score(res["text"], gold) if gold else None,
            "em": 1.0 if normalize(res["text"]) == normalize(gold) and gold else 0.0
        }
        results.append(res_row)
    return summarize(results)

def summarize(rows:List[Dict[str,Any]])->Dict[str,Any]:
    # Aggregate metrics
    f1s  = [r["f1"] for r in rows if r["f1"] is not None]
    ems  = [r["em"] for r in rows if r["em"] is not None]
    ptks = [r["prompt_tokens"] for r in rows]
    otks = [r["output_tokens"] for r in rows]
    lats = [r["latency_ms"] for r in rows]

    def avg(xs): return round(sum(xs)/len(xs), 4) if xs else 0.0
    def p50(xs): 
        xs_sorted=sorted(xs)
        mid=len(xs_sorted)//2
        return xs_sorted[mid] if len(xs_sorted)%2 else round((xs_sorted[mid-1]+xs_sorted[mid])/2,1)
    def p95(xs):
        xs_sorted=sorted(xs)
        idx = max(0, int(0.95*(len(xs_sorted)-1)))
        return xs_sorted[idx]

    report = {
        "n": len(rows),
        "f1_avg": avg(f1s),
        "em_avg": avg(ems),
        "prompt_tokens_avg": avg(ptks),
        "output_tokens_avg": avg(otks),
        "latency_p50_ms": p50(lats) if lats else 0.0,
        "latency_p95_ms": p95(lats) if lats else 0.0,
        "rows": rows
    }
    return report

def save_reports(report:Dict[str,Any], out_prefix:Path):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # JSON
    with open(str(out_prefix)+".json","w",encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    # CSV (summary)
    with open(str(out_prefix)+"_summary.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n","f1_avg","em_avg","prompt_tokens_avg","output_tokens_avg","latency_p50_ms","latency_p95_ms"])
        w.writerow([report["n"], report["f1_avg"], report["em_avg"], report["prompt_tokens_avg"], report["output_tokens_avg"], report["latency_p50_ms"], report["latency_p95_ms"]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--test_path", default="data/processed/test.jsonl")
    ap.add_argument("--adapter_dir", default="results/models/phi3_lora/adapter")
    ap.add_argument("--out_dir", default="results/metrics")
    ap.add_argument("--max_items", type=int, default=0, help="limit for quick runs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Baseline
    rep_baseline = run_setting("baseline", args.model_id, args.test_path, adapter_dir=None, use_rag=False, max_items=args.max_items)
    save_reports(rep_baseline, out_dir / "baseline")

    # 2) LoRA
    rep_lora = run_setting("lora", args.model_id, args.test_path, adapter_dir=args.adapter_dir, use_rag=False, max_items=args.max_items)
    save_reports(rep_lora, out_dir / "lora")

    # 3) RAG+compression (no LoRA)
    rep_rag = run_setting("rag_compress", args.model_id, args.test_path, adapter_dir=None, use_rag=True, max_items=args.max_items)
    save_reports(rep_rag, out_dir / "rag_compress")

    # 4) LoRA + RAG + compression
    rep_lora_rag = run_setting("lora_rag_compress", args.model_id, args.test_path, adapter_dir=args.adapter_dir, use_rag=True, max_items=args.max_items)
    save_reports(rep_lora_rag, out_dir / "lora_rag_compress")

    print("✅ Wrote reports to", out_dir.resolve())

if __name__ == "__main__":
    main()
