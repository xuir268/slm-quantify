# scripts/eval_orchestrator.py
import argparse, json, csv
from pathlib import Path
from typing import Dict, Any, List

from orchestrator import Orchestrator
from verifier import verify

def load_jsonl(path:str)->List[Dict[str,Any]]:
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

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

def summarize(rows:List[Dict[str,Any]])->Dict[str,Any]:
    def avg(xs): return round(sum(xs)/len(xs),4) if xs else 0.0
    def p50(xs):
        xs=sorted(xs); n=len(xs)
        return xs[n//2] if n%2 else round((xs[n//2-1]+xs[n//2])/2,1)
    def p95(xs):
        xs=sorted(xs); 
        return xs[max(0,int(0.95*(len(xs)-1)))]

    f1s=[r["f1"] for r in rows if r.get("f1") is not None]
    ems=[r["em"] for r in rows if r.get("em") is not None]
    p50s=[r["latency_ms"] for r in rows]
    pt=[r["prompt_tokens"] for r in rows]
    ot=[r["output_tokens"] for r in rows]

    return {
        "n": len(rows),
        "f1_avg": avg(f1s),
        "em_avg": avg(ems),
        "prompt_tokens_avg": avg(pt),
        "output_tokens_avg": avg(ot),
        "latency_p50_ms": p50(p50s) if p50s else 0.0,
        "latency_p95_ms": p95(p50s) if p50s else 0.0,
    }

def save_summary(report:Dict[str,Any], out_csv:Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["n","f1_avg","em_avg","prompt_tokens_avg","output_tokens_avg","latency_p50_ms","latency_p95_ms"])
        w.writerow([report["n"], report["f1_avg"], report["em_avg"],
                    report["prompt_tokens_avg"], report["output_tokens_avg"],
                    report["latency_p50_ms"], report["latency_p95_ms"]])

def main(cfg:str, test_path:str, out_dir:str, max_items:int):
    o = Orchestrator(cfg)
    data = load_jsonl(test_path)
    if max_items>0:
        data = data[:max_items]

    # run three modes so plotting sees *_summary.csv files
    for mode in ["router","cascade","ensemble"]:
        o.mode = mode
        rows=[]
        for ex in data:
            q = ex["prompt"]
            gold = ex.get("response","")
            res = o.answer(q)  # returns dict with text, model, conf, ptoks, otoks, latency_ms, bullets
            pred = res["text"]
            rows.append({
                "mode": mode,
                "q": q,
                "gold": gold,
                "pred": pred,
                "prompt_tokens": res.get("ptoks",0),
                "output_tokens": res.get("otoks",0),
                "latency_ms": res.get("latency_ms",0.0),
                "model": res.get("model",""),
                "conf": res.get("conf",0.0),
                "f1": f1_score(pred, gold) if gold else None,
                "em": 1.0 if (gold and (normalize(pred)==normalize(gold))) else 0.0
            })
        report = summarize(rows)
        out_csv = Path(out_dir)/f"orchestrator_{mode}_summary.csv"
        save_summary(report, out_csv)
        print(f"✅ {mode} summary → {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--test_path", default="data/processed/test.jsonl")
    ap.add_argument("--out_dir", default="results/metrics")
    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()
    main(args.cfg, args.test_path, args.out_dir, args.max_items)
