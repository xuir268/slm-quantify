# scripts/eval_orchestrator.py
# Ensure we can import modules from src/ when running this script directly
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SRC  = os.path.join(_ROOT, "src")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse, json, csv
from pathlib import Path
from typing import Dict, Any, List

try:
    from orchestrator import Orchestrator
except ModuleNotFoundError:
    from src.orchestrator import Orchestrator  # fallback if package-style import is needed

try:
    from verifier import verify
except ModuleNotFoundError:
    from src.verifier import verify

def load_jsonl(path:str)->List[Dict[str,Any]]:
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def normalize(s:str)->str:
    """
    Security-aware normalization:
    - lowercase
    - keep letters, digits, spaces and separators: . _ - /
    - collapse whitespace
    """
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9._/\-\s]+", " ", s)
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
    def avg(xs): 
        xs = [x for x in xs if x is not None]
        return round(sum(xs)/len(xs),4) if xs else 0.0
    def p50(xs):
        xs=[x for x in xs if x is not None]
        if not xs: return 0.0
        xs=sorted(xs); n=len(xs)
        return xs[n//2] if n%2 else round((xs[n//2-1]+xs[n//2])/2,1)
    def p95(xs):
        xs=[x for x in xs if x is not None]
        if not xs: return 0.0
        xs=sorted(xs); 
        return xs[max(0,int(0.95*(len(xs)-1)))]

    f1s=[r.get("f1") for r in rows if r.get("f1") is not None]
    ems=[r.get("em") for r in rows if r.get("em") is not None]
    lats=[r.get("latency_ms") for r in rows if r.get("latency_ms") is not None]
    pt  =[r.get("prompt_tokens") for r in rows if r.get("prompt_tokens") is not None]
    ot  =[r.get("output_tokens") for r in rows if r.get("output_tokens") is not None]

    return {
        "n": len(rows),
        "f1_avg": avg(f1s),
        "em_avg": avg(ems),
        "prompt_tokens_avg": avg(pt),
        "output_tokens_avg": avg(ot),
        "latency_p50_ms": p50(lats),
        "latency_p95_ms": p95(lats),
    }

def save_summary(report:Dict[str,Any], out_csv:Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Write deterministic header order
    field_order = ["n","f1_avg","em_avg","prompt_tokens_avg","output_tokens_avg","latency_p50_ms","latency_p95_ms"]
    row = {k: report.get(k, 0) for k in field_order}
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        w.writerow(row)

def main(cfg:str, test_path:str, out_dir:str, max_items:int, only_mode:str=None):
    o = Orchestrator(cfg)
    data = load_jsonl(test_path)
    if max_items>0:
        data = data[:max_items]

    def write_jsonl(path:Path, rows:List[Dict[str,Any]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # run selected modes (default: all three so plotting sees *_summary.csv files)
    modes = [only_mode] if only_mode else ["router","cascade","ensemble"]
    print(f"Running modes: {modes}")
    for mode in modes:
        o.mode = mode
        rows=[]
        for ex in data:
            q = ex["prompt"]
            gold = ex.get("response","")
            try:
                res = o.answer(q)  # dict: text, model, conf, ptoks, otoks, latency_ms, bullets
                pred = res.get("text","")
                bullets = res.get("bullets", [])
                vscore = verify(pred, bullets) if bullets else None
                row = {
                    "mode": mode,
                    "q": q,
                    "gold": gold,
                    "pred": pred,
                    "prompt_tokens": res.get("ptoks",0),
                    "output_tokens": res.get("otoks",0),
                    "latency_ms": res.get("latency_ms",0.0),
                    "model": res.get("model",""),
                    "conf": res.get("conf",0.0),
                    "verifier": vscore,
                    "f1": f1_score(pred, gold) if gold else None,
                    "em": 1.0 if (gold and (normalize(pred)==normalize(gold))) else 0.0
                }
            except Exception as e:
                # record failure but keep going
                row = {
                    "mode": mode,
                    "q": q,
                    "gold": gold,
                    "pred": "",
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": None,
                    "model": "",
                    "conf": 0.0,
                    "verifier": None,
                    "f1": None,
                    "em": 0.0,
                    "error": str(e)
                }
            rows.append(row)
        report = summarize(rows)
        out_csv = Path(out_dir)/f"orchestrator_{mode}_summary.csv"
        save_summary(report, out_csv)
        out_jsonl = Path(out_dir)/f"orchestrator_{mode}.jsonl"
        write_jsonl(out_jsonl, rows)
        print(f"✅ {mode} summary → {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--test_path", default="data/processed/test.jsonl")
    ap.add_argument("--out_dir", default="results/metrics")
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--mode", choices=["router","cascade","ensemble"], help="Run only this mode")
    args = ap.parse_args()
    main(args.cfg, args.test_path, args.out_dir, args.max_items, args.mode)
