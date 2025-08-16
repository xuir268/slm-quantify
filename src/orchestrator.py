import time, yaml, json, inspect
from pathlib import Path
from typing import Dict, Any, List
from inference import load_model, count_tokens
from verifier import verify
from rag_compression import retrieve_topk, compress_to_bullets, build_prompt
import transformers as tf

class Runner:
    def __init__(self, model_id, adapter_dir=None, device="auto", quant: str=None):
        self.tok, self.model, self.device = load_model(model_id, adapter_dir, device, quant)

    def generate(self, prompt:str, gen_kwargs=None)->Dict[str,Any]:
        import torch
        gen_kwargs = gen_kwargs or {"max_new_tokens":200, "temperature":0.7, "top_p":0.9}
        gen_kwargs.setdefault("do_sample", True)
        x = self.tok(prompt, return_tensors="pt").to(self.model.device)
        t0=time.time()
        y = self.model.generate(**x, **gen_kwargs)
        dt=(time.time()-t0)*1000
        text = self.tok.decode(y[0], skip_special_tokens=True)
        # trim prompt echo if present
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return {"text": text, "latency_ms": dt,
                "ptoks": count_tokens(self.tok, prompt),
                "otoks": count_tokens(self.tok, text)}

def build_context(question:str, top_k:int, max_bullets:int):
    ctx = retrieve_topk(question, k=top_k)
    bullets = compress_to_bullets(ctx, max_bullets=max_bullets, query=question)
    prompt = build_prompt(question, bullets)
    return bullets, prompt

class Orchestrator:
    def __init__(self, cfg_path:str):
        cfg = yaml.safe_load(open(cfg_path))
        self.cfg = cfg
        self.runners = {}
        for m in cfg["models"]:
            key = m["name"]
            self.runners[key] = Runner(
                m["id"],
                m.get("adapter","") or None,
                device="auto",
                quant=m.get("quant")
            )
        self.order = cfg["policy"]["order"]
        self.mode  = cfg["policy"]["mode"]
        self.thr   = cfg["verifier"]["threshold"]
        self.top_k = cfg["rag"]["top_k"]
        self.max_b = cfg["rag"]["max_bullets"]
        self.ens_n = cfg["policy"].get("ensemble_size", 2)

    def answer(self, question:str)->Dict[str,Any]:
        bullets, prompt = build_context(question, self.top_k, self.max_b)

        if self.mode == "router":
            # simple heuristic: shorter questions -> phi3, else qwen3b
            target = self.order[0] if len(question)<140 else self.order[1]
            out = self.runners[target].generate(prompt)
            score = verify(out["text"], bullets)
            return {"text": out["text"], "model": target, "conf": score, **out, "bullets": bullets}

        if self.mode == "cascade":
            last=None
            for key in self.order:
                out = self.runners[key].generate(prompt)
                score = verify(out["text"], bullets)
                last = {"text": out["text"], "model": key, "conf": score, **out, "bullets": bullets}
                if score >= self.thr:
                    return last
            return last  # fell through; return strongest

        if self.mode == "ensemble":
            cand=[]
            for key in self.order[:self.ens_n]:
                out = self.runners[key].generate(prompt)
                score = verify(out["text"], bullets)
                cand.append((score, key, out))
            best = max(cand, key=lambda x: x[0])
            s,k,o = best
            return {"text": o["text"], "model": k, "conf": s, **o, "bullets": bullets}

        raise ValueError("Unknown mode")

if __name__ == "__main__":
    import argparse, csv, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to ensemble YAML (e.g., configs/ensemble.yaml)")
    ap.add_argument("--mode", choices=["router","cascade","ensemble"], help="Override policy.mode")
    ap.add_argument("--q", help="Single question to answer")
    ap.add_argument("--in_jsonl", help="Batch eval from JSONL with {'prompt','response'?}")
    ap.add_argument("--out_prefix", help="Output prefix (e.g., results/metrics/run1)")
    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()

    orch = Orchestrator(args.cfg)
    if args.mode:
        orch.mode = args.mode

    def _norm(s: str) -> str:
        import re, string
        s = s.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\s+", " ", s).strip()

    def _write_outputs(prefix: str, rows):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        # JSONL
        with open(prefix + ".jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # CSV summary
        f1s = []
        lats = []
        for r in rows:
            if "gold" in r and r["gold"]:
                p = set(_norm(r.get("text","")).split())
                g = set(_norm(r["gold"]).split())
                if p and g:
                    inter = len(p & g)
                    pre = inter / len(p)
                    rec = inter / len(g)
                    f1s.append(0 if pre + rec == 0 else 2 * pre * rec / (pre + rec))
            if "latency_ms" in r:
                lats.append(r["latency_ms"])
        lats_sorted = sorted(lats)
        n = len(rows)
        p50 = lats_sorted[n//2] if n else 0.0
        p95 = lats_sorted[int(0.95*(n-1))] if n else 0.0
        with open(prefix + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["n","f1_avg","latency_p50_ms","latency_p95_ms"])
            w.writerow([n, round(sum(f1s)/len(f1s),4) if f1s else 0.0, p50, p95])

    if args.q:
        out = orch.answer(args.q)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.in_jsonl:
        rows = []
        with open(args.in_jsonl, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.max_items and i >= args.max_items:
                    break
                ex = json.loads(line)
                q = ex.get("prompt") or ex.get("question") or ""
                o = orch.answer(q)
                if "response" in ex:
                    o["gold"] = ex["response"]
                rows.append(o)
        if args.out_prefix:
            _write_outputs(args.out_prefix, rows)
        else:
            print(json.dumps(rows[:3], ensure_ascii=False, indent=2))
    else:
        print("Nothing to do. Pass --q or --in_jsonl.")
