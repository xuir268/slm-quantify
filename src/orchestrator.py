import time
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from inference import load_model, count_tokens
from verifier import verify
from rag_compression import retrieve_topk, compress_to_bullets, build_prompt

# --- helpers ---
from typing import Union

def _to_text(x: Union[str, List[str], None]) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(str(t) for t in x if t is not None)
    if isinstance(x, str):
        return x
    return str(x)


class Runner:
    """Wraps a (tokenizer, model) pair with deterministic, non-empty generation."""
    def __init__(self, model_id: str, adapter_dir: Optional[str] = None, device: str = "auto", quant: Optional[str] = None):
        self.tok, self.model, self.device = load_model(model_id, adapter_dir, device, quant)

    def generate(self, prompt: str, gen_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import torch
        # Defaults chosen to avoid empty strings and keep runs reproducible.
        gen_kwargs = gen_kwargs or {
            "do_sample": False,
            "max_new_tokens": 128,
            "min_new_tokens": 8,
        }
        x = self.tok(prompt, return_tensors="pt", padding=True).to(self.model.device)
        t0 = time.time()
        with torch.inference_mode():
            y = self.model.generate(
                input_ids=x["input_ids"],
                attention_mask=x.get("attention_mask"),
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.eos_token_id,
                **gen_kwargs,
            )
        dt_ms = (time.time() - t0) * 1000.0
        text = _to_text(self.tok.decode(y[0], skip_special_tokens=True))
        # Trim prompt echo if present
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        # fallback to avoid empty strings breaking downstream metrics
        if not text:
            text = ""
        return {
            "text": text,
            "latency_ms": dt_ms,
            "ptoks": count_tokens(self.tok, prompt),
            "otoks": count_tokens(self.tok, text),
        }


def build_context(question: str, top_k: int, max_bullets: int) -> Tuple[List[str], str]:
    """Retrieve naive context, compress to bullets, and build a final prompt."""
    ctx = retrieve_topk(question, k=top_k)
    bullets = compress_to_bullets(ctx, max_bullets=max_bullets, query=question)
    prompt = build_prompt(question, bullets)
    return bullets, prompt


class Orchestrator:
    """
    Orchestrates multiple SLMs in three modes:
      - router: choose a single model using a simple heuristic
      - cascade: try models in order until verifier confidence >= threshold
      - ensemble: run N models in parallel and select the best by verifier score
    Config (YAML) must include:
      models: [{id, name, adapter?, quant?}, ...]
      rag: {top_k, max_bullets}
      verifier: {threshold}
      policy: {mode, order: [names...], ensemble_size?}
    """
    def __init__(self, cfg_path: str):
        cfg = yaml.safe_load(open(cfg_path, "r"))
        self.cfg = cfg
        self.runners: Dict[str, Runner] = {}
        for m in cfg["models"]:
            name = m["name"]
            self.runners[name] = Runner(
                model_id=m["id"],
                adapter_dir=m.get("adapter") or None,
                device="auto",
                quant=m.get("quant"),
            )
        self.order: List[str] = cfg["policy"]["order"]
        self.mode: str = cfg["policy"]["mode"]
        self.thr: float = float(cfg["verifier"]["threshold"])
        self.top_k: int = int(cfg["rag"]["top_k"])
        self.max_b: int = int(cfg["rag"]["max_bullets"])
        self.ens_n: int = int(cfg["policy"].get("ensemble_size", 2))

        # Optional: global generation overrides
        self.gen_kwargs: Optional[Dict[str, Any]] = cfg.get("generation")

    def _verify_str(self, bullets: List[str]) -> str:
        return "\n".join(bullets) if isinstance(bullets, list) else (bullets or "")

    def _run_one(self, key: str, prompt: str) -> Dict[str, Any]:
        """Run a single model by key with safe error capture."""
        try:
            out = self.runners[key].generate(prompt, self.gen_kwargs)
            out["model"] = key
            out["text"] = _to_text(out.get("text", ""))
            out["latency_ms"] = float(out.get("latency_ms", 0.0) or 0.0)
            out["ptoks"] = int(out.get("ptoks", 0) or 0)
            out["otoks"] = int(out.get("otoks", 0) or 0)
            return out
        except Exception as e:
            return {"text": "", "latency_ms": 0.0, "ptoks": 0, "otoks": 0, "model": key, "error": f"{type(e).__name__}: {e}"}

    def answer(self, question: str) -> Dict[str, Any]:
        bullets, prompt = build_context(question, self.top_k, self.max_b)
        bul_text = self._verify_str(bullets)

        if self.mode == "router":
            # Heuristic: short factual -> first, longer/ambiguous -> second, else fallback third.
            if len(question) < 120:
                choice = self.order[0]
            elif len(question) < 240 and len(self.order) >= 2:
                choice = self.order[1]
            else:
                choice = self.order[-1]
            out = self._run_one(choice, prompt)
            txt = _to_text(out.get("text", ""))
            score = verify(txt, bul_text)
            row = {"q": question, "bullets": bullets, "conf": score, **out}
            row["text"] = txt
            return row

        if self.mode == "cascade":
            last = None
            for key in self.order:
                out = self._run_one(key, prompt)
                txt = _to_text(out.get("text", ""))
                score = verify(txt, bul_text)
                row = {"q": question, "bullets": bullets, "conf": score, **out}
                row["text"] = txt
                last = row
                if score >= self.thr:
                    return row
            return last or {"q": question, "bullets": bullets, "conf": 0.0, "text": "", "latency_ms": 0.0, "ptoks": 0, "otoks": 0, "model": self.order[0]}

        if self.mode == "ensemble":
            cand: List[Tuple[float, str, Dict[str, Any]]] = []
            for key in self.order[: self.ens_n]:
                out = self._run_one(key, prompt)
                txt = _to_text(out.get("text", ""))
                score = verify(txt, bul_text)
                out["text"] = txt
                cand.append((score, key, out))
            if not cand:
                return {"q": question, "bullets": bullets, "conf": 0.0, "text": "", "latency_ms": 0.0, "ptoks": 0, "otoks": 0, "model": self.order[0]}
            s, k, o = max(cand, key=lambda t: t[0])
            return {"q": question, "bullets": bullets, "conf": s, **o}

        raise ValueError(f"Unknown mode: {self.mode}")


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse, csv, os

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to ensemble YAML (e.g., configs/ensemble.yaml)")
    ap.add_argument("--mode", choices=["router", "cascade", "ensemble"], help="Override policy.mode")
    ap.add_argument("--q", help="Single question to answer")
    ap.add_argument("--in_jsonl", help="Batch eval from JSONL with {'prompt','response'?}")
    ap.add_argument("--out_prefix", help="Output prefix (e.g., results/metrics/orchestrator_router)")
    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()

    orch = Orchestrator(args.cfg)
    if args.mode:
        orch.mode = args.mode

    def _norm(s: str) -> str:
        import re, string
        s = s.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\s+", " ", s).strip()

    def _write_outputs(prefix: str, rows: List[Dict[str, Any]]):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        # JSONL with per-row details
        with open(prefix + ".jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Aggregates for plotting
        f1s, ems, lats, ptoks, otoks = [], [], [], [], []
        for r in rows:
            # latency
            lat = r.get("latency_ms")
            if isinstance(lat, (int, float)):
                lats.append(lat)
            # tokens
            for k, col in (("ptoks", ptoks), ("otoks", otoks)):
                v = r.get(k)
                if isinstance(v, (int, float)):
                    col.append(v)
            # quality
            gold = r.get("gold") or ""
            pred = r.get("text") or ""
            if gold:
                p = set(_norm(pred).split()); g = set(_norm(gold).split())
                if p and g:
                    inter = len(p & g)
                    pre = inter / len(p)
                    rec = inter / len(g)
                    f1s.append(0.0 if pre + rec == 0 else 2 * pre * rec / (pre + rec))
                    ems.append(1.0 if _norm(pred) == _norm(gold) else 0.0)

        def p50(xs: List[float]) -> float:
            if not xs: return 0.0
            xs = sorted(xs); n = len(xs)
            return xs[n//2] if n % 2 else (xs[n//2 - 1] + xs[n//2]) / 2

        def p95(xs: List[float]) -> float:
            if not xs: return 0.0
            xs = sorted(xs); idx = max(0, int(0.95 * (len(xs) - 1)))
            return xs[idx]

        def avg(xs: List[float]) -> float:
            return round(sum(xs) / len(xs), 4) if xs else 0.0

        with open(prefix + "_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["n", "f1_avg", "em_avg", "prompt_tokens_avg", "output_tokens_avg", "latency_p50_ms", "latency_p95_ms"])
            w.writerow([
                len(rows), avg(f1s), avg(ems), avg(ptoks), avg(otoks),
                round(p50(lats), 1), round(p95(lats), 1),
            ])

    if args.q:
        out = orch.answer(args.q)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.in_jsonl:
        rows: List[Dict[str, Any]] = []
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
        # Default prefix if not provided
        prefix = args.out_prefix or f"results/metrics/orchestrator_{orch.mode}"
        _write_outputs(prefix, rows)
        print(f"✅ Wrote {orch.mode} → {prefix}_summary.csv")
    else:
        print("Nothing to do. Pass --q or --in_jsonl.")
