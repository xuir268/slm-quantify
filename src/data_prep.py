#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_prep.py
- Ingest raw domain corpus (security/healthcare/custom)
- Clean, dedupe, split
- Generate Q&A pairs (rule-based OR teacher-assisted)
- Write JSONL: train.jsonl, dev.jsonl, test.jsonl
"""

import argparse, json, os, random, re
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
from slugify import slugify

SEED = 42
random.seed(SEED)

# ---------- helpers ----------

def _read_jsonl(path: Path) -> List[dict]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def _write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _dedupe_keep_longest(items: List[str]) -> List[str]:
    seen=set(); out=[]
    for x in sorted(items, key=len, reverse=True):
        k = x.lower()
        if k not in seen:
            seen.add(k); out.append(x)
    # keep input order-ish by length; that’s fine for corpora
    return list(reversed(out))

def _train_dev_test_split(items: List[dict], ratios=(0.8, 0.1, 0.1)) -> Tuple[List[dict], List[dict], List[dict]]:
    random.shuffle(items)
    n=len(items); n_train=int(n*ratios[0]); n_dev=int(n*ratios[1])
    return items[:n_train], items[n_train:n_train+n_dev], items[n_train+n_dev:]

# ---------- corpus loaders ----------

def load_security_corpus(raw_dir: Path) -> List[Dict]:
    """
    Expect either:
      - raw/cve.jsonl with fields: id, summary (or description)
      - OR raw/*.csv with columns: id, description/summary
    Returns list of dicts: {"doc_id", "title", "text"}
    """
    candidates = list(raw_dir.glob("cve.jsonl"))
    rows=[]
    if candidates:
        for r in _read_jsonl(candidates[0]):
            doc_id = r.get("id") or r.get("cve_id") or slugify(r.get("summary",""))[:32]
            text = _clean_text(r.get("summary") or r.get("description") or "")
            title = f"CVE {doc_id}"
            if text: rows.append({"doc_id": str(doc_id), "title": title, "text": text})
    else:
        for csv in raw_dir.glob("*.csv"):
            df = pd.read_csv(csv)
            for _, rr in df.iterrows():
                doc_id = str(rr.get("id") or rr.get("cve_id") or slugify(str(rr.get("description","")))[:32])
                text = _clean_text(str(rr.get("summary") or rr.get("description") or ""))
                title = f"CVE {doc_id}"
                if text: rows.append({"doc_id": doc_id, "title": title, "text": text})

    # dedupe on text
    texts = [r["text"] for r in rows]
    texts = _dedupe_keep_longest(texts)
    # rebuild rows with new ids
    out=[]
    for i, t in enumerate(texts):
        out.append({"doc_id": f"SEC-{i+1:06d}", "title": f"Security-{i+1}", "text": t})
    return out

def load_health_corpus(raw_dir: Path) -> List[Dict]:
    """
    Expect either:
      - raw/pubmed.jsonl with fields: title, abstract
      - OR raw/*.csv with columns: title, abstract
    """
    rows=[]
    jl = raw_dir / "pubmed.jsonl"
    if jl.exists():
        for r in _read_jsonl(jl):
            title = _clean_text(r.get("title",""))
            abstract = _clean_text(r.get("abstract",""))
            if abstract:
                rows.append({"doc_id": slugify(title)[:32] or f"PM-{len(rows)+1}",
                             "title": title or f"PM-{len(rows)+1}", "text": abstract})
    else:
        for csv in raw_dir.glob("*.csv"):
            df = pd.read_csv(csv)
            for _, rr in df.iterrows():
                title = _clean_text(str(rr.get("title","")))
                abstract = _clean_text(str(rr.get("abstract","")))
                if abstract:
                    rows.append({"doc_id": slugify(title)[:32] or f"PM-{len(rows)+1}",
                                 "title": title or f"PM-{len(rows)+1}", "text": abstract})

    texts = [r["text"] for r in rows]
    texts = _dedupe_keep_longest(texts)
    out=[]
    for i, t in enumerate(texts):
        out.append({"doc_id": f"MED-{i+1:06d}", "title": f"Med-{i+1}", "text": t})
    return out

def load_custom_corpus(raw_dir: Path) -> List[Dict]:
    """
    Generic loader:
      - jsonl with {"title","text"} or {"text"}
      - csv with columns title,text OR just text
    """
    rows=[]
    for p in list(raw_dir.glob("*.jsonl")):
        for r in _read_jsonl(p):
            text=_clean_text(r.get("text",""))
            title=_clean_text(r.get("title","")) or f"DOC-{len(rows)+1}"
            if text: rows.append({"doc_id": f"CUS-{len(rows)+1}", "title": title, "text": text})
    for p in list(raw_dir.glob("*.csv")):
        df = pd.read_csv(p)
        for _, rr in df.iterrows():
            text=_clean_text(str(rr.get("text","")))
            title=_clean_text(str(rr.get("title",""))) or f"DOC-{len(rows)+1}"
            if text: rows.append({"doc_id": f"CUS-{len(rows)+1}", "title": title, "text": text})
    # dedupe
    texts=[r["text"] for r in rows]
    texts=_dedupe_keep_longest(texts)
    out=[]
    for i,t in enumerate(texts):
        out.append({"doc_id": f"CUS-{i+1:06d}", "title": f"Doc-{i+1}", "text": t})
    return out

# ---------- Q&A generation ----------

def qa_from_security(text: str) -> List[Tuple[str,str]]:
    """
    Very lightweight rule-based questions for CVE-like text.
    You can replace/augment with a teacher LLM later.
    """
    qas=[]
    base_qs = [
        "What vulnerability is described?",
        "What components or products are affected?",
        "What is the potential impact?",
        "Is there a mitigation or workaround?",
    ]
    for q in base_qs:
        a = _extract_answer_like(text, q)
        qas.append((q, a))
    return qas

def qa_from_health(text: str) -> List[Tuple[str,str]]:
    qas=[]
    base_qs = [
        "What is the main finding?",
        "Which population or dataset was studied?",
        "What method or intervention was used?",
        "What limitation is mentioned?",
    ]
    for q in base_qs:
        a = _extract_answer_like(text, q)
        qas.append((q, a))
    return qas

def _extract_answer_like(text: str, question: str) -> str:
    """
    Tiny heuristic: take first 1-2 sentences as the 'answer'.
    Replace with a call to your teacher model for high quality.
    """
    sents = re.split(r"(?<=[.!?])\s+", text)
    a = " ".join(sents[:2]).strip()
    return a[:600]  # keep short

# ---- Optional: teacher LLM stub (wire your API of choice) ----

def qa_via_teacher(text: str, domain: str, n_pairs: int = 4) -> List[Tuple[str,str]]:
    """
    Replace this with actual API calls to your teacher (e.g., GPT-5).
    Keep it here so the rest of the pipeline stays the same.
    """
    # Pseudocode:
    # prompt = f"Read the passage and generate {n_pairs} helpful Q&A pairs for {domain}.\nPASSAGE:\n{text}"
    # response = call_teacher(prompt)
    # return parse_pairs(response)
    return []  # default to empty so rule-based still works

# ---------- main pipeline ----------

def build_pairs(corpus: List[Dict], domain: str, use_teacher: bool, max_per_doc: int=4) -> List[Dict]:
    out=[]
    for doc in tqdm(corpus, desc="Generating Q&A"):
        text = doc["text"]
        pairs = []
        if use_teacher:
            pairs = qa_via_teacher(text, domain, n_pairs=max_per_doc)
        if not pairs:
            # fallback to rule-based
            if domain == "security":
                pairs = qa_from_security(text)
            elif domain == "health":
                pairs = qa_from_health(text)
            else:
                # generic
                pairs = qa_from_security(text)  # reuse structure

        pairs = pairs[:max_per_doc]
        for i, (q,a) in enumerate(pairs):
            q = _clean_text(q)
            a = _clean_text(a)
            if not q or not a: continue
            out.append({
                "id": f"{doc['doc_id']}-{i+1}",
                "source_id": doc["doc_id"],
                "prompt": q,
                "response": a,
                "meta": {"title": doc["title"], "domain": domain}
            })
    return out

def run(args):
    raw_dir   = Path(args.raw_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load corpus
    if args.domain == "security":
        corpus = load_security_corpus(raw_dir)
    elif args.domain == "health":
        corpus = load_health_corpus(raw_dir)
    else:
        corpus = load_custom_corpus(raw_dir)

    if len(corpus) == 0:
        raise SystemExit(f"No corpus found under {raw_dir}. Put files in raw/ first.")

    # 2) generate Q&A pairs
    pairs = build_pairs(corpus, args.domain, use_teacher=args.use_teacher, max_per_doc=args.max_per_doc)

    # 3) filter very short answers
    pairs = [p for p in pairs if len(p["response"].split()) >= 6]

    # 4) split
    train, dev, test = _train_dev_test_split(pairs, ratios=(args.train_ratio, args.dev_ratio, 1.0-args.train_ratio-args.dev_ratio))

    # 5) write
    _write_jsonl(out_dir/"train.jsonl", train)
    _write_jsonl(out_dir/"dev.jsonl",   dev)
    _write_jsonl(out_dir/"test.jsonl",  test)

    # save a small readme about data
    with (out_dir/"README_DATA.txt").open("w", encoding="utf-8") as f:
        f.write(f"Domain: {args.domain}\nTotal pairs: {len(pairs)}\nTrain/Dev/Test sizes: {len(train)}/{len(dev)}/{len(test)}\nSeed: {SEED}\n")

    print(f"✅ Wrote: {out_dir/'train.jsonl'}, {out_dir/'dev.jsonl'}, {out_dir/'test.jsonl'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["security","health","custom"], required=True,
                    help="Choose your domain to enable sensible defaults.")
    ap.add_argument("--raw_dir", type=str, default="data/raw",
                    help="Folder with raw corpus files (jsonl/csv).")
    ap.add_argument("--out_dir", type=str, default="data/processed",
                    help="Output folder for JSONL train/dev/test.")
    ap.add_argument("--use_teacher", action="store_true",
                    help="If set, calls qa_via_teacher() (you must implement the API call).")
    ap.add_argument("--max_per_doc", type=int, default=4,
                    help="Max Q&A pairs to extract per document.")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    args = ap.parse_args()
    run(args)
