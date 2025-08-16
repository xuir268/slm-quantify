# train_prep.py
import json, argparse
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split

def load_pairs(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_and_clean(pairs_list: List[List[Dict]]) -> List[Dict]:
    all_pairs = []
    seen_ids = set()
    for pairs in pairs_list:
        for p in pairs:
            if p["id"] not in seen_ids and p["prompt"] and p["response"]:
                seen_ids.add(p["id"])
                # Optional cleaning: trim long whitespace
                p["prompt"] = p["prompt"].strip()
                p["response"] = p["response"].strip()
                all_pairs.append(p)
    return all_pairs

def to_alpaca_format(pairs: List[Dict]) -> List[Dict]:
    return [
        {
            "instruction": p["prompt"],
            "input": "",
            "output": p["response"]
        }
        for p in pairs
    ]

def _save_jsonl(items: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_dataset(train, dev, test, out_dir: Path, fmt: str = "jsonl"):
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        _save_jsonl(train, out_dir / "train.jsonl")
        _save_jsonl(dev,   out_dir / "dev.jsonl")
        _save_jsonl(test,  out_dir / "test.jsonl")
    else:
        with open(out_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump(train, f, ensure_ascii=False, indent=2)
        with open(out_dir / "dev.json", "w", encoding="utf-8") as f:
            json.dump(dev, f, ensure_ascii=False, indent=2)
        with open(out_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump(test, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge Q&A pairs and write train/dev/test splits.")
    ap.add_argument("--security", default="processed/security_pairs.json", help="Path to security_pairs.json")
    ap.add_argument("--health",   default="processed/health_pairs.json",   help="Path to health_pairs.json")
    ap.add_argument("--out_dir",  default="data/processed", help="Output directory for splits")
    ap.add_argument("--format",   choices=["jsonl","json"], default="jsonl", help="Output file format")
    ap.add_argument("--alpaca",   action="store_true", help="Convert to Alpaca (instruction/input/output)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Holdout fraction for test+dev")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs_sources = []
    sec_path = Path(args.security)
    hlth_path = Path(args.health)

    if sec_path.exists():
        pairs_sources.append(load_pairs(sec_path))
    else:
        print(f"⚠️  Missing {sec_path}; continuing without security domain.")

    if hlth_path.exists():
        pairs_sources.append(load_pairs(hlth_path))
    else:
        print(f"⚠️  Missing {hlth_path}; continuing without health domain.")

    if not pairs_sources:
        raise FileNotFoundError("No input pair files found. Provide at least one of --security/--health.")

    merged_pairs = merge_and_clean(pairs_sources)

    if args.alpaca:
        data = to_alpaca_format(merged_pairs)
    else:
        # Ensure only the needed fields exist; preserve id/meta if present
        data = []
        for p in merged_pairs:
            row = {"prompt": p["prompt"], "response": p["response"]}
            if "id" in p: row["id"] = p["id"]
            if "meta" in p: row["meta"] = p["meta"]
            data.append(row)

    # Split into train/dev/test
    train, temp = train_test_split(data, test_size=args.test_size, random_state=args.seed)
    dev, test = train_test_split(temp, test_size=0.5, random_state=args.seed)

    out_dir = Path(args.out_dir)
    save_dataset(train, dev, test, out_dir, fmt=args.format)

    def _avg_len(key):
        vals = [len(x.get(key,"")) for x in (train+dev+test)]
        return sum(vals)/len(vals) if vals else 0.0

    print(f"✅ Wrote splits to {out_dir} as {args.format}.")
    print(f"Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")
    if args.alpaca:
        print(f"Avg instruction len: { _avg_len('instruction'):.1f }, Avg output len: { _avg_len('output'):.1f }")
    else:
        print(f"Avg prompt len: { _avg_len('prompt'):.1f }, Avg response len: { _avg_len('response'):.1f }")
