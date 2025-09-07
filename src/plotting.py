import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json

# ---------- simple text normalization + F1 (for fallback when CSV lacks metrics) ----------
import re, string

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _f1(pred: str, gold: str) -> float:
    if not pred or not gold:
        return 0.0
    pt = _normalize_text(pred).split()
    gt = _normalize_text(gold).split()
    if not pt or not gt:
        return 0.0
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    prec = len(common) / len(pt)
    rec  = len(common) / len(gt)
    return 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))

METHOD_ORDER = [
    "baseline",
    "lora",
    "rag_compress",
    "lora_rag_compress",
    "orchestrator_router",
    "orchestrator_cascade",
    "orchestrator_ensemble",
]

CSV_FILES = {
    "baseline":               "baseline_summary.csv",
    "lora":                   "lora_summary.csv",
    "rag_compress":           "rag_compress_summary.csv",
    "lora_rag_compress":      "lora_rag_compress_summary.csv",
    "orchestrator_router":    "orchestrator_router_summary.csv",
    "orchestrator_cascade":   "orchestrator_cascade_summary.csv",
    "orchestrator_ensemble":  "orchestrator_ensemble_summary.csv",
}

JSONL_FILES = {
    "baseline":               "baseline.jsonl",
    "lora":                   "lora.jsonl",
    "rag_compress":           "rag_compress.jsonl",
    "lora_rag_compress":      "lora_rag_compress.jsonl",
    "orchestrator_router":    "orchestrator_router.jsonl",
    "orchestrator_cascade":   "orchestrator_cascade.jsonl",
    "orchestrator_ensemble":  "orchestrator_ensemble.jsonl",
}

COL_MAP = {
    "n": "n",
    "f1_avg": "F1",
    "em_avg": "EM",
    "prompt_tokens_avg": "PromptTokens",
    "output_tokens_avg": "OutputTokens",
    "latency_p50_ms": "LatencyP50",
    "latency_p95_ms": "LatencyP95",
}

def _summary_from_jsonl(path: Path):
    if not path.exists():
        return None
    rows = []
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    rows.append(r)
                except Exception:
                    pass
    except Exception:
        return None
    if not rows:
        return None
    # Collect metrics with flexible field names
    f1s, ems, ptoks, otoks, lats = [], [], [], [], []
    n = 0
    for r in rows:
        n += 1
        # tokens
        pt = r.get("prompt_tokens", r.get("ptoks", 0)) or 0
        ot = r.get("output_tokens", r.get("otoks", 0)) or 0
        ptoks.append(float(pt))
        otoks.append(float(ot))
        # latency
        lm = r.get("latency_ms", 0) or 0
        lats.append(float(lm))
        # f1/em: use provided, else compute from pred/gold or text/gold
        f1v = r.get("f1", None)
        emv = r.get("em", None)
        if f1v is None or emv is None:
            pred = r.get("pred", r.get("text", "")) or ""
            gold = r.get("gold", r.get("answer", "")) or ""
            f1v = _f1(pred, gold) if f1v is None else f1v
            emv = 1.0 if (gold and _normalize_text(pred) == _normalize_text(gold)) else 0.0 if emv is None else emv
        f1s.append(float(f1v))
        ems.append(float(emv))
    # helpers
    def _avg(xs): 
        return round(sum(xs)/len(xs), 4) if xs else 0.0
    def _p50(xs):
        if not xs: return 0.0
        xs_sorted = sorted(xs)
        mid = len(xs_sorted)//2
        return xs_sorted[mid] if len(xs_sorted)%2 else round((xs_sorted[mid-1]+xs_sorted[mid])/2, 1)
    def _p95(xs):
        if not xs: return 0.0
        xs_sorted = sorted(xs)
        idx = max(0, int(0.95*(len(xs_sorted)-1)))
        return xs_sorted[idx]
    return {
        "n": int(n),
        "F1": _avg(f1s),
        "EM": _avg(ems),
        "PromptTokens": _avg(ptoks),
        "OutputTokens": _avg(otoks),
        "LatencyP50": _p50(lats),
        "LatencyP95": _p95(lats),
    }

def load_summaries(metrics_dir: Path):
    rows = []
    for method, fname in CSV_FILES.items():
        f = metrics_dir / fname
        if f.exists():
            try:
                df = pd.read_csv(f)
                if not df.empty:
                    row = df.iloc[0].to_dict()
                    row = {COL_MAP.get(k, k): v for k, v in row.items()}
                    # If CSV lacks metrics (common for orchestrator in some runs), try JSONL fallback
                    need_fallback = False
                    for k in ["F1","EM","LatencyP50","LatencyP95","PromptTokens","OutputTokens"]:
                        if k not in row or (isinstance(row[k], (int,float)) and float(row[k]) == 0.0):
                            need_fallback = True
                            break
                    if need_fallback:
                        jf = JSONL_FILES.get(method)
                        if jf:
                            jpath = metrics_dir / jf
                            jrep = _summary_from_jsonl(jpath)
                            if jrep:
                                # prefer CSV 'n' if present and non-zero; otherwise use JSONL's
                                if not row.get("n"):
                                    row["n"] = jrep["n"]
                                # fill/overwrite zeros with JSONL-derived stats
                                for k in ["F1","EM","LatencyP50","LatencyP95","PromptTokens","OutputTokens"]:
                                    row[k] = jrep.get(k, row.get(k, 0.0))
                    row["Method"] = method
                    rows.append(row)
            except Exception as e:
                print(f"Skip {f}: {e}")
    if not rows:
        raise SystemExit(f"No summary CSVs found in {metrics_dir}")
    out = pd.DataFrame(rows)
    # Ensure all expected columns exist
    for c in ["Method","n","F1","EM","LatencyP50","LatencyP95","PromptTokens","OutputTokens"]:
        if c not in out.columns:
            out[c] = 0.0
    # Order methods
    out["__ord__"] = out["Method"].apply(lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 999)
    out = out.sort_values("__ord__").drop(columns="__ord__").reset_index(drop=True)
    return out

def save_table_png(df: pd.DataFrame, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.4*len(df)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    ax.set_title(title, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def bar_chart(df: pd.DataFrame, y_col: str, ylabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = range(len(df))
    ax.bar(xs, df[y_col].values)  # no explicit colors
    ax.set_xticks(list(xs))
    ax.set_xticklabels(df["Method"], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Method")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def stacked_tokens(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = range(len(df))
    p = ax.bar(xs, df["PromptTokens"].values)
    o = ax.bar(xs, df["OutputTokens"].values, bottom=df["PromptTokens"].values)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(df["Method"], rotation=30, ha="right")
    ax.set_ylabel("Tokens (avg per sample)")
    ax.set_title("Token Usage by Method (Prompt + Output)")
    ax.legend([p, o], ["Prompt", "Output"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def latency_bars(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = range(len(df))
    p50 = ax.bar(xs, df["LatencyP50"].values)
    # overlay p95 as markers
    ax.plot(list(xs), df["LatencyP95"].values, marker="o", linestyle="--")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(df["Method"], rotation=30, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency by Method (p50 bars, p95 line)")
    ax.legend([p50], ["p50"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def optional_latency_vs_f1(jsonl_path: Path, out_path: Path):
    # Per-example scatter (optional)
    if not jsonl_path.exists():
        print(f"[skip] no JSONL found at {jsonl_path}")
        return
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if "f1" in r and "latency_ms" in r:
                    rows.append({"F1": r["f1"] or 0.0, "Latency": r["latency_ms"], "Setting": r.get("setting","")})
            except Exception:
                pass
    if not rows:
        print(f"[skip] JSONL had no usable rows: {jsonl_path}")
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in sorted(df["Setting"].unique()):
        dd = df[df["Setting"] == s]
        ax.scatter(dd["Latency"], dd["F1"], label=s, s=12)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("F1")
    ax.set_title("Per-example Latency vs F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="results/metrics")
    ap.add_argument("--out_dir", default="results/plots")
    ap.add_argument("--jsonl_for_scatter", default="")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summaries(metrics_dir)
    # Reorder columns for readability
    col_order = ["Method","n","F1","EM","LatencyP50","LatencyP95","PromptTokens","OutputTokens"]
    df = df[col_order]
    # Console view
    try:
        import pandas as pd
        with pd.option_context("display.max_columns", None, "display.width", 120):
            print(df.to_string(index=False))
    except Exception:
        print(df)

    # Save table & plots
    save_table_png(df, out_dir / "summary_table.png", "Evaluation Summary")
    bar_chart(df, "F1", "F1", out_dir / "f1_by_method.png")
    bar_chart(df, "EM", "Exact Match", out_dir / "em_by_method.png")
    latency_bars(df, out_dir / "latency_by_method.png")
    stacked_tokens(df, out_dir / "tokens_by_method.png")

    # Optional scatter
    if args.jsonl_for_scatter:
        optional_latency_vs_f1(Path(args.jsonl_for_scatter), out_dir / "latency_vs_f1.png")

    print(f"✅ Plots saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
