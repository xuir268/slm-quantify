import os, json, csv
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

def load_summary_csv(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    if not r:
        return {}
    row = r[0]
    # coerce to float
    out = {k: float(v) for k, v in row.items() if k != ""}
    return out

def find_summaries(metrics_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Looks for files like:
      baseline_summary.csv
      lora_summary.csv
      rag_compress_summary.csv
      lora_rag_compress_summary.csv
      (and any orchestrator_*.csv you add later)
    Returns: {setting_name: summary_dict}
    """
    results = {}
    for p in metrics_dir.glob("*_summary.csv"):
        name = p.stem.replace("_summary", "")  # e.g., "baseline"
        results[name] = load_summary_csv(p)
    return results

# ----------------- plotting helpers -----------------

def _bar(ax, labels, values, title, ylabel):
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(labels, rotation=15, ha="right")

def plot_accuracy(results: Dict[str, Dict[str, float]], out: Path):
    labels = list(results.keys())
    f1 = [results[k].get("f1_avg", 0.0) for k in labels]
    em = [results[k].get("em_avg", 0.0) for k in labels]

    plt.figure(figsize=(8,5))
    ax = plt.gca()
    x = range(len(labels))
    ax.bar([i-0.2 for i in x], f1, width=0.4, label="F1")
    ax.bar([i+0.2 for i in x], em, width=0.4, label="EM")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Accuracy by Setting")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "accuracy_by_setting.png", dpi=180)
    plt.close()

def plot_tokens(results: Dict[str, Dict[str, float]], out: Path):
    labels = list(results.keys())
    pt = [results[k].get("prompt_tokens_avg", 0.0) for k in labels]
    ot = [results[k].get("output_tokens_avg", 0.0) for k in labels]

    plt.figure(figsize=(8,5))
    ax = plt.gca()
    x = range(len(labels))
    ax.bar([i-0.2 for i in x], pt, width=0.4, label="Prompt tokens (avg)")
    ax.bar([i+0.2 for i in x], ot,   width=0.4, label="Output tokens (avg)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Tokens")
    ax.set_title("Token Usage by Setting")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "tokens_by_setting.png", dpi=180)
    plt.close()

def plot_latency(results: Dict[str, Dict[str, float]], out: Path):
    labels = list(results.keys())
    p50 = [results[k].get("latency_p50_ms", 0.0) for k in labels]
    p95 = [results[k].get("latency_p95_ms", 0.0) for k in labels]

    plt.figure(figsize=(8,5))
    ax = plt.gca()
    x = range(len(labels))
    ax.bar([i-0.2 for i in x], p50, width=0.4, label="p50 (ms)")
    ax.bar([i+0.2 for i in x], p95, width=0.4, label="p95 (ms)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Milliseconds")
    ax.set_title("Latency by Setting")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "latency_by_setting.png", dpi=180)
    plt.close()

def main(metrics_dir="results/metrics", out_dir="results/figures"):
    metrics_dir = Path(metrics_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    results = find_summaries(metrics_dir)
    if not results:
        print(f"No *_summary.csv found in {metrics_dir}")
        return
    plot_accuracy(results, out)
    plot_tokens(results, out)
    plot_latency(results, out)
    print(f"✅ Saved plots to {out.resolve()}")

if __name__ == "__main__":
    main()
