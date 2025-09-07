
#!/usr/bin/env bash
set -euo pipefail

# ===========================
# n‑SLM pipeline runner (Colab/Mac/Linux)
# - Builds security dataset if missing
# - Trains LoRA/QLoRA from YAML
# - Runs single‑model eval
# - Runs orchestrator (router/cascade/ensemble)
# - Plots summaries
# Configure via env vars or edit defaults below.
# ===========================

# -------- settings you can tweak (env overrides allowed) --------
MODEL_ID="${MODEL_ID:-microsoft/phi-3-mini-4k-instruct}"
LORA_CFG="${LORA_CFG:-configs/phi3_lora.yaml}"
ENSEMBLE_CFG="${ENSEMBLE_CFG:-configs/ensemble.yaml}"

DATA_OUT="${DATA_OUT:-data/processed}"
TEST_PATH="${TEST_PATH:-data/processed/test.jsonl}"

METRICS_DIR="${METRICS_DIR:-results/metrics}"
FIG_DIR="${FIG_DIR:-results/figures}"

MAX_ITEMS="${MAX_ITEMS:-0}"                 # 0 = full test set
ORCH_ONLY_MODE="${ORCH_ONLY_MODE:-}"        # set to router|cascade|ensemble to force a single mode in eval_orchestrator (if supported)

# Skips (set to 1 to skip step)
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-0}"
SKIP_ORCH="${SKIP_ORCH:-0}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"

# Auto-install deps if missing (useful on Colab)
AUTO_INSTALL="${AUTO_INSTALL:-1}"
# Zip adapters after training (saved alongside results/)
ZIP_ADAPTER="${ZIP_ADAPTER:-0}"
# ---------------------------------------------------------------

echo "== Step 0: env check =="

# Fast HF downloads + tokenizer parallel
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=true

# If running on Apple MPS, avoid watermark errors
python - <<'PY' || true
import os, torch
if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]="0.0"
    print("ℹ️  MPS detected: set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
else:
    pass
PY

# Dep check (and optional auto-install)
if python - <<'PY' 2>/dev/null; then
import transformers, peft, datasets, sentencepiece, yaml, sklearn, pandas, matplotlib
print("✓ deps OK")
PY
then
  echo "✓ deps OK"
else
  echo "⚠️  Missing deps."
  if [[ "${AUTO_INSTALL}" == "1" ]]; then
    echo "→ Installing common packages (this may take a minute)..."
    python -m pip install -q "transformers<5" peft accelerate datasets evaluate sentencepiece faiss-cpu \
         pyyaml tqdm pandas matplotlib scikit-learn hf_transfer bitsandbytes || {
      echo "❌ pip install failed. Please install dependencies manually."; exit 1; }
  else
    echo "Install: pip install 'transformers<5' peft accelerate datasets evaluate sentencepiece faiss-cpu pyyaml tqdm pandas matplotlib scikit-learn hf_transfer bitsandbytes"
    exit 1
  fi
fi

# Show device summary
python - <<'PY'
import torch
cuda = torch.cuda.is_available()
mps  = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
dev  = "cuda" if cuda else ("mps" if mps else "cpu")
cap  = torch.cuda.get_device_name(0) if cuda else ("Apple MPS" if mps else "CPU")
print(f"ℹ️  Device: {dev} | {cap}")
PY

# ---------------------------
# Step 1: dataset (security‑only)
# ---------------------------
if [[ "$SKIP_DATA" == "1" ]]; then
  echo "== Step 1: dataset == (skipped)"
else
  echo "== Step 1: dataset =="
  if [ ! -f "${DATA_OUT}/train.jsonl" ] || [ ! -f "${DATA_OUT}/dev.jsonl" ] || [ ! -f "${DATA_OUT}/test.jsonl" ]; then
    echo "No processed dataset found. Building from HF (CyberNative/Code_Vulnerability_Security_DPO)..."
    python scripts/make_security_sft_from_dpo.py --out_dir "${DATA_OUT}"
  else
    echo "✓ Found processed dataset in ${DATA_OUT}"
  fi
fi

# ---------------------------
# Step 2: (optional) build RAG index
# ---------------------------
echo "== Step 2: (optional) build RAG index =="
# If you already have build_index.py wired to your corpus, uncomment the next line.
# python src/build_index.py

# ---------------------------
# Step 3: train LoRA/QLoRA
# ---------------------------
if [[ "$SKIP_TRAIN" == "1" ]]; then
  echo "== Step 3: train LoRA == (skipped)"
else
  echo "== Step 3: train LoRA (${LORA_CFG}) =="
  # Warn if use_qlora on non‑CUDA
  python - <<'PY'
import os, yaml, torch, sys
cfg_path = os.environ.get("LORA_CFG","configs/phi3_lora.yaml")
cfg = yaml.safe_load(open(cfg_path,'r'))
if cfg.get("use_qlora", False) and not torch.cuda.is_available():
    print("⚠️  use_qlora=True but CUDA not available. This will fail. Set use_qlora=False or run on CUDA.")
PY
  python src/train_lora.py --config "${LORA_CFG}"
fi

# Adapter dir from YAML (robust)
ADAPTER_DIR="$(python - <<'PY'
import json,sys,os
import pathlib as p
import yaml
cfg=p.Path(os.environ.get("LORA_CFG","configs/phi3_lora.yaml"))
y=yaml.safe_load(open(cfg,'r'))
out_dir=y.get('output_dir','results/models/phi3_lora')
print(str(p.Path(out_dir)/'adapter'))
PY
)"

# ---------------------------
# Step 4: single‑model eval
# ---------------------------
if [[ "$SKIP_BASE_EVAL" == "1" ]]; then
  echo "== Step 4: single-model eval == (skipped)"
else
  echo "== Step 4: single-model eval (baseline / lora / rag+compress / lora+rag+compress) =="
  python src/evaluate.py \
    --model_id "${MODEL_ID}" \
    --adapter_dir "${ADAPTER_DIR}" \
    --test_path "${TEST_PATH}" \
    --out_dir "${METRICS_DIR}" \
    --max_items "${MAX_ITEMS}"
fi

# ---------------------------
# Step 5: orchestrator eval
# ---------------------------
if [[ "$SKIP_ORCH" == "1" ]]; then
  echo "== Step 5: n‑SLM orchestrator eval == (skipped)"
else
  echo "== Step 5: n‑SLM orchestrator eval (router/cascade/ensemble) =="
  if [[ -n "${ORCH_ONLY_MODE}" ]]; then
    # Run a single mode if the script supports it
    python scripts/eval_orchestrator.py \
      --cfg "${ENSEMBLE_CFG}" \
      --test_path "${TEST_PATH}" \
      --out_dir "${METRICS_DIR}" \
      --max_items "${MAX_ITEMS}" \
      --mode "${ORCH_ONLY_MODE}" || true
  else
    python scripts/eval_orchestrator.py \
      --cfg "${ENSEMBLE_CFG}" \
      --test_path "${TEST_PATH}" \
      --out_dir "${METRICS_DIR}" \
      --max_items "${MAX_ITEMS}"
  fi
fi

# ---------------------------
# Step 6: plots
# ---------------------------
if [[ "$SKIP_PLOTS" == "1" ]]; then
  echo "== Step 6: plots == (skipped)"
else
  echo "== Step 6: plots =="
  python src/plotting.py || echo "⚠️  plotting failed (non‑fatal)"
fi

# Optional zip of adapter (useful on Colab)
if [[ "$ZIP_ADAPTER" == "1" ]]; then
  base_zip="${ADAPTER_DIR%/adapter}_adapter.zip"
  echo "Zipping adapter to ${base_zip}"
  (cd "$(dirname "${ADAPTER_DIR}")" && zip -qr "$(basename "${base_zip}")" adapter) || true
fi

echo "✅ Done."
echo "Summaries → ${METRICS_DIR}"
echo "Figures   → ${FIG_DIR}"
