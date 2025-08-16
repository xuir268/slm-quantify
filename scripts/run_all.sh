#!/usr/bin/env bash
set -euo pipefail

# -------- settings you can tweak --------
MODEL_ID="${MODEL_ID:-microsoft/phi-3-mini-4k-instruct}"
LORA_CFG="${LORA_CFG:-configs/phi3_lora.yaml}"
ENSEMBLE_CFG="${ENSEMBLE_CFG:-configs/ensemble.yaml}"
DATA_OUT="${DATA_OUT:-data/processed}"
TEST_PATH="${TEST_PATH:-data/processed/test.jsonl}"
METRICS_DIR="${METRICS_DIR:-results/metrics}"
FIG_DIR="${FIG_DIR:-results/figures}"
MAX_ITEMS="${MAX_ITEMS:-0}"           # 0 = use full test set
# ---------------------------------------

echo "== Step 0: env check =="
python -c "import torch, transformers, peft, datasets; print('✓ deps OK')" 2>/dev/null || {
  echo "Missing deps. Install: pip install 'transformers>=4.42' peft datasets accelerate evaluate sentencepiece faiss-cpu pyyaml tqdm pandas matplotlib scikit-learn"
  exit 1
}

echo "== Step 1: dataset =="
if [ ! -f "${DATA_OUT}/train.jsonl" ] || [ ! -f "${DATA_OUT}/dev.jsonl" ] || [ ! -f "${DATA_OUT}/test.jsonl" ]; then
  echo "No processed dataset found. Building from HF (CyberNative/Code_Vulnerability_Security_DPO)..."
  python scripts/make_security_sft_from_dpo.py --out_dir "${DATA_OUT}"
else
  echo "✓ Found processed dataset in ${DATA_OUT}"
fi

echo "== Step 2: (optional) build RAG index =="
# If you already have build_index.py wired to your corpus, uncomment the next line.
# python src/build_index.py

echo "== Step 3: train LoRA (${LORA_CFG}) =="
python src/train_lora.py --config "${LORA_CFG}"

echo "== Step 4: single-model eval (baseline / lora / rag+compress / lora+rag+compress) =="
python src/evaluate.py \
  --model_id "${MODEL_ID}" \
  --adapter_dir "$(python - <<'PY'
import json,sys,os
import pathlib as p
# read output_dir from YAML quickly
import yaml
cfg=p.Path(os.environ.get("LORA_CFG","configs/phi3_lora.yaml"))
with open(cfg,'r') as f: y=yaml.safe_load(f)
out_dir=y.get('output_dir','results/models/phi3_lora')
print(str(p.Path(out_dir)/'adapter'))
PY
)" \
  --test_path "${TEST_PATH}" \
  --out_dir "${METRICS_DIR}" \
  --max_items "${MAX_ITEMS}"

echo "== Step 5: n-SLM orchestrator eval (router/cascade/ensemble) =="
python scripts/eval_orchestrator.py \
  --cfg "${ENSEMBLE_CFG}" \
  --test_path "${TEST_PATH}" \
  --out_dir "${METRICS_DIR}" \
  --max_items "${MAX_ITEMS}"

echo "== Step 6: plots =="
python src/plotting.py

echo "✅ Done. Summaries in ${METRICS_DIR}, figures in ${FIG_DIR}"
