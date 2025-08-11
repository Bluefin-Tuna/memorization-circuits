#!/usr/bin/env bash
set -euo pipefail

# Auto-detect CUDA if available (unless DEVICE already exported) to avoid slow CPU fallback
DEVICE="${DEVICE:-$(python - <<'PY'
try:
	import torch
	print("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
	print("cpu")
PY
)}"

# Sanity check that this works
MODEL_NAME="qwen3-0.6b"
TASK="mmlu"
NUM_EXAMPLES=8
DIGITS=2
TOP_K=10
METHOD="gradient"   # use 'ig' with --steps if you want to test integrated gradients

echo "[INFO] Running quick local test: model=$MODEL_NAME task=$TASK examples=$NUM_EXAMPLES method=$METHOD top_k=$TOP_K device=$DEVICE"
python run_experiment.py \
  --model_names "$MODEL_NAME" \
  --tasks "$TASK" \
  --num_examples_list "$NUM_EXAMPLES" \
  --digits_list "$DIGITS" \
  --top_ks "$TOP_K" \
  --methods "$METHOD" \
  --device "$DEVICE" \
  --debug

echo "[DONE] Local test completed."
