#!/usr/bin/env bash
set -euo pipefail

# Sanity check that this works
MODEL_NAME="qwen3-0.6b"
TASK="boolean"
NUM_EXAMPLES=8
DIGITS=2
TOP_K=200
METHOD="gradient"   # use 'ig' with --steps if you want to test integrated gradients
DEVICE="cpu"

echo "[INFO] Running quick local test: model=$MODEL_NAME task=$TASK examples=$NUM_EXAMPLES method=$METHOD top_k=$TOP_K"
python run_experiment.py \
  --model_name "$MODEL_NAME" \
  --task "$TASK" \
  --num_examples "$NUM_EXAMPLES" \
  --digits "$DIGITS" \
  --top_k "$TOP_K" \
  --method "$METHOD" \
  --device "$DEVICE" \
  --debug

echo "[DONE] Local test completed."
