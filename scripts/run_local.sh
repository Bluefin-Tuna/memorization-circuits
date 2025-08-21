#!/bin/bash
set -euo pipefail

DIGITS=2
NUM_EXAMPLES=10
DTYPE="bf16"
DEVICE="cpu"
OUT_DIR="results_local"

MODELS=(
    "qwen3-0.6b"
)
TASKS=(
    "ioi"
)
METHODS=("eap")
TOP_K_LIST_STR="5,10"
REUSE_THRESHOLDS="95,99,100"

mkdir -p "$OUT_DIR"
RUN_PREFIX="local_sweep_$(date +%Y%m%d_%H%M%S)"

for MODEL_NAME in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
                MODEL_SAFE=${MODEL_NAME//\//-}
                RUN_NAME="${RUN_PREFIX}_${TASK}_${METHOD}_${MODEL_SAFE}"

                echo "[RUNNING] Model: $MODEL_NAME, Task: $TASK, Method: $METHOD, TopKs: $TOP_K_LIST_STR, Reuse: $REUSE_THRESHOLDS, Digits: $DIGITS, Examples: $NUM_EXAMPLES"

                python run_experiment.py \
                    --model_name "$MODEL_NAME" \
                    --task "$TASK" \
                    --top_k_list "$TOP_K_LIST_STR" \
                    --reuse-thresholds "$REUSE_THRESHOLDS" \
                    --perm-trials 2000 \
                    --method "$METHOD" \
                    --digits "$DIGITS" \
                    --num_examples "$NUM_EXAMPLES" \
                    --dtype "$DTYPE" \
                    --device "$DEVICE" \
                    --run-name "$RUN_NAME" \
                    --output-dir "$OUT_DIR" \
                    --debug \
                || { echo "[ERROR] Failed for $RUN_NAME"; continue; }

                echo "[DONE] Completed: $RUN_NAME"
        done
    done
done

echo "[DONE] Sweep finished."