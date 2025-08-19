#!/bin/bash
set -euo pipefail

DIGITS=2
NUM_EXAMPLES=10
DTYPE="bf16"
DEVICE="cpu"
OUT_DIR="results"

MODELS=(
    "qwen3-0.6b"
)
TASKS=(
    "ioi"
)
METHODS=(
    "eap"
)
TOP_K_LIST=(
    5
)

mkdir -p "$OUT_DIR"
RUN_PREFIX="local_sweep_$(date +%Y%m%d_%H%M%S)"

for MODEL_NAME in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            for TOP_K in "${TOP_K_LIST[@]}"; do
                MODEL_SAFE=${MODEL_NAME//\//-}
                RUN_NAME="${RUN_PREFIX}_${TASK}_${METHOD}_k${TOP_K}_${MODEL_SAFE}"

                echo "[RUNNING] Model: $MODEL_NAME, Task: $TASK, Method: $METHOD, TopK: $TOP_K, Digits: $DIGITS, Examples: $NUM_EXAMPLES"

                python run_experiment.py \
                    --model_name "$MODEL_NAME" \
                    --task "$TASK" \
                    --top_k "$TOP_K" \
                    --method "$METHOD" \
                    --digits "$DIGITS" \
                    --num_examples "$NUM_EXAMPLES" \
                    --dtype "$DTYPE" \
                    --device "$DEVICE" \
                    --run-name "$RUN_NAME" \
                    --output-dir "$OUT_DIR/$RUN_NAME" \
                    --debug \
                || { echo "[ERROR] Failed for $RUN_NAME"; continue; }

                echo "[DONE] Completed: $RUN_NAME"
            done
        done
    done
done

echo "[DONE] Sweep finished."