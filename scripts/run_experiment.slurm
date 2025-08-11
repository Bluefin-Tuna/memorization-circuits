#!/bin/bash
# Array SLURM script for circuit reuse experiments (up to 8 jobs)
# Customize paths (LOG_DIR, CACHE paths) to your cluster layout.

#SBATCH --job-name=circuit_reuse
#SBATCH --output=/home/ml6/logs/sbatch/circuit_reuse_%A_%a.out
#SBATCH --error=/home/ml6/logs/sbatch/circuit_reuse_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem=96G
#SBATCH --mail-user=ml6@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --array=0-7
#SBATCH --partition=preempt

set -euo pipefail

# ---- Paths & caches (edit as needed) ----
export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=0
LOG_DIR=/home/ml6/logs/sbatch
mkdir -p "$LOG_DIR" "$HF_HOME"

# ---- Conda env ----
eval "$(conda shell.bash hook)"
conda activate circuit_reuse

# ---- Parameter grids ----
models=("gpt2" "gpt2-medium")
tasks=("addition" "boolean" "mmlu" "mmlu_real")
methods=("gradient" "ig")
top_ks=(5 10)
digits_list=(2)
num_examples_list=(100)
steps_for_ig=10

# Build a list of all (model,task,method,top_k) combos; constrain total
combo_index=0
declare -a model_combo task_combo method_combo topk_combo
for m in "${models[@]}"; do
    for t in "${tasks[@]}"; do
        for meth in "${methods[@]}"; do
            for k in "${top_ks[@]}"; do
                model_combo[$combo_index]="$m"
                task_combo[$combo_index]="$t"
                method_combo[$combo_index]="$meth"
                topk_combo[$combo_index]="$k"
                combo_index=$((combo_index+1))
            done
        done
    done
done

total=${#model_combo[@]}
idx=$SLURM_ARRAY_TASK_ID
if (( idx >= total )); then
    echo "Array index $idx exceeds combo count $total; exiting." >&2
    exit 0
fi

MODEL_NAME=${model_combo[$idx]}
TASK=${task_combo[$idx]}
METHOD=${method_combo[$idx]}
TOP_K=${topk_combo[$idx]}
NUM_EXAMPLES=${num_examples_list[0]}
DIGITS=${digits_list[0]}
STEPS=$steps_for_ig

echo "[INFO] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -> model=$MODEL_NAME task=$TASK method=$METHOD top_k=$TOP_K"
echo "[INFO] Using digits=$DIGITS num_examples=$NUM_EXAMPLES steps=$STEPS"

EXTRA_ARGS=()
if [[ "$METHOD" == "ig" ]]; then
    EXTRA_ARGS+=(--steps "$STEPS")
fi

python -m circuit_reuse.run_experiment \
    --model_name "$MODEL_NAME" \
    --task "$TASK" \
    --num_examples "$NUM_EXAMPLES" \
    --digits "$DIGITS" \
    --top_k "$TOP_K" \
    --method "$METHOD" \
    "${EXTRA_ARGS[@]}" \
    --device cuda

echo "[DONE] Job $SLURM_JOB_ID index $SLURM_ARRAY_TASK_ID finished." 