#!/bin/bash
set -euo pipefail

# Define search space
# models=("qwen3-0.6b")
models=("gpt2")
tasks=("addition")
top_ks=(50)
digits_list=(3)
num_examples_list=(100)

# Enumerate combinations
for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
      for topk in "${top_ks[@]}"; do
        for digits in "${digits_list[@]}"; do
          for numex in "${num_examples_list[@]}"; do
            python run_experiment.py \
                --model_names "$model" \
                --tasks "$task" \
                --top_ks "$topk" \
                --digits_list "$digits" \
                --num_examples_list "$numex" \
                --dtype bf16 \
                --device cpu \
                --run-name "local" \
                --output-dir results
          done
        done
      done
  done
done

echo "[DONE] Completed all experiments"