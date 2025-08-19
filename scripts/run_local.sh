#!/bin/bash
set -euo pipefail

models=("qwen3-0.6b")
tasks=("ioi" "mcqa" "arc_easy" "arc_challenge") 
top_ks=(5)
digits_list=(2)
num_examples_list=(10) 

# Enumerate combinations
for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    for topk in "${top_ks[@]}"; do
      for digits in "${digits_list[@]}"; do
        for numex in "${num_examples_list[@]}"; do
          echo "---"
          echo "[RUNNING] Model: $model, Task: $task, TopK: $topk, Examples: $numex"
          
          python run_experiment.py \
            --model_names "$model" \
            --tasks "$task" \
            --top_ks "$topk" \
            --digits_list "$digits" \
            --num_examples_list "$numex" \
            --dtype bf16 \
            --device cpu \
            --run-name "local_test" \
            --output-dir results \
            --debug
        done
      done
    done
  done
done

echo ""
echo "[DONE] ✅ Completed all local test experiments."