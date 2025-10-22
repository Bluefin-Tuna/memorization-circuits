#!/bin/bash
set -euo pipefail

# Array of models to evaluate
MODELS=(
    "Qwen/Qwen3-VL-4B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
    "google/gemma-3-12b-it"
    "google/gemma-3-4b-it"
)
DOMAINS=(
    "Animals"
    "Logos"
    "National Flags"
    "Chess Pieces"
    "Board Games"
    "Optical Illusions"
    "Patterned Grids"
)
HELD_OUT_DOMAIN="Chess Pieces"
NUM_EXAMPLES=500
NUM_PAIRS=100
TOP_K_HEADS=20
TARGET_LAYERS="early"
MAX_NEW_TOKENS=10
DEVICE="cuda"
DTYPE="bfloat16"
LOAD_IN_8BIT="true"  # Use 8-bit quantization to save memory

# Accept optional model index argument for running single model
MODEL_INDEX=${1:-}

echo "=========================================="
echo "VLM Grid Experiment (Local)"
echo "=========================================="
echo "Models: ${#MODELS[@]}"
echo "Domains: ${DOMAINS[@]}"
echo "Examples per domain: $NUM_EXAMPLES"
echo "Device: $DEVICE"
echo "=========================================="

# Function to run full pipeline for a single model
run_model_experiment() {
    local MODEL=$1
    local MODEL_INDEX=$2

    echo ""
    echo "=========================================="
    echo "Model [$((MODEL_INDEX+1))/${#MODELS[@]}]: $MODEL"
    echo "=========================================="

    # Create necessary directories
    mkdir -p logs

    # Run the full pipeline for all domains
    for DOMAIN in "${DOMAINS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Processing Domain: $DOMAIN"
        echo "=========================================="

        # Output directory with timestamp - includes domain name (spaces replaced with underscores)
        # Save to network volume at /workspace
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        DOMAIN_CLEAN=$(echo "$DOMAIN" | tr ' ' '_')
        OUT_DIR="/workspace/results_full/$(echo $MODEL | tr '/' '_')_${DOMAIN_CLEAN}_${TIMESTAMP}"
        mkdir -p "${OUT_DIR}"

        echo "Output Directory: $OUT_DIR"

        # Run full pipeline (baseline → circuit → heads → ablation)
        # Build command with conditional held-out domain argument
        if [ "$DOMAIN" != "$HELD_OUT_DOMAIN" ]; then
            python vlm_experiment.py \
                --model-name "$MODEL" \
                --split main \
                --domain "$DOMAIN" \
                --num-examples "$NUM_EXAMPLES" \
                --device "$DEVICE" \
                --dtype "$DTYPE" \
                --max-new-tokens "$MAX_NEW_TOKENS" \
                --run-full-pipeline \
                --num-pairs "$NUM_PAIRS" \
                --top-k-heads "$TOP_K_HEADS" \
                --target-layers "$TARGET_LAYERS" \
                --held-out-domain "$HELD_OUT_DOMAIN" \
                --output-dir "$OUT_DIR" \
            || { echo "[ERROR] Pipeline failed for $DOMAIN"; continue; }
        else
            python vlm_experiment.py \
                --model-name "$MODEL" \
                --split main \
                --domain "$DOMAIN" \
                --num-examples "$NUM_EXAMPLES" \
                --device "$DEVICE" \
                --dtype "$DTYPE" \
                --max-new-tokens "$MAX_NEW_TOKENS" \
                --run-full-pipeline \
                --num-pairs "$NUM_PAIRS" \
                --top-k-heads "$TOP_K_HEADS" \
                --target-layers "$TARGET_LAYERS" \
                --output-dir "$OUT_DIR" \
            || { echo "[ERROR] Pipeline failed for $DOMAIN"; continue; }
        fi

        # Generate analysis for this domain
        echo ""
        echo "Generating analysis for $DOMAIN..."
        python vlm_experiment.py \
            --analysis \
            --output-dir "$OUT_DIR" \
        || echo "[WARNING] Analysis encountered errors for $DOMAIN"

        # Create summary report for this domain
        echo ""
        echo "Creating summary report for $DOMAIN..."
        python vlm_experiment.py \
            --create-summary \
            --output-dir "$OUT_DIR"

        # Compress results for this domain
        tar -czf "${OUT_DIR}.tar.gz" "$OUT_DIR"
        echo "Compressed archive: ${OUT_DIR}.tar.gz"

        echo "[DONE] Domain $DOMAIN complete!"
    done

    echo ""
    echo "=========================================="
    echo "Model Complete: $MODEL"
    echo "=========================================="

    # Clean up model cache to save disk space
    echo ""
    echo "Cleaning up model cache for $MODEL..."
    python3 -c "
import shutil
from pathlib import Path
import os

# HuggingFace cache directory (default home directory)
cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'

if cache_dir.exists():
    model_name = '$MODEL'
    # Convert model name to cache format (e.g., 'Qwen/Qwen3-VL-4B' -> 'models--Qwen--Qwen3-VL-4B')
    cache_name = 'models--' + model_name.replace('/', '--')

    # Find and remove all matching cache directories
    removed_size = 0
    for item in cache_dir.iterdir():
        if cache_name in item.name:
            try:
                # Get size before deletion
                if item.is_dir():
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    removed_size += size
                    shutil.rmtree(item)
                    print(f'Removed: {item.name} ({size / 1024**3:.2f} GB)')
            except Exception as e:
                print(f'Warning: Could not remove {item.name}: {e}')

    if removed_size > 0:
        print(f'Total space freed: {removed_size / 1024**3:.2f} GB')
    else:
        print(f'No cache found for {model_name}')
else:
    print('HuggingFace cache directory not found')
" || echo "[WARNING] Model cleanup failed, continuing..."

    echo "[DONE] Model cleanup complete"
}

# Run experiments
if [ -n "$MODEL_INDEX" ]; then
    # Run single model if index provided
    if [ "$MODEL_INDEX" -ge 0 ] && [ "$MODEL_INDEX" -lt "${#MODELS[@]}" ]; then
        run_model_experiment "${MODELS[$MODEL_INDEX]}" "$MODEL_INDEX"
    else
        echo "ERROR: Invalid model index. Must be 0-$((${#MODELS[@]}-1))"
        exit 1
    fi
else
    # Run all models sequentially
    for i in "${!MODELS[@]}"; do
        run_model_experiment "${MODELS[$i]}" "$i" || {
            echo "[ERROR] Model ${MODELS[$i]} failed, continuing to next model..."
            continue
        }
    done
fi

echo ""
echo "=========================================="
echo "All Experiments Complete!"
echo "=========================================="
echo "Results saved in: results_full/"
