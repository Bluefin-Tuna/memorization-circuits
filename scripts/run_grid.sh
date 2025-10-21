#!/bin/bash
set -euo pipefail

# Array of models to evaluate
MODELS=(
    "meta-llama/Llama-3.2-11B-Vision"
    "Qwen/Qwen2-VL-2B-Instruct"
    "Qwen/Qwen2-VL-7B-Instruct"
    "google/paligemma-3-mix-448"
    "google/paligemma-3-mix-224"
)

# Configuration
DOMAINS=("Logos" "Chess" "Board_Games" "Anime" "Fashion")
HELD_OUT_DOMAIN="Chess"
NUM_EXAMPLES=500
NUM_PAIRS=100
TOP_K_HEADS=20
TARGET_LAYERS="early"
MAX_NEW_TOKENS=10
DEVICE="cuda"  # Change to "cuda" if running on GPU locally
DTYPE="bfloat16"  # Change to "bfloat16" if using GPU

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

        # Output directory with timestamp - includes domain name
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUT_DIR="results_full/$(echo $MODEL | tr '/' '_')_${DOMAIN}_${TIMESTAMP}"
        mkdir -p "${OUT_DIR}"

        echo "Output Directory: $OUT_DIR"

        # Determine if this is the held-out domain
        HELD_OUT_ARG=""
        if [ "$DOMAIN" != "$HELD_OUT_DOMAIN" ]; then
            HELD_OUT_ARG="--held-out-domain $HELD_OUT_DOMAIN"
        fi

        # Run full pipeline (baseline → circuit → heads → ablation)
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
            $HELD_OUT_ARG \
            --output-dir "$OUT_DIR" \
        || { echo "[ERROR] Pipeline failed for $DOMAIN"; continue; }

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
