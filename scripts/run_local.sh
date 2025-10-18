#!/bin/bash
set -euo pipefail

# Configuration
MODEL="Qwen/Qwen2-VL-2B-Instruct"
DOMAIN="Logos"
HELD_OUT_DOMAIN="Chess"  # For ablation generalization test
NUM_EXAMPLES=20
NUM_PAIRS=10  # Number of clean/biased pairs for causal tracing
DEVICE="cpu"
DTYPE="auto"
MAX_NEW_TOKENS=5
TOP_K_HEADS=10  # Number of heads to ablate
TARGET_LAYERS="early"  # "early", "all", or comma-separated layer indices
OUT_DIR="results_full_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Full VLM Mechanistic Interpretability Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo "In-domain: $DOMAIN"
echo "Held-out domain: $HELD_OUT_DOMAIN"
echo "Examples: $NUM_EXAMPLES"
echo "Output: $OUT_DIR"
echo "=========================================="

# Step 1: Baseline evaluation (Experiment 0)
echo ""
echo "[1/5] Experiment 0: Baseline evaluation..."
python vlm_experiment.py \
    --model-name "$MODEL" \
    --split main \
    --domain "$DOMAIN" \
    --num-examples "$NUM_EXAMPLES" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output-dir "$OUT_DIR" \
|| { echo "[ERROR] Baseline evaluation failed"; exit 1; }

echo "[DONE] Baseline complete"

# Step 2: Create pairs for circuit analysis
echo ""
echo "[2/5] Creating clean/biased pairs for causal tracing..."
PAIRS_FILE="$OUT_DIR/pairs.json"
METRICS_FILE="$OUT_DIR/"*"__metrics.json"

python3 -c "
import json
import glob
from collections import defaultdict

metrics_files = glob.glob('$METRICS_FILE')
if not metrics_files:
    print('ERROR: No metrics file found')
    exit(1)

with open(metrics_files[0], 'r') as f:
    data = json.load(f)

details = data.get('baseline', {}).get('details', [])

# Group examples by resolution to create compatible pairs
# Example IDs like 'shoe_001_notitle_Q1_px384' where 'px384' is resolution
by_resolution = defaultdict(list)
for item in details:
    item_id = item['id']
    # Extract resolution suffix (everything after _px)
    if '_px' in item_id:
        resolution = item_id.split('_px')[1]
        by_resolution[resolution].append(item)
    else:
        # No resolution suffix, use default group
        by_resolution['default'].append(item)

# Create pairs within same resolution group
pairs = []
for resolution, items in sorted(by_resolution.items()):
    # Pair consecutive items from same resolution
    for i in range(0, len(items) - 1, 2):
        if len(pairs) >= $NUM_PAIRS:
            break
        pairs.append({
            'clean_id': items[i]['id'],
            'biased_id': items[i + 1]['id']
        })
    if len(pairs) >= $NUM_PAIRS:
        break

if len(pairs) == 0:
    print('ERROR: Could not create any pairs')
    exit(1)

with open('$PAIRS_FILE', 'w') as f:
    json.dump(pairs, f, indent=2)

print(f'Created {len(pairs)} resolution-matched pairs for causal tracing')
# Verify first pair
if pairs:
    p = pairs[0]
    clean_res = p['clean_id'].split('_px')[1] if '_px' in p['clean_id'] else 'default'
    biased_res = p['biased_id'].split('_px')[1] if '_px' in p['biased_id'] else 'default'
    match = '✓' if clean_res == biased_res else '✗'
    print(f'{match} Example: {p[\"clean_id\"]} <-> {p[\"biased_id\"]} (resolution: px{clean_res})')
" || { echo "[ERROR] Failed to create pairs file"; exit 1; }

# Step 3: Module-level causal tracing (Experiment 1)
echo ""
echo "[3/5] Experiment 1: Module-level causal tracing..."
python vlm_experiment.py \
    --model-name "$MODEL" \
    --split main \
    --domain "$DOMAIN" \
    --num-examples "$NUM_EXAMPLES" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --run-circuit \
    --pair-file "$PAIRS_FILE" \
    --target-layers "$TARGET_LAYERS" \
    --output-dir "$OUT_DIR" \
|| { echo "[ERROR] Circuit analysis failed"; exit 1; }

echo "[DONE] Module-level causal tracing complete"

# Step 4: Head-level discovery (Experiment 2)
echo ""
echo "[4/5] Experiment 2: Head-level discovery & ranking..."
python vlm_experiment.py \
    --model-name "$MODEL" \
    --split main \
    --domain "$DOMAIN" \
    --num-examples "$NUM_EXAMPLES" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --run-heads \
    --pair-file "$PAIRS_FILE" \
    --target-layers "$TARGET_LAYERS" \
    --output-dir "$OUT_DIR" \
|| { echo "[ERROR] Head analysis failed"; exit 1; }

echo "[DONE] Head-level discovery complete"

# Step 5: Ablation experiments (Experiment 2 continued + Experiment 4)
echo ""
echo "[5/5] Experiment 2-4: Ablation experiments..."
echo "  - Ablating top $TOP_K_HEADS heads"
echo "  - Testing on in-domain ($DOMAIN) and held-out ($HELD_OUT_DOMAIN)"
python vlm_experiment.py \
    --model-name "$MODEL" \
    --split main \
    --domain "$DOMAIN" \
    --num-examples "$NUM_EXAMPLES" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --run-ablation \
    --held-out-domain "$HELD_OUT_DOMAIN" \
    --top-k-heads "$TOP_K_HEADS" \
    --target-layers "$TARGET_LAYERS" \
    --output-dir "$OUT_DIR" \
|| { echo "[ERROR] Ablation experiments failed"; exit 1; }

echo "[DONE] Ablation experiments complete"

# Step 6: Generate all plots (Deliverables 1-4)
echo ""
echo "[6/6] Generating all deliverable figures..."
python vlm_experiment.py \
    --analysis \
    --output-dir "$OUT_DIR" \
|| { echo "[ERROR] Plot generation failed"; exit 1; }

echo "[DONE] All plots generated"

# Summary
echo ""
echo "=========================================="
echo "Full pipeline completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUT_DIR"
echo ""
echo "Deliverables:"
echo "  1. BAER/Accuracy plot: baer_accuracy.png"
echo "  2. Module effects heatmap: *__module_effects.png"
echo "  3. Head importance ranking: *__head_effects.png"
echo "  4. Ablation impact plot: *__ablation_results.png"
echo ""
echo "Generated files:"
ls -lh "$OUT_DIR"