# Understanding Memorization in VLMs

Complete toolkit to localize and analyze memorization/bias circuits in vision-language models using activation patching on the VLMBias dataset.

**Implements full mechanistic interpretability pipeline:**

- Module-level causal tracing (Experiment 1)
- Head-level circuit discovery (Experiment 2)
- Ablation experiments with generalization tests (Experiments 2-4)
- All required deliverable figures

## What's here

- `vlm_experiment.py`: CLI to run full experimental pipeline
- `vlm_analysis/`: Helpers for dataset loading, evaluation, causal tracing, head analysis, ablation, and plotting
- `scripts/run_local.sh`: Automated pipeline executing all experiments

## Install

```bash
pip install -r requirements.txt
```

## Quick Start: Full Pipeline

Run the complete experimental pipeline (baseline → module tracing → head discovery → ablation):

```bash
./scripts/run_local.sh
```

This executes:

1. **Experiment 0:** Baseline evaluation (accuracy, BAER)
2. **Experiment 1:** Module-level causal tracing across vision blocks, merger, and early LM layers
3. **Experiment 2:** Head-level discovery and ranking
4. **Experiments 2-4:** Ablation of top-K heads on in-domain and held-out domains
5. **Analysis:** Generate all deliverable figures

## Individual Experiments

### Experiment 0: Baseline Evaluation

Compute accuracy and BAER (bias-aligned error rate) on VLMBias.

```bash
python vlm_experiment.py \
  --model-name Qwen/Qwen2-VL-2B-Instruct \
  --split main \
  --domain Logos \
  --num-examples 100 \
  --output-dir results
```

**Outputs:** `results/<model>__<domain>__metrics.json`

### Experiment 1: Module-Level Causal Tracing

Patch vision blocks, merger, and language model attention layers to identify which modules causally contribute to bias.

```bash
python vlm_experiment.py \
  --model-name Qwen/Qwen2-VL-2B-Instruct \
  --split main \
  --domain Logos \
  --num-examples 100 \
  --run-circuit \
  --pair-file pairs.json \
  --target-layers early \
  --output-dir results
```

**`pairs.json` format:**

```json
[
	{ "clean_id": "123", "biased_id": "456" },
	{ "clean_id": "789", "biased_id": "1011" }
]
```

**Outputs:** `results/<model>__<domain>__module_effects.json`

### Experiment 2: Head-Level Discovery

Analyze individual attention heads within target modules to identify minimal bias circuit.

```bash
python vlm_experiment.py \
  --model-name Qwen/Qwen2-VL-2B-Instruct \
  --split main \
  --domain Logos \
  --num-examples 100 \
  --run-heads \
  --pair-file pairs.json \
  --target-layers early \
  --output-dir results
```

**Outputs:** `results/<model>__<domain>__head_effects.json`

### Experiments 2-4: Ablation & Generalization

Ablate top-K heads and measure BAER on in-domain and held-out domains.

```bash
python vlm_experiment.py \
  --model-name Qwen/Qwen2-VL-2B-Instruct \
  --split main \
  --domain Logos \
  --num-examples 100 \
  --run-ablation \
  --held-out-domain Chess \
  --top-k-heads 10 \
  --target-layers early \
  --output-dir results
```

**Outputs:** `results/<model>__<domain>__ablation_results.json`

### Generate Plots (All Deliverables)

Use `--analysis` to scan `--output-dir` and generate all figures from existing JSONs.

```bash
python vlm_experiment.py --analysis --output-dir results
```

**Generates:**

1. `baer_accuracy.png` – Accuracy vs BAER across models/domains (Deliverable 1)
2. `*__module_effects.png` – Module-by-layer causal effect heatmap (Deliverable 2)
3. `*__head_effects.png` – Head importance scree plot (Deliverable 3)
4. `*__ablation_results.png` – BAER before/after ablation on in-domain & held-out (Deliverable 4)

## Configuration Options

- `--target-layers`: Which LM layers to analyze (`early` = first 8, `all`, or comma-separated indices)
- `--top-k-heads`: Number of heads to ablate (default: 10)
- `--held-out-domain`: Domain for generalization test (e.g., Chess, Board Games)
- `--device`: Computation device (`cuda`, `cpu`)
- `--dtype`: Model precision (`float16`, `float32`, `bfloat16`, `auto`)

## Architecture Notes

**Qwen2-VL** is a decoder-only VLM without explicit cross-attention. Visual tokens from the vision encoder are merged and concatenated with language tokens, then the language model's self-attention handles cross-modal interaction. The toolkit targets:

- **Vision blocks** (32 layers): Process image patches
- **Merger**: Aggregates visual tokens for language model
- **Language model attention** (early layers): Where visual info integrates with language priors

For other VLMs with explicit cross-attention (LLaVA, BLIP-2), the module filter can be adapted via the `module_filter` parameter to `CircuitAnalyzer`.

## Notes

- Models are loaded via `transformers`; some may require authentication or specific revisions
- CUDA is used if available; CPU fallback is automatic
- The toolkit supports any HuggingFace VLM with minor filter adjustments
