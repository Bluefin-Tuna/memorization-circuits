# Understanding Memorization in VLMs

Minimal toolkit to localize and analyze memorization/bias circuits in vision-language models using activation patching on the VLMBias dataset.

What’s here:

- `vlm_experiment.py`: CLI to run causal analysis and/or baseline evaluation.
- `vlm_analysis/`: Helpers for dataset loading, evaluation, causal tracing, and plotting.

## Install

```bash
pip install -r requirements.txt
```

## Causal analysis (primary)

Run simple causal tracing by patching cross-attention-like modules using pairs of (clean, biased) images answering the same question.

```bash
python vlm_experiment.py \
  --model-name meta-llama/Llama-3.2-11B-Vision \
  --split main \
  --domain Logos \
  --num-examples 50 \
  --run-circuit --pair-file pairs.json \
  --output-dir results
```

`pairs.json` (list of objects):

```json
[
	{ "clean_id": "123", "biased_id": "456" },
	{ "clean_id": "789", "biased_id": "1011" }
]
```

Outputs:

- `results/<model>__<domain>__module_effects.json` (per-module effect scores)

## Optional: Baseline evaluation

Compute accuracy and BAER (bias-aligned error rate) on VLMBias.

```bash
python vlm_experiment.py \
  --model-name meta-llama/Llama-3.2-11B-Vision \
  --split main \
  --domain Logos \
  --num-examples 100 \
  --output-dir results
```

Outputs:

- `results/<model>__<domain>__metrics.json`

## Generate plots from results (no compute)

Use `--analysis` to scan `--output-dir` and write plots from existing JSONs.

```bash
python vlm_experiment.py --analysis --output-dir results
```

Writes:

- `results/baer_accuracy.png` (Accuracy vs BAER across runs)
- `results/<model>__<domain>__module_effects.png` (one per module effects JSON)

## Notes

- Models are loaded via `transformers`; some may require auth or specific revisions.
- `--device` and `--dtype` control compute; CPU fallback is automatic if CUDA is unavailable.
