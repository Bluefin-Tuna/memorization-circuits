# Circuit Reuse Experiments

This repo measures per-example circuits and shared reuse across examples, then ablates the shared circuit vs a size-matched random control to assess causal importance.

High-level flow:

1. Generate a dataset of prompt/target pairs for a task.
2. Attribute component importance (attention heads / MLPs) per example.
3. For each top-K and reuse threshold p, build a shared circuit from components present in at least p% of example circuits.
4. Ablate the shared vs a size-matched random control; measure accuracy drop and permutation significance.

Outputs are saved per run into a folder with:

- metrics.json – aggregated metrics by top-K and threshold
- attributions_train.jsonl – per-example ranked components and scores

## Quick start

```bash
pip install -r requirements.txt
python run_experiment.py \
	--model_name "qwen3-0.6b" \
	--task "ioi" \
	--num_examples 50 \
	--top_k_list "5,10" \
	--reuse-thresholds "95,99,100" \
	--perm-trials 2000 \
	--dtype bf16 --device cpu --debug
```

Then plot:

```bash
python plot_results2.py --results-dir results
```

This writes:

- aggregated_by_task_model_k.csv – flat metrics table
- multiplot_reuse_vs_threshold.png – reuse@p vs p
- multiplot_lift_vs_threshold.png – knockout diff vs p
- multiplot_perm_neglog10p_vs_threshold.png – permutation significance (−log10 p) vs p
- permutation_summary.tex – compact LaTeX table of permutation p-values and observed diffs

## Key CLI args

- --top_k_list: Comma-separated per-example top-K values (e.g., "5,10,25,50").
- --reuse-thresholds: Comma-separated thresholds p as percentages (e.g., "95,96,97,98,99,100").
- --perm-trials: Trials for paired permutation tests between shared vs control ablations.

## Notes

- Attribution methods supported: eap, gradient.
- Results folders under results/, results_pretraining/ have plots and CSVs for analysis.

## References

- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- eap-ig: https://github.com/hannamw/eap-ig

## License

MIT
