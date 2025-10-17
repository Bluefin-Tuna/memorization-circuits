from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional
import torch

from vlm_analysis.dataset import load_vlmbias, VLMExample
from vlm_analysis.models import load_vlm_model
from vlm_analysis.evaluate import evaluate_dataset
from vlm_analysis.plotting import plot_baer_accuracy, plot_module_heatmap
from vlm_analysis.circuit import CircuitAnalyzer


def _parse_args():
    p = argparse.ArgumentParser(description="Run VLMBias experiments for a given model.")
    p.add_argument(
        "--model-name",
        type=str,
        required=False,
        default=None,
        help="HuggingFace identifier of the VLM model (required unless --analysis is set).",
    )
    p.add_argument("--split", type=str, default="main", help="Dataset split to use (default: main).")
    p.add_argument("--domain", type=str, default=None, help="Optional domain/topic filter (e.g., Logos, Chess).")
    p.add_argument("--num-examples", type=int, default=None, help="Number of examples to evaluate (default: all).")
    p.add_argument("--device", type=str, default=None, help="Computation device (e.g., cuda, cpu). Default selects automatically.")
    p.add_argument("--dtype", type=str, default=None, choices=["float16", "float32", "bfloat16", "auto"], help="Model dtype.")
    p.add_argument("--revision", type=str, default=None, help="Model revision or tag.")
    p.add_argument("--output-dir", type=str, default="results", help="Directory to save metrics and plots.")
    p.add_argument("--run-circuit", action="store_true", help="Perform causal tracing over cross-attention modules.")
    p.add_argument(
        "--pair-file",
        type=str,
        default=None,
        help="JSON file mapping clean/biased example IDs for circuit analysis.",
    )
    p.add_argument("--max-new-tokens", type=int, default=5, help="Max new tokens to generate per example.")
    p.add_argument(
        "--analysis",
        action="store_true",
        help="Analysis-only mode: generate plots from existing JSON results instead of running experiments.",
    )
    return p.parse_args()


def _run_analysis(output_dir: Path):
    """Generate plots from previously saved JSON results.

    This scans the output directory for "*__metrics.json" and
    "*__module_effects.json" files, aggregates summary metrics across
    models/domains, and emits plots alongside the JSON files.
    """
    metrics_entries = []
    # Collect metrics across all runs
    for metrics_path in output_dir.glob("*__metrics.json"):
        try:
            with metrics_path.open("r") as f:
                data = json.load(f)
            baseline = data.get("baseline", {})
            entry = {
                "model": data.get("model", metrics_path.stem.split("__")[0]),
                "domain": data.get("domain", "all"),
                "accuracy": baseline.get("accuracy", 0.0),
                "baer": baseline.get("baer", 0.0),
            }
            metrics_entries.append(entry)
        except Exception:
            continue

    # Plot BAER vs Accuracy if we found any metrics
    if metrics_entries:
        baer_plot_path = output_dir / "baer_accuracy.png"
        plot_baer_accuracy(metrics_entries, str(baer_plot_path))
        print(f"[ANALYSIS] Wrote BAER/Accuracy plot -> {baer_plot_path}")
    else:
        print(f"[ANALYSIS] No metrics JSON files found in {output_dir}")

    # For each module effects file, emit a heatmap
    for effects_path in output_dir.glob("*__module_effects.json"):
        try:
            with effects_path.open("r") as f:
                effects = json.load(f)
            # Derive a PNG path next to the JSON
            out_png = effects_path.with_suffix(".png")
            plot_module_heatmap(effects, str(out_png))
            print(f"[ANALYSIS] Wrote module effects heatmap -> {out_png}")
        except Exception as e:
            print(f"[ANALYSIS] Skipped {effects_path}: {e}")


def _load_pairs(pair_file: Path, examples: list[VLMExample]):
    id_to_ex = {ex.id: ex for ex in examples}
    with pair_file.open("r") as f:
        pairs_json = json.load(f)
    pairs = []
    for item in pairs_json:
        clean_id = str(item.get("clean_id"))
        biased_id = str(item.get("biased_id"))
        pairs.append((id_to_ex[clean_id], id_to_ex[biased_id]))
    return pairs


def main():
    args = _parse_args()
    # If analysis-only, generate plots and exit early.
    if args.analysis:
        out_root = Path(args.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        _run_analysis(out_root)
        return
    # Validate required args for experiment mode
    if not args.model_name:
        raise SystemExit("--model-name is required when not running with --analysis")
    # Load dataset
    examples = load_vlmbias(
        split=args.split,
        domain=args.domain,
        return_images=True,
        num_examples=args.num_examples,
    )
    # Load model and processor
    model, processor = load_vlm_model(
        args.model_name,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=args.dtype,
        revision=args.revision,
    )
    # Baseline evaluation
    baseline_results = evaluate_dataset(
        model,
        processor,
        examples,
        max_new_tokens=args.max_new_tokens,
    )
    # Prepare output path
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    domain_str = args.domain or "all"
    model_sanitised = args.model_name.replace("/", "_")
    base_fname = f"{model_sanitised}__{domain_str}"
    metrics_path = out_root / f"{base_fname}__metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "model": args.model_name,
                "domain": domain_str,
                "split": args.split,
                "num_examples": len(examples),
                "baseline": baseline_results,
            },
            f,
            indent=2,
        )
    # Run circuit analysis if requested
    if args.run_circuit:
        pair_file = Path(args.pair_file)
        pairs = _load_pairs(pair_file, examples)
        analyzer = CircuitAnalyzer(model, processor, max_new_tokens=args.max_new_tokens)
        mod_effects = analyzer.analyze_dataset(pairs)
        effects_path = out_root / f"{base_fname}__module_effects.json"
        with effects_path.open("w") as f:
            json.dump(mod_effects, f, indent=2)

    print(f"[DONE] Results written to {metrics_path}")


if __name__ == "__main__":
    main()