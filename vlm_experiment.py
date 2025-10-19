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
    p.add_argument("--run-circuit", action="store_true", help="Perform causal tracing over modules.")
    p.add_argument("--run-heads", action="store_true", help="Perform head-level analysis.")
    p.add_argument("--run-ablation", action="store_true", help="Perform ablation experiments on top heads.")
    p.add_argument("--run-full-pipeline", action="store_true", help="Run complete pipeline (baseline → circuit → heads → ablation).")
    p.add_argument("--num-pairs", type=int, default=100, help="Number of clean/biased pairs to create for circuit analysis.")
    p.add_argument("--top-k-heads", type=int, default=10, help="Number of top heads to analyze/ablate.")
    p.add_argument("--held-out-domain", type=str, default=None, help="Held-out domain for ablation generalization test.")
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
    p.add_argument(
        "--create-summary",
        action="store_true",
        help="Create summary report from results in output-dir.",
    )
    p.add_argument("--target-layers", type=str, default="early", help="Which layers to target: 'early', 'all', or comma-separated indices.")
    return p.parse_args()


def _run_analysis(output_dir: Path):
    """Generate plots from previously saved JSON results.

    This scans the output directory for "*__metrics.json" and
    "*__module_effects.json" files, aggregates summary metrics across
    models/domains, and emits plots alongside the JSON files.
    """
    from vlm_analysis.plotting import plot_head_importance, plot_ablation_impact
    
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
            if not effects:
                print(f"[ANALYSIS] Skipped {effects_path}: empty results (no modules matched filter)")
                continue
            # Derive a PNG path next to the JSON
            out_png = effects_path.with_suffix(".png")
            plot_module_heatmap(effects, str(out_png))
            print(f"[ANALYSIS] Wrote module effects heatmap -> {out_png}")
        except Exception as e:
            print(f"[ANALYSIS] Skipped {effects_path}: {e}")
    
    # Plot head importance from head effects files
    for head_effects_path in output_dir.glob("*__head_effects.json"):
        try:
            with head_effects_path.open("r") as f:
                head_effects = json.load(f)
            if not head_effects:
                print(f"[ANALYSIS] Skipped {head_effects_path}: empty results")
                continue
            
            # Convert to list of (name, score) tuples sorted by score
            head_scores = []
            for key_str, score in head_effects.items():
                # Parse the key which is stored as string like "('module.name', 0)"
                head_scores.append((key_str, score))
            head_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            
            out_png = head_effects_path.with_suffix(".png")
            plot_head_importance(head_scores[:20], str(out_png))  # Top 20 heads
            print(f"[ANALYSIS] Wrote head importance plot -> {out_png}")
        except Exception as e:
            print(f"[ANALYSIS] Skipped {head_effects_path}: {e}")
    
    # Plot ablation impact from ablation results files
    for ablation_path in output_dir.glob("*__ablation_results.json"):
        try:
            with ablation_path.open("r") as f:
                ablation_data = json.load(f)
            if not ablation_data:
                print(f"[ANALYSIS] Skipped {ablation_path}: empty results")
                continue
            
            # Format: {"baseline": {...}, "ablated_in_domain": {...}, "ablated_held_out": {...}}
            ablation_metrics = {}
            if "baseline" in ablation_data and "ablated_in_domain" in ablation_data:
                ablation_metrics["In-domain"] = {
                    "baer_before": ablation_data["baseline"].get("baer", 0.0),
                    "baer_after": ablation_data["ablated_in_domain"].get("baer", 0.0),
                }
            if "ablated_held_out" in ablation_data:
                ablation_metrics["Held-out"] = {
                    "baer_before": ablation_data.get("baseline_held_out", {}).get("baer", 0.0),
                    "baer_after": ablation_data["ablated_held_out"].get("baer", 0.0),
                }
            
            if ablation_metrics:
                out_png = ablation_path.with_suffix(".png")
                plot_ablation_impact(ablation_metrics, str(out_png))
                print(f"[ANALYSIS] Wrote ablation impact plot -> {out_png}")
        except Exception as e:
            print(f"[ANALYSIS] Skipped {ablation_path}: {e}")


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


def _create_pairs_from_metrics(metrics_path: Path, num_pairs: int) -> Path:
    """Create pairs file from metrics JSON, grouping by resolution."""
    from collections import defaultdict
    
    with metrics_path.open("r") as f:
        data = json.load(f)
    
    details = data.get("baseline", {}).get("details", [])
    
    # Group by resolution
    by_resolution = defaultdict(list)
    for item in details:
        item_id = item["id"]
        if "_px" in item_id:
            resolution = item_id.split("_px")[1]
            by_resolution[resolution].append(item)
        else:
            by_resolution["default"].append(item)
    
    # Create pairs within same resolution
    pairs = []
    for resolution, items in sorted(by_resolution.items()):
        for i in range(0, len(items) - 1, 2):
            if len(pairs) >= num_pairs:
                break
            pairs.append({
                "clean_id": items[i]["id"],
                "biased_id": items[i + 1]["id"]
            })
        if len(pairs) >= num_pairs:
            break
    
    if len(pairs) == 0:
        raise ValueError("Could not create any pairs from metrics")
    
    pairs_file = metrics_path.parent / f"{metrics_path.stem.replace('__metrics', '__pairs')}.json"
    with pairs_file.open("w") as f:
        json.dump(pairs, f, indent=2)
    
    print(f"[PAIRS] Created {len(pairs)} resolution-matched pairs")
    return pairs_file


def _create_summary_report(output_dir: Path):
    """Generate summary report from all results in output directory."""
    summary = {
        "timestamp": output_dir.name.split("_")[-1] if "_" in output_dir.name else "",
        "model": "",
        "domains": [],
    }
    
    # Collect all metrics files
    for metrics_file in sorted(output_dir.glob("*__metrics.json")):
        with metrics_file.open() as f:
            data = json.load(f)
        
        if not summary["model"]:
            summary["model"] = data.get("model", "unknown")
        
        domain_summary = {
            "domain": data.get("domain", "unknown"),
            "num_examples": data.get("num_examples", 0),
            "accuracy": data.get("baseline", {}).get("accuracy", 0.0),
            "baer": data.get("baseline", {}).get("baer", 0.0),
        }
        
        # Add ablation results if available
        ablation_file = metrics_file.parent / metrics_file.name.replace("metrics", "ablation_results")
        if ablation_file.exists():
            with ablation_file.open() as f:
                ablation_data = json.load(f)
            
            baseline_baer = ablation_data.get("baseline", {}).get("baer", 0.0)
            ablated_baer = ablation_data.get("ablated_in_domain", {}).get("baer", 0.0)
            domain_summary["ablation"] = {
                "in_domain_baer_reduction": baseline_baer - ablated_baer,
                "in_domain_baer_before": baseline_baer,
                "in_domain_baer_after": ablated_baer,
            }
            
            if "ablated_held_out" in ablation_data:
                baseline_held_out = ablation_data.get("baseline_held_out", {}).get("baer", 0.0)
                ablated_held_out = ablation_data["ablated_held_out"].get("baer", 0.0)
                domain_summary["ablation"]["held_out_baer_reduction"] = baseline_held_out - ablated_held_out
                domain_summary["ablation"]["held_out_baer_before"] = baseline_held_out
                domain_summary["ablation"]["held_out_baer_after"] = ablated_held_out
        
        summary["domains"].append(domain_summary)
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY REPORT")
    print("=" * 50)
    print(f"Model: {summary['model']}")
    print(f"Timestamp: {summary['timestamp']}")
    print()
    
    for domain in summary["domains"]:
        print(f"Domain: {domain['domain']}")
        print(f"  Examples: {domain['num_examples']}")
        print(f"  Accuracy: {domain['accuracy']:.3f}")
        print(f"  BAER: {domain['baer']:.3f}")
        
        if "ablation" in domain:
            abl = domain["ablation"]
            print(f"  In-domain ablation:")
            print(f"    BAER: {abl['in_domain_baer_before']:.3f} → {abl['in_domain_baer_after']:.3f}")
            print(f"    Reduction: {abl['in_domain_baer_reduction']:.3f}")
            
            if "held_out_baer_reduction" in abl:
                print(f"  Held-out ablation:")
                print(f"    BAER: {abl['held_out_baer_before']:.3f} → {abl['held_out_baer_after']:.3f}")
                print(f"    Reduction: {abl['held_out_baer_reduction']:.3f}")
        print()
    
    print(f"Summary saved to: {summary_file}")
    return summary_file


def main():
    args = _parse_args()
    
    # Handle summary creation
    if args.create_summary:
        out_root = Path(args.output_dir)
        _create_summary_report(out_root)
        return
    
    # If analysis-only, generate plots and exit early.
    if args.analysis:
        out_root = Path(args.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        _run_analysis(out_root)
        return
    # Validate required args for experiment mode
    if not args.model_name:
        raise SystemExit("--model-name is required when not running with --analysis")
    
    # Parse target layers
    target_layers = args.target_layers
    if target_layers not in ["early", "all"]:
        try:
            target_layers = [int(x) for x in target_layers.split(",")]
        except ValueError:
            raise SystemExit(f"Invalid --target-layers: {args.target_layers}")
    
    # Load dataset
    examples = load_vlmbias(
        split=args.split,
        domain=args.domain,
        return_images=True,
        num_examples=args.num_examples,
    )
    
    # Load held-out domain if specified
    held_out_examples = None
    if args.held_out_domain:
        held_out_examples = load_vlmbias(
            split=args.split,
            domain=args.held_out_domain,
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
    
    # Prepare output path
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    domain_str = args.domain or "all"
    model_sanitised = args.model_name.replace("/", "_")
    base_fname = f"{model_sanitised}__{domain_str}"
    
    # Handle full pipeline mode
    if args.run_full_pipeline:
        print("[PIPELINE] Running full experimental pipeline")
        
        # Step 1: Baseline evaluation
        print("\n[1/5] Baseline evaluation...")
        baseline_results = evaluate_dataset(
            model,
            processor,
            examples,
            max_new_tokens=args.max_new_tokens,
        )
        
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
        print(f"[BASELINE] Accuracy: {baseline_results['accuracy']:.3f}, BAER: {baseline_results['baer']:.3f}")
        
        # Step 2: Create pairs
        print("\n[2/5] Creating pairs for circuit analysis...")
        pairs_file = _create_pairs_from_metrics(metrics_path, args.num_pairs)
        pairs = _load_pairs(pairs_file, examples)
        
        # Step 3: Module-level analysis
        print("\n[3/5] Module-level causal tracing...")
        analyzer = CircuitAnalyzer(
            model, processor, 
            max_new_tokens=args.max_new_tokens,
            target_layers=target_layers
        )
        mod_effects = analyzer.analyze_dataset(pairs)
        effects_path = out_root / f"{base_fname}__module_effects.json"
        with effects_path.open("w") as f:
            json.dump(mod_effects, f, indent=2)
        print(f"[CIRCUIT] Top module: {max(mod_effects.items(), key=lambda x: abs(x[1]))}")
        
        # Step 4: Head-level analysis
        print("\n[4/5] Head-level discovery...")
        head_effects = analyzer.analyze_heads(pairs)
        head_effects_serializable = {
            f"{module}__head_{head}": effect
            for (module, head), effect in head_effects.items()
        }
        head_effects_path = out_root / f"{base_fname}__head_effects.json"
        with head_effects_path.open("w") as f:
            json.dump(head_effects_serializable, f, indent=2)
        
        sorted_heads = sorted(head_effects.items(), key=lambda x: abs(x[1]), reverse=True)
        top_heads = [head for head, _ in sorted_heads[:args.top_k_heads]]
        print(f"[HEADS] Identified top {len(top_heads)} heads")
        
        # Step 5: Ablation experiments
        print("\n[5/5] Ablation experiments...")
        ablated_in_domain = analyzer.ablate_heads(examples, top_heads)
        
        ablation_results = {
            "top_k": args.top_k_heads,
            "heads_ablated": [{"module": m, "head": h} for m, h in top_heads],
            "baseline": baseline_results,
            "ablated_in_domain": ablated_in_domain,
        }
        
        if held_out_examples:
            baseline_held_out = evaluate_dataset(
                model, processor, held_out_examples, 
                max_new_tokens=args.max_new_tokens
            )
            ablated_held_out = analyzer.ablate_heads(held_out_examples, top_heads)
            ablation_results["baseline_held_out"] = baseline_held_out
            ablation_results["ablated_held_out"] = ablated_held_out
            print(f"[ABLATION] Held-out BAER reduction: {baseline_held_out['baer'] - ablated_held_out['baer']:.3f}")
        
        ablation_path = out_root / f"{base_fname}__ablation_results.json"
        with ablation_path.open("w") as f:
            json.dump(ablation_results, f, indent=2)
        
        print(f"\n[DONE] Full pipeline complete for {domain_str}")
        return
    
    # Baseline evaluation
    baseline_results = evaluate_dataset(
        model,
        processor,
        examples,
        max_new_tokens=args.max_new_tokens,
    )
    
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
    print(f"[BASELINE] Accuracy: {baseline_results['accuracy']:.3f}, BAER: {baseline_results['baer']:.3f}")
    
    # Run circuit analysis if requested
    if args.run_circuit:
        print("[CIRCUIT] Running module-level causal tracing...")
        pair_file = Path(args.pair_file)
        pairs = _load_pairs(pair_file, examples)
        analyzer = CircuitAnalyzer(
            model, processor, 
            max_new_tokens=args.max_new_tokens,
            target_layers=target_layers
        )
        print(f"[CIRCUIT] Analyzing {len(analyzer.modules)} modules across {len(pairs)} pairs...")
        mod_effects = analyzer.analyze_dataset(pairs)
        effects_path = out_root / f"{base_fname}__module_effects.json"
        with effects_path.open("w") as f:
            json.dump(mod_effects, f, indent=2)
        print(f"[CIRCUIT] Module effects written to {effects_path}")
        
        # Show top modules
        sorted_effects = sorted(mod_effects.items(), key=lambda x: abs(x[1]), reverse=True)
        print("[CIRCUIT] Top 5 modules by absolute effect:")
        for name, eff in sorted_effects[:5]:
            print(f"  {name}: {eff:.4f}")
    
    # Run head-level analysis if requested
    if args.run_heads:
        print("[HEADS] Running head-level analysis...")
        pair_file = Path(args.pair_file)
        pairs = _load_pairs(pair_file, examples)
        analyzer = CircuitAnalyzer(
            model, processor,
            max_new_tokens=args.max_new_tokens,
            target_layers=target_layers
        )
        head_effects = analyzer.analyze_heads(pairs)
        
        # Convert tuple keys to strings for JSON serialization
        head_effects_serializable = {
            f"{module}__head_{head}": effect
            for (module, head), effect in head_effects.items()
        }
        
        head_effects_path = out_root / f"{base_fname}__head_effects.json"
        with head_effects_path.open("w") as f:
            json.dump(head_effects_serializable, f, indent=2)
        print(f"[HEADS] Head effects written to {head_effects_path}")
        
        # Show top heads
        sorted_heads = sorted(head_effects.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"[HEADS] Top {min(10, len(sorted_heads))} heads by absolute effect:")
        for (module, head), eff in sorted_heads[:10]:
            layer_num = module.split("layers.")[1].split(".")[0] if "layers." in module else "?"
            print(f"  Layer {layer_num}, Head {head}: {eff:.4f}")
    
    # Run ablation experiments if requested
    if args.run_ablation:
        print("[ABLATION] Running ablation experiments...")
        
        # Load head effects to determine which heads to ablate
        head_effects_path = out_root / f"{base_fname}__head_effects.json"
        if not head_effects_path.exists():
            print("[ABLATION] No head effects file found. Run with --run-heads first.")
            return
        
        with head_effects_path.open("r") as f:
            head_effects_data = json.load(f)
        
        # Parse head effects and get top K
        head_effects = []
        for key, effect in head_effects_data.items():
            # Parse "module__head_N" format
            parts = key.rsplit("__head_", 1)
            if len(parts) == 2:
                module, head_str = parts
                try:
                    head = int(head_str)
                    head_effects.append(((module, head), effect))
                except ValueError:
                    continue
        
        head_effects.sort(key=lambda x: abs(x[1]), reverse=True)
        top_heads = [head for head, _ in head_effects[:args.top_k_heads]]
        
        print(f"[ABLATION] Ablating top {len(top_heads)} heads...")
        for module, head in top_heads[:5]:
            layer = module.split("layers.")[1].split(".")[0] if "layers." in module else "?"
            print(f"  Layer {layer}, Head {head}")
        
        analyzer = CircuitAnalyzer(
            model, processor,
            max_new_tokens=args.max_new_tokens,
            target_layers=target_layers
        )
        
        # Ablate on in-domain examples
        ablated_in_domain = analyzer.ablate_heads(examples, top_heads)
        print(f"[ABLATION] In-domain - Baseline BAER: {baseline_results['baer']:.3f}, Ablated BAER: {ablated_in_domain['baer']:.3f}")
        print(f"[ABLATION] In-domain - BAER reduction: {(baseline_results['baer'] - ablated_in_domain['baer']):.3f}")
        
        ablation_results = {
            "top_k": args.top_k_heads,
            "heads_ablated": [{"module": m, "head": h} for m, h in top_heads],
            "baseline": baseline_results,
            "ablated_in_domain": ablated_in_domain,
        }
        
        # Ablate on held-out domain if provided
        if held_out_examples:
            print(f"[ABLATION] Testing on held-out domain: {args.held_out_domain}")
            baseline_held_out = evaluate_dataset(
                model, processor, held_out_examples,
                max_new_tokens=args.max_new_tokens
            )
            ablated_held_out = analyzer.ablate_heads(held_out_examples, top_heads)
            print(f"[ABLATION] Held-out - Baseline BAER: {baseline_held_out['baer']:.3f}, Ablated BAER: {ablated_held_out['baer']:.3f}")
            print(f"[ABLATION] Held-out - BAER reduction: {(baseline_held_out['baer'] - ablated_held_out['baer']):.3f}")
            
            ablation_results["baseline_held_out"] = baseline_held_out
            ablation_results["ablated_held_out"] = ablated_held_out
        
        ablation_path = out_root / f"{base_fname}__ablation_results.json"
        with ablation_path.open("w") as f:
            json.dump(ablation_results, f, indent=2)
        print(f"[ABLATION] Results written to {ablation_path}")

    print(f"[DONE] All results written to {out_root}")


if __name__ == "__main__":
    main()