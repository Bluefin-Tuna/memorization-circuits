#!/usr/bin/env python3
# Minimal CLI to run circuit reuse experiments.

import argparse
import time
from typing import List
import json
from pathlib import Path
from datetime import datetime
from itertools import product

import torch
from transformer_lens import HookedTransformer

from circuit_reuse.dataset import AdditionDataset, ArithmeticExample, get_dataset
from circuit_reuse.circuit_extraction import CircuitExtractor, compute_shared_circuit
from circuit_reuse.evaluate import evaluate_accuracy, evaluate_accuracy_with_knockout


def _default_run_name():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _prepare_run_dir(output_dir: str, run_name: str | None):
    base = Path(output_dir)
    if run_name is None or run_name.strip() == "":
        run_name = _default_run_name()
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Circuit reuse experiment")
    # ORIGINAL single-value args (kept for backward compatibility)
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the pretrained model to load (TransformerLens)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="addition",
        help="Single task (legacy arg). Ignored if --tasks provided.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Single number of examples (legacy). Ignored if --num_examples_list provided.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Digits for addition (legacy). Ignored if --digits_list provided.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Components per circuit (legacy). Ignored if --top_ks provided.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gradient",
        choices=["gradient", "ig"],
        help="Attribution method (legacy). Ignored if --methods provided.",
    )

    # NEW plural list args for multi-run mode
    parser.add_argument("--model_names", nargs="*", type=str,
                        help="List of model names. Overrides --model_name if provided.")
    parser.add_argument("--tasks", nargs="*", type=str,
                        help="List of tasks. Overrides --task.")
    parser.add_argument("--num_examples_list", nargs="*", type=int,
                        help="List of num_examples values. Overrides --num_examples.")
    parser.add_argument("--digits_list", nargs="*", type=int,
                        help="List of digit counts for addition. Overrides --digits.")
    parser.add_argument("--top_ks", nargs="*", type=int,
                        help="List of top_k values. Overrides --top_k.")
    parser.add_argument("--methods", nargs="*", type=str, choices=["gradient", "ig"],
                        help="List of attribution methods. Overrides --method.")

    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of interpolation steps for integrated gradients",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (cuda or cpu)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose per-example logging (predictions, components).",
    )
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Root directory to store experiment outputs.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional explicit run name; default is timestamp.")
    return parser.parse_args()


def _run_single_combination(
    model: HookedTransformer,
    model_name: str,
    task: str,
    num_examples: int,
    digits: int,
    top_k: int,
    method: str,
    steps: int,
    device: str,
    debug: bool,
    run_dir: Path,
):
    """Execute one (model, task, num_examples, digits, top_k, method) combo and write outputs into run_dir."""
    # DATASET
    if task == "addition":
        dataset = AdditionDataset(num_examples=num_examples, digits=digits)
        print(f"[{model_name}/{task}] Generated {len(dataset)} examples (digits={digits}).")
    else:
        dataset = get_dataset(task, num_examples=num_examples, digits=digits)
        print(f"[{model_name}/{task}] Loaded {len(dataset)} examples.")

    extractor = CircuitExtractor(model, top_k=top_k)

    circuits: List[set] = []
    start = time.time()
    for idx, example in enumerate(dataset):
        if method == "gradient":
            comp_set = extractor.extract_circuit(example.prompt, example.target)
        else:
            comp_set = extractor.extract_circuit_ig(example.prompt, example.target, steps=steps)
        circuits.append(comp_set)
        if debug:
            print(f"[EXTRACT {task}] idx={idx} comps={len(comp_set)}")
        if (idx + 1) % 10 == 0 or (idx + 1) == len(dataset):
            print(f"[{task}] {idx + 1}/{len(dataset)} examples (last circuit size={len(comp_set)})")
        # free intermediate grads right away
        model.zero_grad(set_to_none=True)
    end = time.time()

    shared = compute_shared_circuit(circuits)
    print(f"[{task}] Shared circuit size={len(shared)} (top_k per example={top_k}).")

    baseline_acc = evaluate_accuracy(model, dataset, verbose=debug)
    knockout_acc = evaluate_accuracy_with_knockout(model, dataset, shared, verbose=debug)

    metrics = {
        "model_name": model_name,
        "task": task,
        "num_examples": len(dataset),
        "digits": digits if task == "addition" else None,
        "top_k": top_k,
        "method": method,
        "ig_steps": steps if method == "ig" else None,
        "baseline_accuracy": baseline_acc,
        "knockout_accuracy": knockout_acc,
        "accuracy_drop": baseline_acc - knockout_acc,
        "shared_circuit_size": len(shared),
        "shared_circuit_components": [
            str(c) for c in sorted(shared, key=lambda c: (c.layer, c.kind, c.index))
        ],
        "extraction_seconds": end - start,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (run_dir / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[WARN] Failed to write metrics.json in {run_dir}: {e}")

    try:
        config = {
            "model_name": model_name,
            "task": task,
            "num_examples": num_examples,
            "digits": digits,
            "top_k": top_k,
            "method": method,
            "steps": steps,
            "device": device,
            "debug": debug,
        }
        with (run_dir / "config.json").open("w") as f:
            json.dump(config, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[WARN] Failed to write config.json in {run_dir}: {e}")

    print(f"[DONE] {run_dir}")
    return metrics


def main() -> None:
    """Run experiment end-to-end (single or multi-run)."""
    args = parse_args()

    # Resolve lists (fallback to legacy single-value args)
    model_names = args.model_names if args.model_names else [args.model_name]
    tasks = args.tasks if args.tasks else [args.task]
    methods = args.methods if args.methods else [args.method]
    top_ks = args.top_ks if args.top_ks else [args.top_k]
    digits_list = args.digits_list if args.digits_list else [args.digits]
    num_examples_list = args.num_examples_list if args.num_examples_list else [args.num_examples]

    multi_mode = any(len(lst) > 1 for lst in [model_names, tasks, methods, top_ks, digits_list, num_examples_list])

    # Base directory (single run uses this directly; multi-run uses subdirs)
    base_run_dir = _prepare_run_dir(args.output_dir, args.run_name)

    print(f"Models: {model_names}")
    print(f"Tasks: {tasks}")
    print(f"Methods: {methods}")
    print(f"top_ks: {top_ks}")
    print(f"digits_list: {digits_list}")
    print(f"num_examples_list: {num_examples_list}")
    print(f"Multi-run mode: {multi_mode}")

    total_runs = len(model_names) * len(tasks) * len(methods) * len(top_ks) * len(digits_list) * len(num_examples_list)
    run_counter = 0

    for model_name in model_names:
        print(f"[MODEL LOAD] Loading model {model_name} on {args.device}...")
        model: HookedTransformer = HookedTransformer.from_pretrained(model_name)
        model.to(args.device)
        model.eval()

        for (task, method, top_k, digits, num_examples) in product(tasks, methods, top_ks, digits_list, num_examples_list):
            run_counter += 1
            combo_name = f"{model_name}__{task}__n{num_examples}__d{digits if task=='addition' else 'na'}__k{top_k}__{method}"
            if multi_mode:
                run_dir = base_run_dir / combo_name
            else:
                # Legacy single-run: reuse base directory (no nesting)
                run_dir = base_run_dir
            print(f"\n[RUN {run_counter}/{total_runs}] {combo_name}")
            try:
                _run_single_combination(
                    model=model,
                    model_name=model_name,
                    task=task,
                    num_examples=num_examples,
                    digits=digits,
                    top_k=top_k,
                    method=method,
                    steps=args.steps,
                    device=args.device,
                    debug=args.debug,
                    run_dir=run_dir,
                )
            except torch.cuda.OutOfMemoryError as oom:
                print(f"[OOM] Skipping {combo_name}: {oom}")
            except Exception as e:
                print(f"[ERROR] {combo_name} failed: {e}")
            finally:
                # Clear gradients & cache between combos
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        # Optionally free model before next one
        del model
        torch.cuda.empty_cache()

    print(f"[ALL DONE] Completed {run_counter}/{total_runs} runs. Results root: {base_run_dir.resolve()}")


if __name__ == "__main__":
    main()