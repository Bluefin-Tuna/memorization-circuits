#!/usr/bin/env python3
# Minimal CLI to run circuit reuse experiments.

import argparse
import time
from typing import List
import json
from pathlib import Path
from datetime import datetime

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
        help=(
            "Task to evaluate. Built-in options include 'addition', 'boolean',\n"
            "'mmlu' (toy subset), 'mmlu_real' (load an MMLU subject via the HuggingFace\n"
            "Hub), 'general' (built-in general knowledge), or any dataset under\n"
            "the 'mib' namespace by prefixing with 'mib_' (e.g. 'mib_ioi')."
        ),
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help=(
            "Number of examples to evaluate. For built-in synthetic tasks this\n"
            "controls how many random examples are generated. For HuggingFace\n"
            "datasets (mmlu_real and mib_*), this limits how many examples\n"
            "are loaded from the split."
        ),
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of digits in each operand (addition only)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of components to include in each circuit",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gradient",
        choices=["gradient", "ig"],
        help="Attribution method: 'gradient' for single-step gradients or 'ig' for integrated gradients",
    )
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


def main() -> None:
    """Run experiment end-to-end."""
    args = parse_args()
    print(f"Loading model {args.model_name} on {args.device}...")
    model: HookedTransformer = HookedTransformer.from_pretrained(args.model_name)
    model.to(args.device)

    # Create dataset via factory
    if args.task == "addition":
        dataset = AdditionDataset(num_examples=args.num_examples, digits=args.digits)
        print(
            f"Generated {len(dataset)} examples of {args.digits}-digit addition for evaluation."
        )
    else:
        # boolean and mmlu tasks ignore digits parameter (except addition)
        dataset = get_dataset(args.task, num_examples=args.num_examples, digits=args.digits)
        print(f"Loaded dataset for task '{args.task}' with {len(dataset)} examples.")

    # Create circuit extractor
    extractor = CircuitExtractor(model, top_k=args.top_k)

    # Extract circuits using the chosen attribution method
    circuits: List[set] = []
    start = time.time()
    for idx, example in enumerate(dataset):
        if args.method == "gradient":
            comp_set = extractor.extract_circuit(example.prompt, example.target)
        else:
            # integrated gradients
            comp_set = extractor.extract_circuit_ig(
                example.prompt, example.target, steps=args.steps
            )
        circuits.append(comp_set)
        if args.debug:
            print(f"[EXTRACT] idx={idx} prompt='{example.prompt}' target='{example.target}' components={sorted(comp_set, key=lambda c:(c.layer,c.kind,c.index))}")
        if (idx + 1) % 10 == 0 or (idx + 1) == len(dataset):
            print(
                f"Processed {idx + 1}/{len(dataset)} examples (last circuit size: {len(comp_set)})"
            )
    end = time.time()
    method_name = "gradient" if args.method == "gradient" else f"integrated gradients ({args.steps} steps)"
    print(f"Extraction via {method_name} completed in {end - start:.2f} seconds.")

    # Compute shared circuit
    shared = compute_shared_circuit(circuits)
    print(
        f"Shared circuit contains {len(shared)} components out of {args.top_k} per example."
    )
    if shared:
        for comp in sorted(shared, key=lambda c: (c.layer, c.kind, c.index)):
            print(f"  {comp}")
    else:
        print("No components are shared across all examples.")

    # Evaluate baseline accuracy
    print("Evaluating baseline accuracy (exact sequence)...")
    baseline_acc = evaluate_accuracy(model, dataset, verbose=args.debug)
    print(f"Baseline accuracy: {baseline_acc:.3f}")
    # (All auxiliary diagnostics removed for lean version.)

    # Evaluate with shared circuit removed
    print("Evaluating accuracy with shared circuit knocked out...")
    knockout_acc = evaluate_accuracy_with_knockout(model, dataset, shared, verbose=args.debug)
    print(f"Knockout accuracy: {knockout_acc:.3f}")

    print(
        f"Accuracy drop due to removing shared circuit: {baseline_acc - knockout_acc:.3f}"
    )

    run_dir = _prepare_run_dir(args.output_dir, args.run_name)

    # OPTIONAL: set up logging file (uncomment if desired)
    # import logging, sys
    # logging.basicConfig(
    #     level=logging.INFO,
    #     handlers=[
    #         logging.StreamHandler(sys.stdout),
    #         logging.FileHandler(run_dir / "stdout.log", mode="w")
    #     ]
    # )

    # SAVE results
    try:
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"Failed to write metrics.json: {e}")

    # SAVE config (args)
    try:
        config_path = run_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"Failed to write config.json: {e}")

    # OPTIONAL: save model/artifacts if variables exist
    # if 'model' in locals():
    #     torch.save(model.state_dict(), run_dir / "model.pt")

    print(f"Results saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()