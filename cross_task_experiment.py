"""
Computes a confusion matrix of accuracy drops from ablating shared circuits across tasks.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import torch

from models.olmo_adapter import load_model_any
from circuit_reuse.dataset import get_dataset
from circuit_reuse.evaluate import evaluate_accuracy_with_ablation, evaluate_accuracy
from circuit_reuse.circuit_extraction import Component


def parse_component_str(s: str) -> Component:
    """Parse a string representation of a component into a Component object.

    The expected format is either ``"head[layer=3, index=2]"`` or
    ``"mlp[layer=4, index=0]"``.  Whitespace around commas is ignored.
    """
    kind, rest = s.split("[", 1)
    rest = rest.rstrip("]")
    parts = rest.split(",")
    layer = None
    index = None
    for part in parts:
        k, v = part.split("=")
        k = k.strip()
        v = v.strip()
        if k == "layer":
            layer = int(v)
        elif k == "index":
            index = int(v)
    if layer is None or index is None:
        raise ValueError(f"Unable to parse component string: {s}")
    return Component(layer=layer, kind=kind.strip(), index=index)


def find_metrics_file(results_dir: Path, model_name: str, hf_revision: str, task: str) -> Path:
    """Search ``results_dir`` recursively for a metrics file matching the given settings.

    Returns the path to the first matching metrics.json file.  Raises an
    exception if no match is found.
    """
    for root, _, files in os.walk(results_dir):
        if "metrics.json" in files:
            path = Path(root) / "metrics.json"
            try:
                with path.open() as f:
                    data = json.load(f)
                if (
                    data.get("model_name") == model_name
                    and str(data.get("hf_revision") or "none") == (hf_revision or "none")
                    and data.get("task") == task
                ):
                    return path
            except Exception:
                continue
    raise FileNotFoundError(
        f"No metrics.json found in {results_dir} for model={model_name}, revision={hf_revision}, task={task}."
    )


def load_shared_components(metrics_path: Path, K: int, threshold: int) -> List[Component]:
    """Load the list of shared components for a given K and threshold from a metrics file."""
    with metrics_path.open() as f:
        data = json.load(f)
    by_k = data.get("by_k", {})
    entry = by_k.get(str(K))
    if not entry:
        raise ValueError(f"K={K} not found in metrics file {metrics_path}")
    thresholds = entry.get("thresholds", {})
    thr_entry = thresholds.get(str(threshold))
    if not thr_entry:
        raise ValueError(
            f"threshold {threshold} not found for K={K} in metrics file {metrics_path}"
        )
    return [parse_component_str(c) for c in thr_entry.get("shared_components", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute cross‑task ablation confusion matrix.")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Root directory containing run subdirectories with metrics.json files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name used in the experiments.",
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default="main",
        help="Revision/tag of the model used in the experiments.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma‑separated list of task names to include in the matrix.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="Top‑K percentage used when constructing shared circuits.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Reuse threshold percentage (p) used to define the shared circuit.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=None,
        help="Number of digits for addition task (only used for addition).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device on which to run evaluations.",
    )
    args = parser.parse_args()

    tasks: List[str] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results_dir = Path(args.results_dir)

    # Load model once
    model = load_model_any(args.model_name, device=args.device, revision=args.hf_revision)
    model.eval()

    # Baseline accuracies per task (to avoid recomputation)
    baseline_acc: Dict[str, float] = {}
    baseline_correct: Dict[str, int] = {}
    baseline_total: Dict[str, int] = {}
    # Datasets per task
    datasets: Dict[str, List] = {}
    for task in tasks:
        ds = get_dataset(task, num_examples=args.digits if task == "addition" else None)
        datasets[task] = list(ds)
        bc, bt = evaluate_accuracy(model, datasets[task], task=task)
        baseline_correct[task] = bc
        baseline_total[task] = bt
        baseline_acc[task] = bc / bt if bt > 0 else 0.0

    # For each source task, load shared components and compute drops on all tasks
    matrix: Dict[str, Dict[str, float]] = {t: {} for t in tasks}
    for src_task in tasks:
        metrics_path = find_metrics_file(results_dir, args.model_name, args.hf_revision, src_task)
        shared_components = load_shared_components(metrics_path, args.K, args.threshold)
        for tgt_task in tasks:
            # Evaluate with ablation
            correct, total = evaluate_accuracy_with_ablation(
                model, datasets[tgt_task], task=tgt_task, removed=shared_components
            )
            acc = correct / total if total > 0 else 0.0
            drop = baseline_acc[tgt_task] - acc
            matrix[src_task][tgt_task] = drop * 100.0  # express as percentage points

    # Print header
    header = ["source/task"] + tasks
    print(",".join(header))
    for src_task in tasks:
        row = [src_task] + [f"{matrix[src_task][t]:.3f}" for t in tasks]
        print(",".join(row))


if __name__ == "__main__":
    main()