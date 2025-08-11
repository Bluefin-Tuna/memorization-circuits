#!/usr/bin/env python3
"""
Aggregate and plot experiment metrics produced by run_experiment.py.

Produces one grouped bar plot per attribution method: baseline vs ablation accuracy
with value labels on each bar.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from circuit_reuse.dataset import get_task_display_name, get_model_display_name


# Method display names
METHOD_DISPLAY = {
    "gradient": "Gradient",
    "ig": "Integrated Gradients",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot circuit reuse experiment results.")
    p.add_argument("--results-dir", type=str, default="results",
                   help="Directory containing per-run subdirectories with metrics.json.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory to write plots (default: <results-dir>/plots_<timestamp>).")
    p.add_argument("--save-csv-name", type=str, default="aggregated_metrics.csv",
                   help="Filename for aggregated CSV inside output directory.")
    p.add_argument("--show", action="store_true",
                   help="Show plots interactively (if a display is available).")
    p.add_argument("--sort-by", type=str, default="drop",
                   choices=["drop", "baseline", "ablation", "task"],
                   help="Ordering of tasks within each plot.")
    p.add_argument("--percent", action="store_true", default=True,
                   help="Display accuracies as percentage (0-100%). (Default: True)")
    p.add_argument("--overlay-scores", action="store_true",
                   help="Overlay baseline and post-removal accuracies as connected markers above bars.")
    p.add_argument("--raw", action="store_true",
                   help="Show raw accuracies (0-1). Overrides --percent default.")
    return p.parse_args()


def discover_metrics(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("metrics.json"))


def load_metrics_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r") as f:
            data = json.load(f)
        data["_run_dir"] = str(path.parent.relative_to(path.parents[2])) if len(path.parents) >= 3 else path.parent.name
        data["_run_path"] = str(path.parent)
        return data
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}


def aggregate(paths: List[Path]) -> pd.DataFrame:
    rows = [d for p in paths if (d := load_metrics_json(p))]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    numeric_cols = [
        "baseline_accuracy", "ablation_accuracy", "accuracy_drop", "top_k",
        "shared_circuit_size", "extraction_seconds", "num_examples", "digits", "ig_steps"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def plot_all(df: pd.DataFrame, out_dir: Path, show: bool, sort_by: str, *,
             percent: bool, overlay_scores: bool):
    if df.empty:
        print("[INFO] No data to plot.")
        return

    grouped = (df
               .groupby(["model_display", "task_display", "method"], as_index=False)
               .agg(
                   baseline_accuracy_mean=("baseline_accuracy", "mean"),
                   ablation_accuracy_mean=("ablation_accuracy", "mean"),
                   accuracy_drop_mean=("accuracy_drop", "mean"),
                   runs=("baseline_accuracy", "count")
               ))
    if {"top_k", "model_name"}.intersection(df.columns) and \
       (df[["model_name", "task", "method"]].drop_duplicates().shape[0] != grouped.shape[0]):
        print("[INFO] Aggregation averaged over varying top_k / model_name.")

    methods = grouped["method"].unique()
    saved_files: list[str] = []

    sns.set_theme(style="ticks", context="talk", palette="colorblind")
    plt.rcParams.update({"font.family": "serif"})

    for method in methods:
        method_subset = grouped[grouped["method"] == method]
        for model_disp in sorted(method_subset.model_display.unique()):
            sub = method_subset[method_subset.model_display == model_disp].copy()

            if sort_by == "drop":
                sub = sub.sort_values("accuracy_drop_mean", ascending=False)
            elif sort_by == "baseline":
                sub = sub.sort_values("baseline_accuracy_mean", ascending=False)
            elif sort_by == "ablation":
                sub = sub.sort_values("ablation_accuracy_mean", ascending=False)
            else:
                sub = sub.sort_values("task_display")

            tasks = list(sub.task_display)
            baseline = sub.baseline_accuracy_mean.values * (100 if percent else 1)
            after = sub.ablation_accuracy_mean.values * (100 if percent else 1)
            n_tasks = len(tasks)
            x = np.arange(n_tasks)
            width = 0.40

            fig_w = max(8.0, 1.2 * n_tasks + 2.5)
            fig_h = 6.0
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            bars1 = ax.bar(x - width/2, baseline, width, label="Baseline")
            bars2 = ax.bar(x + width/2, after, width, label="Ablated")

            ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
            if percent:
                ax.set_ylim(0, 105)
            else:
                max_val = float(np.max([baseline.max() if len(baseline) else 0,
                                        after.max() if len(after) else 0]))
                ax.set_ylim(0, max_val * 1.08 if max_val > 0 else 1)

            def _annotate(bar_container, values):
                for rect, val in zip(bar_container, values):
                    h = rect.get_height()
                    label = f"{val:.1f}" if percent else f"{val:.3f}"
                    ax.annotate(label,
                                xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, 2.0),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=11, zorder=5)

            _annotate(bars1, baseline)
            _annotate(bars2, after)

            ax.set_xlabel("Task")
            ax.set_ylabel("Accuracy (%)" if percent else "Accuracy")
            ax.set_xticks(x)
            ax.set_xticklabels(tasks, rotation=0, ha='center')
            ax.set_title(f"{model_disp} – {METHOD_DISPLAY.get(method, method.title())} Attribution")

            ax.legend(loc='upper right', frameon=True)
            fig.tight_layout()

            suffix = 'pct' if percent else 'raw'
            safe_model = model_disp.replace(" ", "_")
            out_path = out_dir / f"{safe_model}_accuracy_{method}_{suffix}.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            saved_files.append(out_path.name)

            if show:
                plt.show()
            plt.close()

    print(f"[INFO] Plots written to: {out_dir}")
    print(f"[INFO] Files: {', '.join(saved_files)}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    metrics_paths = discover_metrics(results_dir)
    
    if not metrics_paths:
        print(f"[INFO] No metrics.json files found under {results_dir}")
        return
        
    print(f"[INFO] Found {len(metrics_paths)} metrics.json files (recursive).")
    df = aggregate(metrics_paths)
    
    if df.empty:
        print("[INFO] Empty DataFrame; skipping CSV and plots.")
        return

    percent = not args.raw
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / f"plots_{timestamp}"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df["task_display"] = df["task"].apply(get_task_display_name)
    df["model_display"] = df["model_name"].apply(get_model_display_name)
    df["method_display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"].str.title())
    csv_path = out_dir / args.save_csv_name
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Aggregated CSV saved to {csv_path}")

    plot_all(
        df, out_dir,
        show=args.show,
        sort_by=args.sort_by,
        percent=percent,
        overlay_scores=args.overlay_scores,
    )


if __name__ == "__main__":
    main()