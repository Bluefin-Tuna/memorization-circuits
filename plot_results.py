#!/usr/bin/env python3
"""
Aggregate and plot experiment metrics produced by run_experiment.py.

Produces one grouped bar plot per attribution method: baseline vs knockout accuracy
with value labels on each bar (no diff annotations, no forced colors).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # still imported but not used for styling (kept to avoid removing dependency)
import numpy as np


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
                   choices=["drop","baseline","knockout","task"],
                   help="Ordering of tasks within each plot.")
    p.add_argument("--percent", action="store_true", default=True,
                   help="Display accuracies as percentage (0-100%). (Default: True)")
    p.add_argument("--overlay-scores", action="store_true",
                   help="Overlay baseline and post-removal accuracies as connected markers above bars.")
    # NOTE: --percent kept for backward compatibility (defaults to True behavior).
    # Add --raw to disable percentage scaling.
    p.add_argument("--raw", action="store_true",
                   help="Show raw accuracies (0-1). Overrides --percent default.")
    return p.parse_args()


def discover_metrics(results_dir: Path) -> List[Path]:
    # REPLACED: now recursive to handle results/slurm_<job>/<combo>/metrics.json
    paths: List[Path] = []
    if not results_dir.exists():
        return paths
    for m in results_dir.rglob("metrics.json"):
        # Skip plotting directories' own metrics if any future additions
        paths.append(m)
    return sorted(paths)


def load_metrics_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r") as f:
            data = json.load(f)
        # Store relative run directory (two levels up possible)
        data["_run_dir"] = str(path.parent.relative_to(path.parents[2])) if len(path.parents) >= 3 else path.parent.name
        data["_run_path"] = str(path.parent)
        return data
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}


def aggregate(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        d = load_metrics_json(p)
        if d:
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    numeric_cols = [
        "baseline_accuracy","knockout_accuracy","accuracy_drop","top_k",
        "shared_circuit_size","extraction_seconds","num_examples","digits","ig_steps"
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
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = (df
               .groupby(["task","method"], as_index=False)
               .agg(
                   baseline_accuracy_mean=("baseline_accuracy","mean"),
                   knockout_accuracy_mean=("knockout_accuracy","mean"),
                   accuracy_drop_mean=("accuracy_drop","mean"),
                   runs=("baseline_accuracy","count")
               ))
    if {"top_k","model_name"}.intersection(df.columns) and \
       (df[["task","method"]].drop_duplicates().shape[0] != grouped.shape[0]):
        print("[INFO] Aggregation averaged over varying top_k / model_name.")

    methods = grouped["method"].unique()
    saved_files: list[str] = []

    for method in methods:
        sub = grouped[grouped["method"] == method].copy()

        if sort_by == "drop":
            sub = sub.sort_values("accuracy_drop_mean", ascending=False)
        elif sort_by == "baseline":
            sub = sub.sort_values("baseline_accuracy_mean", ascending=False)
        elif sort_by == "knockout":
            sub = sub.sort_values("knockout_accuracy_mean", ascending=False)
        else:
            sub = sub.sort_values("task")

        tasks = list(sub.task)
        baseline = sub.baseline_accuracy_mean.values * (100 if percent else 1)
        after = sub.knockout_accuracy_mean.values * (100 if percent else 1)
        n_tasks = len(tasks)
        x = np.arange(n_tasks)
        width = 0.40
        fig_w = max(6.0, 0.8 * n_tasks + 1.5)
        fig_h = 4.3
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        bars1 = ax.bar(x - width/2, baseline, width, label="Baseline")
        bars2 = ax.bar(x + width/2, after, width, label="Knockout")

        # Dynamic y-limit
        max_val = float(np.max([baseline.max() if len(baseline) else 0,
                                after.max() if len(after) else 0, 0]))
        if max_val == 0:
            y_top = 1 if not percent else 5
        else:
            margin = 2 if percent else 0.02
            y_top = max_val + margin
            if percent:
                y_top = min(100, y_top if y_top > max_val else max_val + 2)
        ax.set_ylim(0, y_top)

        # Value annotations (each bar)
        def _annotate(bar_container, values):
            for rect, val in zip(bar_container, values):
                h = rect.get_height()
                if percent:
                    label = f"{val:.1f}"
                    offset = 0.6
                else:
                    label = f"{val:.3f}"
                    offset = 0.01
                ax.text(rect.get_x() + rect.get_width()/2,
                        h + offset,
                        label,
                        ha="center", va="bottom", fontsize=8)

        _annotate(bars1, baseline)
        _annotate(bars2, after)

        ax.set_xticks(x)
        ax.set_xticklabels(tasks,
                           rotation=25 if n_tasks > 6 or any(len(t) > 10 for t in tasks) else 0,
                           ha='right' if n_tasks > 6 else 'center')
        ax.set_ylabel("Accuracy (%)" if percent else "Accuracy")
        ax.set_xlabel("Task")
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout()
        suffix = 'pct' if percent else 'raw'
        out_path = out_dir / f"accuracy_comparison_method_{method}_{suffix}.png"
        plt.savefig(out_path, dpi=170, bbox_inches="tight")
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
    else:
        print(f"[INFO] Found {len(metrics_paths)} metrics.json files (recursive).")
    df = aggregate(metrics_paths)
    # Determine percent flag (raw overrides)
    percent = not args.raw
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / f"plots_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        csv_path = out_dir / args.save_csv_name
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Aggregated CSV saved to {csv_path}")
    else:
        print("[INFO] Empty DataFrame; skipping CSV and plots.")
    plot_all(
        df, out_dir,
        show=args.show,
        sort_by=args.sort_by,
        percent=percent,
        overlay_scores=args.overlay_scores,
    )


if __name__ == "__main__":
    main()