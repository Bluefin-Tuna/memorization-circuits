#!/usr/bin/env python3
"""
Aggregate and plot experiment metrics produced by run_experiment.py.

For each attribution method, produce a clean horizontal grouped bar plot:
Baseline vs Knockout accuracy across tasks (one PNG per method) with Δ annotations.
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
    p.add_argument("--palette", type=str, default="#377eb8",
                   help="Bar color (hex or seaborn palette name). Default: #377eb8")
    return p.parse_args()


def discover_metrics(results_dir: Path) -> List[Path]:
    paths: List[Path] = []
    if not results_dir.exists():
        return paths
    for child in results_dir.iterdir():
        if child.is_dir():
            m = child / "metrics.json"
            if m.is_file():
                paths.append(m)
    return sorted(paths)


def load_metrics_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r") as f:
            data = json.load(f)
        data["_run_dir"] = path.parent.name
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
             percent: bool, overlay_scores: bool, palette: str):
    if df.empty:
        print("[INFO] No data to plot.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use default matplotlib style (do not apply seaborn theme)

    # Aggregate strictly by (task, method)
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

    # Create one bar chart per method
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
        width = 0.38
        fig_w = max(6.0, 0.8 * n_tasks + 1.5)
        fig_h = 4.3
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.bar(x - width/2, baseline, width, label=f"Baseline ({method})")
        ax.bar(x + width/2, after, width, label=f"After Removal ({method})")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=25 if n_tasks > 6 or any(len(t)>10 for t in tasks) else 0, ha='right' if n_tasks > 6 else 'center')
        ax.set_ylabel("Accuracy (%)" if percent else "Accuracy")
        ax.set_xlabel("Task")
        ax.set_ylim(0, 100 if percent else 1.0)
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
        print(f"[INFO] Found {len(metrics_paths)} metrics.json files.")
    df = aggregate(metrics_paths)
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
        percent=args.percent,
        overlay_scores=args.overlay_scores,
        palette=args.palette,
    )


if __name__ == "__main__":
    main()