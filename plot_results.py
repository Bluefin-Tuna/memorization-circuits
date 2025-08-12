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
        "top_k", "shared_circuit_size", "extraction_seconds", 
        "num_examples", "digits", "ig_steps",
        "baseline_train_accuracy", "ablation_train_accuracy", "ablation_control_train_accuracy",
        "baseline_val_accuracy", "ablation_val_accuracy", "ablation_control_val_accuracy",
        "accuracy_drop_train", "accuracy_drop_val", "accuracy_drop_train_val",
        "accuracy_drop_train_control", "accuracy_drop_val_control", "accuracy_drop_train_val_control"
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
                   baseline_train_accuracy_mean=("baseline_train_accuracy", "mean"),
                   ablation_train_accuracy_mean=("ablation_train_accuracy", "mean"),
                   ablation_control_train_accuracy_mean=("ablation_control_train_accuracy", "mean"),
                   baseline_val_accuracy_mean=("baseline_val_accuracy", "mean"),
                   ablation_val_accuracy_mean=("ablation_val_accuracy", "mean"),
                   ablation_control_val_accuracy_mean=("ablation_control_val_accuracy", "mean"),
                   runs=("baseline_train_accuracy", "count")
               ))
    if {"top_k", "model_name"}.intersection(df.columns) and \
       (df[["model_name", "task", "method"]].drop_duplicates().shape[0] != grouped.shape[0]):
        print("[INFO] Aggregation averaged over varying top_k / model_name.")

    methods = grouped["method"].unique()
    saved_files: list[str] = []

    sns.set_theme(style="ticks", context="talk", palette="colorblind")
    plt.rcParams.update({"font.family": "serif"})

    def _create_plot(sub, baseline_col, ablation_col, title_suffix, filename_suffix, *, control_col: str | None = None, control_label: str = "Control Ablated"):
        if sub.empty:
            return
        # Sorting
        if sort_by == "drop" and baseline_col in sub and ablation_col in sub:
            sub = sub.assign(_drop=sub[baseline_col] - sub[ablation_col]).sort_values("_drop", ascending=False)
        elif sort_by == "baseline" and baseline_col in sub:
            sub = sub.sort_values(baseline_col, ascending=False)
        elif sort_by == "ablation" and ablation_col in sub:
            sub = sub.sort_values(ablation_col, ascending=False)
        else:
            sub = sub.sort_values("task_display")
            
        tasks = list(sub.task_display)
        
        base_vals = sub[baseline_col].values * (100 if percent else 1)
        abl_vals = sub[ablation_col].values * (100 if percent else 1)
        
        n_tasks = len(tasks)
        x = np.arange(n_tasks)
        
        width = 0.40
        fig_w = max(8.0, 1.2 * n_tasks + 2.5)
        fig_h = 6.0
        
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        # Determine bar ordering and spacing. We want: Baseline, Ablated, Control (if present)
        has_control = bool(control_col and control_col in sub)
        labels = ["Baseline", "Ablated"]
        value_arrays = [base_vals, abl_vals]
        colors = [None, "#ff8888"]  # Let matplotlib pick for baseline; red-ish for ablated
        if has_control:
            control_vals = sub[control_col].values * (100 if percent else 1)
            labels.append(control_label)
            value_arrays.append(control_vals)
            colors.append("#8888ff")

        n_bars = len(labels)
        # Allocate up to 0.8 of the unit interval for bars, remaining for spacing between groups
        bar_width = 0.8 / n_bars
        # Center bars around each x position
        start_offset = - ( (n_bars - 1) / 2.0 ) * bar_width
        offsets = [start_offset + i * bar_width for i in range(n_bars)]

        bar_containers = []
        for off, lab, vals, col in zip(offsets, labels, value_arrays, colors):
            bc = ax.bar(x + off, vals, bar_width * 0.95, label=lab, color=col, zorder=3, edgecolor=None)
            bar_containers.append(bc)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        
        if percent:
            ax.set_ylim(0, 105)
        else:
            max_val = float(np.max([base_vals.max() if len(base_vals) else 0,
                                    abl_vals.max() if len(abl_vals) else 0]))
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
        
        for bc, vals in zip(bar_containers, value_arrays):
            _annotate(bc, vals)
        
        if overlay_scores:
            ax.plot(x, base_vals, marker='o', color='black', linewidth=1)
            ax.plot(x, abl_vals, marker='s', color='gray', linewidth=1)
        
        ax.set_xlabel("Task")
        ax.set_ylabel("Accuracy (%)" if percent else "Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=0, ha='center')
        ax.set_title(title_suffix)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3 if (control_col and control_col in sub) else 2, frameon=True)
        fig.tight_layout(rect=[0, 0.05, 1, 1])

        out_path = out_dir / filename_suffix
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        saved_files.append(out_path.name)
        if show:
            plt.show()
        plt.close()

    for method in methods:
        method_subset = grouped[grouped["method"] == method]
        for model_disp in sorted(method_subset.model_display.unique()):
            sub = method_subset[method_subset.model_display == model_disp].copy()
            safe_model = model_disp.replace(" ", "_")
            pretty_method = METHOD_DISPLAY.get(method, method.title())
            # Train/Train
            _create_plot(
                sub, 
                "baseline_train_accuracy_mean", 
                "ablation_train_accuracy_mean",
                f"{model_disp} - {pretty_method} Train/Train (Includes Control)", 
                f"{safe_model}_{method}_train_train.png",
                control_col="ablation_control_train_accuracy_mean",
                control_label="Random Control"
            )
            
            # Val/Val if available
            if sub["baseline_val_accuracy_mean"].notna().any():
                _create_plot(
                    sub, 
                    "baseline_val_accuracy_mean", 
                    "ablation_val_accuracy_mean",
                    f"{model_disp} - {pretty_method} Val/Val (Includes Control)", 
                    f"{safe_model}_{method}_val_val.png",
                    control_col="ablation_control_val_accuracy_mean",
                    control_label="Random Control"
                )
            
            # Train/Val if available
            if (sub["baseline_train_accuracy_mean"].notna().any() and 
                sub["ablation_val_accuracy_mean"].notna().any()):
                tv = sub[[
                    "model_display", 
                    "task_display", 
                    "method", 
                    "baseline_train_accuracy_mean", 
                    "ablation_val_accuracy_mean"
                ]].copy()
                
                tv.rename(columns={
                    "baseline_train_accuracy_mean": "baseline_train_mean", 
                    "ablation_val_accuracy_mean": "ablation_val_mean"
                }, inplace=True)
                
                # For Train/Val, also include control if available (train control + val control)
                tv_full = sub[[
                    "model_display", "task_display", "method",
                    "baseline_train_accuracy_mean", "ablation_val_accuracy_mean",
                    "ablation_control_val_accuracy_mean"
                ]].copy()
                _create_plot(
                    tv_full.rename(columns={
                        "baseline_train_accuracy_mean": "baseline_train_accuracy_mean", 
                        "ablation_val_accuracy_mean": "ablation_val_accuracy_mean"
                    }), 
                    "baseline_train_accuracy_mean", 
                    "ablation_val_accuracy_mean",
                    f"{model_disp} - {pretty_method} Train/Val (Includes Control)", 
                    f"{safe_model}_{method}_train_val.png",
                    control_col="ablation_control_val_accuracy_mean",
                    control_label="Random Control"
                )

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