from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
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
    p.add_argument("--no-control", action="store_true",
                   help="If set, do not plot Control results.")
    p.add_argument("--raw", action="store_true",
                   help="Show raw accuracies (0-1). Overrides --percent default.")
    p.add_argument("--ci", type=float, default=0.95,
                   help="Confidence level for error bars (e.g., 0.95 for 95%% CI).")
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
        "baseline_train_accuracy", "ablation_train_accuracy", "control_train_accuracy",
        "baseline_val_accuracy", "ablation_val_accuracy", "control_val_accuracy",
        "accuracy_drop_train", "accuracy_drop_val", "accuracy_drop_train_val",
        "control_accuracy_drop_train", "control_accuracy_drop_val", "control_accuracy_drop_train_val",
        "baseline_train_correct", "baseline_train_total",
        "ablation_train_correct", "ablation_train_total",
        "control_train_correct", "control_train_total",
        "baseline_val_correct", "baseline_val_total",
        "ablation_val_correct", "ablation_val_total",
        "control_val_correct", "control_val_total",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def wilson_score_interval(p: float, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0 or p < 0 or p > 1:
        return (0, 0)
    # Wilson score interval calculation
    p_z = p + z**2 / (2 * n)
    term = z * np.sqrt((p * (1 - p)) / n + z**2 / (4 * n**2))
    denominator = 1 + z**2 / n
    lower = (p_z - term) / denominator
    upper = (p_z + term) / denominator
    # Return error bar sizes (distance from p)
    return p - lower, upper - p

def plot_all(df: pd.DataFrame, out_dir: Path, show: bool, sort_by: str, *,
             percent: bool, overlay_scores: bool, include_control: bool, ci_level: float):
    if df.empty:
        print("[INFO] No data to plot.")
        return

    z_score = stats.norm.ppf((1 + ci_level) / 2)

    agg_spec = {
        "baseline_train_correct": ("baseline_train_correct", "sum"),
        "baseline_train_total": ("baseline_train_total", "sum"),
        "ablation_train_correct": ("ablation_train_correct", "sum"),
        "ablation_train_total": ("ablation_train_total", "sum"),
        "control_train_correct": ("control_train_correct", "sum"),
        "control_train_total": ("control_train_total", "sum"),
        "baseline_val_correct": ("baseline_val_correct", "sum"),
        "baseline_val_total": ("baseline_val_total", "sum"),
        "ablation_val_correct": ("ablation_val_correct", "sum"),
        "ablation_val_total": ("ablation_val_total", "sum"),
        "control_val_correct": ("control_val_correct", "sum"),
        "control_val_total": ("control_val_total", "sum"),
        "shared_circuit_size_mean": ("shared_circuit_size", "mean"),
        "top_k_mean": ("top_k", "mean"),
        "runs": ("baseline_train_accuracy", "count")
    }
    
    final_agg_spec = {k: v for k, v in agg_spec.items() if v[0] in df.columns}
    grouped = (df
               .groupby(["model_display", "task_display", "method"], as_index=False)
               .agg(**final_agg_spec))

    # Calculate accuracies and CIs from aggregated sums
    for split in ["train", "val"]:
        for cat in ["baseline", "ablation", "control"]:
            correct_col = f"{cat}_{split}_correct"
            total_col = f"{cat}_{split}_total"
            acc_col = f"{cat}_{split}_accuracy_mean"
            err_col = f"{cat}_{split}_error"

            if correct_col in grouped.columns and total_col in grouped.columns:
                grouped[acc_col] = grouped[correct_col] / grouped[total_col].replace(0, np.nan)
                errors = grouped.apply(
                    lambda row: wilson_score_interval(row[acc_col], row[total_col], z_score),
                    axis=1
                )
                # Store error tuple and then split into two columns for yerr
                grouped[err_col] = list(errors)

    if {"top_k", "model_name"}.intersection(df.columns) and \
       (df[["model_name", "task", "method"]].drop_duplicates().shape[0] != grouped.shape[0]):
        print("[INFO] Aggregation averaged over varying top_k / model_name.")

    methods = grouped["method"].unique()
    saved_files: list[str] = []

    sns.set_theme(style="ticks", context="talk", palette="colorblind")
    plt.rcParams.update({"font.family": "serif"})
    base_palette = sns.color_palette("colorblind")

    def _create_plot(sub: pd.DataFrame,
                     baseline_col: str,
                     ablation_col: str,
                     title_suffix: str,
                     filename_suffix: str,
                     control_col: Optional[str] = None,
                     control_label: str = "Control"):
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
        has_control = control_col is not None and (control_col in sub)
        ctrl_vals = sub[control_col].values * (100 if percent else 1) if has_control else None
        
        base_err_col = baseline_col.replace("_mean", "_error")
        abl_err_col = ablation_col.replace("_mean", "_error")
        base_err = np.array(sub[base_err_col].tolist()) * (100 if percent else 1) if base_err_col in sub else None
        abl_err = np.array(sub[abl_err_col].tolist()) * (100 if percent else 1) if abl_err_col in sub else None
        
        ctrl_err = None
        if has_control:
            ctrl_err_col = control_col.replace("_mean", "_error")
            if ctrl_err_col in sub:
                ctrl_err = np.array(sub[ctrl_err_col].tolist()) * (100 if percent else 1)

        n_tasks = len(tasks)
        x = np.arange(n_tasks)
        
        # Bars
        labels = ["Baseline", "Ablated"]
        value_arrays = [base_vals, abl_vals]
        error_arrays = [base_err, abl_err]
        # replaced custom hex colors with palette colors
        colors = [base_palette[0], base_palette[1]]
        if has_control:
            labels.append(control_label)
            value_arrays.append(ctrl_vals)
            error_arrays.append(ctrl_err)
            colors.append(base_palette[2])

        n_bars = len(labels)
        bar_width = 0.8 / n_bars
        start_offset = -((n_bars - 1) / 2.0) * bar_width
        offsets = [start_offset + i * bar_width for i in range(n_bars)]

        fig_w = max(8.0, 1.2 * n_tasks + 2.5)
        fig_h = 6.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        bar_containers = []
        for off, lab, vals, err, col in zip(offsets, labels, value_arrays, error_arrays, colors):
            bc = ax.bar(
                x + off,
                vals,
                bar_width * 0.95,
                label=lab,
                color=col,
                yerr=err.T if err is not None else None,
                capsize=3,
                edgecolor='none',
                linewidth=0
            )
            bar_containers.append(bc)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        
        if percent:
            ax.set_ylim(0, 105)
        else:
            max_val = float(np.max([np.max(v) for v in value_arrays if v is not None])) if value_arrays else 1.0
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

        # Annotate bars
        if len(bar_containers) >= 1:
            _annotate(bar_containers[0], base_vals)
        if len(bar_containers) >= 2:
            _annotate(bar_containers[1], abl_vals)
        if has_control and len(bar_containers) >= 3 and ctrl_vals is not None:
            _annotate(bar_containers[2], ctrl_vals)
        
        # Optional overlay
        if overlay_scores:
            ax.plot(x, base_vals, marker='o', color=base_palette[0], linewidth=1, label="_nolegend_")
            ax.plot(x, abl_vals, marker='s', color=base_palette[1], linewidth=1, label="_nolegend_")
            if has_control and ctrl_vals is not None:
                ax.plot(x, ctrl_vals, marker='^', color=base_palette[2], linewidth=1, label="_nolegend_")
        
        ax.set_xlabel("Task")
        ax.set_ylabel("Accuracy (%)" if percent else "Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=0, ha='center')
        ax.set_title(title_suffix)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=n_bars, frameon=True)
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
                f"{model_disp} - {pretty_method} Train/Train",
                f"{safe_model}_{method}_train_train.png",
                control_col="control_train_accuracy_mean" if include_control else None,
                control_label="Control"
            )
            # Val/Val if available
            if sub["baseline_val_accuracy_mean"].notna().any():
                _create_plot(
                    sub,
                    "baseline_val_accuracy_mean",
                    "ablation_val_accuracy_mean",
                    f"{model_disp} - {pretty_method} Val/Val",
                    f"{safe_model}_{method}_val_val.png",
                    control_col="control_val_accuracy_mean" if include_control else None,
                    control_label="Control"
                )
            # Train/Val if available
            if (sub["baseline_train_accuracy_mean"].notna().any() and
                sub["ablation_val_accuracy_mean"].notna().any()):
                tv = sub[[
                    "model_display", 
                    "task_display", 
                    "method", 
                    "baseline_train_accuracy_mean", 
                    "ablation_val_accuracy_mean",
                    "control_val_accuracy_mean"
                ]].copy()
                
                tv.rename(columns={
                    "baseline_train_accuracy_mean": "baseline_train_mean", 
                    "ablation_val_accuracy_mean": "ablation_val_mean",
                    "control_val_accuracy_mean": "control_val_mean"
                }, inplace=True)
                
                tv2 = tv.rename(columns={
                    "baseline_train_mean": "baseline_train_accuracy_mean", 
                    "ablation_val_mean": "ablation_val_accuracy_mean",
                    "control_val_mean": "control_val_accuracy_mean"
                })

                _create_plot(
                    tv2,
                    "baseline_train_accuracy_mean",
                    "ablation_val_accuracy_mean",
                    f"{model_disp} - {pretty_method} Train/Val",
                    f"{safe_model}_{method}_train_val.png",
                    control_col="control_val_accuracy_mean" if include_control else None,
                    control_label="Control"
                )

            # Circuit reuse plot (shared circuit size / top_k). Uses train extraction only; no control.
            if sub["shared_circuit_size_mean"].notna().any() and sub["top_k_mean"].notna().any():
                reuse_df = sub[["model_display", "task_display", "method", "shared_circuit_size_mean", "top_k_mean"]].copy()
                # Avoid division by zero
                reuse_df = reuse_df[reuse_df["top_k_mean"] > 0]
                if not reuse_df.empty:
                    reuse_df["reuse_fraction"] = reuse_df["shared_circuit_size_mean"] / reuse_df["top_k_mean"].clip(lower=1e-9)
                    if percent:
                        reuse_df["reuse_plot_value"] = reuse_df["reuse_fraction"] * 100.0
                    else:
                        reuse_df["reuse_plot_value"] = reuse_df["reuse_fraction"]
                    # Sorting for reuse: by value desc if sort_by in certain modes, else task name
                    if sort_by in {"drop", "baseline", "ablation"}:
                        reuse_df = reuse_df.sort_values("reuse_plot_value", ascending=False)
                    else:
                        reuse_df = reuse_df.sort_values("task_display")
                    tasks = list(reuse_df.task_display)
                    vals = reuse_df["reuse_plot_value"].values
                    x = np.arange(len(tasks))
                    fig_w = max(7.0, 0.9 * len(tasks) + 3.0)
                    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
                    line_color = sns.color_palette("colorblind")[3 % len(sns.color_palette("colorblind"))]
                    ax.plot(x, vals, marker='o', linewidth=2, color=line_color)
                    for xi, v, task in zip(x, vals, tasks):
                        label = f"{v:.1f}%" if percent else f"{v:.2f}"
                        ax.annotate(label,
                                    (xi, v),
                                    xytext=(0, 6 if percent else 0.02),
                                    textcoords="offset points",
                                    ha="center", va="bottom", fontsize=10)
                    ax.set_ylabel("Shared Circuit (% of top_k)" if percent else "Shared Circuit Fraction")
                    ax.set_xlabel("Task")
                    ylim_top = 100 if percent else 1.0
                    ypad = 5 if percent else 0.05
                    ax.set_ylim(0, ylim_top + ypad)
                    ax.set_xticks(x)
                    ax.set_xticklabels(tasks, rotation=0)
                    ax.set_title(f"{model_disp} - {pretty_method} Circuit Reuse")
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    fig.tight_layout()
                    reuse_fname = f"{safe_model}_{method}_reuse.png"
                    plt.savefig(out_dir / reuse_fname, dpi=200, bbox_inches="tight")
                    saved_files.append(reuse_fname)
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
        include_control=not args.no_control,
        ci_level=args.ci,
    )

if __name__ == "__main__":
    main()
