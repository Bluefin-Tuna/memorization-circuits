import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from circuit_reuse.dataset import (
    get_task_display_name,
    get_model_display_name,
)

METHOD_DISPLAY = {"eap": "EAP", "gradient": "Gradient"}


def safe_filename(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))
    s = re.sub(r"_+", "_", s)
    return s.strip("_.")


def discover_metrics(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("metrics.json"))


def load_metrics_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}


def aggregate(paths: List[Path]) -> pd.DataFrame:
    rows = [d for p in paths if (d := load_metrics_json(p))]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_accuracy_and_error(
    correct: np.ndarray,
    total: np.ndarray,
    z_score: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        p = correct / np.where(total == 0, np.nan, total)
    if z_score is None:
        err = np.zeros_like(p, dtype=float)
    else:
        n = total.astype(float)
        err = z_score * np.sqrt(np.clip(p, 0, 1) * np.clip(1 - p, 0, 1) / np.where(n <= 0, np.nan, n))
        err = np.nan_to_num(err)
    return np.nan_to_num(p), err


def grid_fixed_cols(n: int, default_cols: int = 3) -> Tuple[int, int]:
    if n <= 0:
        return 1, 1
    cols = min(default_cols, max(1, n))
    rows = int(math.ceil(n / cols))
    return rows, cols


def style_init():
    # Compact, clean defaults with larger fonts and automatic layout.
    sns.set_theme(style="ticks", context="talk", palette="colorblind")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 20,
        "axes.titleweight": "regular",
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlepad": 6,
        # Constrained layout rcParams must be prefixed with figure.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,
    })


def build_colors_for_models(models: List[str]) -> Dict[str, Tuple[float, float, float]]:
    base = sns.color_palette("colorblind")
    if len(models) > len(base):
        extra = sns.color_palette("tab20")
        base = base + extra
    return {m: base[i % len(base)] for i, m in enumerate(models)}


def plot_cis_per_task_per_model(gk: pd.DataFrame, out_dir: Path, models_order: Optional[List[str]] = None):
    if gk.empty or "cis_mean" not in gk.columns:
        print("[INFO] No data for cis plot.")
        return

    sub = gk.copy()

    # Use a global, consistent model order if provided
    models = models_order if models_order is not None else sorted(sub["model_display"].dropna().unique(), key=str)
    tasks = sorted(sub["task_display"].dropna().unique(), key=str)
    ks = sorted(sub["top_k"].dropna().unique())

    rows, cols = grid_fixed_cols(len(tasks), default_cols=3)

    width_per_ax = max(5.2, 0.9 * max(3, len(models)))
    height_per_ax = 4.0
    fig_w = max(12.0, cols * width_per_ax)
    fig_h = max(5.0, rows * height_per_ax)

    fig, axes = plt.subplots(
        rows, cols, figsize=(fig_w, fig_h), sharey=True, constrained_layout=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = plt.get_cmap("Blues_r")
    grad_vals = np.linspace(0.3, 0.9, max(1, len(ks)))
    k_colors = {k: cmap(grad_vals[i]) for i, k in enumerate(ks)}

    x = np.arange(len(models))
    n_k = max(1, len(ks))
    bar_w = min(0.9 / n_k, 0.9)
    offsets = np.linspace(-(n_k - 1) / 2.0, (n_k - 1) / 2.0, n_k) * bar_w

    for i, task in enumerate(tasks):
        ax = axes[i]
        tdf = sub[sub["task_display"] == task]
        for j, k in enumerate(ks):
            tk = tdf[tdf["top_k"] == k].set_index("model_display").reindex(models)
            vals = tk["cis_mean"].fillna(0.0).values.astype(float)
            bars = ax.bar(
                x + offsets[j],
                vals,
                width=bar_w,
                color=k_colors[k],
                edgecolor="none",
                label=f"k={int(k)}" if i == 0 else None,
            )
    
        ax.set_title(task)
        if i % cols == 0:
            ax.set_ylabel("Circuit identifiability score")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
        ax.margins(x=0.02, y=-0.05)

    for j in range(len(tasks), len(axes)):
        fig.delaxes(axes[j])

    handles = [plt.Rectangle((0, 0), 1, 1, color=k_colors[k]) for k in ks]
    labels = [f"k={int(k)}" for k in ks]
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(ks), 6),
        frameon=True,
        bbox_to_anchor=(0.5, -0.05),
        borderaxespad=0.2,
        handletextpad=0.6,
        columnspacing=0.8,
    )

    out = out_dir / "multiplot_cis_per_task_per_model.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[INFO] Saved {out}")


def plot_drop_by_model_k_per_task(
    gk: pd.DataFrame,
    out_dir: Path,
    split: str,
    percent: bool,
    models_order: Optional[List[str]] = None,
):
    need = {f"baseline_{split}_correct", f"baseline_{split}_total", f"ablation_{split}_correct", f"ablation_{split}_total"}
    if not need.issubset(set(gk.columns)):
        print("[INFO] Missing baseline/ablation columns; skipping drop-by-model-by-k plot.")
        return

    z = stats.norm.ppf(0.5 + 0.95 / 2.0)
    for cat in ["baseline", "ablation"]:
        c = f"{cat}_{split}_correct"
        t = f"{cat}_{split}_total"
        a = f"{cat}_{split}_acc"
        e = f"{cat}_{split}_err"
        gk[a], gk[e] = compute_accuracy_and_error(gk[c].values, gk[t].values, z)

    tasks = sorted(gk["task_display"].dropna().unique(), key=str)
    # Use a global, consistent model order if provided
    models = models_order if models_order is not None else sorted(gk["model_display"].dropna().unique(), key=str)
    ks = sorted(gk["top_k"].dropna().unique())

    # Always scale to percentage (0-100).
    scale = 100.0

    n = len(tasks)
    rows, cols = grid_fixed_cols(n, default_cols=3)

    width_per_ax = max(5.2, 0.9 * max(3, len(models)))
    height_per_ax = 4.0
    fig_w = max(12.0, cols * width_per_ax)
    fig_h = max(5.0, rows * height_per_ax)

    fig, axes = plt.subplots(
        rows, cols, figsize=(fig_w, fig_h), sharey=True, constrained_layout=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = plt.get_cmap("viridis")
    grad_vals = np.linspace(0.3, 0.9, max(1, len(ks)))
    k_colors = {k: cmap(grad_vals[i]) for i, k in enumerate(ks)}

    x = np.arange(len(models))
    n_k = max(1, len(ks))
    bar_w = min(0.9 / n_k, 0.9)
    offsets = np.linspace(-(n_k - 1) / 2.0, (n_k - 1) / 2.0, n_k) * bar_w

    for i, task in enumerate(tasks):
        ax = axes[i]
        sub = gk[gk["task_display"] == task].copy()
        for j, k in enumerate(ks):
            sk = sub[sub["top_k"] == k].set_index("model_display").reindex(models)
            drop_vals = (sk[f"baseline_{split}_acc"] - sk[f"ablation_{split}_acc"]) * scale
            drop_vals = drop_vals.fillna(0.0).astype(float).values
            bars = ax.bar(
                x + offsets[j],
                drop_vals,
                width=bar_w,
                color=k_colors[k],
                edgecolor="none",
                label=f"k={int(k)}" if i == 0 else None,
            )

        ax.set_title(task)

        if i % cols == 0:
            ax.set_ylabel("Accuracy Drop (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
        ax.margins(x=0.02, y=-0.05)

        ax.set_ylim(-60, 110)

    for j in range(len(tasks), len(axes)):
        fig.delaxes(axes[j])

    handles = [plt.Rectangle((0, 0), 1, 1, color=k_colors[k]) for k in ks]
    labels = [f"k={int(k)}" for k in ks]
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(ks), 6),
        frameon=True,
        bbox_to_anchor=(0.5, -0.05),
        borderaxespad=0.2,
        handletextpad=0.6,
        columnspacing=0.8,
    )

    out = out_dir / f"multiplot_drop_by_model_by_k_{split}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[INFO] Saved {out}")


def plot_shared_lift(
    gk: pd.DataFrame,
    out_dir: Path,
    split: str,
    models_order: Optional[List[str]] = None,
    percent: bool = True,
):
    need = {
        f"baseline_{split}_correct", f"baseline_{split}_total",
        f"ablation_{split}_correct", f"ablation_{split}_total",
        f"control_{split}_correct",  f"control_{split}_total",
    }
    if not need.issubset(set(gk.columns)):
        print("[INFO] Missing baseline/ablation/control columns; skipping rel diff plot.")
        return

    # accuracies
    for cat in ["baseline", "ablation", "control"]:
        c = f"{cat}_{split}_correct"
        t = f"{cat}_{split}_total"
        a = f"{cat}_{split}_acc"
        gk[a], _ = compute_accuracy_and_error(gk[c].values, gk[t].values, z_score=None)

    tasks = sorted(gk["task_display"].dropna().unique(), key=str)
    models = models_order if models_order is not None else sorted(gk["model_display"].dropna().unique(), key=str)
    ks = sorted(gk["top_k"].dropna().unique())

    rows, cols = grid_fixed_cols(len(tasks), default_cols=3)
    width_per_ax = max(5.2, 0.9 * max(3, len(models)))
    height_per_ax = 4.0
    fig_w = max(12.0, cols * width_per_ax)
    fig_h = max(5.0, rows * height_per_ax)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()
    
    fig.subplots_adjust(hspace=0.1)

    # color encodes k
    cmap = plt.get_cmap("viridis")
    grad = np.linspace(0.3, 0.9, max(1, len(ks)))
    k_colors = {k: cmap(grad[i]) for i, k in enumerate(ks)}

    x = np.arange(len(models))
    n_k = max(1, len(ks))
    bar_w = min(0.9 / n_k, 0.9)
    offsets = np.linspace(-(n_k - 1) / 2.0, (n_k - 1) / 2.0, n_k) * bar_w

    scale = 100.0 if percent else 1.0

    for i, task in enumerate(tasks):
        ax = axes[i]
        sub = gk[gk["task_display"] == task].copy()

        for j, k in enumerate(ks):
            sk = sub[sub["top_k"] == k].set_index("model_display").reindex(models)

            base = sk[f"baseline_{split}_acc"].values.astype(float)
            shared = sk[f"ablation_{split}_acc"].values.astype(float)
            ctrl = sk[f"control_{split}_acc"].values.astype(float)

            with np.errstate(divide="ignore", invalid="ignore"):
                denom = np.where(base <= 0, np.nan, base)
                rel_diff = (shared - ctrl) / denom

            vals = np.nan_to_num(rel_diff) * scale
            ax.bar(x + offsets[j], vals, width=bar_w, color=k_colors[k], edgecolor="none",
                   label=f"k={int(k)}" if i == 0 else None)

        ax.set_title(task)
        if i % cols == 0:
            ax.set_ylabel("Normalized Lift (%)" if percent else "Normalized Lift")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
        ax.margins(x=0.02, y=-0.05)
        ax.set_ylim(-120 if percent else -1.2, 120 if percent else 1.2)

    for j in range(len(tasks), len(axes)):
        fig.delaxes(axes[j])

    handles = [plt.Rectangle((0, 0), 1, 1, color=k_colors[k]) for k in ks]
    labels = [f"k={int(k)}" for k in ks]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(ks), 6),
               frameon=True, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Shared lift vs control, normalized by baseline", fontsize=20, y=1.1)
    fig.text(0.5, 1.04, "Formula: (shared accuracy - control accuracy) / baseline accuracy",
             ha="center", va="center", fontsize=12, color="0.35")

    out = out_dir / f"multiplot_rel_diff_shared_minus_control_over_base_{split}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[INFO] Saved {out}")
    

def parse_args():
    p = argparse.ArgumentParser(description="New multiplots for circuit reuse experiments.")
    p.add_argument("--results-dir", type=str, default="results", help="Directory containing run subdirs with metrics.json.")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to write plots (default: <results-dir>/plots_<timestamp>).")
    p.add_argument("--show", action="store_true", help="Show plots interactively if a display is available.")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Which split to visualize.")
    p.add_argument("--method", type=str, default="eap", choices=["eap", "gradient"], help="Filter to a single method.")
    return p.parse_args()


def _model_display_with_revision(row: pd.Series) -> str:
    base = get_model_display_name(row.get("model_name"))
    rev = row.get("hf_revision", None)
    if pd.isna(rev) or rev is None or str(rev) == "":
        return base
    m = re.search(r"(?:^|[-_])step(\d+)(?:$|[-_])", str(rev))
    step = f"step{m.group(1)}" if m else None
    return f"{base} {step}" if step else base

def _model_sort_key_from_display(md: str):
    # Extract "stepN" if present; otherwise treat as very large so non-steps go last.
    m = re.search(r"(?:^|\s)step(\d+)\b", md)
    step = int(m.group(1)) if m else 10**12
    # Base is model display without the trailing " stepN"
    base = re.sub(r"\s+step\d+\b", "", md).strip()
    return (base, step, md)

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    paths = discover_metrics(results_dir)
    df = aggregate(paths)
    if df.empty:
        print("[INFO] No rows. Exiting.")
        return

    df["task_display"] = df["task"].apply(get_task_display_name)
    df["model_display"] = df.apply(_model_display_with_revision, axis=1)

    df = df[df["method"] == args.method].copy()
    print(f"[INFO] Using method: {METHOD_DISPLAY.get(args.method, args.method.title())}")
    if df.empty:
        print("[INFO] No rows after method filter. Exiting.")
        return

    cols_sum = [
        "baseline_train_correct", "baseline_train_total",
        "ablation_train_correct", "ablation_train_total",
        "control_train_correct", "control_train_total",
        "baseline_val_correct", "baseline_val_total",
        "ablation_val_correct", "ablation_val_total",
        "control_val_correct", "control_val_total",
    ]
    cols_present = [c for c in cols_sum if c in df.columns]
    gk = (
        df.groupby(["task_display", "model_display", "top_k"], as_index=False)
        .agg(**{c: (c, "sum") for c in cols_present},
             cis_mean=("circuit_identifiability_score", "mean"))
    )

    # Use numeric step ordering within each base model
    models_unique = list(gk["model_display"].dropna().unique())
    models_order = sorted(models_unique, key=_model_sort_key_from_display)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / f"plots_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "aggregated_by_task_model_k.csv"
    gk.to_csv(csv_path, index=False)
    print(f"[INFO] Aggregated CSV saved to {csv_path}")

    style_init()

    # Always use percentage scaling in plots.
    percent = True
    plot_drop_by_model_k_per_task(gk.copy(), out_dir, split=args.split, percent=percent, models_order=models_order)
    plot_cis_per_task_per_model(gk.copy(), out_dir, models_order=models_order)
    plot_shared_lift(gk.copy(), out_dir, split=args.split, models_order=models_order, percent=True)

    if args.show:
        plt.show()
        
if __name__ == "__main__":
    main()
