from __future__ import annotations

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

METHOD_DISPLAY = {
    "eap": "EAP",
    "gradient": "Gradient",
}


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
    sns.set_theme(style="ticks", context="talk", palette="colorblind")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 18,
        "axes.titleweight": "regular",
        "axes.labelsize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })


def build_colors_for_models(models: List[str]) -> Dict[str, Tuple[float, float, float]]:
    base = sns.color_palette("colorblind")
    if len(models) > len(base):
        extra = sns.color_palette("tab20")
        base = base + extra
    return {m: base[i % len(base)] for i, m in enumerate(models)}


def plot_drop_vs_k_by_task(
    gk: pd.DataFrame,
    out_dir: Path,
    split: str,
    percent: bool,
):
    if gk.empty:
        print("[INFO] No data for drop-vs-k.")
        return

    z = stats.norm.ppf(0.5 + 0.95 / 2.0)
    for cat in ["baseline", "ablation"]:
        c = f"{cat}_{split}_correct"
        t = f"{cat}_{split}_total"
        a = f"{cat}_{split}_acc"
        e = f"{cat}_{split}_err"
        gk[a], gk[e] = compute_accuracy_and_error(gk[c].values, gk[t].values, z)

    tasks = sorted(gk["task_display"].dropna().unique(), key=str)
    models = sorted(gk["model_display"].dropna().unique(), key=str)
    colors = build_colors_for_models(models)
    scale = 100.0 if percent else 1.0

    n = len(tasks)
    rows, cols = grid_fixed_cols(n, default_cols=3)
    width_per_ax = 6.6
    height_per_ax = 4.6
    fig_w = max(12.0, cols * width_per_ax)
    fig_h = max(5.5, rows * height_per_ax) + 0.6
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, task in enumerate(tasks):
        ax = axes[i]
        sub = gk[gk["task_display"] == task].copy()
        for mdl in models:
            s = sub[sub["model_display"] == mdl]
            if s.empty:
                continue
            s = s.sort_values("top_k")
            drop = (s[f"baseline_{split}_acc"].values - s[f"ablation_{split}_acc"].values) * scale
            ax.plot(
                s["top_k"].values,
                drop,
                marker="o",
                linewidth=2.0,
                color=colors[mdl],
                label=mdl if i == 0 else None,
            )
        ax.set_xlabel("top_k")
        if i % cols == 0:
            ax.set_ylabel("Baseline - Ablation (pp)" if percent else "Baseline - Ablation")
        ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
        ax.set_ylim(-50 if percent else -0.5, 110 if percent else 1.1)

    for j in range(len(tasks), len(axes)):
        fig.delaxes(axes[j])

    handles = [plt.Line2D([0], [0], color=colors[m], lw=2.0, marker="o") for m in models]
    labels = models
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(models),
        frameon=True,
        borderaxespad=0.4,
        bbox_to_anchor=(0.5, -0.1)
    )

    out = out_dir / f"multiplot_drop_vs_k_{split}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {out}")


def plot_reuse_per_task_per_model(
    gk: pd.DataFrame,
    out_dir: Path,
    percent: bool,
):
    if gk.empty or "shared_circuit_size_mean" not in gk.columns:
        print("[INFO] No data for circuit reuse plot.")
        return

    sub = gk.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        sub["reuse_frac"] = sub["shared_circuit_size_mean"] / sub["top_k"].replace(0, np.nan)
    sub["reuse_frac"] = sub["reuse_frac"].clip(lower=0).fillna(0.0)
    scale = 100.0 if percent else 1.0

    models = sorted(sub["model_display"].dropna().unique(), key=str)
    tasks = sorted(sub["task_display"].dropna().unique(), key=str)
    ks = sorted(sub["top_k"].dropna().unique())

    rows, cols = grid_fixed_cols(len(models), default_cols=3)

    width_per_ax = max(6.0, 1.45 * max(3, len(tasks)))
    height_per_ax = 4.6
    fig_w = max(14.0, cols * width_per_ax)
    fig_h = max(5.8, rows * height_per_ax)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    fig.subplots_adjust(left=0.055, right=0.995, top=0.965, wspace=0.06, hspace=0.30)

    cmap = plt.get_cmap("Blues_r")
    grad_vals = np.linspace(0.25, 0.9, max(1, len(ks)))
    k_colors = {k: cmap(grad_vals[i]) for i, k in enumerate(ks)}

    x = np.arange(len(tasks))
    n_k = max(1, len(ks))
    bar_w = min(0.94 / n_k, 0.94)
    offsets = np.linspace(-(n_k - 1) / 2.0, (n_k - 1) / 2.0, n_k) * bar_w

    for i, mdl in enumerate(models):
        ax = axes[i]
        m = sub[sub["model_display"] == mdl]
        rects_all = []
        for j, k in enumerate(ks):
            mk = m[m["top_k"] == k].set_index("task_display").reindex(tasks)
            vals = (mk["reuse_frac"].values * scale).astype(float)
            rects = ax.bar(
                x + offsets[j],
                vals,
                width=bar_w,
                color=k_colors[k],
                edgecolor="none",
                label=f"k={int(k)}",
            )
            rects_all.append(rects)

        ax.set_title(mdl)
        ax.set_xlabel("Task")
        if i % cols == 0:
            ax.set_ylabel("Circuit reuse (%)" if percent else "Circuit reuse (fraction)")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=0, ha="center")
        if percent:
            ax.set_ylim(0, 110)
        ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
        ax.margins(x=0.02, y=0.035)

        ymin, ymax = ax.get_ylim()
        ypad = (ymax - ymin) * 0.016
        for rects in rects_all:
            for r in rects:
                h = r.get_height()
                if np.isfinite(h):
                    ax.text(
                        r.get_x() + r.get_width() / 2.0,
                        h + ypad,
                        f"{h:.1f}" if percent else f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

    for j in range(len(models), len(axes)):
        fig.delaxes(axes[j])

    handles = [plt.Rectangle((0, 0), 1, 1, color=k_colors[k]) for k in ks]
    labels = [f"k={int(k)}" for k in ks]
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(ks),
        frameon=True,
        borderaxespad=0.4,
        bbox_to_anchor=(0.5, 0)
    )

    out = out_dir / "multiplot_reuse_per_task_per_model.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {out}")


def plot_ablation_vs_control_by_task(
    gk: pd.DataFrame,
    out_dir: Path,
    split: str,
    percent: bool,
):
    need_cols = {f"ablation_{split}_correct", f"ablation_{split}_total", f"control_{split}_correct", f"control_{split}_total"}
    if not need_cols.issubset(set(gk.columns)):
        print("[INFO] Control or ablation columns missing; skipping ablation vs control diff plot.")
        return

    # Compute accuracies for ablation and control
    z = stats.norm.ppf(0.5 + 0.95 / 2.0)
    for cat in ["ablation", "control"]:
        c = f"{cat}_{split}_correct"
        t = f"{cat}_{split}_total"
        a = f"{cat}_{split}_acc"
        e = f"{cat}_{split}_err"
        gk[a], gk[e] = compute_accuracy_and_error(gk[c].values, gk[t].values, z)

    tasks = sorted(gk["task_display"].dropna().unique(), key=str)
    models = sorted(gk["model_display"].dropna().unique(), key=str)
    ks = sorted(gk["top_k"].dropna().unique())

    scale = 100.0 if percent else 1.0

    # Figure layout mirrored to drop_by_model_k_per_task: x-axis models, bars = k (viridis)
    n = len(tasks)
    rows, cols = grid_fixed_cols(n, default_cols=3)
    width_per_ax = max(7.2, 1.6 * max(3, len(models)))
    height_per_ax = 5.0
    fig_w = max(16.0, cols * width_per_ax)
    fig_h = max(6.2, rows * height_per_ax) + 0.4
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    # Colors for k (viridis gradient)
    cmap = plt.get_cmap("viridis")
    grad_vals = np.linspace(0.25, 0.9, max(1, len(ks)))
    k_colors = {k: cmap(grad_vals[i]) for i, k in enumerate(ks)}

    # Grouped bar positions: clusters at each model, bars per k
    x = np.arange(len(models))
    n_k = max(1, len(ks))
    bar_w = min(0.9 / n_k, 0.9)
    offsets = np.linspace(-(n_k - 1) / 2.0, (n_k - 1) / 2.0, n_k) * bar_w

    for i, task in enumerate(tasks):
        ax = axes[i]
        sub = gk[gk["task_display"] == task].copy()
        rects_all = []
        for j, k in enumerate(ks):
            sk = sub[sub["top_k"] == k].set_index("model_display").reindex(models)
            diff_vals = (sk[f"control_{split}_acc"] - sk[f"ablation_{split}_acc"]) * scale
            diff_vals = diff_vals.fillna(0.0).astype(float).values
            rects = ax.bar(
                x + offsets[j],
                diff_vals,
                width=bar_w,
                color=k_colors[k],
                edgecolor="none",
                label=f"k={int(k)}" if i == 0 else None,
            )
            rects_all.append(rects)

        ax.set_title(task)
        ax.set_xlabel("Model")
        if i % cols == 0:
            ax.set_ylabel("Control - Ablation (pp)" if percent else "Control - Ablation")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
        ax.margins(x=0.02, y=0.04)
        ax.set_ylim(-50 if percent else -0.5, 110 if percent else 1.1)

    for j in range(len(tasks), len(axes)):
        fig.delaxes(axes[j])

    handles = [plt.Rectangle((0, 0), 1, 1, color=k_colors[k]) for k in ks]
    labels = [f"k={int(k)}" for k in ks]
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(ks),
        frameon=True,
        borderaxespad=0.4,
        bbox_to_anchor=(0.5, -0.15),
        fontsize=14
    )

    out = out_dir / f"multiplot_ablation_vs_control_{split}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {out}")


def plot_drop_by_model_k_per_task(
    gk: pd.DataFrame,
    out_dir: Path,
    split: str,
    percent: bool,
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
    models = sorted(gk["model_display"].dropna().unique(), key=str)
    ks = sorted(gk["top_k"].dropna().unique())

    scale = 100.0 if percent else 1.0

    n = len(tasks)
    rows, cols = grid_fixed_cols(n, default_cols=3)
    width_per_ax = max(7.2, 1.6 * max(3, len(models)))
    height_per_ax = 5.0
    fig_w = max(16.0, cols * width_per_ax)
    fig_h = max(6.2, rows * height_per_ax) + 0.4
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = plt.get_cmap("viridis")
    grad_vals = np.linspace(0.25, 0.9, max(1, len(ks)))
    k_colors = {k: cmap(grad_vals[i]) for i, k in enumerate(ks)}

    x = np.arange(len(models))
    n_k = max(1, len(ks))
    bar_w = min(0.9 / n_k, 0.9)
    offsets = np.linspace(-(n_k - 1) / 2.0, (n_k - 1) / 2.0, n_k) * bar_w

    for i, task in enumerate(tasks):
        ax = axes[i]
        sub = gk[gk["task_display"] == task].copy()
        rects_all = []
        for j, k in enumerate(ks):
            sk = sub[sub["top_k"] == k].set_index("model_display").reindex(models)
            drop_vals = (sk[f"baseline_{split}_acc"] - sk[f"ablation_{split}_acc"]) * scale
            drop_vals = drop_vals.fillna(0.0).astype(float).values
            rects = ax.bar(
                x + offsets[j],
                drop_vals,
                width=bar_w,
                color=k_colors[k],
                edgecolor="none",
                label=f"k={int(k)}" if i == 0 else None,
            )
            rects_all.append(rects)

        ax.set_title(task)
        ax.set_xlabel("Model")
        if i % cols == 0:
            ax.set_ylabel("Accuracy Drop (pp)" if percent else "Accuracy Drop")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
        ax.margins(x=0.02, y=0.04)

    for j in range(len(tasks), len(axes)):
        fig.delaxes(axes[j])

    handles = [plt.Rectangle((0, 0), 1, 1, color=k_colors[k]) for k in ks]
    labels = [f"k={int(k)}" for k in ks]
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(ks),
        frameon=True,
        borderaxespad=0.4,
        bbox_to_anchor=(0.5, -0.15),
        fontsize=14
    )

    out = out_dir / f"multiplot_drop_by_model_by_k_{split}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {out}")

def parse_args():
    p = argparse.ArgumentParser(description="New multiplots for circuit reuse experiments.")
    p.add_argument("--results-dir", type=str, default="results", help="Directory containing run subdirs with metrics.json.")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to write plots (default: <results-dir>/plots_<timestamp>).")
    p.add_argument("--show", action="store_true", help="Show plots interactively if a display is available.")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Which split to visualize.")
    p.add_argument("--percent", action="store_true", default=True, help="Scale to percentage 0-100 (default True).")
    p.add_argument("--method", type=str, default="eap", choices=["eap", "gradient"], help="Filter to a single method.")
    return p.parse_args()

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    paths = discover_metrics(results_dir)
    df = aggregate(paths)
    if df.empty:
        print("[INFO] No rows. Exiting.")
        return

    df["task_display"] = df["task"].apply(get_task_display_name)
    df["model_display"] = df["model_name"].apply(get_model_display_name)

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
             shared_circuit_size_mean=("shared_circuit_size", "mean"))
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / f"plots_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "aggregated_by_task_model_k.csv"
    gk.to_csv(csv_path, index=False)
    print(f"[INFO] Aggregated CSV saved to {csv_path}")

    style_init()

    plot_drop_vs_k_by_task(gk.copy(), out_dir, split=args.split, percent=args.percent)
    plot_drop_by_model_k_per_task(gk.copy(), out_dir, split=args.split, percent=args.percent)
    plot_reuse_per_task_per_model(gk.copy(), out_dir, percent=args.percent)
    plot_ablation_vs_control_by_task(gk.copy(), out_dir, split=args.split, percent=args.percent)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
