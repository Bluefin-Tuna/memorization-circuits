from __future__ import annotations

import argparse
import json
from datetime import datetime
import re
from pathlib import Path
from typing import Any, Dict, List

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plots for circuit identifiability experiments.")
    p.add_argument("--results-dir", type=str, default="results", help="Dir with per-run subdirs containing metrics.json.")
    p.add_argument("--output-dir", type=str, default=None, help="Where to write plots (default: <results-dir>/plots_<ts>).")
    p.add_argument("--save-csv-name", type=str, default="aggregated_metrics.csv", help="CSV filename inside output dir.")
    p.add_argument("--show", action="store_true", help="Show plots interactively.")
    p.add_argument("--percent", action="store_true", default=True, help="Show accuracies in percent.")
    p.add_argument("--ci", type=float, default=-1, help="Confidence level for CI bars. -1 disables.")
    return p.parse_args()


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
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def to_display_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["task_display"] = df["task"].apply(get_task_display_name)
    df["model_display"] = df["model_name"].apply(get_model_display_name)
    df["method_display"] = df.get("method", pd.Series(dtype=str)).map(METHOD_DISPLAY).fillna(df.get("method", pd.Series(dtype=str)).str.title())
    return df


def compute_ci(p: np.ndarray, n: np.ndarray, z: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        err = z * np.sqrt(np.clip(p, 0, 1) * np.clip(1 - p, 0, 1) / np.where(n <= 0, np.nan, n))
    return np.nan_to_num(err)


def plot_accuracy_bars(df: pd.DataFrame, out_dir: Path, split: str = "train", percent: bool = True, ci_level: float = -1.0, show: bool = False):
    if df.empty:
        return
    cols_needed = {f"baseline_{split}_correct", f"baseline_{split}_total", f"ablation_{split}_correct", f"ablation_{split}_total", f"control_{split}_correct", f"control_{split}_total"}
    if not cols_needed.issubset(set(df.columns)):
        print(f"[INFO] Missing accuracy cols for split={split}; skipping bars.")
        return

    grouped = (
        df.groupby(["model_display", "task_display", "method_display", "top_k"], as_index=False)
        .agg(**{c: (c, "sum") for c in cols_needed})
    )

    scale = 100 if percent else 1
    z = stats.norm.ppf((1 + ci_level) / 2) if ci_level != -1 else None

    for (model_disp, method_disp, k_val), sub in grouped.groupby(["model_display", "method_display", "top_k"], as_index=False):
        safe_model = safe_filename(model_disp)
        safe_method = safe_filename(str(method_disp).lower())
        safe_k = int(k_val)
        sub = sub.sort_values("task_display")
        base_p = sub[f"baseline_{split}_correct"] / sub[f"baseline_{split}_total"].replace(0, np.nan)
        abl_p = sub[f"ablation_{split}_correct"] / sub[f"ablation_{split}_total"].replace(0, np.nan)
        ctrl_p = sub[f"control_{split}_correct"] / sub[f"control_{split}_total"].replace(0, np.nan)

        base_e = compute_ci(base_p.values, sub[f"baseline_{split}_total"].values, z) if z is not None else np.zeros_like(base_p.values)
        abl_e = compute_ci(abl_p.values, sub[f"ablation_{split}_total"].values, z) if z is not None else np.zeros_like(abl_p.values)
        ctrl_e = compute_ci(ctrl_p.values, sub[f"control_{split}_total"].values, z) if z is not None else np.zeros_like(ctrl_p.values)

        tasks = list(sub["task_display"].values)
        x = np.arange(len(tasks))
        width = 0.25

        plt.figure(figsize=(max(8.0, 1.2 * len(tasks) + 2.5), 7))
        plt.bar(x - width, base_p.values * scale, width, label="Baseline", yerr=base_e * scale if z is not None else None, capsize=3)
        plt.bar(x,         abl_p.values * scale, width, label="Ablated (shared)", yerr=abl_e * scale if z is not None else None, capsize=3)
        plt.bar(x + width, ctrl_p.values * scale, width, label="Control (random)", yerr=ctrl_e * scale if z is not None else None, capsize=3)

        plt.xticks(x, tasks, rotation=30, ha="right")
        plt.ylabel("Accuracy (%)" if percent else "Accuracy")
        plt.title(f"{model_disp} • k={safe_k}")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        outp = out_dir / f"{safe_model}/{safe_k}/{safe_model}_{safe_method}_k{safe_k}_{split}.png"
        outp.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(outp, dpi=200)
        if show:
            plt.show()
        plt.close()


def plot_identifiability_vs_k(df: pd.DataFrame, out_dir: Path, show: bool = False):
    if df.empty or "circuit_identifiability_score" not in df.columns:
        return
    gk = (
        df.groupby(["model_display", "task_display", "method_display", "top_k"], as_index=False)
        .agg(cis=("circuit_identifiability_score", "mean"))
        .sort_values("top_k")
    )
    for (model_disp, method_disp), sub in gk.groupby(["model_display", "method_display"], as_index=False):
        safe_model = safe_filename(model_disp)
        safe_method = safe_filename(str(method_disp).lower())
        tasks = sorted(sub["task_display"].unique())
        plt.figure(figsize=(max(8.0, 1.2 * len(tasks) + 2.5), 7))
        cmap = plt.get_cmap("Blues_r")
        ks = sorted(sub["top_k"].unique())
        grads = np.linspace(0.3, 0.9, max(1, len(ks)))
        for i, k in enumerate(ks):
            s = sub[sub["top_k"] == k].set_index("task_display").reindex(tasks)
            plt.plot(np.arange(len(tasks)), s["cis"].values, marker="o", color=cmap(grads[i]), label=f"k={int(k)}")
        plt.xticks(np.arange(len(tasks)), tasks, rotation=30, ha="right")
        plt.ylabel("Circuit identifiability score")
        plt.title(f"{model_disp}")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=min(6, len(ks)))
        outp = out_dir / f"{safe_model}/{safe_model}_{safe_method}_cis_vs_k.png"
        outp.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(outp, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def plot_knockout_diff_vs_k(df: pd.DataFrame, out_dir: Path, split: str = "train", show: bool = False):
    col = f"knockout_diff_{split}"
    if df.empty or col not in df.columns:
        return
    gk = (
        df.groupby(["model_display", "task_display", "method_display", "top_k"], as_index=False)
        .agg(kd=(col, "mean"))
        .sort_values("top_k")
    )
    for (model_disp, method_disp), sub in gk.groupby(["model_display", "method_display"], as_index=False):
        safe_model = safe_filename(model_disp)
        safe_method = safe_filename(str(method_disp).lower())
        tasks = sorted(sub["task_display"].unique())
        plt.figure(figsize=(max(8.0, 1.2 * len(tasks) + 2.5), 7))
        cmap = plt.get_cmap("viridis")
        ks = sorted(sub["top_k"].unique())
        grads = np.linspace(0.3, 0.9, max(1, len(ks)))
        for i, k in enumerate(ks):
            s = sub[sub["top_k"] == k].set_index("task_display").reindex(tasks)
            plt.plot(np.arange(len(tasks)), s["kd"].values, marker="o", color=cmap(grads[i]), label=f"k={int(k)}")
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        plt.xticks(np.arange(len(tasks)), tasks, rotation=30, ha="right")
        plt.ylabel("Knockout diff = (base - shared) / (base - random)")
        plt.title(f"{model_disp} • {split.title()}")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=min(6, len(ks)))
        outp = out_dir / f"{safe_model}/{safe_model}_{safe_method}_knockout_{split}_vs_k.png"
        outp.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(outp, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    paths = discover_metrics(results_dir)
    df = aggregate(paths)
    if df.empty:
        print("[INFO] No metrics.json found.")
        return

    df = to_display_cols(df)

    percent = not args.percent is False
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / f"plots_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / args.save_csv_name
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Aggregated CSV saved to {csv_path}")

    sns.set_theme(style="ticks", context="talk", palette="colorblind")
    plt.rcParams.update({"font.family": "serif"})

    plot_accuracy_bars(df, out_dir, split="train", percent=percent, ci_level=args.ci, show=args.show)
    if df.get("baseline_val_total", pd.Series([0])).sum() > 0:
        plot_accuracy_bars(df, out_dir, split="val", percent=percent, ci_level=args.ci, show=args.show)

    # New plots
    plot_identifiability_vs_k(df, out_dir, show=args.show)
    plot_knockout_diff_vs_k(df, out_dir, split="train", show=args.show)
    if df.get("baseline_val_total", pd.Series([0])).sum() > 0:
        plot_knockout_diff_vs_k(df, out_dir, split="val", show=args.show)


if __name__ == "__main__":
    main()
