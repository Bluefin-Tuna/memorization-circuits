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

from circuit_reuse.dataset import (
    get_task_display_name,
    get_model_display_name,
)

METHOD_DISPLAY = {"eap": "EAP", "gradient": "Gradient"}

def _extract_step_from_revision(rev: str) -> str | None:
    m = re.search(r"(?:^|[-_])step(\d+)(?:$|[-_])", str(rev) if rev is not None else "")
    return f"step{m.group(1)}" if m else None

def _model_display_with_revision(row: pd.Series) -> str:
    base = get_model_display_name(row.get("model_name"))
    rev = row.get("hf_revision", None)
    if pd.isna(rev) or rev is None or str(rev) == "":
        return base
    step = _extract_step_from_revision(str(rev))
    return f"{base} {step}" if step else base

def safe_filename(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))
    s = re.sub(r"_+", "_", s)
    return s.strip("_.")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretty multiplots for circuit identifiability experiments.")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--save-csv-name", type=str, default="aggregated_metrics.csv")
    p.add_argument("--show", action="store_true")
    p.add_argument("--percent", action="store_true", default=True)
    p.add_argument("--ci", type=float, default=-1)
    p.add_argument("--reuse-threshold", type=int, default=100, help="For v2 metrics: pick threshold p to plot.")
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

def _expand_v2(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten metrics.json v2 into rows per (top_k, reuse_threshold)."""
    rows: List[Dict[str, Any]] = []
    base = {
        "version": r.get("version", 1),
        "model_name": r.get("model_name"),
        "hf_revision": r.get("hf_revision"),
        "task": r.get("task"),
        "method": r.get("method"),
        "num_examples": r.get("num_examples"),
        # baseline at top-level
        "baseline_train_accuracy": r.get("baseline_train_accuracy"),
        "baseline_val_accuracy": r.get("baseline_val_accuracy"),
    }
    by_k = (r.get("by_k") or {})
    for k_str, block in by_k.items():
        try:
            K = int(k_str)
        except Exception:
            continue
        thresholds = (block.get("thresholds") or {})
        for p_str, tblock in thresholds.items():
            try:
                P = int(p_str)
            except Exception:
                continue
            tr = (tblock.get("train") or {})
            va = (tblock.get("val") or {})
            row = dict(base)
            row.update({
                "top_k": K,
                "reuse_threshold": P,
                # accuracies to drive accuracy-vs-k
                "ablation_train_accuracy": tr.get("ablation_accuracy"),
                "control_train_accuracy": tr.get("control_accuracy"),
                "ablation_val_accuracy": va.get("ablation_accuracy"),
                "control_val_accuracy": va.get("control_accuracy"),
                # lift (knockout diff)
                "knockout_diff_train": tr.get("knockout_diff"),
                "knockout_diff_val": va.get("knockout_diff"),
            })
            rows.append(row)
    return rows

def aggregate(paths: List[Path]) -> pd.DataFrame:
    expanded: List[Dict[str, Any]] = []
    for p in paths:
        r = load_metrics_json(p)
        if not r:
            continue
        if int(r.get("version", 1)) >= 2 and "by_k" in r:
            expanded.extend(_expand_v2(r))
        else:
            # v1-style: keep as-is (expects *_correct/*_total columns if present)
            expanded.append(r)
    return pd.DataFrame(expanded) if expanded else pd.DataFrame()

def to_display_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "task" in df.columns:
        df["task_display"] = df["task"].apply(get_task_display_name)
    if {"model_name", "hf_revision"}.issubset(df.columns):
        df["model_display"] = df.apply(_model_display_with_revision, axis=1)
    else:
        df["model_display"] = df.get("model_name", pd.Series(dtype=str)).map(get_model_display_name)
    df["method_display"] = df.get("method", pd.Series(dtype=str)).map(METHOD_DISPLAY).fillna(
        df.get("method", pd.Series(dtype=str)).str.title()
    )
    return df

def _setup_style():
    sns.set_theme(style="ticks", context="talk", palette="colorblind")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,
    })

def _calc_acc(long_df: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Return tidy accuracy vs K with columns:
    task_display, model_display, method_display, top_k, condition, accuracy.
    Supports v1 (counts) and v2 (accuracies).
    """
    # v2 path: *_accuracy columns exist
    acc_cols = {f"baseline_{split}_accuracy", f"ablation_{split}_accuracy", f"control_{split}_accuracy"}
    if acc_cols.issubset(long_df.columns):
        g = long_df.groupby(["task_display", "model_display", "method_display", "top_k"], as_index=False).mean(numeric_only=True)
        tidy = pd.DataFrame({
            "task_display": np.repeat(g["task_display"].values, 3),
            "model_display": np.repeat(g["model_display"].values, 3),
            "method_display": np.repeat(g["method_display"].values, 3),
            "top_k": np.repeat(g["top_k"].values, 3),
            "condition": np.tile(["Baseline", "Ablated (shared)", "Control (random)"], len(g)),
            "accuracy": np.concatenate([
                g[f"baseline_{split}_accuracy"].values,
                g[f"ablation_{split}_accuracy"].values,
                g[f"control_{split}_accuracy"].values,
            ]),
        })
        return tidy.replace([np.inf, -np.inf], np.nan).dropna(subset=["accuracy"])

    # v1 path: *_correct/*_total columns exist
    need = {
        f"baseline_{split}_correct", f"baseline_{split}_total",
        f"ablation_{split}_correct", f"ablation_{split}_total",
        f"control_{split}_correct", f"control_{split}_total",
    }
    if need.issubset(set(long_df.columns)):
        g = (
            long_df.groupby(["task_display", "model_display", "method_display", "top_k"], as_index=False)
            .agg({c: "sum" for c in need})
        )
        def ratio(c_key, t_key):
            return g[c_key] / g[t_key].replace(0, np.nan)
        tidy = pd.DataFrame({
            "task_display": np.repeat(g["task_display"].values, 3),
            "model_display": np.repeat(g["model_display"].values, 3),
            "method_display": np.repeat(g["method_display"].values, 3),
            "top_k": np.repeat(g["top_k"].values, 3),
            "condition": np.tile(["Baseline", "Ablated (shared)", "Control (random)"], len(g)),
            "accuracy": np.concatenate([
                ratio(f"baseline_{split}_correct", f"baseline_{split}_total").values,
                ratio(f"ablation_{split}_correct", f"ablation_{split}_total").values,
                ratio(f"control_{split}_correct", f"control_{split}_total").values,
            ]),
        })
        return tidy.replace([np.inf, -np.inf], np.nan).dropna(subset=["accuracy"])

    return pd.DataFrame()

def _save_or_show(fig, path: Path, show: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def _maybe_filter_by_threshold(df: pd.DataFrame, p: int) -> pd.DataFrame:
    if "reuse_threshold" in df.columns:
        return df[df["reuse_threshold"] == p].copy()
    return df

def plot_accuracy_vs_k_by_task(df: pd.DataFrame, out_dir: Path, split: str = "train",
                               percent: bool = True, show: bool = False, p: int | None = None):
    if p is not None:
        df = _maybe_filter_by_threshold(df, p)
    tidy = _calc_acc(df, split)
    if tidy.empty:
        print(f"[INFO] Missing accuracy columns for split={split}. Skipping.")
        return
    if percent:
        tidy = tidy.assign(accuracy=tidy["accuracy"] * 100.0)

    suffix = f"_p{p}" if p is not None else ""
    for method, sub_m in tidy.groupby("method_display"):
        for task, sub in sub_m.groupby("task_display"):
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.lineplot(
                data=sub.sort_values("top_k"),
                x="top_k",
                y="accuracy",
                hue="model_display",
                style="condition",
                markers=True,
                dashes=False,
                ax=ax,
            )
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            ax.set_xlabel("top-k")
            ax.set_ylabel("Accuracy (%)" if percent else "Accuracy")
            title = f"{task} — {method} — {split}"
            if p is not None:
                title += f" — p={p}"
            ax.set_title(title)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
            outp = out_dir / "by_task" / safe_filename(task) / f"{safe_filename(method)}_{split}_accuracy_vs_k{suffix}.png"
            _save_or_show(fig, outp, show)

def plot_knockout_diff_vs_k_by_task(df: pd.DataFrame, out_dir: Path, split: str = "train",
                                    show: bool = False, p: int | None = None):
    col = f"knockout_diff_{split}"
    if p is not None:
        df = _maybe_filter_by_threshold(df, p)
    if df.empty or col not in df.columns:
        return
    gk = (
        df.groupby(["task_display", "model_display", "method_display", "top_k"], as_index=False)
        .agg(kd=(col, "mean"))
    )
    suffix = f"_p{p}" if p is not None else ""
    for method, sub_m in gk.groupby("method_display"):
        for task, sub in sub_m.groupby("task_display"):
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.lineplot(
                data=sub.sort_values("top_k"),
                x="top_k",
                y="kd",
                hue="model_display",
                markers=True,
                dashes=False,
                ax=ax,
            )
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            ax.set_xlabel("top-k")
            ax.set_ylabel("Ablation Impact Ratio")
            title = f"{task} — {method} — {split}"
            if p is not None:
                title += f" — p={p}"
            ax.set_title(title)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
            outp = out_dir / "by_task" / safe_filename(task) / f"{safe_filename(method)}_knockout_{split}_vs_k{suffix}.png"
            _save_or_show(fig, outp, show)

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    paths = discover_metrics(results_dir)
    df = aggregate(paths)
    if df.empty:
        print("[INFO] No metrics.json found.")
        return

    df = to_display_cols(df)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / f"plots_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / args.save_csv_name
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Aggregated CSV saved to {csv_path}")

    _setup_style()

    percent = not args.percent is False
    p = args.reuse_threshold

    # accuracy vs k (v2 ready)
    plot_accuracy_vs_k_by_task(df, out_dir, split="train", percent=percent, show=args.show, p=p)
    if df.get("baseline_val_accuracy", pd.Series([np.nan])).notna().any():
        plot_accuracy_vs_k_by_task(df, out_dir, split="val", percent=percent, show=args.show, p=p)

    # knockout diff vs k (v2 ready)
    # accept either v1 naming or v2 "knockout_diff_train/val"
    if "knockout_diff_train" in df.columns and "knockout_diff_val" in df.columns:
        df = df.rename(columns={"knockout_diff_train": "knockout_diff_train",
                                "knockout_diff_val": "knockout_diff_val"})
    plot_knockout_diff_vs_k_by_task(df.rename(columns={
        "knockout_diff_train": "knockout_diff_train",
        "knockout_diff_val": "knockout_diff_val"
    }), out_dir, split="train", show=args.show, p=p)
    if "knockout_diff_val" in df.columns:
        plot_knockout_diff_vs_k_by_task(df, out_dir, split="val", show=args.show, p=p)

if __name__ == "__main__":
    main()
