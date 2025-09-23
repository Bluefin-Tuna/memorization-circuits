from __future__ import annotations
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from circuit_reuse.dataset import get_task_display_name, get_model_display_name

METHOD_DISPLAY = {"eap": "EAP", "gradient": "Gradient"}


def _extract_step_from_revision(rev: str) -> Optional[str]:
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
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--save-csv-name", type=str, default="aggregated_metrics.csv")
    p.add_argument("--show", action="store_true")
    p.add_argument("--percent", action="store_true", default=True)
    p.add_argument("--ci", type=float, default=-1)

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
    rows: List[Dict[str, Any]] = []

    base = {
        "version": r.get("version", 1),
        "model_name": r.get("model_name"),
        "hf_revision": r.get("hf_revision"),
        "task": r.get("task"),
        "method": r.get("method"),
        "num_examples": r.get("num_examples"),
        "baseline_train_accuracy": r.get("baseline_train_accuracy"),
        "baseline_val_accuracy": r.get("baseline_val_accuracy"),
        "baseline_train_correct": r.get("baseline_train_correct"),
        "baseline_train_total": r.get("baseline_train_total"),
        "baseline_val_correct": r.get("baseline_val_correct"),
        "baseline_val_total": r.get("baseline_val_total"),
    }

    by_k = r.get("by_k") or {}

    for k_str, block in by_k.items():
        try:
            K = int(k_str)
        except Exception:
            continue

        thresholds = block.get("thresholds") or {}

        for p_str, tblock in thresholds.items():
            try:
                P = int(p_str)
            except Exception:
                continue

            tr = tblock.get("train") or {}
            va = tblock.get("val") or {}

            row = dict(base)
            row.update({
                "top_k": K,
                "reuse_threshold": P,
                "reuse_percent": tblock.get("reuse_percent"),
                "shared_circuit_size": tblock.get("shared_circuit_size"),
                "rng_seed": tblock.get("rng_seed"),
                "ablation_train_accuracy": tr.get("ablation_accuracy"),
                "control_train_accuracy": tr.get("control_accuracy"),
                "ablation_val_accuracy": va.get("ablation_accuracy"),
                "control_val_accuracy": va.get("control_accuracy"),
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
            expanded.append(r)

    return pd.DataFrame(expanded) if expanded else pd.DataFrame()


def to_display_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "task" in df.columns:
        df["task_display"] = df["task"].apply(get_task_display_name)
    else:
        df["task_display"] = ""

    if {"model_name", "hf_revision"}.issubset(df.columns):
        df["model_display"] = df.apply(_model_display_with_revision, axis=1)
    else:
        df["model_display"] = df.get("model_name", pd.Series(dtype=str)).map(get_model_display_name)

    df["method_display"] = df.get("method", pd.Series(dtype=str)).map(METHOD_DISPLAY).fillna(
        df.get("method", pd.Series(dtype=str)).str.title()
    )

    return df


def compute_lift(df: pd.DataFrame, split: str) -> pd.Series:
    b = df[f"baseline_{split}_accuracy"].astype(float)
    a = df[f"ablation_{split}_accuracy"].astype(float)
    c = df[f"control_{split}_accuracy"].astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        lift = (a - c) / b

    return pd.to_numeric(lift).replace([np.inf, -np.inf], np.nan)


def compute_ci(p: np.ndarray, n: np.ndarray, z: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        err = z * np.sqrt(np.clip(p, 0, 1) * np.clip(1 - p, 0, 1) / np.where(n <= 0, np.nan, n))

    return np.nan_to_num(err)


def _get_figsize(n_tasks: int, aspect: float = 3 / 4) -> tuple[float, float]:
    width = max(10.0, 0.9 * n_tasks + 3.0)
    height = max(6.0, min(12.0, width * aspect))
    return (width, height)


def _maybe_filter_threshold(df: pd.DataFrame, p: Optional[int]) -> pd.DataFrame:
    if p is None or "reuse_threshold" not in df.columns:
        return df

    return df[df["reuse_threshold"] == p].copy()


def plot_accuracy_bars(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    split: str = "train",
    percent: bool = True,
    ci_level: float = -1.0,
    show: bool = False,
    p: Optional[int] = None
):
    if df.empty:
        return

    df = _maybe_filter_threshold(df, p)

    split_suffix = split
    base_num = f"baseline_{split_suffix}_correct"
    base_den = f"baseline_{split_suffix}_total"
    abl_num = f"ablation_{split_suffix}_correct"
    abl_den = f"ablation_{split_suffix}_total"
    ctrl_num = f"control_{split_suffix}_correct"
    ctrl_den = f"control_{split_suffix}_total"

    group_cols = ["model_display", "task_display", "method_display", "top_k"]

    agg_map: Dict[str, Tuple[str, str]] = {}

    for c in [
        base_num, base_den, abl_num, abl_den, ctrl_num, ctrl_den,
        f"baseline_{split_suffix}_accuracy",
        f"ablation_{split_suffix}_accuracy",
        f"control_{split_suffix}_accuracy"
    ]:
        if c in df.columns:
            agg_map[c] = (c, "mean" if c.endswith("_accuracy") else "sum")

    if not agg_map:
        print(f"[INFO] Missing accuracy columns for split={split}; skipping bars.")
        return

    grouped = df.groupby(group_cols, as_index=False).agg(**agg_map)

    def pick_accuracy(
        sub: pd.DataFrame,
        pref_num: str,
        pref_den: str,
        alt_acc: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if pref_num in sub.columns and pref_den in sub.columns and sub[pref_den].sum() > 0:
            pvals = sub[pref_num] / sub[pref_den].replace(0, np.nan)
            nvals = sub[pref_den]
            return pvals.values, nvals.values
        elif alt_acc in sub.columns:
            return sub[alt_acc].values, None
        else:
            return np.full(len(sub), np.nan), None

    scale = 100 if percent else 1
    z = stats.norm.ppf((1 + ci_level) / 2) if ci_level != -1 else None

    for (model_disp, method_disp, k_val), sub in tqdm(
        grouped.groupby(["model_display", "method_display", "top_k"], as_index=False),
        desc="Plotting accuracy bars"
    ):
        sub = sub.sort_values("task_display")
        tasks = list(sub["task_display"].values)

        base_p, base_n = pick_accuracy(sub, base_num, base_den, f"baseline_{split_suffix}_accuracy")
        abl_p, abl_n = pick_accuracy(sub, abl_num, abl_den, f"ablation_{split_suffix}_accuracy")
        ctrl_p, ctrl_n = pick_accuracy(sub, ctrl_num, ctrl_den, f"control_{split_suffix}_accuracy")

        if np.all(~np.isfinite(base_p)) and np.all(~np.isfinite(abl_p)) and np.all(~np.isfinite(ctrl_p)):
            continue

        base_e = compute_ci(base_p, base_n, z) if (z is not None and base_n is not None) else None
        abl_e = compute_ci(abl_p, abl_n, z) if (z is not None and abl_n is not None) else None
        ctrl_e = compute_ci(ctrl_p, ctrl_n, z) if (z is not None and ctrl_n is not None) else None

        x = np.arange(len(tasks))
        width = 0.25

        fig, ax = plt.subplots(figsize=_get_figsize(len(tasks)))

        b1 = ax.bar(
            x - width, base_p * scale, width, label="Baseline",
            yerr=None if base_e is None else base_e * scale, capsize=3
        )
        b2 = ax.bar(
            x, abl_p * scale, width, label="Ablated (shared)",
            yerr=None if abl_e is None else abl_e * scale, capsize=3
        )
        b3 = ax.bar(
            x + width, ctrl_p * scale, width, label="Control (random)",
            yerr=None if ctrl_e is None else ctrl_e * scale, capsize=3
        )

        for bars in [b1, b2, b3]:
            for r in bars:
                h = r.get_height()
                if np.isfinite(h):
                    ax.annotate(
                        f"{h:.1f}", (r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=30, ha="right")
        ax.set_ylabel("Accuracy (%)" if percent else "Accuracy")

        title = f"{model_disp} k={int(k_val)}"
        if "method_display" in sub.columns and isinstance(method_disp, str) and method_disp:
            title += f" — {method_disp}"
        if p is not None:
            title += f" — p={p}"

        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))

        safe_model = safe_filename(model_disp)
        safe_method = safe_filename(str(method_disp).lower())
        outp = out_dir / f"{safe_model}/{int(k_val)}/{safe_model}_{safe_method}_k{int(k_val)}_{split}"

        if p is not None:
            outp = outp.with_name(outp.name + f"_p{p}")

        outp = outp.with_suffix(".png")
        outp.parent.mkdir(parents=True, exist_ok=True)

        fig.tight_layout()
        fig.savefig(outp, dpi=200, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)


def plot_lift_bars(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    split: str = "train",
    show: bool = False,
    p: Optional[int] = None
):
    if df.empty:
        return

    df = _maybe_filter_threshold(df, p)

    # Compute lift per row
    df = df.copy()
    df["lift"] = compute_lift(df, split)

    group_cols = ["model_display", "task_display", "method_display", "top_k"]
    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(lift=("lift", "mean"))
    )

    for (model_disp, method_disp, k_val), sub in tqdm(
        grouped.groupby(["model_display", "method_display", "top_k"], as_index=False),
        desc="Plotting lift bars"
    ):
        sub = sub.sort_values("task_display")
        tasks = list(sub["task_display"].values)
        vals = sub["lift"].values

        x = np.arange(len(tasks))
        fig, ax = plt.subplots(figsize=_get_figsize(len(tasks)))
        bars = ax.bar(x, vals, width=0.6)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)

        for r, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(
                    f"{v:.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9
                )

        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=30, ha="right")
        ax.set_ylabel("Lift")

        title = f"{model_disp} k={int(k_val)}"
        if isinstance(method_disp, str) and method_disp:
            title += f" — {method_disp}"
        title += f" — {split.title()}"
        if p is not None:
            title += f" — p={p}"

        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        safe_model = safe_filename(model_disp)
        safe_method = safe_filename(str(method_disp).lower())
        outp = out_dir / f"{safe_model}/{int(k_val)}/{safe_model}_{safe_method}_k{int(k_val)}_lift_{split}"

        if p is not None:
            outp = outp.with_name(outp.name + f"_p{p}")

        outp = outp.with_suffix(".png")
        outp.parent.mkdir(parents=True, exist_ok=True)

        fig.tight_layout()
        fig.savefig(outp, dpi=200, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)


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

    sns.set_theme(style="ticks", context="talk", palette="colorblind")
    plt.rcParams.update({"font.family": "serif"})

    reuse_thresholds = df["reuse_threshold"].dropna().unique()

    # Generate plots for both splits into separate subfolders
    for split in ["train", "val"]:
        out_dir_split = out_dir / split
        out_dir_split.mkdir(parents=True, exist_ok=True)

        for reuse_threshold in reuse_thresholds:
            filtered_df = _maybe_filter_threshold(df, reuse_threshold)

            plot_accuracy_bars(
                filtered_df, out_dir_split,
                split=split, percent=args.percent,
                ci_level=args.ci, show=args.show, p=reuse_threshold
            )

            plot_lift_bars(
                filtered_df, out_dir_split,
                split=split, show=args.show, p=reuse_threshold
            )


if __name__ == "__main__":
    main()
