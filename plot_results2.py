import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from circuit_reuse.dataset import get_model_display_name, get_task_display_name

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
            d = json.load(f)
        d["_metrics_path"] = str(path)
        return d
    except Exception:
        return {}

def aggregate(paths: List[Path]) -> pd.DataFrame:
    rows = [d for p in paths if (d := load_metrics_json(p))]
    if not rows:
        return pd.DataFrame()
    expanded: List[Dict[str, Any]] = []
    for r in rows:
        if int(r.get("version", 1)) != 2:
            continue
        base = {
            "model_name": r.get("model_name"),
            "hf_revision": r.get("hf_revision"),
            "task": r.get("task"),
            "method": r.get("method"),
            "num_examples": r.get("num_examples"),
            "baseline_train_accuracy": r.get("baseline_train_accuracy"),
            "baseline_val_accuracy": r.get("baseline_val_accuracy"),
            "_metrics_path": r.get("_metrics_path"),
        }
        by_k = r.get("by_k", {}) or {}
        for k_str, block in by_k.items():
            try:
                K = int(k_str)
            except Exception:
                continue
            thresholds = (block.get("thresholds", {}) or {})
            for p_str, tblock in thresholds.items():
                try:
                    P = int(p_str)
                except Exception:
                    continue
                row = dict(base)
                row.update({
                    "top_k": K,
                    "reuse_threshold": P,
                    "reuse_percent": tblock.get("reuse_percent"),
                    "shared_circuit_size": tblock.get("shared_circuit_size"),
                    "train_knockout_diff": (tblock.get("train", {}) or {}).get("knockout_diff"),
                    "val_knockout_diff": (tblock.get("val", {}) or {}).get("knockout_diff"),
                    "perm_train_p": ((tblock.get("train", {}) or {}).get("permutation", {}) or {}).get("p_value"),
                    "perm_train_obs": ((tblock.get("train", {}) or {}).get("permutation", {}) or {}).get("obs_diff"),
                    "perm_val_p": ((tblock.get("val", {}) or {}).get("permutation", {}) or {}).get("p_value"),
                    "perm_val_obs": ((tblock.get("val", {}) or {}).get("permutation", {}) or {}).get("obs_diff"),
                })
                expanded.append(row)
    return pd.DataFrame(expanded)

def style_init():
    sns.set_theme(style="ticks", context="talk", palette="colorblind")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.dpi": 120,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

def parse_args():
    p = argparse.ArgumentParser(description="Plot reuse@p and lift vs threshold from metrics.json v2, one figure per task.")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()

def _model_display_with_revision(row: pd.Series) -> str:
    base = get_model_display_name(row.get("model_name"))
    rev = row.get("hf_revision")
    return base if not rev or str(rev).strip() == "" else f"{base} ({rev})"

def _save(fig, path: Path, show: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def _per_task_lineplot(df: pd.DataFrame, x: str, y: str, ylabel: str, title_prefix: str, out_name: str, show: bool):
    for method, sub_m in df.groupby("method_display"):
        for task, sub in sub_m.groupby("task_display"):
            fig, ax = plt.subplots(figsize=(9, 5))
            # style by top_k, color by model
            sns.lineplot(
                data=sub.sort_values([x, "top_k"]),
                x=x,
                y=y,
                hue="model_display",
                style="top_k",
                markers=True,
                dashes=False,
                ax=ax,
            )
            if y == "reuse_percent":
                ax.set_ylim(0, 100)
            if y == "neglog10_p":
                ax.axhline(-np.log10(0.05), ls="--", c="0.5", lw=1)
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            ax.set_xlabel("threshold p")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title_prefix} — {task} — {method}")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, title="")
            _save(fig, out_name(task, method), show)

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (results_dir / f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    style_init()
    paths = discover_metrics(results_dir)
    df = aggregate(paths)
    if df.empty:
        print(f"[WARN] No v2 metrics found under {results_dir}")
        return

    df = df.copy()
    df["task_display"] = df["task"].apply(get_task_display_name)
    df["model_display"] = df.apply(_model_display_with_revision, axis=1)
    df["method_display"] = df.get("method", pd.Series(dtype=str)).map(METHOD_DISPLAY).fillna(
        df.get("method", pd.Series(dtype=str)).str.title()
    )

    csv_path = out_dir / "aggregated_by_task_model_k.csv"
    df.to_csv(csv_path, index=False)
    print(f"[WRITE] {csv_path}")

    # reuse@p per task
    def out_reuse(task, method):
        return out_dir / "by_task" / safe_filename(task) / f"{safe_filename(method)}_reuse_vs_threshold.png"
    _per_task_lineplot(df, x="reuse_threshold", y="reuse_percent", ylabel="reuse@p (%)",
                       title_prefix="Reuse vs threshold", out_name=out_reuse, show=args.show)

    # lift per task (train preferred, else val)
    df = df.assign(lift=df["train_knockout_diff"].where(df["train_knockout_diff"].notna(), df["val_knockout_diff"]))
    def out_lift(task, method):
        return out_dir / "by_task" / safe_filename(task) / f"{safe_filename(method)}_lift_vs_threshold.png"
    _per_task_lineplot(df.dropna(subset=["lift"]), x="reuse_threshold", y="lift", ylabel="knockout diff (lift)",
                       title_prefix="Lift vs threshold", out_name=out_lift, show=args.show)

    # permutation significance per task
    df_sig = df.copy()
    df_sig["perm_p"] = df_sig["perm_train_p"].where(df_sig["perm_train_p"].notna(), df_sig["perm_val_p"])
    df_sig = df_sig.dropna(subset=["perm_p"])
    if not df_sig.empty:
        df_sig["neglog10_p"] = -np.log10(np.clip(df_sig["perm_p"].astype(float), 1e-12, 1.0))
        def out_sig(task, method):
            return out_dir / "by_task" / safe_filename(task) / f"{safe_filename(method)}_perm_neglog10p_vs_threshold.png"
        _per_task_lineplot(df_sig, x="reuse_threshold", y="neglog10_p", ylabel="-log10 p",
                           title_prefix="Permutation significance", out_name=out_sig, show=args.show)

    # permutation summary table stays the same
    table_cols = [
        "model_display", "task_display", "method_display", "top_k", "reuse_threshold",
        "perm_p", "perm_obs_diff",
    ]
    df_perm = df.copy()
    df_perm["perm_p"] = df_perm["perm_train_p"].where(df_perm["perm_train_p"].notna(), df_perm["perm_val_p"])
    df_perm["perm_obs_diff"] = df_perm["perm_train_obs"].where(df_perm["perm_train_obs"].notna(), df_perm["perm_val_obs"])
    latex_df = df_perm[table_cols].sort_values(
        ["model_display", "task_display", "method_display", "top_k", "reuse_threshold"]
    )
    latex_path = out_dir / "permutation_summary.tex"
    try:
        with latex_path.open("w") as f:
            f.write(latex_df.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"[WRITE] {latex_path}")
    except Exception as e:
        print(f"[WARN] Failed to write LaTeX table: {e}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
