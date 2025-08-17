from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot circuit reuse experiment results."
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help=(
            "Directory containing per-run subdirectories with metrics.json."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write plots (default: <results-dir>/plots_<timestamp>)."
        ),
    )
    p.add_argument(
        "--save-csv-name",
        type=str,
        default="aggregated_metrics.csv",
        help="Filename for aggregated CSV inside output directory.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (if a display is available).",
    )
    p.add_argument(
        "--sort-by",
        type=str,
        default="drop",
        choices=["drop", "baseline", "ablation", "task"],
        help="Ordering of tasks within each plot.",
    )
    p.add_argument(
        "--percent",
        action="store_true",
        default=True,
        help=(
            "Display accuracies as percentage (0-100%). (Default: True)"
        ),
    )
    p.add_argument(
        "--overlay-scores",
        action="store_true",
        help=(
            "Overlay baseline and post-removal accuracies as connected "
            "markers above bars."
        ),
    )
    p.add_argument(
        "--no-control",
        action="store_true",
        help="If set, do not plot Control results.",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Show raw accuracies (0-1). Overrides --percent default.",
    )
    p.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help=(
            "Confidence level for error bars (e.g., 0.95 for 95% CI). "
            "Use -1 to disable error bars."
        ),
    )
    return p.parse_args()


def discover_metrics(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("metrics.json"))


def load_metrics_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}


def aggregate(paths: List[Path]) -> pd.DataFrame:
    rows = [d for p in paths if (d := load_metrics_json(p))]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def plot_all(
    df: pd.DataFrame,
    out_dir: Path,
    show: bool,
    sort_by: str,
    *,
    percent: bool,
    overlay_scores: bool,
    include_control: bool,
    ci_level: float,
):
    if df.empty:
        print("[INFO] No data to plot.")
        return

    error_bars_enabled = ci_level != -1
    z_score = (
        stats.norm.ppf((1 + ci_level) / 2) if error_bars_enabled else None
    )

    # Filter for methods we want to plot
    if "method" in df.columns:
        df = df[df["method"].isin(METHOD_DISPLAY.keys())].copy()

    grouped = (
        df.groupby(
            ["model_display", "task_display", "method_display"],
            as_index=False,
        )
        .agg(
            baseline_train_correct=("baseline_train_correct", "sum"),
            baseline_train_total=("baseline_train_total", "sum"),
            ablation_train_correct=("ablation_train_correct", "sum"),
            ablation_train_total=("ablation_train_total", "sum"),
            control_train_correct=("control_train_correct", "sum"),
            control_train_total=("control_train_total", "sum"),
            baseline_val_correct=("baseline_val_correct", "sum"),
            baseline_val_total=("baseline_val_total", "sum"),
            ablation_val_correct=("ablation_val_correct", "sum"),
            ablation_val_total=("ablation_val_total", "sum"),
            control_val_correct=("control_val_correct", "sum"),
            control_val_total=("control_val_total", "sum"),
            shared_circuit_size_mean=("shared_circuit_size", "mean"),
            top_k_mean=("top_k", "mean"),
        )
    )

    for split in ["train", "val"]:
        for cat in ["baseline", "ablation", "control"]:
            correct_col = f"{cat}_{split}_correct"
            total_col = f"{cat}_{split}_total"
            acc_col = f"{cat}_{split}_accuracy_mean"
            err_col = f"{cat}_{split}_accuracy_error"

            if correct_col in grouped.columns and total_col in grouped.columns:
                grouped[acc_col] = (
                    grouped[correct_col]
                    / grouped[total_col].replace(0, np.nan)
                )

                if error_bars_enabled:

                    def _sym_err(row: pd.Series) -> float:
                        p = row[acc_col]
                        n = row[total_col]
                        if pd.isna(p) or n <= 0:
                            return 0.0
                        return float(
                            z_score * np.sqrt(p * (1 - p) / n)
                        )

                    grouped[err_col] = grouped.apply(_sym_err, axis=1)
                else:
                    grouped[err_col] = 0.0

    saved_files: List[str] = []

    sns.set_theme(style="ticks", context="talk", palette="colorblind")
    base_palette = sns.color_palette("colorblind")

    def _create_plot(
        sub: pd.DataFrame,
        baseline_col: str,
        ablation_col: str,
        title_suffix: str,
        filename_suffix: str,
        control_col: Optional[str] = None,
        control_label: str = "Control",
    ):
        if sub.empty:
            return

        if sort_by == "drop":
            sub = sub.assign(
                _drop=sub[baseline_col] - sub[ablation_col]
            ).sort_values("_drop", ascending=False)
        elif sort_by == "baseline":
            sub = sub.sort_values(baseline_col, ascending=False)
        elif sort_by == "ablation":
            sub = sub.sort_values(ablation_col, ascending=False)
        else:
            sub = sub.sort_values("task_display")

        tasks = list(sub.task_display)
        scale = 100 if percent else 1

        base_vals = sub[baseline_col].values * scale
        abl_vals = sub[ablation_col].values * scale

        has_control = (
            control_col is not None and control_col in sub.columns
        )
        ctrl_vals = (
            sub[control_col].values * scale if has_control else None
        )

        if error_bars_enabled:
            base_err = (
                sub[baseline_col.replace("_mean", "_error")].values * scale
            )
            abl_err = (
                sub[ablation_col.replace("_mean", "_error")].values * scale
            )
            ctrl_err = (
                sub[control_col.replace("_mean", "_error")].values * scale
                if has_control
                else None
            )
        else:
            base_err = abl_err = ctrl_err = None

        x = np.arange(len(tasks))

        labels = ["Baseline", "Ablated"]
        val_arrays = [base_vals, abl_vals]
        err_arrays = [base_err, abl_err]
        colors = [base_palette[0], base_palette[1]]

        if has_control:
            labels.append(control_label)
            val_arrays.append(ctrl_vals)
            err_arrays.append(ctrl_err)
            colors.append(base_palette[2])

        n_bars = len(labels)
        bar_width = 0.8 / n_bars
        offsets = [
            -((n_bars - 1) / 2.0) * bar_width + i * bar_width
            for i in range(n_bars)
        ]

        fig_w = max(8.0, 1.2 * len(tasks) + 2.5)
        fig, ax = plt.subplots(figsize=(fig_w, 6.0))

        error_kwargs = dict(
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            ecolor="black",
        )

        for off, lab, vals, err, col in zip(
            offsets, labels, val_arrays, err_arrays, colors
        ):
            ax.bar(
                x + off,
                vals,
                bar_width * 0.95,
                label=lab,
                color=col,
                yerr=err if error_bars_enabled else None,
                error_kw=error_kwargs if error_bars_enabled else None,
            )

        ax.set_ylabel("Accuracy (%)" if percent else "Accuracy")

        vals_for_max = [v for v in val_arrays if v is not None]
        if percent:
            ymax = 105
        else:
            ymax = (
                1.05 * float(np.nanmax(vals_for_max))
                if vals_for_max
                else 1.0
            )
        ax.set_ylim(0, ymax)

        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=0, ha="center")
        ax.set_title(title_suffix)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=n_bars,
        )

        fig.tight_layout(rect=[0, 0.05, 1, 1])

        out_path = out_dir / filename_suffix
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        saved_files.append(out_path.name)
        if show:
            plt.show()
        plt.close()

    for (model_disp, method_disp), sub_df in grouped.groupby(
        ["model_display", "method_display"]
    ):
        safe_model = model_disp.replace(" ", "_")
        safe_method = method_disp.lower()

        if (
            "baseline_train_accuracy_mean" in sub_df.columns
            and "ablation_train_accuracy_mean" in sub_df.columns
        ):
            _create_plot(
                sub_df,
                "baseline_train_accuracy_mean",
                "ablation_train_accuracy_mean",
                f"{model_disp} - {method_disp} (Train)",
                f"{safe_model}_{safe_method}_train.png",
                control_col=(
                    "control_train_accuracy_mean" if include_control else None
                ),
            )

        has_val = (
            "baseline_val_accuracy_mean" in sub_df.columns
            and "ablation_val_accuracy_mean" in sub_df.columns
            and sub_df["baseline_val_total"].sum() > 0
        )
        if has_val:
            _create_plot(
                sub_df,
                "baseline_val_accuracy_mean",
                "ablation_val_accuracy_mean",
                f"{model_disp} - {method_disp} (Validation)",
                f"{safe_model}_{safe_method}_val.png",
                control_col=(
                    "control_val_accuracy_mean" if include_control else None
                ),
            )

    print(f"[INFO] Plots written to: {out_dir}")
    files_str = ", ".join(saved_files)
    print(f"[INFO] Files: {files_str}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    df = aggregate(discover_metrics(results_dir))

    if df.empty:
        print("[INFO] Empty DataFrame; skipping CSV and plots.")
        return

    percent = not args.raw

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else results_dir / f"plots_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df["task_display"] = df["task"].apply(get_task_display_name)
    df["model_display"] = df["model_name"].apply(get_model_display_name)
    df["method_display"] = df["method"].map(METHOD_DISPLAY).fillna(
        df["method"].str.title()
    )

    csv_path = out_dir / args.save_csv_name
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Aggregated CSV saved to {csv_path}")

    plot_all(
        df,
        out_dir,
        show=args.show,
        sort_by=args.sort_by,
        percent=percent,
        overlay_scores=args.overlay_scores,
        include_control=not args.no_control,
        ci_level=args.ci,
    )


if __name__ == "__main__":
    main()