from __future__ import annotations
import argparse, json, math, re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from circuit_reuse.dataset import get_task_display_name, get_model_display_name

METHOD_DISPLAY = {"eap": "EAP", "gradient": "Gradient"}

def discover_metrics(results_dir: Path) -> List[Path]:
    if not results_dir.exists(): return []
    return sorted(results_dir.rglob("metrics.json"))

def load_metrics_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r") as f: return json.load(f)
    except Exception as e:
        print(f"[WARN] failed to load {path}: {e}")
        return {}

def aggregate(paths: List[Path]) -> pd.DataFrame:
    rows = [d for p in paths if (d := load_metrics_json(p))]
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def safe_filename(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)); s = re.sub(r"_+", "_", s)
    return s.strip("_.")

def parse_step(tag: Optional[str]) -> Optional[int]:
    if not tag or not isinstance(tag, str): return None
    m = re.search(r"step(\d+)", tag)
    return int(m.group(1)) if m else None

def _safe_div(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b): return float("nan")
    denom = b if abs(b) > 1e-12 else (1e-12 if b >= 0 else -1e-12)
    return float(a / denom)

def _acc_from_row(row: pd.Series, prefix: str, split: str) -> Optional[float]:
    acc_key = f"{prefix}_{split}_accuracy"
    c_key, t_key = f"{prefix}_{split}_correct", f"{prefix}_{split}_total"
    if acc_key in row and pd.notna(row[acc_key]): return float(row[acc_key])
    if c_key in row and t_key in row and row[t_key]: return float(row[c_key]) / float(row[t_key])
    return None

def ensure_air(df: pd.DataFrame, split: str) -> pd.DataFrame:
    col = f"knockout_diff_{split}"
    if col in df.columns and df[col].notna().any(): return df
    out = []
    for _, r in df.iterrows():
        b = _acc_from_row(r, "baseline", split)
        s = _acc_from_row(r, "ablation", split)
        c = _acc_from_row(r, "control", split)
        out.append(float("nan") if (b is None or s is None or c is None) else _safe_div(b - s, b - c))
    df[col] = out
    return df

def human_num(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x): return "--"
    ax = abs(x)
    def trim(s: str) -> str:
        return s.rstrip("0").rstrip(".")
    if ax >= 1e12:  return f"{trim(f'{x/1e12:.{digits}f}')}" + "T"
    if ax >= 1e9:   return f"{trim(f'{x/1e9:.{digits}f}')}" + "B"
    if ax >= 1e6:   return f"{trim(f'{x/1e6:.{digits}f}')}" + "M"
    if ax >= 1e3:   return f"{trim(f'{x/1e3:.{digits}f}')}" + "K"
    if ax >= 1:     return trim(f"{x:.{digits}f}")
    if ax >= 1e-3:  return trim(f"{x:.{digits}f}")
    return "0"

def build_latex_table(df: pd.DataFrame, model_disp: str, task_disp: str, method: str, split: str, digits: int) -> str:
    piv = df.pivot_table(index="step", columns="top_k", values="air", aggfunc="mean")
    if piv.empty: return ""
    piv = piv.sort_index().sort_index(axis=1)
    steps = list(piv.index.astype(int))
    ks = [int(k) for k in piv.columns.tolist()]
    header = "Step & " + " & ".join([f"$k={k}$" for k in ks])
    rows = []
    for s in steps:
        vals = [human_num(piv.loc[s, k], digits) for k in piv.columns]
        rows.append(" & ".join([str(s)] + vals))
    colspec = "r" + "r" * len(ks)
    cap = f"Ablation Impact Ratio by pretraining step for {model_disp} on {task_disp} ({METHOD_DISPLAY.get(method, method.title())}, {split})."
    label = f"tab_air_{safe_filename(model_disp)}_{safe_filename(task_disp)}_{method}_{split}"
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\hline")
    lines.append(header + " \\\\")
    lines.append("\\hline")
    for r in rows: lines.append(r + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{cap}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"

def write_and_print_tables(df: pd.DataFrame, out_dir: Path, split: str, method: str, digits: int) -> None:
    df["step"] = df["hf_revision"].apply(parse_step)
    df = df[df["step"].notna()].copy()
    if df.empty:
        print("[INFO] no rows with parseable steps."); return
    df["task_display"] = df["task"].apply(get_task_display_name)
    df["model_display"] = df["model_name"].apply(get_model_display_name)
    if "method" in df.columns: df = df[df["method"] == method].copy()
    if df.empty:
        print("[INFO] no rows after method filter."); return
    df = ensure_air(df, split=split)
    mcol = f"knockout_diff_{split}"
    g = (
        df.groupby(["model_display", "task_display", "top_k", "step"], as_index=False)[mcol]
        .mean()
        .rename(columns={mcol: "air"})
        .sort_values(["model_display", "task_display", "top_k", "step"])
    )
    if g.empty:
        print("[INFO] nothing to tabulate after aggregation."); return
    table_root = out_dir / "air_tables"
    table_root.mkdir(parents=True, exist_ok=True)
    for (model_disp, task_disp), sub in g.groupby(["model_display", "task_display"], as_index=False):
        tex = build_latex_table(sub, model_disp, task_disp, method, split, digits)
        if not tex: continue
        fname = table_root / f"{safe_filename(model_disp)}_{safe_filename(task_disp)}_{method}_{split}.tex"
        fname.write_text(tex)
        print(tex)

def parse_args():
    p = argparse.ArgumentParser(description="Emit LaTeX tables of Ablation Impact Ratio over pretraining steps.")
    p.add_argument("--results-dir", type=str, default="results_pretraining")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--method", type=str, default="eap", choices=["eap", "gradient"])
    p.add_argument("--digits", type=int, default=3)
    return p.parse_args()

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / f"tables_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_metrics(results_dir)
    df = aggregate(paths)
    if df.empty:
        print("[INFO] no metrics.json found."); return
    (out_dir / "aggregated_raw_metrics.csv").write_text(df.to_csv(index=False))
    write_and_print_tables(df.copy(), out_dir, split=args.split, method=args.method, digits=max(0, args.digits))

if __name__ == "__main__":
    main()
