"""
Plot histograms of attribution score distributions for multiple models and tasks.

Example usage:

    python analysis/plot_attribution_histograms.py \
        --cache-dir cache \
        --models meta/llama-2-7b-hf,meta/llama-2-13b-hf \
        --tasks addition,boolean,ioi \
        --hf_revision main \
        --num_examples 100 \
        --method eap \
        --digits 3 \
        --output histogram.png
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM_BINS = 50  # Number of bins for the histograms

def load_scores(attrib_path: Path) -> List[float]:
    """Load all component scores from a cache JSONL file.

    Returns a flat list of float scores.  If the file does not exist, a
    ``FileNotFoundError`` is raised.
    """
    scores: List[float] = []
    with attrib_path.open("r") as f:
        for line in f:
            data = json.loads(line)
            for comp in data.get("components", []):
                # Score may be positive or negative depending on method; cast to float
                scores.append(float(comp.get("score", 0.0)))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot attribution score histograms for multiple models/tasks.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Path to the cache directory containing attribution JSONL files.",
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma‑separated list of model names (as used when running the experiments).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma‑separated list of task names to include.",
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default="none",
        help="Model revision tag used when constructing the cache filenames (default: 'none').",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        required=True,
        help="Number of examples used in the experiments (part of the cache filename).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="eap",
        help="Attribution method name used when constructing the cache filenames.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=None,
        help="Digit count used for the addition task (if applicable).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the resulting histogram image (e.g. 'hist.png').",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    hf_rev = args.hf_revision or "none"
    # Prepare figure
    nrows = len(models)
    ncols = len(tasks)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            ax = axes[i][j]
            digits_str = str(args.digits) if (task == "addition" and args.digits is not None) else "na"
            attrib_name = (
                f"{model.replace('/', '_')}__{hf_rev}__{task}__{args.method}__"
                f"n{args.num_examples}__d{digits_str}.jsonl"
            )
            attrib_path = cache_dir / attrib_name
            if attrib_path.exists():
                try:
                    scores = load_scores(attrib_path)
                    if scores:
                        ax.hist(scores, bins=NUM_BINS, color="tab:blue", alpha=0.7)
                        ax.set_ylabel("Count")
                        ax.set_xlabel("Attribution score")
                    else:
                        ax.text(0.5, 0.5, "No scores", ha="center", va="center")
                    ax.set_title(f"{model}\n{task}")
                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error loading scores\n{e}",
                        ha="center",
                        va="center",
                    )
                    ax.set_title(f"{model}\n{task}")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Missing cache file",
                    ha="center",
                    va="center",
                )
                ax.set_title(f"{model}\n{task}")
            # improve layout
            ax.tick_params(axis="x", labelrotation=45)
    plt.tight_layout()
    out_path = Path(args.output)
    fig.savefig(out_path)
    print(f"Saved histogram grid to {out_path}")


if __name__ == "__main__":
    main()