from __future__ import annotations

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

matplotlib.use("Agg")


def plot_baer_accuracy(
    metrics,
    outfile: str,
    *,
    title: str = "Baseline Accuracy and BAER",
):
    by_domain = {}
    for m in metrics:
        dom = m.get("domain", "overall")
        by_domain.setdefault(dom, []).append(m)

    n_domains = len(by_domain)
    fig, axes = plt.subplots(1, n_domains, figsize=(5 * n_domains, 5), squeeze=False)
    for ax, (domain, items) in zip(axes[0], by_domain.items()):
        models = [it["model"] for it in items]
        accuracies = [it["accuracy"] for it in items]
        baers = [it["baer"] for it in items]
        x = np.arange(len(models))
        width = 0.35
        bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy")
        bars2 = ax.bar(x + width / 2, baers, width, label="BAER")
        
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Rate")
        ax.set_title(f"{domain}")
        ax.legend()
    plt.suptitle(title)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)


def plot_module_heatmap(
    module_effects,
    outfile: str,
    *,
    title: str = "Module-wise Effect Heatmap",
) -> None:
    names = list(module_effects.keys())
    values = [module_effects[n] for n in names]
    # Create a 2D array: one row per module, single column
    data = np.array(values).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(4, max(2, len(names) * 0.3)))
    cmap = plt.cm.RdBu
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=min(-0.1, min(values)), vmax=max(0.1, max(values)))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xticks([0])
    ax.set_xticklabels(["Effect"])
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Effect (Δ correctness)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close(fig)


def plot_head_importance(
    head_scores,
    outfile: str,
    *,
    title: str = "Head Importance Scree Plot",
) -> None:
    head_names = [h for h, _ in head_scores]
    scores = [s for _, s in head_scores]
    x = np.arange(1, len(head_names) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, scores, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(head_names, rotation=90)
    ax.set_xlabel("Head")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close(fig)


def plot_ablation_impact(
    ablation_metrics,
    outfile: str,
    *,
    title: str = "Ablation Impact on BAER",
) -> None:
    labels = list(ablation_metrics.keys())
    before = [ablation_metrics[l]["baer_before"] for l in labels]
    after = [ablation_metrics[l]["baer_after"] for l in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, before, label="BAER before", marker="o")
    ax.plot(x, after, label="BAER after", marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("BAER")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close(fig)


__all__ = [
    "plot_baer_accuracy",
    "plot_module_heatmap",
    "plot_head_importance",
    "plot_ablation_impact",
]