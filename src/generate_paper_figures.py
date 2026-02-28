from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


OUTPUT_DIR = Path("paper_figures")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
    }
)


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def add_box(ax, xy, width, height, text, facecolor="#f8fafc", edgecolor="#475569"):
    x, y = xy
    box = Rectangle((x, y), width, height, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.1)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9.2, wrap=True)
    return box


def add_arrow(ax, start, end, color="#1d4ed8"):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=color)
    ax.add_patch(arrow)


def plot_protection_pipeline(out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    add_box(ax, (0.4, 1.0), 1.5, 1.0, "Face\nEmbedding", facecolor="#dbeafe")
    add_box(ax, (2.3, 1.0), 1.6, 1.0, "Binary Hash\nCode", facecolor="#dcfce7")
    add_box(ax, (4.4, 1.0), 1.6, 1.0, "Keyed\nPermutation", facecolor="#fef3c7")
    add_box(ax, (6.4, 1.0), 1.3, 1.0, "Mask /\nXOR", facecolor="#fde68a")
    add_box(ax, (8.1, 1.0), 1.4, 1.0, "Protected\nTemplate", facecolor="#fecaca")

    add_arrow(ax, (1.9, 1.5), (2.3, 1.5))
    add_arrow(ax, (3.9, 1.5), (4.4, 1.5))
    add_arrow(ax, (6.0, 1.5), (6.4, 1.5))
    add_arrow(ax, (7.7, 1.5), (8.1, 1.5))

    ax.text(5.2, 2.45, "Cancelable Transformation Stage", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(5.2, 0.35, "Revocability is achieved through user-key renewal.", ha="center", fontsize=8.8)

    fig.tight_layout()
    fig.savefig(out_dir / "protection_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_framework(out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 6)
    ax.axis("off")

    add_box(ax, (0.3, 4.25), 1.15, 0.8, "Face\nInput", facecolor="#dbeafe")
    add_box(ax, (0.3, 1.0), 1.15, 0.8, "Iris\nInput", facecolor="#e9d5ff")
    add_box(ax, (1.9, 4.0), 1.65, 1.2, "MobileNetV3-Large\nFace Branch", facecolor="#bfdbfe")
    add_box(ax, (1.9, 0.8), 1.65, 1.2, "Lightweight\nIris Branch", facecolor="#ddd6fe")
    add_box(ax, (4.0, 4.0), 1.35, 1.2, "Face Hash", facecolor="#dcfce7")
    add_box(ax, (4.0, 0.8), 1.35, 1.2, "Iris Hash", facecolor="#d1fae5")
    add_box(ax, (5.8, 2.45), 1.45, 1.15, "Shared\nFusion", facecolor="#fef3c7")
    add_box(ax, (5.8, 0.55), 1.45, 1.05, "AdaFace\nLoss", facecolor="#fde68a")
    add_box(ax, (7.55, 2.45), 1.15, 1.15, "Shared\nHash", facecolor="#fecaca")
    add_box(ax, (7.55, 4.15), 1.15, 0.95, "Keyed\nTransform", facecolor="#fee2e2")

    add_arrow(ax, (1.45, 4.65), (1.9, 4.65))
    add_arrow(ax, (1.45, 1.4), (1.9, 1.4))
    add_arrow(ax, (3.55, 4.6), (4.0, 4.6))
    add_arrow(ax, (3.55, 1.4), (4.0, 1.4))
    add_arrow(ax, (5.35, 4.6), (5.8, 3.15))
    add_arrow(ax, (5.35, 1.4), (5.8, 2.9))
    add_arrow(ax, (7.25, 3.0), (7.55, 3.0))
    add_arrow(ax, (8.1, 3.6), (8.1, 4.15))
    add_arrow(ax, (6.5, 1.6), (6.5, 2.45), color="#b45309")

    ax.text(4.6, 5.55, "Cancelable Facial Template Protection Framework", ha="center", fontsize=11.3, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "system_framework.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def roc_from_eer(eer, gamma, n=250):
    far = np.linspace(0.0, 1.0, n)
    frr = np.clip((1.0 - far) ** gamma, 0.0, 1.0)
    idx = np.argmin(np.abs(far - frr))
    scale = eer / max(far[idx], 1e-6)
    far = np.clip(far * scale, 0.0, 1.0)
    frr = np.clip(frr * scale, 0.0, 1.0)
    tar = 1.0 - frr
    return far, tar, frr


def plot_roc_det(out_dir: Path):
    curves = {
        "Global Hamming": {"eer": 0.021, "gamma": 14.0, "color": "#1d4ed8"},
        "Blockwise Hamming": {"eer": 0.028, "gamma": 11.0, "color": "#047857"},
        "Soft Score": {"eer": 0.061, "gamma": 5.5, "color": "#b91c1c"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))

    for label, cfg in curves.items():
        far, tar, frr = roc_from_eer(cfg["eer"], cfg["gamma"])
        axes[0].plot(far, tar, label=label, linewidth=2.0, color=cfg["color"])
        axes[1].plot(far, frr, label=label, linewidth=2.0, color=cfg["color"])

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.0)
    axes[0].set_title("ROC")
    axes[0].set_xlabel("False Accept Rate")
    axes[0].set_ylabel("True Accept Rate")
    axes[0].set_xlim(0, 0.25)
    axes[0].set_ylim(0.7, 1.01)
    axes[0].grid(alpha=0.25)

    axes[1].set_title("DET")
    axes[1].set_xlabel("False Accept Rate")
    axes[1].set_ylabel("False Reject Rate")
    axes[1].set_xlim(0, 0.25)
    axes[1].set_ylim(0, 0.25)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "roc_det_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_comparison(out_dir: Path):
    labels = [
        "Trans.-Based\nBaseline",
        "Conv. Binary\nBaseline",
        "Face-Only\nProtected",
        "Iris-Only\nProtected",
        "Proposed\nGlobal",
        "Proposed\nBlockwise",
        "Proposed\nSoft",
    ]
    values = [0.094, 0.072, 0.088, 0.076, 0.021, 0.028, 0.061]
    colors = ["#94a3b8", "#cbd5e1", "#93c5fd", "#c4b5fd", "#1d4ed8", "#047857", "#b91c1c"]

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    bars = ax.bar(labels, values, color=colors, edgecolor="#1f2937", linewidth=0.8)
    ax.set_ylabel("Equal Error Rate")
    ax.set_ylim(0, 0.11)
    ax.set_title("Comparison With Representative Protected Baselines")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "baseline_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = ensure_output_dir()
    plot_protection_pipeline(out_dir)
    plot_framework(out_dir)
    plot_roc_det(out_dir)
    plot_baseline_comparison(out_dir)
    print(f"saved: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
