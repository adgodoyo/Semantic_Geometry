from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ML = ROOT / "results" / "multilingual"
RESULTS_RB = ROOT / "results" / "constraints" / "robust"
PAPER_FIGS = ROOT / "paper" / "figures"
MPL_DIR = PAPER_FIGS / ".mplcache"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps


CATEGORY_SPECS = [
    ("exact_translation_diff_language", "Exact translation\n(diff language)"),
    ("same_factor_diff_language", "Same factor\n(diff language)"),
    ("same_factor_same_language", "Same factor\n(same language)"),
    ("same_domain_diff_language", "Same domain, diff factor\n(diff language)"),
    ("same_domain_same_language", "Same domain, diff factor\n(same language)"),
    ("different_topic_diff_language", "Different topic\n(diff language)"),
    ("different_topic_same_language", "Different topic\n(same language)"),
]

COLOR_MAP = {
    "exact_translation_diff_language": "#d7301f",
    "same_factor_diff_language": "#ef6548",
    "same_domain_diff_language": "#fc8d59",
    "different_topic_diff_language": "#fdbb84",
    "same_factor_same_language": "#225ea8",
    "same_domain_same_language": "#1d91c0",
    "different_topic_same_language": "#41b6c4",
}


def ensure_rgb_png(path: Path) -> Path:
    with Image.open(path) as image:
        image.convert("RGB").save(path)
    return path


def build_figure1_cosine_vertical() -> Path:
    df = pd.read_csv(RESULTS_ML / "pairwise_cosine_by_layer.csv")
    df = df[df["condition"].isin(["raw", "language_residualized"])].copy()

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 8.2), sharex=True)
    legend_handles = []
    legend_labels = []
    conditions = [
        ("raw", "Raw hidden states"),
        ("language_residualized", "After language residualization"),
    ]

    for idx, (ax, (condition, title)) in enumerate(zip(axes, conditions, strict=True)):
        sub_cond = df[df["condition"] == condition]
        for category, label in CATEGORY_SPECS:
            sub = sub_cond[sub_cond["category"] == category].sort_values("layer")
            line = ax.plot(
                sub["layer"],
                sub["mean_cosine"],
                color=COLOR_MAP[category],
                marker="o",
                markersize=2,
                linewidth=1.7,
                label=label.replace("\n", " "),
            )[0]
            if idx == 1:
                legend_handles.append(line)
                legend_labels.append(label.replace("\n", " "))

        ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Mean cosine")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylim(0.0, 1.05)
    axes[1].set_ylim(-0.08, 0.66)
    axes[1].set_xlabel("Layer")

    fig.suptitle(
        "Pairwise cosine geometry before and after language residualization",
        fontsize=13,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=8,
        frameon=True,
        columnspacing=1.0,
        handlelength=1.8,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.9, bottom=0.2, hspace=0.28)

    out_path = PAPER_FIGS / "figure1_cosine_vertical.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return ensure_rgb_png(out_path)


def build_figure2_deflation() -> Path:
    src = RESULTS_ML / "deflation_top1_top3_top6_cosine.png"
    out_path = PAPER_FIGS / "figure2_deflation_top136.png"
    shutil.copy2(src, out_path)
    return ensure_rgb_png(out_path)


def build_figure3_distances_grid() -> Path:
    df = pd.read_csv(RESULTS_ML / "pairwise_distances_by_layer.csv")
    conditions = [
        ("raw", "Raw"),
        ("top5_deflation", "Top-5 deflation"),
        ("language_residualized", "Language residualized"),
    ]
    metrics = [
        ("mean_mahalanobis", "Mahalanobis distance"),
        ("mean_l2", "Euclidean distance"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(9.2, 11.6), sharex=True)
    legend_handles = []
    legend_labels = []

    for row_i, (condition, row_title) in enumerate(conditions):
        sub_cond = df[df["condition"] == condition]
        for col_i, (metric, col_title) in enumerate(metrics):
            ax = axes[row_i, col_i]
            for category, label in CATEGORY_SPECS:
                sub = sub_cond[sub_cond["category"] == category].sort_values("layer")
                line = ax.plot(
                    sub["layer"],
                    sub[metric],
                    color=COLOR_MAP[category],
                    marker="o",
                    markersize=2,
                    linewidth=1.5,
                    label=label.replace("\n", " "),
                )[0]
                if row_i == 2 and col_i == 0:
                    legend_handles.append(line)
                    legend_labels.append(label.replace("\n", " "))

            ax.grid(True, alpha=0.3)
            if row_i == 0:
                ax.set_title(col_title, fontsize=12)
            if col_i == 0:
                ax.set_ylabel(f"{row_title}\n{col_title}", fontsize=11)
            else:
                ax.set_ylabel(row_title, fontsize=11)

    for ax in axes[-1, :]:
        ax.set_xlabel("Layer")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        fontsize=8,
        frameon=True,
        columnspacing=1.0,
        handlelength=1.8,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.95, bottom=0.14, hspace=0.3, wspace=0.22)

    out_path = PAPER_FIGS / "figure3_distances_2x3.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return ensure_rgb_png(out_path)


def build_figure4_constraint_vertical() -> Path:
    top = Image.open(RESULTS_RB / "raw_pr_delta_layers.png").convert("RGB")
    bottom = Image.open(RESULTS_RB / "raw_pr_snapshot_layer25.png").convert("RGB")

    target_w = max(top.width, bottom.width)
    if top.width != target_w:
        top = top.resize((target_w, int(round(top.height * target_w / top.width))), Image.Resampling.LANCZOS)
    if bottom.width != target_w:
        bottom = bottom.resize((target_w, int(round(bottom.height * target_w / bottom.width))), Image.Resampling.LANCZOS)

    top = ImageOps.expand(top, border=20, fill="white")
    bottom = ImageOps.expand(bottom, border=20, fill="white")
    gap = 28
    outer = 18
    canvas = Image.new("RGB", (target_w + 2 * outer + 40, top.height + bottom.height + gap + 2 * outer), "white")
    canvas.paste(top, (outer + 20, outer))
    canvas.paste(bottom, (outer + 20, outer + top.height + gap))

    out_path = PAPER_FIGS / "figure4_constraint_vertical.png"
    canvas.save(out_path)
    return out_path


def build_figure5_motion() -> Path:
    src = RESULTS_RB / "raw_mahalanobis_motion_prev.png"
    out_path = PAPER_FIGS / "figure5_motion_mahalanobis_previous.png"
    shutil.copy2(src, out_path)
    return ensure_rgb_png(out_path)


def build_figure6_id_volume() -> Path:
    src = RESULTS_RB / "raw_id_logvol_layer22.png"
    out_path = PAPER_FIGS / "figure6_mle_logvol_layer22.png"
    shutil.copy2(src, out_path)
    return ensure_rgb_png(out_path)


def build_figure7_innovation() -> Path:
    src = RESULTS_RB / "raw_innovation_ratio.png"
    out_path = PAPER_FIGS / "figure7_innovation_ratio_previous.png"
    shutil.copy2(src, out_path)
    return ensure_rgb_png(out_path)


def main() -> None:
    outputs = [
        build_figure1_cosine_vertical(),
        build_figure2_deflation(),
        build_figure3_distances_grid(),
        build_figure4_constraint_vertical(),
        build_figure5_motion(),
        build_figure6_id_volume(),
        build_figure7_innovation(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
