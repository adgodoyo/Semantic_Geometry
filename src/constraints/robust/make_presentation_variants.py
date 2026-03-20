from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "slides" / "assets"
ASSETS.mkdir(exist_ok=True)
MPL_DIR = ROOT / "results" / "constraints" / "robust" / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONV_RESULTS = ROOT / "results" / "multilingual"
ROBUST_RESULTS = ROOT / "results" / "constraints" / "robust"

LEVEL_COLORS = {
    0: "#412f88",
    1: "#2c73b6",
    2: "#23a3b1",
    3: "#52c569",
    4: "#b8de29",
}

STYLE_MAP = {"refinement": "-", "control": "--"}
STYLE_LABEL = {"refinement": "Refinement", "control": "Control"}


def crop_whitespace(path: Path) -> None:
    img = Image.open(path).convert("RGB")
    bg = Image.new("RGB", img.size, "white")
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        img.crop(bbox).save(path)


def _spans(density: np.ndarray, threshold: float, min_len: int) -> list[tuple[int, int]]:
    idx = np.where(density > threshold)[0]
    if len(idx) == 0:
        return []

    spans = []
    start = idx[0]
    prev = idx[0]
    for x in idx[1:]:
        if x != prev + 1:
            if prev - start + 1 >= min_len:
                spans.append((start, prev))
            start = x
        prev = x
    if prev - start + 1 >= min_len:
        spans.append((start, prev))
    return spans


def _select_grid_spans(
    density: np.ndarray,
    threshold: float,
    min_len: int,
    expected_count: int,
) -> list[tuple[int, int]]:
    spans = _spans(density, threshold, min_len)
    spans = sorted(spans, key=lambda ab: ab[1] - ab[0], reverse=True)[:expected_count]
    spans = sorted(spans)
    if len(spans) == expected_count:
        return spans
    return []


def extract_grid_cells(
    image_path: Path,
    n_rows: int,
    n_cols: int,
    row_threshold: float,
    col_threshold: float,
    row_min_len: int,
    col_min_len: int,
    pad: int = 24,
) -> list[Image.Image]:
    img = Image.open(image_path).convert("RGB")
    gray = np.array(img.convert("L"))
    mask = gray < 245

    row_spans = _select_grid_spans(mask.mean(axis=1), row_threshold, row_min_len, n_rows)
    col_spans = _select_grid_spans(mask.mean(axis=0), col_threshold, col_min_len, n_cols)

    if not row_spans or not col_spans:
        bbox = np.argwhere(mask)
        top, left = bbox.min(axis=0)
        bottom, right = bbox.max(axis=0)
        row_edges = np.linspace(top, bottom + 1, n_rows + 1, dtype=int)
        col_edges = np.linspace(left, right + 1, n_cols + 1, dtype=int)
        row_spans = list(zip(row_edges[:-1], row_edges[1:]))
        col_spans = list(zip(col_edges[:-1], col_edges[1:]))

    cells = []
    for row_start, row_end in row_spans:
        for col_start, col_end in col_spans:
            left = max(col_start - pad, 0)
            top = max(row_start - pad, 0)
            right = min(col_end + pad, img.width)
            bottom = min(row_end + pad, img.height)
            cells.append(img.crop((left, top, right, bottom)))
    return cells


def compose_cells(
    cells: list[Image.Image],
    layout_rows: int,
    layout_cols: int,
    order: list[int],
    out_path: Path,
    gap_x: int = 16,
    gap_y: int = 16,
) -> None:
    ordered = [cells[idx] for idx in order]
    cell_w = max(cell.width for cell in ordered)
    cell_h = max(cell.height for cell in ordered)
    canvas_w = layout_cols * cell_w + (layout_cols - 1) * gap_x
    canvas_h = layout_rows * cell_h + (layout_rows - 1) * gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    for idx, cell in enumerate(ordered):
        row = idx // layout_cols
        col = idx % layout_cols
        x = col * (cell_w + gap_x)
        y = row * (cell_h + gap_y)
        canvas.paste(cell, (x, y))

    canvas.save(out_path)
    crop_whitespace(out_path)


def make_nuisance_variants() -> None:
    raw_lang_cells = extract_grid_cells(
        CONV_RESULTS / "raw_vs_langresid_grid.png",
        n_rows=2,
        n_cols=2,
        row_threshold=0.02,
        col_threshold=0.02,
        row_min_len=220,
        col_min_len=300,
    )
    compose_cells(
        raw_lang_cells,
        layout_rows=2,
        layout_cols=2,
        order=[0, 1, 2, 3],
        out_path=ASSETS / "nuisance_pair_breakdown_raw_langresid_presentation.png",
        gap_x=18,
        gap_y=18,
    )

    top123_cells = extract_grid_cells(
        CONV_RESULTS / "deflation_top1_top3_grid.png",
        n_rows=3,
        n_cols=2,
        row_threshold=0.02,
        col_threshold=0.02,
        row_min_len=220,
        col_min_len=300,
    )
    compose_cells(
        top123_cells,
        layout_rows=2,
        layout_cols=3,
        order=[0, 2, 4, 1, 3, 5],
        out_path=ASSETS / "nuisance_pair_breakdown_top123_presentation.png",
    )

    top456_cells = extract_grid_cells(
        CONV_RESULTS / "deflation_top4_top6_grid.png",
        n_rows=3,
        n_cols=2,
        row_threshold=0.02,
        col_threshold=0.02,
        row_min_len=220,
        col_min_len=300,
    )
    compose_cells(
        top456_cells,
        layout_rows=2,
        layout_cols=3,
        order=[0, 2, 4, 1, 3, 5],
        out_path=ASSETS / "nuisance_pair_breakdown_top456_presentation.png",
    )


def plot_constraints_grid(df: pd.DataFrame, ks: list[int], out_png: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(17.8, 6.9), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    legend_handles = []
    legend_labels = []

    grouped = (
        df.groupby(["condition", "k", "level", "layer"], as_index=False)["pr"]
        .mean()
        .sort_values(["condition", "k", "level", "layer"])
    )

    for idx, (ax, k) in enumerate(zip(axes_flat, ks, strict=True)):
        sub_k = grouped[grouped["k"] == k]
        for level in range(5):
            color = LEVEL_COLORS[level]
            for condition in ["refinement", "control"]:
                sub = sub_k[(sub_k["level"] == level) & (sub_k["condition"] == condition)]
                label = None
                if idx == 0:
                    label = f"{STYLE_LABEL[condition]} L{level}"
                line = ax.plot(
                    sub["layer"],
                    sub["pr"],
                    color=color,
                    linestyle=STYLE_MAP[condition],
                    linewidth=1.8,
                    label=label,
                )[0]
                if label is not None:
                    legend_handles.append(line)
                    legend_labels.append(label)
        ax.set_title("Top-0 / Raw" if k == 0 else f"Top-{k} deflation", fontsize=12, pad=4)
        ax.grid(True, alpha=0.16)
        ax.tick_params(labelsize=9)

    for ax in axes[1, :]:
        ax.set_xlabel("Layer", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Participation ratio", fontsize=10)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        fontsize=9,
        handlelength=2.0,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.05, right=0.997, top=0.985, bottom=0.12, wspace=0.10, hspace=0.14)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    crop_whitespace(out_png)


def make_constraint_variants() -> None:
    top0_top9 = pd.read_csv(ROBUST_RESULTS / "pr_by_layer_top0_top9.csv")
    plot_constraints_grid(
        top0_top9,
        list(range(10)),
        ASSETS / "constraints_top0_top9_purelevels_presentation.png",
    )

    top10_top19 = pd.read_csv(ROBUST_RESULTS / "pr_by_layer_top10_top19.csv")
    plot_constraints_grid(
        top10_top19,
        list(range(10, 20)),
        ASSETS / "constraints_top10_top19_purelevels_presentation.png",
    )


def main() -> None:
    make_nuisance_variants()
    make_constraint_variants()


if __name__ == "__main__":
    main()
