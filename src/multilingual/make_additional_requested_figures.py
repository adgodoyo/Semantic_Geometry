from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ML = ROOT / "results" / "multilingual"
RESULTS_ROBUST = ROOT / "results" / "constraints" / "robust"
MPL_DIR = RESULTS_ML / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

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


def crop_image(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    bg = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.getbbox()
    return rgb.crop(bbox) if bbox else rgb


def crop_whitespace(path: Path) -> None:
    crop_image(Image.open(path)).save(path)


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


def _select_grid_spans(density: np.ndarray, threshold: float, min_len: int, expected_count: int) -> list[tuple[int, int]]:
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


def compose_grid(cells: list[Image.Image], layout_rows: int, layout_cols: int, out_path: Path, gap_x: int = 16, gap_y: int = 16) -> None:
    cell_w = max(cell.width for cell in cells)
    cell_h = max(cell.height for cell in cells)
    canvas_w = layout_cols * cell_w + (layout_cols - 1) * gap_x
    canvas_h = layout_rows * cell_h + (layout_rows - 1) * gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    for idx, cell in enumerate(cells):
        row = idx // layout_cols
        col = idx % layout_cols
        x = col * (cell_w + gap_x)
        y = row * (cell_h + gap_y)
        canvas.paste(cell, (x, y))

    canvas.save(out_path)
    crop_whitespace(out_path)


def make_cosine_top136() -> Path:
    out_path = RESULTS_ML / "deflation_top1_top3_top6_cosine.png"
    top123_cells = extract_grid_cells(
        RESULTS_ML / "deflation_top1_top3_grid.png",
        n_rows=3,
        n_cols=2,
        row_threshold=0.02,
        col_threshold=0.02,
        row_min_len=220,
        col_min_len=420,
        pad=54,
    )
    top456_cells = extract_grid_cells(
        RESULTS_ML / "deflation_top4_top6_grid.png",
        n_rows=3,
        n_cols=2,
        row_threshold=0.02,
        col_threshold=0.02,
        row_min_len=220,
        col_min_len=420,
        pad=90,
    )

    panels = [
        top123_cells[0].convert("RGB"),
        top123_cells[4].convert("RGB"),
        top456_cells[4].convert("RGB"),
    ]

    # Trim the bleed from the neighboring Euclidean panels on the right edge.
    panels[0] = panels[0].crop((0, 10, panels[0].width - 150, panels[0].height))
    panels[1] = panels[1].crop((0, 0, panels[1].width - 150, panels[1].height))
    panels[2] = panels[2].crop((0, 30, panels[2].width - 118, panels[2].height))

    target_h = max(panel.height for panel in panels)
    resized = []
    for panel in panels:
        if panel.height != target_h:
            new_w = int(round(panel.width * target_h / panel.height))
            panel = panel.resize((new_w, target_h), Image.Resampling.LANCZOS)
        resized.append(panel)

    left_space = 72
    right_space = 18
    top_space = 62
    bottom_space = 48
    gap = 24
    canvas_w = left_space + right_space + sum(panel.width for panel in resized) + gap * (len(resized) - 1)
    canvas_h = top_space + bottom_space + target_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    x = left_space
    for panel in resized:
        canvas.paste(panel, (x, top_space))
        x += panel.width + gap

    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 22)
    label_font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 18)

    title = "G2 pair-category breakdown: cosine under blind deflation"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(((canvas_w - title_w) / 2, 14), title, fill="black", font=title_font)

    y_label = "Mean cosine"
    y_bbox = draw.textbbox((0, 0), y_label, font=label_font)
    y_w = y_bbox[2] - y_bbox[0]
    y_h = y_bbox[3] - y_bbox[1]
    y_img = Image.new("RGBA", (y_w + 8, y_h + 8), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_img)
    y_draw.text((4, 4), y_label, fill="black", font=label_font)
    y_rot = y_img.rotate(90, expand=True)
    y_x = int((left_space - y_rot.width) / 2)
    y_y = int(top_space + (target_h - y_rot.height) / 2)
    canvas.paste(y_rot, (y_x, y_y), y_rot)

    # The original top-1 source panel hides numeric x ticks and keeps an internal
    # "Layer" label because it came from the top row of a shared-x figure. Remove
    # that local label and add the missing tick numbers so all three panels match.
    first_panel = resized[0]
    scale = first_panel.height / panels[0].height
    axis_left = int(round(54 * scale))
    axis_right = int(round(960 * scale))
    axis_y = int(round(582 * scale))
    first_x = left_space
    first_y = top_space
    draw.rectangle(
        [
            first_x + int(first_panel.width * 0.28),
            first_y + axis_y + 2,
            first_x + int(first_panel.width * 0.72),
            first_y + first_panel.height,
        ],
        fill="white",
    )
    tick_font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 16)
    tick_y = first_y + axis_y + 10
    for tick in [0, 5, 10, 15, 20, 25, 30]:
        x_tick = first_x + axis_left + int(round((axis_right - axis_left) * (tick / 32.0)))
        text = str(tick)
        bbox = draw.textbbox((0, 0), text, font=tick_font)
        text_w = bbox[2] - bbox[0]
        draw.text((x_tick - text_w / 2, tick_y), text, fill="black", font=tick_font)

    x_label = "Layer"
    x_bbox = draw.textbbox((0, 0), x_label, font=label_font)
    x_w = x_bbox[2] - x_bbox[0]
    draw.text(((canvas_w - x_w) / 2, canvas_h - bottom_space + 10), x_label, fill="black", font=label_font)

    canvas.save(out_path)
    return out_path


def make_raw_top5_langresid_mahalanobis_euclidean() -> Path:
    df = pd.read_csv(RESULTS_ML / "pairwise_distances_by_layer.csv")
    conditions = [
        ("raw", "Raw"),
        ("top5_deflation", "Top-5 deflation"),
        ("language_residualized", "Language residualized"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)

    for col_i, (condition, label_prefix) in enumerate(conditions):
        sub_cond = df[df["condition"] == condition]
        for category, label in CATEGORY_SPECS:
            sub = sub_cond[sub_cond["category"] == category].sort_values("layer")
            axes[0, col_i].plot(
                sub["layer"],
                sub["mean_mahalanobis"],
                label=label.replace("\n", " "),
                color=COLOR_MAP[category],
                marker="o",
                markersize=2,
                linewidth=1.6,
            )
            axes[1, col_i].plot(
                sub["layer"],
                sub["mean_l2"],
                label=label.replace("\n", " "),
                color=COLOR_MAP[category],
                marker="o",
                markersize=2,
                linewidth=1.6,
            )

        axes[0, col_i].set_title(f"{label_prefix}: mean Mahalanobis distance", fontsize=12)
        axes[0, col_i].set_xlabel("Layer")
        axes[0, col_i].set_ylabel("Mean Mahalanobis")
        axes[0, col_i].grid(True, alpha=0.3)

        axes[1, col_i].set_title(f"{label_prefix}: mean Euclidean distance", fontsize=12)
        axes[1, col_i].set_xlabel("Layer")
        axes[1, col_i].set_ylabel("Mean L2")
        axes[1, col_i].grid(True, alpha=0.3)

    axes[1, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Pair-category distances across layers", fontsize=13)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.9, bottom=0.08, wspace=0.14, hspace=0.2)

    out_path = RESULTS_ML / "raw_top5_langresid_mahalanobis_euclidean.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def make_constraints_combined() -> Path:
    left = ImageOps.expand(Image.open(RESULTS_ROBUST / "raw_pr_delta_layers.png").convert("RGB"), border=30, fill="white")
    right = ImageOps.expand(Image.open(RESULTS_ROBUST / "raw_pr_snapshot_layer25.png").convert("RGB"), border=30, fill="white")

    target_h = max(left.height, right.height)
    if left.height != target_h:
        left = left.resize((int(left.width * target_h / left.height), target_h))
    if right.height != target_h:
        right = right.resize((int(right.width * target_h / right.height), target_h))

    gap = 40
    outer_pad = 24
    canvas = Image.new("RGB", (left.width + right.width + gap + 2 * outer_pad, target_h + 2 * outer_pad), "white")
    canvas.paste(left, (outer_pad, outer_pad))
    canvas.paste(right, (left.width + gap + outer_pad, outer_pad))

    out_path = RESULTS_ROBUST / "raw_pr_delta_and_snapshot_combined.png"
    canvas.save(out_path)
    return out_path


def main() -> None:
    outputs = [
        make_cosine_top136(),
        make_raw_top5_langresid_mahalanobis_euclidean(),
        make_constraints_combined(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
