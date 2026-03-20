#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


ROOT = Path(__file__).resolve().parents[1]


TOP_LEVEL_KEEP = {
    Path("README.md"),
    Path("MANIFEST.md"),
    Path(".gitignore"),
    Path("requirements.txt"),
}

SCRIPT_KEEP = {
    Path("run_all.sh"),
    Path("extract_multilingual.sh"),
    Path("extract_constraints.sh"),
    Path("analyze_multilingual.sh"),
    Path("analyze_constraints.sh"),
    Path("build_slides.sh"),
    Path("build_paper.sh"),
    Path("curate_release.sh"),
    Path("build_slide_assets.py"),
    Path("curate_release.py"),
    Path("project_manifest.py"),
    Path("validate_project.py"),
}

MULTILINGUAL_SRC_KEEP = {
    Path("analyze.py"),
    Path("analysis_g_all_families.py"),
    Path("analysis_g_pair_metric_breakdown.py"),
    Path("analysis_g_pair_metric_breakdown_mahalanobis.py"),
    Path("dataset.py"),
    Path("dataset_all_families.py"),
    Path("extract.py"),
    Path("extract_all_families.py"),
    Path("make_additional_requested_figures.py"),
    Path("ml_sentences.py"),
    Path("ml_sentences_f2f5.py"),
}

ROBUST_SRC_KEEP = {
    Path("make_level_motion_graphs.py"),
    Path("make_presentation_variants.py"),
    Path("make_raw_euclidean_delta.py"),
    Path("make_raw_metric_snapshots.py"),
    Path("make_requested_graphs.py"),
    Path("run_robust.py"),
}

PAPER_KEEP = {
    Path("paper.tex"),
    Path("paper.pdf"),
    Path("icml2026.sty"),
}

SLIDE_KEEP = {
    Path("slides.tex"),
    Path("slides.pdf"),
}

MULTILINGUAL_RESULT_KEEP = {
    Path("cosine_gap_summary.png"),
    Path("crosslingual_index_by_family.png"),
    Path("pairwise_cosine_by_layer.csv"),
    Path("pairwise_distances_by_layer.csv"),
    Path("raw_vs_langresid_grid.png"),
    Path("deflation_top1_top3_grid.png"),
    Path("deflation_top1_top3_top6_cosine.png"),
    Path("deflation_top4_top6_grid.png"),
}

CONSTRAINTS_BASE_RESULT_KEEP = {
    Path("full_pythia/acts_full_pythia.npy"),
    Path("control_pythia/acts_control_pythia.npy"),
}

ROBUST_RESULT_KEEP = {
    Path("language_topk_alignment.png"),
    Path("raw_pr_delta_layers.png"),
    Path("raw_pr_snapshot_layer25.png"),
    Path("raw_euclidean_delta_layers.png"),
    Path("raw_langresid_snapshot_layer25.png"),
    Path("raw_motion_from_initial.png"),
    Path("raw_innovation_ratio.png"),
    Path("raw_mahalanobis_motion_prev.png"),
    Path("raw_motion_from_previous.png"),
    Path("raw_id_logvol_delta_layers.png"),
    Path("raw_id_logvol_layer22.png"),
    Path("raw_id_logvol_layer25.png"),
    Path("pr_by_layer_top0_top9.csv"),
    Path("pr_by_layer_top10_top19.csv"),
}

LATEX_BUILD_SUFFIXES = {
    ".aux",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".nav",
    ".out",
    ".snm",
    ".toc",
}

JUNK_DIR_NAMES = {"__pycache__", ".mplcache"}
JUNK_FILE_NAMES = {".DS_Store"}


def _tex_includegraphics_names(tex_path: Path) -> set[str]:
    text = tex_path.read_text(encoding="utf-8")
    return set(re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", text))


def slide_asset_keep() -> set[Path]:
    tex = ROOT / "slides" / "slides.tex"
    return {Path("assets") / name for name in _tex_includegraphics_names(tex)}


def paper_figure_keep() -> set[Path]:
    tex = ROOT / "paper" / "paper.tex"
    return {Path("figures") / name for name in _tex_includegraphics_names(tex)}
