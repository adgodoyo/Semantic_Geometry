#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "slides" / "assets"
TEX_FILE = ROOT / "slides" / "slides.tex"
MPL_DIR = ROOT / "results" / "constraints" / "robust" / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

MULTILINGUAL_RESULTS = ROOT / "results" / "multilingual"
ROBUST_RESULTS = ROOT / "results" / "constraints" / "robust"
HELPER_FILE = ROOT / "src" / "constraints" / "robust" / "make_presentation_variants.py"

COPIED_ASSETS = {
    "nuisance_crosslingual_index_by_family.png": MULTILINGUAL_RESULTS / "crosslingual_index_by_family.png",
    "constraints_snapshot_layer25_purelevels.png": ROBUST_RESULTS / "raw_pr_snapshot_layer25.png",
    "constraints_delta_across_layers.png": ROBUST_RESULTS / "raw_pr_delta_layers.png",
    "constraints_raw_langresid_purelevels.png": ROBUST_RESULTS / "raw_langresid_snapshot_layer25.png",
    "constraints_raw_mle_logvol_snapshot_layer25.png": ROBUST_RESULTS / "raw_id_logvol_layer25.png",
    "constraints_raw_mle_logvol_snapshot_layer22.png": ROBUST_RESULTS / "raw_id_logvol_layer22.png",
    "constraints_raw_mle_logvol_delta_across_layers.png": ROBUST_RESULTS / "raw_id_logvol_delta_layers.png",
    "constraints_raw_euclidean_delta_across_layers.png": ROBUST_RESULTS / "raw_euclidean_delta_layers.png",
    "constraints_raw_level_motion_initial.png": ROBUST_RESULTS / "raw_motion_from_initial.png",
    "constraints_raw_level_motion_previous.png": ROBUST_RESULTS / "raw_motion_from_previous.png",
    "constraints_raw_innovation_previous.png": ROBUST_RESULTS / "raw_innovation_ratio.png",
    "appendix_convergence_2.png": MULTILINGUAL_RESULTS / "cosine_gap_summary.png",
    "appendix_language_topk_alignment.png": ROBUST_RESULTS / "language_topk_alignment.png",
}


def load_helper():
    spec = importlib.util.spec_from_file_location("presentation_variants", HELPER_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {HELPER_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def required_assets_from_tex() -> list[str]:
    text = TEX_FILE.read_text(encoding="utf-8")
    return re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", text)


def clean_assets_dir() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for path in ASSETS_DIR.glob("*"):
        if path.is_file():
            path.unlink()


def copy_assets() -> None:
    for name, source in COPIED_ASSETS.items():
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, ASSETS_DIR / name)


def verify_assets() -> None:
    missing = [name for name in required_assets_from_tex() if not (ASSETS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing presentation assets: {missing}")


def main() -> None:
    clean_assets_dir()
    copy_assets()

    helper = load_helper()
    helper.make_nuisance_variants()
    helper.make_constraint_variants()

    verify_assets()
    for asset in required_assets_from_tex():
        print(asset)


if __name__ == "__main__":
    main()
