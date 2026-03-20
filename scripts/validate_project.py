#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np

from project_manifest import (
    CONSTRAINTS_BASE_RESULT_KEEP,
    JUNK_DIR_NAMES,
    JUNK_FILE_NAMES,
    LATEX_BUILD_SUFFIXES,
    MULTILINGUAL_RESULT_KEEP,
    MULTILINGUAL_SRC_KEEP,
    PAPER_KEEP,
    ROBUST_RESULT_KEEP,
    ROBUST_SRC_KEEP,
    ROOT,
    SCRIPT_KEEP,
    SLIDE_KEEP,
    paper_figure_keep,
    slide_asset_keep,
)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def require_executable(path: Path) -> None:
    require(path)
    if not os.access(path, os.X_OK):
        raise PermissionError(f"Script is not executable: {path}")


def forbid_external_refs(root: Path, snippets: list[str], suffixes: tuple[str, ...]) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet in text:
                raise RuntimeError(f"Found forbidden external reference in {path}: {snippet}")


def relative_file_set(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def require_exact_files(root: Path, expected: set[Path]) -> None:
    actual = relative_file_set(root)
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise RuntimeError(
            f"Unexpected file set in {root}\nMissing: {missing}\nExtra: {extra}"
        )


def assert_no_junk(root: Path) -> None:
    for path in root.rglob("*"):
        if path.name in JUNK_FILE_NAMES:
            raise RuntimeError(f"Unexpected junk file: {path}")
        if path.is_dir() and path.name in JUNK_DIR_NAMES:
            raise RuntimeError(f"Unexpected junk directory: {path}")
        if path.is_file() and path.suffix in LATEX_BUILD_SUFFIXES:
            raise RuntimeError(f"Unexpected LaTeX build artifact: {path}")


def main() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "MANIFEST.md",
        ROOT / ".gitignore",
        ROOT / "requirements.txt",
        ROOT / "data" / "multilingual" / "dataset.csv",
        ROOT / "data" / "multilingual" / "activations.npz",
        ROOT / "data" / "multilingual" / "activations_all_families.npz",
        ROOT / "data" / "constraints" / "semantic_refinement_dataset_v2_400.json",
        ROOT / "data" / "constraints" / "redundant_commonality_controls_400.json",
        ROOT / "results" / "constraints" / "base" / "full_pythia" / "acts_full_pythia.npy",
        ROOT / "results" / "constraints" / "base" / "control_pythia" / "acts_control_pythia.npy",
        ROOT / "paper" / "paper.tex",
        ROOT / "paper" / "paper.pdf",
        ROOT / "slides" / "slides.tex",
        ROOT / "slides" / "slides.pdf",
    ]
    for path in required:
        require(path)

    for path in [
        ROOT / "scripts" / "run_all.sh",
        ROOT / "scripts" / "extract_multilingual.sh",
        ROOT / "scripts" / "extract_constraints.sh",
        ROOT / "scripts" / "analyze_multilingual.sh",
        ROOT / "scripts" / "analyze_constraints.sh",
        ROOT / "scripts" / "build_slides.sh",
        ROOT / "scripts" / "build_paper.sh",
        ROOT / "scripts" / "curate_release.sh",
    ]:
        require_executable(path)

    require_exact_files(ROOT / "scripts", SCRIPT_KEEP)
    require_exact_files(ROOT / "src" / "multilingual", MULTILINGUAL_SRC_KEEP)
    require_exact_files(ROOT / "src" / "constraints" / "robust", ROBUST_SRC_KEEP)
    require_exact_files(ROOT / "results" / "multilingual", MULTILINGUAL_RESULT_KEEP)
    require_exact_files(ROOT / "results" / "constraints" / "base", CONSTRAINTS_BASE_RESULT_KEEP)
    require_exact_files(ROOT / "results" / "constraints" / "robust", ROBUST_RESULT_KEEP)
    require_exact_files(ROOT / "paper", PAPER_KEEP | paper_figure_keep())
    require_exact_files(ROOT / "slides", SLIDE_KEEP | slide_asset_keep())

    assert_no_junk(ROOT)

    forbidden = [
        "presentation_assets/",
        "Concepts_Robust_Pythia/",
        "Convergence_Semantic_Last/",
        "results/constraints/concepts_last/",
        "src/constraints/concepts_last/",
        "codebase/",
    ]
    forbid_external_refs(ROOT / "src", forbidden, (".py",))
    forbid_external_refs(ROOT / "scripts", forbidden, (".py", ".sh"))
    forbid_external_refs(ROOT / "slides", forbidden, (".tex",))
    forbid_external_refs(ROOT / "paper", forbidden, (".tex",))

    if (ROOT / "docs").exists():
        raise RuntimeError("Unexpected docs directory; historical notes should not remain in the curated package.")

    ref = np.load(ROOT / "results" / "constraints" / "base" / "full_pythia" / "acts_full_pythia.npy", mmap_mode="r")
    ctrl = np.load(ROOT / "results" / "constraints" / "base" / "control_pythia" / "acts_control_pythia.npy", mmap_mode="r")
    calib = np.load(ROOT / "data" / "multilingual" / "activations_all_families.npz")

    summary = {
        "acts_full_pythia_shape": tuple(ref.shape),
        "acts_control_pythia_shape": tuple(ctrl.shape),
        "activations_all_families_shape": tuple(calib["activations"].shape),
        "multilingual_results_files": len(MULTILINGUAL_RESULT_KEEP),
        "robust_results_files": len(ROBUST_RESULT_KEEP),
        "slide_assets": len(slide_asset_keep()),
        "paper_figures": len(paper_figure_keep()),
    }
    print(json.dumps(summary, indent=2))
    print("Curated Semantic Geometry package validation passed.")


if __name__ == "__main__":
    main()
