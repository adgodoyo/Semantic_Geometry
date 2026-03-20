#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

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


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def remove_junk(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.name in JUNK_FILE_NAMES:
            remove_path(path)
        elif path.is_dir() and path.name in JUNK_DIR_NAMES:
            shutil.rmtree(path)
        elif path.is_file() and path.suffix in LATEX_BUILD_SUFFIXES:
            path.unlink()


def prune_tree(root: Path, keep_rel_paths: set[Path]) -> None:
    if not root.exists():
        return

    for path in sorted(root.rglob("*"), reverse=True):
        rel = path.relative_to(root)
        if path.is_file():
            if rel not in keep_rel_paths:
                path.unlink()
        elif path.is_dir():
            if path == root:
                continue
            if not any(keep == rel or rel in keep.parents for keep in keep_rel_paths):
                shutil.rmtree(path)

    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and path != root and not any(path.iterdir()):
            path.rmdir()


def main() -> None:
    remove_junk(ROOT)

    remove_path(ROOT / "docs")

    prune_tree(ROOT / "scripts", SCRIPT_KEEP)
    prune_tree(ROOT / "src" / "multilingual", MULTILINGUAL_SRC_KEEP)
    prune_tree(ROOT / "src" / "constraints" / "robust", ROBUST_SRC_KEEP)
    prune_tree(ROOT / "slides", SLIDE_KEEP | slide_asset_keep())
    prune_tree(ROOT / "paper", PAPER_KEEP | paper_figure_keep())
    prune_tree(ROOT / "results" / "multilingual", MULTILINGUAL_RESULT_KEEP)
    prune_tree(ROOT / "results" / "constraints" / "base", CONSTRAINTS_BASE_RESULT_KEEP)
    prune_tree(ROOT / "results" / "constraints" / "robust", ROBUST_RESULT_KEEP)
    remove_junk(ROOT)

    print("Semantic Geometry release view curated.")


if __name__ == "__main__":
    main()
