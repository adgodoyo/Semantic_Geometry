"""Mahalanobis variants of the multilingual pair-breakdown figures.

Creates new figures matching the slide layouts used for:
  - raw vs language residualized
  - top-1 to top-3 deflation
  - top-4 to top-6 deflation

The left-side metric is Mahalanobis distance and the right-side metric remains
Euclidean distance. Existing cosine-based outputs are left untouched.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "multilingual"
PRESENTATION_ASSETS = PROJECT_ROOT / "presentation" / "assets"
MPL_DIR = RESULTS / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

from analysis_g_all_families import (
    CACHE_FILE,
    _build_exact_translation_metadata,
    _residualize,
    _topk_deflate,
)
from analysis_g_pair_metric_breakdown import (
    BASE_CONDITIONS,
    BASE_LABELS,
    CATEGORY_SPECS,
    COLOR_MAP,
    _build_pair_masks,
)


RESULTS.mkdir(exist_ok=True)
PRESENTATION_ASSETS.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAHAL_RANK = 128


def _suite_conditions(deflation_k: int):
    top_name = f"top{deflation_k}_deflation"
    conditions = BASE_CONDITIONS + [top_name]
    labels = dict(BASE_LABELS)
    labels[top_name] = f"Top-{deflation_k} deflation"
    return conditions, labels, top_name


def _whitened_coords_for_mahalanobis(
    X: np.ndarray,
    eps: float = 1e-6,
    max_rank: int = MAHAL_RANK,
) -> np.ndarray:
    """Return coordinates for a regularized truncated Mahalanobis metric.

    The exact full-rank metric is too expensive for repeated all-layer
    pairwise plots at this dataset size. We therefore use the leading
    randomized SVD directions and measure Euclidean distance in that
    whitened subspace.
    """

    X = X.astype(np.float32, copy=False)
    Xc = X - X.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    if n < 2:
        return np.zeros((n, 0), dtype=np.float32)

    k_eff = min(max_rank, n - 1, Xc.shape[1])
    if k_eff < 1:
        return np.zeros((n, 0), dtype=np.float32)

    U, s, _Vt = randomized_svd(Xc, n_components=k_eff, n_iter=3, random_state=0)
    eigvals = (s ** 2) / float(n - 1)
    scale = np.sqrt(n - 1.0) * np.sqrt(eigvals / (eigvals + eps))
    return (U * scale[None, :]).astype(np.float32, copy=False)


def _pairwise_upper_mahalanobis_and_l2(X: np.ndarray, ii: np.ndarray, jj: np.ndarray):
    X = X.astype(np.float32, copy=False)
    Y = _whitened_coords_for_mahalanobis(X)

    sq_y = (Y * Y).sum(axis=1)
    maha_sq = np.clip(sq_y[:, None] + sq_y[None, :] - 2.0 * (Y @ Y.T), 0.0, None)
    mahal = np.sqrt(maha_sq[ii, jj])

    sq_x = (X * X).sum(axis=1)
    l2_sq = np.clip(sq_x[:, None] + sq_x[None, :] - 2.0 * (X @ X.T), 0.0, None)
    l2 = np.sqrt(l2_sq[ii, jj])
    return mahal, l2


def _compute_pair_metric_breakdown_mahalanobis(
    acts: np.ndarray,
    factors: np.ndarray,
    domains: np.ndarray,
    langs_all: np.ndarray,
    exact_key_full: np.ndarray,
    Z_lang_all: np.ndarray,
    deflation_k: int,
) -> pd.DataFrame:
    ii, jj = np.triu_indices(len(factors), k=1)
    masks = _build_pair_masks(factors, domains, langs_all, exact_key_full, ii, jj)
    conditions, _labels, top_name = _suite_conditions(deflation_k)
    transform_fns = {
        "raw": lambda X: X,
        "language_residualized": lambda X: _residualize(X, Z_lang_all),
        top_name: lambda X: _topk_deflate(X, k=deflation_k),
    }

    rows = []
    for layer in range(acts.shape[1]):
        X_layer = acts[:, layer, :].astype(np.float32, copy=False)
        for condition in conditions:
            mahal, l2 = _pairwise_upper_mahalanobis_and_l2(transform_fns[condition](X_layer), ii, jj)
            for category, _label, _desc in CATEGORY_SPECS:
                mask = masks[category]
                rows.append(
                    {
                        "condition": condition,
                        "layer": int(layer),
                        "category": category,
                        "mean_mahalanobis": float(mahal[mask].mean()),
                        "mean_l2": float(l2[mask].mean()),
                        "pair_count": int(mask.sum()),
                    }
                )
    return pd.DataFrame(rows)


def _plot_raw_langresid(df: pd.DataFrame, out_png: Path):
    labels = dict(BASE_LABELS)
    conditions = ["raw", "language_residualized"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)

    for row_i, condition in enumerate(conditions):
        df_cond = df[df["condition"] == condition]
        for category, label, _desc in CATEGORY_SPECS:
            sub = df_cond[df_cond["category"] == category].sort_values("layer")
            axes[row_i, 0].plot(
                sub["layer"],
                sub["mean_mahalanobis"],
                label=label.replace("\n", " "),
                color=COLOR_MAP[category],
                marker="o",
                markersize=2,
            )
            axes[row_i, 1].plot(
                sub["layer"],
                sub["mean_l2"],
                label=label.replace("\n", " "),
                color=COLOR_MAP[category],
                marker="o",
                markersize=2,
            )

        axes[row_i, 0].set_title(f"{labels[condition]}: mean Mahalanobis distance")
        axes[row_i, 0].set_xlabel("Layer")
        axes[row_i, 0].set_ylabel("Mean Mahalanobis")
        axes[row_i, 0].grid(True, alpha=0.3)

        axes[row_i, 1].set_title(f"{labels[condition]}: mean Euclidean distance")
        axes[row_i, 1].set_xlabel("Layer")
        axes[row_i, 1].set_ylabel("Mean L2")
        axes[row_i, 1].grid(True, alpha=0.3)

    axes[0, 1].legend(fontsize=8, loc="best")
    fig.suptitle("G2 pair-category breakdown: raw and language residualized", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_topk_grid(df: pd.DataFrame, ks: list[int], out_png: Path, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)

    for col_i, k in enumerate(ks):
        condition = f"top{k}_deflation"
        df_cond = df[df["condition"] == condition]
        for category, label, _desc in CATEGORY_SPECS:
            sub = df_cond[df_cond["category"] == category].sort_values("layer")
            axes[0, col_i].plot(
                sub["layer"],
                sub["mean_mahalanobis"],
                label=label.replace("\n", " "),
                color=COLOR_MAP[category],
                marker="o",
                markersize=2,
            )
            axes[1, col_i].plot(
                sub["layer"],
                sub["mean_l2"],
                label=label.replace("\n", " "),
                color=COLOR_MAP[category],
                marker="o",
                markersize=2,
            )

        axes[0, col_i].set_title(f"Top-{k} deflation: mean Mahalanobis distance")
        axes[0, col_i].set_xlabel("Layer")
        axes[0, col_i].set_ylabel("Mean Mahalanobis")
        axes[0, col_i].grid(True, alpha=0.3)

        axes[1, col_i].set_title(f"Top-{k} deflation: mean Euclidean distance")
        axes[1, col_i].set_xlabel("Layer")
        axes[1, col_i].set_ylabel("Mean L2")
        axes[1, col_i].grid(True, alpha=0.3)

    axes[1, 0].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def run_pair_metric_breakdown_mahalanobis():
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Missing {CACHE_FILE}. Run extraction first.")

    data = np.load(CACHE_FILE, allow_pickle=True)
    acts = data["activations"]
    sentence_ids = data["sentence_ids"].astype(str)
    factors = data["factor_ids"].astype(str)
    domains = data["domain_ids"].astype(str)
    families = data["surface_families"].astype(str)
    langs_all = data["languages"].astype(str)

    langs_present = sorted(set(langs_all.tolist()))
    lang_to_i = {lang: i for i, lang in enumerate(langs_present)}
    lang_int = np.array([lang_to_i[lang] for lang in langs_all], dtype=int)
    Z_lang_all = np.eye(len(langs_present), dtype=np.float32)[lang_int]

    df_exact_meta = _build_exact_translation_metadata(
        sentence_ids=sentence_ids,
        factors=factors,
        families=families,
        langs=langs_all,
    )
    idx_exact = df_exact_meta["row_idx"].to_numpy(dtype=int)
    keys_exact = df_exact_meta["exact_key"].to_numpy(dtype=str)
    exact_key_full = np.full(len(factors), "", dtype=object)
    exact_key_full[idx_exact] = keys_exact

    frames = []
    for k in range(1, 7):
        df_k = _compute_pair_metric_breakdown_mahalanobis(
            acts=acts,
            factors=factors,
            domains=domains,
            langs_all=langs_all,
            exact_key_full=exact_key_full,
            Z_lang_all=Z_lang_all,
            deflation_k=k,
        )
        keep = [f"top{k}_deflation"]
        if k == 1:
            keep = ["raw", "language_residualized", f"top{k}_deflation"]
        frames.append(df_k[df_k["condition"].isin(keep)].copy())

    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv(RESULTS / "G2_pair_metric_breakdown_mahalanobis_full_layers.csv", index=False)

    raw_lang_png = RESULTS / "G2_pair_metric_breakdown_raw_langresid_mahalanobis.png"
    top123_png = RESULTS / "G2_pair_metric_breakdown_top123_mahalanobis.png"
    top456_png = RESULTS / "G2_pair_metric_breakdown_top456_mahalanobis.png"

    _plot_raw_langresid(
        df_all[df_all["condition"].isin(["raw", "language_residualized"])].copy(),
        raw_lang_png,
    )
    _plot_topk_grid(
        df_all[df_all["condition"].isin([f"top{k}_deflation" for k in [1, 2, 3]])].copy(),
        [1, 2, 3],
        top123_png,
        "G2 pair-category breakdown: top-1, top-2, top-3 deflation",
    )
    _plot_topk_grid(
        df_all[df_all["condition"].isin([f"top{k}_deflation" for k in [4, 5, 6]])].copy(),
        [4, 5, 6],
        top456_png,
        "G2 pair-category breakdown: top-4, top-5, top-6 deflation",
    )

    shutil.copy2(
        raw_lang_png,
        PRESENTATION_ASSETS / "nuisance_pair_breakdown_raw_langresid_mahalanobis_presentation.png",
    )
    shutil.copy2(
        top123_png,
        PRESENTATION_ASSETS / "nuisance_pair_breakdown_top123_mahalanobis_presentation.png",
    )
    shutil.copy2(
        top456_png,
        PRESENTATION_ASSETS / "nuisance_pair_breakdown_top456_mahalanobis_presentation.png",
    )

    print("[G2-Pairs-Mahalanobis] Done. New outputs written without replacing existing figures.")


if __name__ == "__main__":
    run_pair_metric_breakdown_mahalanobis()
