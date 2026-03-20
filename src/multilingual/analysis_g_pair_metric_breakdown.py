"""analysis_g_pair_metric_breakdown.py — Pair-category averages for cosine and L2.

This analysis uses the all-family multilingual cache and reports, at each layer,
mean cosine similarity and mean Euclidean distance for a focused set of pair
categories under three transforms:

1. raw
2. language residualized
3. top-k deflation

Within each transform it reports:

1. exact translations across languages
2. same-factor pairs across languages (exact translations are included here)
3. same-factor pairs within the same language
4. same-domain, different-factor pairs across languages
5. same-domain, different-factor pairs within the same language
6. different-topic pairs across languages
7. different-topic pairs within the same language

Outputs (results/):
  - G2_pair_metric_breakdown_full_layers.csv
  - G2_pair_metric_breakdown_key_layers.csv
  - G2_pair_metric_breakdown_curves.png
  - G2_pair_metric_breakdown_keylayers.png
  - G2_pair_metric_breakdown_summary.md
  - G2_pair_metric_breakdown_full_layers_top3.csv
  - G2_pair_metric_breakdown_key_layers_top3.csv
  - G2_pair_metric_breakdown_curves_top3.png
  - G2_pair_metric_breakdown_keylayers_top3.png
  - G2_pair_metric_breakdown_summary_top3.md
  - G2_pair_metric_breakdown_raw_langresid.png
  - G2_pair_metric_breakdown_top123.png
  - G2_pair_metric_breakdown_top456.png
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_g_all_families import (
    CACHE_FILE,
    CHECK_LAYERS,
    _build_exact_translation_metadata,
    _residualize,
    _topk_deflate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "multilingual"
MPL_DIR = RESULTS / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
RESULTS.mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_CONDITIONS = ["raw", "language_residualized"]
BASE_LABELS = {
    "raw": "Raw",
    "language_residualized": "Language residualized",
}

CATEGORY_SPECS = [
    (
        "exact_translation_diff_language",
        "Exact translation\n(diff language)",
        "Same exact translation key, different language.",
    ),
    (
        "same_factor_diff_language",
        "Same factor\n(diff language)",
        "Same factor, different language. Exact translations are included here.",
    ),
    (
        "same_factor_same_language",
        "Same factor\n(same language)",
        "Same factor, same language.",
    ),
    (
        "same_domain_diff_language",
        "Same domain, diff factor\n(diff language)",
        "Same domain, different factor, different language.",
    ),
    (
        "same_domain_same_language",
        "Same domain, diff factor\n(same language)",
        "Same domain, different factor, same language.",
    ),
    (
        "different_topic_diff_language",
        "Different topic\n(diff language)",
        "Different domain/topic, different language.",
    ),
    (
        "different_topic_same_language",
        "Different topic\n(same language)",
        "Different domain/topic, same language.",
    ),
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


def _normalize(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    return X / nrm


def _pairwise_upper_cosine_and_l2(X: np.ndarray, ii: np.ndarray, jj: np.ndarray):
    X = X.astype(np.float32, copy=False)
    Xn = _normalize(X)
    cosine = (Xn @ Xn.T)[ii, jj]

    sq = (X * X).sum(axis=1)
    l2_sq = np.clip(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0, None)
    l2 = np.sqrt(l2_sq[ii, jj])
    return cosine, l2


def _build_pair_masks(factors, domains, langs_all, exact_key_full, ii, jj):
    same_factor = factors[ii] == factors[jj]
    same_domain = domains[ii] == domains[jj]
    same_lang = langs_all[ii] == langs_all[jj]
    same_exact = (exact_key_full[ii] != "") & (exact_key_full[ii] == exact_key_full[jj])

    return {
        "exact_translation_diff_language": same_exact & (~same_lang),
        "same_factor_diff_language": same_factor & (~same_lang),
        "same_factor_same_language": same_factor & same_lang,
        "same_domain_diff_language": (~same_factor) & same_domain & (~same_lang),
        "same_domain_same_language": (~same_factor) & same_domain & same_lang,
        "different_topic_diff_language": (~same_domain) & (~same_lang),
        "different_topic_same_language": (~same_domain) & same_lang,
    }


def _suite_conditions(deflation_k: int):
    top_name = f"top{deflation_k}_deflation"
    conditions = BASE_CONDITIONS + [top_name]
    labels = dict(BASE_LABELS)
    labels[top_name] = f"Top-{deflation_k} deflation"
    return conditions, labels, top_name


def _compute_pair_metric_breakdown(
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
            cosine, l2 = _pairwise_upper_cosine_and_l2(transform_fns[condition](X_layer), ii, jj)
            for category, _label, _desc in CATEGORY_SPECS:
                mask = masks[category]
                rows.append(
                    {
                        "condition": condition,
                        "layer": int(layer),
                        "category": category,
                        "mean_cosine": float(cosine[mask].mean()),
                        "mean_l2": float(l2[mask].mean()),
                        "pair_count": int(mask.sum()),
                    }
                )
    return pd.DataFrame(rows)


def _plot_curves(df: pd.DataFrame, out_png: Path, deflation_k: int):
    conditions, labels, _top_name = _suite_conditions(deflation_k)
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=True)

    for row_i, condition in enumerate(conditions):
        df_cond = df[df["condition"] == condition]
        for category, label, _desc in CATEGORY_SPECS:
            sub = df_cond[df_cond["category"] == category].sort_values("layer")
            axes[row_i, 0].plot(
                sub["layer"],
                sub["mean_cosine"],
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

        axes[row_i, 0].axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
        axes[row_i, 0].set_title(f"{labels[condition]}: mean cosine")
        axes[row_i, 0].set_xlabel("Layer")
        axes[row_i, 0].set_ylabel("Mean cosine")
        axes[row_i, 0].grid(True, alpha=0.3)

        axes[row_i, 1].set_title(f"{labels[condition]}: mean Euclidean distance")
        axes[row_i, 1].set_xlabel("Layer")
        axes[row_i, 1].set_ylabel("Mean L2")
        axes[row_i, 1].grid(True, alpha=0.3)

    axes[0, 1].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"G2 pair-category breakdown: cosine and Euclidean distance (top-{deflation_k} suite)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_condition_group(df: pd.DataFrame, conditions: list[str], labels: dict, out_png: Path, title: str):
    fig, axes = plt.subplots(len(conditions), 2, figsize=(15, 4.5 * len(conditions)), sharex=True)
    if len(conditions) == 1:
        axes = np.array([axes])

    for row_i, condition in enumerate(conditions):
        df_cond = df[df["condition"] == condition]
        for category, label, _desc in CATEGORY_SPECS:
            sub = df_cond[df_cond["category"] == category].sort_values("layer")
            axes[row_i, 0].plot(
                sub["layer"],
                sub["mean_cosine"],
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

        axes[row_i, 0].axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
        axes[row_i, 0].set_title(f"{labels[condition]}: mean cosine")
        axes[row_i, 0].set_xlabel("Layer")
        axes[row_i, 0].set_ylabel("Mean cosine")
        axes[row_i, 0].grid(True, alpha=0.3)

        axes[row_i, 1].set_title(f"{labels[condition]}: mean Euclidean distance")
        axes[row_i, 1].set_xlabel("Layer")
        axes[row_i, 1].set_ylabel("Mean L2")
        axes[row_i, 1].grid(True, alpha=0.3)

    axes[0, 1].legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_keylayer_heatmaps(df_key: pd.DataFrame, out_png: Path, deflation_k: int):
    conditions, labels, _top_name = _suite_conditions(deflation_k)
    categories = [spec[0] for spec in CATEGORY_SPECS]
    label_rows = [spec[1] for spec in CATEGORY_SPECS]
    layers = sorted(df_key["layer"].unique())

    fig, axes = plt.subplots(3, 2, figsize=(13, 15))

    for row_i, condition in enumerate(conditions):
        df_cond = df_key[df_key["condition"] == condition]
        cos_mat = (
            df_cond.pivot(index="category", columns="layer", values="mean_cosine")
            .reindex(index=categories, columns=layers)
            .values
        )
        l2_mat = (
            df_cond.pivot(index="category", columns="layer", values="mean_l2")
            .reindex(index=categories, columns=layers)
            .values
        )

        im0 = axes[row_i, 0].imshow(cos_mat, cmap="viridis", aspect="auto")
        axes[row_i, 0].set_title(f"{labels[condition]}: mean cosine at key layers")
        axes[row_i, 0].set_xticks(range(len(layers)))
        axes[row_i, 0].set_xticklabels(layers)
        axes[row_i, 0].set_yticks(range(len(label_rows)))
        axes[row_i, 0].set_yticklabels(label_rows)
        for i in range(len(label_rows)):
            for j in range(len(layers)):
                axes[row_i, 0].text(j, i, f"{cos_mat[i, j]:.3f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im0, ax=axes[row_i, 0], fraction=0.046, pad=0.04)

        im1 = axes[row_i, 1].imshow(l2_mat, cmap="viridis_r", aspect="auto")
        axes[row_i, 1].set_title(f"{labels[condition]}: mean L2 at key layers")
        axes[row_i, 1].set_xticks(range(len(layers)))
        axes[row_i, 1].set_xticklabels(layers)
        axes[row_i, 1].set_yticks(range(len(label_rows)))
        axes[row_i, 1].set_yticklabels(label_rows)
        for i in range(len(label_rows)):
            for j in range(len(layers)):
                axes[row_i, 1].text(j, i, f"{l2_mat[i, j]:.3f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im1, ax=axes[row_i, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"G2 pair-category breakdown at layers 0, 16, 32 (top-{deflation_k} suite)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _write_summary(df_full: pd.DataFrame, df_key: pd.DataFrame, out_md: Path, deflation_k: int):
    conditions, _labels, top_name = _suite_conditions(deflation_k)
    pair_counts = (
        df_full[["category", "pair_count"]]
        .drop_duplicates()
        .sort_values("category")
    )
    definitions = pd.DataFrame(
        [{"category": category, "description": desc} for category, _label, desc in CATEGORY_SPECS]
    )

    lines = [
        "# G2 Pair-Category Metric Breakdown",
        "",
        f"This analysis reports mean cosine similarity and mean Euclidean distance for seven pair categories under three transforms: raw, language residualized, and top-{deflation_k} deflation.",
        "",
        "Important: `same_factor_diff_language` is the broad cross-language same-meaning category, and exact translations are a subset of it.",
        "",
        "## Category definitions",
        "",
        definitions.to_markdown(index=False),
        "",
        "## Pair counts",
        "",
        pair_counts.to_markdown(index=False),
    ]

    for condition in conditions:
        key_table = (
            df_key[df_key["condition"] == condition]
            .sort_values(["layer", "category"])
            .round({"mean_cosine": 4, "mean_l2": 4})
        )
        lines.extend(
            [
                "",
                f"## Key-layer values ({condition})",
                "",
                key_table.to_markdown(index=False),
            ]
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def _run_suite(
    acts: np.ndarray,
    factors: np.ndarray,
    domains: np.ndarray,
    langs_all: np.ndarray,
    exact_key_full: np.ndarray,
    Z_lang_all: np.ndarray,
    deflation_k: int,
    suffix: str = "",
):
    df_full = _compute_pair_metric_breakdown(
        acts=acts,
        factors=factors,
        domains=domains,
        langs_all=langs_all,
        exact_key_full=exact_key_full,
        Z_lang_all=Z_lang_all,
        deflation_k=deflation_k,
    )
    df_full.to_csv(RESULTS / f"G2_pair_metric_breakdown_full_layers{suffix}.csv", index=False)

    layers = [layer for layer in CHECK_LAYERS if layer < acts.shape[1]]
    df_key = df_full[df_full["layer"].isin(layers)].copy()
    df_key.to_csv(RESULTS / f"G2_pair_metric_breakdown_key_layers{suffix}.csv", index=False)

    _plot_curves(df_full, RESULTS / f"G2_pair_metric_breakdown_curves{suffix}.png", deflation_k=deflation_k)
    _plot_keylayer_heatmaps(
        df_key,
        RESULTS / f"G2_pair_metric_breakdown_keylayers{suffix}.png",
        deflation_k=deflation_k,
    )
    _write_summary(
        df_full,
        df_key,
        RESULTS / f"G2_pair_metric_breakdown_summary{suffix}.md",
        deflation_k=deflation_k,
    )


def _compute_all_conditions_df(
    acts: np.ndarray,
    factors: np.ndarray,
    domains: np.ndarray,
    langs_all: np.ndarray,
    exact_key_full: np.ndarray,
    Z_lang_all: np.ndarray,
) -> pd.DataFrame:
    frames = []
    for k in range(1, 7):
        df_k = _compute_pair_metric_breakdown(
            acts=acts,
            factors=factors,
            domains=domains,
            langs_all=langs_all,
            exact_key_full=exact_key_full,
            Z_lang_all=Z_lang_all,
            deflation_k=k,
        )
        top_name = f"top{k}_deflation"
        keep = [top_name]
        if k == 1:
            keep = ["raw", "language_residualized", top_name]
        frames.append(df_k[df_k["condition"].isin(keep)].copy())
    return pd.concat(frames, ignore_index=True)


def run_pair_metric_breakdown():
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

    _run_suite(
        acts=acts,
        factors=factors,
        domains=domains,
        langs_all=langs_all,
        exact_key_full=exact_key_full,
        Z_lang_all=Z_lang_all,
        deflation_k=5,
        suffix="",
    )
    _run_suite(
        acts=acts,
        factors=factors,
        domains=domains,
        langs_all=langs_all,
        exact_key_full=exact_key_full,
        Z_lang_all=Z_lang_all,
        deflation_k=3,
        suffix="_top3",
    )

    df_all = _compute_all_conditions_df(
        acts=acts,
        factors=factors,
        domains=domains,
        langs_all=langs_all,
        exact_key_full=exact_key_full,
        Z_lang_all=Z_lang_all,
    )

    all_labels = dict(BASE_LABELS)
    for k in range(1, 7):
        all_labels[f"top{k}_deflation"] = f"Top-{k} deflation"

    _plot_condition_group(
        df_all,
        conditions=["raw", "language_residualized"],
        labels=all_labels,
        out_png=RESULTS / "G2_pair_metric_breakdown_raw_langresid.png",
        title="G2 pair-category breakdown: raw and language residualized",
    )
    _plot_condition_group(
        df_all,
        conditions=["top1_deflation", "top2_deflation", "top3_deflation"],
        labels=all_labels,
        out_png=RESULTS / "G2_pair_metric_breakdown_top123.png",
        title="G2 pair-category breakdown: top-1, top-2, top-3 deflation",
    )
    _plot_condition_group(
        df_all,
        conditions=["top4_deflation", "top5_deflation", "top6_deflation"],
        labels=all_labels,
        out_png=RESULTS / "G2_pair_metric_breakdown_top456.png",
        title="G2 pair-category breakdown: top-4, top-5, top-6 deflation",
    )

    print("[G2-Pairs] Done. Outputs in results/.")


if __name__ == "__main__":
    run_pair_metric_breakdown()
