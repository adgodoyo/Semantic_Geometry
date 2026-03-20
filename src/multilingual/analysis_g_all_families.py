"""analysis_g_all_families.py — Analysis G extension for multilingual F1–F5.

Runs an Analysis-G-style language-pair cosine matrix using the expanded
multilingual dataset (new F2–F5 translations), while preserving old outputs.

Outputs (results/):
  - G2_language_matrix_all_families.csv
  - G2_language_matrix_all_families.png
  - G2_language_matrix_by_family.csv
  - G2_language_matrix_by_family.png
  - G2_exact_translation_matrix.csv
  - G2_exact_translation_matrix.png
  - G2_exact_vs_all_layerwise.csv
  - G2_exact_vs_all_layerwise.png
  - G2_exact_translation_alignment.csv
  - G2_exact_translation_alignment.png
  - G2_crosslingual_index_by_family.csv
  - G2_crosslingual_index_by_family_lang_resid.csv
  - G2_crosslingual_index_by_family_top3_deflation.csv
  - G2_crosslingual_index_by_family_top5_deflation.csv
  - G2_crosslingual_index_by_family.png
  - G2_crosslingual_index_by_family_lang_resid.png
  - G2_crosslingual_index_by_family_top3_deflation.png
  - G2_crosslingual_index_by_family_top5_deflation.png
  - G2_deflation_focus_lang_resid.csv
  - G2_deflation_focus_lang_resid.png
  - G2_deflation_focus_top3.csv
  - G2_deflation_focus_top3.png
  - G2_deflation_focus_top5.csv
  - G2_deflation_focus_top5.png
  - G2_deflation_focus_comparison.png
  - G2_summary.md
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "multilingual"
RESULTS = PROJECT_ROOT / "results" / "multilingual"
MPL_DIR = RESULTS / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

CACHE_FILE = DATA_DIR / "activations_all_families.npz"
RESULTS.mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANG_ORDER = ["en", "de", "es", "ar", "zh"]
LANG_DISPLAY = ["EN", "DE", "ES", "AR", "ZH"]
FAMILY_ORDER = ["F1", "F2", "F3", "F4", "F5"]
CHECK_LAYERS = [0, 16, 32]


def _normalize(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    return X / nrm


def _matrix_for_subset(Xn: np.ndarray, factors: np.ndarray, langs: np.ndarray, lang_to_i: dict):
    """Compute mean cosine matrix for same-factor pairs for a subset.

    Returns:
      mean_mat (n_lang, n_lang), cnt_mat (n_lang, n_lang)
    """
    n_lang = len(lang_to_i)
    mat = np.zeros((n_lang, n_lang), dtype=np.float64)
    cnt = np.zeros((n_lang, n_lang), dtype=np.int64)

    unique_factors = np.unique(factors)

    for fac in unique_factors:
        idx = np.where(factors == fac)[0]
        if len(idx) < 2:
            continue

        Xf = Xn[idx]
        lf = langs[idx]
        cos = Xf @ Xf.T

        m = len(idx)
        for a in range(m):
            la = lang_to_i[lf[a]]
            for b in range(a + 1, m):
                lb = lang_to_i[lf[b]]
                v = float(cos[a, b])
                mat[la, lb] += v
                mat[lb, la] += v
                cnt[la, lb] += 1
                cnt[lb, la] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_mat = np.where(cnt > 0, mat / cnt, np.nan)

    return mean_mat, cnt


def _record_matrix_rows(rows, group_name, layer, mean_mat, cnt_mat, langs_present):
    for i, li in enumerate(langs_present):
        for j, lj in enumerate(langs_present):
            rows.append(
                {
                    "group": group_name,
                    "layer": int(layer),
                    "lang_i": li,
                    "lang_j": lj,
                    "mean_cosine": float(mean_mat[i, j]),
                    "pair_count": int(cnt_mat[i, j]),
                }
            )


def _crosslingual_index(mean_mat: np.ndarray):
    """Return (diag_mean, offdiag_mean, gap)."""
    n = mean_mat.shape[0]
    diag = []
    off = []
    for i in range(n):
        for j in range(n):
            if np.isnan(mean_mat[i, j]):
                continue
            if i == j:
                diag.append(mean_mat[i, j])
            else:
                off.append(mean_mat[i, j])
    diag_mean = float(np.mean(diag)) if diag else np.nan
    off_mean = float(np.mean(off)) if off else np.nan
    return diag_mean, off_mean, diag_mean - off_mean


def _residualize(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Remove from X the linear component explained by Z (with intercept)."""
    X = X.astype(np.float32, copy=False)
    Z = Z.astype(np.float32, copy=False)
    ones = np.ones((Z.shape[0], 1), dtype=np.float32)
    Zaug = np.concatenate([ones, Z], axis=1)
    W = np.linalg.lstsq(Zaug, X, rcond=None)[0]
    return X - Zaug @ W


def _topk_deflate(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Deflate top-k global directions using centered SVD."""
    X = X.astype(np.float32, copy=False)
    Xc = X - X.mean(axis=0, keepdims=True)
    # N x H with N < H; full_matrices=False keeps SVD tractable.
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    k_eff = max(1, min(k, Vt.shape[0]))
    Vk = Vt[:k_eff].T  # (H, k)
    return Xc - (Xc @ Vk) @ Vk.T


def _compute_layerwise_curve(
    acts: np.ndarray,
    factors: np.ndarray,
    families: np.ndarray,
    langs_all: np.ndarray,
    lang_to_i: dict,
    transform_fn,
) -> pd.DataFrame:
    """Compute layerwise cross-lingual indices for ALL + F1..F5 groups."""
    all_groups = ["ALL"] + FAMILY_ORDER
    group_masks = {"ALL": np.ones(len(families), dtype=bool)}
    for fam in FAMILY_ORDER:
        group_masks[fam] = families == fam

    rows = []
    for l in range(acts.shape[1]):
        X_proc = transform_fn(acts[:, l, :].astype(np.float32), l)
        for group in all_groups:
            mask = group_masks[group]
            fac_g = factors[mask]
            lan_g = langs_all[mask]
            Xn = _normalize(X_proc[mask])
            mean_mat, _ = _matrix_for_subset(Xn, fac_g, lan_g, lang_to_i)
            diag_mean, off_mean, gap = _crosslingual_index(mean_mat)
            rows.append(
                {
                    "group": group,
                    "layer": int(l),
                    "diag_mean": diag_mean,
                    "offdiag_mean": off_mean,
                    "diag_minus_offdiag": gap,
                }
            )
    return pd.DataFrame(rows)


def _plot_layerwise_curve(df_curve: pd.DataFrame, out_png: Path, title: str):
    """Plot offdiag and diag-offdiag layerwise curves for ALL + families."""
    all_groups = ["ALL"] + FAMILY_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    ax = axes[0]
    for group in all_groups:
        sub = df_curve[df_curve["group"] == group].sort_values("layer")
        ax.plot(sub["layer"], sub["offdiag_mean"], label=group, marker="o", markersize=2)
    ax.set_title("Cross-language mean cosine (off-diagonal)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean off-diagonal cosine")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)

    ax = axes[1]
    for group in all_groups:
        sub = df_curve[df_curve["group"] == group].sort_values("layer")
        ax.plot(sub["layer"], sub["diag_minus_offdiag"], label=group, marker="o", markersize=2)
    ax.set_title("Language specificity (diag - offdiag)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean diagonal - off-diagonal cosine")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _build_exact_translation_metadata(
    sentence_ids: np.ndarray,
    factors: np.ndarray,
    families: np.ndarray,
    langs: np.ndarray,
) -> pd.DataFrame:
    """Build metadata for exact-translation pairs keyed by (factor, family, variant 0/1).

    English has 12 sentences/factor; exact multilingual mapping uses:
      F1: EN indices 00,01
      F2: EN indices 02,03
      F3: EN indices 04,05 (first 2 only; 06,07 excluded)
      F4: EN indices 08,09
      F5: EN indices 10,11
    """
    en_map = {
        "F1": {0: 0, 1: 1},
        "F2": {2: 0, 3: 1},
        "F3": {4: 0, 5: 1},
        "F4": {8: 0, 9: 1},
        "F5": {10: 0, 11: 1},
    }

    rows = []
    for i, sid in enumerate(sentence_ids.astype(str)):
        fac = str(factors[i])
        fam = str(families[i])
        lan = str(langs[i])
        variant = None

        if lan == "en":
            # English ids follow: "{factor}_{idx:02d}"
            try:
                idx = int(sid.rsplit("_", 1)[1])
            except Exception:
                idx = None
            if idx is not None and fam in en_map and idx in en_map[fam]:
                variant = en_map[fam][idx]
        else:
            # Multilingual ids are either:
            #   F1: "{factor}_ml_{lang}_{idx:02d}"
            #   F2-F5: "{factor}_ml_{family}_{lang}_{idx:02d}"
            parts = sid.split("_")
            if len(parts) == 4 and parts[1] == "ml":
                # F1 form
                fam_p = "F1"
                lan_p = parts[2]
                idx_p = int(parts[3])
                if fam_p == fam and lan_p == lan:
                    variant = idx_p if idx_p in (0, 1) else None
            elif len(parts) == 5 and parts[1] == "ml":
                fam_p = parts[2]
                lan_p = parts[3]
                idx_p = int(parts[4])
                if fam_p == fam and lan_p == lan:
                    variant = idx_p if idx_p in (0, 1) else None

        if variant is None:
            continue

        key = f"{fac}|{fam}|{variant}"
        rows.append(
            {
                "row_idx": i,
                "sentence_id": sid,
                "factor_id": fac,
                "family": fam,
                "language": lan,
                "variant": int(variant),
                "exact_key": key,
            }
        )

    df = pd.DataFrame(rows)
    return df


def _exact_alignment_summary(df_exact: pd.DataFrame, langs_present: list) -> pd.DataFrame:
    """Summarize exact-translation key coverage by family and language."""
    n_factors = df_exact["factor_id"].nunique()
    expected = n_factors * 2  # two exact variants per family/factor

    out = []
    for fam in FAMILY_ORDER:
        sub_f = df_exact[df_exact["family"] == fam]
        for lan in langs_present:
            sub = sub_f[sub_f["language"] == lan]
            keys_found = int(sub["exact_key"].nunique())
            dup_keys = int(sub.duplicated(subset=["exact_key"]).sum())
            out.append(
                {
                    "family": fam,
                    "language": lan,
                    "keys_found": keys_found,
                    "expected_keys": expected,
                    "coverage": keys_found / expected if expected > 0 else np.nan,
                    "duplicate_keys": dup_keys,
                }
            )
    return pd.DataFrame(out)


def _plot_exact_alignment(df_align: pd.DataFrame, langs_present: list, out_png: Path):
    """Plot coverage heatmap for exact-translation key matching."""
    mat = (
        df_align.pivot(index="family", columns="language", values="coverage")
        .reindex(index=FAMILY_ORDER, columns=langs_present)
        .values
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(langs_present)))
    ax.set_yticks(range(len(FAMILY_ORDER)))
    ax.set_xticklabels([x.upper() for x in langs_present])
    ax.set_yticklabels(FAMILY_ORDER)
    ax.set_title("Exact-translation key coverage (family × language)")
    for i in range(len(FAMILY_ORDER)):
        for j in range(len(langs_present)):
            v = mat[i, j]
            ax.text(j, i, f"{100*v:.0f}%", ha="center", va="center",
                    color="white" if v < 0.6 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Coverage")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _exact_matrix_for_subset(
    Xn: np.ndarray,
    exact_keys: np.ndarray,
    langs: np.ndarray,
    lang_to_i: dict,
):
    """Compute language matrix using exact translation keys only.

    Off-diagonal (li, lj): mean cosine across rows with the same exact_key.
    Diagonal is undefined (NaN), since there is one sentence per key/language.
    """
    n_lang = len(lang_to_i)
    mat = np.full((n_lang, n_lang), np.nan, dtype=np.float64)
    cnt = np.zeros((n_lang, n_lang), dtype=np.int64)

    # key -> (language -> row)
    key_to_lang_row = {}
    for r, (k, lan) in enumerate(zip(exact_keys, langs)):
        key_to_lang_row.setdefault(k, {})
        # Keep first row if duplicates exist (duplicates are reported separately).
        if lan not in key_to_lang_row[k]:
            key_to_lang_row[k][lan] = r

    for li, lan_i in enumerate(lang_to_i.keys()):
        for lj, lan_j in enumerate(lang_to_i.keys()):
            if li == lj:
                continue
            vals = []
            for _k, mp in key_to_lang_row.items():
                if lan_i in mp and lan_j in mp:
                    a = mp[lan_i]
                    b = mp[lan_j]
                    vals.append(float(Xn[a] @ Xn[b]))
            if vals:
                mat[li, lj] = float(np.mean(vals))
                cnt[li, lj] = len(vals)

    return mat, cnt


def _plot_exact_matrix_heatmaps(mats_exact: dict, langs_present: list, disp_present: list, out_png: Path, title: str):
    """Plot exact-translation matrices for selected layers."""
    vals = np.concatenate([m[~np.isnan(m)] for m in mats_exact.values() if np.any(~np.isnan(m))])
    vmin = max(-1.0, float(np.nanmin(vals)) - 0.02)
    vmax = min(1.0, float(np.nanmax(vals)) + 0.02)
    fig, axes = plt.subplots(1, len(mats_exact), figsize=(5 * len(mats_exact), 4))
    if len(mats_exact) == 1:
        axes = [axes]
    for ax, l in zip(axes, mats_exact.keys()):
        mat = mats_exact[l]
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("lightgray")
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(langs_present)))
        ax.set_yticks(range(len(langs_present)))
        ax.set_xticklabels(disp_present)
        ax.set_yticklabels(disp_present)
        ax.set_title(f"Layer {l}")
        mid = (vmin + vmax) / 2.0
        for i in range(len(langs_present)):
            for j in range(len(langs_present)):
                v = mat[i, j]
                txt = "—" if np.isnan(v) else f"{v:.3f}"
                ax.text(
                    j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if (not np.isnan(v) and v < mid) else "black",
                )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _compute_exact_outputs(
    acts: np.ndarray,
    factors: np.ndarray,
    langs_all: np.ndarray,
    idx_exact: np.ndarray,
    keys_exact: np.ndarray,
    langs_exact: np.ndarray,
    lang_to_i: dict,
    langs_present: list,
    disp_present: list,
    layers: list,
    transform_fn,
    matrix_csv: Path,
    matrix_png: Path,
    compare_csv: Path,
    compare_png: Path,
    title_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute exact-translation matrices and all-vs-exact layerwise comparison."""
    rows_exact = []
    mats_exact = {}
    curve_cmp_rows = []
    en_i = lang_to_i["en"]
    zh_i = lang_to_i["zh"]
    offdiag_mask = ~np.eye(len(langs_present), dtype=bool)

    for l in range(acts.shape[1]):
        X_proc = transform_fn(acts[:, l, :].astype(np.float32), l)
        X_all_n = _normalize(X_proc)
        X_exact_n = _normalize(X_proc[idx_exact])

        mat_all, _ = _matrix_for_subset(X_all_n, factors, langs_all, lang_to_i)
        mat_exact, cnt_exact = _exact_matrix_for_subset(X_exact_n, keys_exact, langs_exact, lang_to_i)

        if l in layers:
            mats_exact[l] = mat_exact
            for i, li in enumerate(langs_present):
                for j, lj in enumerate(langs_present):
                    rows_exact.append(
                        {
                            "layer": int(l),
                            "lang_i": li,
                            "lang_j": lj,
                            "mean_cosine": float(mat_exact[i, j]),
                            "pair_count": int(cnt_exact[i, j]),
                        }
                    )

        curve_cmp_rows.append(
            {
                "layer": int(l),
                "offdiag_all_pairs": float(np.nanmean(mat_all[offdiag_mask])),
                "offdiag_exact_translation": float(np.nanmean(mat_exact[offdiag_mask])),
                "enzh_all_pairs": float(mat_all[en_i, zh_i]),
                "enzh_exact_translation": float(mat_exact[en_i, zh_i]),
            }
        )

    df_exact_matrix = pd.DataFrame(rows_exact)
    df_exact_matrix.to_csv(matrix_csv, index=False)
    _plot_exact_matrix_heatmaps(mats_exact, langs_present, disp_present, matrix_png, f"{title_prefix} language-pair cosine")

    df_cmp = pd.DataFrame(curve_cmp_rows)
    df_cmp.to_csv(compare_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    ax = axes[0]
    ax.plot(df_cmp["layer"], df_cmp["offdiag_all_pairs"], label="All pairs", marker="o", markersize=2)
    ax.plot(df_cmp["layer"], df_cmp["offdiag_exact_translation"], label="Exact translations", marker="o", markersize=2)
    ax.set_title("Off-diagonal mean cosine")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean cross-language cosine")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(df_cmp["layer"], df_cmp["enzh_all_pairs"], label="EN-ZH all pairs", marker="o", markersize=2)
    ax.plot(df_cmp["layer"], df_cmp["enzh_exact_translation"], label="EN-ZH exact translations", marker="o", markersize=2)
    ax.set_title("EN-ZH cosine")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.suptitle(f"{title_prefix}: all-pairs vs exact translations", fontsize=12)
    plt.tight_layout()
    plt.savefig(compare_png, dpi=150)
    plt.close()

    return df_exact_matrix, df_cmp


def _compute_deflation_focus_curve(
    acts: np.ndarray,
    factors: np.ndarray,
    domains: np.ndarray,
    families: np.ndarray,
    langs_all: np.ndarray,
    exact_key_full: np.ndarray,
    transform_fn,
) -> pd.DataFrame:
    """Compute category-wise cosine curves after a global transform per layer.

    Categories:
      - exact_translation: same exact key, different language
      - within_family_same_meaning: same factor and same family, excluding exact_translation
      - same_topic_diff_meaning: same domain, different factor
      - different_topics: different domain
    """
    ii, jj = np.triu_indices(len(factors), k=1)
    same_factor = factors[ii] == factors[jj]
    same_domain = domains[ii] == domains[jj]
    same_family = families[ii] == families[jj]
    same_lang = langs_all[ii] == langs_all[jj]
    same_exact = (exact_key_full[ii] != "") & (exact_key_full[ii] == exact_key_full[jj])

    masks = {
        "exact_translation": same_exact & (~same_lang),
        "within_family_same_meaning": same_factor & same_family & (~(same_exact & (~same_lang))),
        "same_topic_diff_meaning": (~same_factor) & same_domain,
        "different_topics": ~same_domain,
    }

    rows = []
    for l in range(acts.shape[1]):
        X = transform_fn(acts[:, l, :].astype(np.float32), l)
        Xn = _normalize(X)
        cos = (Xn @ Xn.T)[ii, jj]
        for category, mask in masks.items():
            rows.append(
                {
                    "layer": int(l),
                    "category": category,
                    "mean_cosine": float(cos[mask].mean()),
                    "pair_count": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def _plot_deflation_focus(df: pd.DataFrame, out_png: Path, title: str):
    category_order = [
        "exact_translation",
        "within_family_same_meaning",
        "same_topic_diff_meaning",
        "different_topics",
    ]
    label_map = {
        "exact_translation": "Exact translations",
        "within_family_same_meaning": "Within family, same meaning",
        "same_topic_diff_meaning": "Same topic, different meaning",
        "different_topics": "Different topics",
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    for cat in category_order:
        sub = df[df["category"] == cat].sort_values("layer")
        ax.plot(sub["layer"], sub["mean_cosine"], label=label_map[cat], marker="o", markersize=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean cosine")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_deflation_focus_comparison(
    df_lang: pd.DataFrame,
    df_top5: pd.DataFrame,
    df_top3: pd.DataFrame,
    out_png: Path,
):
    """Plot aligned 3-panel comparison with shared y-axis."""
    category_order = [
        "exact_translation",
        "within_family_same_meaning",
        "same_topic_diff_meaning",
        "different_topics",
    ]
    label_map = {
        "exact_translation": "Exact translations",
        "within_family_same_meaning": "Within family, same meaning",
        "same_topic_diff_meaning": "Same topic, different meaning",
        "different_topics": "Different topics",
    }
    color_map = {
        "exact_translation": "C0",
        "within_family_same_meaning": "C1",
        "same_topic_diff_meaning": "C2",
        "different_topics": "C3",
    }

    panels = [
        ("Language residualized", df_lang),
        ("Top-5 deflation", df_top5),
        ("Top-3 deflation", df_top3),
    ]

    all_vals = pd.concat([df_lang["mean_cosine"], df_top5["mean_cosine"], df_top3["mean_cosine"]], ignore_index=True)
    y_min = float(all_vals.min()) - 0.03
    y_max = float(all_vals.max()) + 0.03

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (title, df) in zip(axes, panels):
        for cat in category_order:
            sub = df[df["category"] == cat].sort_values("layer")
            ax.plot(
                sub["layer"],
                sub["mean_cosine"],
                label=label_map[cat],
                color=color_map[cat],
                marker="o",
                markersize=2,
            )
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_min, y_max)
    axes[0].set_ylabel("Mean cosine")
    axes[0].legend(fontsize=8)
    fig.suptitle("G2. Deflation-first pair-category focus comparison", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def run_analysis_g2():
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {CACHE_FILE}. Run: python extract_all_families.py"
        )

    data = np.load(CACHE_FILE, allow_pickle=True)
    acts = data["activations"]
    sentence_ids = data["sentence_ids"].astype(str)
    factors = data["factor_ids"].astype(str)
    domains = data["domain_ids"].astype(str)
    families = data["surface_families"].astype(str)
    langs_all = data["languages"].astype(str)

    langs_present = [l for l in LANG_ORDER if l in set(langs_all)]
    disp_present = [LANG_DISPLAY[LANG_ORDER.index(l)] for l in langs_present]
    lang_to_i = {l: i for i, l in enumerate(langs_present)}
    lang_int = np.array([lang_to_i[l] for l in langs_all], dtype=int)
    Z_lang_all = np.eye(len(langs_present), dtype=np.float32)[lang_int]

    layers = [l for l in CHECK_LAYERS if l < acts.shape[1]]

    rows_matrix = []
    rows_index = []

    all_group_mats = {"ALL": {}}
    family_group_mats = {fam: {} for fam in FAMILY_ORDER}

    # ── Group ALL families together ───────────────────────────────────────
    for l in layers:
        Xn = _normalize(acts[:, l, :].astype(np.float32))
        mean_mat, cnt_mat = _matrix_for_subset(Xn, factors, langs_all, lang_to_i)
        all_group_mats["ALL"][l] = mean_mat
        _record_matrix_rows(rows_matrix, "ALL", l, mean_mat, cnt_mat, langs_present)

        diag_mean, off_mean, gap = _crosslingual_index(mean_mat)
        rows_index.append(
            {
                "group": "ALL",
                "layer": int(l),
                "diag_mean": diag_mean,
                "offdiag_mean": off_mean,
                "diag_minus_offdiag": gap,
            }
        )

    # ── Per-family matrices ────────────────────────────────────────────────
    for fam in FAMILY_ORDER:
        fam_mask = families == fam
        fac_f = factors[fam_mask]
        lan_f = langs_all[fam_mask]

        for l in layers:
            Xn = _normalize(acts[fam_mask, l, :].astype(np.float32))
            mean_mat, cnt_mat = _matrix_for_subset(Xn, fac_f, lan_f, lang_to_i)
            family_group_mats[fam][l] = mean_mat
            _record_matrix_rows(rows_matrix, fam, l, mean_mat, cnt_mat, langs_present)

            diag_mean, off_mean, gap = _crosslingual_index(mean_mat)
            rows_index.append(
                {
                    "group": fam,
                    "layer": int(l),
                    "diag_mean": diag_mean,
                    "offdiag_mean": off_mean,
                    "diag_minus_offdiag": gap,
                }
            )

    df_matrix = pd.DataFrame(rows_matrix)
    df_matrix.to_csv(RESULTS / "G2_language_matrix_by_family.csv", index=False)

    df_index = pd.DataFrame(rows_index)
    df_index.to_csv(RESULTS / "G2_crosslingual_index_by_family.csv", index=False)

    # Keep a dedicated ALL-families output compatible with old G naming style
    df_all = df_matrix[df_matrix["group"] == "ALL"].copy()
    df_all.to_csv(RESULTS / "G2_language_matrix_all_families.csv", index=False)

    # ── Plot 1: ALL-families matrix at selected layers ─────────────────────
    mats_all = all_group_mats["ALL"]
    vals = np.concatenate([m.ravel() for m in mats_all.values()])
    vmin = max(-1.0, float(np.nanmin(vals)) - 0.02)
    vmax = min(1.0, float(np.nanmax(vals)) + 0.02)

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4))
    if len(layers) == 1:
        axes = [axes]

    for ax, l in zip(axes, layers):
        mat = mats_all[l]
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(langs_present)))
        ax.set_yticks(range(len(langs_present)))
        ax.set_xticklabels(disp_present)
        ax.set_yticklabels(disp_present)
        ax.set_title(f"Layer {l}")

        mid = (vmin + vmax) / 2.0
        for i in range(len(langs_present)):
            for j in range(len(langs_present)):
                v = mat[i, j]
                txt = "—" if np.isnan(v) else f"{v:.3f}"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if (not np.isnan(v) and v < mid) else "black",
                )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("G2. Language-pair cosine similarity (ALL families)", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS / "G2_language_matrix_all_families.png", dpi=150)
    plt.close()

    # ── Plot 2: Per-family matrix grid (rows=families, cols=layers) ───────
    n_rows = len(FAMILY_ORDER)
    n_cols = len(layers)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.7 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    vals_f = np.concatenate([family_group_mats[f][l].ravel() for f in FAMILY_ORDER for l in layers])
    vmin_f = max(-1.0, float(np.nanmin(vals_f)) - 0.02)
    vmax_f = min(1.0, float(np.nanmax(vals_f)) + 0.02)
    mid_f = (vmin_f + vmax_f) / 2.0

    for ri, fam in enumerate(FAMILY_ORDER):
        for ci, l in enumerate(layers):
            ax = axes[ri, ci] if n_cols > 1 else axes[ri, 0]
            mat = family_group_mats[fam][l]
            im = ax.imshow(mat, vmin=vmin_f, vmax=vmax_f, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(langs_present)))
            ax.set_yticks(range(len(langs_present)))
            ax.set_xticklabels(disp_present, fontsize=8)
            ax.set_yticklabels(disp_present, fontsize=8)

            if ri == 0:
                ax.set_title(f"Layer {l}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(fam, fontsize=10, fontweight="bold")

            for i in range(len(langs_present)):
                for j in range(len(langs_present)):
                    v = mat[i, j]
                    txt = "—" if np.isnan(v) else f"{v:.2f}"
                    ax.text(
                        j,
                        i,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if (not np.isnan(v) and v < mid_f) else "black",
                    )

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("G2. Language-pair cosine similarity by surface family", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS / "G2_language_matrix_by_family.png", dpi=150)
    plt.close()

    # ── Plot 3: Original layerwise cross-lingual index curves (raw) ───────
    df_curve = _compute_layerwise_curve(
        acts=acts,
        factors=factors,
        families=families,
        langs_all=langs_all,
        lang_to_i=lang_to_i,
        transform_fn=lambda X, _l: X,
    )
    df_curve.to_csv(RESULTS / "G2_crosslingual_index_by_family_full_layers.csv", index=False)
    _plot_layerwise_curve(
        df_curve,
        RESULTS / "G2_crosslingual_index_by_family.png",
        "G2. Layerwise cross-lingual convergence indices by family (raw)",
    )
    # Keep existing key-layer CSV name intact (0,16,32)
    df_curve[df_curve["layer"].isin(layers)].to_csv(
        RESULTS / "G2_crosslingual_index_by_family.csv", index=False
    )

    # ── Plot 4: Layerwise indices with language residualization ────────────
    df_curve_lang = _compute_layerwise_curve(
        acts=acts,
        factors=factors,
        families=families,
        langs_all=langs_all,
        lang_to_i=lang_to_i,
        transform_fn=lambda X, _l: _residualize(X, Z_lang_all),
    )
    df_curve_lang.to_csv(
        RESULTS / "G2_crosslingual_index_by_family_lang_resid_full_layers.csv", index=False
    )
    df_curve_lang[df_curve_lang["layer"].isin(layers)].to_csv(
        RESULTS / "G2_crosslingual_index_by_family_lang_resid.csv", index=False
    )
    _plot_layerwise_curve(
        df_curve_lang,
        RESULTS / "G2_crosslingual_index_by_family_lang_resid.png",
        "G2. Layerwise cross-lingual convergence indices by family (language residualized)",
    )

    # ── Plot 5: Layerwise indices with top-5 dominant-direction deflation ──
    df_curve_defl = _compute_layerwise_curve(
        acts=acts,
        factors=factors,
        families=families,
        langs_all=langs_all,
        lang_to_i=lang_to_i,
        transform_fn=lambda X, _l: _topk_deflate(X, k=5),
    )
    df_curve_defl.to_csv(
        RESULTS / "G2_crosslingual_index_by_family_top5_deflation_full_layers.csv",
        index=False,
    )
    df_curve_defl[df_curve_defl["layer"].isin(layers)].to_csv(
        RESULTS / "G2_crosslingual_index_by_family_top5_deflation.csv", index=False
    )
    _plot_layerwise_curve(
        df_curve_defl,
        RESULTS / "G2_crosslingual_index_by_family_top5_deflation.png",
        "G2. Layerwise cross-lingual convergence indices by family (top-5 deflation)",
    )

    # ── Plot 6: Layerwise indices with top-3 dominant-direction deflation ──
    df_curve_defl3 = _compute_layerwise_curve(
        acts=acts,
        factors=factors,
        families=families,
        langs_all=langs_all,
        lang_to_i=lang_to_i,
        transform_fn=lambda X, _l: _topk_deflate(X, k=3),
    )
    df_curve_defl3.to_csv(
        RESULTS / "G2_crosslingual_index_by_family_top3_deflation_full_layers.csv",
        index=False,
    )
    df_curve_defl3[df_curve_defl3["layer"].isin(layers)].to_csv(
        RESULTS / "G2_crosslingual_index_by_family_top3_deflation.csv", index=False
    )
    _plot_layerwise_curve(
        df_curve_defl3,
        RESULTS / "G2_crosslingual_index_by_family_top3_deflation.png",
        "G2. Layerwise cross-lingual convergence indices by family (top-3 deflation)",
    )

    # ── Exact-translation analysis and alignment check ─────────────────────
    df_exact_meta = _build_exact_translation_metadata(
        sentence_ids=sentence_ids,
        factors=factors,
        families=families,
        langs=langs_all,
    )

    df_align = _exact_alignment_summary(df_exact_meta, langs_present)
    df_align.to_csv(RESULTS / "G2_exact_translation_alignment.csv", index=False)
    _plot_exact_alignment(
        df_align, langs_present, RESULTS / "G2_exact_translation_alignment.png"
    )

    idx_exact = df_exact_meta["row_idx"].to_numpy(dtype=int)
    keys_exact = df_exact_meta["exact_key"].to_numpy(dtype=str)
    langs_exact = df_exact_meta["language"].to_numpy(dtype=str)
    exact_key_full = np.full(len(factors), "", dtype=object)
    exact_key_full[idx_exact] = keys_exact

    df_exact_matrix, df_cmp = _compute_exact_outputs(
        acts=acts,
        factors=factors,
        langs_all=langs_all,
        idx_exact=idx_exact,
        keys_exact=keys_exact,
        langs_exact=langs_exact,
        lang_to_i=lang_to_i,
        langs_present=langs_present,
        disp_present=disp_present,
        layers=layers,
        transform_fn=lambda X, _l: X,
        matrix_csv=RESULTS / "G2_exact_translation_matrix.csv",
        matrix_png=RESULTS / "G2_exact_translation_matrix.png",
        compare_csv=RESULTS / "G2_exact_vs_all_layerwise.csv",
        compare_png=RESULTS / "G2_exact_vs_all_layerwise.png",
        title_prefix="G2. Exact-translation",
    )

    df_exact_matrix_lang, df_cmp_lang = _compute_exact_outputs(
        acts=acts,
        factors=factors,
        langs_all=langs_all,
        idx_exact=idx_exact,
        keys_exact=keys_exact,
        langs_exact=langs_exact,
        lang_to_i=lang_to_i,
        langs_present=langs_present,
        disp_present=disp_present,
        layers=layers,
        transform_fn=lambda X, _l: _residualize(X, Z_lang_all),
        matrix_csv=RESULTS / "G2_exact_translation_matrix_lang_resid.csv",
        matrix_png=RESULTS / "G2_exact_translation_matrix_lang_resid.png",
        compare_csv=RESULTS / "G2_exact_vs_all_layerwise_lang_resid.csv",
        compare_png=RESULTS / "G2_exact_vs_all_layerwise_lang_resid.png",
        title_prefix="G2. Exact-translation (language residualized)",
    )

    df_exact_matrix_top5, df_cmp_top5 = _compute_exact_outputs(
        acts=acts,
        factors=factors,
        langs_all=langs_all,
        idx_exact=idx_exact,
        keys_exact=keys_exact,
        langs_exact=langs_exact,
        lang_to_i=lang_to_i,
        langs_present=langs_present,
        disp_present=disp_present,
        layers=layers,
        transform_fn=lambda X, _l: _topk_deflate(X, k=5),
        matrix_csv=RESULTS / "G2_exact_translation_matrix_top5_deflation.csv",
        matrix_png=RESULTS / "G2_exact_translation_matrix_top5_deflation.png",
        compare_csv=RESULTS / "G2_exact_vs_all_layerwise_top5_deflation.csv",
        compare_png=RESULTS / "G2_exact_vs_all_layerwise_top5_deflation.png",
        title_prefix="G2. Exact-translation (top-5 deflation)",
    )

    # Pair-category focus after global top-k deflation
    df_focus_top5 = _compute_deflation_focus_curve(
        acts=acts,
        factors=factors,
        domains=domains,
        families=families,
        langs_all=langs_all,
        exact_key_full=exact_key_full,
        transform_fn=lambda X, _l: _topk_deflate(X, k=5),
    )
    df_focus_top5.to_csv(RESULTS / "G2_deflation_focus_top5.csv", index=False)
    _plot_deflation_focus(
        df_focus_top5,
        RESULTS / "G2_deflation_focus_top5.png",
        "G2. Top-5 deflation across all sentences, then pair-category focus",
    )

    df_focus_lang = _compute_deflation_focus_curve(
        acts=acts,
        factors=factors,
        domains=domains,
        families=families,
        langs_all=langs_all,
        exact_key_full=exact_key_full,
        transform_fn=lambda X, _l: _residualize(X, Z_lang_all),
    )
    df_focus_lang.to_csv(RESULTS / "G2_deflation_focus_lang_resid.csv", index=False)
    _plot_deflation_focus(
        df_focus_lang,
        RESULTS / "G2_deflation_focus_lang_resid.png",
        "G2. Language residualization across all sentences, then pair-category focus",
    )

    df_focus_top3 = _compute_deflation_focus_curve(
        acts=acts,
        factors=factors,
        domains=domains,
        families=families,
        langs_all=langs_all,
        exact_key_full=exact_key_full,
        transform_fn=lambda X, _l: _topk_deflate(X, k=3),
    )
    df_focus_top3.to_csv(RESULTS / "G2_deflation_focus_top3.csv", index=False)
    _plot_deflation_focus(
        df_focus_top3,
        RESULTS / "G2_deflation_focus_top3.png",
        "G2. Top-3 deflation across all sentences, then pair-category focus",
    )
    _plot_deflation_focus_comparison(
        df_focus_lang,
        df_focus_top5,
        df_focus_top3,
        RESULTS / "G2_deflation_focus_comparison.png",
    )

    # ── Markdown summary ───────────────────────────────────────────────────
    lines = [
        "# Analysis G2 — Language-Pair Similarity Across All Families",
        "",
        "This is an extension of Analysis G using the expanded multilingual dataset (F1–F5).",
        "",
        "## Dataset used",
        "",
        "- English base: 288",
        "- Multilingual F1: 192",
        "- Multilingual F2–F5: 768",
        "- Total: 1248 sentences",
        "",
        "## Cross-lingual index at key layers",
        "",
        df_index.sort_values(["group", "layer"]).round(4).to_markdown(index=False),
        "",
        "## Additional layerwise variants (key layers 0/16/32)",
        "",
        "### Language residualized",
        "",
        df_curve_lang[df_curve_lang["layer"].isin(layers)]
        .sort_values(["group", "layer"])
        .round(4)
        .to_markdown(index=False),
        "",
        "### Top-5 dominant-direction deflation",
        "",
        df_curve_defl[df_curve_defl["layer"].isin(layers)]
        .sort_values(["group", "layer"])
        .round(4)
        .to_markdown(index=False),
        "",
        "### Top-3 dominant-direction deflation",
        "",
        df_curve_defl3[df_curve_defl3["layer"].isin(layers)]
        .sort_values(["group", "layer"])
        .round(4)
        .to_markdown(index=False),
        "",
        "## Exact translation matching check",
        "",
        df_align.sort_values(["family", "language"]).round(4).to_markdown(index=False),
        "",
        "## Exact-translation comparison variants (ALL families, key layers)",
        "",
        "### Raw",
        "",
        df_cmp[df_cmp["layer"].isin(layers)].round(4).to_markdown(index=False),
        "",
        "### Language residualized",
        "",
        df_cmp_lang[df_cmp_lang["layer"].isin(layers)].round(4).to_markdown(index=False),
        "",
        "### Top-5 dominant-direction deflation",
        "",
        df_cmp_top5[df_cmp_top5["layer"].isin(layers)].round(4).to_markdown(index=False),
        "",
        "## Deflation-first pair-category focus (key layers 0/16/32)",
        "",
        "### Language residualization across all sentences",
        "",
        df_focus_lang[df_focus_lang["layer"].isin(layers)]
        .sort_values(["category", "layer"])
        .round(4)
        .to_markdown(index=False),
        "",
        "### Top-5 deflation across all sentences",
        "",
        df_focus_top5[df_focus_top5["layer"].isin(layers)]
        .sort_values(["category", "layer"])
        .round(4)
        .to_markdown(index=False),
        "",
        "### Top-3 deflation across all sentences",
        "",
        df_focus_top3[df_focus_top3["layer"].isin(layers)]
        .sort_values(["category", "layer"])
        .round(4)
        .to_markdown(index=False),
        "",
        "## Key numeric takeaway (ALL families)",
    ]

    all_idx = df_index[df_index["group"] == "ALL"].set_index("layer")
    for l in layers:
        row = all_idx.loc[l]
        lines.append(
            f"- Layer {l}: diag={row['diag_mean']:.4f}, offdiag={row['offdiag_mean']:.4f}, gap={row['diag_minus_offdiag']:.4f}"
        )

    (RESULTS / "G2_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("[G2] Done. Outputs:")
    print("  results/G2_language_matrix_all_families.csv")
    print("  results/G2_language_matrix_all_families.png")
    print("  results/G2_language_matrix_by_family.csv")
    print("  results/G2_language_matrix_by_family.png")
    print("  results/G2_crosslingual_index_by_family.csv")
    print("  results/G2_crosslingual_index_by_family_full_layers.csv")
    print("  results/G2_crosslingual_index_by_family.png")
    print("  results/G2_crosslingual_index_by_family_lang_resid.csv")
    print("  results/G2_crosslingual_index_by_family_lang_resid_full_layers.csv")
    print("  results/G2_crosslingual_index_by_family_lang_resid.png")
    print("  results/G2_crosslingual_index_by_family_top5_deflation.csv")
    print("  results/G2_crosslingual_index_by_family_top5_deflation_full_layers.csv")
    print("  results/G2_crosslingual_index_by_family_top5_deflation.png")
    print("  results/G2_crosslingual_index_by_family_top3_deflation.csv")
    print("  results/G2_crosslingual_index_by_family_top3_deflation_full_layers.csv")
    print("  results/G2_crosslingual_index_by_family_top3_deflation.png")
    print("  results/G2_exact_translation_alignment.csv")
    print("  results/G2_exact_translation_alignment.png")
    print("  results/G2_exact_translation_matrix.csv")
    print("  results/G2_exact_translation_matrix.png")
    print("  results/G2_exact_translation_matrix_lang_resid.csv")
    print("  results/G2_exact_translation_matrix_lang_resid.png")
    print("  results/G2_exact_translation_matrix_top5_deflation.csv")
    print("  results/G2_exact_translation_matrix_top5_deflation.png")
    print("  results/G2_exact_vs_all_layerwise.csv")
    print("  results/G2_exact_vs_all_layerwise.png")
    print("  results/G2_exact_vs_all_layerwise_lang_resid.csv")
    print("  results/G2_exact_vs_all_layerwise_lang_resid.png")
    print("  results/G2_exact_vs_all_layerwise_top5_deflation.csv")
    print("  results/G2_exact_vs_all_layerwise_top5_deflation.png")
    print("  results/G2_deflation_focus_lang_resid.csv")
    print("  results/G2_deflation_focus_lang_resid.png")
    print("  results/G2_deflation_focus_top5.csv")
    print("  results/G2_deflation_focus_top5.png")
    print("  results/G2_deflation_focus_top3.csv")
    print("  results/G2_deflation_focus_top3.png")
    print("  results/G2_deflation_focus_comparison.png")
    print("  results/G2_summary.md")


if __name__ == "__main__":
    run_analysis_g2()
