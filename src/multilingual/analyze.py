"""
analyze.py — Analyses of semantic convergence across Pythia 2.8B layers.

Analysis A  Within-vs-between meaning similarity, COSINE + L2 (§9-A)
Analysis B  Meaning classifier under cross-form generalization (§9-B)
Analysis C  Surface-form classifier — 5-fold stratified CV (§9-C) [fixed]
Analysis D  Shared low-rank semantic subspace (§9-D)
Analysis E  Four-condition residualization test (§9-E):
              (1) raw
              (2) style: residualize by F1-F5 family one-hot
              (3) ling: residualize by actual linguistic features
              (4) lang: residualize by language identity one-hot (en/de/es/ar/zh)
            Reports cosine + L2 gaps for within, same-domain, diff-domain comparisons.
Analysis F  Family-pair cosine similarity matrix (5×5: F1…F5 transitions)
Analysis G  Language-pair cosine similarity matrix (5×5: EN/DE/ES/AR/ZH)

All results are saved as CSV + PNG files in results/
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold

import dataset as ds

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "multilingual"
RESULTS    = PROJECT_ROOT / "results" / "multilingual"
MPL_DIR = RESULTS / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

CACHE_FILE = DATA_DIR / "activations.npz"
RESULTS.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Shared helpers ─────────────────────────────────────────────────────────────

def lr_accuracy(X_train, y_train, X_test, y_test, n_pca=50, max_iter=2000):
    """PCA-reduce then fit logistic regression; return test accuracy."""
    k = min(n_pca, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=k)
    Xtr = pca.fit_transform(X_train)
    Xte = pca.transform(X_test)
    clf = LogisticRegression(max_iter=max_iter, C=1.0, solver="lbfgs", random_state=0)
    clf.fit(Xtr, y_train)
    return accuracy_score(y_test, clf.predict(Xte))


def residualize(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Remove from X the component explained by Z (with intercept) via least-squares.

    X : (N, H)   — activation matrix
    Z : (N, K)   — regressor matrix (style one-hot or linguistic features)
    Returns residuals (N, H).
    """
    # Add explicit intercept column so the regression can remove global mean
    ones = np.ones((Z.shape[0], 1), dtype=Z.dtype)
    Zaug = np.concatenate([ones, Z], axis=1)           # (N, K+1)
    W = np.linalg.lstsq(Zaug, X, rcond=None)[0]       # (K+1, H)
    return X - Zaug @ W                                # (N, H)


def compute_ling_features(sentences: list) -> np.ndarray:
    """Compute a z-scored (N, 15) matrix of surface-linguistic features.

    Captures actual measurable surface variation — distinct from the
    categorical F1-F5 family labels used in the style residualization.

    Features:
      n_tokens, n_chars, avg_token_len, type_token_ratio, n_commas,
      n_semicolons, starts_the, starts_it, starts_what, starts_this,
      starts_as, starts_when_upon, has_that, has_which, has_copula_is_a
    """
    rows = []
    for s in sentences:
        tokens  = s.split()
        n_tok   = len(tokens)
        n_char  = len(s)
        avg_tl  = sum(len(t) for t in tokens) / max(n_tok, 1)
        unique  = len(set(t.lower().rstrip(".,;:") for t in tokens))
        ttr     = unique / max(n_tok, 1)
        n_com   = s.count(",")
        n_semi  = s.count(";")
        sl      = s.lower()
        rows.append([
            n_tok, n_char, avg_tl, ttr,
            n_com, n_semi,
            int(sl.startswith("the ")),
            int(sl.startswith("it ")),
            int(sl.startswith("what ")),
            int(sl.startswith("this ")),
            int(sl.startswith("as ")),
            int(sl.startswith("when ") or sl.startswith("upon ")),
            int(" that " in sl),
            int(" which " in sl),
            int(" is a " in sl or " is an " in sl),
        ])
    F = np.array(rows, dtype=np.float32)
    # Z-score; StandardScaler zeroes out any constant column automatically
    return StandardScaler().fit_transform(F).astype(np.float32)


def _cosine_gap(M: np.ndarray, ii, jj, within_mask):
    """Vectorised cosine within/between gap on upper-triangle pairs."""
    nrm  = np.linalg.norm(M, axis=1, keepdims=True) + 1e-10
    Mn   = M / nrm
    sims = (Mn @ Mn.T)[ii, jj]
    w = sims[within_mask].mean()
    b = sims[~within_mask].mean()
    return w, b, w - b


def _metric_gaps(M, ii, jj, within_mask, same_dom_mask, diff_dom_mask):
    """Returns (cos_within, cos_same, cos_diff, l2_within, l2_same, l2_diff).

    Computes both cosine similarity and L2 distance for all three pair types.
    For cosine: higher within = better (directions align).
    For L2: lower within = better (spatially closer).
    """
    M = M.astype(np.float32)
    # ── Cosine ──────────────────────────────────────────────────────────────
    nrm = np.linalg.norm(M, axis=1, keepdims=True) + 1e-10
    Mn  = M / nrm
    cos_all = (Mn @ Mn.T)[ii, jj]
    cos_w = cos_all[within_mask].mean()
    cos_s = cos_all[same_dom_mask].mean()
    cos_d = cos_all[diff_dom_mask].mean()
    # ── L2 ──────────────────────────────────────────────────────────────────
    sq     = (M * M).sum(axis=1)
    l2_sq  = np.clip(sq[:, None] + sq[None, :] - 2.0 * (M @ M.T), 0.0, None)
    l2_all = np.sqrt(l2_sq[ii, jj])
    l2_w   = l2_all[within_mask].mean()
    l2_s   = l2_all[same_dom_mask].mean()
    l2_d   = l2_all[diff_dom_mask].mean()
    return cos_w, cos_s, cos_d, l2_w, l2_s, l2_d


# ── Load cache ─────────────────────────────────────────────────────────────────

def load_cache():
    print(f"[analyze] Loading activations from {CACHE_FILE}")
    data        = np.load(CACHE_FILE, allow_pickle=True)
    acts        = data["activations"]           # (N, L, H)
    df          = ds.build_full_dataframe()     # rows aligned with acts
    factor_ids  = data["factor_ids"]
    domain_ids  = data["domain_ids"]
    families    = data["surface_families"]
    splits      = data["splits"]
    languages   = data["languages"]
    sentences   = df["sentence_text"].tolist()
    print(f"[analyze] Shape: {acts.shape}  →  N={acts.shape[0]}, L={acts.shape[1]}, H={acts.shape[2]}")
    lang_counts = pd.Series(languages).value_counts().to_dict()
    print(f"[analyze] Language breakdown: {lang_counts}")
    return acts, factor_ids, domain_ids, families, splits, languages, df, sentences


# ── Analysis A: Within vs Between — cosine + L2 ───────────────────────────────

def analysis_A(acts, factor_ids, domain_ids):
    print("\n[A] Within-vs-between meaning similarity (cosine direction + L2 spatial) ...")
    t0 = time.time()
    N, L, H = acts.shape

    fac_arr = np.array(factor_ids)
    dom_arr = np.array(domain_ids)

    ii, jj = np.triu_indices(N, k=1)
    within_mask   = fac_arr[ii] == fac_arr[jj]
    same_dom_mask = (~within_mask) & (dom_arr[ii] == dom_arr[jj])
    diff_dom_mask = (~within_mask) & (dom_arr[ii] != dom_arr[jj])

    within_cos  = np.zeros(L);  btwn_same_cos = np.zeros(L);  btwn_diff_cos = np.zeros(L)
    within_l2   = np.zeros(L);  btwn_same_l2  = np.zeros(L);  btwn_diff_l2  = np.zeros(L)

    for l in range(L):
        X   = acts[:, l, :].astype(np.float32)
        nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
        Xn  = X / nrm

        # ── Cosine (direction only) ─────────────────────────────────────────
        cos_pairs = (Xn @ Xn.T)[ii, jj]
        within_cos[l]   = cos_pairs[within_mask].mean()
        btwn_same_cos[l]= cos_pairs[same_dom_mask].mean()
        btwn_diff_cos[l]= cos_pairs[diff_dom_mask].mean()

        # ── L2 (direction + magnitude; captures spatial clustering) ─────────
        # ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 * xi . xj
        sq = (X * X).sum(axis=1)
        l2_sq = np.clip(sq[:, None] + sq[None, :] - 2 * (X @ X.T), 0, None)
        l2    = np.sqrt(l2_sq[ii, jj])
        within_l2[l]   = l2[within_mask].mean()
        btwn_same_l2[l]= l2[same_dom_mask].mean()
        btwn_diff_l2[l]= l2[diff_dom_mask].mean()

        if l % 8 == 0:
            cos_gap = within_cos[l] - btwn_same_cos[l]
            l2_ratio = within_l2[l] / (btwn_same_l2[l] + 1e-10)
            print(f"  layer {l:2d}/{L-1}  cos_gap={cos_gap:.4f}  "
                  f"l2_ratio={l2_ratio:.3f}  "
                  f"(within_l2={within_l2[l]:.1f}  btwn_l2={btwn_same_l2[l]:.1f})")

    df_out = pd.DataFrame({
        "layer":               range(L),
        "within_cos":          within_cos,
        "btwn_same_cos":       btwn_same_cos,
        "btwn_diff_cos":       btwn_diff_cos,
        "cos_gap":             within_cos - btwn_same_cos,
        "within_l2":           within_l2,
        "btwn_same_l2":        btwn_same_l2,
        "btwn_diff_l2":        btwn_diff_l2,
        # Ratio < 1 → same-meaning sentences are spatially closer
        "l2_ratio_within_same":within_l2 / (btwn_same_l2 + 1e-10),
        # Keep backward-compatible names
        "within_factor":            within_cos,
        "between_same_domain":      btwn_same_cos,
        "between_diff_domain":      btwn_diff_cos,
        "within_minus_same_domain": within_cos - btwn_same_cos,
    })
    df_out.to_csv(RESULTS / "A_similarity.csv", index=False)

    # 2-row figure
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    layers = np.arange(L)

    ax = axes[0]
    ax.plot(layers, within_cos,    label="Within factor",        marker="o", markersize=3)
    ax.plot(layers, btwn_same_cos, label="Between, same domain", marker="s", markersize=3)
    ax.plot(layers, btwn_diff_cos, label="Between, diff domain", marker="^", markersize=3)
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("A.1 — Cosine similarity by group (direction alignment)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(layers, within_l2,    label="Within factor",        marker="o", markersize=3)
    ax.plot(layers, btwn_same_l2, label="Between, same domain", marker="s", markersize=3)
    ax.plot(layers, btwn_diff_l2, label="Between, diff domain", marker="^", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 distance")
    ax.set_title("A.2 — Euclidean distance by group  (smaller within = same spatial region)")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("A. Within-vs-between meaning similarity per layer", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS / "A_similarity.png", dpi=150)
    plt.close()

    best_layer = int(np.nanargmax(within_cos - btwn_same_cos))
    print(f"  [A] Best layer (cosine gap): {best_layer}")
    print(f"  [A] Done in {time.time()-t0:.1f} s")
    return df_out, best_layer


# ── Analysis B: Meaning classifier (unchanged) ────────────────────────────────

def analysis_B(acts, factor_ids, splits):
    print("\n[B] Meaning classifier under cross-form generalization ...")
    t0 = time.time()
    N, L, H = acts.shape

    fac_arr  = np.array(factor_ids)
    spl_arr  = np.array(splits)
    train_mask = spl_arr == "train"
    val_mask   = spl_arr == "val"
    test_mask  = spl_arr == "test"

    le    = LabelEncoder().fit(fac_arr)
    y_all = le.transform(fac_arr)

    acc_train = np.zeros(L); acc_val = np.zeros(L); acc_test = np.zeros(L)

    for l in range(L):
        X = acts[:, l, :]
        acc_train[l] = lr_accuracy(X[train_mask], y_all[train_mask],
                                   X[train_mask], y_all[train_mask])
        acc_val[l]   = lr_accuracy(X[train_mask], y_all[train_mask],
                                   X[val_mask],   y_all[val_mask])
        acc_test[l]  = lr_accuracy(X[train_mask], y_all[train_mask],
                                   X[test_mask],  y_all[test_mask])
        if l % 8 == 0:
            print(f"  layer {l:2d}/{L-1}  train={acc_train[l]:.3f}  "
                  f"val(F3)={acc_val[l]:.3f}  test(F4+F5)={acc_test[l]:.3f}")

    df_out = pd.DataFrame({"layer": range(L), "train_acc": acc_train,
                           "val_acc_F3": acc_val, "test_acc_F4F5": acc_test})
    df_out.to_csv(RESULTS / "B_classifier.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(L), acc_train, label="Train (F1+F2)", marker="o", markersize=3)
    ax.plot(np.arange(L), acc_val,   label="Val (F3)",      marker="s", markersize=3)
    ax.plot(np.arange(L), acc_test,  label="Test (F4+F5)",  marker="^", markersize=3)
    ax.axhline(1 / len(le.classes_), color="gray", linestyle="--", label="Chance")
    ax.set_xlabel("Layer"); ax.set_ylabel("Accuracy")
    ax.set_title("B. Meaning classifier accuracy per layer (cross-form generalization)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "B_classifier.png", dpi=150)
    plt.close()

    best_layer = int(np.argmax(acc_val))
    print(f"  [B] Best layer by val accuracy: {best_layer}  (val={acc_val[best_layer]:.3f})")
    print(f"  [B] Done in {time.time()-t0:.1f} s")
    return df_out, best_layer


# ── Analysis C: Surface-form classifier — 5-fold stratified CV ───────────────

def analysis_C(acts, families):
    print("\n[C] Surface-form classifier — 5-fold stratified CV (all 5 families in every fold) ...")
    t0 = time.time()
    N, L, H = acts.shape

    fam_arr = np.array(families)
    le      = LabelEncoder()
    y_fam   = le.fit_transform(fam_arr)
    n_fam   = len(le.classes_)
    chance  = 1.0 / n_fam

    # Each fold trains and tests on all 5 families — corrects the original bug
    # (original code trained on F1/F2/F3, tested on F4/F5 → unseen labels → 0% by construction)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = np.zeros(L)

    for l in range(L):
        X = acts[:, l, :]
        fold_accs = []
        for tr_idx, te_idx in skf.split(X, y_fam):
            fold_accs.append(lr_accuracy(
                X[tr_idx], y_fam[tr_idx],
                X[te_idx], y_fam[te_idx],
                n_pca=min(50, len(tr_idx) - 1, H),
            ))
        acc[l] = np.mean(fold_accs)
        if l % 8 == 0:
            print(f"  layer {l:2d}/{L-1}  surface_cv_acc={acc[l]:.3f}  (chance={chance:.3f})")

    df_out = pd.DataFrame({"layer": range(L), "surface_family_acc_cv": acc})
    df_out.to_csv(RESULTS / "C_surface_classifier.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(L), acc, marker="o", markersize=3, label="5-fold CV accuracy")
    ax.axhline(chance, color="gray", linestyle="--", label=f"Chance ({chance:.2f})")
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean CV accuracy")
    ax.set_title("C. Surface-form family classifier accuracy per layer (5-fold stratified CV)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "C_surface_classifier.png", dpi=150)
    plt.close()

    print(f"  [C] Done in {time.time()-t0:.1f} s")
    return df_out


# ── Analysis D: Low-rank semantic subspace ────────────────────────────────────

def analysis_D(acts, factor_ids, best_layer_B):
    print(f"\n[D] Low-rank semantic subspace (best layer from B = {best_layer_B}) ...")
    t0 = time.time()
    N, L, H = acts.shape

    fac_arr = np.array(factor_ids)
    le = LabelEncoder()
    y  = le.fit_transform(fac_arr)

    k_values     = [2, 4, 8, 12, 16, 24, 32, 64, 128]
    results_rows = []

    for l in [best_layer_B, 0, L // 2, L - 1]:
        X   = acts[:, l, :]
        pca = PCA()
        Xp  = pca.fit_transform(X)
        expl = np.cumsum(pca.explained_variance_ratio_)
        print(f"  layer {l}:  dims for 50% var = {np.searchsorted(expl, 0.50)+1}  "
              f"|  80% = {np.searchsorted(expl, 0.80)+1}  "
              f"|  95% = {np.searchsorted(expl, 0.95)+1}")
        for k in k_values:
            if k >= X.shape[0]:
                continue
            sil = silhouette_score(Xp[:, :k], y, metric="cosine")
            results_rows.append({
                "layer": l, "n_components": k,
                "silhouette_cosine": sil,
                "var_explained": expl[k - 1] if k <= len(expl) else np.nan,
            })

    df_out = pd.DataFrame(results_rows)
    df_out.to_csv(RESULTS / "D_subspace.csv", index=False)

    sub = df_out[df_out["layer"] == best_layer_B]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["n_components"], sub["silhouette_cosine"], marker="o")
    ax.axhline(0, color="gray", linestyle="--", label="Zero (no cluster structure)")
    ax.set_xlabel("PCA components retained"); ax.set_ylabel("Silhouette score (cosine)")
    ax.set_title(f"D. Low-rank subspace quality (layer {best_layer_B})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "D_subspace.png", dpi=150)
    plt.close()

    print(f"  [D] Done in {time.time()-t0:.1f} s")
    return df_out


# ── Analysis E: Four-condition residualization ────────────────────────────────

def analysis_E(acts, factor_ids, domain_ids, families, sentences, languages):
    print("\n[E] Residualization test — four conditions:")
    print("    (1) raw  (2) style: remove F-label effect  "
          "(3) ling: remove linguistic-feature effect  "
          "(4) lang: remove language-identity effect")
    t0 = time.time()
    N, L, H = acts.shape

    fac_arr = np.array(factor_ids)
    dom_arr = np.array(domain_ids)
    fam_arr = np.array(families)
    lan_arr = np.array(languages)

    # ── Regressors ──────────────────────────────────────────────────────────
    # (2) style: family one-hot (F1-F5 + ml → encode all present labels)
    fam_le  = LabelEncoder()
    y_fam   = fam_le.fit_transform(fam_arr)
    Z_style = np.eye(len(fam_le.classes_), dtype=np.float32)[y_fam]   # (N, n_fam) one-hot

    # (3) ling: surface linguistic features
    Z_ling  = compute_ling_features(sentences)                          # (N, 15) z-scored

    # (4) lang: language one-hot (en/de/es/ar/zh)
    lang_le = LabelEncoder()
    y_lang  = lang_le.fit_transform(lan_arr)
    Z_lang  = np.eye(len(lang_le.classes_), dtype=np.float32)[y_lang]  # (N, n_lang) one-hot

    # ── Pair masks ──────────────────────────────────────────────────────────
    ii, jj        = np.triu_indices(N, k=1)
    within_mask   = fac_arr[ii] == fac_arr[jj]
    same_dom_mask = (~within_mask) & (dom_arr[ii] == dom_arr[jj])
    diff_dom_mask = (~within_mask) & (dom_arr[ii] != dom_arr[jj])

    # condition names and regressors (None = no residualization)
    conditions = [
        ("raw",   None),
        ("style", Z_style),
        ("ling",  Z_ling),
        ("lang",  Z_lang),
    ]

    results_rows = []

    for l in range(L):
        X_raw = acts[:, l, :].astype(np.float32)
        row = {"layer": l}

        for cond_name, Z in conditions:
            X = residualize(X_raw, Z) if Z is not None else X_raw
            cos_w, cos_s, cos_d, l2_w, l2_s, l2_d = _metric_gaps(
                X, ii, jj, within_mask, same_dom_mask, diff_dom_mask
            )
            # store raw metrics
            row[f"{cond_name}_cos_within"]    = cos_w
            row[f"{cond_name}_cos_same"]      = cos_s
            row[f"{cond_name}_cos_diff"]      = cos_d
            row[f"{cond_name}_l2_within"]     = l2_w
            row[f"{cond_name}_l2_same"]       = l2_s
            row[f"{cond_name}_l2_diff"]       = l2_d
            # derived gap metrics
            row[f"{cond_name}_cos_gap_same"]  = cos_w - cos_s   # positive = within clusters
            row[f"{cond_name}_cos_gap_diff"]  = cos_w - cos_d
            row[f"{cond_name}_l2_gap_same"]   = l2_s - l2_w     # positive = within spatially closer
            row[f"{cond_name}_l2_gap_diff"]   = l2_d - l2_w

        # backward-compatible aliases (raw → _orig, style → _resid/_style)
        row["within_orig"]   = row["raw_cos_within"]
        row["between_orig"]  = row["raw_cos_same"]
        row["gap_orig"]      = row["raw_cos_gap_same"]
        row["within_resid"]  = row["style_cos_within"]
        row["between_resid"] = row["style_cos_same"]
        row["gap_resid"]     = row["style_cos_gap_same"]
        row["within_style"]  = row["style_cos_within"]
        row["between_style"] = row["style_cos_same"]
        row["gap_style"]     = row["style_cos_gap_same"]
        row["within_ling"]   = row["ling_cos_within"]
        row["between_ling"]  = row["ling_cos_same"]
        row["gap_ling"]      = row["ling_cos_gap_same"]

        results_rows.append(row)

        if l % 8 == 0:
            print(f"  layer {l:2d}/{L-1}  "
                  f"cos_gap_same: raw={row['raw_cos_gap_same']:.4f}  "
                  f"style={row['style_cos_gap_same']:.4f}  "
                  f"ling={row['ling_cos_gap_same']:.4f}  "
                  f"lang={row['lang_cos_gap_same']:.4f}")

    df_out = pd.DataFrame(results_rows)
    df_out.to_csv(RESULTS / "E_residualization.csv", index=False)

    layers = df_out["layer"].values

    # ── Plot 1: Cosine gaps (same-domain and diff-domain) for all 4 conditions ──
    cond_labels = {
        "raw":   "(1) Raw",
        "style": "(2) Style residualized",
        "ling":  "(3) Ling. residualized",
        "lang":  "(4) Language residualized",
    }
    colors = ["C0", "C1", "C2", "C3"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    ax = axes[0]
    for (cond, label), col in zip(cond_labels.items(), colors):
        ax.plot(layers, df_out[f"{cond}_cos_gap_same"],
                label=label, color=col, marker="o", markersize=2)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine gap (within − same-domain)")
    ax.set_title("E.1 — Cosine gap: within vs same-domain")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for (cond, label), col in zip(cond_labels.items(), colors):
        ax.plot(layers, df_out[f"{cond}_cos_gap_diff"],
                label=label, color=col, marker="o", markersize=2)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine gap (within − diff-domain)")
    ax.set_title("E.2 — Cosine gap: within vs diff-domain")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("E. Cosine similarity gaps — 4 residualization conditions", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS / "E_cosine_gaps.png", dpi=150)
    plt.close()

    # ── Plot 2: L2 gaps (same-domain and diff-domain) for all 4 conditions ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    ax = axes[0]
    for (cond, label), col in zip(cond_labels.items(), colors):
        ax.plot(layers, df_out[f"{cond}_l2_gap_same"],
                label=label, color=col, marker="o", markersize=2)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 gap (same-domain − within)")
    ax.set_title("E.3 — L2 gap: within vs same-domain")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for (cond, label), col in zip(cond_labels.items(), colors):
        ax.plot(layers, df_out[f"{cond}_l2_gap_diff"],
                label=label, color=col, marker="o", markersize=2)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 gap (diff-domain − within)")
    ax.set_title("E.4 — L2 gap: within vs diff-domain")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("E. L2 distance gaps — 4 residualization conditions", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS / "E_l2_gaps.png", dpi=150)
    plt.close()

    # ── Plot 3: Backward-compatible 3-panel cosine-gap figure ──────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    specs = [
        ("(1) Raw representations",                                      "gap_orig",  "C0"),
        ("(2) Style residualized\n(remove F-family label effect)",       "gap_style", "C1"),
        ("(3) Language residualized\n(remove linguistic feature effect)", "gap_ling",  "C2"),
    ]
    for ax, (title, col, color) in zip(axes, specs):
        ax.plot(layers, df_out[col], marker="o", markersize=3, color=color)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Within − Between cosine similarity (same-domain)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle("E. Residualization test — semantic gap across three conditions", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS / "E_residualization.png", dpi=150)
    plt.close()

    for cond in ["raw", "style", "ling", "lang"]:
        col = f"{cond}_cos_gap_same"
        best_l = df_out.loc[df_out[col].idxmax(), "layer"]
        print(f"  [E] Best layer ({cond:5s}): {best_l}  cos_gap_same={df_out[col].max():.4f}")

    # ── Per-family convergence breakdown (English only, 3 conditions) ────────
    # For each surface family Fi, measure how well its representations converge
    # toward shared meaning across different surface forms.
    # Gap = mean_cosine(same-meaning × diff-family pairs involving Fi)
    #       − mean_cosine(diff-meaning pairs involving Fi)
    # Positive gap → Fi converges to shared semantic representation.
    print("  [E] Computing per-family convergence breakdown (English only) ...")

    en_mask_pf   = lan_arr == "en"
    en_idx_pf    = np.where(en_mask_pf)[0]
    n_en_pf      = len(en_idx_pf)
    fac_en_pf    = fac_arr[en_mask_pf]
    fam_en_pf    = fam_arr[en_mask_pf]
    en_fam_labs  = ["F1", "F2", "F3", "F4", "F5"]
    fam_to_i_pf  = {f: i for i, f in enumerate(en_fam_labs)}
    fam_int_pf   = np.array([fam_to_i_pf[f] for f in fam_en_pf])

    Z_style_pf  = np.eye(len(en_fam_labs), dtype=np.float32)[fam_int_pf]
    sents_pf    = [sentences[i] for i in en_idx_pf]
    Z_ling_pf   = compute_ling_features(sents_pf)

    ii_pf, jj_pf = np.triu_indices(n_en_pf, k=1)
    same_fac_pf  = fac_en_pf[ii_pf] == fac_en_pf[jj_pf]
    diff_fac_pf  = ~same_fac_pf
    diff_fam_pf  = fam_int_pf[ii_pf] != fam_int_pf[jj_pf]

    pf_conditions = [
        ("raw",   None),
        ("style", Z_style_pf),
        ("ling",  Z_ling_pf),
    ]

    per_fam_rows = []
    for l in range(L):
        X_all_pf = acts[:, l, :].astype(np.float32)
        X_en_pf  = X_all_pf[en_idx_pf]
        for cond_name_pf, Z_pf in pf_conditions:
            Xr  = residualize(X_en_pf, Z_pf) if Z_pf is not None else X_en_pf
            nrm = np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-10
            Xn  = Xr / nrm
            cos = (Xn @ Xn.T)[ii_pf, jj_pf]
            for fi, fn in enumerate(en_fam_labs):
                inv_fi = (fam_int_pf[ii_pf] == fi) | (fam_int_pf[jj_pf] == fi)
                cross  = inv_fi & same_fac_pf & diff_fam_pf
                btwn   = inv_fi & diff_fac_pf
                w = float(cos[cross].mean()) if cross.sum() > 0 else np.nan
                b = float(cos[btwn].mean())  if btwn.sum()  > 0 else np.nan
                g = (w - b) if not (np.isnan(w) or np.isnan(b)) else np.nan
                per_fam_rows.append({
                    "layer": l, "condition": cond_name_pf, "family": fn,
                    "within_cross_cos": w, "between_cos": b, "convergence_gap": g,
                })
        if l % 8 == 0:
            print(f"  per-family layer {l}/{L-1}")

    df_fam = pd.DataFrame(per_fam_rows)
    df_fam.to_csv(RESULTS / "E_per_family.csv", index=False)

    pf_cond_titles = {
        "raw":   "(1) Raw representations",
        "style": "(2) Style residualized\n(remove family-label effect)",
        "ling":  "(3) Ling. residualized\n(remove surface-feature effect)",
    }
    fam_colors = ["C0", "C1", "C2", "C3", "C4"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ci, (cond_name_pf, _) in enumerate(pf_conditions):
        ax = axes[ci]
        sub = df_fam[df_fam["condition"] == cond_name_pf]
        for fi, fn in enumerate(en_fam_labs):
            row = sub[sub["family"] == fn].sort_values("layer")
            ax.plot(row["layer"], row["convergence_gap"],
                    label=fn, color=fam_colors[fi], marker="o", markersize=2)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cross-form convergence gap")
        ax.set_title(pf_cond_titles[cond_name_pf])
        ax.legend(fontsize=9, title="Family")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "E. Per-family cross-form convergence  "
        "(same-meaning × cross-form gap − different-meaning baseline)\n"
        "English only  |  positive = that surface form converges to shared meaning",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(RESULTS / "E_per_family.png", dpi=150)
    plt.close()
    print(f"  [E] Per-family breakdown saved → E_per_family.png")

    print(f"  [E] Done in {time.time()-t0:.1f} s")
    return df_out


# ── Analysis F: Family-pair similarity matrix — 3 residualization conditions ──

def analysis_F(acts, factor_ids, families, languages, sentences):
    """5×5 family-pair cosine similarity matrix across 3 residualization conditions.

    Entry (Fi, Fj) = mean cosine similarity of all same-factor pairs where
    one sentence is from surface family Fi and the other from Fj.
    Three conditions:
      (1) raw  (2) style residualized (remove F-label effect)
      (3) ling residualized (remove surface-feature effect)
    Only English sentences (language=="en") are used; multilingual sentences also
    carry surface_family="F1" which would confound the matrix.
    """
    print("\n[F] Family-pair cosine similarity — 3 residualization conditions (English only) ...")
    t0 = time.time()
    N, L, H = acts.shape

    fac_arr  = np.array(factor_ids)
    fam_arr  = np.array(families)
    lang_arr = np.array(languages)

    en_mask = lang_arr == "en"
    en_idx  = np.where(en_mask)[0]
    n_en    = len(en_idx)

    en_fam_labels = ["F1", "F2", "F3", "F4", "F5"]
    n_fam    = len(en_fam_labels)
    fam_to_i = {f: i for i, f in enumerate(en_fam_labels)}

    fac_en  = fac_arr[en_mask]
    fam_en  = fam_arr[en_mask]
    fam_int = np.array([fam_to_i[f] for f in fam_en])

    # English-only regressors
    Z_style_en = np.eye(n_fam, dtype=np.float32)[fam_int]
    sents_en   = [sentences[i] for i in en_idx]
    Z_ling_en  = compute_ling_features(sents_en)

    conditions_F = [
        ("raw",   None),
        ("style", Z_style_en),
        ("ling",  Z_ling_en),
    ]

    check_layers = [l for l in [0, 16, 32] if l < L]

    all_results = {}   # (cond_name, layer) → mat_norm (5×5)
    rows_out    = []

    for cond_name, Z_en in conditions_F:
        for l in check_layers:
            X_en = acts[en_idx, l, :].astype(np.float32)
            if Z_en is not None:
                X_en = residualize(X_en, Z_en)
            nrm     = np.linalg.norm(X_en, axis=1, keepdims=True) + 1e-10
            Xn      = X_en / nrm
            cos_mat = Xn @ Xn.T   # (n_en, n_en)

            mat = np.zeros((n_fam, n_fam))
            cnt = np.zeros((n_fam, n_fam))

            for a in range(n_en):
                for b in range(a + 1, n_en):
                    if fac_en[a] != fac_en[b]:
                        continue
                    fi, fj = fam_int[a], fam_int[b]
                    v = cos_mat[a, b]
                    mat[fi, fj] += v;  mat[fj, fi] += v
                    cnt[fi, fj] += 1;  cnt[fj, fi] += 1

            with np.errstate(invalid="ignore"):
                mat_norm = np.where(cnt > 0, mat / cnt, np.nan)

            all_results[(cond_name, l)] = mat_norm
            for fi, fn_i in enumerate(en_fam_labels):
                for fj, fn_j in enumerate(en_fam_labels):
                    rows_out.append({
                        "condition": cond_name, "layer": l,
                        "family_i": fn_i, "family_j": fn_j,
                        "mean_cosine": mat_norm[fi, fj],
                    })

        # Print summary for raw condition
        if cond_name == "raw":
            for l in check_layers:
                m = all_results[("raw", l)]
                print(f"  layer {l:2d} [raw]  F1-F1={m[0,0]:.4f}  "
                      f"F1-F5={m[0,4]:.4f}  F1-F2={m[0,1]:.4f}")

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(RESULTS / "F_family_matrix.csv", index=False)

    # ── Plot: 3 conditions (rows) × 3 layers (cols) ────────────────────────
    cond_row_labels = {
        "raw":   "(1) Raw",
        "style": "(2) Style\nresidualised",
        "ling":  "(3) Ling.\nresidualised",
    }
    n_cond = len(conditions_F)
    n_cols  = len(check_layers)

    fig, axes = plt.subplots(n_cond, n_cols, figsize=(5.2 * n_cols, 4.2 * n_cond))
    if n_cond == 1:
        axes = axes[np.newaxis, :]

    for ci, (cond_name, _) in enumerate(conditions_F):
        for lj, l in enumerate(check_layers):
            ax = axes[ci, lj]
            mat = all_results[(cond_name, l)]
            im  = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
            ax.set_xticks(range(n_fam)); ax.set_xticklabels(en_fam_labels, fontsize=8)
            ax.set_yticks(range(n_fam)); ax.set_yticklabels(en_fam_labels, fontsize=8)
            if ci == 0:
                ax.set_title(f"Layer {l}", fontsize=10)
            if lj == 0:
                ax.set_ylabel(cond_row_labels[cond_name], fontsize=9)
            for fi in range(n_fam):
                for fj in range(n_fam):
                    v = mat[fi, fj]
                    if not np.isnan(v):
                        ax.text(fj, fi, f"{v:.3f}", ha="center", va="center",
                                fontsize=6.5,
                                color="white" if v < 0.75 else "black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "F. Family-pair cosine similarity  "
        "(rows = residualization condition, cols = layer)\n"
        "Diagonal = within-family same-meaning  |  Off-diagonal = cross-family same-meaning",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(RESULTS / "F_family_matrix.png", dpi=150)
    plt.close()

    print(f"  [F] Done in {time.time()-t0:.1f} s")
    return df_out


# ── Analysis G: Language-pair cosine similarity matrix ───────────────────────

def analysis_G(acts, factor_ids, languages):
    """Compute a 5×5 matrix of mean cosine similarity for all (li, lj) language pairs.

    Entry (li, lj) = mean cosine similarity of all same-factor pairs where
    one sentence has language li and the other has language lj.
    Diagonal = within-language same-meaning similarity.
    Off-diagonal = cross-language same-meaning similarity.
    """
    print("\n[G] Language-pair cosine similarity matrix ...")
    t0 = time.time()
    N, L, H = acts.shape

    fac_arr  = np.array(factor_ids)
    lang_arr = np.array(languages)

    lang_labels = ["en", "de", "es", "ar", "zh"]
    lang_display = ["EN", "DE", "ES", "AR", "ZH"]
    # Only include languages that are actually present
    present_langs = [l for l in lang_labels if l in lang_arr]
    present_disp  = [lang_display[lang_labels.index(l)] for l in present_langs]
    lang_to_i = {l: i for i, l in enumerate(present_langs)}
    n_lang = len(present_langs)

    lang_int = np.array([lang_to_i.get(l, -1) for l in lang_arr])
    valid_mask = lang_int >= 0

    check_layers = [0, 16, 32]
    check_layers = [l for l in check_layers if l < L]

    all_matrices = {}
    rows_out = []

    for l in check_layers:
        X_all = acts[:, l, :].astype(np.float32)
        X_v   = X_all[valid_mask]
        nrm   = np.linalg.norm(X_v, axis=1, keepdims=True) + 1e-10
        Xn    = X_v / nrm

        fac_v  = fac_arr[valid_mask]
        lang_v = lang_int[valid_mask]

        mat = np.zeros((n_lang, n_lang))
        cnt = np.zeros((n_lang, n_lang))

        n_v = len(fac_v)
        cos_mat = Xn @ Xn.T   # (n_v, n_v)

        for a in range(n_v):
            for b in range(a + 1, n_v):
                if fac_v[a] != fac_v[b]:
                    continue
                li = lang_v[a]
                lj = lang_v[b]
                v  = cos_mat[a, b]
                mat[li, lj] += v
                mat[lj, li] += v
                cnt[li, lj] += 1
                cnt[lj, li] += 1

        # Diagonal: within-language same-factor pairs
        for a in range(n_v):
            for b in range(a + 1, n_v):
                if fac_v[a] != fac_v[b]:
                    continue
                if lang_v[a] != lang_v[b]:
                    continue
                li = lang_v[a]
                # Already accumulated above (li == lj case)

        with np.errstate(invalid="ignore"):
            mat_norm = np.where(cnt > 0, mat / cnt, np.nan)

        # For diagonal (li == lj), we need within-language pairs
        # Re-compute diagonal explicitly to avoid double-counting
        for li in range(n_lang):
            idx_li = np.where(lang_v == li)[0]
            pairs_val = []
            for p in range(len(idx_li)):
                for q in range(p + 1, len(idx_li)):
                    a, b = idx_li[p], idx_li[q]
                    if fac_v[a] == fac_v[b]:
                        pairs_val.append(cos_mat[a, b])
            mat_norm[li, li] = np.mean(pairs_val) if pairs_val else np.nan

        all_matrices[l] = mat_norm
        for li, ln_i in enumerate(present_langs):
            for lj, ln_j in enumerate(present_langs):
                rows_out.append({
                    "layer": l,
                    "lang_i": ln_i,
                    "lang_j": ln_j,
                    "mean_cosine": mat_norm[li, lj],
                })
        print(f"  layer {l:2d}  EN-EN={mat_norm[0,0]:.4f}  EN-DE={mat_norm[0,1]:.4f}  "
              f"EN-ZH={mat_norm[0,4]:.4f}" if n_lang >= 5 else
              f"  layer {l:2d}  {mat_norm}")

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(RESULTS / "G_language_matrix.csv", index=False)

    # ── Plot: 3 heatmaps side by side ──────────────────────────────────────
    # Determine colorbar range from data
    all_vals = np.concatenate([m.ravel() for m in all_matrices.values()])
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    # Give a bit of margin and ensure range is not degenerate
    vmin = max(0.0, vmin - 0.02)
    vmax = min(1.0, vmax + 0.02)

    fig, axes = plt.subplots(1, len(check_layers), figsize=(5 * len(check_layers), 4))
    if len(check_layers) == 1:
        axes = [axes]

    for ax, l in zip(axes, check_layers):
        mat = all_matrices[l]
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_xticks(range(n_lang)); ax.set_xticklabels(present_disp)
        ax.set_yticks(range(n_lang)); ax.set_yticklabels(present_disp)
        ax.set_title(f"Layer {l}")
        ax.set_xlabel("Language j")
        ax.set_ylabel("Language i")
        for li in range(n_lang):
            for lj in range(n_lang):
                v = mat[li, lj]
                txt = f"{v:.3f}" if not np.isnan(v) else "—"
                mid = (vmin + vmax) / 2
                ax.text(lj, li, txt, ha="center", va="center",
                        fontsize=7, color="white" if v < mid else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("G. Language-pair cosine similarity matrix (same-factor pairs)", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS / "G_language_matrix.png", dpi=150)
    plt.close()

    print(f"  [G] Done in {time.time()-t0:.1f} s")
    return df_out


# ── Analysis H: 15×15 family × pair-type block-diagonal heatmap ───────────────

def analysis_H(acts, factor_ids, domain_ids, families, languages):
    """15×15 block-diagonal heatmap: 5 families × 3 pair types, after language residualization.

    Rows/cols (15 total):
      0–4:   F1-W … F5-W   (within-factor  = same meaning)
      5–9:   F1-S … F5-S   (same-domain, diff-factor = confusable)
      10–14: F1-D … F5-D   (different-domain = unrelated)

    Entry (block_pt × fi, block_pt × fj): mean cosine or mean L2 of English pairs
    of pair-type pt where one sentence is from family Fi and the other from Fj.
    Off-diagonal 5×5 blocks = NaN (gray).

    Language residualization applied to all 480 sentences before selecting English subset.
    Plot: 2 rows (cosine, L2) × 3 cols (layer 0, 16, 32).
    """
    print("\n[H] 15×15 family×pair-type heatmap (language-residualised, English) ...")
    t0 = time.time()
    N, L, H = acts.shape

    fac_arr  = np.array(factor_ids)
    dom_arr  = np.array(domain_ids)
    fam_arr  = np.array(families)
    lang_arr = np.array(languages)

    present_langs = sorted(set(languages))
    lang_to_i     = {ln: i for i, ln in enumerate(present_langs)}
    n_lang        = len(present_langs)
    lang_int_all  = np.array([lang_to_i[ln] for ln in lang_arr])
    Z_lang_all    = np.eye(n_lang, dtype=np.float32)[lang_int_all]

    en_mask = lang_arr == "en"
    en_idx  = np.where(en_mask)[0]
    n_en    = len(en_idx)

    en_fam_labels = ["F1", "F2", "F3", "F4", "F5"]
    n_fam    = len(en_fam_labels)
    fam_to_i = {f: i for i, f in enumerate(en_fam_labels)}

    fac_en  = fac_arr[en_idx]
    dom_en  = dom_arr[en_idx]
    fam_en  = fam_arr[en_idx]
    fam_int = np.array([fam_to_i[f] for f in fam_en])

    ii_en, jj_en = np.triu_indices(n_en, k=1)
    same_fac  = fac_en[ii_en] == fac_en[jj_en]
    same_dom  = dom_en[ii_en]  == dom_en[jj_en]
    pair_types = [
        ("W", "Within-factor\n(same meaning)", same_fac),
        ("S", "Same-domain\n(confusable)",      (~same_fac) & same_dom),
        ("D", "Diff-domain\n(unrelated)",        (~same_fac) & (~same_dom)),
    ]
    n_pt  = len(pair_types)
    SIZE  = n_fam * n_pt
    labels_15 = [f"F{f+1}-{suf}" for suf in ["W", "S", "D"] for f in range(n_fam)]

    check_layers = sorted(set([l for l in [0, 16, 32] if l < L]))

    cos_mats = {}
    l2_mats  = {}
    rows_out = []

    for l in check_layers:
        X_all = acts[:, l, :].astype(np.float32)
        X_res = residualize(X_all, Z_lang_all)
        X_en  = X_res[en_idx]

        nrm      = np.linalg.norm(X_en, axis=1, keepdims=True) + 1e-10
        Xn       = X_en / nrm
        cos_pairs = (Xn @ Xn.T)[ii_en, jj_en]

        sq       = (X_en * X_en).sum(axis=1)
        l2_sq    = np.clip(sq[:, None] + sq[None, :] - 2.0 * (X_en @ X_en.T), 0.0, None)
        l2_pairs = np.sqrt(l2_sq[ii_en, jj_en])

        cos_mat15 = np.full((SIZE, SIZE), np.nan)
        l2_mat15  = np.full((SIZE, SIZE), np.nan)

        for pt_idx, (pt_label, _, pt_mask) in enumerate(pair_types):
            row_off = pt_idx * n_fam
            acc_cos = np.zeros((n_fam, n_fam))
            acc_l2  = np.zeros((n_fam, n_fam))
            cnt     = np.zeros((n_fam, n_fam))
            fi_arr  = fam_int[ii_en[pt_mask]]
            fj_arr  = fam_int[jj_en[pt_mask]]
            np.add.at(acc_cos, (fi_arr, fj_arr), cos_pairs[pt_mask])
            np.add.at(acc_cos, (fj_arr, fi_arr), cos_pairs[pt_mask])
            np.add.at(acc_l2,  (fi_arr, fj_arr), l2_pairs[pt_mask])
            np.add.at(acc_l2,  (fj_arr, fi_arr), l2_pairs[pt_mask])
            np.add.at(cnt,     (fi_arr, fj_arr), 1)
            np.add.at(cnt,     (fj_arr, fi_arr), 1)
            with np.errstate(invalid="ignore"):
                blk_cos = np.where(cnt > 0, acc_cos / cnt, np.nan)
                blk_l2  = np.where(cnt > 0, acc_l2  / cnt, np.nan)
            cos_mat15[row_off:row_off+n_fam, row_off:row_off+n_fam] = blk_cos
            l2_mat15 [row_off:row_off+n_fam, row_off:row_off+n_fam] = blk_l2
            for fi, fn_i in enumerate(en_fam_labels):
                for fj, fn_j in enumerate(en_fam_labels):
                    rows_out.append({"layer": l, "pair_type": pt_label,
                                     "family_i": fn_i, "family_j": fn_j,
                                     "mean_cosine": blk_cos[fi, fj],
                                     "mean_l2":     blk_l2[fi, fj]})

        cos_mats[l] = cos_mat15
        l2_mats[l]  = l2_mat15
        w = np.nanmean([cos_mat15[f, f] for f in range(n_fam)])
        s = np.nanmean([cos_mat15[n_fam+f, n_fam+f] for f in range(n_fam)])
        d = np.nanmean([cos_mat15[2*n_fam+f, 2*n_fam+f] for f in range(n_fam)])
        print(f"  layer {l:2d}  cos diag W={w:.4f}  S={s:.4f}  D={d:.4f}")

    pd.DataFrame(rows_out).to_csv(RESULTS / "H_family_pair_structure.csv", index=False)

    n_cols = len(check_layers)
    fig, axes = plt.subplots(2, n_cols, figsize=(6.5 * n_cols, 12))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for ri, (metric_name, mats_dict, vmin_f, vmax_f, cmap_name, cbar_label) in enumerate([
        ("COSINE", cos_mats, 0.0, 1.0, "viridis",   "Mean cosine similarity"),
        ("L2",     l2_mats,  None, None, "viridis_r", "Mean L2 distance"),
    ]):
        all_vals = np.concatenate([m.ravel() for m in mats_dict.values()])
        vmin = vmin_f if vmin_f is not None else np.nanmin(all_vals)
        vmax = vmax_f if vmax_f is not None else np.nanmax(all_vals)
        cmap = plt.cm.get_cmap(cmap_name).copy()
        cmap.set_bad("lightgray")

        for ci, l in enumerate(check_layers):
            ax  = axes[ri, ci]
            mat = mats_dict[l]
            im  = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
            ax.set_xticks(range(SIZE)); ax.set_xticklabels(labels_15, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(SIZE)); ax.set_yticklabels(labels_15, fontsize=7)
            for sep in [n_fam, 2*n_fam]:
                ax.axhline(sep - 0.5, color="black", linewidth=1.5)
                ax.axvline(sep - 0.5, color="black", linewidth=1.5)
            mid_val = (vmin + vmax) / 2
            for row in range(SIZE):
                for col in range(SIZE):
                    v = mat[row, col]
                    if not np.isnan(v):
                        ax.text(col, row, f"{v:.2f}", ha="center", va="center",
                                fontsize=5.5, color="white" if v < mid_val else "black")
            if ri == 0: ax.set_title(f"Layer {l}", fontsize=11, fontweight="bold")
            if ci == 0: ax.set_ylabel(metric_name, fontsize=10, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=cbar_label)

    fig.suptitle(
        "H. 15×15 Family × Pair-type structured similarity (language-residualised, English only)\n"
        "Blocks: W = within-factor  |  S = same-domain confusable  |  D = different-domain\n"
        "Off-diagonal blocks are undefined (gray)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(RESULTS / "H_family_pair_structure.png", dpi=150)
    plt.close()
    print(f"  [H] Done in {time.time()-t0:.1f} s")
    return pd.DataFrame(rows_out)


# ── Analysis I: Meaning classifier with residualization — cross-language train ─

def analysis_I(acts, factor_ids, families, languages, splits, sentences):
    """Meaning classifier: train F1+F2+multilingual, test F3/F4/F5, under residualization.

    Extends Analysis B in two ways:
      1. Training includes ALL languages (en F1+F2 + ar/de/es/zh F1).
      2. Activations are residualized before classification (raw / lang / ling).

    For each condition × layer:
      - Residualize all 480 activations globally.
      - Fit logistic regression on residualized train set.
      - Report accuracy on F3 (val), F4, and F5 (test) separately.

    Plot: 1 row × 3 conditions (raw / lang-resid / ling-resid), each showing
          F3/F4/F5 accuracy curves + chance baseline.
    """
    print("\n[I] Meaning classifier with residualization — train across languages ...")
    t0 = time.time()
    N, L, H = acts.shape

    fac_arr  = np.array(factor_ids)
    fam_arr  = np.array(families)
    lang_arr = np.array(languages)
    spl_arr  = np.array(splits)

    le    = LabelEncoder().fit(fac_arr)
    y_all = le.transform(fac_arr)
    chance = 1.0 / len(le.classes_)

    # Train: F1+F2 English  +  all multilingual F1 equivalents
    train_mask = (spl_arr == "train") | (spl_arr == "ml")
    f3_mask    = spl_arr == "val"
    f4_mask    = (spl_arr == "test") & (fam_arr == "F4")
    f5_mask    = (spl_arr == "test") & (fam_arr == "F5")

    print(f"  Train (F1+F2 en + multilingual F1): {train_mask.sum()}  "
          f"F3={f3_mask.sum()}  F4={f4_mask.sum()}  F5={f5_mask.sum()}")

    # Residualization regressors — fit on all N sentences globally
    present_langs = sorted(set(languages))
    lang_to_i     = {ln: i for i, ln in enumerate(present_langs)}
    n_lang        = len(present_langs)
    Z_lang = np.eye(n_lang, dtype=np.float32)[
        np.array([lang_to_i[ln] for ln in lang_arr])]
    Z_ling = compute_ling_features(sentences)

    conditions = [("raw", None), ("lang", Z_lang), ("ling", Z_ling)]

    rows_out = []
    for cond_name, Z in conditions:
        acc_f3 = np.zeros(L)
        acc_f4 = np.zeros(L)
        acc_f5 = np.zeros(L)

        for l in range(L):
            X_all = acts[:, l, :].astype(np.float32)
            Xr    = residualize(X_all, Z) if Z is not None else X_all

            acc_f3[l] = lr_accuracy(Xr[train_mask], y_all[train_mask],
                                    Xr[f3_mask],    y_all[f3_mask])
            acc_f4[l] = lr_accuracy(Xr[train_mask], y_all[train_mask],
                                    Xr[f4_mask],    y_all[f4_mask])
            acc_f5[l] = lr_accuracy(Xr[train_mask], y_all[train_mask],
                                    Xr[f5_mask],    y_all[f5_mask])

            if l % 8 == 0:
                print(f"  [{cond_name}] layer {l:2d}/{L-1}  "
                      f"F3={acc_f3[l]:.3f}  F4={acc_f4[l]:.3f}  F5={acc_f5[l]:.3f}")

        for l in range(L):
            rows_out.append({"condition": cond_name, "layer": l,
                             "acc_F3": acc_f3[l], "acc_F4": acc_f4[l], "acc_F5": acc_f5[l]})

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(RESULTS / "I_classifier_residualized.csv", index=False)

    # Plot: 1 row × 3 conditions
    cond_titles = {
        "raw":  "(1) Raw activations",
        "lang": "(2) Language residualized",
        "ling": "(3) Linguistic residualized",
    }
    layers = np.arange(L)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = {"acc_F3": "#1f77b4", "acc_F4": "#ff7f0e", "acc_F5": "#2ca02c"}
    labels = {"acc_F3": "F3 — syntactic paraphrase",
              "acc_F4": "F4 — definition-like",
              "acc_F5": "F5 — embedded context"}

    for ax, (cond_name, _) in zip(axes, conditions):
        sub = df_out[df_out.condition == cond_name].sort_values("layer")
        for col in ["acc_F3", "acc_F4", "acc_F5"]:
            vals = sub[col].values
            ax.plot(layers, vals, label=labels[col], color=colors[col],
                    marker="o", markersize=3, linewidth=1.5)
            best_l = int(np.argmax(vals))
            ax.annotate(f"{vals[best_l]:.2f}",
                        xy=(best_l, vals[best_l]),
                        xytext=(best_l + 0.5, vals[best_l] + 0.025),
                        fontsize=7.5, color=colors[col],
                        arrowprops=dict(arrowstyle="-", color=colors[col], lw=0.8))
        ax.axhline(chance, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Chance ({chance:.2f})")
        ax.set_xlabel("Layer"); ax.set_title(cond_titles[cond_name], fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Test accuracy")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle(
        "I. Meaning classifier — train on F1+F2 (all 5 languages), test on F3 / F4 / F5 (English)\n"
        "Each panel shows a different residualization of the activations before fitting",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(RESULTS / "I_classifier_residualized.png", dpi=150)
    plt.close()

    # Summary table
    print(f"\n  Peak accuracy per (condition, surface family):")
    print(f"  {'condition':<6}  {'F3 acc':>7}  {'F3 layer':>8}  "
          f"{'F4 acc':>7}  {'F4 layer':>8}  {'F5 acc':>7}  {'F5 layer':>8}")
    for cond_name, _ in conditions:
        sub = df_out[df_out.condition == cond_name].sort_values("layer")
        l_f3 = int(sub.acc_F3.idxmax()) % L
        l_f4 = int(sub.acc_F4.idxmax()) % L
        l_f5 = int(sub.acc_F5.idxmax()) % L
        print(f"  {cond_name:<6}  {sub.acc_F3.max():>7.3f}  {l_f3:>8d}  "
              f"{sub.acc_F4.max():>7.3f}  {l_f4:>8d}  "
              f"{sub.acc_F5.max():>7.3f}  {l_f5:>8d}")

    print(f"  [I] Done in {time.time()-t0:.1f} s")
    return df_out


# ── Summary table ─────────────────────────────────────────────────────────────

def write_summary(res_A, res_B, res_C, res_D, res_E, best_A, best_B, languages=None):
    print("\n[analyze] Writing summary ...")

    # E summary: cos_gap_same for all 4 conditions at key layers
    key_layers = [0, 8, 16, 24, 32]
    key_layers = [l for l in key_layers if l < len(res_E)]
    e_cols = ["layer", "raw_cos_gap_same", "style_cos_gap_same",
              "ling_cos_gap_same", "lang_cos_gap_same"]
    e_sub = res_E[res_E["layer"].isin(key_layers)][e_cols].round(4)

    lines = [
        "# Experiment Summary — Semantic Convergence in Pythia 2.8B\n",
        f"Best semantic layer by Analysis A (cosine gap):   **{best_A}**\n",
        f"Best semantic layer by Analysis B (val accuracy): **{best_B}**\n",
        "\n## Analysis A — Within vs Between (cosine + L2)\n",
        res_A[["layer", "within_cos", "btwn_same_cos", "cos_gap",
               "within_l2", "btwn_same_l2", "l2_ratio_within_same"]]
            .round(4).to_markdown(index=False),
        "\n## Analysis B — Meaning classifier\n",
        res_B.round(4).to_markdown(index=False),
        "\n## Analysis C — Surface-form classifier (5-fold CV, corrected)\n",
        res_C.round(4).to_markdown(index=False),
        "\n## Analysis E — Residualization (four conditions, cosine gap same-domain)\n",
        e_sub.to_markdown(index=False),
    ]
    with open(RESULTS / "summary.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[analyze] Summary saved to {RESULTS}/summary.md")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_analyses():
    t_start = time.time()
    acts, factor_ids, domain_ids, families, splits, languages, df, sentences = load_cache()

    res_A, best_A = analysis_A(acts, factor_ids, domain_ids)
    res_B, best_B = analysis_B(acts, factor_ids, splits)
    res_C         = analysis_C(acts, families)
    res_D         = analysis_D(acts, factor_ids, best_B)
    res_E         = analysis_E(acts, factor_ids, domain_ids, families, sentences, languages)
    res_F         = analysis_F(acts, factor_ids, families, languages, sentences)
    res_G         = analysis_G(acts, factor_ids, languages)
    res_H         = analysis_H(acts, factor_ids, domain_ids, families, languages)
    res_I         = analysis_I(acts, factor_ids, families, languages, splits, sentences)

    write_summary(res_A, res_B, res_C, res_D, res_E, best_A, best_B, languages)

    total = time.time() - t_start
    print(f"\n[analyze] All analyses complete in {total:.1f} s")
    print(f"[analyze] Results in: {RESULTS.resolve()}/")


if __name__ == "__main__":
    run_analyses()
