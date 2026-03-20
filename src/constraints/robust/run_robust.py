#!/usr/bin/env python3
"""Robust layerwise Pythia analysis with multilingual language calibration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.extmath import randomized_svd


ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "constraints" / "robust"
RESULTS_DIR.mkdir(exist_ok=True)
MPL_DIR = RESULTS_DIR / ".mplcache"
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = ROOT / "data"
REF_DATASET = DATA_DIR / "constraints" / "semantic_refinement_dataset_v2_400.json"
CTRL_DATASET = DATA_DIR / "constraints" / "redundant_commonality_controls_400.json"
REF_ACTS = ROOT / "results" / "constraints" / "base" / "full_pythia" / "acts_full_pythia.npy"
CTRL_ACTS = ROOT / "results" / "constraints" / "base" / "control_pythia" / "acts_control_pythia.npy"
CALIBRATION_ACTS = DATA_DIR / "multilingual" / "activations_all_families.npz"

SURFACE_MAP = {"F1": 0, "F2": 1, "F3": 2, "F4": 3}
CTRL_SURFACE_MAP = {
    0: "F1", 1: "F1",
    2: "F2", 3: "F2",
    4: "F3", 5: "F3",
    6: "F4", 7: "F4",
}

SEM_DIM = 12
SURFACE_DIM = 3
LANG_DIM = 4
DEFLATION_K = [3, 5]
ALIGNMENT_K = [1, 3, 5, 10]
METRICS = ["pr", "mle_id", "log_vol"]
STATIC_BRANCHES = [
    "raw",
    "sem_proj",
    "whitened_sem",
    "surface_resid",
    "lang_resid",
    "surface_lang_resid",
    "deflate_top3",
    "deflate_top5",
]
FLOW_BRANCHES = [
    "raw",
    "surface_resid",
    "lang_resid",
    "surface_lang_resid",
    "deflate_top3",
    "deflate_top5",
]
BRANCH_LABELS = {
    "raw": "Raw",
    "sem_proj": "Sem-proj",
    "whitened_sem": "White-sem",
    "surface_resid": "Surface resid",
    "lang_resid": "Lang resid",
    "surface_lang_resid": "Surface+lang resid",
    "deflate_top3": "Deflate top-3",
    "deflate_top5": "Deflate top-5",
}
COND_COLORS = {
    "refinement": "#1f77b4",
    "control": "#d95f02",
}


def load_refinement_df() -> pd.DataFrame:
    raw = json.load(open(REF_DATASET))
    df = pd.DataFrame(raw["examples"])[
        ["chain_id", "level", "concept", "surface_family", "sentence"]
    ].copy()
    df["chain_level"] = df["chain_id"] + "__L" + df["level"].astype(str)
    df["chain_level_id"] = pd.factorize(df["chain_level"])[0]
    df["surface_id"] = df["surface_family"].map(SURFACE_MAP)
    return df


def load_control_df() -> pd.DataFrame:
    raw = json.load(open(CTRL_DATASET))
    rows = []
    for chain in raw["chains"]:
        chain_id = chain["chain_id"]
        for level_block in chain["levels"]:
            for idx, sentence in enumerate(level_block["examples"]):
                rows.append(
                    {
                        "chain_id": chain_id,
                        "level": level_block["level"],
                        "concept": level_block["concept"],
                        "surface_family": CTRL_SURFACE_MAP[idx],
                        "sentence": sentence,
                    }
                )
    df = pd.DataFrame(rows)
    df["chain_level"] = df["chain_id"] + "__L" + df["level"].astype(str)
    df["chain_level_id"] = pd.factorize(df["chain_level"])[0]
    df["surface_id"] = df["surface_family"].map(SURFACE_MAP)
    return df


def orthonormal_basis(W: np.ndarray) -> np.ndarray:
    if W.size == 0:
        return np.zeros((W.shape[0], 0), dtype=np.float32)
    Q, _ = np.linalg.qr(W.astype(np.float32, copy=False))
    return Q.astype(np.float32, copy=False)


def concat_bases(*bases: np.ndarray) -> np.ndarray:
    mats = [b for b in bases if b is not None and b.size]
    if not mats:
        width = bases[0].shape[0] if bases else 0
        return np.zeros((width, 0), dtype=np.float32)
    return orthonormal_basis(np.concatenate(mats, axis=1))


def lda_basis(X: np.ndarray, labels: np.ndarray, max_dim: int) -> np.ndarray:
    n_classes = len(np.unique(labels))
    k = min(max_dim, n_classes - 1, X.shape[0] - 1, X.shape[1])
    if k < 1:
        return np.zeros((X.shape[1], 0), dtype=np.float32)
    try:
        lda = LinearDiscriminantAnalysis(n_components=k)
        lda.fit(X, labels)
        return orthonormal_basis(lda.scalings_[:, :k])
    except Exception:
        return np.zeros((X.shape[1], 0), dtype=np.float32)


def label_mean_basis(X: np.ndarray, labels: np.ndarray, max_dim: int) -> np.ndarray:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return np.zeros((X.shape[1], 0), dtype=np.float32)
    means = np.stack([X[labels == label].mean(axis=0) for label in unique_labels], axis=0)
    means = means - means.mean(axis=0, keepdims=True)
    if min(means.shape) < 2:
        return np.zeros((X.shape[1], 0), dtype=np.float32)
    _, _, Vt = np.linalg.svd(means, full_matrices=False)
    k_eff = min(max_dim, unique_labels.size - 1, Vt.shape[0], X.shape[1])
    return Vt[:k_eff].T.astype(np.float32, copy=False)


def top_pc_basis(X: np.ndarray, k: int) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    if min(Xc.shape) < 2:
        return np.zeros((X.shape[1], 0), dtype=np.float32)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    k_eff = min(k, Vt.shape[0], X.shape[1])
    return Vt[:k_eff].T.astype(np.float32, copy=False)


def top_pc_basis_family(X: np.ndarray, max_k: int) -> dict[int, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    if min(Xc.shape) < 2:
        return {k: np.zeros((X.shape[1], 0), dtype=np.float32) for k in range(1, max_k + 1)}
    max_eff = min(max_k, Xc.shape[0] - 1, X.shape[1])
    if max_eff < 1:
        return {k: np.zeros((X.shape[1], 0), dtype=np.float32) for k in range(1, max_k + 1)}
    _, _, Vt = randomized_svd(
        Xc,
        n_components=max_eff,
        n_iter=3,
        random_state=0,
    )
    family = {}
    for k in range(1, max_k + 1):
        k_eff = min(k, max_eff)
        family[k] = Vt[:k_eff].T.astype(np.float32, copy=False)
    return family


def project_coords(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros((X.shape[0], 0), dtype=np.float32)
    return (X @ basis).astype(np.float32, copy=False)


def residualize_basis(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return X.astype(np.float32, copy=False)
    return (X - (X @ basis) @ basis.T).astype(np.float32, copy=False)


def whiten_coords(Y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if Y.shape[1] == 0:
        return Y.astype(np.float32, copy=False)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    if min(Yc.shape) < 2:
        return Yc.astype(np.float32, copy=False)
    _, s, Vt = np.linalg.svd(Yc, full_matrices=False)
    eigs = (s ** 2) / max(Yc.shape[0] - 1, 1)
    keep = eigs > 1e-10
    if not np.any(keep):
        return Yc.astype(np.float32, copy=False)
    V = Vt[keep].T
    return ((Yc @ V) / np.sqrt(eigs[keep] + eps)).astype(np.float32, copy=False)


def covariance_eigs(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    if Xc.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    gram = (Xc @ Xc.T) / max(Xc.shape[0] - 1, 1)
    eigs = np.linalg.eigvalsh(gram)[::-1]
    return eigs[eigs > 1e-12].astype(np.float32, copy=False)


def participation_ratio(X: np.ndarray) -> float:
    eigs = covariance_eigs(X)
    if eigs.size == 0:
        return 0.0
    return float((eigs.sum() ** 2) / np.square(eigs).sum())


def stable_rank(X: np.ndarray) -> float:
    eigs = covariance_eigs(X)
    if eigs.size == 0 or eigs[0] <= 0:
        return 0.0
    return float(eigs.sum() / eigs[0])


def log_volume(X: np.ndarray) -> float:
    eigs = covariance_eigs(X)
    if eigs.size == 0:
        return float("-inf")
    return float(0.5 * np.log(eigs).sum())


def mle_intrinsic_dim(X: np.ndarray, k: int = 5) -> float:
    n = X.shape[0]
    k_eff = min(k, n - 2)
    if n < 4 or k_eff < 2:
        return 0.0
    X = X.astype(np.float32, copy=False)
    sq = (X * X).sum(axis=1)
    d2 = np.clip(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0, None)
    np.fill_diagonal(d2, np.inf)
    dists = np.sqrt(np.sort(d2, axis=1)[:, :k_eff])
    r_k = dists[:, -1]
    log_ratios = np.log((r_k[:, None] + 1e-12) / (dists[:, :-1] + 1e-12))
    local = (k_eff - 1) / (log_ratios.sum(axis=1) + 1e-12)
    return float(np.median(local))


def metric_bundle(X: np.ndarray) -> dict[str, float]:
    return {
        "pr": participation_ratio(X),
        "sr": stable_rank(X),
        "mle_id": mle_intrinsic_dim(X),
        "log_vol": log_volume(X),
    }


def selectivity_score(X_proj: np.ndarray, sem_labels: np.ndarray, surf_labels: np.ndarray) -> float:
    if X_proj.shape[1] == 0:
        return 0.0
    clf = LogisticRegression(max_iter=400, solver="lbfgs", random_state=0)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    Xp = np.nan_to_num(X_proj, nan=0.0, posinf=0.0, neginf=0.0)
    sem_acc = cross_val_score(clf, Xp, sem_labels, cv=cv, scoring="accuracy").mean()
    surf_acc = cross_val_score(clf, Xp, surf_labels, cv=cv, scoring="accuracy").mean()
    return float(sem_acc - surf_acc)


def compute_language_bases_and_alignment(cal_acts: np.ndarray, lang_labels: np.ndarray):
    lang_bases: list[np.ndarray] = []
    rows: list[dict[str, float]] = []
    max_k = max(ALIGNMENT_K)
    for layer in range(cal_acts.shape[1]):
        if layer % 4 == 0:
            print(f"[calibration] layer {layer}/{cal_acts.shape[1] - 1}")
        X = cal_acts[:, layer, :].astype(np.float32, copy=False)
        lang_basis = label_mean_basis(X, lang_labels, LANG_DIM)
        lang_bases.append(lang_basis)
        pc_family = top_pc_basis_family(X, max_k)
        for k in ALIGNMENT_K:
            pc_basis = pc_family[k]
            if lang_basis.shape[1] == 0 or pc_basis.shape[1] == 0:
                overlap = 0.0
            else:
                overlap = float(
                    np.trace(lang_basis.T @ pc_basis @ pc_basis.T @ lang_basis) / lang_basis.shape[1]
                )
            rows.append({"layer": layer, "k": k, "overlap": overlap})
    return lang_bases, pd.DataFrame(rows)


def compute_condition_models(name: str, acts: np.ndarray, df: pd.DataFrame):
    sem_labels = df["chain_level_id"].values
    surface_labels = df["surface_id"].values
    sem_bases: list[np.ndarray] = []
    surface_bases: list[np.ndarray] = []
    top_bases: dict[int, list[np.ndarray]] = {k: [] for k in DEFLATION_K}
    selectivity = np.zeros(acts.shape[1], dtype=np.float32)

    print(f"[{name}] computing layerwise bases and selectivity")
    for layer in range(acts.shape[1]):
        if layer % 4 == 0:
            print(f"[{name}] layer {layer}/{acts.shape[1] - 1}")
        X = acts[:, layer, :].astype(np.float32, copy=False)
        sem_basis = lda_basis(X, sem_labels, SEM_DIM)
        surface_basis = label_mean_basis(X, surface_labels, SURFACE_DIM)
        pc_family = top_pc_basis_family(X, max(DEFLATION_K))
        sem_bases.append(sem_basis)
        surface_bases.append(surface_basis)
        for k in DEFLATION_K:
            top_bases[k].append(pc_family[k])
        selectivity[layer] = selectivity_score(project_coords(X, sem_basis), sem_labels, surface_labels)
    return {
        "selectivity": selectivity,
        "sem_bases": sem_bases,
        "surface_bases": surface_bases,
        "top_bases": top_bases,
    }


def build_group_index(df: pd.DataFrame):
    chains = sorted(df["chain_id"].unique())
    chain_arr = df["chain_id"].values
    level_arr = df["level"].values.astype(int)
    group_index = {}
    for chain in chains:
        chain_mask = chain_arr == chain
        for level in range(5):
            group_index[(chain, level)] = np.where(chain_mask & (level_arr == level))[0]
    return chains, group_index


def collect_static_rows(
    condition: str,
    acts: np.ndarray,
    df: pd.DataFrame,
    models: dict,
    lang_bases: list[np.ndarray],
    chains: list[str],
    group_index: dict[tuple[str, int], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    print(f"[{condition}] collecting static geometry rows")
    for layer in range(acts.shape[1]):
        X = acts[:, layer, :].astype(np.float32, copy=False)
        sem_basis = models["sem_bases"][layer]
        surface_basis = models["surface_bases"][layer]
        lang_basis = lang_bases[layer]
        surf_lang_basis = concat_bases(surface_basis, lang_basis)
        sem_proj = project_coords(X, sem_basis)
        branches = {
            "raw": X,
            "sem_proj": sem_proj,
            "whitened_sem": whiten_coords(sem_proj),
            "surface_resid": residualize_basis(X, surface_basis),
            "lang_resid": residualize_basis(X, lang_basis),
            "surface_lang_resid": residualize_basis(X, surf_lang_basis),
            "deflate_top3": residualize_basis(X, models["top_bases"][3][layer]),
            "deflate_top5": residualize_basis(X, models["top_bases"][5][layer]),
        }
        for branch, Xb in branches.items():
            for chain in chains:
                for level in range(5):
                    idx = group_index[(chain, level)]
                    metrics = metric_bundle(Xb[idx])
                    rows.append(
                        {
                            "condition": condition,
                            "layer": layer,
                            "branch": branch,
                            "chain": chain,
                            "level": level,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def summarise_static_deltas(df_static: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for condition in sorted(df_static["condition"].unique()):
        for branch in STATIC_BRANCHES:
            sub = df_static[(df_static["condition"] == condition) & (df_static["branch"] == branch)]
            for layer in sorted(sub["layer"].unique()):
                layer_sub = sub[sub["layer"] == layer]
                for metric in METRICS:
                    pivot = layer_sub.pivot(index="chain", columns="level", values=metric)
                    deltas = (pivot[4] - pivot[0]).dropna().astype(float)
                    rows.append(
                        {
                            "condition": condition,
                            "layer": int(layer),
                            "branch": branch,
                            "metric": metric,
                            "mean_delta": float(deltas.mean()),
                            "std_delta": float(deltas.std(ddof=0)),
                            "n_chains": int(deltas.shape[0]),
                        }
                    )
    return pd.DataFrame(rows)


def build_branch_states(
    acts: np.ndarray,
    models: dict,
    lang_bases: list[np.ndarray],
    branch: str,
) -> np.ndarray:
    out = np.empty_like(acts, dtype=np.float32)
    for layer in range(acts.shape[1]):
        X = acts[:, layer, :].astype(np.float32, copy=False)
        surface_basis = models["surface_bases"][layer]
        lang_basis = lang_bases[layer]
        if branch == "raw":
            Xt = X
        elif branch == "surface_resid":
            Xt = residualize_basis(X, surface_basis)
        elif branch == "lang_resid":
            Xt = residualize_basis(X, lang_basis)
        elif branch == "surface_lang_resid":
            Xt = residualize_basis(X, concat_bases(surface_basis, lang_basis))
        elif branch == "deflate_top3":
            Xt = residualize_basis(X, models["top_bases"][3][layer])
        elif branch == "deflate_top5":
            Xt = residualize_basis(X, models["top_bases"][5][layer])
        else:
            raise ValueError(f"Unsupported flow branch: {branch}")
        out[:, layer, :] = Xt
    return out


def collect_flow_rows(
    condition: str,
    acts: np.ndarray,
    df: pd.DataFrame,
    models: dict,
    lang_bases: list[np.ndarray],
    chains: list[str],
    group_index: dict[tuple[str, int], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for branch in FLOW_BRANCHES:
        print(f"[{condition}] collecting flow rows for branch={branch}")
        states = build_branch_states(acts, models, lang_bases, branch)
        flows = states[:, 1:, :] - states[:, :-1, :]
        for transition in range(flows.shape[1]):
            F = flows[:, transition, :]
            for chain in chains:
                for level in range(5):
                    idx = group_index[(chain, level)]
                    metrics = metric_bundle(F[idx])
                    rows.append(
                        {
                            "condition": condition,
                            "transition": transition,
                            "branch": branch,
                            "chain": chain,
                            "level": level,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def summarise_flow_deltas(df_flow: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for condition in sorted(df_flow["condition"].unique()):
        for branch in FLOW_BRANCHES:
            sub = df_flow[(df_flow["condition"] == condition) & (df_flow["branch"] == branch)]
            for transition in sorted(sub["transition"].unique()):
                trans_sub = sub[sub["transition"] == transition]
                for metric in METRICS:
                    pivot = trans_sub.pivot(index="chain", columns="level", values=metric)
                    deltas = (pivot[4] - pivot[0]).dropna().astype(float)
                    rows.append(
                        {
                            "condition": condition,
                            "transition": int(transition),
                            "branch": branch,
                            "metric": metric,
                            "mean_delta": float(deltas.mean()),
                            "std_delta": float(deltas.std(ddof=0)),
                            "n_chains": int(deltas.shape[0]),
                        }
                    )
    return pd.DataFrame(rows)


def plot_selectivity(layer_df: pd.DataFrame) -> None:
    wide = (
        layer_df.pivot(index="layer", columns="condition", values="selectivity")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for condition in ["refinement", "control"]:
        ax.plot(
            wide.index,
            wide[condition],
            label=condition.capitalize(),
            color=COND_COLORS[condition],
            linewidth=2,
        )
    shared_layer = int(wide[["refinement", "control"]].min(axis=1).idxmax())
    ref_best = int(wide["refinement"].idxmax())
    ctrl_best = int(wide["control"].idxmax())
    ax.axvline(ref_best, color=COND_COLORS["refinement"], linestyle="--", alpha=0.6)
    ax.axvline(ctrl_best, color=COND_COLORS["control"], linestyle="--", alpha=0.6)
    ax.axvline(shared_layer, color="black", linestyle=":", linewidth=2, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Selectivity")
    ax.set_title("Layer selection: condition-specific peaks plus shared comparison layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "selectivity_layers.png", dpi=150)
    plt.close(fig)


def plot_language_alignment(df_align: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in ALIGNMENT_K:
        sub = df_align[df_align["k"] == k].sort_values("layer")
        ax.plot(sub["layer"], sub["overlap"], label=f"top-{k}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Subspace overlap")
    ax.set_title("How much top principal directions align with language directions")
    ax.legend(title="PC deflation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "language_topk_alignment.png", dpi=150)
    plt.close(fig)


def plot_static_gap_heatmap(df_delta: pd.DataFrame, metric: str) -> None:
    sub = df_delta[df_delta["metric"] == metric]
    rows = []
    for branch in STATIC_BRANCHES:
        branch_sub = sub[sub["branch"] == branch]
        wide = branch_sub.pivot(index="layer", columns="condition", values="mean_delta").sort_index()
        rows.append((branch, (wide["refinement"] - wide["control"]).values))
    mat = np.vstack([vals for _branch, vals in rows])
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([BRANCH_LABELS[b] for b, _ in rows])
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(range(mat.shape[1]))
    ax.set_xlabel("Layer")
    ax.set_title(f"Static {metric}: refinement minus control delta by branch and layer")
    fig.colorbar(im, ax=ax, label="Refinement - control")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"static_gap_heatmap_{metric}.png", dpi=150)
    plt.close(fig)


def plot_static_metric(df_delta: pd.DataFrame, metric: str, shared_layer: int) -> None:
    branches = [
        "raw",
        "whitened_sem",
        "surface_resid",
        "lang_resid",
        "surface_lang_resid",
        "deflate_top5",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, branch in zip(axes.flat, branches):
        sub = df_delta[(df_delta["metric"] == metric) & (df_delta["branch"] == branch)]
        for condition in ["refinement", "control"]:
            row = sub[sub["condition"] == condition].sort_values("layer")
            ax.plot(
                row["layer"],
                row["mean_delta"],
                color=COND_COLORS[condition],
                label=condition.capitalize(),
                linewidth=2,
            )
            ax.fill_between(
                row["layer"],
                row["mean_delta"] - row["std_delta"],
                row["mean_delta"] + row["std_delta"],
                color=COND_COLORS[condition],
                alpha=0.15,
            )
        ax.axvline(shared_layer, color="black", linestyle=":", alpha=0.8)
        ax.set_title(BRANCH_LABELS[branch])
        ax.grid(True, alpha=0.3)
        if metric == "mle_id":
            ax.set_ylabel("Delta L4-L0")
        else:
            ax.set_ylabel("Delta L4-L0")
    axes[0, 0].legend()
    fig.suptitle(f"Static {metric}: level-4 minus level-0 difference across layers", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"static_delta_{metric}.png", dpi=150)
    plt.close(fig)


def plot_flow_gap_heatmap(df_flow_delta: pd.DataFrame, metric: str) -> None:
    sub = df_flow_delta[df_flow_delta["metric"] == metric]
    rows = []
    for branch in FLOW_BRANCHES:
        branch_sub = sub[sub["branch"] == branch]
        wide = branch_sub.pivot(index="transition", columns="condition", values="mean_delta").sort_index()
        rows.append((branch, (wide["refinement"] - wide["control"]).values))
    mat = np.vstack([vals for _branch, vals in rows])
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([BRANCH_LABELS[b] for b, _ in rows])
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(range(mat.shape[1]))
    ax.set_xlabel("Layer transition")
    ax.set_title(f"Flow {metric}: refinement minus control delta by branch and transition")
    fig.colorbar(im, ax=ax, label="Refinement - control")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"flow_gap_heatmap_{metric}.png", dpi=150)
    plt.close(fig)


def plot_flow_metric(df_flow_delta: pd.DataFrame, metric: str) -> None:
    branches = FLOW_BRANCHES
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, branch in zip(axes.flat, branches):
        sub = df_flow_delta[(df_flow_delta["metric"] == metric) & (df_flow_delta["branch"] == branch)]
        for condition in ["refinement", "control"]:
            row = sub[sub["condition"] == condition].sort_values("transition")
            ax.plot(
                row["transition"],
                row["mean_delta"],
                color=COND_COLORS[condition],
                label=condition.capitalize(),
                linewidth=2,
            )
            ax.fill_between(
                row["transition"],
                row["mean_delta"] - row["std_delta"],
                row["mean_delta"] + row["std_delta"],
                color=COND_COLORS[condition],
                alpha=0.15,
            )
        ax.set_title(BRANCH_LABELS[branch])
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Layer transition")
        ax.set_ylabel("Delta L4-L0")
    axes[0, 0].legend()
    fig.suptitle(f"Flow {metric}: level-4 minus level-0 difference by transition", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"flow_delta_{metric}.png", dpi=150)
    plt.close(fig)


def write_summary(
    layer_df: pd.DataFrame,
    df_align: pd.DataFrame,
    static_delta: pd.DataFrame,
    flow_delta: pd.DataFrame,
    shared_layer: int,
) -> None:
    wide_layers = layer_df.pivot(index="layer", columns="condition", values="selectivity").sort_index()
    ref_best = int(wide_layers["refinement"].idxmax())
    ctrl_best = int(wide_layers["control"].idxmax())

    shared_static = static_delta[
        (static_delta["layer"] == shared_layer) & (static_delta["metric"] == "mle_id")
    ]
    pivot_static = shared_static.pivot(index="branch", columns="condition", values="mean_delta")
    pivot_static["gap_ref_minus_ctrl"] = pivot_static["refinement"] - pivot_static["control"]
    best_static_branch = pivot_static["gap_ref_minus_ctrl"].idxmax()
    static_mean = (
        static_delta[static_delta["metric"] == "mle_id"]
        .groupby(["branch", "condition"])["mean_delta"]
        .mean()
        .unstack("condition")
    )
    static_mean["gap_ref_minus_ctrl"] = static_mean["refinement"] - static_mean["control"]
    best_static_mean_branch = static_mean["gap_ref_minus_ctrl"].idxmax()

    flow_mle = flow_delta[flow_delta["metric"] == "mle_id"]
    flow_pivot = (
        flow_mle.pivot_table(
            index="transition",
            columns=["branch", "condition"],
            values="mean_delta",
        )
        .sort_index()
    )
    best_flow_branch = None
    best_flow_transition = None
    best_flow_gap = float("-inf")
    for branch in FLOW_BRANCHES:
        key_ref = (branch, "refinement")
        key_ctrl = (branch, "control")
        if key_ref not in flow_pivot.columns or key_ctrl not in flow_pivot.columns:
            continue
        gap_curve = flow_pivot[key_ref] - flow_pivot[key_ctrl]
        local_transition = int(gap_curve.idxmax())
        local_gap = float(gap_curve.loc[local_transition])
        if local_gap > best_flow_gap:
            best_flow_branch = branch
            best_flow_transition = local_transition
            best_flow_gap = local_gap

    align_top5 = df_align[df_align["k"] == 5]
    best_align_row = align_top5.loc[align_top5["overlap"].idxmax()]

    lines = [
        "# Robust Pythia Summary",
        "",
        f"- Refinement best layer: {ref_best}",
        f"- Control best layer: {ctrl_best}",
        f"- Shared comparison layer (max-min selectivity): {shared_layer}",
        "",
        "## Static separation at the shared layer",
        "",
        f"- Strongest MLE separation branch: `{best_static_branch}`",
        f"- Shared-layer delta gap (refinement - control): "
        f"{pivot_static.loc[best_static_branch, 'gap_ref_minus_ctrl']:.4f}",
        f"- Strongest mean-across-layers MLE branch: `{best_static_mean_branch}`",
        f"- Mean layerwise gap: {static_mean.loc[best_static_mean_branch, 'gap_ref_minus_ctrl']:.4f}",
        "",
        "## Dynamic separation across transitions",
        "",
        f"- Strongest peak flow MLE branch: `{best_flow_branch}`",
        f"- Best transition: {best_flow_transition}",
        f"- Peak transition gap (refinement - control): {best_flow_gap:.4f}",
        "",
        "## Dominant directions vs language",
        "",
        f"- Highest top-5 / language overlap layer: {int(best_align_row['layer'])}",
        f"- Overlap value: {best_align_row['overlap']:.4f}",
        "",
        "## Recommendation",
        "",
        "- Use the multilingual calibration projector as the language-control ablation.",
        "- Do not subtract a single transferred English mean vector from the concept data.",
        "- Use the shared comparison layer for cross-condition snapshots and the full layerwise curves for interpretation.",
    ]
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines))


def main() -> None:
    print("[load] reading datasets and cached activations")
    ref_df = load_refinement_df()
    ctrl_df = load_control_df()
    ref_chains, ref_group_index = build_group_index(ref_df)
    ctrl_chains, ctrl_group_index = build_group_index(ctrl_df)
    ref_acts = np.load(REF_ACTS, mmap_mode=None).astype(np.float32, copy=False)
    ctrl_acts = np.load(CTRL_ACTS, mmap_mode=None).astype(np.float32, copy=False)
    cal = np.load(CALIBRATION_ACTS, allow_pickle=True)
    cal_acts = cal["activations"].astype(np.float32, copy=False)
    cal_langs = cal["languages"].astype(str)

    print("[calibration] learning layerwise language bases")
    lang_bases, df_align = compute_language_bases_and_alignment(cal_acts, cal_langs)
    df_align.to_csv(RESULTS_DIR / "language_topk_alignment.csv", index=False)

    ref_models = compute_condition_models("refinement", ref_acts, ref_df)
    ctrl_models = compute_condition_models("control", ctrl_acts, ctrl_df)

    layer_df = pd.DataFrame(
        {
            "layer": np.arange(ref_acts.shape[1]),
            "refinement": ref_models["selectivity"],
            "control": ctrl_models["selectivity"],
        }
    )
    layer_df["shared_score"] = np.minimum(layer_df["refinement"], layer_df["control"])
    shared_layer = int(layer_df["shared_score"].idxmax())

    layer_rows = pd.concat(
        [
            layer_df[["layer", "refinement", "shared_score"]].rename(
                columns={"refinement": "selectivity"}
            ).assign(condition="refinement"),
            layer_df[["layer", "control", "shared_score"]].rename(
                columns={"control": "selectivity"}
            ).assign(condition="control"),
        ],
        ignore_index=True,
    )
    layer_rows.to_csv(RESULTS_DIR / "layer_selection.csv", index=False)

    static_ref = collect_static_rows(
        "refinement", ref_acts, ref_df, ref_models, lang_bases, ref_chains, ref_group_index
    )
    static_ctrl = collect_static_rows(
        "control", ctrl_acts, ctrl_df, ctrl_models, lang_bases, ctrl_chains, ctrl_group_index
    )
    static_all = pd.concat([static_ref, static_ctrl], ignore_index=True)
    static_all.to_csv(RESULTS_DIR / "static_level_metrics.csv", index=False)
    static_delta = summarise_static_deltas(static_all)
    static_delta.to_csv(RESULTS_DIR / "static_delta_by_layer.csv", index=False)

    flow_ref = collect_flow_rows(
        "refinement", ref_acts, ref_df, ref_models, lang_bases, ref_chains, ref_group_index
    )
    flow_ctrl = collect_flow_rows(
        "control", ctrl_acts, ctrl_df, ctrl_models, lang_bases, ctrl_chains, ctrl_group_index
    )
    flow_all = pd.concat([flow_ref, flow_ctrl], ignore_index=True)
    flow_all.to_csv(RESULTS_DIR / "flow_level_metrics.csv", index=False)
    flow_delta = summarise_flow_deltas(flow_all)
    flow_delta.to_csv(RESULTS_DIR / "flow_delta_by_transition.csv", index=False)

    plot_selectivity(layer_rows)
    plot_language_alignment(df_align)
    plot_static_metric(static_delta, "mle_id", shared_layer)
    plot_static_metric(static_delta, "log_vol", shared_layer)
    plot_static_gap_heatmap(static_delta, "mle_id")
    plot_static_gap_heatmap(static_delta, "log_vol")
    plot_flow_metric(flow_delta, "mle_id")
    plot_flow_gap_heatmap(flow_delta, "mle_id")
    write_summary(layer_rows, df_align, static_delta, flow_delta, shared_layer)

    print("[done] results written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
