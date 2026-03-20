"""All plotting functions. Each function saves a PNG and returns the path."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from config import N_LEVELS

PALETTE = sns.color_palette("husl", 10)
LEVEL_COLORS = sns.color_palette("YlOrRd", N_LEVELS)
sns.set_theme(style="whitegrid", font_scale=1.1)


# ── helpers ──────────────────────────────────────────────────────────────────

def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Approach A ───────────────────────────────────────────────────────────────

def plot_selectivity_curve(selectivity: np.ndarray,
                           out_path: str,
                           title: str = "Semantic Selectivity per Layer") -> str:
    """selectivity: (n_layers,) S_ℓ = acc_sem − acc_surf"""
    fig, ax = plt.subplots(figsize=(8, 4))
    layers = np.arange(len(selectivity))
    ax.plot(layers, selectivity, "o-", color="#2d6a9f", lw=2, ms=6)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    best = int(np.argmax(selectivity))
    ax.axvline(best, color="#e05c2d", ls="--", lw=1.2,
               label=f"Best layer L{best} ({selectivity[best]:.2f})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Selectivity (sem acc − surf acc)")
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(layers)
    return _save(fig, out_path)


def plot_pr_trajectory(pr_by_chain: dict,
                       out_path: str,
                       measure: str = "PR",
                       title: str = "Participation Ratio vs Semantic Level") -> str:
    """
    pr_by_chain: {chain_id: np.ndarray of shape (n_levels,)}
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("tab10", len(pr_by_chain))
    for i, (chain, vals) in enumerate(pr_by_chain.items()):
        levels = np.arange(len(vals))
        ax.plot(levels, vals, "o-", color=colors[i], lw=1.8, ms=6,
                label=chain.replace("_", " "))
    ax.set_xlabel("Semantic constraint level (L0 = base, L4 = fully constrained)")
    ax.set_ylabel(measure)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xticks(range(N_LEVELS))
    return _save(fig, out_path)


def plot_pr_comparison_ab(pr_A: dict, pr_B: dict,
                          out_path: str,
                          approach_labels=("Semantic subspace (A)", "ZCA-whitened (B)")) -> str:
    """Side-by-side PR trajectories for Approach A vs B."""
    chains = list(pr_A.keys())
    n = len(chains)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    colors = sns.color_palette("tab10", n)
    for i, chain in enumerate(chains):
        for ax, d, label in zip(axes, [pr_A, pr_B], approach_labels):
            vals = d.get(chain, np.zeros(N_LEVELS))
            ax.plot(range(len(vals)), vals, "o-", color=colors[i],
                    lw=1.6, ms=5, label=chain.replace("_", " ") if ax == axes[0] else "")
    for ax, label in zip(axes, approach_labels):
        ax.set_title(label)
        ax.set_xlabel("Level")
        ax.set_ylabel("PR")
        ax.set_xticks(range(N_LEVELS))
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Participation Ratio: Semantic Subspace vs Whitening Control", fontsize=12)
    return _save(fig, out_path)


def plot_eigenspectrum_heatmap(spectra: np.ndarray,
                               row_labels: list[str],
                               out_path: str,
                               title: str = "Eigenspectrum (normalised)") -> str:
    """
    spectra: (n_rows, n_components) normalised eigenvalues per row.
    row_labels: e.g. ["chain@L0", "chain@L1", ...]
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(row_labels) * 0.35)))
    sns.heatmap(spectra, ax=ax, cmap="magma_r", xticklabels=range(1, spectra.shape[1] + 1),
                yticklabels=row_labels, cbar_kws={"label": "Normalised eigenvalue"},
                vmin=0.0, vmax=0.30,
                linewidths=0.3, linecolor="white")
    ax.set_xlabel("Component rank")
    ax.set_ylabel("")
    ax.set_title(title)
    return _save(fig, out_path)


def plot_pca_scatter(X: np.ndarray,
                     chain_ids: np.ndarray,
                     level_labels: np.ndarray,
                     out_path: str,
                     title: str = "PCA of Semantic Subspace") -> str:
    """2D PCA scatter coloured by level, shaped by chain."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    chains = sorted(set(chain_ids))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]
    colors = sns.color_palette("YlOrRd", N_LEVELS)
    for ci, chain in enumerate(chains):
        mask = chain_ids == chain
        for l in range(N_LEVELS):
            m = mask & (level_labels == l)
            if m.sum() == 0:
                continue
            ax.scatter(Z[m, 0], Z[m, 1],
                       color=colors[l],
                       marker=markers[ci % len(markers)],
                       s=40, alpha=0.8,
                       label=f"{chain[:10]}·L{l}" if l == 0 else "")
    # Custom legend: level colours
    for l in range(N_LEVELS):
        ax.scatter([], [], color=colors[l], s=30, label=f"Level {l}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3)
    return _save(fig, out_path)


def plot_layer_pr_heatmap(pr_matrix: np.ndarray,
                          chain_ids: list[str],
                          out_path: str,
                          title: str = "PR by Chain and Layer") -> str:
    """
    pr_matrix: (n_chains, n_layers)
    """
    fig, ax = plt.subplots(figsize=(12, max(4, len(chain_ids) * 0.55)))
    df_hm = pd.DataFrame(pr_matrix,
                         index=[c.replace("_", " ") for c in chain_ids],
                         columns=[f"L{i}" for i in range(pr_matrix.shape[1])])
    sns.heatmap(df_hm, ax=ax, cmap="viridis", annot=True, fmt=".1f",
                cbar_kws={"label": "PR"}, linewidths=0.3)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    return _save(fig, out_path)


# ── Approach C (flow) ─────────────────────────────────────────────────────────

def plot_flow_magnitude(flow_mag: np.ndarray,
                        out_path: str,
                        title: str = "Mean Flow Magnitude per Layer Transition") -> str:
    """flow_mag: (n_transitions,)"""
    fig, ax = plt.subplots(figsize=(8, 4))
    t = np.arange(len(flow_mag))
    ax.bar(t, flow_mag, color="#4a90d9", alpha=0.85)
    ax.set_xlabel("Layer transition (ℓ → ℓ+1)")
    ax.set_ylabel("‖Δh‖ (mean over sentences)")
    ax.set_title(title)
    ax.set_xticks(t)
    ax.set_xticklabels([f"{i}→{i+1}" for i in t], rotation=45, fontsize=8)
    return _save(fig, out_path)


def plot_flow_pr_layers(flow_pr: np.ndarray,
                        out_path: str,
                        title: str = "Flow PR per Layer Transition") -> str:
    """flow_pr: (n_transitions,)"""
    fig, ax = plt.subplots(figsize=(8, 4))
    t = np.arange(len(flow_pr))
    ax.plot(t, flow_pr, "o-", color="#e05c2d", lw=2, ms=6)
    ax.set_xlabel("Layer transition (ℓ → ℓ+1)")
    ax.set_ylabel("Participation Ratio of flow cloud")
    ax.set_title(title)
    ax.set_xticks(t)
    ax.set_xticklabels([f"{i}→{i+1}" for i in t], rotation=45, fontsize=8)
    return _save(fig, out_path)


def plot_centroid_distances(dist_matrix: np.ndarray,
                            chain_id: str,
                            out_path: str) -> str:
    """
    dist_matrix: (n_levels-1, n_layers) — consecutive-level L2 distances.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("RdYlBu", dist_matrix.shape[0])
    for i, row in enumerate(dist_matrix):
        ax.plot(range(len(row)), row, "o-", color=colors[i], lw=1.8, ms=5,
                label=f"L{i}→L{i+1}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("‖μ_{l+1} − μ_l‖ in hidden space")
    ax.set_title(f"Centroid Distance Trajectory — {chain_id.replace('_', ' ')}")
    ax.legend(fontsize=9)
    return _save(fig, out_path)


def plot_cumulative_flow_pr(cum_pr: np.ndarray,
                            per_trans_pr: np.ndarray,
                            out_path: str,
                            title: str = "Cumulative vs Per-transition Flow PR") -> str:
    """Plot cumulative flow PR (directions accumulated across layers) vs per-transition PR.

    cum_pr: (n_transitions,) — PR of stacked flows from L0 to transition t.
    per_trans_pr: (n_transitions,) — PR of flows at each individual transition.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    t = np.arange(len(cum_pr))
    ax.plot(t, cum_pr, "o-", color="#2d6a9f", lw=2, ms=6,
            label="Cumulative (L0→t stacked)")
    ax.plot(t, per_trans_pr, "s--", color="#e05c2d", lw=1.5, ms=5, alpha=0.75,
            label="Per-transition")
    ax.set_xlabel("Layer transition (ℓ → ℓ+1)")
    ax.set_ylabel("Participation Ratio")
    ax.set_title(title)
    ax.set_xticks(t)
    ax.set_xticklabels([f"{i}→{i+1}" for i in t], rotation=45, fontsize=8)
    ax.legend()
    return _save(fig, out_path)


def plot_flow_alignment(alignment: np.ndarray,
                        out_path: str,
                        title: str = "Intra-class Flow Alignment per Transition") -> str:
    """alignment: (n_transitions,) — mean cosine similarity within semantic class."""
    fig, ax = plt.subplots(figsize=(8, 4))
    t = np.arange(len(alignment))
    ax.fill_between(t, 0, alignment, alpha=0.3, color="#2d9f6a")
    ax.plot(t, alignment, "o-", color="#2d9f6a", lw=2, ms=6)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Layer transition (ℓ → ℓ+1)")
    ax.set_ylabel("Mean intra-class cosine similarity")
    ax.set_title(title)
    ax.set_xticks(t)
    ax.set_xticklabels([f"{i}→{i+1}" for i in t], rotation=45, fontsize=8)
    return _save(fig, out_path)


def plot_all_chains_centroid_heatmap(all_dists: dict,
                                     out_path: str,
                                     layer: int = 8) -> str:
    """
    all_dists: {chain_id: dist_matrix (n_levels-1, n_layers)}
    Show per-chain, per-level consecutive centroid distance at a fixed layer.
    """
    chains = list(all_dists.keys())
    n_chains = len(chains)
    n_gaps = N_LEVELS - 1
    mat = np.zeros((n_chains, n_gaps))
    for ci, chain in enumerate(chains):
        d = all_dists[chain]
        if layer < d.shape[1]:
            mat[ci] = d[:, layer]

    fig, ax = plt.subplots(figsize=(8, max(4, n_chains * 0.5)))
    sns.heatmap(mat, ax=ax, cmap="YlOrRd",
                xticklabels=[f"L{i}→L{i+1}" for i in range(n_gaps)],
                yticklabels=[c.replace("_", " ") for c in chains],
                annot=True, fmt=".1f",
                cbar_kws={"label": "‖Δμ‖"})
    ax.set_title(f"Consecutive Level Centroid Distances at Layer {layer}")
    return _save(fig, out_path)


def plot_geometry_summary(df: pd.DataFrame, out_path: str) -> str:
    """
    df columns: chain, level, pr_A, pr_B_zca, pr_B_deflate5, er_A, rr_A
    Faceted line plots.
    """
    measures = [c for c in df.columns if c.startswith("pr_") or c.startswith("er_")]
    n = len(measures)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    chains = df["chain"].unique()
    colors = sns.color_palette("tab10", len(chains))

    for idx, meas in enumerate(measures):
        ax = axes[idx // cols][idx % cols]
        for ci, chain in enumerate(chains):
            sub = df[df["chain"] == chain].sort_values("level")
            ax.plot(sub["level"], sub[meas], "o-", color=colors[ci],
                    lw=1.5, ms=4, label=chain.replace("_", " ") if idx == 0 else "")
        ax.set_title(meas.upper().replace("_", " "))
        ax.set_xlabel("Level")
        ax.set_xticks(range(N_LEVELS))

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Geometry Measures vs Semantic Level", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _save(fig, out_path)
