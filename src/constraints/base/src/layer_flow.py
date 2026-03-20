"""Approach C — Layer-flow / vector field analysis.

Studies how representations CHANGE from layer to layer (Δh_ℓ = h_{ℓ+1} − h_ℓ)
rather than their static geometry at any single layer.

Key functions:
  - compute_flows(acts) -> flows  shape (N, n_layers-1, D)
  - flow_pr_per_transition(flows) -> (n_transitions,)
  - centroid_distance_matrix(acts, chain_mask, level_labels) -> dict
  - level_divergence_trajectory(acts, level_labels) -> (n_layers, n_level_pairs)
  - flow_alignment(flows, labels) -> per-transition mean intra-class cosine similarity
"""
import numpy as np
from geometry import all_measures, participation_ratio, effective_rank


def compute_flows(acts: np.ndarray) -> np.ndarray:
    """
    Compute layer-to-layer displacement vectors.

    Args:
        acts: (N, n_layers, D)

    Returns:
        flows: (N, n_layers-1, D)  where flows[:,ℓ,:] = h_{ℓ+1} - h_ℓ
    """
    return acts[:, 1:, :] - acts[:, :-1, :]   # shape (N, n_layers-1, D)


def flow_magnitude_per_transition(flows: np.ndarray) -> np.ndarray:
    """Mean ‖Δh_ℓ‖ per layer transition, shape (n_transitions,)."""
    return np.linalg.norm(flows, axis=-1).mean(axis=0)   # (n_transitions,)


def flow_pr_per_transition(flows: np.ndarray) -> np.ndarray:
    """Participation ratio of the flow cloud at each transition."""
    n_trans = flows.shape[1]
    prs = np.zeros(n_trans)
    for t in range(n_trans):
        prs[t] = participation_ratio(flows[:, t, :])
    return prs


def flow_er_per_transition(flows: np.ndarray) -> np.ndarray:
    """Effective rank of flow cloud at each transition."""
    n_trans = flows.shape[1]
    ers = np.zeros(n_trans)
    for t in range(n_trans):
        ers[t] = effective_rank(flows[:, t, :])
    return ers


def centroid_trajectory(acts: np.ndarray,
                        chain_df_idx: np.ndarray,
                        level_labels: np.ndarray,
                        n_levels: int = 5) -> np.ndarray:
    """
    Compute per-level centroid at each layer.

    Args:
        acts: (N, n_layers, D)
        chain_df_idx: boolean or integer mask selecting rows for this chain
        level_labels: (N,) level 0..4 for all rows in this chain
        n_levels: number of levels

    Returns:
        centroids: (n_levels, n_layers, D)
    """
    X = acts[chain_df_idx]              # (N_chain, n_layers, D)
    lvl = level_labels                  # (N_chain,)
    n_layers = X.shape[1]
    D = X.shape[2]
    centroids = np.zeros((n_levels, n_layers, D))
    for l in range(n_levels):
        mask = (lvl == l)
        if mask.sum() > 0:
            centroids[l] = X[mask].mean(axis=0)
    return centroids


def pairwise_centroid_distances(centroids: np.ndarray) -> np.ndarray:
    """
    Per-layer pairwise L2 distances between level centroids.

    Args:
        centroids: (n_levels, n_layers, D)

    Returns:
        distances: (n_levels, n_levels, n_layers)
    """
    n_levels, n_layers, D = centroids.shape
    dists = np.zeros((n_levels, n_levels, n_layers))
    for ℓ in range(n_layers):
        C = centroids[:, ℓ, :]        # (n_levels, D)
        diff = C[:, None, :] - C[None, :, :]   # (n_levels, n_levels, D)
        dists[:, :, ℓ] = np.linalg.norm(diff, axis=-1)
    return dists


def consecutive_level_distance(centroids: np.ndarray) -> np.ndarray:
    """
    Mean L2 distance between consecutive level centroids, per layer.

    Returns:
        (n_levels-1, n_layers)
    """
    n_levels, n_layers, D = centroids.shape
    result = np.zeros((n_levels - 1, n_layers))
    for l in range(n_levels - 1):
        result[l] = np.linalg.norm(centroids[l + 1] - centroids[l], axis=-1)
    return result


def flow_alignment_score(flows: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Intra-class cosine similarity of flow vectors per transition.

    High score → semantically similar sentences move in the same direction.

    Returns:
        alignment: (n_transitions,)
    """
    n_trans = flows.shape[1]
    scores = np.zeros(n_trans)
    unique_labels = np.unique(labels)

    for t in range(n_trans):
        F = flows[:, t, :]          # (N, D)
        norms = np.linalg.norm(F, axis=-1, keepdims=True) + 1e-10
        F_normed = F / norms

        sim_sum, count = 0.0, 0
        for lab in unique_labels:
            idx = np.where(labels == lab)[0]
            if len(idx) < 2:
                continue
            F_cls = F_normed[idx]   # (k, D)
            G = F_cls @ F_cls.T     # (k, k) cosine similarity matrix
            # Mean upper triangle (excluding diagonal)
            k = len(idx)
            pairs = (G.sum() - k) / (k * (k - 1)) if k > 1 else 0.0
            sim_sum += pairs
            count += 1
        scores[t] = sim_sum / count if count > 0 else 0.0

    return scores


def cumulative_flow_pr(acts: np.ndarray,
                       chain_idx: np.ndarray,
                       level_labels: np.ndarray,
                       layer_range: list[int]) -> np.ndarray:
    """
    For a chain, compute cumulative flow PR as levels are added, at a given layer.

    Returns:
        pr_matrix: (n_levels, len(layer_range))
    """
    n_levels = int(level_labels.max()) + 1
    result = np.zeros((n_levels, len(layer_range)))
    flows = compute_flows(acts[chain_idx])   # (N_chain, n_layers-1, D)

    for li, layer in enumerate(layer_range):
        if layer >= flows.shape[1]:
            continue
        for l_max in range(n_levels):
            mask = level_labels <= l_max
            if mask.sum() < 2:
                continue
            result[l_max, li] = participation_ratio(flows[mask, layer, :])

    return result


def cumulative_flow_pr_across_layers(flows: np.ndarray) -> np.ndarray:
    """PR of the flow field accumulated across layer transitions.

    At each step t, stacks all displacement vectors from transitions 0..t into
    a single matrix of shape ((t+1)*N, D) and computes its PR.

    This answers: "how many independent directions have been used cumulatively
    by all layer transitions up to and including transition t?"

    Unlike per-transition PR (which measures diversity *within* one step),
    this accumulates directions across steps — it grows only when a new
    transition introduces directions not already spanned by prior transitions.

    Args:
        flows: (N, n_transitions, D)

    Returns:
        cum_pr: (n_transitions,)
    """
    N, n_trans, D = flows.shape
    cum_pr = np.zeros(n_trans)
    for t in range(n_trans):
        # Stack all flows from transitions 0..t: shape ((t+1)*N, D)
        stacked = flows[:, : t + 1, :].reshape(-1, D)
        cum_pr[t] = participation_ratio(stacked)
    return cum_pr


def cumulative_flow_er_across_layers(flows: np.ndarray) -> np.ndarray:
    """Effective rank of flow field accumulated across layer transitions.

    Same logic as cumulative_flow_pr_across_layers but using ER (entropy-based).

    Args:
        flows: (N, n_transitions, D)

    Returns:
        cum_er: (n_transitions,)
    """
    N, n_trans, D = flows.shape
    cum_er = np.zeros(n_trans)
    for t in range(n_trans):
        stacked = flows[:, : t + 1, :].reshape(-1, D)
        cum_er[t] = effective_rank(stacked)
    return cum_er
