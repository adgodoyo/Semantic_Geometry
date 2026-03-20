"""Shared geometric measures applied to any N×D representation matrix."""
import numpy as np
from scipy.linalg import svd
from config import NOISE_FLOOR


def _safe_svd(X: np.ndarray) -> np.ndarray:
    """Return singular values of centred matrix X (shape N×D)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    if Xc.shape[0] < 2:
        return np.zeros(1)
    _, s, _ = svd(Xc, full_matrices=False, check_finite=False)
    eigs = s ** 2 / max(Xc.shape[0] - 1, 1)   # sample covariance eigenvalues
    return eigs[eigs > 0]


def participation_ratio(X: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ²  — effective dimensionality."""
    eigs = _safe_svd(X)
    if len(eigs) == 0:
        return 0.0
    return float(eigs.sum() ** 2 / (eigs ** 2).sum())


def effective_rank(X: np.ndarray) -> float:
    """ER = exp(H) where H = −Σ pᵢ log pᵢ, pᵢ = λᵢ / Σλ."""
    eigs = _safe_svd(X)
    if len(eigs) == 0:
        return 0.0
    p = eigs / eigs.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(H))


def recoverable_rank(X: np.ndarray, threshold: float = NOISE_FLOOR) -> int:
    """Number of eigenvalues > threshold * λ_max."""
    eigs = _safe_svd(X)
    if len(eigs) == 0:
        return 0
    return int((eigs > threshold * eigs[0]).sum())


def stable_rank(X: np.ndarray) -> float:
    """Stable rank = ‖A‖_F² / ‖A‖_2² = Σλ / λ_max.

    Less sensitive to a single dominant eigenvalue than PR.
    Returns 1.0 when all variance is in one direction.
    """
    eigs = _safe_svd(X)
    if len(eigs) == 0 or eigs[0] == 0:
        return 0.0
    return float(eigs.sum() / eigs[0])


def mle_intrinsic_dim(X: np.ndarray, k: int = 5) -> float:
    """Maximum likelihood estimate of intrinsic dimensionality (Levina & Bickel 2004).

    Uses k-NN distances to estimate local dimensionality at each point,
    then returns the median across points. Robust to anisotropy and to the
    N << D regime because it operates on pairwise distances, not the covariance.

    Args:
        X: (N, D) matrix of representations.
        k: number of nearest neighbours to use (default 5; needs N > k+1).

    Returns:
        Estimated intrinsic dimensionality (float).
    """
    from sklearn.neighbors import NearestNeighbors
    N = X.shape[0]
    if N <= k + 1:
        return 0.0
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    dists = dists[:, 1:]                       # (N, k) — exclude self (distance 0)
    r_k = dists[:, -1]                         # (N,) k-th neighbour distance
    # log(r_k / r_j) for j = 1..k-1 → shape (N, k-1)
    log_ratios = np.log(r_k[:, None] / (dists[:, :-1] + 1e-12))
    local_dim = (k - 1) / (log_ratios.sum(axis=1) + 1e-12)
    return float(np.median(local_dim))


def log_volume(X: np.ndarray, n_components: int | None = None) -> float:
    """Log-volume of the Gaussian confidence ellipsoid in the effective PCA subspace.

    Defined as (1/2) * Σ log(λᵢ) over the top-k positive eigenvalues.
    Captures the full volumetric expansion of the point cloud: grows when
    new points spread into genuinely new directions, not just when one
    direction already occupied gets a bit wider.

    When n_components is None, uses all positive eigenvalues (up to numerical rank).

    Returns -inf when the point cloud is degenerate (rank < 1).
    """
    eigs = _safe_svd(X)
    if len(eigs) == 0:
        return float("-inf")
    if n_components is not None:
        eigs = eigs[:n_components]
    pos = eigs[eigs > 1e-12]
    if len(pos) == 0:
        return float("-inf")
    return float(0.5 * np.sum(np.log(pos)))


def eigenspectrum(X: np.ndarray, n_components: int = 20) -> np.ndarray:
    """Return top-n normalised eigenvalues (always length n_components, zero-padded)."""
    eigs = _safe_svd(X)
    out = np.zeros(n_components)
    k = min(len(eigs), n_components)
    out[:k] = eigs[:k]
    if out.sum() > 0:
        out = out / out.sum()
    return out


def all_measures(X: np.ndarray) -> dict:
    """Compute PR, ER, RR, stable rank, MLE intrinsic dim, and log-volume in one call."""
    eigs = _safe_svd(X)
    if len(eigs) == 0:
        return {"pr": 0.0, "er": 0.0, "rr": 0, "sr": 0.0, "mle_id": 0.0, "log_vol": float("-inf")}
    pr = float(eigs.sum() ** 2 / (eigs ** 2).sum())
    p = eigs / eigs.sum()
    er = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    rr = int((eigs > NOISE_FLOOR * eigs[0]).sum())
    sr = float(eigs.sum() / eigs[0])
    mle_id = mle_intrinsic_dim(X)
    pos = eigs[eigs > 1e-12]
    lv = float(0.5 * np.sum(np.log(pos))) if len(pos) > 0 else float("-inf")
    return {"pr": pr, "er": er, "rr": rr, "sr": sr, "mle_id": mle_id, "log_vol": lv}
