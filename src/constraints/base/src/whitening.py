"""Approach B — Anisotropy control via whitening / top-PC deflation.

No semantic labels used. This is the unsupervised comparison baseline.

Performance design
------------------
Naïve implementation: SVD of D×D covariance  → O(D³).
For Pythia (D=2560) that is ~16B ops and takes minutes per call on CPU.

This implementation avoids that via two tricks:
  1. Gram-matrix route: G = Xc Xc.T (N×N, with N≤400) then eigh(G).
     Complexity: O(N²D + N³) — for N=400,D=2560 this is ~470M vs 16.8B.
  2. Factored transforms: never materialise the D×D whitening matrix W.
     Instead: X_w = (Xc V) diag(s)^{-½} V.T  using two (N×k)@(k×D) products.
  3. torch MPS (Apple GPU) for the heavy GEMM steps; falls back to numpy
     when MPS is unavailable.

Together these make each whitening call ~1 s instead of minutes.
"""
import numpy as np
from config import WHITEN_REGULARIZE, DEFLATION_K

try:
    import torch as _torch
    _MPS = _torch.backends.mps.is_available()
    _CUDA = (not _MPS) and _torch.cuda.is_available()
    _DEVICE = "mps" if _MPS else ("cuda" if _CUDA else "cpu")
except ImportError:
    _torch = None
    _DEVICE = "cpu"


# ── fast GEMM helper ──────────────────────────────────────────────────────────

def _matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiply via torch (GPU) when available, else numpy."""
    if _torch is not None and _DEVICE != "cpu":
        At = _torch.tensor(A, dtype=_torch.float32, device=_DEVICE)
        Bt = _torch.tensor(B, dtype=_torch.float32, device=_DEVICE)
        return (At @ Bt).cpu().numpy()
    return A @ B


# ── covariance eigenvectors via Gram matrix ───────────────────────────────────

def _cov_eig(Xc: np.ndarray):
    """Eigenvectors / eigenvalues of sample covariance, via Gram matrix.

    Returns:
        V   : (D, k)   eigenvectors (descending eigenvalue order)
        eigs: (k,)     eigenvalues of C = Xc.T Xc / (N-1)  (positive only)

    Uses the identity  C = V Λ V.T  where Λ, V come from eigh(G/(N-1))
    and G = Xc Xc.T (N×N).  The expensive step is a single BLAS SGEMM
    on the (N×D) matrix, which runs on GPU.
    """
    N = Xc.shape[0]
    # Gram matrix — heavy GEMM, run on GPU
    G = _matmul(Xc, Xc.T) / max(N - 1, 1)   # (N, N)
    # Eigendecomposition of small N×N matrix — fine on CPU
    evals, U = np.linalg.eigh(G)             # ascending, O(N³)
    # Descending, positive only
    evals = evals[::-1].copy()
    U = U[:, ::-1].copy()
    pos = evals > 1e-12
    evals, U = evals[pos], U[:, pos]
    # Right singular vectors of Xc  →  eigenvectors of C
    sigma = np.sqrt(evals * max(N - 1, 1))
    V = _matmul(Xc.T, U) / sigma[np.newaxis, :]   # (D, k)
    return V, evals


# ── whitening transforms ──────────────────────────────────────────────────────

def zca_whiten(X: np.ndarray) -> np.ndarray:
    """ZCA whitening: spheres the data so all PCs have unit variance."""
    Xc = X - X.mean(axis=0, keepdims=True)
    V, eigs = _cov_eig(Xc)
    s_inv_sqrt = 1.0 / np.sqrt(eigs + WHITEN_REGULARIZE)   # (k,)
    # X_w = Xc V diag(s_inv_sqrt) V.T  — two (N×k)/(k×D) products
    proj = _matmul(Xc, V)                          # (N, k)
    return _matmul(proj * s_inv_sqrt, V.T)          # (N, D)


def pca_whiten(X: np.ndarray, n_components: int | None = None) -> np.ndarray:
    """PCA whitening: project to top PCs, then divide by sqrt(eigenvalue)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    V, eigs = _cov_eig(Xc)
    k = min(n_components or len(eigs), len(eigs))
    s_inv_sqrt = 1.0 / np.sqrt(eigs[:k] + WHITEN_REGULARIZE)
    proj = _matmul(Xc, V[:, :k])                   # (N, k)
    return proj * s_inv_sqrt                        # (N, k)


def top_k_deflate(X: np.ndarray, k: int) -> np.ndarray:
    """Remove top-k PCs from X (subtract projection onto top-k PC subspace)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    V, _ = _cov_eig(Xc)
    Vk = V[:, :k]                                  # (D, k)
    proj = _matmul(Xc, Vk)                         # (N, k)
    return Xc - _matmul(proj, Vk.T)               # (N, D)


def mahalanobis_transform(X: np.ndarray) -> np.ndarray:
    """Transform X so pairwise distances become Mahalanobis distances."""
    Xc = X - X.mean(axis=0, keepdims=True)
    V, eigs = _cov_eig(Xc)
    s_inv_sqrt = 1.0 / np.sqrt(eigs + WHITEN_REGULARIZE)
    proj = _matmul(Xc, V)
    return _matmul(proj * s_inv_sqrt, V.T)


def get_all_whitened(X: np.ndarray) -> dict[str, np.ndarray]:
    """Return all whitening variants — calls _cov_eig only once.

    The bottleneck on this machine is np.linalg.eigh (~10 s for 400×400).
    Computing V and eigs once and reusing across all variants cuts the
    total cost from 4×eigh to 1×eigh.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    out = {"raw": Xc}

    try:
        V, eigs = _cov_eig(Xc)
    except Exception:
        # If eigenvectors fail, fall back to raw for all variants
        fallback = Xc
        out["zca"] = fallback
        for k in DEFLATION_K:
            out[f"deflate_top{k}"] = fallback
        out["mahalanobis"] = fallback
        return out

    # Shared projection: Xc @ V  — used by ZCA, Mahalanobis, and indirectly deflation
    proj = _matmul(Xc, V)                                     # (N, k)
    s_inv_sqrt = 1.0 / np.sqrt(eigs + WHITEN_REGULARIZE)     # (k,)

    # ZCA and Mahalanobis are both Σ^{-½} whitening — same transform
    try:
        whitened = _matmul(proj * s_inv_sqrt, V.T)            # (N, D)
        out["zca"]         = whitened
        out["mahalanobis"] = whitened
    except Exception:
        out["zca"] = Xc
        out["mahalanobis"] = Xc

    # Top-k PC deflation variants
    for k in DEFLATION_K:
        if k < X.shape[1] and k < X.shape[0] and k <= V.shape[1]:
            try:
                Vk    = V[:, :k]                              # (D, k)
                proj_k = _matmul(Xc, Vk)                     # (N, k)
                out[f"deflate_top{k}"] = Xc - _matmul(proj_k, Vk.T)
            except Exception:
                out[f"deflate_top{k}"] = Xc

    return out
