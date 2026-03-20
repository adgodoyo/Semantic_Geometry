"""Approach A — Semantic subspace isolation via LDA.

Provides:
  - learn_nuisance_projector(X, surface_labels) -> P_nuis (D×D)
  - learn_semantic_projector(X, semantic_labels) -> P_sem (D×D) and directions (D×k)
  - project_semantic(X, P_sem) -> X_sem
  - residualise_nuisance(X, P_nuis) -> X_res
  - selectivity_score(X_proj, sem_labels, surf_labels) -> float
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config import SEM_SUBSPACE_DIM, NUIS_SUBSPACE_DIM


def _lda_directions(X: np.ndarray, labels: np.ndarray, n_components: int) -> np.ndarray:
    """Fit LDA and return the (D × n_components) discriminant direction matrix."""
    n_classes = len(np.unique(labels))
    k = min(n_components, n_classes - 1)
    if k < 1:
        return np.zeros((X.shape[1], 1))
    lda = LinearDiscriminantAnalysis(n_components=k)
    lda.fit(X, labels)
    return lda.scalings_[:, :k]   # shape (D, k)


def _projector_from_directions(W: np.ndarray) -> np.ndarray:
    """Orthonormal projector P = V V^T from direction matrix W (D×k)."""
    V, _ = np.linalg.qr(W)       # orthonormalise columns
    return V @ V.T               # (D × D) projector


def learn_nuisance_projector(X: np.ndarray, surface_labels: np.ndarray) -> np.ndarray:
    """P_nuis: projection onto the nuisance (surface_family) subspace."""
    W = _lda_directions(X, surface_labels, NUIS_SUBSPACE_DIM)
    return _projector_from_directions(W)


def learn_semantic_projector(X: np.ndarray,
                              semantic_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P_sem and direction matrix V (D×k) from chain×level labels."""
    W = _lda_directions(X, semantic_labels, SEM_SUBSPACE_DIM)
    P = _projector_from_directions(W)
    return P, W


def project_semantic(X: np.ndarray, P_sem: np.ndarray) -> np.ndarray:
    """Project X into the semantic subspace."""
    return X @ P_sem.T


def residualise_nuisance(X: np.ndarray, P_nuis: np.ndarray) -> np.ndarray:
    """Remove nuisance directions: h̃ = (I − P_nuis) h."""
    return X - X @ P_nuis.T


def selectivity_score(X_proj: np.ndarray,
                      sem_labels: np.ndarray,
                      surf_labels: np.ndarray,
                      cv: int = 3) -> float:
    """
    S = accuracy(semantic recovery) − accuracy(nuisance recovery)
    Both measured by cross-validated logistic regression on X_proj.
    """
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=0)
    cv_ = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    # Clamp X to avoid numerical issues
    X_ = np.nan_to_num(X_proj, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        sem_acc = cross_val_score(clf, X_, sem_labels, cv=cv_, scoring="accuracy").mean()
    except Exception:
        sem_acc = 0.0
    try:
        surf_acc = cross_val_score(clf, X_, surf_labels, cv=cv_, scoring="accuracy").mean()
    except Exception:
        surf_acc = 0.0
    return float(sem_acc - surf_acc)


def build_semantic_reps(acts: np.ndarray,
                        sem_labels: np.ndarray,
                        surf_labels: np.ndarray,
                        layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a given layer, return:
      X_proj   — projected into semantic subspace
      X_res    — nuisance-residualised
      P_sem    — semantic projector
    """
    X = acts[:, layer, :]
    P_nuis = learn_nuisance_projector(X, surf_labels)
    P_sem, _ = learn_semantic_projector(X, sem_labels)
    X_proj = project_semantic(X, P_sem)
    X_res = residualise_nuisance(X, P_nuis)
    return X_proj, X_res, P_sem
