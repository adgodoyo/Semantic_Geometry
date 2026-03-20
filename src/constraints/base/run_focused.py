"""
Focused analysis — MLE ID and log-vol as primary metrics.

Three tracks for each model (BERT-base, Pythia-2.8B):

  A. Static geometry at best semantic layer
     4 branches: raw | sem-projected | residualised | whitened-sem
     Per chain × level (0-4) → level trajectory + Δ(L4-L0)

  B. Layer trajectory
     At every layer ℓ: project activations into fixed P_sem (from best layer)
     → MLE ID and log-vol vs depth, refinement vs control

  C. Semantic flow (the residualised stream)
     Δh_sem(ℓ) = P_sem · (h_{ℓ+1} − h_ℓ)
     Stack all layer-transition displacement vectors per chain × level
     → MLE ID / log-vol of the update field, refinement vs control
     + cumulative MLE ID as transitions are stacked

All results saved to results/focused/
"""

import sys, os, json, logging
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

sys.path.insert(0, "src")
from data_loader import load_dataset, load_control_dataset
from subspace import (learn_semantic_projector, learn_nuisance_projector,
                      project_semantic, residualise_nuisance)
from whitening import zca_whiten
from geometry import mle_intrinsic_dim, log_volume, participation_ratio
from config import (ALL_CHAINS, CONTROL_CHAINS, N_LEVELS,
                    FULL_DIR, CONTROL_BERT_DIR, CONTROL_PYTHIA_DIR,
                    RESULTS_DIR)

# ── output dir ────────────────────────────────────────────────────────────────
FOCUSED_DIR = os.path.join(RESULTS_DIR, "focused")
os.makedirs(FOCUSED_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(FOCUSED_DIR, "run_focused.log"), mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── best semantic layers (from full run) ─────────────────────────────────────
BEST_LAYERS = {"bert": 10, "pythia": 15}

MODELS = {
    "bert": {
        "label":        "BERT-base",
        "ref_acts":     os.path.join(FULL_DIR, "acts_full.npy"),
        "ctrl_acts":    os.path.join(CONTROL_BERT_DIR, "acts_control_bert.npy"),
        "best_layer":   10,
    },
    "pythia": {
        "label":        "Pythia-2.8B",
        "ref_acts":     os.path.join(RESULTS_DIR, "full_pythia", "acts_full_pythia.npy"),
        "ctrl_acts":    os.path.join(CONTROL_PYTHIA_DIR, "acts_control_pythia.npy"),
        "best_layer":   15,
    },
}

BRANCHES = ["raw", "sem", "res", "whi"]
BRANCH_LABELS = {"raw": "Raw", "sem": "Sem-projected",
                 "res": "Residualised", "whi": "Whitened-sem"}
METRICS = ["mle_id", "log_vol"]
METRIC_LABELS = {"mle_id": "MLE Intrinsic Dim", "log_vol": "Log-Vol"}

# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_metrics(X: np.ndarray) -> dict:
    if X.shape[0] < 7:
        return {"mle_id": np.nan, "log_vol": np.nan}
    return {"mle_id": mle_intrinsic_dim(X), "log_vol": log_volume(X)}


def _build_branches(acts_all: np.ndarray, df, chains: list[str],
                    best_layer: int, P_sem: np.ndarray, P_nuis: np.ndarray,
                    whi_transform) -> dict:
    """Return {branch: {chain: [metrics_L0, …, metrics_L4]}}."""
    results = {b: {c: [] for c in chains} for b in BRANCHES}
    for chain in chains:
        for level in range(N_LEVELS):
            mask = (df["chain_id"] == chain) & (df["level"] == level)
            X = acts_all[mask.values, best_layer, :]          # (8, D)
            X_sem = X @ P_sem.T                               # (8, D_sem)
            X_res = residualise_nuisance(X, P_nuis)           # (8, D)
            X_whi = whi_transform(X_sem)                      # (8, D_sem)
            results["raw"][chain].append(_safe_metrics(X))
            results["sem"][chain].append(_safe_metrics(X_sem))
            results["res"][chain].append(_safe_metrics(X_res))
            results["whi"][chain].append(_safe_metrics(X_whi))
    return results


def _layer_trajectory(acts_all: np.ndarray, df, chains: list[str],
                      n_layers: int, P_sem: np.ndarray) -> dict:
    """Return {chain: (n_layers, N_LEVELS, 2)} — [mle_id, log_vol] per layer×level."""
    out = {}
    for chain in chains:
        mat = np.full((n_layers, N_LEVELS, 2), np.nan)
        for layer in range(n_layers):
            for level in range(N_LEVELS):
                mask = (df["chain_id"] == chain) & (df["level"] == level)
                X_sem = acts_all[mask.values, layer, :] @ P_sem.T
                m = _safe_metrics(X_sem)
                mat[layer, level, 0] = m["mle_id"]
                mat[layer, level, 1] = m["log_vol"]
        out[chain] = mat
    return out


def _semantic_flow(acts_all: np.ndarray, df, chains: list[str],
                   n_layers: int, P_sem: np.ndarray) -> dict:
    """
    Δh_sem(ℓ) = (h_{ℓ+1} − h_ℓ) @ P_sem.T
    Returns:
      flow_per_trans : {chain: (n_trans, N_LEVELS, 2)}   per-transition metrics
      cumulative     : {chain: (n_trans, N_LEVELS, 2)}   cumulative stack metrics
    """
    n_trans = n_layers - 1
    flow_pt = {}
    cumulative = {}
    for chain in chains:
        ft  = np.full((n_trans, N_LEVELS, 2), np.nan)
        cum = np.full((n_trans, N_LEVELS, 2), np.nan)
        for level in range(N_LEVELS):
            mask = (df["chain_id"] == chain) & (df["level"] == level)
            X = acts_all[mask.values, :, :]        # (N_sent, n_layers, D)
            deltas = np.diff(X, axis=1)            # (N_sent, n_trans, D)
            # Project into semantic subspace
            dsem = deltas @ P_sem.T                # (N_sent, n_trans, D_sem)
            stacked = []
            for t in range(n_trans):
                batch = dsem[:, t, :]              # (N_sent, D_sem)
                m = _safe_metrics(batch)
                ft[t, level, 0] = m["mle_id"]
                ft[t, level, 1] = m["log_vol"]
                stacked.append(batch)
                # cumulative: stack all transitions 0..t
                cum_X = np.concatenate(stacked, axis=0)  # ((t+1)*N, D_sem)
                mc = _safe_metrics(cum_X)
                cum[t, level, 0] = mc["mle_id"]
                cum[t, level, 1] = mc["log_vol"]
        flow_pt[chain]  = ft
        cumulative[chain] = cum
    return flow_pt, cumulative


# ── plotting helpers ──────────────────────────────────────────────────────────

_COL = {"ref": "#2196F3", "ctrl": "#FF5722"}
_LS  = {"ref": "-", "ctrl": "--"}

def _mw_label(a, b):
    """Mann-Whitney U p-value string."""
    if len(a) < 3 or len(b) < 3:
        return ""
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return f"p={p:.2f}"


def plot_static(ref_branches, ctrl_branches, model_label, save_path):
    """4 branches × 2 metrics — level trajectories + Δ bar chart."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey="row")
    fig.suptitle(f"{model_label} — Static geometry at best semantic layer", fontsize=13)
    levels = list(range(N_LEVELS))

    for col, branch in enumerate(BRANCHES):
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            ref_vals = np.array([[ref_branches[branch][c][l][metric] for l in levels]
                                  for c in ALL_CHAINS])      # (10, 5)
            ctrl_vals = np.array([[ctrl_branches[branch][c][l][metric] for l in levels]
                                   for c in CONTROL_CHAINS]) # (10, 5)
            for vals, key in [(ref_vals, "ref"), (ctrl_vals, "ctrl")]:
                mu = np.nanmean(vals, axis=0)
                se = np.nanstd(vals, axis=0) / np.sqrt(vals.shape[0])
                ax.plot(levels, mu, color=_COL[key], ls=_LS[key],
                        lw=2, label=("Refinement" if key == "ref" else "Control"))
                ax.fill_between(levels, mu - se, mu + se,
                                alpha=0.15, color=_COL[key])
            if row == 0:
                ax.set_title(BRANCH_LABELS[branch], fontsize=10)
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            ax.set_xlabel("Level" if row == 1 else "")
            ax.set_xticks(levels)
            if row == 0 and col == 3:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  → {save_path}")


def plot_delta_summary(ref_branches, ctrl_branches, model_label, save_path):
    """Δ(L4-L0) bar chart: 4 branches × 2 metrics, refinement vs control + significance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_label} — Δ(L4−L0) by branch", fontsize=13)
    x = np.arange(len(BRANCHES))
    width = 0.35

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        ref_deltas_by_branch = []
        ctrl_deltas_by_branch = []
        for branch in BRANCHES:
            ref_d = np.array([ref_branches[branch][c][4][metric] -
                               ref_branches[branch][c][0][metric]
                               for c in ALL_CHAINS])
            ctrl_d = np.array([ctrl_branches[branch][c][4][metric] -
                                ctrl_branches[branch][c][0][metric]
                                for c in CONTROL_CHAINS])
            ref_deltas_by_branch.append(ref_d)
            ctrl_deltas_by_branch.append(ctrl_d)

        for i, (ref_d, ctrl_d) in enumerate(zip(ref_deltas_by_branch, ctrl_deltas_by_branch)):
            r_mu, r_se = np.nanmean(ref_d), np.nanstd(ref_d) / np.sqrt(len(ref_d))
            c_mu, c_se = np.nanmean(ctrl_d), np.nanstd(ctrl_d) / np.sqrt(len(ctrl_d))
            ax.bar(x[i] - width/2, r_mu, width, color=_COL["ref"],
                   yerr=r_se, capsize=4, alpha=0.85)
            ax.bar(x[i] + width/2, c_mu, width, color=_COL["ctrl"],
                   yerr=c_se, capsize=4, alpha=0.85)
            label = _mw_label(ref_d[~np.isnan(ref_d)], ctrl_d[~np.isnan(ctrl_d)])
            if label:
                y_top = max(abs(r_mu) + r_se, abs(c_mu) + c_se) + 0.05
                ax.text(x[i], y_top if r_mu >= 0 else -y_top,
                        label, ha="center", va="bottom", fontsize=9)

        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([BRANCH_LABELS[b] for b in BRANCHES], rotation=15, fontsize=9)
        ax.set_ylabel(f"Δ {METRIC_LABELS[metric]}")
        ax.set_title(METRIC_LABELS[metric])
        from matplotlib.patches import Patch
        if ax_idx == 1:
            ax.legend(handles=[Patch(color=_COL["ref"], label="Refinement"),
                                Patch(color=_COL["ctrl"], label="Control")],
                      fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  → {save_path}")


def plot_layer_trajectory(ref_traj, ctrl_traj, model_label, n_layers, save_path):
    """MLE ID and log-vol vs layer depth, one line per level (refinement solid, control dashed)."""
    fig, axes = plt.subplots(2, N_LEVELS, figsize=(18, 7), sharey="row")
    fig.suptitle(f"{model_label} — Layer trajectory (semantic subspace)", fontsize=13)
    layers = list(range(n_layers))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, N_LEVELS))

    for level in range(N_LEVELS):
        for row, metric_idx in enumerate([0, 1]):
            ax = axes[row, level]
            # refinement
            ref_mat = np.array([ref_traj[c][:, level, metric_idx]
                                 for c in ALL_CHAINS])     # (10, n_layers)
            mu_r = np.nanmean(ref_mat, axis=0)
            se_r = np.nanstd(ref_mat, axis=0) / np.sqrt(ref_mat.shape[0])
            ax.plot(layers, mu_r, color=cmap[level], lw=2,
                    label=f"Ref L{level}")
            ax.fill_between(layers, mu_r - se_r, mu_r + se_r,
                            alpha=0.2, color=cmap[level])
            # control
            ctrl_mat = np.array([ctrl_traj[c][:, level, metric_idx]
                                  for c in CONTROL_CHAINS])
            mu_c = np.nanmean(ctrl_mat, axis=0)
            se_c = np.nanstd(ctrl_mat, axis=0) / np.sqrt(ctrl_mat.shape[0])
            ax.plot(layers, mu_c, color=cmap[level], lw=2, ls="--",
                    label=f"Ctrl L{level}")
            ax.fill_between(layers, mu_c - se_c, mu_c + se_c,
                            alpha=0.1, color=cmap[level])
            ax.set_title(f"Level {level}" if row == 0 else "")
            if level == 0:
                ax.set_ylabel(METRIC_LABELS[METRICS[row]], fontsize=9)
            ax.set_xlabel("Layer" if row == 1 else "")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  → {save_path}")


def _stacked_flow_pr_whitening(acts_all: np.ndarray, df, chains: list[str],
                                W_sem: np.ndarray, n_layers: int) -> dict:
    """
    For each (chain, level) stack all (n_trans × N_sent) flow vectors in 12-D
    semantic space and compute PR under three anisotropy controls:
      0 — raw PR
      1 — ZCA-whitened PR  (N=256 >> D=12, non-degenerate)
      2 — Top-5 deflated PR

    Returns {chain: ndarray (N_levels, 3)}.
    """
    n_trans = n_layers - 1
    D_sem   = W_sem.shape[1]
    reg     = 1e-4

    def _zca_pr(X):
        Xc = X - X.mean(0)
        C  = Xc.T @ Xc / max(len(Xc) - 1, 1)
        vals, vecs = np.linalg.eigh(C)
        pos = vals > 1e-12
        if not pos.any():
            return float("nan")
        s_inv = 1.0 / np.sqrt(vals[pos] + reg)
        Xw    = Xc @ vecs[:, pos] * s_inv
        return participation_ratio(Xw)

    def _deflate_pr(X, k=5):
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        Vk = Vt[:k].T
        return participation_ratio(Xc - (Xc @ Vk) @ Vk.T)

    out = {}
    for chain in chains:
        mask   = df["chain_id"] == chain
        acts_c = acts_all[mask.values]            # (40, n_layers, D)
        df_c   = df[mask].reset_index(drop=True)
        vals   = np.zeros((N_LEVELS, 3))

        for lvl in range(N_LEVELS):
            m  = df_c["level"] == lvl
            a  = acts_c[m.values]                 # (N_sent, n_layers, D)
            # stack all transitions
            rows = []
            for t in range(n_trans):
                delta = a[:, t + 1, :] - a[:, t, :]   # (N_sent, D)
                rows.append(delta @ W_sem)             # (N_sent, D_sem)
            X = np.concatenate(rows, axis=0)           # (n_trans * N_sent, D_sem)

            vals[lvl, 0] = participation_ratio(X)
            vals[lvl, 1] = _zca_pr(X)
            vals[lvl, 2] = _deflate_pr(X, k=min(5, D_sem - 1))

        out[chain] = vals
    return out


def plot_flow_pr_whitening(ref_pr: dict, ctrl_pr: dict,
                            ref_chains: list[str], ctrl_chains: list[str],
                            model_label: str, save_path: str):
    """2-panel: flow cloud PR under Raw and Top-5 deflated.
    ZCA is omitted from the figure (N>>D makes it degenerate at PR=D=12).
    """
    # indices: 0=raw, 2=deflate5  (1=ZCA excluded — always 12.0)
    panels = [(0, "Raw PR",
               "Control more isotropic (higher raw PR)\n→ updates spread in many directions"),
              (2, "PR after Top-5 deflation",
               "Refinement higher after removing 5 dominant directions\n→ richer residual structure (reversal)")]

    ref_mat  = np.array([ref_pr[c]  for c in ref_chains])    # (10, 5, 3)
    ctrl_mat = np.array([ctrl_pr[c] for c in ctrl_chains])
    levels   = list(range(N_LEVELS))
    xlabels  = ["L0 (base)", "L1", "L2", "L3", "L4 (full)"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"{model_label} — Flow Cloud PR: Anisotropy Controls\n"
                 "Stacked flow: 32 transitions × 8 sentences = 256 vectors per (chain, level)",
                 fontsize=12, fontweight="bold")

    for ax, (mi, panel_title, subtitle) in zip(axes, panels):
        ref_mean  = ref_mat[:, :, mi].mean(0);  ref_sd  = ref_mat[:, :, mi].std(0)
        ctrl_mean = ctrl_mat[:, :, mi].mean(0); ctrl_sd = ctrl_mat[:, :, mi].std(0)

        ax.fill_between(levels, ref_mean - ref_sd,  ref_mean + ref_sd,
                        alpha=0.18, color=_COL["ref"])
        ax.fill_between(levels, ctrl_mean - ctrl_sd, ctrl_mean + ctrl_sd,
                        alpha=0.18, color=_COL["ctrl"])
        for vals in ref_mat[:, :, mi]:
            ax.plot(levels, vals, "-", color=_COL["ref"],  lw=0.4, alpha=0.3)
        for vals in ctrl_mat[:, :, mi]:
            ax.plot(levels, vals, "-", color=_COL["ctrl"], lw=0.4, alpha=0.3)

        ax.plot(levels, ref_mean,  "o-",  color=_COL["ref"],  lw=2.5, ms=8,
                label="Refinement (independent)")
        ax.plot(levels, ctrl_mean, "s--", color=_COL["ctrl"], lw=2.5, ms=8,
                label="Control (redundant)")

        # significance stars with offset
        y_max = max((ref_mean + ref_sd).max(), (ctrl_mean + ctrl_sd).max())
        y_range = y_max - min((ref_mean - ref_sd).min(), (ctrl_mean - ctrl_sd).min())
        for l in levels:
            rv = ref_mat[:, l, mi]; cv = ctrl_mat[:, l, mi]
            star = _mw_label(rv, cv)
            if star and star.startswith("*"):
                y = max(ref_mean[l] + ref_sd[l], ctrl_mean[l] + ctrl_sd[l]) + y_range * 0.06
                ax.text(l, y, star, ha="center", va="bottom", fontsize=13, color="black")

        ax.set_title(panel_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(subtitle, fontsize=9, color="#444444")
        ax.set_ylabel("Participation Ratio", fontsize=11)
        ax.set_xticks(levels); ax.set_xticklabels(xlabels, fontsize=10)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  → {save_path}")


def plot_flow_pr_deflation_sweep(ref_pr_sweep: dict, ctrl_pr_sweep: dict,
                                  ref_chains: list[str], ctrl_chains: list[str],
                                  k_values: list[int], model_label: str,
                                  save_path: str):
    """
    Multi-panel figure: one panel per k in k_values.
    Each panel shows flow cloud PR (after deflating k PCs) vs semantic level,
    refinement vs control.  Lets the reader see where the raw→reversal transition
    happens and how stable the separation is across k.
    """
    import seaborn as sns
    from scipy.stats import mannwhitneyu as _mwu

    n_k   = len(k_values)
    ncols = min(4, n_k)
    nrows = (n_k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                              squeeze=False)
    fig.suptitle(f"{model_label} — Flow Cloud PR: Deflation Sensitivity (k = 0 … {k_values[-1]})\n"
                 "k = 0 is raw PR; each k removes the k largest shared directions",
                 fontsize=12, fontweight="bold")

    levels  = list(range(N_LEVELS))
    xlbls   = ["L0", "L1", "L2", "L3", "L4"]

    for idx, k in enumerate(k_values):
        ax   = axes[idx // ncols][idx % ncols]
        r_mat  = np.array([ref_pr_sweep[c][:, idx]  for c in ref_chains])   # (10, 5)
        c_mat  = np.array([ctrl_pr_sweep[c][:, idx] for c in ctrl_chains])

        r_mean = r_mat.mean(0); r_sd = r_mat.std(0)
        c_mean = c_mat.mean(0); c_sd = c_mat.std(0)

        ax.fill_between(levels, r_mean - r_sd, r_mean + r_sd, alpha=0.15, color=_COL["ref"])
        ax.fill_between(levels, c_mean - c_sd, c_mean + c_sd, alpha=0.15, color=_COL["ctrl"])
        for v in r_mat: ax.plot(levels, v, "-", color=_COL["ref"],  lw=0.35, alpha=0.25)
        for v in c_mat: ax.plot(levels, v, "-", color=_COL["ctrl"], lw=0.35, alpha=0.25)
        ax.plot(levels, r_mean, "o-",  color=_COL["ref"],  lw=2.2, ms=6,
                label="Ref" if idx == 0 else "")
        ax.plot(levels, c_mean, "s--", color=_COL["ctrl"], lw=2.2, ms=6,
                label="Ctrl" if idx == 0 else "")

        # significance stars
        y_range = max((r_mean+r_sd).max(), (c_mean+c_sd).max()) - \
                  min((r_mean-r_sd).min(), (c_mean-c_sd).min())
        for l in levels:
            _, p = _mwu(r_mat[:, l], c_mat[:, l], alternative="two-sided")
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
            if sig:
                y = max(r_mean[l]+r_sd[l], c_mean[l]+c_sd[l]) + y_range*0.07
                ax.text(l, y, sig, ha="center", va="bottom", fontsize=11)

        # direction label: ref > ctrl or ctrl > ref at L4
        direction = "Ref > Ctrl" if r_mean[-1] > c_mean[-1] else "Ctrl > Ref"
        ax.set_title(f"k = {k}  ({direction})", fontsize=11, fontweight="bold")
        ax.set_xticks(levels); ax.set_xticklabels(xlbls, fontsize=9)
        ax.set_ylabel("PR" if idx % ncols == 0 else "")
        ax.grid(True, alpha=0.3)

    # legend in first panel
    axes[0][0].legend(fontsize=10)
    # hide unused axes
    for idx in range(n_k, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  → {save_path}")


def plot_flow(ref_flow_pt, ctrl_flow_pt, ref_cum, ctrl_cum,
              model_label, n_layers, save_path):
    """Semantic flow: per-transition + cumulative MLE ID and log-vol."""
    n_trans = n_layers - 1
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{model_label} — Semantic flow (Δh projected into P_sem)", fontsize=13)
    titles = ["Per-transition MLE ID", "Per-transition Log-Vol",
              "Cumulative MLE ID", "Cumulative Log-Vol"]
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, N_LEVELS))
    trans = list(range(n_trans))

    for panel, (flow_ref, flow_ctrl, metric_idx, title) in enumerate([
        (ref_flow_pt,  ctrl_flow_pt,  0, titles[0]),
        (ref_flow_pt,  ctrl_flow_pt,  1, titles[1]),
        (ref_cum,      ctrl_cum,      0, titles[2]),
        (ref_cum,      ctrl_cum,      1, titles[3]),
    ]):
        ax = axes[panel // 2, panel % 2]
        ax.set_title(title, fontsize=10)
        for level in range(N_LEVELS):
            ref_mat = np.array([flow_ref[c][:, level, metric_idx]
                                 for c in ALL_CHAINS])
            ctrl_mat = np.array([flow_ctrl[c][:, level, metric_idx]
                                  for c in CONTROL_CHAINS])
            mu_r = np.nanmean(ref_mat, axis=0)
            mu_c = np.nanmean(ctrl_mat, axis=0)
            ax.plot(trans, mu_r, color=cmap[level], lw=1.5,
                    label=f"Ref L{level}")
            ax.plot(trans, mu_c, color=cmap[level], lw=1.5, ls="--",
                    label=f"Ctrl L{level}")
        ax.set_xlabel("Layer transition")
        ax.set_ylabel("MLE ID" if metric_idx == 0 else "Log-Vol")
        ax.grid(True, alpha=0.3)
        if panel == 0:
            # legend for ref vs ctrl only
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0], [0], color="grey", lw=2, label="Refinement"),
                Line2D([0], [0], color="grey", lw=2, ls="--", label="Control"),
            ], fontsize=8, loc="upper right")
        # colorbar-style level annotation
        sm = plt.cm.ScalarMappable(cmap="plasma",
                                   norm=plt.Normalize(vmin=0, vmax=N_LEVELS-1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Semantic level", shrink=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  → {save_path}")


# ── serialise results ─────────────────────────────────────────────────────────

def _to_json(obj):
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


# ── main ──────────────────────────────────────────────────────────────────────

def run_model(key: str):
    cfg = MODELS[key]
    model_label = cfg["label"]
    best_layer  = cfg["best_layer"]
    log.info("=" * 60)
    log.info(f"MODEL: {model_label}  (best_layer=L{best_layer})")
    log.info("=" * 60)

    # Load activations
    acts_ref  = np.load(cfg["ref_acts"],  mmap_mode="r").astype(np.float32)
    acts_ctrl = np.load(cfg["ctrl_acts"], mmap_mode="r").astype(np.float32)
    n_layers  = acts_ref.shape[1]
    log.info(f"  acts_ref  shape: {acts_ref.shape}")
    log.info(f"  acts_ctrl shape: {acts_ctrl.shape}")

    # Load DataFrames
    df_ref  = load_dataset()
    df_ctrl = load_control_dataset()

    # ── fit projectors at best layer on EACH condition's own data ─────────────
    X_ref_best  = acts_ref[:, best_layer, :]
    X_ctrl_best = acts_ctrl[:, best_layer, :]

    log.info("  Fitting projectors (refinement) …")
    P_sem_ref,  W_sem_ref  = learn_semantic_projector(
        X_ref_best, df_ref["chain_level_id"].values)
    P_nuis_ref    = learn_nuisance_projector(
        X_ref_best, df_ref["surface_id"].values)

    log.info("  Fitting projectors (control) …")
    P_sem_ctrl, W_sem_ctrl = learn_semantic_projector(
        X_ctrl_best, df_ctrl["chain_level_id"].values)
    P_nuis_ctrl   = learn_nuisance_projector(
        X_ctrl_best, df_ctrl["surface_id"].values)

    # ── ZCA transforms (fit on global semantic projections, applied to subsets) ─
    log.info("  Building ZCA transforms …")
    from whitening import _cov_eig
    from config import WHITEN_REGULARIZE

    def _make_whi(acts_all, P_sem):
        """Pre-compute ZCA on full dataset; return closure that applies it to subsets."""
        X_sem_all = (acts_all[:, best_layer, :] @ P_sem.T).astype(np.float32)
        mu = X_sem_all.mean(0, keepdims=True)
        Xc = X_sem_all - mu
        V, eigs = _cov_eig(Xc)                                    # fit on N=400
        s_inv_sqrt = 1.0 / np.sqrt(eigs + WHITEN_REGULARIZE)      # (k,)
        # W = V diag(s_inv_sqrt) V.T   (D×D whitening matrix)
        # Apply factored: (X - mu) @ V * s_inv_sqrt @ V.T
        def _whi(X):
            Xc_sub = X.astype(np.float32) - mu
            proj   = Xc_sub @ V                         # (N, k)
            return (proj * s_inv_sqrt) @ V.T            # (N, D_sem)
        return _whi

    whi_ref  = _make_whi(acts_ref,  P_sem_ref)
    whi_ctrl = _make_whi(acts_ctrl, P_sem_ctrl)

    # ── Part A: Static geometry ───────────────────────────────────────────────
    log.info("  Part A — static geometry …")
    ref_branches  = _build_branches(acts_ref,  df_ref,  ALL_CHAINS,
                                     best_layer, P_sem_ref,  P_nuis_ref,  whi_ref)
    ctrl_branches = _build_branches(acts_ctrl, df_ctrl, CONTROL_CHAINS,
                                     best_layer, P_sem_ctrl, P_nuis_ctrl, whi_ctrl)

    # ── Part B: Layer trajectory ──────────────────────────────────────────────
    log.info("  Part B — layer trajectory …")
    ref_traj  = _layer_trajectory(acts_ref,  df_ref,  ALL_CHAINS,     n_layers, P_sem_ref)
    ctrl_traj = _layer_trajectory(acts_ctrl, df_ctrl, CONTROL_CHAINS, n_layers, P_sem_ctrl)

    # ── Part C: Semantic flow ─────────────────────────────────────────────────
    log.info("  Part C — semantic flow …")
    ref_flow_pt,  ref_cum  = _semantic_flow(
        acts_ref,  df_ref,  ALL_CHAINS,     n_layers, P_sem_ref)
    ctrl_flow_pt, ctrl_cum = _semantic_flow(
        acts_ctrl, df_ctrl, CONTROL_CHAINS, n_layers, P_sem_ctrl)

    # ── Part D: Stacked flow cloud PR under whitening ─────────────────────────
    log.info("  Part D — stacked flow cloud PR (raw / ZCA / top-5) …")
    ref_flow_pr_whi  = _stacked_flow_pr_whitening(
        acts_ref,  df_ref,  ALL_CHAINS,     W_sem_ref,  n_layers)
    ctrl_flow_pr_whi = _stacked_flow_pr_whitening(
        acts_ctrl, df_ctrl, CONTROL_CHAINS, W_sem_ctrl, n_layers)

    # ── Summary stats ─────────────────────────────────────────────────────────
    log.info(f"\n  {'Branch':<14}  {'Δ MLE ref':>10}  {'Δ MLE ctrl':>10}  "
             f"{'Δ logvol ref':>12}  {'Δ logvol ctrl':>13}  {'sig MLE':>8}  {'sig lv':>8}")
    for branch in BRANCHES:
        ref_mle  = np.array([ref_branches[branch][c][4]["mle_id"] -
                              ref_branches[branch][c][0]["mle_id"]  for c in ALL_CHAINS])
        ctrl_mle = np.array([ctrl_branches[branch][c][4]["mle_id"] -
                              ctrl_branches[branch][c][0]["mle_id"] for c in CONTROL_CHAINS])
        ref_lv   = np.array([ref_branches[branch][c][4]["log_vol"] -
                              ref_branches[branch][c][0]["log_vol"] for c in ALL_CHAINS])
        ctrl_lv  = np.array([ctrl_branches[branch][c][4]["log_vol"] -
                              ctrl_branches[branch][c][0]["log_vol"] for c in CONTROL_CHAINS])
        sig_mle  = _mw_label(ref_mle[~np.isnan(ref_mle)], ctrl_mle[~np.isnan(ctrl_mle)])
        sig_lv   = _mw_label(ref_lv[~np.isnan(ref_lv)],   ctrl_lv[~np.isnan(ctrl_lv)])
        log.info(f"  {BRANCH_LABELS[branch]:<14}  "
                 f"{np.nanmean(ref_mle):+.3f}±{np.nanstd(ref_mle):.3f}  "
                 f"{np.nanmean(ctrl_mle):+.3f}±{np.nanstd(ctrl_mle):.3f}  "
                 f"{np.nanmean(ref_lv):+.3f}±{np.nanstd(ref_lv):.3f}  "
                 f"{np.nanmean(ctrl_lv):+.3f}±{np.nanstd(ctrl_lv):.3f}  "
                 f"{sig_mle:>8}  {sig_lv:>8}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    tag = key
    plot_static(ref_branches, ctrl_branches, model_label,
                os.path.join(FOCUSED_DIR, f"static_{tag}.png"))
    plot_delta_summary(ref_branches, ctrl_branches, model_label,
                       os.path.join(FOCUSED_DIR, f"delta_{tag}.png"))
    plot_layer_trajectory(ref_traj, ctrl_traj, model_label, n_layers,
                          os.path.join(FOCUSED_DIR, f"layer_traj_{tag}.png"))
    plot_flow(ref_flow_pt, ctrl_flow_pt, ref_cum, ctrl_cum, model_label, n_layers,
              os.path.join(FOCUSED_DIR, f"flow_{tag}.png"))
    plot_flow_pr_whitening(ref_flow_pr_whi, ctrl_flow_pr_whi,
                           ALL_CHAINS, CONTROL_CHAINS, model_label,
                           os.path.join(FOCUSED_DIR, f"flow_pr_whitening_{tag}.png"))

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "model": model_label, "best_layer": best_layer,
        "static_ref":   ref_branches,
        "static_ctrl":  ctrl_branches,
        "layer_traj_ref":  {c: ref_traj[c].tolist()  for c in ALL_CHAINS},
        "layer_traj_ctrl": {c: ctrl_traj[c].tolist() for c in CONTROL_CHAINS},
        "flow_ref":        {c: ref_flow_pt[c].tolist()  for c in ALL_CHAINS},
        "flow_ctrl":       {c: ctrl_flow_pt[c].tolist() for c in CONTROL_CHAINS},
        "cum_flow_ref":    {c: ref_cum[c].tolist()  for c in ALL_CHAINS},
        "cum_flow_ctrl":   {c: ctrl_cum[c].tolist() for c in CONTROL_CHAINS},
        "flow_pr_whi_ref":  {c: ref_flow_pr_whi[c].tolist()  for c in ALL_CHAINS},
        "flow_pr_whi_ctrl": {c: ctrl_flow_pr_whi[c].tolist() for c in CONTROL_CHAINS},
    }
    json_path = os.path.join(FOCUSED_DIR, f"focused_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(_to_json(out), f, indent=2)
    log.info(f"  → {json_path}")


if __name__ == "__main__":
    for model_key in ["bert", "pythia"]:
        run_model(model_key)
    log.info("FOCUSED ANALYSIS COMPLETE")
