"""
Flow analysis supplement — all §7 figures not produced by run_focused.py.

Produces (saved to results/focused/):
  flow_mle_comparison.png       — per-transition flow MLE by semantic level
  flow_mle_delta_bars.png       — per-chain Δ flow MLE (L4-L0)
  flow_magnitude_comparison.png — mean ‖Δh‖ per layer transition, both conditions
  flow_stacked_cloud.png        — stacked cloud MLE ID and log-volume
  flow_pr_whitening_pythia.png  — stacked cloud PR: raw vs top-5 deflated
  flow_pr_deflation_sweep.png   — PR sensitivity across k=5..10

Also saves results/focused/flow_analysis_pythia.json with all numerical data.

Run with:
    python3 run_flow_analysis.py

Prerequisites:
    results/full_pythia/acts_full_pythia.npy
    results/control_pythia/acts_control_pythia.npy
    (produced by run_full.py)
"""

import sys, os, json, logging
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, "src")
from data_loader import load_dataset, load_control_dataset
from subspace import learn_semantic_projector
from geometry import mle_intrinsic_dim, log_volume, participation_ratio
from config import ALL_CHAINS, CONTROL_CHAINS, N_LEVELS, RESULTS_DIR

FOCUSED_DIR = os.path.join(RESULTS_DIR, "focused")
os.makedirs(FOCUSED_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(FOCUSED_DIR, "run_flow_analysis.log"), mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.1)

BEST_LAYER = 15
COL = {"ref": "#2d6a9f", "ctrl": "#e07b2d"}

# ── helpers ───────────────────────────────────────────────────────────────────

def _mw_sig(a, b):
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""), p


def _save(fig, name):
    path = os.path.join(FOCUSED_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  → {path}")
    return path


def _top_k_deflate(X, k):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vk = Vt[:k].T
    return Xc - (Xc @ Vk) @ Vk.T


# ── data loading ──────────────────────────────────────────────────────────────

def load_all():
    log.info("Loading activations ...")
    acts_ref  = np.load(os.path.join(RESULTS_DIR, "full_pythia", "acts_full_pythia.npy")).astype(np.float32)
    acts_ctrl = np.load(os.path.join(RESULTS_DIR, "control_pythia", "acts_control_pythia.npy")).astype(np.float32)
    log.info(f"  ref  {acts_ref.shape}  ctrl {acts_ctrl.shape}")

    df_ref  = load_dataset()
    df_ctrl = load_control_dataset()

    log.info("Learning semantic projectors ...")
    _, W_ref  = learn_semantic_projector(acts_ref[:,  BEST_LAYER, :],
                                         df_ref["chain_level_id"].values)
    _, W_ctrl = learn_semantic_projector(acts_ctrl[:, BEST_LAYER, :],
                                         df_ctrl["chain_level_id"].values)
    log.info(f"  W_sem shape: {W_ref.shape}")
    return acts_ref, acts_ctrl, df_ref, df_ctrl, W_ref, W_ctrl


# ── compute flow vectors ──────────────────────────────────────────────────────

def compute_flow_vectors(acts, df, chains, W_sem):
    """
    Returns:
      flow  : {chain: (N_levels, n_trans, N_sent, D_sem)}
      mag   : {chain: (n_trans,)}   mean raw ‖Δh‖ over levels and sentences
    """
    n_trans = acts.shape[1] - 1
    D_sem   = W_sem.shape[1]
    flow, mag = {}, {}
    for chain in chains:
        mask   = df["chain_id"] == chain
        acts_c = acts[mask.values]
        df_c   = df[mask].reset_index(drop=True)
        lf     = np.zeros((N_LEVELS, n_trans, 8, D_sem))
        m_arr  = np.zeros(n_trans)
        for lvl in range(N_LEVELS):
            m = df_c["level"] == lvl
            a = acts_c[m.values]                          # (8, n_layers, D)
            for t in range(n_trans):
                delta      = a[:, t + 1, :] - a[:, t, :]  # (8, D)
                m_arr[t]  += np.linalg.norm(delta, axis=1).mean() / N_LEVELS
                lf[lvl, t] = delta @ W_sem                 # (8, D_sem)
        flow[chain] = lf
        mag[chain]  = m_arr
    return flow, mag


# ── per-transition flow MLE ───────────────────────────────────────────────────

def compute_pertrans_mle(flow, chains):
    """Mean MLE over all 32 transitions per (chain, level). Returns (10, 5)."""
    result = {}
    for chain in chains:
        arr = flow[chain]           # (5, 32, 8, D_sem)
        result[chain] = np.array([
            [mle_intrinsic_dim(arr[lvl, t]) for lvl in range(N_LEVELS)]
            for t in range(arr.shape[1])
        ]).mean(axis=0)             # mean over transitions → (5,)
    return result                   # {chain: (5,)}


def plot_pertrans_mle(ref_mle, ctrl_mle, ref_chains, ctrl_chains):
    """flow_mle_comparison.png and flow_mle_delta_bars.png"""
    ref_mat  = np.array([ref_mle[c]  for c in ref_chains])    # (10, 5)
    ctrl_mat = np.array([ctrl_mle[c] for c in ctrl_chains])
    levels   = np.arange(N_LEVELS)
    xlabels  = ["L0\n(base)", "L1", "L2", "L3", "L4\n(full)"]

    # ── Figure 1: trajectory ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, mat, lbl, ls in [("ref", ref_mat, "Refinement (independent)", "-"),
                                 ("ctrl", ctrl_mat, "Control (redundant)", "--")]:
        mu = mat.mean(0); sd = mat.std(0)
        ax.fill_between(levels, mu - sd, mu + sd, alpha=0.15, color=COL[cond])
        for v in mat:
            ax.plot(levels, v, ls, color=COL[cond], lw=0.4, alpha=0.3)
        ax.plot(levels, mu, f"o{ls}", color=COL[cond], lw=2.2, ms=7, label=lbl)

    y_range = (ref_mat.max() - ref_mat.min()) * 0.15
    for l in levels:
        sig, _ = _mw_sig(ref_mat[:, l], ctrl_mat[:, l])
        if sig:
            y = max(ref_mat[:, l].mean() + ref_mat[:, l].std(),
                    ctrl_mat[:, l].mean() + ctrl_mat[:, l].std()) + y_range * 0.3
            ax.text(l, y, sig, ha="center", fontsize=12)

    ax.set_xlabel("Semantic constraint level (L0 = base, L4 = fully constrained)", fontsize=11)
    ax.set_ylabel("Flow MLE ID\n(mean over all 32 layer transitions)", fontsize=10)
    ax.set_title("Flow Field Complexity vs Semantic Level\n"
                 "Refinement updates converge; redundant-modifier updates fragment", fontsize=11)
    ax.set_xticks(levels); ax.set_xticklabels(xlabels)
    ax.legend(fontsize=10)
    _save(fig, "flow_mle_comparison.png")

    # ── Figure 2: per-chain delta bars ────────────────────────────────────────
    ref_d  = ref_mat[:, 4]  - ref_mat[:, 0]
    ctrl_d = ctrl_mat[:, 4] - ctrl_mat[:, 0]
    ref_names  = [c.replace("_", "\n") for c in ref_chains]
    ctrl_names = [c.replace("_common", "").replace("_", "\n") for c in ctrl_chains]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, deltas, names, cond, title in [
        (axes[0], ref_d,  ref_names,  "ref",  "Refinement — Δ Flow MLE ID (L4 − L0)"),
        (axes[1], ctrl_d, ctrl_names, "ctrl", "Control — Δ Flow MLE ID (L4 − L0)"),
    ]:
        colors = [COL[cond] if d < 0 else sns.desaturate(COL[cond], 0.5) for d in deltas]
        ax.bar(range(10), deltas, color=colors, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(deltas.mean(), color=COL[cond], lw=1.8, ls="--",
                   label=f"Mean = {deltas.mean():.2f}")
        ax.set_xticks(range(10)); ax.set_xticklabels(names, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Δ Flow MLE ID")
        ax.legend(fontsize=9)

    sig, p = _mw_sig(ref_d, ctrl_d)
    pooled = np.sqrt((ref_d.std() ** 2 + ctrl_d.std() ** 2) / 2)
    d_val  = (ref_d.mean() - ctrl_d.mean()) / pooled
    fig.suptitle(f"Δ Flow MLE ID: Independent vs Redundant Constraints\n"
                 f"Mann-Whitney {sig} (p={p:.4f}), Cohen's d = {d_val:.2f}", fontsize=12)
    fig.tight_layout()
    _save(fig, "flow_mle_delta_bars.png")

    return ref_mat, ctrl_mat


# ── flow magnitude comparison ─────────────────────────────────────────────────

def plot_flow_magnitude_comparison(mag_ref, mag_ctrl, ref_chains, ctrl_chains):
    """flow_magnitude_comparison.png — both conditions on same axes."""
    ref_mat  = np.array([mag_ref[c]  for c in ref_chains])    # (10, 32)
    ctrl_mat = np.array([mag_ctrl[c] for c in ctrl_chains])
    transitions = np.arange(32)

    fig, ax = plt.subplots(figsize=(12, 5))
    for cond, mat, lbl, ls in [("ref",  ref_mat,  "Refinement (independent)", "-"),
                                 ("ctrl", ctrl_mat, "Control (redundant)",      "--")]:
        mu = mat.mean(0); sd = mat.std(0)
        ax.fill_between(transitions, mu - sd, mu + sd, alpha=0.18, color=COL[cond])
        for v in mat:
            ax.plot(transitions, v, ls, color=COL[cond], lw=0.35, alpha=0.3)
        ax.plot(transitions, mu, ls, color=COL[cond], lw=2.3, label=lbl)

    ax.axvspan(9.5, 24.5, alpha=0.07, color="green")
    ymax = max((ref_mat + ref_mat.std(0)).max(), (ctrl_mat + ctrl_mat.std(0)).max())
    ax.text(17, ymax * 1.01, "Semantic zone (L10–L25)",
            ha="center", va="bottom", fontsize=9, color="darkgreen",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("Layer transition (ℓ → ℓ+1)", fontsize=11)
    ax.set_ylabel("Mean ‖Δh‖ over sentences", fontsize=11)
    ax.set_title("Flow Magnitude per Layer Transition — Refinement vs Control\n"
                 "Shaded band = ±1 SD across chains", fontsize=11)
    ax.set_xticks(np.arange(0, 32, 2))
    ax.set_xticklabels([f"{i}→{i+1}" for i in range(0, 32, 2)], rotation=45, fontsize=7.5)
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    _save(fig, "flow_magnitude_comparison.png")


# ── stacked flow cloud geometry ───────────────────────────────────────────────

def compute_stacked_cloud(flow, chains, k_deflate=5):
    """
    For each (chain, level): stack all (32 × 8) flow vectors.
    Returns {chain: (N_levels, 4)} — [mle_id, pr_raw, logvol, pr_deflated].
    """
    out = {}
    for chain in chains:
        arr  = flow[chain]          # (5, 32, 8, D_sem)
        vals = np.zeros((N_LEVELS, 4))
        for lvl in range(N_LEVELS):
            X = arr[lvl].reshape(-1, arr.shape[-1])    # (256, D_sem)
            vals[lvl, 0] = mle_intrinsic_dim(X, k=10)
            vals[lvl, 1] = participation_ratio(X)
            vals[lvl, 2] = log_volume(X)
            k = min(k_deflate, X.shape[1] - 1)
            vals[lvl, 3] = participation_ratio(_top_k_deflate(X, k))
        out[chain] = vals
    return out


def plot_stacked_cloud(ref_sc, ctrl_sc, ref_chains, ctrl_chains):
    """flow_stacked_cloud.png — MLE ID and logVol side by side."""
    ref_mat  = np.array([ref_sc[c]  for c in ref_chains])
    ctrl_mat = np.array([ctrl_sc[c] for c in ctrl_chains])
    levels   = np.arange(N_LEVELS)
    xlabels  = ["L0\n(base)", "L1", "L2", "L3", "L4\n(full)"]

    panels = [
        (0, "MLE Intrinsic Dimensionality\nof stacked flow cloud",
         "Control spans more directions across full network depth"),
        (2, "Log-volume of stacked flow cloud\n(total update space)",
         "Refinement compresses more as constraints accumulate"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stacked Flow Cloud: All Transitions × All Sentences per (Chain, Level)",
                 fontsize=12, fontweight="bold")

    for ax, (mi, ylabel, subtitle) in zip(axes, panels):
        for cond, mat, lbl, ls in [("ref",  ref_mat,  "Refinement", "o-"),
                                    ("ctrl", ctrl_mat, "Control",    "s--")]:
            mu = mat[:, :, mi].mean(0); sd = mat[:, :, mi].std(0)
            ax.fill_between(levels, mu - sd, mu + sd, alpha=0.15, color=COL[cond])
            for v in mat[:, :, mi]:
                ax.plot(levels, v, "-", color=COL[cond], lw=0.4, alpha=0.3)
            ax.plot(levels, mu, ls, color=COL[cond], lw=2.2, ms=7, label=lbl)

        y_r = (max(ref_mat[:,:,mi].max(), ctrl_mat[:,:,mi].max()) -
               min(ref_mat[:,:,mi].min(), ctrl_mat[:,:,mi].min()))
        for l in levels:
            sig, _ = _mw_sig(ref_mat[:, l, mi], ctrl_mat[:, l, mi])
            if sig:
                y = max(ref_mat[:, l, mi].mean() + ref_mat[:, l, mi].std(),
                        ctrl_mat[:, l, mi].mean() + ctrl_mat[:, l, mi].std()) + y_r * 0.07
                ax.text(l, y, sig, ha="center", fontsize=12)

        ax.set_title(subtitle, fontsize=10, fontweight="bold")
        ax.set_xlabel(ylabel, fontsize=9, color="#444")
        ax.set_ylabel("Value")
        ax.set_xticks(levels); ax.set_xticklabels(xlabels)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "flow_stacked_cloud.png")
    return ref_mat, ctrl_mat


# ── PR whitening (raw + top-5) ────────────────────────────────────────────────

def compute_pr_whitening(flow, chains):
    """Returns {chain: (N_levels, 2)} — [pr_raw, pr_deflate5]."""
    out = {}
    for chain in chains:
        arr  = flow[chain]
        vals = np.zeros((N_LEVELS, 2))
        for lvl in range(N_LEVELS):
            X = arr[lvl].reshape(-1, arr.shape[-1])
            vals[lvl, 0] = participation_ratio(X)
            k = min(5, X.shape[1] - 1)
            vals[lvl, 1] = participation_ratio(_top_k_deflate(X, k))
        out[chain] = vals
    return out


def plot_pr_whitening(ref_pr, ctrl_pr, ref_chains, ctrl_chains):
    """flow_pr_whitening_pythia.png — 2-panel: raw PR and top-5 deflated PR."""
    ref_mat  = np.array([ref_pr[c]  for c in ref_chains])
    ctrl_mat = np.array([ctrl_pr[c] for c in ctrl_chains])
    levels   = np.arange(N_LEVELS)
    xlabels  = ["L0 (base)", "L1", "L2", "L3", "L4 (full)"]

    panels = [
        (0, "Raw PR",
         "Control more isotropic (higher raw PR)\n→ updates spread in many directions"),
        (1, "PR after Top-5 deflation",
         "Refinement higher after removing 5 dominant directions\n→ richer residual structure (reversal)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Pythia-2.8B — Flow Cloud PR: Anisotropy Controls\n"
                 "Stacked flow: 32 transitions × 8 sentences = 256 vectors per (chain, level)",
                 fontsize=12, fontweight="bold")

    for ax, (mi, panel_title, subtitle) in zip(axes, panels):
        for cond, mat, lbl, ls in [("ref",  ref_mat,  "Refinement (independent)", "o-"),
                                    ("ctrl", ctrl_mat, "Control (redundant)",      "s--")]:
            mu = mat[:, :, mi].mean(0); sd = mat[:, :, mi].std(0)
            ax.fill_between(levels, mu - sd, mu + sd, alpha=0.18, color=COL[cond])
            for v in mat[:, :, mi]:
                ax.plot(levels, v, "-", color=COL[cond], lw=0.4, alpha=0.3)
            ax.plot(levels, mu, ls, color=COL[cond], lw=2.5, ms=8, label=lbl)

        y_r = (max(ref_mat[:,:,mi].max(), ctrl_mat[:,:,mi].max()) -
               min(ref_mat[:,:,mi].min(), ctrl_mat[:,:,mi].min()))
        for l in levels:
            sig, _ = _mw_sig(ref_mat[:, l, mi], ctrl_mat[:, l, mi])
            if sig:
                y = max(ref_mat[:, l, mi].mean() + ref_mat[:, l, mi].std(),
                        ctrl_mat[:, l, mi].mean() + ctrl_mat[:, l, mi].std()) + y_r * 0.06
                ax.text(l, y, sig, ha="center", va="bottom", fontsize=13)

        ax.set_title(panel_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(subtitle, fontsize=9, color="#444444")
        ax.set_ylabel("Participation Ratio", fontsize=11)
        ax.set_xticks(levels); ax.set_xticklabels(xlabels, fontsize=10)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "flow_pr_whitening_pythia.png")
    return ref_mat, ctrl_mat


# ── deflation sensitivity sweep k = 5..10 ────────────────────────────────────

def compute_deflation_sweep(flow, chains, k_values):
    """Returns {chain: (N_levels, len(k_values))} — PR after deflating k PCs."""
    D_sem = list(flow.values())[0].shape[-1]
    out   = {}
    for chain in chains:
        arr  = flow[chain]
        vals = np.zeros((N_LEVELS, len(k_values)))
        for lvl in range(N_LEVELS):
            X  = arr[lvl].reshape(-1, D_sem)
            Xc = X - X.mean(0)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            for ki, k in enumerate(k_values):
                if k >= D_sem:
                    vals[lvl, ki] = float("nan")
                    continue
                Vk = Vt[:k].T
                vals[lvl, ki] = participation_ratio(Xc - (Xc @ Vk) @ Vk.T)
        out[chain] = vals
    return out


def plot_deflation_sweep(ref_sw, ctrl_sw, ref_chains, ctrl_chains, k_values):
    """flow_pr_deflation_sweep.png — one panel per k."""
    ref_mat  = np.array([ref_sw[c]  for c in ref_chains])
    ctrl_mat = np.array([ctrl_sw[c] for c in ctrl_chains])
    levels   = np.arange(N_LEVELS)
    xlabels  = ["L0", "L1", "L2", "L3", "L4"]

    ncols = min(4, len(k_values))
    nrows = (len(k_values) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"Pythia-2.8B — Flow Cloud PR: Deflation Sensitivity (k = {k_values[0]}–{k_values[-1]})\n"
                 "k removes the k largest shared PC directions before computing PR",
                 fontsize=12, fontweight="bold")

    for idx, k in enumerate(k_values):
        ax    = axes[idx // ncols][idx % ncols]
        r_m   = ref_mat[:, :, idx].mean(0);  r_sd = ref_mat[:, :, idx].std(0)
        c_m   = ctrl_mat[:, :, idx].mean(0); c_sd = ctrl_mat[:, :, idx].std(0)
        y_r   = max((r_m + r_sd).max(), (c_m + c_sd).max()) - \
                min((r_m - r_sd).min(), (c_m - c_sd).min())

        for cond, mu, sd, ls in [("ref",  r_m, r_sd, "o-"),
                                   ("ctrl", c_m, c_sd, "s--")]:
            ax.fill_between(levels, mu - sd, mu + sd, alpha=0.15, color=COL[cond])
            ax.plot(levels, mu, ls, color=COL[cond], lw=2.2, ms=6)

        for l in levels:
            sig, _ = _mw_sig(ref_mat[:, l, idx], ctrl_mat[:, l, idx])
            if sig:
                y = max(r_m[l] + r_sd[l], c_m[l] + c_sd[l]) + y_r * 0.07
                ax.text(l, y, sig, ha="center", va="bottom", fontsize=11)

        direction = "Ref > Ctrl" if r_m[-1] > c_m[-1] else "Ctrl > Ref"
        ax.set_title(f"k = {k}  ({direction})", fontsize=11, fontweight="bold")
        ax.set_xticks(levels); ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel("PR" if idx % ncols == 0 else "")
        ax.grid(True, alpha=0.3)

    # single legend in first panel
    axes[0][0].plot([], [], "o-",  color=COL["ref"],  lw=2, label="Refinement")
    axes[0][0].plot([], [], "s--", color=COL["ctrl"], lw=2, label="Control")
    axes[0][0].legend(fontsize=9)
    for idx in range(len(k_values), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "flow_pr_deflation_sweep.png")


# ── save JSON ─────────────────────────────────────────────────────────────────

def _to_json(obj):
    if isinstance(obj, dict):  return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):  return [_to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.integer, int)):  return int(obj)
    return obj


def save_json(ref_mle, ctrl_mle, ref_sc, ctrl_sc,
              ref_pr_whi, ctrl_pr_whi, ref_sw, ctrl_sw, k_values):
    out = {
        "model": "Pythia-2.8B",
        "best_layer": BEST_LAYER,
        "pertrans_mle_ref":  {c: ref_mle[c].tolist()  for c in ALL_CHAINS},
        "pertrans_mle_ctrl": {c: ctrl_mle[c].tolist() for c in CONTROL_CHAINS},
        "stacked_cloud_ref":  {c: ref_sc[c].tolist()  for c in ALL_CHAINS},
        "stacked_cloud_ctrl": {c: ctrl_sc[c].tolist() for c in CONTROL_CHAINS},
        "stacked_cloud_columns": ["mle_id", "pr_raw", "log_vol", "pr_deflate5"],
        "pr_whitening_ref":  {c: ref_pr_whi[c].tolist()  for c in ALL_CHAINS},
        "pr_whitening_ctrl": {c: ctrl_pr_whi[c].tolist() for c in CONTROL_CHAINS},
        "pr_whitening_columns": ["pr_raw", "pr_deflate5"],
        "deflation_sweep_ref":  {c: ref_sw[c].tolist()  for c in ALL_CHAINS},
        "deflation_sweep_ctrl": {c: ctrl_sw[c].tolist() for c in CONTROL_CHAINS},
        "deflation_sweep_k_values": k_values,
    }
    path = os.path.join(FOCUSED_DIR, "flow_analysis_pythia.json")
    with open(path, "w") as f:
        json.dump(_to_json(out), f, indent=2)
    log.info(f"  → {path}")


# ── print stats table ─────────────────────────────────────────────────────────

def print_stats(ref_mle, ctrl_mle, ref_sc, ctrl_sc, ref_pr_whi, ctrl_pr_whi,
                ref_sw, ctrl_sw, k_values):
    ref_m  = np.array([ref_mle[c]  for c in ALL_CHAINS])
    ctrl_m = np.array([ctrl_mle[c] for c in CONTROL_CHAINS])
    rd = ref_m[:, 4] - ref_m[:, 0]; cd = ctrl_m[:, 4] - ctrl_m[:, 0]
    sig, p = _mw_sig(rd, cd)
    pooled = np.sqrt((rd.std()**2 + cd.std()**2) / 2)
    d_val  = (rd.mean() - cd.mean()) / pooled
    log.info(f"\n=== Per-transition flow MLE Δ(L4-L0) ===")
    log.info(f"  Ref: {rd.mean():+.3f}±{rd.std():.3f}  Ctrl: {cd.mean():+.3f}±{cd.std():.3f}  {sig}(p={p:.4f})  d={d_val:.2f}")

    log.info(f"\n=== Stacked cloud logVol Δ(L4-L0) ===")
    ref_lv  = np.array([ref_sc[c][:, 2]  for c in ALL_CHAINS])
    ctrl_lv = np.array([ctrl_sc[c][:, 2] for c in CONTROL_CHAINS])
    rd = ref_lv[:, 4]-ref_lv[:, 0]; cd = ctrl_lv[:, 4]-ctrl_lv[:, 0]
    sig, p = _mw_sig(rd, cd)
    pooled = np.sqrt((rd.std()**2 + cd.std()**2) / 2)
    d_val  = (rd.mean() - cd.mean()) / pooled
    log.info(f"  Ref: {rd.mean():+.3f}±{rd.std():.3f}  Ctrl: {cd.mean():+.3f}±{cd.std():.3f}  {sig}(p={p:.4f})  d={d_val:.2f}")

    log.info(f"\n=== Deflation sweep direction at L4 ===")
    ref_sw_mat  = np.array([ref_sw[c]  for c in ALL_CHAINS])
    ctrl_sw_mat = np.array([ctrl_sw[c] for c in CONTROL_CHAINS])
    for ki, k in enumerate(k_values):
        rv4 = ref_sw_mat[:, 4, ki]; cv4 = ctrl_sw_mat[:, 4, ki]
        sig, p = _mw_sig(rv4, cv4)
        direction = "Ref > Ctrl" if rv4.mean() > cv4.mean() else "Ctrl > Ref"
        log.info(f"  k={k}: {direction} (ref={rv4.mean():.2f}, ctrl={cv4.mean():.2f})  {sig}(p={p:.4f})")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    K_SWEEP = [5, 6, 7, 8, 9, 10]

    acts_ref, acts_ctrl, df_ref, df_ctrl, W_ref, W_ctrl = load_all()
    n_layers = acts_ref.shape[1]

    log.info("Computing flow vectors ...")
    flow_ref,  mag_ref  = compute_flow_vectors(acts_ref,  df_ref,  ALL_CHAINS,     W_ref)
    flow_ctrl, mag_ctrl = compute_flow_vectors(acts_ctrl, df_ctrl, CONTROL_CHAINS, W_ctrl)

    log.info("Per-transition flow MLE ...")
    ref_mle  = compute_pertrans_mle(flow_ref,  ALL_CHAINS)
    ctrl_mle = compute_pertrans_mle(flow_ctrl, CONTROL_CHAINS)
    plot_pertrans_mle(ref_mle, ctrl_mle, ALL_CHAINS, CONTROL_CHAINS)

    log.info("Flow magnitude comparison ...")
    plot_flow_magnitude_comparison(mag_ref, mag_ctrl, ALL_CHAINS, CONTROL_CHAINS)

    log.info("Stacked cloud geometry ...")
    ref_sc  = compute_stacked_cloud(flow_ref,  ALL_CHAINS)
    ctrl_sc = compute_stacked_cloud(flow_ctrl, CONTROL_CHAINS)
    plot_stacked_cloud(ref_sc, ctrl_sc, ALL_CHAINS, CONTROL_CHAINS)

    log.info("PR whitening ...")
    ref_pr_whi  = compute_pr_whitening(flow_ref,  ALL_CHAINS)
    ctrl_pr_whi = compute_pr_whitening(flow_ctrl, CONTROL_CHAINS)
    plot_pr_whitening(ref_pr_whi, ctrl_pr_whi, ALL_CHAINS, CONTROL_CHAINS)

    log.info(f"Deflation sweep k={K_SWEEP[0]}..{K_SWEEP[-1]} ...")
    ref_sw  = compute_deflation_sweep(flow_ref,  ALL_CHAINS,     K_SWEEP)
    ctrl_sw = compute_deflation_sweep(flow_ctrl, CONTROL_CHAINS, K_SWEEP)
    plot_deflation_sweep(ref_sw, ctrl_sw, ALL_CHAINS, CONTROL_CHAINS, K_SWEEP)

    log.info("Saving JSON ...")
    save_json(ref_mle, ctrl_mle, ref_sc, ctrl_sc,
              ref_pr_whi, ctrl_pr_whi, ref_sw, ctrl_sw, K_SWEEP)

    print_stats(ref_mle, ctrl_mle, ref_sc, ctrl_sc, ref_pr_whi, ctrl_pr_whi,
                ref_sw, ctrl_sw, K_SWEEP)

    log.info("FLOW ANALYSIS COMPLETE")
