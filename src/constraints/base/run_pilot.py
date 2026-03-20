"""
Pilot experiment — 3 chains: hiking_boot, rescue_drone, emergency_vehicle.

Runs all three approaches for BERT and then Pythia-2.8B, saving:
  results/pilot/               BERT plots + geometry_summary.csv + results.json
  results/pilot_pythia/        Pythia plots + geometry_summary.csv + results.json
  results/pilot/run_pilot.log  Full progress log (both models)

The results.json files capture every numerical result so future analyses
can load them directly without re-running the pipeline.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (PILOT_CHAINS, N_LEVELS,
                    PILOT_DIR, SEM_SUBSPACE_DIM,
                    MODEL_NAME, N_LAYERS, POOLING,
                    PYTHIA_MODEL_NAME, PYTHIA_N_LAYERS, PYTHIA_POOLING,
                    PYTHIA_TORCH_DTYPE, PYTHIA_PILOT_DIR)
from data_loader import load_dataset, filter_chains, get_chain_info
from extract_activations import extract_and_cache
from subspace import (learn_nuisance_projector, learn_semantic_projector,
                      project_semantic, residualise_nuisance, selectivity_score)
from whitening import get_all_whitened
from layer_flow import (compute_flows, flow_magnitude_per_transition,
                        flow_pr_per_transition, flow_er_per_transition,
                        centroid_trajectory, consecutive_level_distance,
                        flow_alignment_score,
                        cumulative_flow_pr_across_layers,
                        cumulative_flow_er_across_layers)
from geometry import all_measures, eigenspectrum
from visualize import (plot_selectivity_curve, plot_pr_trajectory,
                       plot_pr_comparison_ab, plot_eigenspectrum_heatmap,
                       plot_pca_scatter, plot_flow_magnitude, plot_flow_pr_layers,
                       plot_centroid_distances, plot_flow_alignment,
                       plot_all_chains_centroid_heatmap, plot_geometry_summary,
                       plot_cumulative_flow_pr)


# ── JSON serialisation helper ──────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Serialize numpy arrays/scalars to JSON-compatible Python types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (v != v or v == float("inf") or v == float("-inf")) else v
        return super().default(obj)


def save_results_json(results: dict, path: str) -> None:
    """Dump the full numerical results dict to a JSON file.

    All numpy arrays are converted to nested lists; inf/nan → null.
    Load with:
        import json, numpy as np
        with open("results.json") as f:
            r = json.load(f)
        pr_A = {k: np.array(v) for k, v in r["geometry_A"]["pr"].items()}
    """
    with open(path, "w") as f:
        json.dump(results, f, cls=_NumpyEncoder, indent=2)


# ── logging setup ──────────────────────────────────────────────────────────────

def _make_logger(log_path: str) -> logging.Logger:
    """Create a logger that writes to both stdout and a log file."""
    logger = logging.getLogger("pilot")
    logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers if run() is called more than once in a session
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── analysis kernel ────────────────────────────────────────────────────────────

def run_analysis(acts: np.ndarray,
                 df: pd.DataFrame,
                 chain_info: dict,
                 output_dir: str,
                 model_label: str,
                 n_layers: int,
                 logger: logging.Logger,
                 chains: list | None = None) -> dict:
    """Run all three approaches on pre-extracted activations.

    Returns a dict with every numerical result. Also saves:
      output_dir/geometry_summary.csv
      output_dir/results.json        ← full results, loadable without re-running
    """
    os.makedirs(output_dir, exist_ok=True)

    sem_labels   = df["chain_level_id"].values
    surf_labels  = df["surface_id"].values
    level_labels = df["level"].values
    chain_labels = df["chain_id"].values
    chains       = chains if chains is not None else PILOT_CHAINS
    all_layer_indices = list(range(n_layers))
    # Middle 50% of layers for dense whitening analysis
    dense = list(range(max(0, n_layers // 4), min(n_layers, 3 * n_layers // 4)))

    # ── APPROACH A — semantic subspace isolation ──────────────────────────────
    logger.info(f"[{model_label}] APPROACH A — semantic subspace")
    selectivity = np.zeros(n_layers)
    pr_A  = {c: np.zeros(N_LEVELS) for c in chains}
    er_A  = {c: np.zeros(N_LEVELS) for c in chains}
    rr_A  = {c: np.zeros(N_LEVELS) for c in chains}
    sr_A  = {c: np.zeros(N_LEVELS) for c in chains}
    mle_A = {c: np.zeros(N_LEVELS) for c in chains}
    lv_A  = {c: np.zeros(N_LEVELS) for c in chains}

    for layer in tqdm(all_layer_indices, desc=f"  [{model_label}] selectivity"):
        X = acts[:, layer, :]
        P_sem, _ = learn_semantic_projector(X, sem_labels)
        X_proj   = project_semantic(X, P_sem)
        selectivity[layer] = selectivity_score(X_proj, sem_labels, surf_labels)
        logger.debug(f"  layer L{layer:02d} selectivity={selectivity[layer]:.4f}")

    best_layer = int(np.argmax(selectivity))
    logger.info(f"[{model_label}] Best layer: L{best_layer} "
                f"(selectivity={selectivity[best_layer]:.3f})")

    X_best         = acts[:, best_layer, :]
    P_nuis_best    = learn_nuisance_projector(X_best, surf_labels)
    P_sem_best, _  = learn_semantic_projector(X_best, sem_labels)
    X_proj_best    = project_semantic(X_best, P_sem_best)

    for chain in chains:
        cmask = chain_labels == chain
        logger.info(f"[{model_label}] Approach A geometry — chain: {chain}")
        for l_max in range(N_LEVELS):
            cumask = cmask & (level_labels <= l_max)
            if cumask.sum() < 2:
                continue
            m = all_measures(X_proj_best[cumask])
            pr_A[chain][l_max]  = m["pr"]
            er_A[chain][l_max]  = m["er"]
            rr_A[chain][l_max]  = m["rr"]
            sr_A[chain][l_max]  = m["sr"]
            mle_A[chain][l_max] = m["mle_id"]
            lv_A[chain][l_max]  = m["log_vol"]
            logger.debug(f"    {chain} L{l_max}: PR={m['pr']:.3f} SR={m['sr']:.3f} "
                         f"MLE={m['mle_id']:.3f} logVol={m['log_vol']:.3f}")

    # Eigenspectra
    spectra_rows, spectra_labels_list = [], []
    for chain in chains:
        cmask = chain_labels == chain
        for l_max in range(N_LEVELS):
            cumask = cmask & (level_labels <= l_max)
            if cumask.sum() < 2:
                continue
            sp = eigenspectrum(X_proj_best[cumask], n_components=10)
            spectra_rows.append(sp)
            concept = chain_info[chain].get(l_max, f"L{l_max}")
            spectra_labels_list.append(f"{chain[:8]}·L{l_max} ({concept[:18]})")
    spectra_arr = np.array(spectra_rows)

    # ── APPROACH B — whitening ────────────────────────────────────────────────
    logger.info(f"[{model_label}] APPROACH B — whitening ({len(dense)} layers)")
    pr_B = {variant: {c: np.zeros(N_LEVELS) for c in chains}
            for variant in ["raw", "zca", "deflate_top5", "mahalanobis"]}

    for layer in tqdm(dense, desc=f"  [{model_label}] whitening"):
        X = acts[:, layer, :]
        whitened = get_all_whitened(X)
        for variant, Xw in whitened.items():
            if variant not in pr_B:
                pr_B[variant] = {c: np.zeros(N_LEVELS) for c in chains}
            for chain in chains:
                cmask = chain_labels == chain
                for l_max in range(N_LEVELS):
                    cumask = cmask & (level_labels <= l_max)
                    if cumask.sum() < 2:
                        continue
                    pr_B[variant][chain][l_max] += (
                        all_measures(Xw[cumask])["pr"] / len(dense))
        logger.debug(f"  [{model_label}] whitening L{layer} done")

    # ── APPROACH C — layer flow ───────────────────────────────────────────────
    logger.info(f"[{model_label}] APPROACH C — layer flow (raw)")
    flows      = compute_flows(acts)                    # (N, n_layers-1, D)
    flow_mag   = flow_magnitude_per_transition(flows)
    flow_pr    = flow_pr_per_transition(flows)
    flow_er    = flow_er_per_transition(flows)
    flow_align = flow_alignment_score(flows, sem_labels)

    logger.info(f"[{model_label}] Computing cumulative flow PR across layers (raw)")
    cum_flow_pr = cumulative_flow_pr_across_layers(flows)
    cum_flow_er = cumulative_flow_er_across_layers(flows)

    # ── APPROACH C (sem) — flow inside the semantic subspace ─────────────────
    # Project ALL layers through P_sem_best (learned at best_layer).
    # This removes nuisance directions before measuring flow geometry,
    # as required by §9.2 of the instructions.
    logger.info(f"[{model_label}] APPROACH C — layer flow (semantic subspace, dim={SEM_SUBSPACE_DIM})")
    N_total, n_layers_total, D_full = acts.shape
    acts_sem = (acts.reshape(-1, D_full) @ P_sem_best.T).reshape(N_total, n_layers_total, -1)
    flows_sem      = compute_flows(acts_sem)
    flow_pr_sem    = flow_pr_per_transition(flows_sem)
    flow_er_sem    = flow_er_per_transition(flows_sem)
    flow_align_sem = flow_alignment_score(flows_sem, sem_labels)
    cum_flow_pr_sem = cumulative_flow_pr_across_layers(flows_sem)
    cum_flow_er_sem = cumulative_flow_er_across_layers(flows_sem)
    logger.info(f"[{model_label}] Sem-subspace cumulative flow PR: "
                f"{cum_flow_pr_sem[0]:.1f} → {cum_flow_pr_sem[-1]:.1f}")

    centroid_dists = {}
    for chain in chains:
        cmask = (chain_labels == chain)
        cent  = centroid_trajectory(acts, cmask, level_labels[cmask])
        centroid_dists[chain] = consecutive_level_distance(cent)
        logger.debug(f"  [{model_label}] centroid trajectory done: {chain}")

    # ── Summary DataFrame ─────────────────────────────────────────────────────
    rows = []
    for chain in chains:
        for l_max in range(N_LEVELS):
            rows.append({
                "chain":         chain,
                "level":         l_max,
                "pr_A":          pr_A[chain][l_max],
                "er_A":          er_A[chain][l_max],
                "rr_A":          rr_A[chain][l_max],
                "sr_A":          sr_A[chain][l_max],
                "mle_id_A":      mle_A[chain][l_max],
                "log_vol_A":     lv_A[chain][l_max],
                "pr_B_raw":      pr_B["raw"][chain][l_max],
                "pr_B_zca":      pr_B["zca"][chain][l_max],
                "pr_B_deflate5": pr_B.get("deflate_top5", {}).get(chain, np.zeros(N_LEVELS))[l_max],
                "pr_B_mahal":    pr_B["mahalanobis"][chain][l_max],
            })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(output_dir, "geometry_summary.csv"), index=False)
    logger.info(f"[{model_label}] CSV saved → geometry_summary.csv")

    # ── JSON results cache ────────────────────────────────────────────────────
    # Every numerical result is serialised here so future analyses can load
    # results.json directly, skipping the full extraction + geometry pipeline.
    #
    # Loading example:
    #   import json, numpy as np
    #   r = json.load(open("results/pilot/results.json"))
    #   pr_A_hb = np.array(r["geometry_A"]["pr"]["hiking_boot"])   # shape (5,)
    #   cum_pr  = np.array(r["flow"]["cum_pr"])                     # shape (n_transitions,)
    results_dict = {
        "model":      model_label,
        "n_layers":   n_layers,
        "best_layer": best_layer,
        "chains":     chains,
        "selectivity": selectivity,
        "geometry_A": {
            "pr":      pr_A,
            "er":      er_A,
            "rr":      rr_A,
            "sr":      sr_A,
            "mle_id":  mle_A,
            "log_vol": lv_A,
        },
        "geometry_B": {
            variant: data
            for variant, data in pr_B.items()
        },
        "eigenspectra": {
            "matrix": spectra_arr,
            "labels": spectra_labels_list,
        },
        "flow": {
            "magnitude":   flow_mag,
            "pr":          flow_pr,
            "er":          flow_er,
            "alignment":   flow_align,
            "cum_pr":      cum_flow_pr,
            "cum_er":      cum_flow_er,
        },
        "flow_sem": {
            "pr":       flow_pr_sem,
            "er":       flow_er_sem,
            "alignment": flow_align_sem,
            "cum_pr":   cum_flow_pr_sem,
            "cum_er":   cum_flow_er_sem,
        },
        "centroid_dists": {c: centroid_dists[c] for c in chains},
    }
    json_path = os.path.join(output_dir, "results.json")
    save_results_json(results_dict, json_path)
    logger.info(f"[{model_label}] Results JSON saved → {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info(f"[{model_label}] Plotting …")

    plot_selectivity_curve(
        selectivity, os.path.join(output_dir, "selectivity_curve.png"),
        title=f"Semantic Selectivity — {model_label} (pilot)")

    plot_pr_trajectory(
        pr_A, os.path.join(output_dir, "pr_trajectory_A.png"),
        title=f"PR vs Level — Approach A (Semantic Subspace) — {model_label}")

    plot_pr_trajectory(
        mle_A, os.path.join(output_dir, "mle_id_trajectory_A.png"),
        measure="MLE Intrinsic Dim",
        title=f"MLE Intrinsic Dim vs Level — {model_label}")

    plot_pr_trajectory(
        sr_A, os.path.join(output_dir, "stable_rank_trajectory_A.png"),
        measure="Stable Rank",
        title=f"Stable Rank vs Level — {model_label}")

    plot_pr_trajectory(
        lv_A, os.path.join(output_dir, "log_vol_trajectory_A.png"),
        measure="Log-Volume",
        title=f"Log-Volume vs Level — {model_label}")

    plot_pr_trajectory(
        pr_B["zca"], os.path.join(output_dir, "pr_trajectory_B_zca.png"),
        measure="PR",
        title=f"PR vs Level — Approach B (ZCA Whitening) — {model_label}")

    plot_pr_trajectory(
        pr_B.get("deflate_top5", pr_B["raw"]),
        os.path.join(output_dir, "pr_trajectory_B_deflate5.png"),
        measure="PR",
        title=f"PR vs Level — Approach B (Top-5 PC Deflation) — {model_label}")

    plot_pr_comparison_ab(
        pr_A, pr_B["zca"], os.path.join(output_dir, "pr_comparison_AB.png"))

    if len(spectra_arr) > 0:
        plot_eigenspectrum_heatmap(
            spectra_arr, spectra_labels_list,
            os.path.join(output_dir, "eigenspectrum_heatmap.png"))

    plot_pca_scatter(
        X_proj_best, chain_labels, level_labels,
        os.path.join(output_dir, "pca_scatter_best_layer.png"),
        title=f"Semantic Subspace PCA — Layer L{best_layer} — {model_label}")

    plot_flow_magnitude(
        flow_mag, os.path.join(output_dir, "flow_magnitude.png"))

    plot_flow_pr_layers(
        flow_pr, os.path.join(output_dir, "flow_pr_layers.png"))

    plot_flow_alignment(
        flow_align, os.path.join(output_dir, "flow_alignment.png"))

    plot_cumulative_flow_pr(
        cum_flow_pr, flow_pr,
        os.path.join(output_dir, "flow_pr_cumulative.png"),
        title=f"Cumulative vs Per-transition Flow PR — {model_label} (raw space)")

    plot_cumulative_flow_pr(
        cum_flow_pr_sem, flow_pr_sem,
        os.path.join(output_dir, "flow_pr_cumulative_sem.png"),
        title=f"Cumulative vs Per-transition Flow PR — {model_label} (semantic subspace)")

    for chain in chains:
        plot_centroid_distances(
            centroid_dists[chain], chain,
            os.path.join(output_dir, f"centroid_dist_{chain}.png"))
        logger.debug(f"  [{model_label}] centroid plot saved: {chain}")

    plot_all_chains_centroid_heatmap(
        centroid_dists,
        os.path.join(output_dir, "centroid_dist_heatmap.png"),
        layer=best_layer)

    plot_geometry_summary(
        summary_df, os.path.join(output_dir, "geometry_summary_plot.png"))

    logger.info(f"[{model_label}] All plots saved → {output_dir}")

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info(f"[{model_label}] Best semantic layer: L{best_layer} "
                f"(S={selectivity[best_layer]:.3f})")
    logger.info(f"[{model_label}] PR / MLE-id / stable-rank / log-vol (L0→L4):")
    for chain in chains:
        dpr = pr_A[chain][-1] - pr_A[chain][0]
        dml = mle_A[chain][-1] - mle_A[chain][0]
        dsr = sr_A[chain][-1] - sr_A[chain][0]
        dlv = lv_A[chain][-1] - lv_A[chain][0]
        logger.info(f"  {chain:<22}  ΔPR={dpr:+.2f}  ΔMLE={dml:+.2f}  "
                    f"ΔSR={dsr:+.2f}  ΔlogVol={dlv:+.2f}")

    logger.info(f"[{model_label}] Cumulative flow PR (raw space):      "
                f"{cum_flow_pr[0]:.1f} → {cum_flow_pr[-1]:.1f}")
    logger.info(f"[{model_label}] Cumulative flow PR (semantic subspace): "
                f"{cum_flow_pr_sem[0]:.1f} → {cum_flow_pr_sem[-1]:.1f}")

    return results_dict


# ── entry point ────────────────────────────────────────────────────────────────

def run():
    os.makedirs(PILOT_DIR, exist_ok=True)
    log_path = os.path.join(PILOT_DIR, "run_pilot.log")
    logger   = _make_logger(log_path)

    logger.info("=" * 60)
    logger.info("PILOT EXPERIMENT — 3 chains")
    logger.info("=" * 60)

    df_all     = load_dataset()
    df         = filter_chains(df_all, PILOT_CHAINS)
    chain_info = get_chain_info(df)
    logger.info(f"Sentences: {len(df)}  |  Chains: {len(PILOT_CHAINS)}")

    # ── BERT ──────────────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("MODEL: BERT-base-uncased")
    logger.info("─" * 60)
    bert_cache = os.path.join(PILOT_DIR, "acts_pilot.npy")
    bert_acts  = extract_and_cache(
        df, bert_cache,
        model_name=MODEL_NAME, n_layers=N_LAYERS, pooling=POOLING,
        torch_dtype="float32")
    logger.info(f"BERT activations shape: {bert_acts.shape}")

    bert_results = run_analysis(
        bert_acts, df, chain_info,
        output_dir=PILOT_DIR,
        model_label="BERT-base",
        n_layers=N_LAYERS,
        logger=logger)

    # ── Pythia 2.8B ───────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("MODEL: Pythia-2.8B")
    logger.info("─" * 60)
    os.makedirs(PYTHIA_PILOT_DIR, exist_ok=True)
    pythia_cache = os.path.join(PYTHIA_PILOT_DIR, "acts_pilot_pythia.npy")
    pythia_acts  = extract_and_cache(
        df, pythia_cache,
        model_name=PYTHIA_MODEL_NAME,
        n_layers=PYTHIA_N_LAYERS,
        pooling=PYTHIA_POOLING,
        torch_dtype=PYTHIA_TORCH_DTYPE,
        batch_size=4)           # smaller batch for 2.8B memory footprint
    logger.info(f"Pythia activations shape: {pythia_acts.shape}")

    pythia_results = run_analysis(
        pythia_acts, df, chain_info,
        output_dir=PYTHIA_PILOT_DIR,
        model_label="Pythia-2.8B",
        n_layers=PYTHIA_N_LAYERS,
        logger=logger)

    logger.info("=" * 60)
    logger.info("PILOT COMPLETE")
    logger.info(f"  BERT results   → {PILOT_DIR}")
    logger.info(f"  Pythia results → {PYTHIA_PILOT_DIR}")
    logger.info(f"  Log            → {log_path}")
    logger.info("=" * 60)

    return {"bert": bert_results, "pythia": pythia_results}


if __name__ == "__main__":
    run()
