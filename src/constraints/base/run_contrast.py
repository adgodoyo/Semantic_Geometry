"""
Contrast experiment — refinement chains vs. redundant-commonality controls.

For each model (BERT, Pythia-2.8B):
  1. Extract/cache activations for the control dataset.
  2. Run run_analysis() (same pipeline as run_full.py).
  3. Load the already-computed full-run results JSON.
  4. Produce comparison plots showing per-metric growth (ΔL0→L4) for
     refinement chains vs. control chains.

Outputs:
  results/control_bert/        BERT control geometry + results.json
  results/control_pythia/      Pythia control geometry + results.json
  results/contrast/            All comparison plots
  results/contrast/contrast.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    MODEL_NAME, N_LAYERS, POOLING,
    PYTHIA_MODEL_NAME, PYTHIA_N_LAYERS, PYTHIA_POOLING, PYTHIA_TORCH_DTYPE,
    CONTROL_BERT_DIR, CONTROL_PYTHIA_DIR, CONTRAST_DIR,
    FULL_DIR, N_LEVELS, CONTROL_CHAINS, ALL_CHAINS,
)
from data_loader import load_control_dataset, get_chain_info
from extract_activations import extract_and_cache
from run_pilot import run_analysis, _make_logger

FULL_PYTHIA_DIR = os.path.join(RESULTS_DIR, "full_pythia")


# ── Comparison plots ───────────────────────────────────────────────────────────

def _delta(metric_dict: dict, chains: list) -> dict:
    """ΔL0→L4 for each chain: last value minus first value."""
    return {c: float(np.array(metric_dict[c])[-1] - np.array(metric_dict[c])[0])
            for c in chains if c in metric_dict}


def plot_delta_comparison(ref_results: dict, ctrl_results: dict,
                          model_label: str, out_dir: str) -> None:
    """
    Bar chart: ΔL0→L4 per metric, refinement vs control chains.
    Each bar = mean across chains; error bars = std across chains.
    """
    metrics = [
        ("pr",      "ΔPR",      "geometry_A"),
        ("mle_id",  "ΔMLE ID",  "geometry_A"),
        ("sr",      "ΔStable Rank", "geometry_A"),
        ("log_vol", "ΔLog-Vol", "geometry_A"),
    ]
    ref_chains  = ref_results["chains"]
    ctrl_chains = ctrl_results["chains"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    fig.suptitle(f"Refinement vs Control — ΔL0→L4  ({model_label})", fontsize=13)

    for ax, (key, label, block) in zip(axes, metrics):
        ref_deltas  = list(_delta(ref_results[block][key],  ref_chains).values())
        ctrl_deltas = list(_delta(ctrl_results[block][key], ctrl_chains).values())

        means = [np.mean(ref_deltas), np.mean(ctrl_deltas)]
        stds  = [np.std(ref_deltas),  np.std(ctrl_deltas)]
        colors = ["#2196F3", "#FF7043"]
        bars = ax.bar(["Refinement", "Control"], means, yerr=stds,
                      color=colors, capsize=5, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Δ (L4 − L0)")
        # Scatter individual chains
        for i, vals in enumerate([ref_deltas, ctrl_deltas]):
            ax.scatter([i] * len(vals), vals, color="black", s=18, zorder=5, alpha=0.6)

    plt.tight_layout()
    path = os.path.join(out_dir, f"delta_comparison_{model_label.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_trajectory_comparison(ref_results: dict, ctrl_results: dict,
                                model_label: str, out_dir: str) -> None:
    """
    Mean trajectory per level (averaged across chains) for refinement vs control.
    One subplot per metric.
    """
    metrics = [
        ("pr",      "PR",           "geometry_A"),
        ("mle_id",  "MLE ID",       "geometry_A"),
        ("sr",      "Stable Rank",  "geometry_A"),
        ("log_vol", "Log-Vol",      "geometry_A"),
    ]
    levels = list(range(N_LEVELS))

    ref_chains  = ref_results["chains"]
    ctrl_chains = ctrl_results["chains"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
    fig.suptitle(f"Mean Trajectory: Refinement vs Control  ({model_label})", fontsize=13)

    for ax, (key, label, block) in zip(axes, metrics):
        ref_mat  = np.array([ref_results[block][key][c]  for c in ref_chains
                             if c in ref_results[block][key]])
        ctrl_mat = np.array([ctrl_results[block][key][c] for c in ctrl_chains
                             if c in ctrl_results[block][key]])

        ref_mean  = ref_mat.mean(0)  if len(ref_mat)  else np.zeros(N_LEVELS)
        ctrl_mean = ctrl_mat.mean(0) if len(ctrl_mat) else np.zeros(N_LEVELS)
        ref_std   = ref_mat.std(0)   if len(ref_mat)  else np.zeros(N_LEVELS)
        ctrl_std  = ctrl_mat.std(0)  if len(ctrl_mat) else np.zeros(N_LEVELS)

        ax.plot(levels, ref_mean,  color="#2196F3", marker="o", label="Refinement")
        ax.fill_between(levels, ref_mean - ref_std, ref_mean + ref_std,
                        color="#2196F3", alpha=0.18)
        ax.plot(levels, ctrl_mean, color="#FF7043", marker="s", label="Control")
        ax.fill_between(levels, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std,
                        color="#FF7043", alpha=0.18)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Semantic level (constraints added)")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, f"trajectory_comparison_{model_label.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_whitening_comparison(ref_results: dict, ctrl_results: dict,
                               model_label: str, out_dir: str) -> None:
    """
    Approach B (anisotropy-controlled) PR: refinement vs control, all whitening variants.
    """
    variants = ["raw", "zca", "deflate_top5", "mahalanobis"]
    labels   = ["Raw",  "ZCA", "Top-5 deflate", "Mahalanobis"]
    levels   = list(range(N_LEVELS))

    ref_chains  = ref_results["chains"]
    ctrl_chains = ctrl_results["chains"]

    fig, axes = plt.subplots(1, len(variants), figsize=(18, 4))
    fig.suptitle(f"Approach B PR (whitened): Refinement vs Control  ({model_label})", fontsize=12)

    for ax, (var, lbl) in zip(axes, zip(variants, labels)):
        ref_b  = ref_results.get("geometry_B",  {}).get(var, {})
        ctrl_b = ctrl_results.get("geometry_B", {}).get(var, {})

        ref_mat  = np.array([ref_b[c]  for c in ref_chains  if c in ref_b])
        ctrl_mat = np.array([ctrl_b[c] for c in ctrl_chains if c in ctrl_b])

        for mat, color, name in [(ref_mat, "#2196F3", "Refinement"),
                                  (ctrl_mat, "#FF7043", "Control")]:
            if len(mat) == 0:
                continue
            mean = mat.mean(0)
            std  = mat.std(0)
            ax.plot(levels, mean, color=color, marker="o", label=name)
            ax.fill_between(levels, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Semantic level")
        ax.set_ylabel("PR")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, f"whitening_comparison_{model_label.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_flow_comparison(ref_results: dict, ctrl_results: dict,
                          model_label: str, out_dir: str) -> None:
    """
    Cumulative flow PR: raw and semantic-subspace, refinement vs control.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Cumulative Flow PR: Refinement vs Control  ({model_label})", fontsize=12)

    for ax, (flow_key, title) in zip(axes, [
        ("flow",     "Raw space"),
        ("flow_sem", "Semantic subspace"),
    ]):
        ref_cum  = np.array(ref_results.get(flow_key, {}).get("cum_pr", []))
        ctrl_cum = np.array(ctrl_results.get(flow_key, {}).get("cum_pr", []))

        if len(ref_cum):
            ax.plot(ref_cum,  color="#2196F3", label="Refinement")
        if len(ctrl_cum):
            ax.plot(ctrl_cum, color="#FF7043", label="Control")
        ax.set_title(title)
        ax.set_xlabel("Layer transition")
        ax.set_ylabel("Cumulative PR")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, f"flow_comparison_{model_label.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def save_contrast_csv(ref_results: dict, ctrl_results: dict,
                       model_label: str, out_dir: str) -> str:
    """Save a CSV with per-chain Δ metrics for both conditions."""
    metrics = [("pr", "geometry_A"), ("mle_id", "geometry_A"),
               ("sr", "geometry_A"), ("log_vol", "geometry_A")]
    rows = []
    for condition, results, chains in [
        ("refinement", ref_results,  ref_results["chains"]),
        ("control",    ctrl_results, ctrl_results["chains"]),
    ]:
        for chain in chains:
            row = {"condition": condition, "chain": chain}
            for key, block in metrics:
                vals = np.array(results[block].get(key, {}).get(chain, [0] * N_LEVELS))
                row[f"delta_{key}"] = float(vals[-1] - vals[0]) if len(vals) >= 2 else 0.0
                row[f"L0_{key}"]    = float(vals[0])  if len(vals) else 0.0
                row[f"L4_{key}"]    = float(vals[-1]) if len(vals) else 0.0
            rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, f"contrast_deltas_{model_label.replace(' ', '_')}.csv")
    df.to_csv(path, index=False)
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(CONTROL_BERT_DIR,   exist_ok=True)
    os.makedirs(CONTROL_PYTHIA_DIR, exist_ok=True)
    os.makedirs(CONTRAST_DIR,       exist_ok=True)

    log_path = os.path.join(CONTRAST_DIR, "contrast.log")
    logger   = _make_logger(log_path)
    logger.handlers[0].stream  # ensure log file is opened fresh

    logger.info("=" * 60)
    logger.info("CONTRAST EXPERIMENT — refinement vs redundant controls")
    logger.info("=" * 60)

    # ── Load control dataset ──────────────────────────────────────────────────
    df_ctrl    = load_control_dataset()
    chain_info = get_chain_info(df_ctrl)
    logger.info(f"Control sentences: {len(df_ctrl)}  |  Chains: {len(CONTROL_CHAINS)}")

    # ── BERT control ──────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("MODEL: BERT-base-uncased  [control dataset]")
    logger.info("─" * 60)
    bert_ctrl_cache = os.path.join(CONTROL_BERT_DIR, "acts_control_bert.npy")
    bert_ctrl_acts  = extract_and_cache(
        df_ctrl, bert_ctrl_cache,
        model_name=MODEL_NAME, n_layers=N_LAYERS, pooling=POOLING,
        torch_dtype="float32")
    logger.info(f"BERT control activations shape: {bert_ctrl_acts.shape}")

    bert_ctrl_results = run_analysis(
        bert_ctrl_acts, df_ctrl, chain_info,
        output_dir=CONTROL_BERT_DIR,
        model_label="BERT-base (control)",
        n_layers=N_LAYERS,
        logger=logger,
        chains=CONTROL_CHAINS)

    # ── Pythia control ────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("MODEL: Pythia-2.8B  [control dataset]")
    logger.info("─" * 60)
    pythia_ctrl_cache = os.path.join(CONTROL_PYTHIA_DIR, "acts_control_pythia.npy")
    pythia_ctrl_acts  = extract_and_cache(
        df_ctrl, pythia_ctrl_cache,
        model_name=PYTHIA_MODEL_NAME, n_layers=PYTHIA_N_LAYERS,
        pooling=PYTHIA_POOLING, torch_dtype=PYTHIA_TORCH_DTYPE,
        batch_size=4)
    logger.info(f"Pythia control activations shape: {pythia_ctrl_acts.shape}")

    pythia_ctrl_results = run_analysis(
        pythia_ctrl_acts, df_ctrl, chain_info,
        output_dir=CONTROL_PYTHIA_DIR,
        model_label="Pythia-2.8B (control)",
        n_layers=PYTHIA_N_LAYERS,
        logger=logger,
        chains=CONTROL_CHAINS)

    # ── Load refinement results (already computed) ────────────────────────────
    logger.info("─" * 60)
    logger.info("Loading refinement results from JSON cache …")
    bert_ref_path   = os.path.join(FULL_DIR, "results.json")
    pythia_ref_path = os.path.join(FULL_PYTHIA_DIR, "results.json")

    if not os.path.exists(bert_ref_path):
        logger.warning(f"  BERT full results not found at {bert_ref_path} — run run_full.py first")
        return
    if not os.path.exists(pythia_ref_path):
        logger.warning(f"  Pythia full results not found at {pythia_ref_path} — run run_full.py first")
        return

    with open(bert_ref_path)   as f: bert_ref_results   = json.load(f)
    with open(pythia_ref_path) as f: pythia_ref_results  = json.load(f)
    logger.info("  Loaded refinement JSONs.")

    # ── Comparison plots ──────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("Generating comparison plots …")

    for model_label, ref_res, ctrl_res in [
        ("BERT-base",   bert_ref_results,   bert_ctrl_results),
        ("Pythia-2.8B", pythia_ref_results, pythia_ctrl_results),
    ]:
        p1 = plot_delta_comparison(ref_res, ctrl_res, model_label, CONTRAST_DIR)
        p2 = plot_trajectory_comparison(ref_res, ctrl_res, model_label, CONTRAST_DIR)
        p3 = plot_whitening_comparison(ref_res, ctrl_res, model_label, CONTRAST_DIR)
        p4 = plot_flow_comparison(ref_res, ctrl_res, model_label, CONTRAST_DIR)
        p5 = save_contrast_csv(ref_res, ctrl_res, model_label, CONTRAST_DIR)
        logger.info(f"  [{model_label}] plots saved to {CONTRAST_DIR}")

        # Console summary
        for metric, block in [("pr", "geometry_A"), ("mle_id", "geometry_A")]:
            ref_deltas  = [float(np.array(ref_res[block][metric].get(c, [0,0]))[-1]
                              - np.array(ref_res[block][metric].get(c, [0,0]))[0])
                           for c in ref_res["chains"] if c in ref_res[block][metric]]
            ctrl_deltas = [float(np.array(ctrl_res[block][metric].get(c, [0,0]))[-1]
                               - np.array(ctrl_res[block][metric].get(c, [0,0]))[0])
                            for c in ctrl_res["chains"] if c in ctrl_res[block][metric]]
            logger.info(f"  [{model_label}] Δ{metric.upper():<8} "
                        f"refinement={np.mean(ref_deltas):+.3f}±{np.std(ref_deltas):.3f}  "
                        f"control={np.mean(ctrl_deltas):+.3f}±{np.std(ctrl_deltas):.3f}")

    logger.info("=" * 60)
    logger.info("CONTRAST COMPLETE")
    logger.info(f"  Plots → {CONTRAST_DIR}")
    logger.info(f"  Log   → {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
