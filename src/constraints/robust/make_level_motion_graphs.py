#!/usr/bin/env python3
"""Level-specific motion graphs for raw Pythia activations."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "constraints" / "robust"
MPL_DIR = RESULTS_DIR / ".mplcache"
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from run_robust import CTRL_ACTS, REF_ACTS, load_control_df, load_refinement_df


PLOT_LEVELS = [0, 1, 2, 4]
LEVEL_LABELS = {0: "L0", 1: "L1", 2: "L2", 4: "L4"}
REF_COLORS = {
    0: "#b7d4ea",
    1: "#7fb3d5",
    2: "#3f88c5",
    4: "#145a96",
}
CTRL_COLORS = {
    0: "#f6c18b",
    1: "#f0a35e",
    2: "#e67e22",
    4: "#b65a14",
}
METRIC_SPECS = {
    "euclid_init": {
        "title": "Raw level motion from initial configuration",
        "ylabel": "Mean Euclidean distance from layer 0",
        "filename": "requested_raw_level_motion_euclidean_from_initial.png",
    },
    "euclid_prev": {
        "title": "Raw level motion from previous layer",
        "ylabel": "Mean Euclidean distance from previous layer",
        "filename": "requested_raw_level_motion_euclidean_from_previous.png",
    },
    "mahal_init": {
        "title": "Raw level motion from initial configuration",
        "ylabel": "Subspace Mahalanobis distance from layer 0",
        "filename": "requested_raw_level_motion_mahalanobis_from_initial.png",
    },
    "mahal_prev": {
        "title": "Raw level motion from previous layer",
        "ylabel": "Subspace Mahalanobis distance from previous layer",
        "filename": "requested_raw_level_motion_mahalanobis_from_previous.png",
    },
    "innovation_prev": {
        "title": "Raw movement into new directions",
        "ylabel": "Orthogonal innovation ratio vs previous-layer subspace",
        "filename": "requested_raw_level_motion_innovation_vs_previous.png",
    },
}


def build_pure_index(df: pd.DataFrame) -> tuple[list[str], dict[tuple[str, int], np.ndarray]]:
    chains = sorted(df["chain_id"].unique())
    chain_arr = df["chain_id"].to_numpy()
    level_arr = df["level"].to_numpy(dtype=int)
    index = {}
    for chain in chains:
        chain_mask = chain_arr == chain
        for level in PLOT_LEVELS:
            index[(chain, level)] = np.where(chain_mask & (level_arr == level))[0]
    return chains, index


def reference_subspace(X_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xc = X_ref - X_ref.mean(axis=0, keepdims=True)
    if min(Xc.shape) < 2:
        return np.zeros((X_ref.shape[1], 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigs = (s ** 2) / max(Xc.shape[0] - 1, 1)
    keep = eigs > 1e-10
    if not np.any(keep):
        return np.zeros((X_ref.shape[1], 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return Vt[keep].T.astype(np.float32, copy=False), eigs[keep].astype(np.float32, copy=False)


def euclidean_step(D: np.ndarray) -> np.ndarray:
    return np.linalg.norm(D, axis=1)


def subspace_mahalanobis(D: np.ndarray, basis: np.ndarray, eigs: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros((D.shape[0],), dtype=np.float32)
    proj = D @ basis
    ridge = max(float(np.median(eigs)), 1e-6)
    md2 = np.sum((proj ** 2) / (eigs[None, :] + ridge), axis=1)
    return np.sqrt(np.clip(md2, 0.0, None))


def innovation_ratio(D: np.ndarray, basis: np.ndarray) -> np.ndarray:
    total_sq = np.sum(D ** 2, axis=1)
    if basis.shape[1] == 0:
        return np.where(total_sq > 1e-12, 1.0, 0.0).astype(np.float32)
    proj_sq = np.sum((D @ basis) ** 2, axis=1)
    orth_sq = np.clip(total_sq - proj_sq, 0.0, None)
    return np.sqrt(orth_sq) / (np.sqrt(total_sq) + 1e-8)


def compute_condition_metrics(
    condition: str,
    acts: np.ndarray,
    chains: list[str],
    pure_index: dict[tuple[str, int], np.ndarray],
) -> pd.DataFrame:
    rows = []
    num_layers = acts.shape[1]
    for chain in chains:
        for level in PLOT_LEVELS:
            idx = pure_index[(chain, level)]
            X0 = acts[idx, 0, :].astype(np.float32, copy=False)
            basis0, eigs0 = reference_subspace(X0)
            rows.append(
                {
                    "metric": "euclid_init",
                    "condition": condition,
                    "chain": chain,
                    "level": level,
                    "layer": 0,
                    "value": 0.0,
                }
            )
            rows.append(
                {
                    "metric": "mahal_init",
                    "condition": condition,
                    "chain": chain,
                    "level": level,
                    "layer": 0,
                    "value": 0.0,
                }
            )
            rows.append(
                {
                    "metric": "euclid_prev",
                    "condition": condition,
                    "chain": chain,
                    "level": level,
                    "layer": 0,
                    "value": np.nan,
                }
            )
            rows.append(
                {
                    "metric": "mahal_prev",
                    "condition": condition,
                    "chain": chain,
                    "level": level,
                    "layer": 0,
                    "value": np.nan,
                }
            )
            rows.append(
                {
                    "metric": "innovation_prev",
                    "condition": condition,
                    "chain": chain,
                    "level": level,
                    "layer": 0,
                    "value": np.nan,
                }
            )

            for layer in range(1, num_layers):
                X_cur = acts[idx, layer, :].astype(np.float32, copy=False)
                X_prev = acts[idx, layer - 1, :].astype(np.float32, copy=False)

                D_init = X_cur - X0
                D_prev = X_cur - X_prev

                basis_prev, eigs_prev = reference_subspace(X_prev)

                metric_values = {
                    "euclid_init": float(euclidean_step(D_init).mean()),
                    "euclid_prev": float(euclidean_step(D_prev).mean()),
                    "mahal_init": float(subspace_mahalanobis(D_init, basis0, eigs0).mean()),
                    "mahal_prev": float(subspace_mahalanobis(D_prev, basis_prev, eigs_prev).mean()),
                    "innovation_prev": float(innovation_ratio(D_prev, basis_prev).mean()),
                }

                for metric, value in metric_values.items():
                    rows.append(
                        {
                            "metric": metric,
                            "condition": condition,
                            "chain": chain,
                            "level": level,
                            "layer": layer,
                            "value": value,
                        }
                    )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["metric", "condition", "level", "layer"], as_index=False)["value"]
        .agg(mean="mean", std=lambda x: x.std(ddof=0), n="count")
        .rename(columns={"n": "n_chains"})
    )
    return summary


def plot_metric(summary: pd.DataFrame, metric: str) -> None:
    spec = METRIC_SPECS[metric]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for condition in ["refinement", "control"]:
        sub_cond = summary[(summary["metric"] == metric) & (summary["condition"] == condition)]
        for level in PLOT_LEVELS:
            sub = sub_cond[sub_cond["level"] == level].sort_values("layer")
            color = REF_COLORS[level] if condition == "refinement" else CTRL_COLORS[level]
            linestyle = "-" if condition == "refinement" else "--"
            label = f"{condition.capitalize()} {LEVEL_LABELS[level]}"
            ax.plot(
                sub["layer"],
                sub["mean"],
                color=color,
                linestyle=linestyle,
                linewidth=2.6,
                label=label,
            )
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel(spec["ylabel"], fontsize=14)
    ax.set_title(spec["title"], fontsize=18)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / spec["filename"], dpi=150)
    plt.close(fig)


def combine_pair(left_name: str, right_name: str, out_name: str) -> None:
    left = Image.open(RESULTS_DIR / left_name).convert("RGB")
    right = Image.open(RESULTS_DIR / right_name).convert("RGB")
    target_h = max(left.height, right.height)
    if left.height != target_h:
        left = left.resize((int(round(left.width * target_h / left.height)), target_h))
    if right.height != target_h:
        right = right.resize((int(round(right.width * target_h / right.height)), target_h))
    gap = 24
    canvas = Image.new("RGB", (left.width + right.width + gap, target_h), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    canvas.save(RESULTS_DIR / out_name)


def main() -> None:
    ref_df = load_refinement_df()
    ctrl_df = load_control_df()
    ref_acts = np.load(REF_ACTS, mmap_mode="r")
    ctrl_acts = np.load(CTRL_ACTS, mmap_mode="r")

    ref_chains, ref_index = build_pure_index(ref_df)
    ctrl_chains, ctrl_index = build_pure_index(ctrl_df)

    df = pd.concat(
        [
            compute_condition_metrics("refinement", ref_acts, ref_chains, ref_index),
            compute_condition_metrics("control", ctrl_acts, ctrl_chains, ctrl_index),
        ],
        ignore_index=True,
    )
    df.to_csv(RESULTS_DIR / "requested_raw_level_motion_by_chain.csv", index=False)

    summary = summarize(df)
    summary.to_csv(RESULTS_DIR / "requested_raw_level_motion_summary.csv", index=False)

    for metric in METRIC_SPECS:
        plot_metric(summary, metric)

    combine_pair(
        "requested_raw_level_motion_euclidean_from_initial.png",
        "requested_raw_level_motion_mahalanobis_from_initial.png",
        "requested_raw_level_motion_initial_combined.png",
    )
    combine_pair(
        "requested_raw_level_motion_euclidean_from_previous.png",
        "requested_raw_level_motion_mahalanobis_from_previous.png",
        "requested_raw_level_motion_previous_combined.png",
    )


if __name__ == "__main__":
    main()
