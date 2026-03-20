#!/usr/bin/env python3
"""Compute raw pure-level Euclidean and Mahalanobis spread summaries."""

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

from run_robust import COND_COLORS, CTRL_ACTS, REF_ACTS, load_control_df, load_refinement_df, whiten_coords


LEVELS = [0, 1, 2, 3, 4]


def build_pure_index(df: pd.DataFrame) -> tuple[list[str], dict[tuple[str, int], np.ndarray]]:
    chains = sorted(df["chain_id"].unique())
    chain_arr = df["chain_id"].to_numpy()
    level_arr = df["level"].to_numpy(dtype=int)
    index = {}
    for chain in chains:
        chain_mask = chain_arr == chain
        for level in LEVELS:
            index[(chain, level)] = np.where(chain_mask & (level_arr == level))[0]
    return chains, index


def mean_pairwise_l2(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return 0.0
    X = X.astype(np.float32, copy=False)
    sq = np.sum(X * X, axis=1)
    l2_sq = np.clip(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0, None)
    ii, jj = np.triu_indices(X.shape[0], k=1)
    return float(np.sqrt(l2_sq[ii, jj]).mean())


def compute_level_metrics(
    condition: str,
    acts: np.ndarray,
    chains: list[str],
    pure_index: dict[tuple[str, int], np.ndarray],
) -> pd.DataFrame:
    rows = []
    for layer in range(acts.shape[1]):
        X_layer = acts[:, layer, :].astype(np.float32, copy=False)
        X_mahal = whiten_coords(X_layer)
        for chain in chains:
            for level in LEVELS:
                idx = pure_index[(chain, level)]
                rows.append(
                    {
                        "metric": "euclidean",
                        "condition": condition,
                        "layer": layer,
                        "chain": chain,
                        "level": level,
                        "value": mean_pairwise_l2(X_layer[idx]),
                    }
                )
                rows.append(
                    {
                        "metric": "mahalanobis",
                        "condition": condition,
                        "layer": layer,
                        "chain": chain,
                        "level": level,
                        "value": mean_pairwise_l2(X_mahal[idx]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_delta(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["euclidean", "mahalanobis"]:
        for condition in ["refinement", "control"]:
            for layer in sorted(df["layer"].unique()):
                vals = (
                    df[
                        (df["metric"] == metric)
                        & (df["condition"] == condition)
                        & (df["layer"] == layer)
                    ]
                    .pivot(index="chain", columns="level", values="value")
                    .reindex(columns=LEVELS)
                )
                delta = vals[4] - vals[0]
                rows.append(
                    {
                        "metric": metric,
                        "condition": condition,
                        "layer": int(layer),
                        "mean_delta": float(delta.mean()),
                        "std_delta": float(delta.std(ddof=0)),
                        "n_chains": int(delta.shape[0]),
                    }
                )
    return pd.DataFrame(rows)


def plot_delta(df_delta: pd.DataFrame, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    sub_metric = df_delta[df_delta["metric"] == metric].copy()
    for condition in ["refinement", "control"]:
        sub = sub_metric[sub_metric["condition"] == condition].sort_values("layer")
        x = sub["layer"].to_numpy()
        mu = sub["mean_delta"].to_numpy()
        sd = sub["std_delta"].to_numpy()
        ax.plot(x, mu, color=COND_COLORS[condition], linewidth=3, label=condition.capitalize())
        ax.fill_between(x, mu - sd, mu + sd, color=COND_COLORS[condition], alpha=0.18)
    ax.set_xlabel("Layer", fontsize=14)
    if metric == "euclidean":
        ax.set_ylabel("Euclidean delta: pure L4 minus pure L0", fontsize=14)
        ax.set_title("Raw pure-level Euclidean delta across layers: L4 minus L0", fontsize=18)
        filename = "requested_raw_euclidean_delta_across_layers.png"
    else:
        ax.set_ylabel("Mahalanobis delta: pure L4 minus pure L0", fontsize=14)
        ax.set_title("Raw pure-level Mahalanobis delta across layers: L4 minus L0", fontsize=18)
        filename = "requested_raw_mahalanobis_delta_across_layers.png"
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    ref_df = load_refinement_df()
    ctrl_df = load_control_df()
    ref_acts = np.load(REF_ACTS, mmap_mode="r")
    ctrl_acts = np.load(CTRL_ACTS, mmap_mode="r")

    ref_chains, ref_index = build_pure_index(ref_df)
    ctrl_chains, ctrl_index = build_pure_index(ctrl_df)

    df_levels = pd.concat(
        [
            compute_level_metrics("refinement", ref_acts, ref_chains, ref_index),
            compute_level_metrics("control", ctrl_acts, ctrl_chains, ctrl_index),
        ],
        ignore_index=True,
    )
    df_levels.to_csv(RESULTS_DIR / "requested_raw_distance_pure_level_metrics.csv", index=False)
    df_levels[df_levels["metric"] == "euclidean"].to_csv(
        RESULTS_DIR / "requested_raw_euclidean_pure_level_metrics.csv", index=False
    )
    df_levels[df_levels["metric"] == "mahalanobis"].to_csv(
        RESULTS_DIR / "requested_raw_mahalanobis_pure_level_metrics.csv", index=False
    )

    df_delta = summarize_delta(df_levels)
    df_delta.to_csv(RESULTS_DIR / "requested_raw_distance_delta_across_layers_summary.csv", index=False)
    df_delta[df_delta["metric"] == "euclidean"].to_csv(
        RESULTS_DIR / "requested_raw_euclidean_delta_across_layers_summary.csv", index=False
    )
    df_delta[df_delta["metric"] == "mahalanobis"].to_csv(
        RESULTS_DIR / "requested_raw_mahalanobis_delta_across_layers_summary.csv", index=False
    )
    plot_delta(df_delta, "euclidean")
    plot_delta(df_delta, "mahalanobis")


if __name__ == "__main__":
    main()
