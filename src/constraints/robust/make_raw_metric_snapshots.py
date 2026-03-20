#!/usr/bin/env python3
"""Create raw pure-level layer-25 snapshots for metrics beyond participation ratio."""

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

from run_robust import COND_COLORS


LEVELS = [0, 1, 2, 3, 4]
LEVEL_LABELS = [f"L{level}" for level in LEVELS]
SNAPSHOT_LAYERS = [25, 22]
METRIC_SPECS = {
    "mle_id": {
        "ylabel": "MLE ID",
    },
    "log_vol": {
        "ylabel": "Log-volume",
    },
}
DELTA_SPECS = {
    "mle_id": {
        "title": "Raw pure-level MLE ID delta across layers: L4 minus L0",
        "ylabel": "MLE ID delta: pure L4 minus pure L0",
        "filename": "requested_raw_mle_id_delta_across_layers.png",
    },
    "log_vol": {
        "title": "Raw pure-level log-volume delta across layers: L4 minus L0",
        "ylabel": "Log-volume delta: pure L4 minus pure L0",
        "filename": "requested_raw_log_vol_delta_across_layers.png",
    },
}


def snapshot_spec(metric: str, layer_value: int) -> dict[str, str]:
    return {
        "title": f"Raw pure-level comparison at layer {layer_value}",
        "ylabel": METRIC_SPECS[metric]["ylabel"],
        "filename": f"requested_raw_{metric}_snapshot_layer{layer_value}_purelevels.png",
    }


def combined_snapshot_filename(layer_value: int) -> str:
    return f"requested_raw_mle_logvol_snapshot_layer{layer_value}_purelevels.png"


def summarize_metric(df: pd.DataFrame, metric: str, layer_value: int) -> pd.DataFrame:
    sub = df[(df["layer"] == layer_value) & (df["branch"] == "raw")].copy()
    rows = []
    for condition in ["refinement", "control"]:
        vals = (
            sub[sub["condition"] == condition]
            .pivot(index="chain", columns="level", values=metric)
            .reindex(columns=LEVELS)
        )
        mu = vals.mean(axis=0).to_numpy()
        sd = vals.std(axis=0, ddof=0).to_numpy()
        for level, mean_value, std_value in zip(LEVELS, mu, sd, strict=True):
            rows.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "layer": layer_value,
                    "branch": "raw",
                    "level": level,
                    "mean": float(mean_value),
                    "std": float(std_value),
                }
            )
    return pd.DataFrame(rows)


def plot_metric(summary: pd.DataFrame, metric: str, layer_value: int) -> None:
    spec = snapshot_spec(metric, layer_value)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(LEVELS))
    for condition in ["refinement", "control"]:
        sub = summary[summary["condition"] == condition].sort_values("level")
        mu = sub["mean"].to_numpy()
        sd = sub["std"].to_numpy()
        ax.plot(x, mu, color=COND_COLORS[condition], linewidth=3, label=condition.capitalize())
        ax.fill_between(x, mu - sd, mu + sd, color=COND_COLORS[condition], alpha=0.18)
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_title(spec["title"], fontsize=18)
    ax.set_ylabel(spec["ylabel"], fontsize=14)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / spec["filename"], dpi=150)
    plt.close(fig)


def plot_combined(summary_df: pd.DataFrame, layer_value: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    x = np.arange(len(LEVELS))
    for ax, metric in zip(axes, METRIC_SPECS, strict=True):
        spec = snapshot_spec(metric, layer_value)
        summary = summary_df[summary_df["metric"] == metric]
        for condition in ["refinement", "control"]:
            sub = summary[summary["condition"] == condition].sort_values("level")
            mu = sub["mean"].to_numpy()
            sd = sub["std"].to_numpy()
            ax.plot(x, mu, color=COND_COLORS[condition], linewidth=3, label=condition.capitalize())
            ax.fill_between(x, mu - sd, mu + sd, color=COND_COLORS[condition], alpha=0.18)
        ax.set_xticks(x)
        ax.set_xticklabels(LEVEL_LABELS)
        ax.set_title(spec["ylabel"], fontsize=16)
        ax.set_ylabel(spec["ylabel"], fontsize=13)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=12)
    fig.suptitle(f"Raw pure-level comparison at layer {layer_value}", fontsize=18)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / combined_snapshot_filename(layer_value), dpi=150)
    plt.close(fig)


def summarize_delta_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[df["branch"] == "raw"].copy()
    rows = []
    for condition in ["refinement", "control"]:
        for layer in sorted(sub["layer"].unique()):
            vals = (
                sub[(sub["condition"] == condition) & (sub["layer"] == layer)]
                .pivot(index="chain", columns="level", values=metric)
                .reindex(columns=LEVELS)
            )
            delta = vals[4] - vals[0]
            rows.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "layer": int(layer),
                    "branch": "raw",
                    "mean_delta": float(delta.mean()),
                    "std_delta": float(delta.std(ddof=0)),
                    "n_chains": int(delta.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def plot_delta_metric(summary: pd.DataFrame, metric: str) -> None:
    spec = DELTA_SPECS[metric]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = summary["layer"].drop_duplicates().sort_values().to_numpy()
    for condition in ["refinement", "control"]:
        sub = summary[summary["condition"] == condition].sort_values("layer")
        mu = sub["mean_delta"].to_numpy()
        sd = sub["std_delta"].to_numpy()
        ax.plot(x, mu, color=COND_COLORS[condition], linewidth=3, label=condition.capitalize())
        ax.fill_between(x, mu - sd, mu + sd, color=COND_COLORS[condition], alpha=0.18)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel(spec["ylabel"], fontsize=14)
    ax.set_title(spec["title"], fontsize=18)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / spec["filename"], dpi=150)
    plt.close(fig)


def plot_delta_combined(summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
    for ax, metric in zip(axes, DELTA_SPECS, strict=True):
        spec = DELTA_SPECS[metric]
        summary = summary_df[summary_df["metric"] == metric].sort_values("layer")
        x = summary["layer"].drop_duplicates().to_numpy()
        for condition in ["refinement", "control"]:
            sub = summary[summary["condition"] == condition].sort_values("layer")
            mu = sub["mean_delta"].to_numpy()
            sd = sub["std_delta"].to_numpy()
            ax.plot(x, mu, color=COND_COLORS[condition], linewidth=3, label=condition.capitalize())
            ax.fill_between(x, mu - sd, mu + sd, color=COND_COLORS[condition], alpha=0.18)
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_ylabel(spec["ylabel"], fontsize=13)
        ax.set_title(spec["ylabel"], fontsize=16)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=12)
    fig.suptitle("Raw pure-level delta across layers: L4 minus L0", fontsize=18)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "requested_raw_mle_logvol_delta_across_layers.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "static_level_metrics.csv")
    snapshot_rows = []
    for layer_value in SNAPSHOT_LAYERS:
        layer_rows = []
        for metric in METRIC_SPECS:
            summary = summarize_metric(df, metric, layer_value)
            plot_metric(summary, metric, layer_value)
            layer_rows.append(summary)
        layer_df = pd.concat(layer_rows, ignore_index=True)
        plot_combined(layer_df, layer_value)
        layer_df.to_csv(
            RESULTS_DIR / f"requested_raw_metric_snapshots_layer{layer_value}_summary.csv",
            index=False,
        )
        snapshot_rows.append(layer_df)
    pd.concat(snapshot_rows, ignore_index=True).to_csv(
        RESULTS_DIR / "requested_raw_metric_snapshots_summary.csv",
        index=False,
    )

    delta_rows = []
    for metric in DELTA_SPECS:
        summary = summarize_delta_metric(df, metric)
        plot_delta_metric(summary, metric)
        delta_rows.append(summary)
    delta_df = pd.concat(delta_rows, ignore_index=True)
    plot_delta_combined(delta_df)
    delta_df.to_csv(
        RESULTS_DIR / "requested_raw_metric_delta_across_layers_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
