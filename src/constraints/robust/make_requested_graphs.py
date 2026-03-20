#!/usr/bin/env python3
"""Recreate the user's preferred figure styles with corrected layerwise analysis."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "constraints" / "robust"
MPL_DIR = RESULTS_DIR / ".mplcache"
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_robust import (
    ROOT as _ROOT_CHECK,
    COND_COLORS,
    BRANCH_LABELS,
    REF_ACTS,
    CTRL_ACTS,
    CALIBRATION_ACTS,
    load_refinement_df,
    load_control_df,
    compute_language_bases_and_alignment,
    compute_condition_models,
    residualize_basis,
    whiten_coords,
    top_pc_basis_family,
)


assert ROOT == _ROOT_CHECK

LEVELS = list(range(5))
LEVEL_LABELS = ["L0", "L0-L1", "L0-L2", "L0-L3", "L0-L4"]
WHITEN_BRANCHES = ["raw", "zca_raw", "deflate_top5", "lang_resid"]
WHITEN_LABELS = {
    "raw": "Raw",
    "zca_raw": "ZCA",
    "deflate_top5": "Top-5 deflation",
    "lang_resid": "Language residualized",
}
DELTA_CORE_BRANCHES = ["raw", "sem_proj", "surface_resid", "whitened_sem"]
DELTA_EXT_BRANCHES = [
    "raw",
    "sem_proj",
    "whitened_sem",
    "lang_resid",
    "surface_lang_resid",
    "deflate_top5",
]
FLOW_EXT_BRANCHES = [
    "raw",
    "surface_resid",
    "lang_resid",
    "surface_lang_resid",
    "deflate_top3",
    "deflate_top5",
]
METRIC_LABELS = {
    "mle_id": "Delta MLE ID (L4-L0)",
    "log_vol": "Delta Log-volume (L4-L0)",
}


def build_cumulative_index(df: pd.DataFrame):
    chains = sorted(df["chain_id"].unique())
    chain_arr = df["chain_id"].values
    level_arr = df["level"].values.astype(int)
    index = {}
    for chain in chains:
        chain_mask = chain_arr == chain
        for lmax in LEVELS:
            index[(chain, lmax)] = np.where(chain_mask & (level_arr <= lmax))[0]
    return chains, index


def build_pure_index(df: pd.DataFrame):
    chains = sorted(df["chain_id"].unique())
    chain_arr = df["chain_id"].values
    level_arr = df["level"].values.astype(int)
    index = {}
    for chain in chains:
        chain_mask = chain_arr == chain
        for level in LEVELS:
            index[(chain, level)] = np.where(chain_mask & (level_arr == level))[0]
    return chains, index


def whitening_branch_states(
    X: np.ndarray,
    lang_basis: np.ndarray,
    top5_basis: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "raw": X.astype(np.float32, copy=False),
        "zca_raw": whiten_coords(X),
        "deflate_top5": residualize_basis(X, top5_basis),
        "lang_resid": residualize_basis(X, lang_basis),
    }


def participation_ratio_gram(X: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    if Xc.shape[0] < 2:
        return 0.0
    gram = (Xc @ Xc.T) / max(Xc.shape[0] - 1, 1)
    eigs = np.linalg.eigvalsh(gram)[::-1]
    eigs = eigs[eigs > 1e-12]
    if eigs.size == 0:
        return 0.0
    return float((eigs.sum() ** 2) / np.square(eigs).sum())


def compute_whitening_like_pr(
    condition: str,
    acts: np.ndarray,
    chains: list[str],
    cum_index: dict[tuple[str, int], np.ndarray],
    models: dict,
    lang_bases: list[np.ndarray],
) -> pd.DataFrame:
    rows = []
    for layer in range(acts.shape[1]):
        X = acts[:, layer, :].astype(np.float32, copy=False)
        branches = whitening_branch_states(
            X,
            lang_bases[layer],
            models["top_bases"][5][layer],
        )
        for branch, Xb in branches.items():
            for chain in chains:
                for lmax in LEVELS:
                    idx = cum_index[(chain, lmax)]
                    rows.append(
                        {
                            "condition": condition,
                            "layer": layer,
                            "branch": branch,
                            "chain": chain,
                            "level_set": lmax,
                            "pr": participation_ratio_gram(Xb[idx]),
                        }
                    )
    return pd.DataFrame(rows)


def compute_whitening_pure_level_pr(
    condition: str,
    acts: np.ndarray,
    chains: list[str],
    pure_index: dict[tuple[str, int], np.ndarray],
    models: dict,
    lang_bases: list[np.ndarray],
) -> pd.DataFrame:
    rows = []
    for layer in range(acts.shape[1]):
        X = acts[:, layer, :].astype(np.float32, copy=False)
        branches = whitening_branch_states(
            X,
            lang_bases[layer],
            models["top_bases"][5][layer],
        )
        for branch, Xb in branches.items():
            for chain in chains:
                for level in LEVELS:
                    idx = pure_index[(chain, level)]
                    rows.append(
                        {
                            "condition": condition,
                            "layer": layer,
                            "branch": branch,
                            "chain": chain,
                            "level": level,
                            "pr": participation_ratio_gram(Xb[idx]),
                        }
                    )
    return pd.DataFrame(rows)


def plot_whitening_snapshot_at_layer(
    df_pr: pd.DataFrame,
    layer_value: int,
    filename: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(WHITEN_BRANCHES), figsize=(18, 4))
    fig.suptitle(title, fontsize=13)
    for ax, branch in zip(axes, WHITEN_BRANCHES):
        sub = df_pr[(df_pr["layer"] == layer_value) & (df_pr["branch"] == branch)]
        for condition in ["refinement", "control"]:
            vals = (
                sub[sub["condition"] == condition]
                .pivot(index="chain", columns="level_set", values="pr")
                .sort_index(axis=1)
            )
            mu = vals.mean(axis=0).values
            sd = vals.std(axis=0, ddof=0).values
            x = np.arange(len(LEVELS))
            ax.plot(x, mu, color=COND_COLORS[condition], linewidth=2, label=condition.capitalize())
            ax.fill_between(x, mu - sd, mu + sd, color=COND_COLORS[condition], alpha=0.15)
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels(LEVEL_LABELS, rotation=30, ha="right")
        ax.set_title(WHITEN_LABELS[branch])
        ax.set_ylabel("Participation ratio")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def plot_whitening_snapshot(df_pr: pd.DataFrame, shared_layer: int) -> None:
    plot_whitening_snapshot_at_layer(
        df_pr,
        shared_layer,
        "requested_whitening_snapshot_shared_layer.png",
        f"Corrected whitening-style comparison at shared layer {shared_layer}",
    )


def compute_whitening_pure_level_snapshot_data(
    condition: str,
    acts: np.ndarray,
    chains: list[str],
    pure_index: dict[tuple[str, int], np.ndarray],
    models: dict,
    lang_bases: list[np.ndarray],
    layer_value: int,
) -> pd.DataFrame:
    rows = []
    X = acts[:, layer_value, :].astype(np.float32, copy=False)
    branches = whitening_branch_states(
        X,
        lang_bases[layer_value],
        models["top_bases"][5][layer_value],
    )
    for branch, Xb in branches.items():
        for chain in chains:
            for level in LEVELS:
                idx = pure_index[(chain, level)]
                rows.append(
                    {
                        "condition": condition,
                        "layer": layer_value,
                        "branch": branch,
                        "chain": chain,
                        "level": level,
                        "pr": participation_ratio_gram(Xb[idx]),
                    }
                )
    return pd.DataFrame(rows)


def plot_whitening_pure_level_snapshot(
    df_pure: pd.DataFrame,
    layer_value: int,
    filename: str,
    title: str,
) -> None:
    pure_labels = ["L0", "L1", "L2", "L3", "L4"]
    fig, axes = plt.subplots(1, len(WHITEN_BRANCHES), figsize=(18, 4))
    fig.suptitle(title, fontsize=13)
    for ax, branch in zip(axes, WHITEN_BRANCHES):
        sub = df_pure[(df_pure["layer"] == layer_value) & (df_pure["branch"] == branch)]
        for condition in ["refinement", "control"]:
            vals = (
                sub[sub["condition"] == condition]
                .pivot(index="chain", columns="level", values="pr")
                .sort_index(axis=1)
            )
            mu = vals.mean(axis=0).values
            sd = vals.std(axis=0, ddof=0).values
            x = np.arange(len(LEVELS))
            ax.plot(x, mu, color=COND_COLORS[condition], linewidth=2, label=condition.capitalize())
            ax.fill_between(x, mu - sd, mu + sd, color=COND_COLORS[condition], alpha=0.15)
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels(pure_labels)
        ax.set_title(WHITEN_LABELS[branch])
        ax.set_ylabel("Participation ratio")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def plot_whitening_heatmaps(df_pr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, len(WHITEN_BRANCHES), figsize=(18, 8), sharex=True, sharey=True)
    fig.suptitle("Whitening-style cumulative PR across all layers", fontsize=13)
    for row_i, condition in enumerate(["refinement", "control"]):
        for col_i, branch in enumerate(WHITEN_BRANCHES):
            ax = axes[row_i, col_i]
            sub = df_pr[(df_pr["condition"] == condition) & (df_pr["branch"] == branch)]
            mat = (
                sub.groupby(["layer", "level_set"])["pr"]
                .mean()
                .unstack("level_set")
                .reindex(index=range(33), columns=LEVELS)
                .T
                .values
            )
            im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower")
            if row_i == 0:
                ax.set_title(WHITEN_LABELS[branch])
            if col_i == 0:
                ax.set_ylabel(f"{condition.capitalize()}\nCumulative level set")
            ax.set_xlabel("Layer")
            ax.set_yticks(range(len(LEVELS)))
            ax.set_yticklabels(LEVEL_LABELS)
            ax.set_xticks(range(0, 33, 4))
            ax.set_xticklabels(range(0, 33, 4))
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Mean PR")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "requested_whitening_heatmaps.png", dpi=150)
    plt.close(fig)


def plot_whitening_delta_across_layers(df_pr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(WHITEN_BRANCHES), figsize=(18, 4), sharey=True)
    fig.suptitle("Whitening-style PR difference across layers: pure L4 minus pure L0", fontsize=13)
    for ax, branch in zip(axes, WHITEN_BRANCHES):
        sub = df_pr[df_pr["branch"] == branch]
        for condition in ["refinement", "control"]:
            deltas = []
            for layer in range(33):
                row = (
                    sub[(sub["condition"] == condition) & (sub["layer"] == layer)]
                    .pivot(index="chain", columns="level", values="pr")
                    .sort_index(axis=1)
                )
                d = row[4] - row[0]
                deltas.append((float(d.mean()), float(d.std(ddof=0))))
            mu = np.array([x[0] for x in deltas])
            sd = np.array([x[1] for x in deltas])
            x = np.arange(33)
            ax.plot(x, mu, color=COND_COLORS[condition], linewidth=2, label=condition.capitalize())
            ax.fill_between(x, mu - sd, mu + sd, color=COND_COLORS[condition], alpha=0.15)
        ax.set_title(WHITEN_LABELS[branch])
        ax.set_xlabel("Layer")
        ax.set_ylabel("PR delta: pure L4 minus pure L0")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "requested_whitening_delta_across_layers.png", dpi=150)
    plt.close(fig)


def plot_top5_pr_by_layer(df_pr: pd.DataFrame) -> None:
    sub = df_pr[df_pr["branch"] == "deflate_top5"].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.viridis(np.linspace(0.12, 0.88, len(LEVELS)))
    for level_set, color in zip(LEVELS, cmap):
        for condition, ls in [("refinement", "-"), ("control", "--")]:
            row = (
                sub[(sub["condition"] == condition) & (sub["level_set"] == level_set)]
                .groupby("layer")["pr"]
                .mean()
                .reindex(range(33))
            )
            label = f"{condition.capitalize()} {LEVEL_LABELS[level_set]}"
            ax.plot(
                row.index,
                row.values,
                color=color,
                linestyle=ls,
                linewidth=2,
                label=label,
            )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Participation ratio")
    ax.set_title("Top-5 deflation: cumulative PR across layers")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "requested_top5_pr_by_layer.png", dpi=150)
    plt.close(fig)


def compute_top5_pure_level_pr(
    condition: str,
    acts: np.ndarray,
    chains: list[str],
    pure_index: dict[tuple[str, int], np.ndarray],
) -> pd.DataFrame:
    rows = []
    for layer in range(acts.shape[1]):
        X = acts[:, layer, :].astype(np.float32, copy=False)
        top5_basis = top_pc_basis_family(X, 5)[5]
        X_def = residualize_basis(X, top5_basis)
        for chain in chains:
            for level in LEVELS:
                idx = pure_index[(chain, level)]
                rows.append(
                    {
                        "condition": condition,
                        "layer": layer,
                        "chain": chain,
                        "level": level,
                        "pr": participation_ratio_gram(X_def[idx]),
                    }
                )
    return pd.DataFrame(rows)


def plot_top5_pure_level_pr_by_layer(df_pr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.viridis(np.linspace(0.12, 0.88, len(LEVELS)))
    pure_labels = ["L0", "L1", "L2", "L3", "L4"]
    for level, color, label in zip(LEVELS, cmap, pure_labels):
        for condition, ls in [("refinement", "-"), ("control", "--")]:
            row = (
                df_pr[(df_pr["condition"] == condition) & (df_pr["level"] == level)]
                .groupby("layer")["pr"]
                .mean()
                .reindex(range(33))
            )
            ax.plot(
                row.index,
                row.values,
                color=color,
                linestyle=ls,
                linewidth=2,
                label=f"{condition.capitalize()} {label}",
            )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Participation ratio")
    ax.set_title("Top-5 deflation: PR across layers using only the 8 sentences of each level")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "requested_top5_pure_level_pr_by_layer.png", dpi=150)
    plt.close(fig)


def plot_delta_snapshot(
    df_delta: pd.DataFrame,
    shared_layer: int,
    branches: list[str],
    filename: str,
    title: str,
) -> None:
    metrics = ["mle_id", "log_vol"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13)
    x = np.arange(len(branches))
    width = 0.35
    for ax, metric in zip(axes, metrics):
        sub = df_delta[(df_delta["layer"] == shared_layer) & (df_delta["metric"] == metric)]
        for i, condition in enumerate(["refinement", "control"]):
            vals = []
            errs = []
            for branch in branches:
                row = sub[(sub["condition"] == condition) & (sub["branch"] == branch)]
                vals.append(float(row["mean_delta"].iloc[0]))
                errs.append(float(row["std_delta"].iloc[0]))
            offset = -width / 2 if condition == "refinement" else width / 2
            ax.bar(
                x + offset,
                vals,
                width,
                yerr=errs,
                capsize=4,
                color=COND_COLORS[condition],
                alpha=0.85,
                label=condition.capitalize(),
            )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([BRANCH_LABELS[b] for b in branches], rotation=20, ha="right")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def plot_delta_across_layers(
    df_delta: pd.DataFrame,
    branches: list[str],
    filename: str,
    title: str,
) -> None:
    metrics = ["mle_id", "log_vol"]
    fig, axes = plt.subplots(2, len(branches), figsize=(4.2 * len(branches), 8), sharex=True)
    fig.suptitle(title, fontsize=13)
    for col_i, branch in enumerate(branches):
        for row_i, metric in enumerate(metrics):
            ax = axes[row_i, col_i]
            sub = df_delta[(df_delta["branch"] == branch) & (df_delta["metric"] == metric)]
            for condition in ["refinement", "control"]:
                row = sub[sub["condition"] == condition].sort_values("layer")
                ax.plot(
                    row["layer"],
                    row["mean_delta"],
                    color=COND_COLORS[condition],
                    linewidth=2,
                    label=condition.capitalize(),
                )
                ax.fill_between(
                    row["layer"],
                    row["mean_delta"] - row["std_delta"],
                    row["mean_delta"] + row["std_delta"],
                    color=COND_COLORS[condition],
                    alpha=0.15,
                )
            if row_i == 0:
                ax.set_title(BRANCH_LABELS[branch])
            if col_i == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_xlabel("Layer")
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def write_notes(shared_layer: int) -> None:
    lines = [
        "# Requested Graph Notes",
        "",
        "## How the original figures were made",
        "",
        "- `whitening_comparison_Pythia-2.8B.png` used cumulative level-set PR trajectories.",
        "- But those trajectories were averaged over a dense middle-layer band, not shown layer by layer.",
        "- `delta_pythia.png` used per-level `L4-L0` deltas at a single fixed layer.",
        "- Full ZCA PR in the `N << D` regime is close to a rank-counting upper bound, so it can look strong while being mostly algebraic.",
        "",
        "## What is corrected here",
        "",
        f"- The whitening-style snapshot now uses one explicit shared layer: `{shared_layer}`.",
        "- The across-layer whitening figures show the same cumulative PR information for every layer `0..32`.",
        "- `requested_whitening_delta_across_layers.png` is now corrected: it uses pure-level `PR(L4) - PR(L0)`, not cumulative sets.",
        "- The delta-style snapshot uses the corrected robust static metrics at the shared layer.",
        "- The across-layer delta figures show those same `L4-L0` statistics across all layers.",
        "",
        "## Files",
        "",
        "- `requested_whitening_snapshot_shared_layer.png`",
        "- `requested_whitening_snapshot_layer10.png`",
        "- `requested_whitening_snapshot_layer10_purelevels.png`",
        "- `requested_whitening_heatmaps.png`",
        "- `requested_whitening_delta_across_layers.png`",
        "- `requested_whitening_pure_level_pr.csv`",
        "- `requested_top5_pr_by_layer.png`",
        "- `requested_top5_pure_level_pr_by_layer.png`",
        "- `requested_delta_snapshot_shared_layer_core.png`",
        "- `requested_delta_snapshot_shared_layer_extended.png`",
        "- `requested_delta_across_layers_core.png`",
        "- `requested_delta_across_layers_extended.png`",
        "- `requested_flow_across_transitions_extended.png`",
    ]
    (RESULTS_DIR / "requested_graphs_notes.md").write_text("\n".join(lines))


def plot_flow_across_transitions(df_flow: pd.DataFrame) -> None:
    metric = "mle_id"
    branches = FLOW_EXT_BRANCHES
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    fig.suptitle("Flow delta across transitions (requested-style view)", fontsize=13)
    for ax, branch in zip(axes.flat, branches):
        sub = df_flow[(df_flow["branch"] == branch) & (df_flow["metric"] == metric)]
        for condition in ["refinement", "control"]:
            row = sub[sub["condition"] == condition].sort_values("transition")
            ax.plot(
                row["transition"],
                row["mean_delta"],
                color=COND_COLORS[condition],
                linewidth=2,
                label=condition.capitalize(),
            )
            ax.fill_between(
                row["transition"],
                row["mean_delta"] - row["std_delta"],
                row["mean_delta"] + row["std_delta"],
                color=COND_COLORS[condition],
                alpha=0.15,
            )
        ax.set_title(BRANCH_LABELS[branch])
        ax.set_xlabel("Transition")
        ax.set_ylabel("Delta MLE ID (L4-L0)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "requested_flow_across_transitions_extended.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print("[requested] loading data")
    ref_df = load_refinement_df()
    ctrl_df = load_control_df()
    ref_acts = np.load(REF_ACTS).astype(np.float32, copy=False)
    ctrl_acts = np.load(CTRL_ACTS).astype(np.float32, copy=False)
    cal = np.load(CALIBRATION_ACTS, allow_pickle=True)
    cal_acts = cal["activations"].astype(np.float32, copy=False)
    cal_langs = cal["languages"].astype(str)

    print("[requested] recomputing robust models needed for whitening-style plots")
    lang_bases, _ = compute_language_bases_and_alignment(cal_acts, cal_langs)
    ref_models = compute_condition_models("refinement-requested", ref_acts, ref_df)
    ctrl_models = compute_condition_models("control-requested", ctrl_acts, ctrl_df)

    layer_sel = pd.read_csv(RESULTS_DIR / "layer_selection.csv")
    shared_layer = int(layer_sel.groupby("layer")["shared_score"].first().idxmax())

    ref_chains, ref_cum = build_cumulative_index(ref_df)
    ctrl_chains, ctrl_cum = build_cumulative_index(ctrl_df)
    ref_pure_chains, ref_pure = build_pure_index(ref_df)
    ctrl_pure_chains, ctrl_pure = build_pure_index(ctrl_df)

    print("[requested] computing corrected whitening-style cumulative PR")
    df_pr = pd.concat(
        [
            compute_whitening_like_pr("refinement", ref_acts, ref_chains, ref_cum, ref_models, lang_bases),
            compute_whitening_like_pr("control", ctrl_acts, ctrl_chains, ctrl_cum, ctrl_models, lang_bases),
        ],
        ignore_index=True,
    )
    df_pr.to_csv(RESULTS_DIR / "requested_whitening_cumulative_pr.csv", index=False)

    print("[requested] making whitening-style figures")
    plot_whitening_snapshot(df_pr, shared_layer)
    plot_whitening_snapshot_at_layer(
        df_pr,
        10,
        "requested_whitening_snapshot_layer10.png",
        "Corrected whitening-style comparison at layer 10",
    )
    plot_whitening_heatmaps(df_pr)
    plot_top5_pr_by_layer(df_pr)

    print("[requested] computing pure-level whitening PR across layers")
    df_whitening_pure = pd.concat(
        [
            compute_whitening_pure_level_pr(
                "refinement", ref_acts, ref_pure_chains, ref_pure, ref_models, lang_bases
            ),
            compute_whitening_pure_level_pr(
                "control", ctrl_acts, ctrl_pure_chains, ctrl_pure, ctrl_models, lang_bases
            ),
        ],
        ignore_index=True,
    )
    df_whitening_pure.to_csv(RESULTS_DIR / "requested_whitening_pure_level_pr.csv", index=False)
    plot_whitening_delta_across_layers(df_whitening_pure)

    print("[requested] computing pure-level top-5 PR across layers")
    df_top5_pure = pd.concat(
        [
            compute_top5_pure_level_pr("refinement", ref_acts, ref_pure_chains, ref_pure),
            compute_top5_pure_level_pr("control", ctrl_acts, ctrl_pure_chains, ctrl_pure),
        ],
        ignore_index=True,
    )
    df_top5_pure.to_csv(RESULTS_DIR / "requested_top5_pure_level_pr.csv", index=False)
    plot_top5_pure_level_pr_by_layer(df_top5_pure)

    print("[requested] making pure-level whitening-style snapshot at layer 10")
    df_pure_snapshot = pd.concat(
        [
            compute_whitening_pure_level_snapshot_data(
                "refinement", ref_acts, ref_pure_chains, ref_pure, ref_models, lang_bases, 10
            ),
            compute_whitening_pure_level_snapshot_data(
                "control", ctrl_acts, ctrl_pure_chains, ctrl_pure, ctrl_models, lang_bases, 10
            ),
        ],
        ignore_index=True,
    )
    df_pure_snapshot.to_csv(
        RESULTS_DIR / "requested_whitening_snapshot_layer10_purelevels.csv",
        index=False,
    )
    plot_whitening_pure_level_snapshot(
        df_pure_snapshot,
        10,
        "requested_whitening_snapshot_layer10_purelevels.png",
        "Corrected whitening-style comparison at layer 10 using pure levels only",
    )

    print("[requested] making delta-style figures from robust static outputs")
    df_delta = pd.read_csv(RESULTS_DIR / "static_delta_by_layer.csv")
    plot_delta_snapshot(
        df_delta,
        shared_layer,
        DELTA_CORE_BRANCHES,
        "requested_delta_snapshot_shared_layer_core.png",
        f"Corrected delta-style summary at shared layer {shared_layer} (core branches)",
    )
    plot_delta_snapshot(
        df_delta,
        shared_layer,
        DELTA_EXT_BRANCHES,
        "requested_delta_snapshot_shared_layer_extended.png",
        f"Corrected delta-style summary at shared layer {shared_layer} (extended branches)",
    )
    plot_delta_across_layers(
        df_delta,
        DELTA_CORE_BRANCHES,
        "requested_delta_across_layers_core.png",
        "Delta-style summary across all layers (core branches)",
    )
    plot_delta_across_layers(
        df_delta,
        DELTA_EXT_BRANCHES,
        "requested_delta_across_layers_extended.png",
        "Delta-style summary across all layers (extended branches)",
    )

    print("[requested] making requested-style flow figure")
    df_flow = pd.read_csv(RESULTS_DIR / "flow_delta_by_transition.csv")
    plot_flow_across_transitions(df_flow)

    write_notes(shared_layer)
    print("[requested] done")


if __name__ == "__main__":
    main()
