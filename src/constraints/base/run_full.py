"""
Full experiment — all 10 chains, both BERT and Pythia-2.8B.

Saves to:
  results/full/               BERT plots + geometry_summary.csv + results.json
  results/full_pythia/        Pythia plots + geometry_summary.csv + results.json
  results/full/run_full.log   Full progress log (both models)

Reuses run_analysis() from run_pilot.py — all measures, JSON cache, same pipeline.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import logging
from run_pilot import run_analysis, _make_logger, save_results_json

from config import (ALL_CHAINS, N_LEVELS,
                    RESULTS_DIR,
                    MODEL_NAME, N_LAYERS, POOLING,
                    PYTHIA_MODEL_NAME, PYTHIA_N_LAYERS, PYTHIA_POOLING,
                    PYTHIA_TORCH_DTYPE)
from data_loader import load_dataset, get_chain_info
from extract_activations import extract_and_cache

FULL_DIR        = os.path.join(RESULTS_DIR, "full")
FULL_PYTHIA_DIR = os.path.join(RESULTS_DIR, "full_pythia")


def run():
    os.makedirs(FULL_DIR, exist_ok=True)
    log_path = os.path.join(FULL_DIR, "run_full.log")
    logger   = _make_logger(log_path)

    logger.info("=" * 60)
    logger.info("FULL EXPERIMENT — 10 chains")
    logger.info("=" * 60)

    df_all     = load_dataset()
    chain_info = get_chain_info(df_all)
    logger.info(f"Sentences: {len(df_all)}  |  Chains: {len(ALL_CHAINS)}")

    # ── BERT ──────────────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("MODEL: BERT-base-uncased")
    logger.info("─" * 60)
    bert_cache = os.path.join(FULL_DIR, "acts_full.npy")
    bert_acts  = extract_and_cache(
        df_all, bert_cache,
        model_name=MODEL_NAME, n_layers=N_LAYERS, pooling=POOLING,
        torch_dtype="float32")
    logger.info(f"BERT activations shape: {bert_acts.shape}")

    bert_results = run_analysis(
        bert_acts, df_all, chain_info,
        output_dir=FULL_DIR,
        model_label="BERT-base (full)",
        n_layers=N_LAYERS,
        logger=logger,
        chains=ALL_CHAINS)

    # ── Pythia 2.8B ───────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("MODEL: Pythia-2.8B")
    logger.info("─" * 60)
    os.makedirs(FULL_PYTHIA_DIR, exist_ok=True)
    pythia_cache = os.path.join(FULL_PYTHIA_DIR, "acts_full_pythia.npy")
    pythia_acts  = extract_and_cache(
        df_all, pythia_cache,
        model_name=PYTHIA_MODEL_NAME,
        n_layers=PYTHIA_N_LAYERS,
        pooling=PYTHIA_POOLING,
        torch_dtype=PYTHIA_TORCH_DTYPE,
        batch_size=4)
    logger.info(f"Pythia activations shape: {pythia_acts.shape}")

    pythia_results = run_analysis(
        pythia_acts, df_all, chain_info,
        output_dir=FULL_PYTHIA_DIR,
        model_label="Pythia-2.8B (full)",
        n_layers=PYTHIA_N_LAYERS,
        logger=logger,
        chains=ALL_CHAINS)

    logger.info("=" * 60)
    logger.info("FULL EXPERIMENT COMPLETE")
    logger.info(f"  BERT results   → {FULL_DIR}")
    logger.info(f"  Pythia results → {FULL_PYTHIA_DIR}")
    logger.info(f"  Log            → {log_path}")
    logger.info("=" * 60)

    return {"bert": bert_results, "pythia": pythia_results}


if __name__ == "__main__":
    run()
