#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1

resolve_python() {
  local candidate
  for candidate in "${PYTHON_BIN:-}" "$(command -v python3 2>/dev/null || true)" /opt/anaconda3/bin/python3 /usr/bin/python3; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if "$candidate" - <<'PY' >/dev/null 2>&1
import numpy
PY
    then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "No suitable Python interpreter found. Set PYTHON_BIN to an environment with numpy installed." >&2
  exit 1
}

PYTHON_BIN="$(resolve_python)"
export PYTHON_BIN

copy_result() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
  fi
}

if [[ ! -f "$ROOT/results/constraints/base/full_pythia/acts_full_pythia.npy" ]]; then
  echo "Missing refinement activations in results/constraints/base/full_pythia" >&2
  echo "Run scripts/extract_constraints.sh first." >&2
  exit 1
fi

if [[ ! -f "$ROOT/results/constraints/base/control_pythia/acts_control_pythia.npy" ]]; then
  echo "Missing control activations in results/constraints/base/control_pythia" >&2
  echo "Run scripts/extract_constraints.sh first." >&2
  exit 1
fi

if [[ ! -f "$ROOT/data/multilingual/activations_all_families.npz" ]]; then
  echo "Missing multilingual calibration cache in data/multilingual" >&2
  echo "Run scripts/extract_multilingual.sh first." >&2
  exit 1
fi

cd "$ROOT/src/constraints/robust"

"$PYTHON_BIN" -B run_robust.py
"$PYTHON_BIN" -B make_requested_graphs.py
"$PYTHON_BIN" -B make_raw_metric_snapshots.py
"$PYTHON_BIN" -B make_raw_euclidean_delta.py
"$PYTHON_BIN" -B make_level_motion_graphs.py

copy_result "$ROOT/results/constraints/robust/paper_raw_delta_across_layers.png" "$ROOT/results/constraints/robust/raw_pr_delta_layers.png"
copy_result "$ROOT/results/constraints/robust/paper_raw_snapshot_layer25_purelevels.png" "$ROOT/results/constraints/robust/raw_pr_snapshot_layer25.png"
copy_result "$ROOT/results/constraints/robust/requested_raw_euclidean_delta_across_layers.png" "$ROOT/results/constraints/robust/raw_euclidean_delta_layers.png"
copy_result "$ROOT/results/constraints/robust/requested_raw_langresid_snapshot_layer25_purelevels.png" "$ROOT/results/constraints/robust/raw_langresid_snapshot_layer25.png"
copy_result "$ROOT/results/constraints/robust/requested_raw_level_motion_initial_combined.png" "$ROOT/results/constraints/robust/raw_motion_from_initial.png"
copy_result "$ROOT/results/constraints/robust/requested_raw_level_motion_innovation_vs_previous.png" "$ROOT/results/constraints/robust/raw_innovation_ratio.png"
copy_result "$ROOT/results/constraints/robust/requested_raw_level_motion_mahalanobis_from_previous.png" "$ROOT/results/constraints/robust/raw_mahalanobis_motion_prev.png"
copy_result "$ROOT/results/constraints/robust/requested_raw_level_motion_previous_combined.png" "$ROOT/results/constraints/robust/raw_motion_from_previous.png"
copy_result "$ROOT/results/constraints/robust/requested_raw_mle_logvol_delta_across_layers.png" "$ROOT/results/constraints/robust/raw_id_logvol_delta_layers.png"
copy_result "$ROOT/results/constraints/robust/requested_raw_mle_logvol_snapshot_layer22_purelevels.png" "$ROOT/results/constraints/robust/raw_id_logvol_layer22.png"
copy_result "$ROOT/results/constraints/robust/requested_raw_mle_logvol_snapshot_layer25_purelevels.png" "$ROOT/results/constraints/robust/raw_id_logvol_layer25.png"
copy_result "$ROOT/results/constraints/robust/requested_top0_top9_pure_level_pr.csv" "$ROOT/results/constraints/robust/pr_by_layer_top0_top9.csv"
copy_result "$ROOT/results/constraints/robust/requested_top10_top19_pure_level_pr.csv" "$ROOT/results/constraints/robust/pr_by_layer_top10_top19.csv"
