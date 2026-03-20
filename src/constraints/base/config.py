from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "constraints"
RESULTS_ROOT = PROJECT_ROOT / "results" / "constraints" / "base"

DATA_PATH = DATA_DIR / "semantic_refinement_dataset_v2_400.json"
CONTROL_DATA_PATH = DATA_DIR / "redundant_commonality_controls_400.json"
RESULTS_DIR = RESULTS_ROOT
PILOT_DIR = RESULTS_DIR / "pilot"
FULL_DIR = RESULTS_DIR / "full"
CONTROL_BERT_DIR = RESULTS_DIR / "control_bert"
CONTROL_PYTHIA_DIR = RESULTS_DIR / "control_pythia"
CONTRAST_DIR = RESULTS_DIR / "contrast"

# ── BERT ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
POOLING = "mean_content"   # mean over non-[CLS]/[SEP] tokens
N_LAYERS = 13              # embedding (L0) + 12 transformer layers

# ── Pythia 2.8B ───────────────────────────────────────────────────────────────
PYTHIA_MODEL_NAME  = "EleutherAI/pythia-2.8b"
PYTHIA_POOLING     = "mean_all"    # mean over all non-padding tokens (no CLS/SEP)
PYTHIA_N_LAYERS    = 33            # embedding (L0) + 32 transformer layers
PYTHIA_TORCH_DTYPE = "float32"     # full precision for geometry analysis
PYTHIA_PILOT_DIR = RESULTS_DIR / "pilot_pythia"

# Layer indices to focus dense analysis on
DENSE_LAYERS = list(range(4, 11))   # L4–L10
ALL_LAYERS = list(range(N_LAYERS))  # L0–L12

# Semantic subspace (Approach A)
SEM_SUBSPACE_DIM = 12    # LDA components for semantic projection
NUIS_SUBSPACE_DIM = 3    # LDA components for nuisance (surface_family has 4 classes → max 3)

# Whitening (Approach B)
DEFLATION_K = [5, 10]    # top-k PC deflation variants
WHITEN_REGULARIZE = 1e-6 # epsilon for numerical stability

# Geometry
NOISE_FLOOR = 0.01        # threshold for recoverable rank: λ > noise_floor * λ_max

# Pilot chains (minimal viable experiment)
PILOT_CHAINS = ["hiking_boot", "rescue_drone", "emergency_vehicle"]

# All chains
ALL_CHAINS = [
    "medical_consent", "rescue_drone", "graph_database", "refugee_shelter",
    "forensic_hospital", "employment_agreement", "emergency_vehicle",
    "hiking_boot", "research_university", "apartment_building",
]

# Redundant-commonality control chains
CONTROL_CHAINS = [
    "fruit_common", "bread_common", "coffee_common", "snow_common", "fire_common",
    "bird_common",  "tree_common",  "river_common",  "house_common", "dog_common",
]

SURFACE_FAMILIES = ["F1", "F2", "F3", "F4"]
N_LEVELS = 5
