"""Load and structure the semantic refinement dataset."""
import json
import pandas as pd
from config import DATA_PATH, CONTROL_DATA_PATH

# Surface pattern → family mapping for control dataset
# 8 examples per level follow 4 surface patterns × 2 sentences each
_CTRL_SURF_MAP = {0: "F1", 1: "F1", 2: "F2", 3: "F2",
                  4: "F3", 5: "F3", 6: "F4", 7: "F4"}


def load_dataset() -> pd.DataFrame:
    """Return the full dataset as a DataFrame."""
    with open(DATA_PATH) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["examples"])
    df["chain_level"] = df["chain_id"] + "__L" + df["level"].astype(str)
    df["chain_level_id"] = pd.factorize(df["chain_level"])[0]
    df["surface_id"] = df["surface_family"].map({"F1": 0, "F2": 1, "F3": 2, "F4": 3})
    df["level_id"] = df["level"]
    return df


def load_control_dataset() -> pd.DataFrame:
    """Return the redundant-commonality control dataset as a DataFrame.

    Same column schema as load_dataset(): chain_id, level, sentence,
    surface_family (inferred from example index within each level),
    chain_level_id, surface_id, level_id.
    """
    with open(CONTROL_DATA_PATH) as f:
        raw = json.load(f)
    rows = []
    for chain in raw["chains"]:
        chain_id = chain["chain_id"]
        for lvl in chain["levels"]:
            for idx, sentence in enumerate(lvl["examples"]):
                rows.append({
                    "chain_id":        chain_id,
                    "level":           lvl["level"],
                    "concept":         lvl["concept"],
                    "added_dimension": lvl["added_dimension"],
                    "surface_family":  _CTRL_SURF_MAP.get(idx, "F1"),
                    "sentence":        sentence,
                })
    df = pd.DataFrame(rows)
    df["chain_level"]    = df["chain_id"] + "__L" + df["level"].astype(str)
    df["chain_level_id"] = pd.factorize(df["chain_level"])[0]
    df["surface_id"]     = df["surface_family"].map({"F1": 0, "F2": 1, "F3": 2, "F4": 3})
    df["level_id"]       = df["level"]
    return df


def filter_chains(df: pd.DataFrame, chain_ids: list[str]) -> pd.DataFrame:
    return df[df["chain_id"].isin(chain_ids)].reset_index(drop=True)


def get_chain_info(df: pd.DataFrame) -> dict:
    """Return metadata: {chain_id: {level: concept_name}}."""
    info = {}
    for _, row in df.iterrows():
        c = row["chain_id"]
        l = row["level"]
        if c not in info:
            info[c] = {}
        if l not in info[c]:
            info[c][l] = row["concept"]
    return info
