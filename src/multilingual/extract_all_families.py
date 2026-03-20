"""extract_all_families.py — Extract activations for multilingual F1–F5 dataset.

Writes a separate cache so existing activations/results are preserved.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import dataset_all_families as ds_af

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "multilingual"
RESULTS_DIR = PROJECT_ROOT / "results" / "multilingual"
MPL_DIR = RESULTS_DIR / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

MODEL_NAME = "EleutherAI/pythia-2.8b"
BATCH_SIZE = 4
MAX_LENGTH = 64
OUTPUT_FILE = DATA_DIR / "activations_all_families.npz"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_activations_all_families():
    t0 = time.time()
    device = get_device()
    print(f"[extract-af] Device: {device}")

    df = ds_af.build_full_dataframe_all_families()
    sentences = df["sentence_text"].tolist()
    n = len(sentences)
    print(f"[extract-af] Sentences: {n}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    total_layers = n_layers + 1
    print(f"[extract-af] Layers={total_layers} Hidden={hidden_size}")

    acts = np.zeros((n, total_layers, hidden_size), dtype=np.float32)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[extract-af] Batches: {n_batches} (batch_size={BATCH_SIZE})")

    for b in range(n_batches):
        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, n)
        if b % 20 == 0:
            elapsed = time.time() - t0
            pct = 100.0 * start / max(n, 1)
            print(f"  batch {b+1:4d}/{n_batches}  ({pct:5.1f}%)  elapsed={elapsed:6.1f}s")

        inputs = tokenizer(
            sentences[start:end],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = torch.stack(outputs.hidden_states, dim=0)  # (L, B, T, H)
        mask = inputs["attention_mask"].unsqueeze(0).unsqueeze(-1).float()
        summed = (hidden_states.float() * mask).sum(dim=2)
        lengths = mask.sum(dim=2)
        pooled = (summed / lengths).cpu().numpy()                  # (L, B, H)

        acts[start:end] = pooled.transpose(1, 0, 2)

    elapsed = time.time() - t0
    print(f"[extract-af] Extraction complete in {elapsed:.1f}s")

    np.savez_compressed(
        OUTPUT_FILE,
        activations=acts,
        sentence_ids=df["sentence_id"].values,
        factor_ids=df["factor_id"].values,
        domain_ids=df["domain_id"].values,
        surface_families=df["surface_family"].values,
        splits=df["split"].values,
        languages=df["language"].values,
    )
    print(f"[extract-af] Saved {OUTPUT_FILE} shape={acts.shape} size={acts.nbytes/1e6:.1f}MB")


if __name__ == "__main__":
    extract_activations_all_families()
