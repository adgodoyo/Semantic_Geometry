"""
extract.py — Extract and cache hidden-state activations from Pythia 2.8B.

For every sentence in the dataset:
  - Run a forward pass with output_hidden_states=True
  - Mean-pool token representations (excluding padding) at each layer
  - Store as a float32 numpy array of shape (n_sentences, n_layers, hidden_size)

Cache: activations.npz  (overwrites if exists)
"""

import os
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

import dataset as ds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "multilingual"
RESULTS_DIR = PROJECT_ROOT / "results" / "multilingual"
MPL_DIR = RESULTS_DIR / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME  = "EleutherAI/pythia-2.8b"
BATCH_SIZE  = 4          # sentences per forward pass (safe for ~16 GB unified memory)
MAX_LENGTH  = 64         # max tokens per sentence; all sentences are ≤ 30 tokens
OUTPUT_FILE = DATA_DIR / "activations.npz"

# ── Device ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Main extraction ──────────────────────────────────────────────────────────

def extract_activations():
    t0 = time.time()
    device = get_device()
    print(f"[extract] Device : {device}")

    # ── Load dataset ──────────────────────────────────────────────────────
    df = ds.build_full_dataframe()
    sentences = df["sentence_text"].tolist()
    n = len(sentences)
    print(f"[extract] Sentences : {n}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"[extract] Loading tokenizer from cache: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[extract] Loading model from cache: {MODEL_NAME}  (this may take ~30 s)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,   # native for Apple Silicon; converted to float32 on store
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    n_layers = model.config.num_hidden_layers  # 32 for pythia-2.8b
    hidden_size = model.config.hidden_size     # 2560

    # +1 for the embedding layer (index 0)
    total_layers = n_layers + 1
    print(f"[extract] Model layers : {n_layers} transformer + 1 embedding = {total_layers} total")
    print(f"[extract] Hidden size  : {hidden_size}")
    elapsed = time.time() - t0
    print(f"[extract] Model loaded in {elapsed:.1f} s")

    # ── Run batched extraction ────────────────────────────────────────────
    all_activations = np.zeros((n, total_layers, hidden_size), dtype=np.float32)

    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[extract] Processing {n_batches} batches of size {BATCH_SIZE} ...")

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, n)
        batch_sentences = sentences[start:end]

        # Progress every 10 batches
        if batch_idx % 10 == 0:
            pct = 100 * start / n
            elapsed = time.time() - t0
            eta = (elapsed / max(start, 1)) * (n - start) if start > 0 else "—"
            eta_str = f"{eta:.0f} s" if isinstance(eta, float) else eta
            print(f"  batch {batch_idx+1:3d}/{n_batches}  ({pct:5.1f}%)  elapsed={elapsed:.1f}s  ETA≈{eta_str}")

        inputs = tokenizer(
            batch_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        attention_mask = inputs["attention_mask"]  # (B, seq_len)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (total_layers,) tensors each (B, seq_len, H)
        hidden_states = torch.stack(outputs.hidden_states, dim=0)  # (L, B, T, H)

        # Mean-pool over non-padding token positions
        # mask: (1, B, T, 1) for broadcasting
        mask = attention_mask.unsqueeze(0).unsqueeze(-1).float()  # (1, B, T, 1)
        summed = (hidden_states.float() * mask).sum(dim=2)         # (L, B, H)
        lengths = mask.sum(dim=2)                                  # (1, B, 1)
        pooled = (summed / lengths).cpu().numpy()                  # (L, B, H)

        # pooled shape: (total_layers, batch_size, hidden_size)
        all_activations[start:end] = pooled.transpose(1, 0, 2)    # (B, L, H)

    elapsed = time.time() - t0
    print(f"[extract] Extraction complete in {elapsed:.1f} s")

    # ── Save cache ────────────────────────────────────────────────────────
    print(f"[extract] Saving activations to {OUTPUT_FILE}  "
          f"(shape={all_activations.shape}, "
          f"size≈{all_activations.nbytes / 1e6:.1f} MB)")
    np.savez_compressed(
        OUTPUT_FILE,
        activations=all_activations,
        sentence_ids=df["sentence_id"].values,
        factor_ids=df["factor_id"].values,
        domain_ids=df["domain_id"].values,
        surface_families=df["surface_family"].values,
        splits=df["split"].values,
        languages=df["language"].values,
    )
    print(f"[extract] Done. Total time: {elapsed:.1f} s")
    return all_activations, df


if __name__ == "__main__":
    extract_activations()
