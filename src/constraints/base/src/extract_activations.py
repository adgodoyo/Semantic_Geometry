"""Extract hidden states for all sentences at all layers.

Supports both encoder models (BERT-style, mean_content pooling) and
decoder-only models (Pythia/GPT-style, mean_all pooling).

Returns a (N, n_layers, hidden_dim) tensor cached to disk.
"""
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from config import MODEL_NAME, N_LAYERS, POOLING


def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor,
          strategy: str) -> torch.Tensor:
    """Pool token representations to a single sentence vector.

    strategy="mean_content": exclude CLS (pos 0) and SEP (last attended pos).
                             Appropriate for BERT-style encoder models.
    strategy="mean_all":     mean over all non-padding tokens.
                             Appropriate for causal LMs (Pythia, GPT, etc.)
                             where there is no CLS/SEP.
    """
    if strategy == "mean_content":
        mask = attention_mask.clone().float()
        mask[:, 0] = 0.0  # CLS
        for i in range(mask.shape[0]):
            last = int(attention_mask[i].sum().item()) - 1
            if last > 0:
                mask[i, last] = 0.0
    else:  # mean_all
        mask = attention_mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom


def extract_and_cache(df: pd.DataFrame,
                      cache_path: str,
                      model_name: str | None = None,
                      n_layers: int | None = None,
                      pooling: str | None = None,
                      torch_dtype: str = "float32",
                      batch_size: int = 16,
                      device: str | None = None) -> np.ndarray:
    """Extract hidden states for all sentences in df.

    Args:
        df: DataFrame with a "sentence" column.
        cache_path: Path to .npy cache file (loaded if exists, saved otherwise).
        model_name: HuggingFace model id. Defaults to config.MODEL_NAME.
        n_layers: Total number of layers to extract. Defaults to config.N_LAYERS.
        pooling: "mean_content" (BERT) or "mean_all" (causal LM).
                 Defaults to config.POOLING.
        torch_dtype: Model weight dtype — "float32", "float16", or "bfloat16".
                     Use "float32" for full precision (recommended for geometry
                     analysis). Use "float16"/"bfloat16" to halve memory for
                     large models (Pythia 2.8B ~5.6 GB → ~2.8 GB).
        batch_size: Sentences per forward pass.
        device: torch device string. Auto-detected if None.

    Returns:
        acts: np.ndarray of shape (N, n_layers, hidden_dim), always float32.
    """
    if os.path.exists(cache_path):
        print(f"  Loading cached activations from {cache_path}")
        return np.load(cache_path)

    model_name = model_name or MODEL_NAME
    n_layers   = n_layers or N_LAYERS
    pooling    = pooling or POOLING

    dtype_map = {"float32": torch.float32,
                 "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    t_dtype = dtype_map.get(torch_dtype, torch.float32)

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"  Model: {model_name}  |  dtype: {torch_dtype}  |  "
          f"Pooling: {pooling}  |  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Pythia / GPT-style tokenizers have no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(model_name, output_hidden_states=True,
                                      torch_dtype=t_dtype)
    model.eval().to(device)

    sentences = df["sentence"].tolist()
    N = len(sentences)
    hidden_dim = model.config.hidden_size

    all_acts = np.zeros((N, n_layers, hidden_dim), dtype=np.float32)

    for start in tqdm(range(0, N, batch_size), desc=f"Extracting {model_name}"):
        batch_sents = sentences[start: start + batch_size]
        enc = tokenizer(batch_sents, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        hidden_states = out.hidden_states  # tuple len = n_layers

        for layer_idx, hs in enumerate(hidden_states):
            if layer_idx >= n_layers:
                break
            pooled = _pool(hs, enc["attention_mask"], pooling)
            all_acts[start: start + len(batch_sents), layer_idx] = (
                pooled.cpu().float().numpy()
            )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, all_acts)
    print(f"  Saved activations → {cache_path}")

    # Explicitly free the model so downstream geometry analysis (numpy/scipy)
    # is not throttled by memory pressure from a multi-GB model still in RAM.
    import gc
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_acts
