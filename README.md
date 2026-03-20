# Semantic Geometry

Probes the hidden-state geometry of transformer language models (BERT-base and Pythia-2.8B) across two settings:

- **Multilingual** — whether a stable semantic hierarchy survives across five languages (EN/DE/ES/AR/ZH) and multiple surface forms once language and phrasing nuisance is removed.
- **Constraints** — whether adding genuine semantic constraints increases recoverable geometric structure, and whether redundant modifiers leave it flat.

## Repository layout

```
data/           Input datasets and cached activation tensors
src/            Analysis source code
  constraints/  Constraint experiment (base pipeline + robustness checks)
  multilingual/ Multilingual experiment
results/        Output figures, CSVs, and JSON summaries
scripts/        Ordered entry points for extraction, analysis, and build
slides/         Beamer deck source and PDF
paper/          Paper source and PDF
```

## Prerequisites

```
pip install -r requirements.txt
```

- `torch`, `transformers`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `Pillow`, `tqdm`
- Hugging Face access to `EleutherAI/pythia-2.8b` if re-extracting activations
- Enough disk for activation caches (several GB)

## Running

Run everything in order:

```bash
bash scripts/run_all.sh
```

Or step by step:

```bash
bash scripts/extract_multilingual.sh
bash scripts/extract_constraints.sh
bash scripts/analyze_multilingual.sh
bash scripts/analyze_constraints.sh
bash scripts/build_slides.sh
bash scripts/build_paper.sh
bash scripts/curate_release.sh
python scripts/validate_project.py
```

`curate_release.sh` strips intermediate files back to the release view. `validate_project.py` checks that all expected outputs are present and the build graph is intact.

## What the analyses produce

Each analysis run saves CSVs and PNGs to `results/`. The multilingual pipeline covers within/between similarity (cosine + L2), meaning and surface-form classifiers, shared low-rank subspace geometry, residualization under style/linguistic/language control, and family- and language-pair similarity matrices. The constraints pipeline covers the same core geometry measures plus robustness checks and raw Euclidean delta snapshots.
