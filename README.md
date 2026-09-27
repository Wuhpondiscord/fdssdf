---
title: Voynich Structure Lab
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Voynich Structure Lab

A falsification-first computational workbench for structural analysis of the Voynich Manuscript.

This repository is deliberately **not** a "Voynich decipherer." Its first milestone is a reproducible Phase 0/Phase 2 harness: normalize a transcription without assuming spaces are words, measure multi-scale structure, compare against progressively stronger null models, and make every later model earn its complexity.

## Implemented now

- Gradio UI for Hugging Face Spaces.
- CPU-first analysis; ordinary app use does not request a GPU.
- Glyph entropy, order-1 conditional entropy, adjacent-glyph mutual information, conventional-token statistics, edge mutual information, positional entropy, and Zipf slope.
- Null models: i.i.d. glyph surrogate, first-order Markov surrogate, token-order shuffle.
- Lightweight BPE-style unit discovery over a continuous glyph stream.
- Optional ZeroGPU co-occurrence/PPMI-SVD probe behind an explicit button.
- Automatic GitHub → Hugging Face Space mirroring.
- Manual recovery push from the app using `HF_TOKEN`.

## Hugging Face target

Default Space: `wuhp/ghtest`

Override with `HF_SPACE_REPO`.

### Secrets

Do **not** commit a Hugging Face token.

For automatic GitHub → Hugging Face mirroring, add a GitHub Actions secret named `HF_TOKEN` to this repository. It should be a Hugging Face write token with access to `wuhp/ghtest`.

For the in-app manual push button, also add `HF_TOKEN` as a Hugging Face Space secret.

## ZeroGPU behavior

Only `src/voynich_lab/gpu_probe.py` uses `@spaces.GPU`. Loading data, preprocessing, metrics, null-model generation, unit discovery, and the UI itself stay on CPU and do not request ZeroGPU.

## Local run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Run tests with:

```bash
pytest -q
```

## Current scientific scope

Whitespace is treated as an observation, not a verified word boundary. Metrics involving whitespace-delimited strings are explicitly labeled as conventional transcription-token metrics.

Next priorities are IVTFF-aware parsing, uncertain-space handling, page/quire metadata, quire-held-out validation, stronger position-conditioned nulls, published-result reproduction, and preregistered multivariate falsification tests.

GitHub `main` is the source of truth; `.github/workflows/mirror-to-hf.yml` mirrors it to `wuhp/ghtest`.
