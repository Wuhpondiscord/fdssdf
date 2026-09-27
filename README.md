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
- **IVTFF-aware parsing** (`ivtff.py`): real locus/folio metadata and the actual certain (`.`) vs. uncertain (`,`) word-space distinction from Zandbergen's IVTFF 2.0 spec, plus alternate-reading (`[a:b]`) resolution — instead of treating all whitespace as equivalent.
- **Nested null ladder** (`surrogates.py`, wired into `harness.py` as N0–N6): i.i.d. glyph, order-1/3/5 Markov, block shuffle, conventional-token shuffle, and position-conditioned Markov. N7 (quire-conditioned) is available separately once you supply a folio→quire map.
- **Long-range structure module** (`longrange.py`): lagged mutual information, block entropy, and a DFA-style fluctuation exponent — three independent measures, since no single estimator's slope should count as "long-range structure" on its own.
- **Cross-fit (held-out) unit discovery** (`segmentation.py`): `learn_bpe_merges` / `apply_bpe_merges` / `cross_fit_bpe_by_quire` so merge tables are learned on training quires only and applied unchanged to held-out quires, rather than the original in-sample-only BPE (still available, now clearly labeled as exploratory).
- **Two-layer adversarial validation**: `scorecard.py` (locked, interpretable feature-vector discrepancy / D-infinity — layer 1) and `discriminator.py` (classifier two-sample test with held-out AUC — layer 2, deliberately independent of layer 1).
- **Replication gate** (`replication_targets.py`): published 2025–2026 target statistics (Rozanova & Temerev; Parisel) with citations and a tolerance-based comparison, so new results get checked against the literature rather than eyeballed.
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

Whitespace is treated as an observation, not a verified word boundary. Metrics involving whitespace-delimited strings are explicitly labeled as conventional transcription-token metrics. IVTFF-parsed input additionally distinguishes certain vs. uncertain separators rather than collapsing them.

Known gaps, in priority order:

1. **Quire metadata.** IVTFF does not encode quire membership. `ivtff.load_quire_map` expects a `folio,quire` CSV that this repo does not ship — source one from a published transcription site and verify it against your transcription version before trusting any quire-held-out result.
2. **Generator library.** No Naibbe, self-citation, or grille-cipher generator is implemented yet. For anything you intend to report, prefer the original authors' released implementations over an approximation built from a paper description, and keep "meaningful cipher" and "meaningless pseudo-text" generator families explicitly separate (they are not the same claim).
3. **Consensus segmentation.** Only cross-fit BPE exists. The review recommends running BPE/MDL, a segmental HSMM, and motif discovery independently and reporting where they agree (a consensus graph), not treating any one method's output as "the units."
4. **Currier A/B as a stratification variable**, not an assumption — needs the quire map above plus a proper mixture-model fit to be genuinely useful rather than hand-labeled.
5. **Mahalanobis-style covariance-aware discrepancy** in `scorecard.py` (currently only the simpler D-infinity per-feature z-score is implemented; a covariance-aware statistic needs enough Monte Carlo replicates to estimate feature covariance stably).
6. Preregistering which features are fit/validation/sealed *before* tuning a generator, per the review's "sealed test" recommendation — currently a discipline the researcher has to impose manually, not something the code enforces.

GitHub `main` is the source of truth; `.github/workflows/mirror-to-hf.yml` mirrors it to `wuhp/ghtest`.
