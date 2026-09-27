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

This repository is deliberately **not** a "Voynich decipherer." Its current focus is a reproducible structural-analysis harness: parse the transcription without assuming spaces are linguistic words, measure structure at several scales, compare against progressively stronger null models, and require later models to beat explicit validation tests.

## Implemented now

- Gradio UI for Hugging Face Spaces.
- CPU-first analysis; ordinary app use does not request a GPU.
- Glyph entropy, order-1 conditional entropy, adjacent-glyph mutual information, conventional-token statistics, edge mutual information, positional entropy, and Zipf slope.
- **IVTFF-aware parsing** (`ivtff.py`): locus/folio metadata, native page variables such as `$Q`, `$L`, `$H`, and `$I`, certain (`.`) versus uncertain (`,`) separators, and alternate-reading (`[a:b]`) resolution.
- **Nested null ladder** (`surrogates.py`, wired into `harness.py` as N0–N6): i.i.d. glyph, order-1/3/5 Markov, block shuffle, conventional-token shuffle, and position-conditioned Markov.
- **Long-range structure module** (`longrange.py`): lagged mutual information with shuffled finite-sample baseline, block entropy, and a DFA-style fluctuation exponent.
- **Exploratory stream BPE** remains available for discovery, but it is kept separate from reportable literature reproduction.
- **Published-method within-token BPE** (`segmentation.py`): merge rules are learned only inside conventional transcription tokens, pair counts are token-frequency weighted, rules are applied unchanged to held-out quire vocabularies, and unit-bigram scoring may cross token boundaries only within the same manuscript line.
- **Two-layer adversarial validation**: `scorecard.py` (locked interpretable feature-vector discrepancy / D-infinity) and `discriminator.py` (group-aware classifier two-sample test with held-out AUC).
- **Paper-exact replication layer** (`published_replication.py`): separate preprocessing/null contracts for character entropy, token order, glyph-edge order, cross-fit unit scale, and erased-space separator crossing.
- **Replication gate** (`replication_targets.py`): verified Rozanova & Temerev targets are computed automatically from IVTFF input; separate Parisel statistics remain available for manual comparison.
- Optional ZeroGPU co-occurrence/PPMI-SVD probe behind an explicit button.
- Automatic GitHub → Hugging Face Space mirroring.
- Ordinary pytest CI plus a pinned full-corpus literature-reproduction integration test.

## Published replication

The automatic Rozanova & Temerev path follows the public reproduction code rather than forcing all statistics through one generic normalized stream. This matters because the paper uses different corpus-cleaning and null-model conventions for different headline results.

The integration workflow pins the authors' final reproduction archive at:

```text
lrozanova/voynich-units
956a7c4fc39981f4d116fa3f4edfccce6d065571
```

It downloads that revision's `ZL3b.txt` and runs this repository's implementation end-to-end. The current reproduced values are:

| Statistic | This implementation | Published value |
|---|---:|---:|
| Composite-collapsed character H(next\|current) | 2.689692 | 2.69 bits |
| Shuffle-corrected capped-token order / H(T) | 0.00793728 | 0.79% |
| Last-glyph → next-token first-glyph excess MI | 0.197240 | 0.197 bits |
| Cross-fit BPE dependence gap, 0 merges | 1.685848 | 1.686 bits |
| Cross-fit BPE dependence gap, 16 merges | 1.422645 | 1.423 bits |
| Cross-fit BPE dependence gap, 32 merges | 1.378640 | 1.379 bits |
| Cross-fit BPE dependence gap, 64 merges | 1.489944 | 1.490 bits |
| Held-out BPE minimum | 32 merges | 32 merges |
| Certain-separator crossing after 64 erased-space BPE merges | 661 / 26,447 = 0.0249934 | 2.5% |

The same Table-8 reproduction also yields `477 / 2,350 = 0.202979` for uncertain separators and `97,402 / 162,330 = 0.600025` across all eligible intra-line positions.

### Why there are separate replication paths

- **Character entropy** uses the paper's entropy-specific `P`-locus cleaner, composite-collapsed EVA, and then erases token and line boundaries from the final symbol stream.
- **Unit-scale BPE** uses the paper's stricter token cleaner, native `$Q` groups, within-token merge learning, and leave-one-quire-out rule application.
- **Edge MI** uses 100 within-line shuffles with NumPy seed `20260816`.
- **Token succession** uses the 2,000-type cap and a separate Python `random.Random(20260810)` within-line shuffle procedure; the denominator is full capped-token marginal entropy `H(T)`.
- **Separator crossing** erases spaces first, learns BPE within manuscript-line strings, and evaluates hidden separator positions at the paper's 64-merge checkpoint.

The pinned integration check lives in `.github/workflows/replication-integration.yml`. Ordinary unit/regression tests live in `.github/workflows/tests.yml`.

## Hugging Face target

Default Space: `wuhp/ghtest`

Override with `HF_SPACE_REPO`.

### Secrets

Do **not** commit a Hugging Face token.

For automatic GitHub → Hugging Face mirroring, add a GitHub Actions secret named `HF_TOKEN` to this repository. It should be a Hugging Face write token with access to `wuhp/ghtest`.

For the in-app manual push button, also add `HF_TOKEN` as a Hugging Face Space secret.

## ZeroGPU behavior

Only `src/voynich_lab/gpu_probe.py` uses `@spaces.GPU`. Loading data, preprocessing, metrics, null-model generation, unit discovery, literature replication, and the UI itself stay on CPU and do not request ZeroGPU.

## Local run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Run tests with:

```bash
python -m pytest -q
```

## Current scientific scope

Whitespace is treated as an observation, not a verified linguistic word boundary. Metrics involving whitespace-delimited strings are explicitly labeled as conventional transcription-token metrics. IVTFF-parsed input additionally distinguishes certain from uncertain separators instead of collapsing them.

Native `$Q` quire metadata is used when it is present in the loaded transcription. The literature-reproduction path intentionally follows the preprocessing choices of the cited public reproduction code; the generic exploratory tabs remain separate so a convenient descriptive metric is not accidentally presented as a reproduced paper statistic.

Known gaps, in priority order:

1. **Generator library.** No Naibbe, self-citation, or grille-cipher generator is implemented yet. For reportable comparisons, prefer the original authors' released implementations over an approximation reconstructed only from prose, and keep meaningful-cipher and meaningless-pseudotext generator families separate.
2. **Consensus segmentation.** BPE is implemented, including paper-exact held-out BPE, but an independent segmental HSMM and motif-discovery path are not yet implemented. A later consensus analysis should report where independent methods agree rather than treating BPE output as identified linguistic units.
3. **Currier stratification.** Native `$L` labels are parsed, but a proper stratified/mixture analysis and held-out evaluation should be added before drawing model-level conclusions from Currier A/B differences.
4. **Covariance-aware discrepancy.** `scorecard.py` currently uses the simpler D-infinity / per-feature z-score framework. A Mahalanobis-style statistic needs enough Monte Carlo replicates for stable covariance estimation and validation.
5. **Sealed-test enforcement.** The code supports held-out analyses, but it does not yet enforce a preregistered fit/validation/sealed partition before generator tuning.
6. **Independent literature families.** The automatic gate currently reproduces the verified Rozanova & Temerev pipeline. Parisel's positional-class statistics are kept separate until their full preprocessing/estimation pipeline is implemented and independently checked.

GitHub `main` is the source of truth; `.github/workflows/mirror-to-hf.yml` mirrors it to `wuhp/ghtest`.
