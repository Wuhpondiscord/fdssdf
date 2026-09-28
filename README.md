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
- **Currier stratification** (`currier.py`): Currier A/B boundary profiles with stratum-local PMI anchors, native `$Q` resampling, and both independent-stratum and joint-quire bootstrap schemes.
- **Line-start decomposition** (`currier_linestart.py`): paragraph-first versus other-line JSD, six-glyph enrichment, Currier-stratified JSD, relabel nulls, and quire bootstrap, with executable parity checked against the authors' pinned script.
- **Naibbe meaningful-cipher baseline** (`generators.py`): the paper-facing 52-card `naibbe.py` algorithm, exact 414-row substitution table, isolated seeded RNG, and provenance metadata. CI compares seeded plaintext respacing, canonical ciphertext, final 3%-space-dropped ciphertext, and ambiguity retry count directly against the pinned original implementation.
- **Timm/Schinner self-citation pseudotext baseline** (`selfcitation_*.py`): the released Java generator is ported through RNG, glyph parsing, can-follow rules, source choice, statistics, slim morphing, and full generator orchestration. Dedicated CI gates compare the Python port against the pinned upstream Java implementation.
- **Replication gate** (`replication_targets.py`): verified Rozanova & Temerev targets are computed automatically from IVTFF input; separate Parisel statistics remain available for manual comparison.
- Optional ZeroGPU co-occurrence/PPMI-SVD probe behind an explicit button.
- Automatic GitHub → Hugging Face Space mirroring.
- Ordinary pytest CI plus pinned full-corpus literature-reproduction integration tests.

## Published replication

The automatic Rozanova & Temerev path follows the public reproduction code rather than forcing all statistics through one generic normalized stream. This matters because the paper uses different corpus-cleaning and null-model conventions for different headline results.

The integration workflows pin the authors' final reproduction archive at:

```text
lrozanova/voynich-units
956a7c4fc39981f4d116fa3f4edfccce6d065571
```

They download that revision's `ZL3b.txt` and run this repository's implementation end-to-end. The current reproduced values are:

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

### Currier A/B reproduction

The Currier boundary-profile implementation recomputes the PMI normalization separately inside A and B, rather than forcing both strata onto one pooled scale. On the pinned archive it reproduces the released analysis values:

| Boundary class | Currier A | Currier B | A − B |
|---|---:|---:|---:|
| uncertain separators | 0.493805 | 0.502914 | -0.009109 |
| first in line | ≈0.104 | ≈0.069 | ≈+0.036 |
| mid-line | ≈0.125 | ≈0.037 | ≈+0.088 |
| certain separators | ≈0.094 | ≈-0.003 | ≈+0.096 |
| line break | ≈-0.010 | ≈-0.041 | ≈+0.031 |

The uncertain-separator rates are `0.08803` for A and `0.07959` for B; raw uncertain-minus-certain PMI gaps are `1.53252` and `2.52171` bits respectively.

The paper-level 1,500-draw quire bootstrap is reproduced separately. For the uncertain-index A−B difference, the 90% intervals are:

- independent A/B quire resampling: `[-0.05152, 0.03714]`
- joint pooled-quire resampling: `[-0.04842, 0.03479]`

Both include zero. This is why the project does not present the small pooled A/B difference as a stand-alone dialect-significance result.

### Line-start executable-vs-stated discrepancy

The pinned final `reproduce_dialect_linestart.py` contains two different objects: values **computed by the executable analysis** and older hard-coded `STATED` constants. Our implementation is checked directly against the executable output and matches it to floating-point precision.

For the primary `pstart` + composite-collapsed analysis on the pinned final archive:

| Quantity | Pinned executable (and this repo) | Bundled `STATED` constant |
|---|---:|---:|
| JSD, paragraph-first lines | 0.527549 | 0.485 |
| JSD, other lines | 0.179247 | 0.176 |
| JSD, pooled | 0.202997 | 0.179 |
| first/other JSD ratio | 2.94314 | 2.75 |

The same mismatch occurs in several enrichment ratios. For example, executable paragraph-first enrichment is `p=24.3071`, `t=11.9128`, `k=3.01265`, `f=5.02975`, `y=0.38133`, `d=0.32563`, while the bundled constants are `23.7, 18.2, 6.43, 4.47, 0.18, 0.26`.

This repository does **not** alter the implementation to force the final archive to match stale constants. `.github/workflows/upstream-linestart-equivalence.yml` downloads the authors' exact pinned executable and verifies our computed JSD/enrichment/Currier-stratified line-start fields against it to machine precision, while also asserting that the executable and `STATED` values materially differ.

The pinned deterministic integration check lives in `.github/workflows/replication-integration.yml`; the full 1,500-draw Currier uncertainty check lives in `.github/workflows/currier-bootstrap-integration.yml`; executable line-start parity lives in `.github/workflows/upstream-linestart-equivalence.yml`; ordinary unit/regression tests live in `.github/workflows/tests.yml`.

## Meaningful-cipher generator baseline: Naibbe

Naibbe is intentionally kept separate from the N0–N6 null-surrogate ladder. It consumes meaningful plaintext and produces Voynich-like ciphertext, so placing it in the same dropdown as corpus-preserving null models would conflate two different experimental questions.

The implementation pins `greshko/naibbe-cipher` at commit:

```text
f2675ec5dd275268bc64dd48ea64fc0e0e9827a2
```

The baseline follows the paper-facing `naibbe.py` defaults: respacing parameter 17, 52-card deck, 3% random output-space removal, and ambiguity-safe bigrams that reject accidental unigram glyph types. The imported `references/naibbe_tables.csv` is byte-identical to the pinned upstream asset (414 mapping rows).

`.github/workflows/naibbe-equivalence.yml` downloads and executes that exact upstream implementation, then requires exact equality with this repository for seeded respaced plaintext, canonical ciphertext, final respaced ciphertext, and ambiguity retry count. The local wrapper uses a private `random.Random(seed)` so Naibbe generation does not perturb the global RNG state used by other experiments.

The newer upstream `naibbe_v2.py` is not silently substituted: it changes the default to a 78-card deck and adds an additional cross-bigram collision rule, so it is treated as a distinct future generator variant. Third-party licensing and the requested paper citation are recorded in `THIRD_PARTY_NOTICES.md`.

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

Native `$Q` and `$L` metadata are used where present. Currier A/B comparisons are quire-aware and use stratum-local normalization. The literature-reproduction path intentionally follows the preprocessing choices of the cited public reproduction code; the generic exploratory tabs remain separate so a convenient descriptive metric is not accidentally presented as a reproduced paper statistic.

Known gaps, in priority order:

1. **Generator library.** The paper-facing Naibbe 52-card meaningful-cipher baseline and the Timm/Schinner meaningless self-citation pseudotext family are both implemented and source-equivalence-checked. The remaining major generator gap is a reproducible Rugg-style grille-cipher implementation sourced from original released code/data rather than a prose-only approximation.
2. **Consensus segmentation.** BPE is implemented, including paper-exact held-out BPE, but an independent segmental HSMM and motif-discovery path are not yet implemented. A later consensus analysis should report where independent methods agree rather than treating BPE output as identified linguistic units.
3. **Covariance-aware discrepancy.** `scorecard.py` currently uses the simpler D-infinity / per-feature z-score framework. A Mahalanobis-style statistic needs enough Monte Carlo replicates for stable covariance estimation and validation.
4. **Sealed-test enforcement.** The code supports held-out analyses, but it does not yet enforce a preregistered fit/validation/sealed partition before generator tuning.
5. **Independent literature families.** The automatic gate currently reproduces the verified Rozanova & Temerev pipeline. Parisel's positional-class statistics are kept separate until their full preprocessing/estimation pipeline is implemented and independently checked.

GitHub `main` is the source of truth; `.github/workflows/mirror-to-hf.yml` mirrors it to `wuhp/ghtest`.
