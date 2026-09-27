"""Locked-scorecard adversarial validation (layer 1 of 2).

Per the review, "fooling a discriminator" should not be the first test a
generator faces, because a black-box classifier can pass or fail for
reasons unrelated to the hypothesis (line-length artifacts, vocabulary
leakage, too few samples). The recommended design is:

  1. A small, preregistered, interpretable feature vector (this module).
  2. A black-box classifier two-sample test as a second, independent check
     for residual detectable structure (see discriminator.py).

A generator "survives" layer 1 only if its worst standardized discrepancy
(D_inf = max |z_j|) across *every* feature is small -- not if it wins on
one favorite metric. This directly discourages the failure mode where a
hoax-generator is quietly tuned against whichever statistic it already
matches, then presented as if it explained the others too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .metrics import compute_metrics

DEFAULT_FEATURES = (
    "glyph_entropy_bits",
    "conditional_entropy_order1_bits",
    "adjacent_glyph_mi_bits",
    "cross_token_edge_mi_bits",
    "hapax_fraction",
    "mean_token_length",
    "zipf_loglog_slope",
)


@dataclass
class ScorecardResult:
    observed: dict[str, float]
    generator_mean: dict[str, float]
    generator_sd: dict[str, float]
    z_scores: dict[str, float]
    d_infinity: float
    worst_feature: str
    n_replicates: int


def run_scorecard(
    text: str,
    generator_fn: Callable[[str, int], str],
    features: tuple[str, ...] = DEFAULT_FEATURES,
    n_replicates: int = 200,
    seed: int = 0,
) -> ScorecardResult:
    """Compare observed text's feature vector against the Monte Carlo
    distribution a candidate generator produces at matched size.

    generator_fn must have signature (text, seed) -> str, matching the
    existing surrogates.* functions, so any surrogate or future Naibbe/
    self-citation/grille implementation can be plugged in without changes
    here.
    """
    observed_all = compute_metrics(text)
    observed = {f: float(observed_all[f]) for f in features}

    samples: dict[str, list[float]] = {f: [] for f in features}
    for r in range(max(1, int(n_replicates))):
        gen_text = generator_fn(text, seed + r)
        gen_metrics = compute_metrics(gen_text)
        for f in features:
            v = gen_metrics.get(f, float("nan"))
            samples[f].append(float(v))

    gen_mean: dict[str, float] = {}
    gen_sd: dict[str, float] = {}
    z: dict[str, float] = {}
    for f in features:
        arr = np.array([v for v in samples[f] if v == v])  # drop NaN
        mean = float(np.mean(arr)) if arr.size else float("nan")
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
        gen_mean[f] = mean
        gen_sd[f] = sd
        if sd and sd > 0 and sd == sd and observed[f] == observed[f]:
            z[f] = (observed[f] - mean) / sd
        else:
            z[f] = float("nan")

    finite_z = {k: v for k, v in z.items() if v == v}
    if finite_z:
        worst_feature = max(finite_z, key=lambda k: abs(finite_z[k]))
        d_inf = abs(finite_z[worst_feature])
    else:
        worst_feature = ""
        d_inf = float("nan")

    return ScorecardResult(
        observed=observed,
        generator_mean=gen_mean,
        generator_sd=gen_sd,
        z_scores=z,
        d_infinity=d_inf,
        worst_feature=worst_feature,
        n_replicates=int(n_replicates),
    )


def scorecard_to_rows(result: ScorecardResult) -> list[dict[str, object]]:
    rows = []
    for f in result.observed:
        rows.append(
            {
                "feature": f,
                "observed": result.observed[f],
                "generator_mean": result.generator_mean[f],
                "generator_sd": result.generator_sd[f],
                "z_score": result.z_scores[f],
                "is_worst_feature": f == result.worst_feature,
            }
        )
    return rows
