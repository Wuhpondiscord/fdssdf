"""Interpretable Monte-Carlo scorecard for generator falsification."""
from __future__ import annotations

from dataclasses import dataclass, field
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
    feature_p_values: dict[str, float]
    feature_p_holm: dict[str, float]
    d_infinity: float
    worst_feature: str
    global_p_value: float
    n_replicates: int
    # Monte-Carlo resolution diagnostics. An empirical p-value can never fall below 1/(n+1), and Holm then
    # multiplies the smallest one by the number of features, so a huge |z| can still show a "large" p.
    min_resolvable_p: float = float("nan")
    outside_null_range: dict[str, bool] = field(default_factory=dict)
    holm_can_reject_at_0_05: bool = True
    resolution_note: str = ""


def _holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    valid = sorted((p, k) for k, p in pvals.items() if p == p)
    m = len(valid)
    out = {k: float("nan") for k in pvals}
    running = 0.0
    for rank, (p, key) in enumerate(valid, start=1):
        adjusted = min(1.0, (m - rank + 1) * p)
        running = max(running, adjusted)
        out[key] = running
    return out


def run_scorecard(
    text: str,
    generator_fn: Callable[[str, int], str],
    features: tuple[str, ...] = DEFAULT_FEATURES,
    n_replicates: int = 200,
    seed: int = 0,
) -> ScorecardResult:
    n_replicates = max(3, int(n_replicates))
    observed_all = compute_metrics(text)
    observed = {f: float(observed_all[f]) for f in features}
    matrix = np.full((n_replicates, len(features)), np.nan, dtype=float)
    for r in range(n_replicates):
        gen_metrics = compute_metrics(generator_fn(text, seed + r))
        for j, f in enumerate(features):
            matrix[r, j] = float(gen_metrics.get(f, float("nan")))

    gen_mean: dict[str, float] = {}
    gen_sd: dict[str, float] = {}
    z: dict[str, float] = {}
    pvals: dict[str, float] = {}
    outside: dict[str, bool] = {}
    for j, f in enumerate(features):
        arr = matrix[:, j]
        arr = arr[np.isfinite(arr)]
        mean = float(np.mean(arr)) if arr.size else float("nan")
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
        gen_mean[f], gen_sd[f] = mean, sd
        if np.isfinite(sd) and sd > 0 and np.isfinite(observed[f]):
            z[f] = (observed[f] - mean) / sd
        elif np.isfinite(observed[f]) and np.isfinite(mean):
            z[f] = 0.0 if observed[f] == mean else float("inf")
        else:
            z[f] = float("nan")
        if arr.size and np.isfinite(observed[f]):
            center = float(np.median(arr))
            obs_dev = abs(observed[f] - center)
            exceed = int(np.sum(np.abs(arr - center) >= obs_dev))
            pvals[f] = (exceed + 1.0) / (arr.size + 1.0)
        else:
            pvals[f] = float("nan")
        outside[f] = bool(arr.size and np.isfinite(observed[f]) and (observed[f] < arr.min() or observed[f] > arr.max()))

    finite_z = {k: v for k, v in z.items() if not np.isnan(v)}
    if finite_z:
        worst_feature = max(finite_z, key=lambda k: abs(finite_z[k]))
        d_inf = float(abs(finite_z[worst_feature]))
    else:
        worst_feature, d_inf = "", float("nan")

    null_d: list[float] = []
    for i in range(n_replicates):
        others = np.delete(matrix, i, axis=0)
        means = np.nanmean(others, axis=0)
        sds = np.nanstd(others, axis=0, ddof=1)
        vals = matrix[i]
        zi: list[float] = []
        for v, m, s in zip(vals, means, sds):
            if not np.isfinite(v) or not np.isfinite(m):
                continue
            if np.isfinite(s) and s > 0:
                zi.append(abs((v - m) / s))
            else:
                zi.append(0.0 if v == m else float("inf"))
        if zi:
            null_d.append(max(zi))
    if null_d and not np.isnan(d_inf):
        global_p = (1.0 + sum(v >= d_inf for v in null_d)) / (len(null_d) + 1.0)
    else:
        global_p = float("nan")

    n_tested = sum(1 for v in pvals.values() if v == v)
    min_p = 1.0 / (n_replicates + 1.0)
    holm_ok = bool(n_tested == 0 or min_p * n_tested <= 0.05)
    if holm_ok:
        note = ""
    else:
        need = int(np.ceil(n_tested / 0.05)) - 1
        note = (
            f"With {n_replicates} replicates the smallest attainable empirical p is {min_p:.3g}, so Holm "
            f"across {n_tested} features cannot fall below {min_p * n_tested:.3g}. That is a resolution limit, "
            f"not weak evidence: use >= {need} replicates for alpha=0.05, and read 'outside_null_range' "
            "(observed lies beyond every simulated replicate) alongside the z-score."
        )

    return ScorecardResult(
        observed=observed,
        generator_mean=gen_mean,
        generator_sd=gen_sd,
        z_scores=z,
        feature_p_values=pvals,
        feature_p_holm=_holm_adjust(pvals),
        d_infinity=d_inf,
        worst_feature=worst_feature,
        global_p_value=float(global_p),
        n_replicates=n_replicates,
        min_resolvable_p=float(min_p),
        outside_null_range=outside,
        holm_can_reject_at_0_05=holm_ok,
        resolution_note=note,
    )


def scorecard_to_rows(result: ScorecardResult) -> list[dict[str, object]]:
    return [{
        "feature": f,
        "observed": result.observed[f],
        "generator_mean": result.generator_mean[f],
        "generator_sd": result.generator_sd[f],
        "z_score": result.z_scores[f],
        "empirical_p": result.feature_p_values[f],
        "holm_p": result.feature_p_holm[f],
        "outside_null_range": result.outside_null_range.get(f, False),
        "is_worst_feature": f == result.worst_feature,
    } for f in result.observed]
