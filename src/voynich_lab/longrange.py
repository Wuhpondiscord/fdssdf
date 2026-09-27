"""Long-range dependence statistics and finite-sample baselines."""
from __future__ import annotations
from collections import Counter
from math import log2
import random
import numpy as np
from .metrics import glyph_stream, mutual_information_pairs


def lagged_mutual_information(stream: str, max_lag: int = 20) -> dict[int, float]:
    stream = glyph_stream(stream)
    out: dict[int, float] = {}
    n = len(stream)
    for k in range(1, max(1, int(max_lag)) + 1):
        if n <= k:
            break
        out[k] = mutual_information_pairs(list(stream[:n-k]), list(stream[k:]))
    return out


def lagged_mi_with_shuffle_baseline(stream: str, max_lag: int = 20, n_permutations: int = 20, seed: int = 0) -> list[dict[str, float]]:
    """Raw lagged MI plus a finite-sample plug-in-bias baseline."""
    stream = glyph_stream(stream)
    observed = lagged_mutual_information(stream, max_lag=max_lag)
    rng = random.Random(seed)
    baseline: dict[int, list[float]] = {k: [] for k in observed}
    chars = list(stream)
    for _ in range(max(1, int(n_permutations))):
        perm = chars.copy()
        rng.shuffle(perm)
        pm = lagged_mutual_information("".join(perm), max_lag=max_lag)
        for k in baseline:
            if k in pm:
                baseline[k].append(pm[k])
    rows = []
    for k, value in observed.items():
        vals = baseline[k]
        mean = float(np.mean(vals)) if vals else float("nan")
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        rows.append({
            "lag": k,
            "mutual_information_bits": value,
            "shuffle_mean_bits": mean,
            "shuffle_sd_bits": sd,
            "excess_mi_bits": value - mean if mean == mean else float("nan"),
        })
    return rows


def block_entropy(stream: str, block_sizes: tuple[int, ...] = (1, 2, 3, 4, 5, 6)) -> dict[int, float]:
    stream = glyph_stream(stream)
    out: dict[int, float] = {}
    for b in block_sizes:
        b = int(b)
        if len(stream) < b:
            break
        blocks = [stream[i:i+b] for i in range(len(stream)-b+1)]
        counts = Counter(blocks)
        total = sum(counts.values())
        h = -sum((c/total)*log2(c/total) for c in counts.values())
        out[b] = h / b
    return out


def dfa_fluctuation(stream: str, indicator_glyphs: set[str] | None = None, box_sizes: tuple[int, ...] = (10,20,40,80,160,320)) -> tuple[dict[int, float], float]:
    stream = glyph_stream(stream)
    if not stream:
        return {}, float("nan")
    if indicator_glyphs is None:
        indicator_glyphs = {Counter(stream).most_common(1)[0][0]}
    x = np.array([1.0 if ch in indicator_glyphs else -1.0 for ch in stream])
    x -= x.mean()
    profile = np.cumsum(x)
    fluctuations: dict[int, float] = {}
    for box in box_sizes:
        box = int(box)
        if box < 4 or len(profile) < box * 2:
            continue
        n_boxes = len(profile) // box
        vals = []
        for i in range(n_boxes):
            seg = profile[i*box:(i+1)*box]
            t = np.arange(box)
            coeff = np.polyfit(t, seg, 1)
            trend = np.polyval(coeff, t)
            vals.append(np.sqrt(np.mean((seg-trend)**2)))
        fluctuations[box] = float(np.mean(vals))
    if len(fluctuations) >= 2:
        boxes = np.array(sorted(fluctuations))
        f = np.array([fluctuations[b] for b in boxes])
        valid = f > 0
        alpha = float(np.polyfit(np.log(boxes[valid]), np.log(f[valid]), 1)[0]) if valid.sum() >= 2 else float("nan")
    else:
        alpha = float("nan")
    return fluctuations, alpha
