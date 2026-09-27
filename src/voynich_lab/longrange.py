"""Long-range dependence statistics.

The existing harness only measures local structure (order-1 conditional
entropy, adjacent mutual information). Per the review, a generator should
never get credit for "long-range structure" on the strength of one
estimator's favorable slope, so this module deliberately implements three
independent measures rather than one:

1. lagged_mutual_information: I(G_t; G_{t+k}) for a range of k
2. block_entropy: H(G_1..G_n) approximated over increasing window size
3. dfa_fluctuation: a detrended-fluctuation-style RMS exponent over an
   indicator random walk, in the spirit of Schinner's spectral/fractal work

All three should be reported together; a generator "passing" one while
failing the other two is not evidence of matching long-range structure.
"""

from __future__ import annotations

from collections import Counter
from math import log2

import numpy as np

from .metrics import glyph_stream, mutual_information_pairs


def lagged_mutual_information(stream: str, max_lag: int = 20) -> dict[int, float]:
    out: dict[int, float] = {}
    n = len(stream)
    for k in range(1, max(1, int(max_lag)) + 1):
        if n <= k:
            break
        xs = list(stream[: n - k])
        ys = list(stream[k:])
        out[k] = mutual_information_pairs(xs, ys)
    return out


def block_entropy(stream: str, block_sizes: tuple[int, ...] = (1, 2, 3, 4, 5, 6)) -> dict[int, float]:
    """H(block) per symbol for increasing block length. A flattening slope
    (entropy rate approaching a plateau quickly) indicates mostly local
    structure; a slowly-converging slope indicates longer-range dependence.
    """
    out: dict[int, float] = {}
    for b in block_sizes:
        b = int(b)
        if len(stream) < b:
            break
        blocks = [stream[i : i + b] for i in range(len(stream) - b + 1)]
        counts = Counter(blocks)
        total = sum(counts.values())
        h = -sum((c / total) * log2(c / total) for c in counts.values())
        out[b] = h / b  # per-symbol entropy, so blocks are comparable
    return out


def dfa_fluctuation(
    stream: str,
    indicator_glyphs: set[str] | None = None,
    box_sizes: tuple[int, ...] = (10, 20, 40, 80, 160, 320),
) -> tuple[dict[int, float], float]:
    """Detrended-fluctuation-style analysis over a random-walk built from an
    indicator sequence (1 if glyph in indicator_glyphs else -1). Returns
    (per-box-size RMS fluctuation, estimated scaling exponent alpha via a
    log-log fit). alpha ~ 0.5 indicates no long-range correlation;
    alpha > 0.5 indicates persistent long-range correlation.

    If indicator_glyphs is not given, the single most frequent glyph is used
    (a common convention in this line of work, since results can depend on
    which glyph/class is chosen -- report results for more than one choice
    rather than treating a single alpha as definitive).
    """
    stream = glyph_stream(stream) if any(c.isspace() for c in stream) else stream
    if not stream:
        return {}, float("nan")
    if indicator_glyphs is None:
        most_common = Counter(stream).most_common(1)[0][0]
        indicator_glyphs = {most_common}

    x = np.array([1.0 if ch in indicator_glyphs else -1.0 for ch in stream])
    x = x - x.mean()
    profile = np.cumsum(x)

    fluctuations: dict[int, float] = {}
    for box in box_sizes:
        box = int(box)
        if box < 4 or len(profile) < box * 2:
            continue
        n_boxes = len(profile) // box
        rms_vals = []
        for i in range(n_boxes):
            seg = profile[i * box : (i + 1) * box]
            t = np.arange(box)
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            rms_vals.append(np.sqrt(np.mean((seg - trend) ** 2)))
        fluctuations[box] = float(np.mean(rms_vals))

    if len(fluctuations) >= 2:
        boxes = np.array(sorted(fluctuations))
        f = np.array([fluctuations[b] for b in boxes])
        valid = f > 0
        if valid.sum() >= 2:
            alpha, _ = np.polyfit(np.log(boxes[valid]), np.log(f[valid]), 1)
        else:
            alpha = float("nan")
    else:
        alpha = float("nan")
    return fluctuations, float(alpha)
