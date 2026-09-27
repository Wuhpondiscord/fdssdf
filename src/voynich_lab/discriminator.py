"""Classifier two-sample test (layer 2 of adversarial validation).

Deliberately separate from scorecard.py's locked feature vector. A
black-box classifier can distinguish real vs. generated text for reasons
that have nothing to do with the hypothesis under test (block-length
artifacts, vocabulary leakage, alphabet-normalization differences). Its
result is informative only in one direction:

  held-out AUC >> 0.5  =>  the generator omits *some* detectable structure
  held-out AUC ~= 0.5  =>  this classifier, at this sample size, found no
                           difference -- NOT proof the generator is
                           historically or semantically correct.

Requires scikit-learn (see requirements.txt).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .metrics import glyph_stream


@dataclass
class DiscriminatorResult:
    held_out_auc: float
    n_real_blocks: int
    n_generated_blocks: int
    block_size: int
    note: str


def _block_features(block: str) -> list[float]:
    """Small, generic block-level feature set (kept separate from and
    smaller than the scorecard's feature set, so this layer is testing for
    *residual* structure rather than re-deriving the same signal)."""
    from collections import Counter
    from math import log2

    n = len(block)
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0]
    counts = Counter(block)
    entropy = -sum((c / n) * log2(c / n) for c in counts.values())
    n_types = len(counts)
    most_common_frac = counts.most_common(1)[0][1] / n
    # simple run-length proxy: fraction of positions where char repeats previous char
    repeats = sum(1 for i in range(1, n) if block[i] == block[i - 1]) / max(1, n - 1)
    return [entropy, n_types / n, most_common_frac, repeats]


def _make_blocks(text: str, block_size: int) -> list[str]:
    stream = glyph_stream(text)
    return [stream[i : i + block_size] for i in range(0, len(stream) - block_size + 1, block_size)]


def classifier_two_sample_test(
    real_text: str,
    generator_fn: Callable[[str, int], str],
    block_size: int = 200,
    n_generated_replicates: int = 5,
    seed: int = 0,
    test_fraction: float = 0.3,
) -> DiscriminatorResult:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "classifier_two_sample_test requires scikit-learn; add it to requirements.txt"
        ) from exc

    real_blocks = _make_blocks(real_text, block_size)
    gen_blocks: list[str] = []
    for r in range(max(1, int(n_generated_replicates))):
        gen_blocks.extend(_make_blocks(generator_fn(real_text, seed + r), block_size))

    if len(real_blocks) < 4 or len(gen_blocks) < 4:
        return DiscriminatorResult(
            held_out_auc=float("nan"),
            n_real_blocks=len(real_blocks),
            n_generated_blocks=len(gen_blocks),
            block_size=block_size,
            note="Too few blocks for a meaningful held-out AUC; use a smaller block_size or more text.",
        )

    X = np.array([_block_features(b) for b in real_blocks + gen_blocks])
    y = np.array([1] * len(real_blocks) + [0] * len(gen_blocks))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probs))

    note = (
        "AUC near 0.5 means this classifier found no residual difference at this "
        "block size/feature set -- it does not certify the generator as correct. "
        "AUC well above 0.5 means detectable structure remains unmodeled."
    )
    return DiscriminatorResult(
        held_out_auc=auc,
        n_real_blocks=len(real_blocks),
        n_generated_blocks=len(gen_blocks),
        block_size=block_size,
        note=note,
    )
