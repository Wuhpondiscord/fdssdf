"""Classifier two-sample test with group-aware held-out evaluation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil, log2
from typing import Callable

import numpy as np

from .metrics import conventional_tokens, edge_mutual_information, normalize_text


FEATURE_NAMES = (
    "glyph_entropy",
    "glyph_type_ratio",
    "dominant_glyph_share",
    "adjacent_repeat_rate",
    "space_rate",
    "mean_token_length",
    "token_length_cv",
    "token_type_ratio",
    "hapax_type_fraction",
    "cross_token_edge_mi",
)


@dataclass
class DiscriminatorResult:
    held_out_auc: float
    auc_sd: float
    auc_ci_low: float
    auc_ci_high: float
    n_splits: int
    n_real_blocks: int
    n_generated_blocks: int
    block_size: int
    note: str
    feature_names: tuple[str, ...] = FEATURE_NAMES


def _safe(value: float) -> float:
    return float(value) if value == value and np.isfinite(value) else 0.0


def _block_features(block: str) -> list[float]:
    """Features that see both glyph composition and conventional boundaries.

    Earlier versions erased whitespace before blocking, so layer 2 could not
    directly see the token/boundary structure highlighted by the scorecard.
    Blocks now preserve their internal spaces and include token-length and
    cross-boundary edge features in addition to the original four glyph-only
    composition statistics.
    """
    clean = normalize_text(block)
    glyphs = "".join(ch for ch in clean if not ch.isspace())
    n = len(glyphs)
    if n == 0:
        return [0.0] * len(FEATURE_NAMES)

    counts = Counter(glyphs)
    entropy = -sum((c / n) * log2(c / n) for c in counts.values())
    repeats = sum(1 for i in range(1, n) if glyphs[i] == glyphs[i - 1]) / max(1, n - 1)
    tokens = conventional_tokens(clean)
    token_counts = Counter(tokens)
    lengths = np.asarray([len(t) for t in tokens], dtype=float)
    mean_len = float(np.mean(lengths)) if len(lengths) else 0.0
    length_cv = float(np.std(lengths) / mean_len) if len(lengths) > 1 and mean_len > 0 else 0.0
    token_type_ratio = len(token_counts) / max(1, len(tokens))
    hapax = (
        sum(1 for c in token_counts.values() if c == 1) / len(token_counts)
        if token_counts
        else 0.0
    )
    edge_mi = edge_mutual_information(tokens)
    spaces = sum(1 for ch in clean if ch.isspace())
    space_rate = spaces / max(1, len(clean))

    return [
        entropy,
        len(counts) / n,
        counts.most_common(1)[0][1] / n,
        repeats,
        space_rate,
        mean_len,
        length_cv,
        token_type_ratio,
        hapax,
        _safe(edge_mi),
    ]


def _make_blocks(text: str, block_size: int) -> list[str]:
    """Chunk normalized text by approximate glyph count while retaining spaces.

    Token boundaries are kept intact whenever possible. A single token longer
    than ``block_size`` forms its own block. This lets the classifier use
    boundary features without leaking neighboring blocks across a random split.
    """
    clean = normalize_text(text)
    target = max(1, int(block_size))
    blocks: list[str] = []
    current: list[str] = []
    glyph_count = 0

    for line in clean.splitlines():
        for token in line.split():
            token_len = len(token)
            if current and glyph_count + token_len > target:
                blocks.append(" ".join(current))
                current = []
                glyph_count = 0
            current.append(token)
            glyph_count += token_len
            if glyph_count >= target:
                blocks.append(" ".join(current))
                current = []
                glyph_count = 0
    if current:
        blocks.append(" ".join(current))
    return blocks


def _macro_group_ids(n_blocks: int, prefix: str, target_groups: int = 8) -> list[str]:
    if n_blocks <= 0:
        return []
    width = max(1, ceil(n_blocks / max(2, target_groups)))
    return [f"{prefix}:{i // width}" for i in range(n_blocks)]


def classifier_two_sample_test(
    real_text: str,
    generator_fn: Callable[[str, int], str],
    block_size: int = 200,
    n_generated_replicates: int = 5,
    seed: int = 0,
    test_fraction: float = 0.3,
    n_splits: int = 12,
) -> DiscriminatorResult:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("classifier_two_sample_test requires scikit-learn") from exc

    real_blocks = _make_blocks(real_text, int(block_size))
    blocks = list(real_blocks)
    labels = [1] * len(real_blocks)
    groups = _macro_group_ids(len(real_blocks), "real")
    n_generated_replicates = max(1, int(n_generated_replicates))
    n_gen = 0
    for r in range(n_generated_replicates):
        gb = _make_blocks(generator_fn(real_text, seed + r), int(block_size))
        blocks.extend(gb)
        labels.extend([0] * len(gb))
        groups.extend(_macro_group_ids(len(gb), f"gen{r}"))
        n_gen += len(gb)

    if len(real_blocks) < 6 or n_gen < 6 or len(set(groups)) < 6:
        return DiscriminatorResult(
            float("nan"), float("nan"), float("nan"), float("nan"), 0,
            len(real_blocks), n_gen, int(block_size),
            "Too few independent block groups. Use a smaller block size or more text.",
        )

    X = np.asarray([_block_features(b) for b in blocks], dtype=float)
    y = np.asarray(labels, dtype=int)
    groups_arr = np.asarray(groups)
    splitter = GroupShuffleSplit(
        n_splits=max(3, int(n_splits)) * 3,
        test_size=test_fraction,
        random_state=seed,
    )
    aucs: list[float] = []
    for train_idx, test_idx in splitter.split(X, y, groups_arr):
        if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[test_idx], probs)))
        if len(aucs) >= max(3, int(n_splits)):
            break

    if not aucs:
        mean_auc = sd = lo = hi = float("nan")
    else:
        arr = np.asarray(aucs)
        mean_auc = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        lo, hi = (
            (float(x) for x in np.quantile(arr, [0.025, 0.975]))
            if len(arr) > 1
            else (mean_auc, mean_auc)
        )

    note = (
        "AUC is averaged over group-aware contiguous holdouts. Blocks retain internal token boundaries; "
        "the feature vector includes glyph composition, token-length/type statistics and cross-token edge MI. "
        "This is still posterior-predictive because the generator is fit to the full input; quire-held-out "
        "evaluation remains preferable when IVTFF $Q metadata are available."
    )
    return DiscriminatorResult(
        mean_auc, sd, lo, hi, len(aucs), len(real_blocks), n_gen, int(block_size), note
    )
