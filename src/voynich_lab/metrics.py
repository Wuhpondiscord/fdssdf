from __future__ import annotations

from collections import Counter, defaultdict
from math import log2
import re
from typing import Iterable

import numpy as np

SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [SPACE_RE.sub(" ", line.strip()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def conventional_tokens(text: str) -> list[str]:
    return [tok for tok in re.split(r"\s+", normalize_text(text)) if tok]


def glyph_stream(text: str, keep_spaces: bool = False) -> str:
    text = normalize_text(text)
    if keep_spaces:
        return text.replace("\n", " ")
    return "".join(ch for ch in text if not ch.isspace())


def entropy(symbols: Iterable[str]) -> float:
    seq = list(symbols)
    n = len(seq)
    if n == 0:
        return float("nan")
    counts = Counter(seq)
    return -sum((c / n) * log2(c / n) for c in counts.values())


def conditional_entropy_order1(stream: str) -> float:
    if len(stream) < 2:
        return float("nan")
    prev_counts = Counter(stream[:-1])
    pair_counts = Counter(zip(stream[:-1], stream[1:]))
    n_pairs = len(stream) - 1
    h = 0.0
    for (a, _b), c_ab in pair_counts.items():
        p_ab = c_ab / n_pairs
        p_b_given_a = c_ab / prev_counts[a]
        h -= p_ab * log2(p_b_given_a)
    return h


def mutual_information_pairs(xs: list[str], ys: list[str]) -> float:
    if len(xs) != len(ys) or not xs:
        return float("nan")
    n = len(xs)
    cx, cy, cxy = Counter(xs), Counter(ys), Counter(zip(xs, ys))
    mi = 0.0
    for (x, y), c in cxy.items():
        pxy = c / n
        px = cx[x] / n
        py = cy[y] / n
        mi += pxy * log2(pxy / (px * py))
    return mi


def adjacent_glyph_mi(stream: str) -> float:
    if len(stream) < 2:
        return float("nan")
    return mutual_information_pairs(list(stream[:-1]), list(stream[1:]))


def edge_mutual_information(tokens: list[str]) -> float:
    tokens = [t for t in tokens if t]
    if len(tokens) < 2:
        return float("nan")
    left = [tokens[i][-1] for i in range(len(tokens) - 1)]
    right = [tokens[i + 1][0] for i in range(len(tokens) - 1)]
    return mutual_information_pairs(left, right)


def zipf_slope(tokens: list[str]) -> float:
    counts = np.array(sorted(Counter(tokens).values(), reverse=True), dtype=float)
    if len(counts) < 3:
        return float("nan")
    ranks = np.arange(1, len(counts) + 1, dtype=float)
    slope, _intercept = np.polyfit(np.log(ranks), np.log(counts), 1)
    return float(slope)


def positional_entropy_by_decile(text: str) -> dict[str, float]:
    bins: dict[int, list[str]] = defaultdict(list)
    for line in normalize_text(text).splitlines():
        chars = [c for c in line if not c.isspace()]
        if not chars:
            continue
        denom = max(1, len(chars) - 1)
        for i, ch in enumerate(chars):
            b = min(9, int((i / denom) * 10))
            bins[b].append(ch)
    return {f"decile_{b}": entropy(chars) for b, chars in sorted(bins.items())}


def compute_metrics(text: str) -> dict[str, float | int]:
    text = normalize_text(text)
    stream = glyph_stream(text)
    tokens = conventional_tokens(text)
    token_counts = Counter(tokens)
    return {
        "glyph_count": len(stream),
        "glyph_types": len(set(stream)),
        "conventional_token_count": len(tokens),
        "conventional_token_types": len(token_counts),
        "hapax_fraction": (sum(1 for c in token_counts.values() if c == 1) / len(token_counts)) if token_counts else float("nan"),
        "mean_token_length": float(np.mean([len(t) for t in tokens])) if tokens else float("nan"),
        "glyph_entropy_bits": entropy(stream),
        "conditional_entropy_order1_bits": conditional_entropy_order1(stream),
        "adjacent_glyph_mi_bits": adjacent_glyph_mi(stream),
        "cross_token_edge_mi_bits": edge_mutual_information(tokens),
        "zipf_loglog_slope": zipf_slope(tokens),
    }
