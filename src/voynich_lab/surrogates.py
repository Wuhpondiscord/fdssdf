from __future__ import annotations

from collections import Counter, defaultdict
import random

from .metrics import glyph_stream, normalize_text


def iid_glyph_surrogate(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    stream = glyph_stream(text)
    if not stream:
        return ""
    counts = Counter(stream)
    alphabet = list(counts)
    weights = [counts[ch] for ch in alphabet]
    return "".join(rng.choices(alphabet, weights=weights, k=len(stream)))


def markov1_surrogate(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    stream = glyph_stream(text)
    if len(stream) < 2:
        return stream
    transitions: dict[str, list[str]] = defaultdict(list)
    for a, b in zip(stream[:-1], stream[1:]):
        transitions[a].append(b)
    out = [rng.choice(stream)]
    for _ in range(len(stream) - 1):
        choices = transitions.get(out[-1])
        out.append(rng.choice(choices) if choices else rng.choice(stream))
    return "".join(out)


def token_shuffle_surrogate(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    tokens = normalize_text(text).split()
    rng.shuffle(tokens)
    return " ".join(tokens)


def markov_k_surrogate(text: str, order: int = 2, seed: int = 0) -> str:
    """Order-k Markov surrogate (N3/N4 on the nested null ladder). Falls
    back to a lower order wherever a k-gram context was never observed,
    rather than raising, so sparse high-order contexts degrade gracefully
    instead of crashing on short inputs."""
    rng = random.Random(seed)
    stream = glyph_stream(text)
    k = max(1, int(order))
    if len(stream) <= k:
        return stream
    transitions: dict[str, list[str]] = defaultdict(list)
    for i in range(len(stream) - k):
        ctx = stream[i : i + k]
        transitions[ctx].append(stream[i + k])
    out = list(stream[:k])
    for _ in range(len(stream) - k):
        ctx = "".join(out[-k:])
        choices = transitions.get(ctx)
        if not choices:
            # back off to order-1 on the fly rather than fail
            choices = transitions.get(out[-1], list(stream))
        out.append(rng.choice(choices))
    return "".join(out)


def block_shuffle_surrogate(text: str, block_size: int = 3, seed: int = 0) -> str:
    """Shuffle fixed-size glyph blocks rather than single glyphs (N5):
    preserves short local motifs of length <= block_size while destroying
    structure between blocks."""
    rng = random.Random(seed)
    stream = glyph_stream(text)
    b = max(1, int(block_size))
    blocks = [stream[i : i + b] for i in range(0, len(stream), b)]
    rng.shuffle(blocks)
    return "".join(blocks)


def position_conditioned_markov_surrogate(text: str, n_bins: int = 10, seed: int = 0) -> str:
    """Order-1 Markov surrogate whose transition table is conditioned on
    normalized within-line position decile (N6): tests whether apparent
    structure is explainable by known line-position effects (e.g.
    line-initial/final asymmetries) rather than needing a "real" mechanism."""
    rng = random.Random(seed)
    lines = [
        [ch for ch in line if not ch.isspace()]
        for line in normalize_text(text).splitlines()
    ]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""

    def _bin(i: int, n: int) -> int:
        denom = max(1, n - 1)
        return min(n_bins - 1, int((i / denom) * n_bins))

    transitions: dict[tuple[int, str], list[str]] = defaultdict(list)
    for ln in lines:
        for i in range(len(ln) - 1):
            key = (_bin(i, len(ln)), ln[i])
            transitions[key].append(ln[i + 1])
    all_glyphs = [ch for ln in lines for ch in ln]

    out_lines: list[str] = []
    for ln in lines:
        if not ln:
            continue
        out = [rng.choice(all_glyphs)]
        for i in range(len(ln) - 1):
            key = (_bin(i, len(ln)), out[-1])
            choices = transitions.get(key) or all_glyphs
            out.append(rng.choice(choices))
        out_lines.append("".join(out))
    return "\n".join(out_lines)


def quire_conditioned_surrogate(
    text_by_quire: dict[str, str], seed: int = 0
) -> dict[str, str]:
    """Order-1 Markov surrogate fit *separately per quire* (N7): returns one
    surrogate string per quire key so quire-level heterogeneity in the real
    manuscript can be compared against quire-level heterogeneity that a
    single global model would predict is absent."""
    return {
        quire: markov1_surrogate(sub_text, seed=seed)
        for quire, sub_text in text_by_quire.items()
    }
