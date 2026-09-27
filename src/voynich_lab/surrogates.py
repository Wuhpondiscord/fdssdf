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
