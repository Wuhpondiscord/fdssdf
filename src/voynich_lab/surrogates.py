from __future__ import annotations

from collections import Counter, defaultdict
import random

from .metrics import glyph_stream, normalize_text


def _restore_layout(template: str, replacement_stream: str) -> str:
    """Put generated glyphs back into the template's whitespace geometry."""
    template = normalize_text(template)
    expected = sum(1 for ch in template if not ch.isspace())
    if len(replacement_stream) != expected:
        raise ValueError(f"replacement stream has {len(replacement_stream)} glyphs; expected {expected}")
    it = iter(replacement_stream)
    out: list[str] = []
    for ch in template:
        out.append(ch if ch.isspace() else next(it))
    return "".join(out)


def iid_glyph_surrogate(text: str, seed: int = 0, preserve_layout: bool = True) -> str:
    rng = random.Random(seed)
    stream = glyph_stream(text)
    if not stream:
        return ""
    counts = Counter(stream)
    alphabet = list(counts)
    weights = [counts[ch] for ch in alphabet]
    generated = "".join(rng.choices(alphabet, weights=weights, k=len(stream)))
    return _restore_layout(text, generated) if preserve_layout else generated


def markov1_surrogate(text: str, seed: int = 0, preserve_layout: bool = True) -> str:
    return markov_k_surrogate(text, order=1, seed=seed, preserve_layout=preserve_layout)


def markov_k_surrogate(text: str, order: int = 2, seed: int = 0, preserve_layout: bool = True) -> str:
    """Variable-backoff order-k Markov surrogate."""
    rng = random.Random(seed)
    stream = glyph_stream(text)
    k = max(1, int(order))
    if len(stream) <= k:
        return normalize_text(text) if preserve_layout else stream
    tables: dict[int, dict[str, list[str]]] = {}
    for ctx_len in range(1, k + 1):
        table: dict[str, list[str]] = defaultdict(list)
        for i in range(len(stream) - ctx_len):
            table[stream[i : i + ctx_len]].append(stream[i + ctx_len])
        tables[ctx_len] = table
    counts = Counter(stream)
    alphabet = list(counts)
    weights = [counts[ch] for ch in alphabet]
    start = rng.randrange(0, len(stream) - k + 1)
    out = list(stream[start : start + k])
    while len(out) < len(stream):
        choices = None
        for ctx_len in range(min(k, len(out)), 0, -1):
            ctx = "".join(out[-ctx_len:])
            choices = tables[ctx_len].get(ctx)
            if choices:
                break
        if choices:
            out.append(rng.choice(choices))
        else:
            out.append(rng.choices(alphabet, weights=weights, k=1)[0])
    generated = "".join(out[: len(stream)])
    return _restore_layout(text, generated) if preserve_layout else generated


def token_shuffle_surrogate(text: str, seed: int = 0) -> str:
    """Shuffle token identities while preserving line token counts."""
    rng = random.Random(seed)
    lines = [line.split() for line in normalize_text(text).splitlines()]
    flat = [tok for line in lines for tok in line]
    rng.shuffle(flat)
    out_lines: list[str] = []
    offset = 0
    for line in lines:
        n = len(line)
        out_lines.append(" ".join(flat[offset : offset + n]))
        offset += n
    return "\n".join(out_lines)


def block_shuffle_surrogate(text: str, block_size: int = 3, seed: int = 0, preserve_layout: bool = True) -> str:
    rng = random.Random(seed)
    stream = glyph_stream(text)
    b = max(1, int(block_size))
    blocks = [stream[i : i + b] for i in range(0, len(stream), b)]
    rng.shuffle(blocks)
    generated = "".join(blocks)
    return _restore_layout(text, generated) if preserve_layout else generated


def position_conditioned_markov_surrogate(text: str, n_bins: int = 10, seed: int = 0) -> str:
    """Order-1 Markov null conditioned on normalized within-line position."""
    rng = random.Random(seed)
    clean = normalize_text(text)
    raw_lines = clean.splitlines()
    glyph_lines = [[ch for ch in line if not ch.isspace()] for line in raw_lines]
    glyph_lines = [ln for ln in glyph_lines if ln]
    if not glyph_lines:
        return ""

    def _bin(i: int, n: int) -> int:
        denom = max(1, n - 1)
        return min(n_bins - 1, int((i / denom) * n_bins))

    transitions: dict[tuple[int, str], list[str]] = defaultdict(list)
    starts: dict[int, list[str]] = defaultdict(list)
    all_glyphs = [ch for ln in glyph_lines for ch in ln]
    for ln in glyph_lines:
        starts[_bin(0, len(ln))].append(ln[0])
        for i in range(len(ln) - 1):
            transitions[(_bin(i, len(ln)), ln[i])].append(ln[i + 1])

    generated_lines: list[str] = []
    for ln in glyph_lines:
        out = [rng.choice(starts.get(0) or all_glyphs)]
        for i in range(len(ln) - 1):
            choices = transitions.get((_bin(i, len(ln)), out[-1])) or all_glyphs
            out.append(rng.choice(choices))
        generated_lines.append("".join(out))

    result_lines: list[str] = []
    gi = 0
    for raw_line in raw_lines:
        if not any(not ch.isspace() for ch in raw_line):
            result_lines.append(raw_line)
            continue
        result_lines.append(_restore_layout(raw_line, generated_lines[gi]))
        gi += 1
    return "\n".join(result_lines)


def quire_conditioned_surrogate(text_by_quire: dict[str, str], seed: int = 0) -> dict[str, str]:
    return {
        quire: markov1_surrogate(sub_text, seed=seed + i)
        for i, (quire, sub_text) in enumerate(sorted(text_by_quire.items()))
    }
