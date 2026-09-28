from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable
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


def markov_k_surrogate(text: str, *, order: int = 2, seed: int = 0, preserve_layout: bool = True) -> str:
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


def block_shuffle_surrogate(text: str, *, block_size: int = 3, seed: int = 0, preserve_layout: bool = True) -> str:
    rng = random.Random(seed)
    stream = glyph_stream(text)
    b = max(1, int(block_size))
    blocks = [stream[i : i + b] for i in range(0, len(stream), b)]
    rng.shuffle(blocks)
    generated = "".join(blocks)
    return _restore_layout(text, generated) if preserve_layout else generated


def position_conditioned_markov_surrogate(text: str, *, n_bins: int = 10, seed: int = 0) -> str:
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


def bind_generator(fn: Callable[..., str], **params) -> Callable[[str, int], str]:
    """Bind hyperparameters and return a callable honouring the ``(text, seed) -> text`` contract.

    ``run_scorecard`` and ``classifier_two_sample_test`` call ``generator_fn(text, seed)`` positionally.
    ``functools.partial(fn, order=3)`` breaks that contract (the seed collides with, or is silently
    consumed as, a hyperparameter), so anything that has hyperparameters should be bound through here.
    """

    def bound(text: str, seed: int = 0) -> str:
        return fn(text, seed=seed, **params)

    bound.__name__ = f"{getattr(fn, '__name__', 'generator')}_bound"
    bound.__doc__ = f"{getattr(fn, '__name__', 'generator')} with {params!r} bound"
    return bound


_BOS = "\x02"  # beginning-of-line context padding; never appears in transcriptions


def boundary_markov_surrogate(text: str, *, order: int = 3, seed: int = 0, max_line_symbols: int = 600) -> str:
    """Order-k Markov null over glyphs **plus** word-space and end-of-line symbols.

    Every other Markov null in the ladder is fit to the space-free glyph stream and then has the original
    spaces pasted back in. That silently answers "what if spaces carried no information?", and all of
    them overshoot the observed cross-boundary coupling by ~5x. This null instead lets the chain
    generate its own word boundaries, i.e. the standard character n-gram model of running text with
    spaces as ordinary symbols. It is the fair baseline for the claim that boundary-edge structure
    exceeds what a local character n-gram already explains.

    Generates the same number of lines as the input; each line ends when the chain emits an
    end-of-line symbol, so total size (and token length) are *emergent*, not matched.
    """
    rng = random.Random(seed)
    lines = [ln.split() for ln in normalize_text(text).splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    k = max(1, int(order))
    tables: dict[int, dict[str, list[str]]] = {c: defaultdict(list) for c in range(1, k + 1)}
    for toks in lines:
        seq = [_BOS] * k + list(" ".join(toks)) + ["\n"]
        for i in range(k, len(seq)):
            nxt = seq[i]
            for c in range(1, k + 1):
                tables[c]["".join(seq[i - c : i])].append(nxt)

    out_lines: list[str] = []
    for _ in range(len(lines)):
        out = [_BOS] * k
        emitted = 0
        while emitted < int(max_line_symbols):
            choices = None
            for c in range(k, 0, -1):
                choices = tables[c].get("".join(out[-c:]))
                if choices:
                    break
            nxt = rng.choice(choices)
            if nxt == "\n":
                break
            out.append(nxt)
            emitted += 1
        out_lines.append("".join(out[k:]).strip())
    return "\n".join(out_lines)


def quire_conditioned_surrogate(text_by_quire: dict[str, str], seed: int = 0) -> dict[str, str]:
    return {
        quire: markov1_surrogate(sub_text, seed=seed + i)
        for i, (quire, sub_text) in enumerate(sorted(text_by_quire.items()))
    }


# Nested null ladder (N0-N6, plus boundary-aware N2b/N3b). N7 stratified nulls are built per document with
# stratified.stratified_generator because they need per-line quire / Currier labels, not just raw text.
#
# Every entry honours the ``(text, seed) -> text`` contract that run_scorecard and
# classifier_two_sample_test call positionally. Hyperparameters are bound with ``bind_generator`` --
# never ``functools.partial``, which lets the seed collide with (or be silently consumed as) a hyperparameter.
#
# N1-N4 and N6 are fit to the space-free glyph stream and get the original spaces pasted back, i.e. they
# assume boundaries carry no information. N2b/N3b let the chain generate its own word boundaries, which is
# the fair baseline for any claim about boundary-edge structure.
#
# Kept here (not in harness.py) so it is importable without pandas, matching the lightweight CI environment.
NULL_LADDER: dict[str, Callable[[str, int], str]] = {
    "N0_iid_glyph": iid_glyph_surrogate,
    "N1_markov1_glyph": markov1_surrogate,
    "N2_markov3_glyph": bind_generator(markov_k_surrogate, order=3),
    "N2b_boundary_markov2": bind_generator(boundary_markov_surrogate, order=2),
    "N3_markov5_glyph": bind_generator(markov_k_surrogate, order=5),
    "N3b_boundary_markov3": bind_generator(boundary_markov_surrogate, order=3),
    "N4_block_shuffle_3": bind_generator(block_shuffle_surrogate, block_size=3),
    "N5_token_shuffle": token_shuffle_surrogate,
    "N6_position_conditioned_markov": bind_generator(position_conditioned_markov_surrogate, n_bins=10),
}
