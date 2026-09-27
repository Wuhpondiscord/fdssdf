from __future__ import annotations

from collections import Counter

from .metrics import glyph_stream


def _pair_counts(symbols: list[str]) -> Counter[tuple[str, str]]:
    return Counter(zip(symbols[:-1], symbols[1:]))


def _merge_pair(symbols: list[str], pair: tuple[str, str], merged: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def discover_bpe_units(text: str, merges: int = 32) -> tuple[list[str], list[dict[str, object]]]:
    symbols = list(glyph_stream(text))
    history: list[dict[str, object]] = []
    for step in range(max(0, int(merges))):
        counts = _pair_counts(symbols)
        if not counts:
            break
        pair, freq = counts.most_common(1)[0]
        if freq < 2:
            break
        merged = "".join(pair)
        symbols = _merge_pair(symbols, pair, merged)
        history.append({"step": step + 1, "left": pair[0], "right": pair[1], "merged": merged, "frequency_before_merge": freq, "sequence_length_after": len(symbols)})
    return symbols, history
