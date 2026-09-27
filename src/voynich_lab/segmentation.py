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
    """In-sample BPE. Kept for quick exploration, but note: learning and
    evaluating merges on the same text can manufacture apparent structure.
    Use learn_bpe_merges / apply_bpe_merges for any result you intend to
    report, so the held-out discipline the 2026 unit-discovery paper used
    is actually followed rather than merely mentioned in the README."""
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


def learn_bpe_merges(train_text: str, merges: int = 64) -> list[tuple[str, str]]:
    """Learn an ordered merge table on training data only. Returns the merge
    sequence, not the segmented training text, so it can be re-applied
    unchanged to held-out data."""
    symbols = list(glyph_stream(train_text))
    merge_table: list[tuple[str, str]] = []
    for _ in range(max(0, int(merges))):
        counts = _pair_counts(symbols)
        if not counts:
            break
        pair, freq = counts.most_common(1)[0]
        if freq < 2:
            break
        merge_table.append(pair)
        symbols = _merge_pair(symbols, pair, "".join(pair))
    return merge_table


def apply_bpe_merges(text: str, merge_table: list[tuple[str, str]]) -> list[str]:
    """Apply a merge table learned elsewhere, in the order it was learned.
    This is the operation that must be run on held-out quires."""
    symbols = list(glyph_stream(text))
    for pair in merge_table:
        symbols = _merge_pair(symbols, pair, "".join(pair))
    return symbols


def cross_fit_bpe_by_quire(
    text_by_quire: dict[str, str], merges: int = 64
) -> dict[str, dict[str, object]]:
    """Leave-one-quire-out cross-fitting: for each held-out quire, learn
    merges on every other quire and segment the held-out quire with that
    frozen merge table. Returns per-quire held-out unit counts/lengths so
    unit-scale stability across folds can be assessed directly, mirroring
    the review's "held-out description length / boundary stability /
    cross-quire stability" comparison table.
    """
    results: dict[str, dict[str, object]] = {}
    quires = list(text_by_quire)
    for held_out in quires:
        train_text = "\n".join(t for q, t in text_by_quire.items() if q != held_out)
        merge_table = learn_bpe_merges(train_text, merges=merges)
        segmented = apply_bpe_merges(text_by_quire[held_out], merge_table)
        results[held_out] = {
            "n_merges_learned": len(merge_table),
            "held_out_unit_count": len(segmented),
            "held_out_unit_types": len(set(segmented)),
            "mean_unit_length": (sum(len(u) for u in segmented) / len(segmented)) if segmented else float("nan"),
        }
    return results
