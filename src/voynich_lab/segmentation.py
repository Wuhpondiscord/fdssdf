from __future__ import annotations

from collections import Counter
from math import log2

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


def learn_bpe_merges(train_text: str, merges: int = 64) -> list[tuple[str, str]]:
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
    symbols = list(glyph_stream(text))
    for pair in merge_table:
        symbols = _merge_pair(symbols, pair, "".join(pair))
    return symbols


def _entropy_units(units: list[str]) -> float:
    if not units:
        return float("nan")
    counts = Counter(units)
    n = len(units)
    return -sum((c / n) * log2(c / n) for c in counts.values())


def unit_dependence_gap(units: list[str]) -> float:
    """Adjacent-unit dependence D = H(U[t+1]) - H(U[t+1]|U[t]), bits."""
    if len(units) < 2:
        return float("nan")
    left = units[:-1]
    right = units[1:]
    n = len(left)
    left_counts = Counter(left)
    pair_counts = Counter(zip(left, right))
    h_cond = 0.0
    for (a, _b), c_ab in pair_counts.items():
        p_ab = c_ab / n
        p_b_given_a = c_ab / left_counts[a]
        h_cond -= p_ab * log2(p_b_given_a)
    return _entropy_units(right) - h_cond


def cross_fit_bpe_by_quire(text_by_quire: dict[str, str], merges: int = 64) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for held_out in text_by_quire:
        train_text = "\n".join(t for q, t in text_by_quire.items() if q != held_out)
        merge_table = learn_bpe_merges(train_text, merges=merges)
        segmented = apply_bpe_merges(text_by_quire[held_out], merge_table)
        results[held_out] = {
            "n_merges_learned": len(merge_table),
            "held_out_unit_count": len(segmented),
            "held_out_unit_types": len(set(segmented)),
            "mean_unit_length": (sum(len(u) for u in segmented) / len(segmented)) if segmented else float("nan"),
            "dependence_gap_bits": unit_dependence_gap(segmented),
        }
    return results


def cross_fit_bpe_scale_curve(
    text_by_quire: dict[str, str], checkpoints: tuple[int, ...] = (0, 16, 32, 64)
) -> tuple[list[dict[str, object]], int | None]:
    """Leave-one-quire-out BPE scale curve with glyph-weighted aggregation."""
    if len(text_by_quire) < 2:
        return [], None
    checkpoints = tuple(sorted({max(0, int(c)) for c in checkpoints}))
    max_merges = max(checkpoints, default=0)
    fold_tables: dict[str, list[tuple[str, str]]] = {}
    for held_out in text_by_quire:
        train = "\n".join(t for q, t in text_by_quire.items() if q != held_out)
        fold_tables[held_out] = learn_bpe_merges(train, merges=max_merges)

    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        weighted_sum = 0.0
        total_weight = 0
        fold_values: dict[str, float] = {}
        for held_out, held_text in text_by_quire.items():
            units = apply_bpe_merges(held_text, fold_tables[held_out][:checkpoint])
            gap = unit_dependence_gap(units)
            fold_values[held_out] = gap
            weight = len(glyph_stream(held_text))
            if gap == gap and weight > 0:
                weighted_sum += gap * weight
                total_weight += weight
        rows.append({
            "merges": checkpoint,
            "glyph_weighted_dependence_gap_bits": weighted_sum / total_weight if total_weight else float("nan"),
            "folds": len(fold_values),
            "fold_values": fold_values,
        })
    finite = [r for r in rows if r["glyph_weighted_dependence_gap_bits"] == r["glyph_weighted_dependence_gap_bits"]]
    selected = min(finite, key=lambda r: r["glyph_weighted_dependence_gap_bits"])["merges"] if finite else None
    return rows, int(selected) if selected is not None else None
