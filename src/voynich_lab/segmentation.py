from __future__ import annotations

from collections import Counter
from math import log2

from .metrics import glyph_stream, normalize_text


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
    """Exploratory BPE over the continuous space-erased glyph stream.

    This intentionally remains separate from the reportable literature
    reproduction below. Rozanova & Temerev's unit-scale experiment learns
    merges *inside conventional tokens*; use the token-BPE helpers for that.
    """
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
        history.append(
            {
                "step": step + 1,
                "left": pair[0],
                "right": pair[1],
                "merged": merged,
                "frequency_before_merge": freq,
                "sequence_length_after": len(symbols),
            }
        )
    return symbols, history


def learn_bpe_merges(train_text: str, merges: int = 64) -> list[tuple[str, str]]:
    """Exploratory continuous-stream merge table kept for backwards compatibility."""
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
    """Apply the exploratory continuous-stream merge table."""
    symbols = list(glyph_stream(text))
    for pair in merge_table:
        symbols = _merge_pair(symbols, pair, "".join(pair))
    return symbols


# ---------------------------------------------------------------------------
# Published unit-scale method: BPE is learned/applied within token types.
# ---------------------------------------------------------------------------

TokenLines = list[list[str]]
BPERule = tuple[str, str, str]


def text_to_token_lines(text: str) -> TokenLines:
    """Convert normalized transcription text into manuscript lines of tokens."""
    return [line.split() for line in normalize_text(text).splitlines() if line.split()]


def learn_token_bpe_rules(lines: TokenLines, max_merges: int = 64) -> list[BPERule]:
    """Learn frequency-weighted within-token BPE rules.

    Mirrors ``analysis/reproduce_unit_scale.py`` in the authors' public
    reproduction repository: token types have their own segmentations, pair
    counts are weighted by token frequency, and ties are broken by the pair
    itself after frequency (``max(..., key=(count, pair))``).
    """
    word_frequency = Counter(word for line in lines for word in line)
    segmentations: dict[str, tuple[str, ...]] = {
        word: tuple(word) for word in word_frequency
    }
    rules: list[BPERule] = []
    for _ in range(max(0, int(max_merges))):
        pair_frequency: Counter[tuple[str, str]] = Counter()
        for word, units in segmentations.items():
            frequency = word_frequency[word]
            for pair in zip(units, units[1:]):
                pair_frequency[pair] += frequency
        if not pair_frequency:
            break
        (left, right), count = max(
            pair_frequency.items(), key=lambda item: (item[1], item[0])
        )
        if count < 2:
            break
        merged = left + right
        rules.append((left, right, merged))
        for word, units in list(segmentations.items()):
            segmentations[word] = tuple(
                _merge_pair(list(units), (left, right), merged)
            )
    return rules


def apply_token_bpe_rules(
    lines: TokenLines,
    rules: list[BPERule],
    merge_count: int | None = None,
) -> dict[str, tuple[str, ...]]:
    """Apply training BPE rules to held-out token types, never across spaces."""
    segmentations: dict[str, tuple[str, ...]] = {
        word: tuple(word) for line in lines for word in line
    }
    n_rules = len(rules) if merge_count is None else max(0, int(merge_count))
    for left, right, merged in rules[:n_rules]:
        for word, units in list(segmentations.items()):
            segmentations[word] = tuple(
                _merge_pair(list(units), (left, right), merged)
            )
    return segmentations


def unit_stream_stats(
    lines: TokenLines,
    segmentation: dict[str, tuple[str, ...]],
) -> dict[str, float | int]:
    """Exact unit-scale scoring convention used by the public reproduction.

    BPE units stay token-internal, but adjacent-unit bigrams cross conventional
    token boundaries *within each manuscript line*. They never cross a line
    boundary. ``gap`` is H1 - H2 where H2 = H(next | previous).
    """
    unigram: Counter[str] = Counter()
    bigram: Counter[tuple[str, str]] = Counter()
    glyphs = 0
    for line in lines:
        units: list[str] = []
        for word in line:
            units.extend(segmentation[word])
            glyphs += len(word)
        unigram.update(units)
        bigram.update(zip(units, units[1:]))

    n_units = sum(unigram.values())
    if not n_units or not glyphs:
        return {
            "types": 0,
            "units": 0,
            "glyphs": glyphs,
            "mean_len": float("nan"),
            "H1": float("nan"),
            "H2": float("nan"),
            "bits_per_glyph": float("nan"),
            "gap": float("nan"),
        }

    h1 = -sum(
        (count / n_units) * log2(count / n_units) for count in unigram.values()
    )
    n_bigrams = sum(bigram.values())
    if n_bigrams:
        h12 = -sum(
            (count / n_bigrams) * log2(count / n_bigrams)
            for count in bigram.values()
        )
        previous = Counter()
        for (left, _right), count in bigram.items():
            previous[left] += count
        h_prev = -sum(
            (count / n_bigrams) * log2(count / n_bigrams)
            for count in previous.values()
        )
        h2 = h12 - h_prev
    else:
        h2 = float("nan")

    return {
        "types": len(unigram),
        "units": n_units,
        "glyphs": glyphs,
        "mean_len": glyphs / n_units,
        "H1": h1,
        "H2": h2,
        "bits_per_glyph": (h2 * n_units / glyphs) if h2 == h2 else float("nan"),
        "gap": (h1 - h2) if h2 == h2 else float("nan"),
    }


def cross_fit_bpe_by_quire(
    text_by_quire: dict[str, str], merges: int = 64
) -> dict[str, dict[str, object]]:
    """Leave-one-quire-out within-token BPE at one fixed merge count."""
    results: dict[str, dict[str, object]] = {}
    quires = list(text_by_quire)
    if len(quires) < 2:
        return results
    for held_out in quires:
        training_lines = [
            line
            for quire, text in text_by_quire.items()
            if quire != held_out
            for line in text_to_token_lines(text)
        ]
        held_lines = text_to_token_lines(text_by_quire[held_out])
        rules = learn_token_bpe_rules(training_lines, max_merges=int(merges))
        segmentation = apply_token_bpe_rules(
            held_lines, rules, merge_count=int(merges)
        )
        stats = unit_stream_stats(held_lines, segmentation)
        results[held_out] = {
            "n_merges_learned": len(rules),
            "held_out_unit_count": stats["units"],
            "held_out_unit_types": stats["types"],
            "mean_unit_length": stats["mean_len"],
            "H1_bits": stats["H1"],
            "H2_conditional_bits": stats["H2"],
            "bits_per_glyph": stats["bits_per_glyph"],
            "dependence_gap_bits": stats["gap"],
            "held_out_glyphs": stats["glyphs"],
        }
    return results


def cross_fit_bpe_scale_curve(
    text_by_quire: dict[str, str],
    checkpoints: tuple[int, ...] = (0, 16, 32, 64),
) -> tuple[list[dict[str, object]], int | None]:
    """Published-method leave-one-quire-out BPE scale curve.

    For each omitted quire, rules are learned on all remaining quires, applied
    to held-out token types, and scored only on the held-out quire. Fold
    statistics are aggregated with held-out glyph counts as weights, matching
    the public reproduction driver.
    """
    if len(text_by_quire) < 2:
        return [], None
    checkpoints = tuple(sorted({max(0, int(c)) for c in checkpoints}))
    if not checkpoints:
        return [], None
    max_merges = max(checkpoints)

    folds: dict[str, dict[int, dict[str, float | int]]] = {}
    rules_by_quire: dict[str, list[BPERule]] = {}
    for held_out in text_by_quire:
        training_lines = [
            line
            for quire, text in text_by_quire.items()
            if quire != held_out
            for line in text_to_token_lines(text)
        ]
        held_lines = text_to_token_lines(text_by_quire[held_out])
        rules = learn_token_bpe_rules(training_lines, max_merges=max_merges)
        rules_by_quire[held_out] = rules
        folds[held_out] = {}
        for checkpoint in checkpoints:
            segmentation = apply_token_bpe_rules(
                held_lines, rules, merge_count=checkpoint
            )
            folds[held_out][checkpoint] = unit_stream_stats(
                held_lines, segmentation
            )

    rows: list[dict[str, object]] = []
    aggregate_metrics = ("mean_len", "H1", "H2", "bits_per_glyph", "gap")
    for checkpoint in checkpoints:
        weights = [
            float(folds[q][checkpoint]["glyphs"]) for q in text_by_quire
        ]
        total_weight = sum(weights)
        aggregated: dict[str, float] = {}
        for metric in aggregate_metrics:
            values = [
                float(folds[q][checkpoint][metric]) for q in text_by_quire
            ]
            usable = [
                (value, weight)
                for value, weight in zip(values, weights)
                if value == value and weight > 0
            ]
            denom = sum(weight for _value, weight in usable)
            aggregated[metric] = (
                sum(value * weight for value, weight in usable) / denom
                if denom
                else float("nan")
            )
        fold_values = {
            q: float(folds[q][checkpoint]["gap"]) for q in text_by_quire
        }
        rows.append(
            {
                "merges": checkpoint,
                "glyph_weighted_dependence_gap_bits": aggregated["gap"],
                "mean_unit_length": aggregated["mean_len"],
                "H1_bits": aggregated["H1"],
                "H2_conditional_bits": aggregated["H2"],
                "bits_per_glyph": aggregated["bits_per_glyph"],
                "glyphs": int(total_weight),
                "folds": len(text_by_quire),
                "fold_values": fold_values,
                "merges_learned_by_fold": {
                    q: len(rules_by_quire[q]) for q in text_by_quire
                },
            }
        )

    finite = [
        row
        for row in rows
        if row["glyph_weighted_dependence_gap_bits"]
        == row["glyph_weighted_dependence_gap_bits"]
    ]
    selected = (
        min(finite, key=lambda row: row["glyph_weighted_dependence_gap_bits"])[
            "merges"
        ]
        if finite
        else None
    )
    return rows, int(selected) if selected is not None else None
