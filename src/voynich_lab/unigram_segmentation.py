from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .metrics import normalize_text


_LOG2 = math.log(2.0)
_NEG_INF = float("-inf")


@dataclass(frozen=True)
class LineObservation:
    """One transcription line with whitespace erased but boundary positions retained."""

    glyphs: str
    whitespace_boundaries: frozenset[int]


@dataclass(frozen=True)
class UnigramSegmentationModel:
    """Finite-vocabulary unigram latent-segmentation model.

    ``log_probabilities`` are natural-log token probabilities learned by EM.
    The fixed length penalty is kept separate so experiments can explicitly
    test whether longer latent units are being preferred merely because they
    reduce token count.
    """

    log_probabilities: dict[str, float]
    max_unit_length: int
    length_penalty_bits: float
    iterations_run: int
    training_partition_bits: float
    vocabulary_size: int

    def unit_score(self, unit: str) -> float:
        return self.log_probabilities[unit] - (
            self.length_penalty_bits * max(0, len(unit) - 1) * _LOG2
        )


@dataclass(frozen=True)
class SegmentationConsensusResult:
    summary: dict[str, float | int]
    line_rows: tuple[dict[str, object], ...]
    unit_rows: tuple[dict[str, object], ...]
    preview: str
    model: UnigramSegmentationModel
    bpe_rules: tuple[tuple[str, str], ...]


def line_observations(text: str) -> list[LineObservation]:
    """Build line-bounded glyph sequences without assuming spaces are words.

    Whitespace is removed from the sequence given to both segmentation models,
    but its positions are retained as an observed transcription-boundary layer
    for later comparison. No learned unit is allowed to cross a manuscript line
    boundary in this module.
    """

    observations: list[LineObservation] = []
    for raw_line in normalize_text(text).splitlines():
        tokens = raw_line.split()
        if not tokens:
            continue
        glyphs = "".join(tokens)
        if not glyphs:
            continue
        boundaries: set[int] = set()
        cursor = 0
        for token in tokens[:-1]:
            cursor += len(token)
            if 0 < cursor < len(glyphs):
                boundaries.add(cursor)
        observations.append(LineObservation(glyphs, frozenset(boundaries)))
    return observations


def _logsumexp(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return _NEG_INF
    maximum = max(values)
    if maximum == _NEG_INF:
        return _NEG_INF
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _substring_counts(sequences: Sequence[str], max_unit_length: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sequence in sequences:
        n = len(sequence)
        for start in range(n):
            stop_max = min(n, start + max_unit_length)
            for stop in range(start + 1, stop_max + 1):
                counts[sequence[start:stop]] += 1
    return counts


def candidate_vocabulary(
    sequences: Sequence[str],
    *,
    max_unit_length: int = 8,
    min_count: int = 2,
    max_vocab: int = 1024,
) -> tuple[str, ...]:
    """Construct a bounded candidate vocabulary without using spaces.

    Every observed single glyph is protected so every sequence remains
    segmentable. Longer candidates are ranked by recurring compression
    opportunity ``frequency * (length - 1)``; this is only candidate selection,
    not the EM objective itself.
    """

    max_unit_length = max(1, int(max_unit_length))
    min_count = max(1, int(min_count))
    max_vocab = max(1, int(max_vocab))
    counts = _substring_counts(sequences, max_unit_length)
    singles = sorted(unit for unit in counts if len(unit) == 1)
    if max_vocab < len(singles):
        max_vocab = len(singles)
    longer = [
        unit
        for unit, count in counts.items()
        if len(unit) > 1 and count >= min_count
    ]
    longer.sort(
        key=lambda unit: (
            counts[unit] * (len(unit) - 1),
            counts[unit],
            len(unit),
            unit,
        ),
        reverse=True,
    )
    keep = singles + longer[: max(0, max_vocab - len(singles))]
    return tuple(keep)


def _initial_log_probabilities(
    sequences: Sequence[str], vocabulary: Sequence[str]
) -> dict[str, float]:
    counts = _substring_counts(sequences, max(len(unit) for unit in vocabulary))
    # Temper raw substring counts so high-frequency single glyphs do not consume
    # nearly all probability mass at initialization.
    weights = {unit: math.sqrt(float(counts[unit])) for unit in vocabulary}
    total = sum(weights.values())
    return {unit: math.log(weight / total) for unit, weight in weights.items()}


def _matching_units(
    sequence: str,
    start: int,
    vocabulary: set[str],
    max_unit_length: int,
):
    stop_max = min(len(sequence), start + max_unit_length)
    for stop in range(start + 1, stop_max + 1):
        unit = sequence[start:stop]
        if unit in vocabulary:
            yield stop, unit


def _forward_backward_expected_counts(
    sequence: str,
    log_probabilities: dict[str, float],
    *,
    max_unit_length: int,
    length_penalty_bits: float,
) -> tuple[Counter[str], float]:
    n = len(sequence)
    vocabulary = set(log_probabilities)

    def score(unit: str) -> float:
        return log_probabilities[unit] - (
            length_penalty_bits * max(0, len(unit) - 1) * _LOG2
        )

    forward = [_NEG_INF] * (n + 1)
    forward[0] = 0.0
    for start in range(n):
        if forward[start] == _NEG_INF:
            continue
        for stop, unit in _matching_units(
            sequence, start, vocabulary, max_unit_length
        ):
            candidate = forward[start] + score(unit)
            forward[stop] = _logsumexp((forward[stop], candidate))

    log_partition = forward[n]
    if log_partition == _NEG_INF:
        raise RuntimeError("unigram segmentation vocabulary cannot cover a sequence")

    backward = [_NEG_INF] * (n + 1)
    backward[n] = 0.0
    for start in range(n - 1, -1, -1):
        values: list[float] = []
        for stop, unit in _matching_units(
            sequence, start, vocabulary, max_unit_length
        ):
            if backward[stop] != _NEG_INF:
                values.append(score(unit) + backward[stop])
        backward[start] = _logsumexp(values)

    expected: Counter[str] = Counter()
    for start in range(n):
        if forward[start] == _NEG_INF:
            continue
        for stop, unit in _matching_units(
            sequence, start, vocabulary, max_unit_length
        ):
            if backward[stop] == _NEG_INF:
                continue
            log_posterior = (
                forward[start]
                + score(unit)
                + backward[stop]
                - log_partition
            )
            expected[unit] += math.exp(log_posterior)
    return expected, log_partition


def fit_unigram_segmenter(
    sequences: Sequence[str],
    *,
    max_unit_length: int = 8,
    min_count: int = 2,
    max_vocab: int = 1024,
    iterations: int = 12,
    smoothing: float = 1e-3,
    length_penalty_bits: float = 0.0,
    convergence_tol: float = 1e-6,
) -> UnigramSegmentationModel:
    """Fit a SentencePiece-like finite unigram segmentation model by EM.

    This is intentionally independent of pair-merging BPE. Candidate units are
    recurring substrings, while the latent segmentation itself is inferred by
    summing over *all* tokenizations allowed by the vocabulary at each EM step.
    """

    sequences = tuple(str(sequence) for sequence in sequences if sequence)
    if not sequences:
        raise ValueError("at least one non-empty sequence is required")
    max_unit_length = max(1, int(max_unit_length))
    iterations = max(1, int(iterations))
    smoothing = max(0.0, float(smoothing))
    length_penalty_bits = max(0.0, float(length_penalty_bits))
    vocabulary = candidate_vocabulary(
        sequences,
        max_unit_length=max_unit_length,
        min_count=min_count,
        max_vocab=max_vocab,
    )
    if not vocabulary:
        raise ValueError("candidate vocabulary is empty")
    log_probabilities = _initial_log_probabilities(sequences, vocabulary)

    last_objective: float | None = None
    iterations_run = 0
    partition_bits = float("nan")
    for iteration in range(iterations):
        expected: Counter[str] = Counter()
        total_log_partition = 0.0
        for sequence in sequences:
            counts, log_partition = _forward_backward_expected_counts(
                sequence,
                log_probabilities,
                max_unit_length=max_unit_length,
                length_penalty_bits=length_penalty_bits,
            )
            expected.update(counts)
            total_log_partition += log_partition

        total = sum(expected.values()) + smoothing * len(vocabulary)
        if total <= 0:
            raise RuntimeError("EM produced zero expected unit mass")
        log_probabilities = {
            unit: math.log((expected[unit] + smoothing) / total)
            for unit in vocabulary
        }
        iterations_run = iteration + 1
        partition_bits = -total_log_partition / _LOG2
        if last_objective is not None:
            scale = max(1.0, abs(last_objective))
            if abs(partition_bits - last_objective) / scale <= convergence_tol:
                break
        last_objective = partition_bits

    return UnigramSegmentationModel(
        log_probabilities=log_probabilities,
        max_unit_length=max_unit_length,
        length_penalty_bits=length_penalty_bits,
        iterations_run=iterations_run,
        training_partition_bits=partition_bits,
        vocabulary_size=len(vocabulary),
    )


def viterbi_segment(
    sequence: str, model: UnigramSegmentationModel
) -> tuple[str, ...]:
    """Return the highest-scoring latent unit sequence under a fitted model."""

    sequence = str(sequence)
    if not sequence:
        return ()
    vocabulary = set(model.log_probabilities)
    n = len(sequence)
    best = [_NEG_INF] * (n + 1)
    previous: list[tuple[int, str] | None] = [None] * (n + 1)
    best[0] = 0.0
    epsilon = 1e-12
    for start in range(n):
        if best[start] == _NEG_INF:
            continue
        for stop, unit in _matching_units(
            sequence, start, vocabulary, model.max_unit_length
        ):
            candidate = best[start] + model.unit_score(unit)
            old = best[stop]
            old_unit = previous[stop][1] if previous[stop] is not None else ""
            if (
                candidate > old + epsilon
                or (
                    abs(candidate - old) <= epsilon
                    and (len(unit), unit) > (len(old_unit), old_unit)
                )
            ):
                best[stop] = candidate
                previous[stop] = (start, unit)
    if previous[n] is None:
        raise RuntimeError("Viterbi decoder could not cover the sequence")
    units: list[str] = []
    cursor = n
    while cursor > 0:
        step = previous[cursor]
        if step is None:
            raise RuntimeError("broken Viterbi backpointer")
        start, unit = step
        units.append(unit)
        cursor = start
    units.reverse()
    return tuple(units)


def _merge_units(
    units: Sequence[str], pair: tuple[str, str]
) -> tuple[str, ...]:
    merged = pair[0] + pair[1]
    out: list[str] = []
    index = 0
    while index < len(units):
        if (
            index + 1 < len(units)
            and units[index] == pair[0]
            and units[index + 1] == pair[1]
        ):
            out.append(merged)
            index += 2
        else:
            out.append(units[index])
            index += 1
    return tuple(out)


def learn_line_bpe_rules(
    sequences: Sequence[str], merges: int = 32
) -> tuple[tuple[str, str], ...]:
    """Learn BPE while respecting line boundaries and ignoring whitespace.

    This is an exploratory comparator, not the paper-exact within-token BPE in
    ``segmentation.py``. Its only purpose is to provide an independent boundary
    set on the *same* line-bounded, space-erased sequences used by the unigram
    model.
    """

    segmentations = [tuple(sequence) for sequence in sequences if sequence]
    rules: list[tuple[str, str]] = []
    for _ in range(max(0, int(merges))):
        counts: Counter[tuple[str, str]] = Counter()
        for units in segmentations:
            counts.update(zip(units, units[1:]))
        if not counts:
            break
        pair, count = max(counts.items(), key=lambda item: (item[1], item[0]))
        if count < 2:
            break
        rules.append(pair)
        segmentations = [_merge_units(units, pair) for units in segmentations]
    return tuple(rules)


def apply_line_bpe_rules(
    sequence: str, rules: Sequence[tuple[str, str]]
) -> tuple[str, ...]:
    units: tuple[str, ...] = tuple(sequence)
    for pair in rules:
        units = _merge_units(units, pair)
    return units


def unit_boundaries(units: Sequence[str]) -> frozenset[int]:
    boundaries: set[int] = set()
    cursor = 0
    for unit in units[:-1]:
        cursor += len(unit)
        boundaries.add(cursor)
    return frozenset(boundaries)


def _segment_from_boundaries(sequence: str, boundaries: Iterable[int]) -> tuple[str, ...]:
    cuts = sorted({int(pos) for pos in boundaries if 0 < int(pos) < len(sequence)})
    units: list[str] = []
    start = 0
    for stop in cuts + [len(sequence)]:
        units.append(sequence[start:stop])
        start = stop
    return tuple(units)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def segmentation_consensus(
    text: str,
    *,
    bpe_merges: int = 32,
    max_unit_length: int = 8,
    min_count: int = 2,
    max_vocab: int = 1024,
    iterations: int = 12,
    length_penalty_bits: float = 0.0,
    preview_lines: int = 12,
) -> SegmentationConsensusResult:
    """Compare BPE and unigram boundaries without treating either as truth."""

    observations = line_observations(text)
    if not observations:
        raise ValueError("no non-empty transcription lines available")
    sequences = [observation.glyphs for observation in observations]
    model = fit_unigram_segmenter(
        sequences,
        max_unit_length=max_unit_length,
        min_count=min_count,
        max_vocab=max_vocab,
        iterations=iterations,
        length_penalty_bits=length_penalty_bits,
    )
    bpe_rules = learn_line_bpe_rules(sequences, merges=bpe_merges)

    total_positions = 0
    shared = 0
    union = 0
    bpe_boundary_count = 0
    unigram_boundary_count = 0
    whitespace_count = 0
    bpe_at_whitespace = 0
    unigram_at_whitespace = 0
    shared_at_whitespace = 0
    whitespace_recovered_bpe = 0
    whitespace_recovered_unigram = 0
    whitespace_recovered_shared = 0
    agreement_positions = 0
    line_rows: list[dict[str, object]] = []
    unit_counts: dict[str, Counter[str]] = {
        "unigram": Counter(),
        "bpe": Counter(),
        "consensus_intersection": Counter(),
    }
    preview_parts: list[str] = []

    for line_number, observation in enumerate(observations, start=1):
        sequence = observation.glyphs
        unigram_units = viterbi_segment(sequence, model)
        bpe_units = apply_line_bpe_rules(sequence, bpe_rules)
        unigram_boundaries = unit_boundaries(unigram_units)
        bpe_boundaries = unit_boundaries(bpe_units)
        shared_boundaries = unigram_boundaries & bpe_boundaries
        union_boundaries = unigram_boundaries | bpe_boundaries
        whitespace = observation.whitespace_boundaries
        consensus_units = _segment_from_boundaries(sequence, shared_boundaries)

        internal_positions = max(0, len(sequence) - 1)
        total_positions += internal_positions
        shared += len(shared_boundaries)
        union += len(union_boundaries)
        bpe_boundary_count += len(bpe_boundaries)
        unigram_boundary_count += len(unigram_boundaries)
        whitespace_count += len(whitespace)
        bpe_at_whitespace += len(bpe_boundaries & whitespace)
        unigram_at_whitespace += len(unigram_boundaries & whitespace)
        shared_at_whitespace += len(shared_boundaries & whitespace)
        whitespace_recovered_bpe += len(whitespace & bpe_boundaries)
        whitespace_recovered_unigram += len(whitespace & unigram_boundaries)
        whitespace_recovered_shared += len(whitespace & shared_boundaries)
        agreement_positions += internal_positions - len(union_boundaries) + len(shared_boundaries)

        unit_counts["unigram"].update(unigram_units)
        unit_counts["bpe"].update(bpe_units)
        unit_counts["consensus_intersection"].update(consensus_units)
        line_rows.append(
            {
                "line": line_number,
                "glyphs": len(sequence),
                "transcription_space_boundaries": len(whitespace),
                "unigram_units": len(unigram_units),
                "bpe_units": len(bpe_units),
                "shared_boundaries": len(shared_boundaries),
                "boundary_union": len(union_boundaries),
                "boundary_jaccard": _safe_ratio(
                    len(shared_boundaries), len(union_boundaries)
                ),
                "unigram_space_precision": _safe_ratio(
                    len(unigram_boundaries & whitespace), len(unigram_boundaries)
                ),
                "bpe_space_precision": _safe_ratio(
                    len(bpe_boundaries & whitespace), len(bpe_boundaries)
                ),
            }
        )
        if len(preview_parts) < max(0, int(preview_lines)):
            preview_parts.append(
                "\n".join(
                    (
                        f"L{line_number} glyphs:    {sequence}",
                        f"   unigram: {'·'.join(unigram_units)}",
                        f"   BPE:     {'·'.join(bpe_units)}",
                        f"   shared:  {'·'.join(consensus_units)}",
                    )
                )
            )

    summary: dict[str, float | int] = {
        "lines": len(observations),
        "glyphs": sum(len(observation.glyphs) for observation in observations),
        "possible_internal_boundaries": total_positions,
        "unigram_vocabulary_size": model.vocabulary_size,
        "unigram_iterations": model.iterations_run,
        "unigram_training_partition_bits": model.training_partition_bits,
        "bpe_rules_learned": len(bpe_rules),
        "unigram_boundaries": unigram_boundary_count,
        "bpe_boundaries": bpe_boundary_count,
        "shared_boundaries": shared,
        "boundary_union": union,
        "boundary_jaccard": _safe_ratio(shared, union),
        "all_position_agreement": _safe_ratio(agreement_positions, total_positions),
        "fraction_unigram_boundaries_shared": _safe_ratio(shared, unigram_boundary_count),
        "fraction_bpe_boundaries_shared": _safe_ratio(shared, bpe_boundary_count),
        "transcription_space_boundaries": whitespace_count,
        "fraction_spaces_recovered_unigram": _safe_ratio(
            whitespace_recovered_unigram, whitespace_count
        ),
        "fraction_spaces_recovered_bpe": _safe_ratio(
            whitespace_recovered_bpe, whitespace_count
        ),
        "fraction_spaces_recovered_by_shared_boundaries": _safe_ratio(
            whitespace_recovered_shared, whitespace_count
        ),
        "fraction_unigram_boundaries_at_spaces": _safe_ratio(
            unigram_at_whitespace, unigram_boundary_count
        ),
        "fraction_bpe_boundaries_at_spaces": _safe_ratio(
            bpe_at_whitespace, bpe_boundary_count
        ),
        "fraction_shared_boundaries_at_spaces": _safe_ratio(
            shared_at_whitespace, shared
        ),
    }

    unit_rows: list[dict[str, object]] = []
    for method, counts in unit_counts.items():
        total = sum(counts.values())
        for rank, (unit, count) in enumerate(counts.most_common(50), start=1):
            unit_rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "unit": unit,
                    "glyph_length": len(unit),
                    "count": count,
                    "frequency": count / total if total else float("nan"),
                }
            )

    return SegmentationConsensusResult(
        summary=summary,
        line_rows=tuple(line_rows),
        unit_rows=tuple(unit_rows),
        preview="\n\n".join(preview_parts),
        model=model,
        bpe_rules=bpe_rules,
    )
