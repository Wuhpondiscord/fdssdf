from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.unigram_segmentation import (
    apply_line_bpe_rules,
    candidate_vocabulary,
    fit_unigram_segmenter,
    learn_line_bpe_rules,
    line_observations,
    segmentation_consensus,
    unit_boundaries,
    viterbi_segment,
)


def test_line_observations_erase_spaces_but_preserve_positions():
    observations = line_observations("ab cd e\nxy z\n")
    assert [obs.glyphs for obs in observations] == ["abcde", "xyz"]
    assert observations[0].whitespace_boundaries == frozenset({2, 4})
    assert observations[1].whitespace_boundaries == frozenset({2})


def test_candidate_vocabulary_always_keeps_observed_single_glyphs():
    vocab = candidate_vocabulary(
        ["abcabc"], max_unit_length=3, min_count=2, max_vocab=2
    )
    assert {"a", "b", "c"}.issubset(vocab)


def test_unigram_fit_is_deterministic_and_reconstructs_sequences():
    sequences = ["abababab", "ababab", "babababa"]
    first = fit_unigram_segmenter(
        sequences, max_unit_length=4, min_count=2, max_vocab=64, iterations=8
    )
    second = fit_unigram_segmenter(
        sequences, max_unit_length=4, min_count=2, max_vocab=64, iterations=8
    )
    assert first.log_probabilities == second.log_probabilities
    for sequence in sequences:
        units = viterbi_segment(sequence, first)
        assert "".join(units) == sequence
        assert all(unit in first.log_probabilities for unit in units)
    assert any(
        len(unit) > 1
        for sequence in sequences
        for unit in viterbi_segment(sequence, first)
    )


def test_line_bpe_never_creates_cross_line_units():
    # If lines were concatenated, the boundary pair 'b'+'c' would be eligible.
    sequences = ["ab", "cd", "ab", "cd"]
    rules = learn_line_bpe_rules(sequences, merges=4)
    assert ("b", "c") not in rules
    for sequence in sequences:
        units = apply_line_bpe_rules(sequence, rules)
        assert "".join(units) == sequence


def test_unit_boundaries_use_glyph_offsets_not_token_indices():
    assert unit_boundaries(("ab", "c", "def")) == frozenset({2, 3})


def test_segmentation_consensus_reports_method_and_space_agreement():
    text = "ab ab ab\nab ab ab\nxy xy xy\nxy xy xy\n"
    first = segmentation_consensus(
        text,
        bpe_merges=8,
        max_unit_length=4,
        max_vocab=64,
        iterations=8,
        preview_lines=2,
    )
    second = segmentation_consensus(
        text,
        bpe_merges=8,
        max_unit_length=4,
        max_vocab=64,
        iterations=8,
        preview_lines=2,
    )

    assert first.summary == second.summary
    assert first.line_rows == second.line_rows
    assert first.unit_rows == second.unit_rows
    assert first.preview == second.preview
    assert first.summary["lines"] == 4
    assert first.summary["transcription_space_boundaries"] == 8
    assert 0.0 <= first.summary["boundary_jaccard"] <= 1.0
    assert 0.0 <= first.summary["all_position_agreement"] <= 1.0
    for key in (
        "fraction_spaces_recovered_unigram",
        "fraction_spaces_recovered_bpe",
        "fraction_spaces_recovered_by_shared_boundaries",
    ):
        value = first.summary[key]
        assert math.isnan(value) or 0.0 <= value <= 1.0
    assert "unigram:" in first.preview
    assert "BPE:" in first.preview
    assert "shared:" in first.preview


def test_consensus_handles_input_without_transcription_spaces():
    result = segmentation_consensus(
        "abababab\nabababab\n",
        bpe_merges=4,
        max_unit_length=4,
        max_vocab=32,
        iterations=5,
    )
    assert result.summary["transcription_space_boundaries"] == 0
    assert math.isnan(result.summary["fraction_spaces_recovered_unigram"])
    assert math.isnan(result.summary["fraction_spaces_recovered_bpe"])
