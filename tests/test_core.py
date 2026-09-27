from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.discriminator import classifier_two_sample_test
from voynich_lab.ivtff import is_ivtff, parse_ivtff
from voynich_lab.longrange import block_entropy, dfa_fluctuation, lagged_mutual_information, lagged_mi_with_shuffle_baseline
from voynich_lab.metrics import compute_metrics, glyph_stream, normalize_text
from voynich_lab.replication_targets import TARGETS, compare_to_targets
from voynich_lab.scorecard import run_scorecard
from voynich_lab.segmentation import (
    apply_bpe_merges,
    apply_token_bpe_rules,
    cross_fit_bpe_by_quire,
    cross_fit_bpe_scale_curve,
    discover_bpe_units,
    learn_bpe_merges,
    learn_token_bpe_rules,
    unit_stream_stats,
)
from voynich_lab.surrogates import block_shuffle_surrogate, iid_glyph_surrogate, markov1_surrogate, markov_k_surrogate, position_conditioned_markov_surrogate

SAMPLE = "qokeedy qokedy\nchedy qokeedy"
LONG_SAMPLE = " ".join(["qokeedy", "qokedy", "chedy", "shedy", "daiin", "otedy"] * 50)
IVTFF_SNIPPET = """#=IVTFF Eva- 2.0
<f1r> <! $Q=A $P=A $I=H $L=A $H=1 >
<f1r.1,@P0;U> fachys.ykal.ar,ataiin.shol
<f1r.2,@P0;U> shory.cth[e:o]res.y.k[a:o]l
<f1r.3,@P0;U> qokedy.qokeedy
<f2r> <! $Q=B $P=A $I=T $L=B $H=2 >
<f2r.1,@P0;U> chedy.qokeedy.daiin
"""


def _whitespace_mask(text: str):
    return [ch if ch.isspace() else "X" for ch in normalize_text(text)]


def test_metrics_basic():
    m = compute_metrics(SAMPLE)
    assert m["glyph_count"] > 0
    assert m["glyph_types"] > 1
    assert m["conventional_token_count"] == 4


def test_surrogate_lengths_and_layout():
    raw_len = len(glyph_stream(SAMPLE))
    for fn in (iid_glyph_surrogate, markov1_surrogate):
        out = fn(SAMPLE, seed=1)
        assert len(glyph_stream(out)) == raw_len
        assert _whitespace_mask(out) == _whitespace_mask(SAMPLE)


def test_bpe_reduces_or_preserves_length():
    units, history = discover_bpe_units(SAMPLE, 8)
    assert len(units) <= len(glyph_stream(SAMPLE))
    assert len(history) <= 8


def test_markov_k_surrogate_matches_length_and_is_deterministic():
    a = markov_k_surrogate(SAMPLE, order=5, seed=7)
    b = markov_k_surrogate(SAMPLE, order=5, seed=7)
    assert a == b
    assert len(glyph_stream(a)) == len(glyph_stream(SAMPLE))
    assert _whitespace_mask(a) == _whitespace_mask(SAMPLE)


def test_block_shuffle_preserves_multiset_and_layout():
    out = block_shuffle_surrogate(LONG_SAMPLE, block_size=3, seed=2)
    assert sorted(glyph_stream(out)) == sorted(glyph_stream(LONG_SAMPLE))
    assert _whitespace_mask(out) == _whitespace_mask(LONG_SAMPLE)


def test_position_conditioned_markov_runs_and_preserves_layout():
    out = position_conditioned_markov_surrogate(LONG_SAMPLE, n_bins=5, seed=3)
    assert len(out) > 0
    assert _whitespace_mask(out) == _whitespace_mask(LONG_SAMPLE)


def test_ivtff_parses_locus_and_folio():
    doc = parse_ivtff(IVTFF_SNIPPET)
    assert is_ivtff(IVTFF_SNIPPET)
    assert len(doc.lines) == 4
    assert doc.lines[0].folio == "f1r"
    assert doc.lines[-1].folio == "f2r"


def test_ivtff_page_metadata_propagates_to_loci():
    doc = parse_ivtff(IVTFF_SNIPPET)
    assert doc.quires_present() == ["A", "B"]
    assert doc.lines[0].quire == "A"
    assert doc.lines[0].currier_language == "A"
    assert doc.lines[-1].quire == "B"
    assert doc.lines[-1].hand == "2"
    assert doc.lines[-1].illustration_type == "T"


def test_ivtff_separator_certainty():
    doc = parse_ivtff(IVTFF_SNIPPET)
    assert doc.lines[0].separators == ["certain", "certain", "uncertain", "certain"]


def test_ivtff_alternates_are_token_local_and_primary_resolves():
    doc = parse_ivtff(IVTFF_SNIPPET)
    row = doc.lines[1].tokens
    flags = {t.text: t.had_alternates for t in row}
    assert flags["ctheres"] is True and flags["kal"] is True
    assert flags["shory"] is False and flags["y"] is False


def test_ivtff_analysis_stream_excludes_markup():
    clean = normalize_text(IVTFF_SNIPPET)
    assert "$Q" not in clean and "<f1r" not in clean and "@P0" not in clean
    assert "fachys ykal ar ataiin shol" in clean
    assert compute_metrics(IVTFF_SNIPPET)["glyph_count"] == len(glyph_stream(clean))


def test_ivtff_uncertain_merge_policy_changes_boundary_only():
    doc = parse_ivtff(IVTFF_SNIPPET)
    split = doc.to_analysis_text("split")
    merged = doc.to_analysis_text("merge")
    assert "ar ataiin" in split
    assert "arataiin" in merged
    assert glyph_stream(split) == glyph_stream(merged)


def test_lagged_mutual_information_is_finite_and_nonnegative():
    mi = lagged_mutual_information(glyph_stream(LONG_SAMPLE), max_lag=5)
    assert len(mi) == 5
    for v in mi.values():
        assert v == v and v >= -1e-9


def test_lagged_mi_reports_shuffle_bias_baseline():
    rows = lagged_mi_with_shuffle_baseline(LONG_SAMPLE, max_lag=4, n_permutations=5, seed=1)
    assert len(rows) == 4
    assert all("excess_mi_bits" in r and "shuffle_mean_bits" in r for r in rows)


def test_block_entropy_keys():
    be = block_entropy(glyph_stream(LONG_SAMPLE), block_sizes=(1, 2, 3))
    assert set(be) == {1, 2, 3}


def test_dfa_fluctuation_returns_alpha():
    fluct, alpha = dfa_fluctuation(LONG_SAMPLE, box_sizes=(4, 8, 16))
    assert isinstance(fluct, dict)
    assert isinstance(alpha, float)


def test_learn_and_apply_bpe_merges_no_leakage():
    merge_table = learn_bpe_merges(LONG_SAMPLE, merges=10)
    held_out = "qokeedy chedy qokedy"
    segmented = apply_bpe_merges(held_out, merge_table)
    assert len(segmented) <= len(glyph_stream(held_out))


def test_token_bpe_never_learns_across_conventional_token_boundary():
    lines = [["ab", "cd"] for _ in range(10)]
    rules = learn_token_bpe_rules(lines, max_merges=4)
    assert all((left, right) != ("b", "c") for left, right, _merged in rules)
    segmented = apply_token_bpe_rules([["ab", "cd"]], rules, merge_count=4)
    assert set(segmented) == {"ab", "cd"}


def test_token_bpe_pair_counts_are_weighted_by_token_frequency():
    lines = [["ab"], ["ab"], ["ab"], ["ac"]]
    rules = learn_token_bpe_rules(lines, max_merges=1)
    assert rules[0][:2] == ("a", "b")


def test_unit_stream_stats_cross_tokens_within_line_but_not_lines():
    lines = [["a", "b"], ["a", "c"]]
    segmentation = {word: tuple(word) for line in lines for word in line}
    stats = unit_stream_stats(lines, segmentation)
    assert abs(stats["H1"] - 1.5) < 1e-12
    assert abs(stats["H2"] - 1.0) < 1e-12
    assert abs(stats["gap"] - 0.5) < 1e-12


def test_cross_fit_bpe_by_quire_runs_per_quire():
    text_by_quire = {"Q1": LONG_SAMPLE, "Q2": ("chedy qokeedy daiin shedy " * 20)}
    results = cross_fit_bpe_by_quire(text_by_quire, merges=8)
    assert set(results) == {"Q1", "Q2"}
    assert all(r["held_out_unit_count"] > 0 for r in results.values())
    assert all("dependence_gap_bits" in r for r in results.values())
    assert all("H1_bits" in r and "bits_per_glyph" in r for r in results.values())


def test_crossfit_bpe_scale_curve_uses_quire_folds():
    doc = parse_ivtff(IVTFF_SNIPPET)
    rows, selected = cross_fit_bpe_scale_curve(doc.text_by_quire(), checkpoints=(0, 2, 4))
    assert [r["merges"] for r in rows] == [0, 2, 4]
    assert selected in {0, 2, 4}
    assert all(r["folds"] == 2 for r in rows)
    assert all("H1_bits" in r and "H2_conditional_bits" in r for r in rows)
    assert all("bits_per_glyph" in r and "mean_unit_length" in r for r in rows)


def test_scorecard_has_global_and_adjusted_p_values():
    result = run_scorecard(LONG_SAMPLE, iid_glyph_surrogate, n_replicates=20, seed=0)
    assert result.d_infinity == result.d_infinity
    assert result.worst_feature
    assert 0 < result.global_p_value <= 1
    assert set(result.feature_p_holm) == set(result.observed)
    assert all((p != p) or (0 <= p <= 1) for p in result.feature_p_holm.values())


def test_classifier_uses_multiple_grouped_splits():
    result = classifier_two_sample_test(LONG_SAMPLE, iid_glyph_surrogate, block_size=20, n_generated_replicates=3, seed=0, n_splits=5)
    assert result.n_splits >= 3
    assert 0 <= result.held_out_auc <= 1
    assert result.auc_ci_low <= result.held_out_auc <= result.auc_ci_high
    assert "group-aware" in result.note.lower()


def test_replication_targets_are_verified_and_corrected():
    assert TARGETS["char_conditional_entropy_bits"].value == 2.69
    assert TARGETS["cross_boundary_edge_mi_bits"].value == 0.197
    assert TARGETS["certain_separator_unit_crossing_rate"].value == 0.025
    assert "uncertain_separator_crossing_rate" not in TARGETS
    rows = compare_to_targets({"bpe_crossfit_selected_merges": 32, "token_succession_entropy_fraction": 0.009})
    assert all(r["reproduced"] is True for r in rows)


def test_replication_mismatch_is_explicit():
    rows = compare_to_targets({"end_to_start_flow_proportion": 0.1}, tolerance=0.1)
    assert rows[0]["reproduced"] is False
    assert rows[0]["mismatch_reason"] == "UNRESOLVED"
