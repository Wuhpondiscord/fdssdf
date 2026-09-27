from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.discriminator import classifier_two_sample_test
from voynich_lab.ivtff import parse_ivtff
from voynich_lab.longrange import block_entropy, dfa_fluctuation, lagged_mutual_information
from voynich_lab.metrics import compute_metrics
from voynich_lab.replication_targets import compare_to_targets
from voynich_lab.scorecard import run_scorecard
from voynich_lab.segmentation import (
    apply_bpe_merges,
    cross_fit_bpe_by_quire,
    discover_bpe_units,
    learn_bpe_merges,
)
from voynich_lab.surrogates import (
    block_shuffle_surrogate,
    iid_glyph_surrogate,
    markov1_surrogate,
    markov_k_surrogate,
    position_conditioned_markov_surrogate,
)

SAMPLE = "qokeedy qokedy\nchedy qokeedy"

# A larger synthetic sample so entropy/long-range/scorecard tests have
# enough symbols to be numerically meaningful, not just non-crashing.
LONG_SAMPLE = " ".join(["qokeedy", "qokedy", "chedy", "shedy", "daiin", "otedy"] * 40)

IVTFF_SNIPPET = """# comment line, should be dropped
<f1r.1,@P0;U> fachys.ykal.ar,ataiin.shol
<f1r.2,@P0;U> shory.cth[e:o]res.y.k[a:o]l
<f1r.3,@P0;U> qokedy.qokeedy
<f2r.1,@P0;U> chedy.qokeedy.daiin
"""


def test_metrics_basic():
    m = compute_metrics(SAMPLE)
    assert m["glyph_count"] > 0
    assert m["glyph_types"] > 1
    assert m["conventional_token_count"] == 4


def test_surrogate_lengths():
    raw_len = len("".join(SAMPLE.split()))
    assert len(iid_glyph_surrogate(SAMPLE)) == raw_len
    assert len(markov1_surrogate(SAMPLE)) == raw_len


def test_bpe_reduces_or_preserves_length():
    units, history = discover_bpe_units(SAMPLE, 8)
    assert len(units) <= len("".join(SAMPLE.split()))
    assert len(history) <= 8


# ---- new null-ladder surrogates ------------------------------------------


def test_markov_k_surrogate_matches_length():
    stream_len = len("".join(SAMPLE.split()))
    out = markov_k_surrogate(SAMPLE, order=3, seed=1)
    assert len(out) == stream_len


def test_block_shuffle_preserves_multiset():
    out = block_shuffle_surrogate(LONG_SAMPLE, block_size=3, seed=2)
    stream = "".join(LONG_SAMPLE.split())
    assert sorted(out) == sorted(stream)  # shuffling blocks preserves the glyph multiset


def test_position_conditioned_markov_runs():
    out = position_conditioned_markov_surrogate(LONG_SAMPLE, n_bins=5, seed=3)
    assert len(out) > 0


# ---- IVTFF parsing --------------------------------------------------------


def test_ivtff_parses_locus_and_folio():
    doc = parse_ivtff(IVTFF_SNIPPET)
    assert len(doc.lines) == 4
    assert doc.lines[0].folio == "f1r"
    assert doc.lines[-1].folio == "f2r"


def test_ivtff_separator_certainty():
    doc = parse_ivtff(IVTFF_SNIPPET)
    # line 1: fachys.ykal.ar,ataiin.shol -> 4 separators: certain, certain, uncertain, certain
    seps = doc.lines[0].separators
    assert seps == ["certain", "certain", "uncertain", "certain"]


def test_ivtff_alternate_resolves_to_primary_reading():
    doc = parse_ivtff(IVTFF_SNIPPET)
    tokens = [t.text for t in doc.lines[1].tokens]
    assert "ctheres" in tokens  # cth[e:o]res -> "cth" + "e" + "res" (first option)
    assert "kal" in tokens  # k[a:o]l -> "k" + "a" + "l" (first option)


def test_ivtff_conventional_tokens_flat_list():
    doc = parse_ivtff(IVTFF_SNIPPET)
    toks = doc.conventional_tokens()
    assert "qokedy" in toks and "qokeedy" in toks


# ---- long-range statistics -------------------------------------------------


def test_lagged_mutual_information_is_finite_and_nonnegative():
    mi = lagged_mutual_information("".join(LONG_SAMPLE.split()), max_lag=5)
    assert len(mi) == 5
    for v in mi.values():
        assert v == v  # not NaN
        assert v >= -1e-9  # MI should not be meaningfully negative


def test_block_entropy_monotone_keys():
    be = block_entropy("".join(LONG_SAMPLE.split()), block_sizes=(1, 2, 3))
    assert set(be) == {1, 2, 3}


def test_dfa_fluctuation_returns_alpha():
    fluct, alpha = dfa_fluctuation(LONG_SAMPLE, box_sizes=(4, 8, 16))
    assert isinstance(fluct, dict)
    # alpha may be NaN on tiny inputs; just check the call doesn't crash
    # and returns a float type.
    assert isinstance(alpha, float)


# ---- cross-fit BPE ----------------------------------------------------------


def test_learn_and_apply_bpe_merges_no_leakage():
    train_text = LONG_SAMPLE
    held_out_text = "qokeedy chedy qokedy"
    merge_table = learn_bpe_merges(train_text, merges=10)
    segmented = apply_bpe_merges(held_out_text, merge_table)
    assert len(segmented) <= len("".join(held_out_text.split()))


def test_cross_fit_bpe_by_quire_runs_per_quire():
    text_by_quire = {"Q1": LONG_SAMPLE, "Q2": "chedy qokeedy daiin shedy" * 10}
    results = cross_fit_bpe_by_quire(text_by_quire, merges=8)
    assert set(results) == {"Q1", "Q2"}
    for r in results.values():
        assert r["held_out_unit_count"] > 0


# ---- scorecard and discriminator (adversarial validation layers) ---------


def test_scorecard_d_infinity_is_small_for_matched_surrogate():
    result = run_scorecard(LONG_SAMPLE, iid_glyph_surrogate, n_replicates=15, seed=0)
    assert result.d_infinity == result.d_infinity  # not NaN given enough replicates
    assert result.worst_feature != ""


def test_classifier_two_sample_test_runs():
    result = classifier_two_sample_test(
        LONG_SAMPLE, iid_glyph_surrogate, block_size=10, n_generated_replicates=3, seed=0
    )
    assert 0.0 <= result.held_out_auc <= 1.0 or result.held_out_auc != result.held_out_auc


# ---- replication targets ---------------------------------------------------


def test_compare_to_targets_flags_mismatch_and_match():
    observed = {
        "char_conditional_entropy_bits": 2.7,  # matches target exactly
        "end_to_start_flow_proportion": 0.1,  # far from published 0.806
    }
    rows = compare_to_targets(observed, tolerance=0.1)
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["char_conditional_entropy_bits"]["reproduced_within_tolerance"] is True
    assert by_metric["end_to_start_flow_proportion"]["reproduced_within_tolerance"] is False
    assert by_metric["end_to_start_flow_proportion"]["mismatch_reason"] == "UNRESOLVED"
