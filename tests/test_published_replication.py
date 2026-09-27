from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.published_replication import (
    automatic_replication_metrics,
    collapse_composites,
    entropy_corpus_tokens,
    paper_character_conditional_entropy,
    paper_order_metrics,
    strict_space_records,
    unit_scale_lines_by_quire,
    unit_scale_text_by_quire,
)

RAW = """#=IVTFF Eva- 2.0
<f1r> <! $Q=A $P=A $I=H $L=A $H=1 >
<f1r.1,@P0;U> cth.ee,shol
<f1r.2,@L0;U> label.shouldnot
<f1r.3,@P0;U> a'b.cd
<f1r.4,@P0;U> bad?.ok
<f1r.5,@P0;U> qok[e:a]dy.daiin
<f2r> <! $Q=B $P=A $I=T $L=B $H=2 >
<f2r.1,@P0;U> chedy.qokeedy.daiin
<f2r.2,@P0;U> shol,shory.qokedy
"""


def test_composite_collapse_matches_public_substitution_order():
    assert collapse_composites("cthcheeiin") == "TCEN"


def test_entropy_corpus_is_p_only_but_keeps_apostrophe_path():
    tokens = entropy_corpus_tokens(RAW)
    assert "label" not in tokens and "shouldnot" not in tokens
    assert "a'b" in tokens
    assert "bad?" not in tokens and "ok" in tokens
    assert "qokedy" in tokens


def test_unit_scale_cleaner_uses_p_loci_and_skips_bad_pieces_individually():
    grouped = unit_scale_lines_by_quire(RAW)
    assert set(grouped) == {"A", "B"}
    flat_a = [token for line in grouped["A"] for token in line]
    assert "label" not in flat_a and "shouldnot" not in flat_a
    assert "ab" in flat_a and "a'b" not in flat_a
    assert "bad?" not in flat_a and "ok" in flat_a
    assert "qokedy" in flat_a


def test_strict_space_cleaner_rejects_whole_line_when_a_piece_is_invalid():
    records = strict_space_records(RAW)
    token_lines = [record["tokens"] for record in records]
    assert ["bad?", "ok"] not in token_lines
    assert not any("ok" in line and len(line) == 1 for line in token_lines)
    assert any(line == ["ab", "cd"] for line in token_lines)
    assert all(len(record["tokens"]) >= 2 for record in records)


def test_unit_scale_text_preserves_manuscript_lines_inside_each_quire():
    rendered = unit_scale_text_by_quire(RAW)
    assert set(rendered) == {"A", "B"}
    assert "label" not in rendered["A"]
    assert "\n" in rendered["A"]


def test_paper_entropy_and_order_metrics_are_finite_and_deterministic():
    h2 = paper_character_conditional_entropy(RAW)
    assert h2 == h2 and h2 >= 0
    first = paper_order_metrics(RAW, shuffles=5, seed=17)
    second = paper_order_metrics(RAW, shuffles=5, seed=17)
    assert first == second
    assert first["cross_boundary_edge_mi_bits"] == first["cross_boundary_edge_mi_bits"]
    assert first["token_succession_entropy_fraction"] == first["token_succession_entropy_fraction"]
    assert first["order_lines"] == len(strict_space_records(RAW))


def test_automatic_replication_emits_only_unambiguous_targets():
    observed, diagnostics = automatic_replication_metrics(
        RAW,
        shuffles=5,
        order_seed=23,
        bpe_checkpoints=(0, 2),
    )
    assert "char_conditional_entropy_bits" in observed
    assert "cross_boundary_edge_mi_bits" in observed
    assert "token_succession_entropy_fraction" in observed
    assert "bpe_crossfit_gap_0_bits" in observed
    assert "bpe_crossfit_gap_2_bits" in observed
    assert "bpe_crossfit_selected_merges" in observed
    assert "certain_separator_unit_crossing_rate" not in observed
    assert diagnostics["unit_scale_quires"] == ["A", "B"]
