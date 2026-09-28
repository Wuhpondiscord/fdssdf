"""Regression tests for the generator contract, boundary-aware null, stratification, and scorecard resolution.

Each test pins a specific defect or claim found while evaluating the repository on the real ZL3b corpus:

* ``partial(markov_k_surrogate, order=3)(text, seed)`` crashed (seed collided with ``order``) and
  ``position_conditioned_markov_surrogate(text, seed)`` silently took the seed as ``n_bins``.
* All space-free Markov nulls paste original spaces back, so none could be a fair baseline for
  boundary-edge structure.
* Global nulls are stationary by construction, so a block classifier separates them from a
  heterogeneous manuscript for reasons unrelated to local structure.
* Holm-corrected Monte-Carlo p-values cannot resolve small p at low replicate counts.
"""

from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.ivtff import parse_ivtff
from voynich_lab.metrics import compute_metrics
from voynich_lab.scorecard import run_scorecard, scorecard_to_rows
from voynich_lab.stratified import heterogeneity_control, line_strata, stratified_generator
from voynich_lab.surrogates import (
    NULL_LADDER as DEFAULT_GENERATORS,
    bind_generator,
    block_shuffle_surrogate,
    boundary_markov_surrogate,
    iid_glyph_surrogate,
    markov_k_surrogate,
    position_conditioned_markov_surrogate,
    token_shuffle_surrogate,
)


def _corpus(n_lines: int = 60, alphabet: str = "abcdehkloqrsty", seed: int = 0) -> str:
    rng = random.Random(seed)
    vocab = ["".join(rng.choice(alphabet) for _ in range(rng.randint(2, 6))) for _ in range(40)]
    return "\n".join(" ".join(rng.choice(vocab) for _ in range(rng.randint(3, 8))) for _ in range(n_lines))


TEXT = _corpus()


def _planted_edge_corpus(n_lines: int = 400, seed: int = 1) -> str:
    """The last glyph of a token (a/b) deterministically selects the first glyph (x/y) of the next."""
    rng = random.Random(seed)
    lines = []
    for _ in range(n_lines):
        toks, first = [], rng.choice("xy")
        for _ in range(6):
            last = rng.choice("ab")
            toks.append(first + "".join(rng.choice("kol") for _ in range(rng.randint(1, 3))) + last)
            first = "x" if last == "a" else "y"
        lines.append(" ".join(toks))
    return "\n".join(lines)


# ---- generator contract ------------------------------------------------------


@pytest.mark.parametrize("name", list(DEFAULT_GENERATORS))
def test_default_generators_honour_text_seed_contract(name):
    fn = DEFAULT_GENERATORS[name]
    a, b, c = fn(TEXT, 1), fn(TEXT, 1), fn(TEXT, 2)
    assert a == b, "same seed must be reproducible"
    assert a != c, "seed must actually change the output (it must not be consumed as a hyperparameter)"


@pytest.mark.parametrize("name", list(DEFAULT_GENERATORS))
def test_run_scorecard_accepts_every_default_generator(name):
    result = run_scorecard(TEXT, DEFAULT_GENERATORS[name], n_replicates=3, seed=0)
    assert result.n_replicates == 3


def test_hyperparameters_are_keyword_only():
    with pytest.raises(TypeError):
        markov_k_surrogate(TEXT, 3)
    with pytest.raises(TypeError):
        block_shuffle_surrogate(TEXT, 3)
    with pytest.raises(TypeError):
        position_conditioned_markov_surrogate(TEXT, 5)
    with pytest.raises(TypeError):
        boundary_markov_surrogate(TEXT, 3)


def test_bind_generator_does_not_consume_seed_as_hyperparameter():
    bound = bind_generator(position_conditioned_markov_surrogate, n_bins=4)
    assert bound(TEXT, 5) == position_conditioned_markov_surrogate(TEXT, n_bins=4, seed=5)


# ---- boundary-aware null -------------------------------------------------------


def test_boundary_markov_generates_its_own_wellformed_boundaries():
    out = boundary_markov_surrogate(TEXT, order=3, seed=4)
    lines = out.split("\n")
    assert len(lines) == len(TEXT.split("\n"))
    assert out == boundary_markov_surrogate(TEXT, order=3, seed=4)
    assert " " in out
    assert set(out) <= set(TEXT) | {"\n"}
    for line in lines:
        assert line == line.strip() and "  " not in line


def test_boundary_markov_reproduces_planted_edge_coupling_but_token_shuffle_does_not():
    text = _planted_edge_corpus()
    observed = compute_metrics(text)["cross_token_edge_mi_bits"]
    boundary = compute_metrics(boundary_markov_surrogate(text, order=2, seed=0))["cross_token_edge_mi_bits"]
    shuffled = compute_metrics(token_shuffle_surrogate(text, seed=0))["cross_token_edge_mi_bits"]
    assert observed > 0.5
    assert boundary > 0.5 * observed
    assert shuffled < 0.25 * observed


# ---- stratification --------------------------------------------------------------


def _two_strata():
    a = _corpus(30, alphabet="abc", seed=2)
    b = _corpus(30, alphabet="xyz", seed=3)
    return a + "\n" + b, ["A"] * 30 + ["B"] * 30


def test_stratified_generator_keeps_strata_apart_and_preserves_layout():
    text, labels = _two_strata()
    out = stratified_generator(labels, token_shuffle_surrogate)(text, 7).split("\n")
    src = text.split("\n")
    assert [len(l.split()) for l in out] == [len(l.split()) for l in src]
    assert all(set(l) <= set("abc ") for l in out[:30])
    assert all(set(l) <= set("xyz ") for l in out[30:])
    assert sorted(t for l in out[:30] for t in l.split()) == sorted(t for l in src[:30] for t in l.split())


def test_stratified_generator_rejects_a_different_layout():
    text, labels = _two_strata()
    with pytest.raises(ValueError):
        stratified_generator(labels, token_shuffle_surrogate)(text + "\nextra line", 0)


def test_line_strata_reads_quire_and_currier_from_page_variables():
    raw = (
        "#=IVTFF Eva- 2.0 M 5\n"
        "<f1r>      <! $Q=A $P=A $F=a $B=1 $I=T $L=A $H=1 $C=1 $X=V>\n"
        "<f1r.1,@P0>       fachys.ykal.ar\n"
        "<f1r.2,+P0>       sory.ckhar.or\n"
        "<f26r>     <! $Q=C $P=A $F=a $B=1 $I=H $L=B $H=1 $C=1 $X=V>\n"
        "<f26r.1,@P0>      daiin.chedy.qokeedy\n"
    )
    doc = parse_ivtff(raw)
    assert line_strata(doc, "quire") == ["A", "A", "C"]
    assert line_strata(doc, "currier") == ["A", "A", "B"]
    with pytest.raises(ValueError):
        line_strata(doc, "hand-waving")


def test_heterogeneity_control_attributes_separability_to_stratum_composition():
    text, labels = _two_strata()
    result = heterogeneity_control(
        text, labels, iid_glyph_surrogate, block_size=40, n_generated_replicates=3, seed=0, n_splits=6
    )
    assert result.auc_global > 0.85
    assert result.auc_drop > 0.25
    assert result.n_strata == 2


# ---- scorecard resolution ------------------------------------------------------------


def test_scorecard_flags_holm_resolution_limit_at_low_replicates():
    r = run_scorecard(TEXT, iid_glyph_surrogate, n_replicates=10, seed=0)
    assert r.min_resolvable_p == pytest.approx(1 / 11)
    assert r.holm_can_reject_at_0_05 is False
    assert "resolution" in r.resolution_note


def test_scorecard_reports_resolvable_holm_at_high_replicates_and_outside_range():
    text = _planted_edge_corpus(n_lines=120)
    r = run_scorecard(text, token_shuffle_surrogate, n_replicates=150, seed=0)
    assert r.holm_can_reject_at_0_05 is True and r.resolution_note == ""
    assert r.outside_null_range["cross_token_edge_mi_bits"] is True
    rows = {row["feature"]: row for row in scorecard_to_rows(r)}
    assert rows["cross_token_edge_mi_bits"]["outside_null_range"] is True
