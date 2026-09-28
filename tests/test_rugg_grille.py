from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.discriminator import FEATURE_NAMES, _block_features, classifier_two_sample_test
from voynich_lab.grille import RuggGrilleConfig, generate_rugg_grille, generate_rugg_grille_text
from voynich_lab.metrics import conventional_tokens

SAMPLE = "\n".join(
    [
        "qokeedy qokedy chedy shedy daiin otedy",
        "chedy qokeedy daiin shedy qokedy qokeedy",
        "daiin chedy shedy qokedy otedy qokeedy",
    ]
    * 30
)


def test_rugg_generator_is_seed_deterministic_and_contract_compatible():
    a = generate_rugg_grille_text(SAMPLE, seed=7)
    b = generate_rugg_grille_text(SAMPLE, seed=7)
    c = generate_rugg_grille_text(SAMPLE, seed=8)
    assert a == b
    assert a != c


def test_rugg_generator_matches_line_and_token_counts_not_exact_layout():
    result = generate_rugg_grille(SAMPLE, seed=3)
    assert len(result.text.splitlines()) == len(SAMPLE.splitlines())
    assert len(conventional_tokens(result.text)) == len(conventional_tokens(SAMPLE))
    assert result.inventory.bpe_rules >= 0
    assert result.provenance["family"] == "table_grille"
    assert "exact reconstruction" in result.provenance["method_note"]


def test_rugg_config_rejects_impossible_grille_count():
    config = RuggGrilleConfig(max_grille_offset=0, n_grilles=2)
    try:
        generate_rugg_grille(SAMPLE, seed=1, config=config)
    except ValueError as exc:
        assert "n_grilles" in str(exc)
    else:
        raise AssertionError("expected impossible grille count to be rejected")


def test_discriminator_features_include_boundary_statistics():
    features = _block_features("qokeedy qokedy chedy shedy daiin otedy")
    assert len(features) == len(FEATURE_NAMES)
    assert "cross_token_edge_mi" in FEATURE_NAMES
    assert "mean_token_length" in FEATURE_NAMES
    assert all(value == value for value in features)


def test_rugg_generator_runs_through_layer2_contract():
    result = classifier_two_sample_test(
        SAMPLE,
        generate_rugg_grille_text,
        block_size=40,
        n_generated_replicates=2,
        seed=4,
        n_splits=3,
    )
    assert result.n_splits >= 3
    assert 0 <= result.held_out_auc <= 1
    assert result.feature_names == FEATURE_NAMES
