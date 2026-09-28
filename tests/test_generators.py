from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.generators import (
    NAIBBE_FAMILY,
    NAIBBE_UPSTREAM_COMMIT,
    NaibbeConfig,
    clean_naibbe_line,
    generate_naibbe,
    load_naibbe_glyph_map,
)


def test_naibbe_defaults_pin_paper_facing_baseline():
    cfg = NaibbeConfig()
    assert cfg.respacing == 17
    assert cfg.use_78_card_deck is False
    assert cfg.space_removal_rate == 0.03
    assert cfg.unambiguous is True
    assert NAIBBE_FAMILY == "meaningful_cipher"
    assert NAIBBE_UPSTREAM_COMMIT == "f2675ec5dd275268bc64dd48ea64fc0e0e9827a2"


def test_naibbe_mapping_asset_is_complete_and_pinned_at_sentinels():
    mapping = load_naibbe_glyph_map()
    assert len(mapping) == 414
    assert mapping["unigram_alpha_a"] == "ol"
    assert mapping["suffix_gamma2_z"] == "alsy"


def test_clean_naibbe_line_matches_latin_normalization_rules():
    assert clean_naibbe_line("Wâ J K ß Ø, æther!") == "uuaicsso aether".replace(" ", "")
    assert clean_naibbe_line("Quædam—vōx 123") == "quaedamvox"


def test_naibbe_is_deterministic_for_a_seed_and_changes_across_seeds():
    text = "In principio erat Verbum et Verbum erat apud Deum."
    first = generate_naibbe(text, seed=2025)
    second = generate_naibbe(text, seed=2025)
    other = generate_naibbe(text, seed=2026)
    assert first == second
    assert first.ciphertext
    assert first.canonical_ciphertext
    assert first.respaced_plaintext
    assert first.ciphertext != other.ciphertext


def test_naibbe_private_rng_does_not_mutate_global_random_state():
    random.seed(123456)
    before = random.getstate()
    generate_naibbe("Arma virumque cano Troiae qui primus ab oris", seed=7)
    after = random.getstate()
    assert before == after


def test_naibbe_preserves_blank_and_terminal_lines():
    result = generate_naibbe("Lorem ipsum\n\nDolor sit amet\n", seed=11)
    final_lines = result.ciphertext.split("\n")
    canonical_lines = result.canonical_ciphertext.split("\n")
    respaced_lines = result.respaced_plaintext.split("\n")
    assert len(final_lines) == len(canonical_lines) == len(respaced_lines) == 4
    assert final_lines[1] == canonical_lines[1] == respaced_lines[1] == ""
    assert final_lines[-1] == canonical_lines[-1] == respaced_lines[-1] == ""
    assert result.ciphertext.endswith("\n")


def test_naibbe_provenance_marks_plaintext_cipher_family():
    result = generate_naibbe("Vita brevis ars longa", seed=3)
    provenance = result.provenance
    assert provenance["family"] == "meaningful_cipher"
    assert provenance["input_kind"] == "meaningful_plaintext"
    assert provenance["upstream_commit"] == NAIBBE_UPSTREAM_COMMIT
    assert provenance["config"]["use_78_card_deck"] is False
