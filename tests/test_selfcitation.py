from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.selfcitation import (
    JavaRandom,
    SELFCITATION_FAMILY,
    SELFCITATION_UPSTREAM_COMMIT,
    SelfCitationConfig,
    canonical_selfcitation_config,
)


def test_canonical_selfcitation_config_matches_released_executable():
    cfg = canonical_selfcitation_config()
    assert cfg.lines_to_create == 1200
    assert cfg.max_line_length == 55
    assert cfg.min_line_length == 15
    assert cfg.lines_per_page == 29
    assert cfg.max_repeat_count == 3
    assert cfg.random_mode == "pseudo"
    assert cfg.random_seed == 19
    assert cfg.can_follow == "curveline"
    assert cfg.source_chooser == "page"
    assert cfg.same_position_probability == 28
    assert cfg.morph == "slim"
    assert (cfg.add_remove_probability, cfg.combine_split_probability, cfg.replace_probability) == (20, 30, 50)
    assert cfg.reuse_last_probability == 10
    assert cfg.combined_dismiss_as_source_probability == 30
    assert cfg.suggestions == "top"
    assert cfg.suggestions_probability == 40
    assert cfg.currier_type == "B"


def test_selfcitation_provenance_keeps_pseudotext_separate_from_cipher_family():
    cfg = canonical_selfcitation_config()
    provenance = cfg.provenance
    assert provenance["family"] == SELFCITATION_FAMILY == "meaningless_pseudotext"
    assert provenance["upstream_commit"] == SELFCITATION_UPSTREAM_COMMIT
    assert provenance["reference_config"] == "executable/conf.properties"


def test_java_random_is_deterministic_and_rand_zero_does_not_advance_state():
    first = JavaRandom(19)
    second = JavaRandom(19)
    before = first._seed
    assert first.rand(0) == 0
    assert first._seed == before
    assert [first.rand(100) for _ in range(100)] == [second.rand(100) for _ in range(100)]


def test_java_random_power_of_two_and_rejection_paths_stay_in_bounds():
    rng = JavaRandom(123456789)
    for bound in (1, 2, 4, 8, 16, 3, 7, 10, 97, 1000, 2_000_000_000):
        values = [rng.next_int(bound) for _ in range(200)]
        assert all(0 <= value < bound for value in values)


def test_java_random_unbounded_values_are_signed_32_bit():
    rng = JavaRandom(1)
    values = [rng.next_int() for _ in range(100)]
    assert all(-(1 << 31) <= value < (1 << 31) for value in values)
    assert any(value < 0 for value in values)
    assert any(value >= 0 for value in values)


def test_selfcitation_config_rejects_invalid_morph_mix():
    cfg = replace(SelfCitationConfig(), replace_probability=49)
    with pytest.raises(ValueError, match="sum to 100"):
        cfg.validate()
