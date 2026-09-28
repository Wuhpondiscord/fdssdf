from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.generator_adapters import (
    estimate_selfcitation_lines,
    selfcitation_reference_matched,
)


def test_estimate_selfcitation_lines_prefers_observed_line_count():
    text = "alpha beta\ngamma delta\nepsilon zeta\n"
    assert estimate_selfcitation_lines(text) == 3


def test_estimate_selfcitation_lines_uses_character_fallback_for_one_line():
    assert estimate_selfcitation_lines("a" * 90) == 2
    assert estimate_selfcitation_lines("a" * 91) == 3


def test_estimate_selfcitation_lines_respects_upper_bound():
    text = "\n".join("x" for _ in range(2000))
    assert estimate_selfcitation_lines(text) == 1200


def test_selfcitation_reference_adapter_is_seed_deterministic_and_line_matched():
    reference = "a\nb\nc\nd\ne\n"
    first = selfcitation_reference_matched(reference, seed=123)
    second = selfcitation_reference_matched(reference, seed=123)
    third = selfcitation_reference_matched(reference, seed=124)

    assert first == second
    assert first != third
    assert len(first.splitlines()) == 5
