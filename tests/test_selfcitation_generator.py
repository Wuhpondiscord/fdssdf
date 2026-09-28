from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.selfcitation import GenerateType, GlyphGroup, JavaRandom, canonical_selfcitation_config
from voynich_lab.selfcitation_generator import (
    SelfCitationTextGenerator,
    generate_selfcitation,
    search_shorter_glyph,
)


def _expected_initial_output(config):
    generator = SelfCitationTextGenerator(config)
    return generator.trim_initial_line(config.initial_line.split(" "))


def test_search_shorter_glyph_matches_upstream_first_shortest_rule():
    assert search_shorter_glyph("a") is None
    assert search_shorter_glyph("xyz") is None
    assert search_shorter_glyph("iiin").tokens == ("n",)
    assert search_shorter_glyph("ee").tokens == ("e",)
    assert search_shorter_glyph("cth").tokens == ("ch",)


def test_generator_wires_one_shared_rng_into_all_generation_components():
    rng = JavaRandom(19)
    generator = SelfCitationTextGenerator(random_number_generator=rng)
    assert generator.random_number_generator is rng
    assert generator.statistics.random_number_generator is rng
    assert generator.source_group_chooser.random_number_generator is rng
    assert generator.group_morpher.random_number_generator is rng


def test_initial_line_populates_statistics_but_not_generated_line_arrays():
    generator = SelfCitationTextGenerator()
    expected_initial = generator.trim_initial_line(generator.config.initial_line.split(" "))
    assert generator.generated_text == [expected_initial]
    assert generator.line_arrays == []
    assert generator.paragraph_initial_line_arrays == []
    assert generator.statistics.stat_tokens == len(generator.config.initial_line.split(" "))


def test_trim_initial_line_preserves_upstream_overrun_quirk():
    config = replace(canonical_selfcitation_config(), max_line_length=40)
    generator = SelfCitationTextGenerator(config)
    result = generator.trim_initial_line(("a" * 39, "bbbb"))
    assert result == ("a" * 39) + " bbbb"
    assert len(result) == 44


def test_try_to_trim_uses_shorter_substitution_and_shorten_type():
    generator = SelfCitationTextGenerator()
    result = generator.try_to_trim(GlyphGroup("cthdy"), 3)
    assert str(result) == "chdy"
    assert result.generate_type == GenerateType.SHORTEN


def test_try_to_trim_returns_original_when_no_change_is_available():
    generator = SelfCitationTextGenerator()
    source = GlyphGroup("dy")
    assert generator.try_to_trim(source, 2) is source


def test_small_canonical_generation_is_seed_deterministic():
    config = replace(canonical_selfcitation_config(), lines_to_create=8)
    first = generate_selfcitation(config)
    second = generate_selfcitation(config)
    assert first.lines == second.lines
    assert len(first.lines) == 8
    assert first.lines[0] == _expected_initial_output(config)
    assert all(isinstance(line, str) for line in first.lines)


def test_generate_prefix_request_can_return_only_initial_lines_without_rng_work():
    config = replace(canonical_selfcitation_config(), lines_to_create=8)
    result = generate_selfcitation(config, lines_to_create=1)
    assert result.lines == (_expected_initial_output(config),)
