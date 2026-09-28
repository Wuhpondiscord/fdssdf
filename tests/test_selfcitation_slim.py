from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.selfcitation import GenerateType, GlyphGroup
from voynich_lab.selfcitation_slim import (
    FINAL_SUBSTITUTION_MAP,
    SUBSTITUTION_MAP,
    SlimGroupMorpher,
    similar_glyph,
)


class QueueRandom:
    def __init__(self, *values):
        self.values = list(values)
        self.calls = []

    def rand(self, maximum):
        self.calls.append(maximum)
        if not self.values:
            raise AssertionError(f"unexpected rand({maximum})")
        value = self.values.pop(0)
        assert maximum == 0 or 0 <= value < maximum
        return value


def test_every_substitution_row_is_cumulative_and_ends_at_100():
    assert SUBSTITUTION_MAP
    assert FINAL_SUBSTITUTION_MAP
    for table in (SUBSTITUTION_MAP, FINAL_SUBSTITUTION_MAP):
        for substitutions in table.values():
            probabilities = [sub.probability for sub in substitutions]
            assert probabilities == sorted(probabilities)
            assert probabilities[-1] == 100


def test_final_substitution_table_overrides_normal_table_only_when_requested():
    normal = similar_glyph("ol", False)
    final = similar_glyph("ol", True)
    assert normal is SUBSTITUTION_MAP["ol"]
    assert final is FINAL_SUBSTITUTION_MAP["ol"]
    assert similar_glyph("ch", True) is SUBSTITUTION_MAP["ch"]
    assert similar_glyph("z", True) is None


def test_choose_substitution_uses_strict_greater_than_cumulative_boundary():
    # k: t<77, p<94, f<100. A draw equal to a threshold falls through.
    for draw, expected in ((76, "t"), (77, "p"), (93, "p"), (94, "f"), (99, "f")):
        rng = QueueRandom(draw)
        morpher = SlimGroupMorpher(random_number_generator=rng)
        chosen = morpher._choose_substitution(SUBSTITUTION_MAP["k"])
        assert chosen.tokens == (expected,)
        assert rng.calls == [100]


def test_previous_dy_group_forces_q_prefix_on_first_attempt_below_90():
    rng = QueueRandom(89)
    morpher = SlimGroupMorpher(random_number_generator=rng)
    assert morpher._choose_prefix(0, GlyphGroup("chedy"), []) == "q"
    assert rng.calls == [100]


def test_previous_dy_q_rule_stops_at_90():
    rng = QueueRandom(90)
    morpher = SlimGroupMorpher(random_number_generator=rng)
    assert morpher._choose_prefix(0, GlyphGroup("chedy"), []) == "d"


def test_q_prefix_replaces_initial_o_with_qo_without_extra_rng_after_choice():
    rng = QueueRandom(80)
    morpher = SlimGroupMorpher(random_number_generator=rng)
    result = morpher._try_to_add_prefix(GlyphGroup("ol"), None, False)
    assert str(result) == "qol"
    assert result.generate_type == GenerateType.ADD
    assert rng.calls == [100]


def test_remove_line_initial_glyph_rewrites_gallow_before_e():
    morpher = SlimGroupMorpher(random_number_generator=QueueRandom())
    source = GlyphGroup("okedy")
    result = morpher.remove_line_initial_glyph(source)
    assert str(result) == "chedy"
    assert result.generate_type == GenerateType.INITIAL


def test_calc_length_counts_characters_not_tokens():
    morpher = SlimGroupMorpher(random_number_generator=QueueRandom())
    assert morpher.calc_length(["qo", "ch", "e", "dy"]) == 7


def test_replace_position_list_shrinks_during_java_style_loop():
    # One-token 'k' starts with positions [0,0]. Java removes one entry on the
    # first attempt and then exits because j==remaining size. The trace therefore
    # contains one position draw and one substitution draw, plus legality draws
    # only if CurveLine requires them.
    rng = QueueRandom(0, 0)
    morpher = SlimGroupMorpher(random_number_generator=rng)
    result = morpher.replace_random_token(GlyphGroup("k"), False)
    assert result in (GlyphGroup("k"), GlyphGroup("t", GenerateType.REPLACE))
    assert rng.calls[:2] == [2, 100]
