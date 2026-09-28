from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.selfcitation import GlyphGroup
from voynich_lab.selfcitation_morph import (
    BaseGroupMorpher,
    Substitution,
    contains_first_line_gallow,
    random_gallow,
    random_line_initial_glyph,
    replace_first_line_gallow,
    replace_gallows,
    replace_group_final_glyph,
    replace_ingroup_glyph,
    replace_line_final_glyph,
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
        assert 0 <= value < maximum or maximum == 0
        return value


class PrimitiveMorpher(BaseGroupMorpher):
    def add_gallow(self, glyph_group, is_paragraph_initial, is_line_initial):
        return glyph_group

    def remove_line_initial_glyph(self, glyph_group):
        return glyph_group

    def replace_random_token(self, glyph_group, is_paragraph_initial):
        return glyph_group


def test_glyph_replacement_tables_match_upstream_examples():
    assert replace_line_final_glyph("ol") == "om"
    assert replace_line_final_glyph("iiin") == "im"
    assert replace_line_final_glyph("ch") == "ch"
    assert replace_ingroup_glyph("dy") == "da"
    assert replace_ingroup_glyph("n") == "r"
    assert replace_group_final_glyph("k") == "t"
    assert replace_group_final_glyph("a") == "y"
    assert contains_first_line_gallow("cph")
    assert contains_first_line_gallow("cfh")
    assert not contains_first_line_gallow("cth")
    assert replace_first_line_gallow("cphcfh", "k") == "ckhckh"
    assert replace_gallows("kptf", "t") == "tttt"


def test_random_gallow_exact_thresholds():
    cases = [
        (False, 34, "k"),
        (False, 35, "t"),
        (False, 49, "t"),
        (False, 50, "k"),
        (False, 89, "k"),
        (False, 90, "t"),
        (False, 99, "t"),
        (True, 34, "k"),
        (True, 35, "t"),
        (True, 49, "t"),
        (True, 50, "p"),
        (True, 89, "p"),
        (True, 90, "f"),
        (True, 99, "f"),
    ]
    for first_line, value, expected in cases:
        rng = QueueRandom(value)
        assert random_gallow(first_line, rng) == expected
        assert rng.calls == [100]


def test_random_line_initial_exact_thresholds():
    for value, expected in ((45, "o"), (46, "y"), (75, "y"), (76, "d"), (90, "d"), (91, "s"), (99, "s")):
        rng = QueueRandom(value)
        assert random_line_initial_glyph(rng) == expected
        assert rng.calls == [100]


def test_probability_clamping_keeps_at_least_twenty_percent_replace():
    morpher = PrimitiveMorpher()
    morpher.set_probabilities(20, 30)
    assert (morpher.add_remove_probability, morpher.combine_split_probability, morpher.replace_probability) == (20, 30, 50)

    morpher.set_probabilities(90, 90)
    assert (morpher.add_remove_probability, morpher.combine_split_probability, morpher.replace_probability) == (80, 0, 20)

    morpher.set_probabilities(70, 50)
    assert (morpher.add_remove_probability, morpher.combine_split_probability, morpher.replace_probability) == (70, 10, 20)


def test_canfollow_error_rate_uses_java_less_than_or_equal_boundary():
    allowed = PrimitiveMorpher(
        random_number_generator=QueueRandom(5),
        method_can_follow_error_rate=5,
    )
    assert allowed.can_follow_each_other_before("b", "b")

    denied = PrimitiveMorpher(
        random_number_generator=QueueRandom(6),
        method_can_follow_error_rate=5,
    )
    assert not denied.can_follow_each_other_before("b", "b")


def test_line_initial_helper_preserves_released_self_check_quirk():
    # 0 chooses the add path and then 'o'; 99 skips the d/ch -> gallow branch.
    # Upstream then calls canFollowToInitalGlyph("o", "o", candidate), so the
    # duplicate-prefix guard rejects its own candidate under CurveLineCanFollow.
    rng = QueueRandom(0, 0, 99)
    morpher = PrimitiveMorpher(random_number_generator=rng)
    source = GlyphGroup("chedy")
    assert morpher.add_line_initial_glyph(source) == source
    assert rng.calls == [100, 100, 100]


def test_choose_subgroups_matches_three_java_branches():
    source = GlyphGroup("qotchedy")

    prefix_rng = QueueRandom(39, 1)
    prefix = PrimitiveMorpher(random_number_generator=prefix_rng)
    assert prefix.choose_subgroups(source) == ["qo", "t"]
    assert prefix_rng.calls == [100, source.get_token_count() - 1]

    drop_last = PrimitiveMorpher(random_number_generator=QueueRandom(40))
    assert drop_last.choose_subgroups(source) == source.copy_tokens()[:-1]

    combinable = PrimitiveMorpher(random_number_generator=QueueRandom(60))
    assert combinable.choose_subgroups(GlyphGroup("olal")) == ["al"]


def test_split_position_and_split_result_match_curveline_rules():
    morpher = PrimitiveMorpher()
    source = GlyphGroup("olal")
    assert morpher.calc_split_position(source) == 1
    result = morpher.split_glyphgroup(source)
    assert [str(group) for group in result] == ["ol"]


def test_delete_prefix_rewrites_new_initial_gallow_with_same_random_draw():
    rng = QueueRandom(0)
    morpher = PrimitiveMorpher(random_number_generator=rng)
    result = morpher.try_to_delete_prefix(GlyphGroup("okedy"))
    assert str(result) == "chedy"
    assert result.generate_type.value == "DELETE"
    assert rng.calls == [100]


def test_gallow_replacement_helpers_consume_one_draw_per_matching_token():
    normal = PrimitiveMorpher(random_number_generator=QueueRandom(0, 35))
    result = normal.replace_first_line_gallows(GlyphGroup("pchalcfh"))
    assert str(result) == "kchalcth"
    assert normal.random_number_generator.calls == [100, 100]

    first_line = PrimitiveMorpher(random_number_generator=QueueRandom(50))
    result = first_line.replace_with_first_line_gallows(GlyphGroup("kchedy"))
    assert str(result) == "pchedy"
    assert first_line.random_number_generator.calls == [100]


def test_substitution_value_helpers_match_java_shape():
    substitution = Substitution(("k", "ch"), 50)
    assert substitution.count() == 2
    assert substitution.first() == "k"
    assert substitution.last() == "ch"
    assert substitution.get(1) == "ch"
    assert substitution.probability == 50
