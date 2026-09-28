from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.selfcitation import GenerateType, GlyphGroup
from voynich_lab.selfcitation_canfollow import CurveLineCanFollow


def test_released_word_final_setting_accepts_reference_start_and_final_boundaries():
    rule = CurveLineCanFollow(use_word_final_substitutions=True)
    assert rule.can_follow_each_other_after("", "qo")
    assert rule.can_follow_each_other_before("", "qotchedy")
    assert rule.can_follow_each_other_after("dy", "")
    assert rule.can_follow_each_other_before("dy", "")


def test_duplicate_prefix_and_suffix_are_rejected_before_type_rules():
    rule = CurveLineCanFollow()
    assert not rule.can_follow_each_other_before("ch", "chedy")
    assert not rule.can_follow_each_other_after("chedy", "dy")


def test_combinable_ligature_exceptions_are_applied_first():
    rule = CurveLineCanFollow()
    assert rule.can_follow_each_other_before("ol", "a")
    assert rule.can_follow_each_other_after("r", "ol")
    assert rule.can_follow_each_other_before("r", "ol")
    assert rule.can_follow_each_other_after("ol", "a")


def test_initial_groups_bypass_generated_start_validation_only():
    rule = CurveLineCanFollow()
    initial = GlyphGroup("xyz", GenerateType.INITIAL)
    generated = GlyphGroup("xyz", GenerateType.REPLACE)
    assert rule.has_valid_start_glyph(initial)
    assert not rule.has_valid_start_glyph(generated)


def test_known_reference_words_are_valid_under_curveline_rule():
    rule = CurveLineCanFollow()
    for word in (
        "pchal",
        "shal",
        "shorchdy",
        "okeor",
        "okain",
        "shedy",
        "pchedy",
        "qotchedy",
        "qotar",
        "ol",
        "lkar",
    ):
        assert rule.is_valid(GlyphGroup(word, GenerateType.INITIAL)), word


def test_generated_validity_checks_start_and_every_adjacent_pair():
    rule = CurveLineCanFollow()
    assert rule.is_valid(GlyphGroup("qotchedy", GenerateType.REPLACE))
    assert not rule.is_valid(GlyphGroup("xyz", GenerateType.REPLACE))


def test_disabling_word_final_substitutions_changes_empty_boundary_handling():
    enabled = CurveLineCanFollow(use_word_final_substitutions=True)
    disabled = CurveLineCanFollow(use_word_final_substitutions=False)
    assert enabled.can_follow_each_other_after("", "qo")
    assert not disabled.can_follow_each_other_after("", "qo")
    assert enabled.can_follow_each_other_before("dy", "")
    assert not disabled.can_follow_each_other_before("dy", "")
