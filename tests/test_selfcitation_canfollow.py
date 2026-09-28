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
    # Glyph.allCombinableLigature also contains the released word-final om/am.
    assert rule.can_follow_each_other_before("in", "om")
    assert rule.can_follow_each_other_after("am", "y")


def test_initial_groups_bypass_generated_start_validation_only():
    rule = CurveLineCanFollow()
    initial = GlyphGroup("byz", GenerateType.INITIAL)
    generated = GlyphGroup("byz", GenerateType.REPLACE)
    assert rule.has_valid_start_glyph(initial)
    assert not rule.has_valid_start_glyph(generated)


def test_initial_status_does_not_bypass_internal_pair_validation():
    rule = CurveLineCanFollow()
    # The released seed line contains shorchdy, but initial status only bypasses
    # start-glyph validation; the internal sh/or/ch/dy transitions still apply.
    assert not rule.is_valid(GlyphGroup("shorchdy", GenerateType.INITIAL))


def test_generated_validity_checks_start_and_every_adjacent_pair():
    rule = CurveLineCanFollow()
    assert rule.is_valid(GlyphGroup("qotchedy", GenerateType.REPLACE))
    assert not rule.is_valid(GlyphGroup("byz", GenerateType.REPLACE))


def test_disabling_word_final_substitutions_changes_empty_boundary_handling():
    enabled = CurveLineCanFollow(use_word_final_substitutions=True)
    disabled = CurveLineCanFollow(use_word_final_substitutions=False)
    assert enabled.can_follow_each_other_after("", "qo")
    assert not disabled.can_follow_each_other_after("", "qo")
    assert enabled.can_follow_each_other_before("dy", "")
    assert not disabled.can_follow_each_other_before("dy", "")
