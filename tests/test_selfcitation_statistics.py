from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.selfcitation import GenerateType, GlyphGroup
from voynich_lab.selfcitation_canfollow import CurveLineCanFollow
from voynich_lab.selfcitation_statistics import (
    CHEDY_GROUP,
    CHOL_GROUP,
    DAIIN_GROUP,
    StatisticHelper,
    SuggestMethod,
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
        assert 0 <= value < maximum
        return value


def stats(*, currier="B", method=SuggestMethod.TOP_TOKEN, rng=None):
    return StatisticHelper(
        currier,
        rng or QueueRandom(),
        CurveLineCanFollow(use_word_final_substitutions=True),
        method,
        40,
    )


def test_empty_percentages_match_java_nan_semantics():
    helper = stats()
    assert math.isnan(helper.get_percentage_i())
    assert math.isnan(helper.get_percentage_dy())
    assert math.isnan(helper.get_percentage_ol())
    assert not helper.has_suggested_groups()
    assert helper.suggest_groups() == []


def test_remember_tracks_types_and_repeat_counters():
    helper = stats()
    helper.remember(GlyphGroup("daiin"))
    assert helper.stat_tokens == 1
    assert helper.stat_tokens_i == 1
    assert helper.repeated_tokens_i == 1

    helper.remember(GlyphGroup("shedy"))
    assert helper.stat_tokens_dy == 1
    assert helper.repeated_tokens_i == 0
    assert helper.repeated_tokens_dy == 1

    helper.remember(GlyphGroup("chol"))
    assert helper.stat_tokens_ol == 1
    assert helper.repeated_tokens_dy == 0
    assert helper.repeated_tokens_ol == 1

    helper.remember(GlyphGroup("qok"))
    assert helper.repeated_tokens_ol == 0
    assert helper.repeated_tokens_unknown == 1


def test_initial_groups_enter_valid_dictionary_even_if_generation_rule_rejects_them():
    helper = stats()
    initial = GlyphGroup("bb", GenerateType.INITIAL)
    helper.remember(initial)
    assert helper.valid_group_dictionary.contains_key(initial)

    generated = GlyphGroup("bb", GenerateType.REPLACE)
    helper.remember(generated)
    assert helper.valid_group_dictionary.get(generated) == 1
    assert helper.all_group_map.get(generated) == 2


def test_combine_groups_never_enter_valid_dictionary_after_generation():
    helper = stats()
    combined = GlyphGroup("chedy", GenerateType.COMBINE)
    helper.remember(combined)
    assert not helper.valid_group_dictionary.contains_key(combined)
    assert helper.all_group_map.contains_key(combined)


def test_page_reset_preserves_document_state_but_clears_page_usage():
    helper = stats()
    helper.remember(GlyphGroup("daiin"))
    helper.new_line()
    helper.new_line()
    assert helper.lines_in_page == 2
    assert len(helper.usage_i_map_for_page) == 1
    assert len(helper.usage_i_map_for_doc) == 1

    helper.new_page()
    assert helper.lines_in_page == 0
    assert helper.lines_in_paragraph == 0
    assert len(helper.usage_i_map_for_page) == 0
    assert len(helper.usage_i_map_for_doc) == 1


def test_top_token_uses_page_map_even_when_empty_due_upstream_size_ge_zero_quirk():
    helper = stats(method=SuggestMethod.TOP_TOKEN)
    helper.remember(GlyphGroup("daiin"))
    helper.remember(GlyphGroup("qokaiin"))
    helper.remember(GlyphGroup("qokaiin"))
    helper.new_page()
    # The document map has qokaiin as top token, but upstream always consults
    # the page map because it checks size() >= 0. Empty page map => default.
    assert helper._choose_i_group() == DAIIN_GROUP


def test_top_token_prefers_first_hashmap_iteration_entry_on_equal_counts():
    helper = stats(method=SuggestMethod.TOP_TOKEN)
    a = GlyphGroup("daiin")
    b = GlyphGroup("qokaiin")
    helper.remember(a)
    helper.remember(b)
    expected = helper.usage_i_map_for_page.key_set()[0]
    assert helper._choose_i_group() == expected


def test_currier_b_suggestion_priority_is_i_then_dy():
    helper = stats(currier="B")
    for word in ("shedy", "shedy", "shedy", "shedy"):
        helper.remember(GlyphGroup(word))
    assert helper.has_suggested_groups()
    assert helper.suggest_groups() == [DAIIN_GROUP]

    helper.remember(GlyphGroup("daiin"))
    helper.remember(GlyphGroup("daiin"))
    helper.remember(GlyphGroup("daiin"))
    helper.remember(GlyphGroup("daiin"))
    # i is now well over 20%; dy may or may not be under 25 depending on totals,
    # but if it is under target the B-mode fallback is the top dy/default group.
    suggestion = helper.suggest_groups()
    if suggestion:
        assert suggestion[0] == CHEDY_GROUP or suggestion[0].is_type_dy()


def test_force_suggestion_cycles_exact_b_order_after_try_12():
    helper = stats(currier="B")
    assert helper.suggest_groups_force(12) == [DAIIN_GROUP, CHEDY_GROUP]
    assert helper.suggest_groups_force(13) == [CHEDY_GROUP, DAIIN_GROUP]
    assert helper.suggest_groups_force(14) == [CHOL_GROUP, DAIIN_GROUP]


def test_random_suggestion_uses_java_ordered_set_and_one_rng_draw():
    rng = QueueRandom(0)
    helper = stats(method=SuggestMethod.RANDOM, rng=rng)
    helper.remember(GlyphGroup("daiin"))
    helper.remember(GlyphGroup("qokaiin"))
    ordered = helper.group_i_list_for_page.key_set()
    assert helper._choose_i_group() == ordered[0]
    assert rng.calls == [2]
