from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.selfcitation import GenerateType, GlyphGroup, JavaRandom
from voynich_lab.selfcitation_chooser import (
    ChooserStatisticsSnapshot,
    JavaGlyphGroupMap,
    PageSourceGroupChooser,
    calc_line_position,
    choose_randomly,
    remove_initial_gallows,
)


class TraceRandom:
    def __init__(self, seed: int):
        self.inner = JavaRandom(seed)
        self.calls: list[tuple[int, int]] = []

    def rand(self, maximum: int) -> int:
        result = self.inner.rand(maximum)
        self.calls.append((maximum, result))
        return result


def make_dictionary(words):
    dictionary = JavaGlyphGroupMap()
    for word in words:
        dictionary.increment(GlyphGroup(word))
    return dictionary


def test_java_glyph_group_map_matches_default_hashmap_capacity_boundaries():
    dictionary = JavaGlyphGroupMap()
    assert dictionary.capacity == 0
    for index in range(1, 13):
        dictionary.put(GlyphGroup(f"x{index}"), 1)
        assert dictionary.capacity == 16
    dictionary.put(GlyphGroup("x13"), 1)
    assert dictionary.capacity == 32
    for index in range(14, 25):
        dictionary.put(GlyphGroup(f"x{index}"), 1)
    assert dictionary.capacity == 32
    dictionary.put(GlyphGroup("x25"), 1)
    assert dictionary.capacity == 64


def test_java_glyph_group_map_updates_do_not_change_key_order_or_size():
    dictionary = make_dictionary(["daiin", "chedy", "chol"])
    before = [str(group) for group in dictionary.key_set()]
    dictionary.increment(GlyphGroup("chedy", GenerateType.REPLACE))
    assert len(dictionary) == 3
    assert dictionary.get(GlyphGroup("chedy")) == 2
    assert [str(group) for group in dictionary.key_set()] == before


def test_calc_line_position_uses_character_position_not_word_index():
    source = [GlyphGroup("daiin"), GlyphGroup("ol"), GlyphGroup("qotchedy")]
    assert calc_line_position([], source) == 0
    assert calc_line_position([GlyphGroup("chol")], source) == 0
    assert calc_line_position([GlyphGroup("qotchedy")], source) == 1
    assert calc_line_position([GlyphGroup("qotchedy"), GlyphGroup("daiin")], source) == 2


def test_remove_initial_gallows_removes_one_token_but_keeps_single_gallow():
    source = [
        GlyphGroup("pchal", GenerateType.INITIAL),
        GlyphGroup("k", GenerateType.REPLACE),
        GlyphGroup("chedy", GenerateType.ADD),
    ]
    result = remove_initial_gallows(source)
    assert [str(group) for group in result] == ["chal", "k", "chedy"]
    assert result[0].generate_type == GenerateType.INITIAL
    assert result[1].generate_type == GenerateType.REPLACE
    assert result[2].generate_type == GenerateType.ADD


def test_choose_randomly_consumes_exactly_two_dictionary_draws():
    rng = TraceRandom(19)
    dictionary = make_dictionary(["daiin", "chedy", "chol", "pchal"])
    chosen = choose_randomly(dictionary, rng)
    assert len(chosen) == 2
    assert rng.calls == [(4, rng.calls[0][1]), (4, rng.calls[1][1])]


def test_page_chooser_random_mode_consumes_unconditional_paragraph_draw_first():
    rng = TraceRandom(19)
    stats = ChooserStatisticsSnapshot(
        valid_group_dictionary=make_dictionary(["daiin", "chedy", "chol", "pchal"]),
        lines_in_page=0,
    )
    chooser = PageSourceGroupChooser(rng)
    result = chooser.choose_source_group([], [], [], stats, False, True, 1)
    assert len(result) == 2
    assert [maximum for maximum, _ in rng.calls] == [100, 4, 4]


def test_page_chooser_local_mode_uses_page_line_and_position_draws():
    rng = TraceRandom(55)
    stats = ChooserStatisticsSnapshot(
        valid_group_dictionary=make_dictionary(["daiin", "chedy"]),
        lines_in_page=5,
    )
    generated = [
        [GlyphGroup("daiin"), GlyphGroup("ol"), GlyphGroup("chedy")],
        [GlyphGroup("chol"), GlyphGroup("qotchedy"), GlyphGroup("lkar")],
        [GlyphGroup("pchal"), GlyphGroup("shedy"), GlyphGroup("okeor")],
    ]
    chooser = PageSourceGroupChooser(rng)
    result = chooser.choose_source_group(
        generated,
        [],
        [GlyphGroup("daiin")],
        stats,
        False,
        False,
        1,
    )
    assert len(result) == 2
    # paragraph-mode probe, page-line probe, same-position probe; a fourth call
    # appears only when the same-position probability test fails.
    assert [maximum for maximum, _ in rng.calls][:3] == [100, 3, 100]


def test_page_chooser_paragraph_initial_mode_draws_line_then_position():
    rng = TraceRandom(1)
    stats = ChooserStatisticsSnapshot(
        valid_group_dictionary=make_dictionary(["daiin", "chedy"]),
        lines_in_page=2,
    )
    paragraphs = [
        [GlyphGroup("pchal"), GlyphGroup("daiin")],
        [GlyphGroup("tchedy"), GlyphGroup("chol")],
    ]
    chooser = PageSourceGroupChooser(rng)
    result = chooser.choose_source_group(
        [[GlyphGroup("daiin")]], paragraphs, [], stats, True, True, 1
    )
    assert len(result) == 2
    assert [maximum for maximum, _ in rng.calls] == [100, 2, 2]


def test_page_chooser_single_suggestion_consumes_two_random_fallback_draws():
    rng = TraceRandom(0)
    stats = ChooserStatisticsSnapshot(
        valid_group_dictionary=make_dictionary(["daiin", "chedy", "chol"]),
        lines_in_page=3,
        suggestions_probability=100,
        suggested_groups=[GlyphGroup("pchal")],
    )
    chooser = PageSourceGroupChooser(rng)
    result = chooser.choose_source_group(
        [[GlyphGroup("daiin")]], [], [], stats, False, False, 1
    )
    assert len(result) == 2
    # unconditional paragraph draw; suggestion-probability draw; then two
    # chooseRandomly draws even though only its first group is appended.
    assert [maximum for maximum, _ in rng.calls] == [100, 100, 3, 3]
    assert str(result[0]) == "chal"


def test_page_chooser_two_suggestions_adds_no_dictionary_draws():
    rng = TraceRandom(0)
    stats = ChooserStatisticsSnapshot(
        valid_group_dictionary=make_dictionary(["daiin", "chedy", "chol"]),
        suggestions_probability=100,
        suggested_groups=[GlyphGroup("daiin"), GlyphGroup("chedy")],
    )
    chooser = PageSourceGroupChooser(rng)
    result = chooser.choose_source_group(
        [[GlyphGroup("daiin")]], [], [], stats, False, False, 1
    )
    assert [str(group) for group in result] == ["daiin", "chedy"]
    assert [maximum for maximum, _ in rng.calls] == [100, 100]
