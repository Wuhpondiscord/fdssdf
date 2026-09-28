from __future__ import annotations

from enum import Enum
import math
from typing import Protocol

from .selfcitation import GenerateType, GlyphGroup
from .selfcitation_chooser import JavaGlyphGroupMap
from .selfcitation_canfollow import CurveLineCanFollow


DAIIN_GROUP = GlyphGroup("daiin", GenerateType.INITIAL)
CHOL_GROUP = GlyphGroup("chol", GenerateType.INITIAL)
CHEDY_GROUP = GlyphGroup("chedy", GenerateType.INITIAL)
CHEODY_GROUP = GlyphGroup("cheody", GenerateType.INITIAL)

I_MIN_PERCENTAGE = 0.20
OL_MIN_PERCENTAGE = 0.25
DY_MIN_PERCENTAGE = 0.25


class RandomLike(Protocol):
    def rand(self, maximum: int) -> int: ...


class SuggestMethod(str, Enum):
    TOP_TOKEN = "top"
    DAIIN_OL_CHEDY = "daiin_ol_chedy"
    RANDOM = "random"
    NONE = "none"


def _java_float_ratio(numerator: int, denominator: int) -> float:
    # Java float division yields NaN for 0/0 rather than raising.
    if denominator == 0:
        return math.nan if numerator == 0 else math.copysign(math.inf, numerator)
    return float(numerator) / float(denominator)


class JavaGlyphGroupSet:
    """HashSet<GlyphGroup> model backed by the parity-tested HashMap ordering."""

    def __init__(self) -> None:
        self._map = JavaGlyphGroupMap()

    def __len__(self) -> int:
        return len(self._map)

    def add(self, group: GlyphGroup) -> None:
        if not self._map.contains_key(group):
            self._map.put(group, 1)

    def clear(self) -> None:
        self._map = JavaGlyphGroupMap()

    def key_set(self) -> list[GlyphGroup]:
        return self._map.key_set()


class StatisticHelper:
    """Generation-relevant port of upstream ``StatisticHelper``.

    The VMS-validity counters are informational only and do not feed generation.
    They can be supplied with a separate validator later; canonical generation
    uses CurveLine validity for dictionary admission exactly as the Java code.
    """

    def __init__(
        self,
        currier_type: str,
        random_number_generator: RandomLike,
        can_follow: CurveLineCanFollow,
        suggest_method: SuggestMethod | str = SuggestMethod.TOP_TOKEN,
        suggestions_probability: int = 40,
        *,
        i_min_percentage: float = I_MIN_PERCENTAGE,
        ol_min_percentage: float = OL_MIN_PERCENTAGE,
        dy_min_percentage: float = DY_MIN_PERCENTAGE,
    ) -> None:
        self.currier_type = str(currier_type).upper()
        self.random_number_generator = random_number_generator
        self.can_follow = can_follow
        self.suggest_method = SuggestMethod(suggest_method)
        self.suggestions_probability = int(suggestions_probability)
        self.i_min_percentage = float(i_min_percentage)
        self.ol_min_percentage = float(ol_min_percentage)
        self.dy_min_percentage = float(dy_min_percentage)

        self.valid_group_dictionary = JavaGlyphGroupMap()
        self.all_group_map = JavaGlyphGroupMap()

        self.lines_in_page = 0
        self.lines_in_paragraph = 0
        self.stat_tokens = 0
        self.stat_tokens_i = 0
        self.stat_tokens_dy = 0
        self.stat_tokens_ol = 0
        self.repeated_tokens_i = 0
        self.repeated_tokens_dy = 0
        self.repeated_tokens_ol = 0
        self.repeated_tokens_unknown = 0

        self.group_i_list_for_doc = JavaGlyphGroupSet()
        self.group_dy_list_for_doc = JavaGlyphGroupSet()
        self.group_ol_list_for_doc = JavaGlyphGroupSet()
        self.usage_i_map_for_doc = JavaGlyphGroupMap()
        self.usage_dy_map_for_doc = JavaGlyphGroupMap()
        self.usage_ol_map_for_doc = JavaGlyphGroupMap()

        self.group_i_list_for_page = JavaGlyphGroupSet()
        self.group_dy_list_for_page = JavaGlyphGroupSet()
        self.group_ol_list_for_page = JavaGlyphGroupSet()
        self.usage_i_map_for_page = JavaGlyphGroupMap()
        self.usage_dy_map_for_page = JavaGlyphGroupMap()
        self.usage_ol_map_for_page = JavaGlyphGroupMap()

    def get_percentage_dy(self) -> float:
        return _java_float_ratio(self.stat_tokens_dy, self.stat_tokens)

    def get_percentage_i(self) -> float:
        return _java_float_ratio(self.stat_tokens_i, self.stat_tokens)

    def get_percentage_ol(self) -> float:
        return _java_float_ratio(self.stat_tokens_ol, self.stat_tokens)

    def new_line(self) -> None:
        self.lines_in_page += 1
        self.lines_in_paragraph += 1

    def new_page(self) -> None:
        self.lines_in_page = 0
        self.lines_in_paragraph = 0
        self.group_i_list_for_page.clear()
        self.group_dy_list_for_page.clear()
        self.group_ol_list_for_page.clear()
        self.usage_i_map_for_page = JavaGlyphGroupMap()
        self.usage_dy_map_for_page = JavaGlyphGroupMap()
        self.usage_ol_map_for_page = JavaGlyphGroupMap()

    def new_paragraph(self) -> None:
        self.lines_in_paragraph = 0

    @staticmethod
    def _add_to_dictionary(group: GlyphGroup, dictionary: JavaGlyphGroupMap) -> None:
        dictionary.increment(group)

    @staticmethod
    def _add_to_usage_map(group: GlyphGroup, usage_map: JavaGlyphGroupMap) -> None:
        current = usage_map.get(group)
        usage_map.put(group, 1 if current is None else current + 1)

    def remember(self, glyph_group: GlyphGroup) -> None:
        if (
            glyph_group.generate_type == GenerateType.INITIAL
            or (
                self.can_follow.is_valid(glyph_group)
                and glyph_group.get_token_count() < 8
                and glyph_group.generate_type != GenerateType.COMBINE
            )
        ):
            self._add_to_dictionary(glyph_group, self.valid_group_dictionary)
        self._add_to_dictionary(glyph_group, self.all_group_map)

        self.stat_tokens += 1
        if glyph_group.is_type_i():
            self._add_to_usage_map(glyph_group, self.usage_i_map_for_doc)
            self._add_to_usage_map(glyph_group, self.usage_i_map_for_page)
            self.group_i_list_for_doc.add(glyph_group)
            self.group_i_list_for_page.add(glyph_group)
            self.repeated_tokens_dy = 0
            self.repeated_tokens_ol = 0
            self.repeated_tokens_unknown = 0
            self.stat_tokens_i += 1
            self.repeated_tokens_i += 1
        elif glyph_group.is_type_dy():
            self._add_to_usage_map(glyph_group, self.usage_dy_map_for_doc)
            self._add_to_usage_map(glyph_group, self.usage_dy_map_for_page)
            self.group_dy_list_for_doc.add(glyph_group)
            self.group_dy_list_for_page.add(glyph_group)
            self.repeated_tokens_unknown = 0
            self.repeated_tokens_i = 0
            self.repeated_tokens_ol = 0
            self.stat_tokens_dy += 1
            self.repeated_tokens_dy += 1
        elif glyph_group.is_type_ol():
            self._add_to_usage_map(glyph_group, self.usage_ol_map_for_doc)
            self._add_to_usage_map(glyph_group, self.usage_ol_map_for_page)
            self.group_ol_list_for_doc.add(glyph_group)
            self.group_ol_list_for_page.add(glyph_group)
            self.repeated_tokens_unknown = 0
            self.repeated_tokens_i = 0
            self.repeated_tokens_dy = 0
            self.stat_tokens_ol += 1
            self.repeated_tokens_ol += 1
        else:
            self.repeated_tokens_i = 0
            self.repeated_tokens_dy = 0
            self.repeated_tokens_ol = 0
            self.repeated_tokens_unknown += 1

    def has_suggested_groups(self) -> bool:
        if self.suggest_method == SuggestMethod.NONE:
            return False
        percentage_i = self.get_percentage_i()
        percentage_dy = self.get_percentage_dy()
        percentage_ol = self.get_percentage_ol()
        if self.currier_type == "B":
            return percentage_i < self.i_min_percentage or percentage_dy < self.dy_min_percentage
        if self.currier_type == "A":
            return percentage_i < self.i_min_percentage or percentage_ol < self.ol_min_percentage
        return percentage_i < self.i_min_percentage

    def suggest_groups(self) -> list[GlyphGroup]:
        result: list[GlyphGroup] = []
        percentage_i = self.get_percentage_i()
        percentage_dy = self.get_percentage_dy()
        percentage_ol = self.get_percentage_ol()
        if self.currier_type == "B":
            if percentage_i < self.i_min_percentage:
                result.append(self._choose_i_group())
            elif percentage_dy < self.dy_min_percentage:
                result.append(self._choose_dy_group())
        elif self.currier_type == "A":
            if percentage_i < self.i_min_percentage:
                result.append(self._choose_i_group())
            elif percentage_ol < self.ol_min_percentage:
                result.append(self._choose_ol_group())
        elif percentage_i < self.i_min_percentage:
            result.append(self._choose_i_group())
        return result

    def suggest_groups_force(self, try_count: int) -> list[GlyphGroup]:
        if try_count < 12:
            result = self.suggest_groups()
            if result:
                return result
        else:
            result = []

        mod = try_count % 3
        if self.currier_type == "B":
            if mod == 0:
                result.extend((self._choose_i_group(), self._choose_dy_group()))
            elif mod == 1:
                result.extend((self._choose_dy_group(), self._choose_i_group()))
            else:
                result.extend((self._choose_ol_group(), self._choose_i_group()))
        elif self.currier_type == "A":
            if mod == 0:
                result.extend((self._choose_i_group(), self._choose_ol_group()))
            elif mod == 1:
                result.extend((self._choose_ol_group(), self._choose_i_group()))
            else:
                result.extend((self._choose_dy_group(), self._choose_i_group()))
        else:
            result.extend((self._choose_i_group(), self._choose_ol_group()))
        return result

    def _choose_one_element_randomly(self, groups: JavaGlyphGroupSet) -> GlyphGroup:
        ordered = groups.key_set()
        index = self.random_number_generator.rand(len(ordered))
        return ordered[index]

    @staticmethod
    def _choose_top_group(default_group: GlyphGroup, usage_map: JavaGlyphGroupMap) -> GlyphGroup:
        max_group = default_group
        max_count = 0
        for group in usage_map.key_set():
            count = usage_map.get(group, 0) or 0
            if count > max_count:
                max_count = count
                max_group = group
        return max_group

    def _choose_ol_group(self) -> GlyphGroup:
        if self.suggest_method == SuggestMethod.RANDOM:
            if len(self.group_ol_list_for_page) > 0:
                return self._choose_one_element_randomly(self.group_ol_list_for_page)
            return self._choose_one_element_randomly(self.group_ol_list_for_doc)
        if self.suggest_method == SuggestMethod.TOP_TOKEN:
            # Preserve upstream's `size() >= 0` quirk: the page map always wins.
            return self._choose_top_group(CHOL_GROUP, self.usage_ol_map_for_page)
        return CHOL_GROUP

    def _choose_dy_group(self) -> GlyphGroup:
        default_group = CHEODY_GROUP if self.currier_type == "A" else CHEDY_GROUP
        if self.suggest_method == SuggestMethod.RANDOM:
            if len(self.group_dy_list_for_page) > 0:
                return self._choose_one_element_randomly(self.group_dy_list_for_page)
            return self._choose_one_element_randomly(self.group_dy_list_for_doc)
        if self.suggest_method == SuggestMethod.TOP_TOKEN:
            return self._choose_top_group(default_group, self.usage_dy_map_for_page)
        return default_group

    def _choose_i_group(self) -> GlyphGroup:
        if self.suggest_method == SuggestMethod.RANDOM:
            if len(self.group_i_list_for_page) > 0:
                return self._choose_one_element_randomly(self.group_i_list_for_page)
            return self._choose_one_element_randomly(self.group_i_list_for_doc)
        if self.suggest_method == SuggestMethod.TOP_TOKEN:
            return self._choose_top_group(DAIIN_GROUP, self.usage_i_map_for_page)
        return DAIIN_GROUP

    def get_suggestions_probability(self) -> int:
        return self.suggestions_probability
