from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from voynich_lab.selfcitation import GenerateType, GlyphGroup, JavaRandom
from voynich_lab.selfcitation_canfollow import CurveLineCanFollow


class RandomSource(Protocol):
    def rand(self, maximum: int) -> int: ...


GALLOW_GLYPHS: tuple[str, ...] = ("k", "t", "p", "f")
LINE_INITIAL_GLYPHS: tuple[str, ...] = ("o", "y", "d", "s")
LINE_FINAL_GLYPHS: tuple[str, ...] = ("m", "g")
COMBINABLE_LIGATURES: tuple[str, ...] = ("ol", "al", "or", "ar")
ALL_COMBINABLE_LIGATURES: dict[str, tuple[str, ...]] = {
    "ol": ("l", "r", "s"),
    "al": ("l", "r", "s"),
    "or": ("l", "r", "s"),
    "ar": ("l", "r", "s"),
    "om": ("l", "r", "s"),
    "am": ("l", "r", "s"),
}
PREFIX_GLYPHS: dict[str, tuple[str, ...]] = {
    "l": ("k", "t", "p", "f", "d", "ch", "sh", "o", "a", "e", "i", "o"),
    "o": ("k", "t", "p", "f", "d", "ch", "sh"),
    "y": ("k", "t", "p", "f", "d", "ch", "sh"),
    "ch": ("k", "t", "p", "f", "d", "ol", "or", "al", "ar"),
    "sh": ("k", "t", "p", "f", "d", "ol", "or", "al", "ar"),
    "q": (),
    "d": ("a",),
    "x": ("ol", "or", "al", "ar"),
}
FINAL_LINE_REPLACEMENTS: dict[str, str] = {
    "ol": "om",
    "or": "om",
    "al": "am",
    "ar": "am",
    "om": "og",
    "am": "ag",
    "im": "mg",
    "in": "n",
    "iin": "im",
    "iiin": "im",
}
INGROUP_GLYPH_REPLACEMENTS: dict[str, str] = {
    "y": "o",
    "m": "r",
    "g": "r",
    "n": "r",
    "in": "ir",
    "iin": "iir",
    "dy": "da",
}
GROUP_FINAL_GLYPH_REPLACEMENTS: dict[str, str] = {
    "o": "y",
    "a": "y",
    "k": "t",
    "t": "k",
    "p": "f",
    "f": "p",
}
PARAGRAPH_STARTS_WITH_GALLOW_PROBABILITY = 94
MORPH_REUSED_PROBABILITY = 50
MIN_REPLACE_PROBABILITY = 20


@dataclass(frozen=True)
class Substitution:
    tokens: tuple[str, ...]
    probability: int

    def __init__(self, tokens: Sequence[str], probability: int) -> None:
        object.__setattr__(self, "tokens", tuple(str(token) for token in tokens))
        object.__setattr__(self, "probability", int(probability))

    def count(self) -> int:
        return len(self.tokens)

    def get(self, index: int) -> str:
        return self.tokens[index]

    def first(self) -> str:
        return self.tokens[0]

    def last(self) -> str:
        return self.tokens[-1]


def is_gallow(token: str) -> bool:
    return token in GALLOW_GLYPHS


def is_combinable_ligature(token: str) -> bool:
    return token in COMBINABLE_LIGATURES


def replace_line_final_glyph(token: str) -> str:
    return FINAL_LINE_REPLACEMENTS.get(token, token)


def replace_ingroup_glyph(token: str) -> str:
    return INGROUP_GLYPH_REPLACEMENTS.get(token, token)


def replace_group_final_glyph(token: str) -> str:
    return GROUP_FINAL_GLYPH_REPLACEMENTS.get(token, token)


def contains_first_line_gallow(text: str) -> bool:
    return "p" in text or "f" in text


def replace_first_line_gallow(text: str, new_gallow: str) -> str:
    result = text
    for gallow in GALLOW_GLYPHS:
        if gallow in ("p", "f") and gallow in text:
            result = result.replace(gallow, new_gallow)
    return result


def replace_gallows(text: str, new_gallow: str) -> str:
    result = text
    for gallow in GALLOW_GLYPHS:
        if gallow in text:
            result = result.replace(gallow, new_gallow)
    return result


def random_line_initial_glyph(random_number_generator: RandomSource) -> str:
    rand = random_number_generator.rand(100)
    if rand <= 45:
        return LINE_INITIAL_GLYPHS[0]
    if rand <= 75:
        return LINE_INITIAL_GLYPHS[1]
    if rand <= 90:
        return LINE_INITIAL_GLYPHS[2]
    if rand <= 99:
        return LINE_INITIAL_GLYPHS[3]
    return LINE_INITIAL_GLYPHS[0]


def random_gallow(first_line: bool, random_number_generator: RandomSource) -> str:
    rand = random_number_generator.rand(100)
    if rand <= 34:
        return GALLOW_GLYPHS[0]
    if rand <= 49:
        return GALLOW_GLYPHS[1]
    if rand <= 89:
        return GALLOW_GLYPHS[2] if first_line else GALLOW_GLYPHS[0]
    if rand <= 99:
        return GALLOW_GLYPHS[3] if first_line else GALLOW_GLYPHS[1]
    return GALLOW_GLYPHS[0]


class BaseGroupMorpher:
    """Port of the released ``AbstractBaseGroupMorpher`` for CurveLine mode.

    The released executable uses ``CurveLineCanFollow``. That implementation
    ignores Java's ``resultingGlyphGroup`` argument, so the Python wrappers keep
    the exact RNG/error-rate behavior without manufacturing unused objects.
    """

    def __init__(
        self,
        *,
        random_number_generator: RandomSource | None = None,
        can_follow: CurveLineCanFollow | None = None,
        method_can_follow_error_rate: int = 0,
        reuse_last_probability: int = 10,
        currier_type: str = "B",
    ) -> None:
        self.random_number_generator = random_number_generator or JavaRandom(19)
        self.can_follow = can_follow or CurveLineCanFollow(use_word_final_substitutions=True)
        self.method_can_follow_error_rate = int(method_can_follow_error_rate)
        self.reuse_last_probability = int(reuse_last_probability)
        self.currier_type = str(currier_type)
        self.combine_split_probability = 10
        self.add_remove_probability = 30
        self.replace_probability = 60
        self.morph_reused_probability = MORPH_REUSED_PROBABILITY
        self.min_replace_probability = MIN_REPLACE_PROBABILITY

    def set_probabilities(self, add_remove_probability: int, combine_split_probability: int) -> None:
        max_add_remove_probability = 100 - self.min_replace_probability
        self.add_remove_probability = min(int(add_remove_probability), max_add_remove_probability)
        combine_split_probability = int(combine_split_probability)
        if self.add_remove_probability + combine_split_probability > max_add_remove_probability:
            self.combine_split_probability = min(
                combine_split_probability,
                max_add_remove_probability - self.add_remove_probability,
            )
        else:
            self.combine_split_probability = combine_split_probability
        self.replace_probability = (
            100 - self.add_remove_probability - self.combine_split_probability
        )

    def can_follow_each_other_before(self, glyph_to_add: str, group2: str) -> bool:
        can_follow_before = self.can_follow.can_follow_each_other_before(glyph_to_add, group2)
        if not can_follow_before and self.method_can_follow_error_rate > 0:
            rand = self.random_number_generator.rand(100)
            if rand <= self.method_can_follow_error_rate:
                return True
        return can_follow_before

    def can_follow_each_other_after(self, group1: str, glyph_to_add: str) -> bool:
        can_follow_after = self.can_follow.can_follow_each_other_after(group1, glyph_to_add)
        if not can_follow_after and self.method_can_follow_error_rate > 0:
            rand = self.random_number_generator.rand(100)
            if rand <= self.method_can_follow_error_rate:
                return True
        return can_follow_after

    def can_follow_to_initial_glyph(self, glyph_to_add: str, group2: str) -> bool:
        return self.can_follow.can_follow_to_initial_glyph(glyph_to_add, group2)

    def is_replacable(
        self,
        substitution: Substitution,
        glyph_group: GlyphGroup,
        pos: int,
        remove_next: bool,
    ) -> bool:
        last = "" if pos == 0 else glyph_group.get_token(pos - 1)
        next_pos = pos + 2 if remove_next else pos + 1
        next_token = "" if next_pos >= glyph_group.get_token_count() else glyph_group.get_token(next_pos)
        last_ok = self.can_follow_each_other_after(last, substitution.first())
        next_ok = self.can_follow_each_other_before(substitution.last(), next_token)
        if (
            last_ok
            and next_ok
            and glyph_group.contains(substitution.first())
            and not is_combinable_ligature(substitution.first())
        ):
            compare_value = 80
            if substitution.first() == last or substitution.last() == next_token:
                compare_value = 100 - self.method_can_follow_error_rate
            rand = self.random_number_generator.rand(100)
            if rand < compare_value:
                return False
        return last_ok and next_ok

    def handle_gallows(
        self,
        is_paragraph_initial: bool,
        is_line_initial: bool,
        groups: list[GlyphGroup],
    ) -> None:
        if not groups:
            return
        if is_line_initial:
            rand = self.random_number_generator.rand(100)
            if is_paragraph_initial and rand < PARAGRAPH_STARTS_WITH_GALLOW_PROBABILITY:
                groups[0] = self.add_gallow(groups[0], True, True)
            else:
                groups[0] = self.add_line_initial_glyph(groups[0])
        elif groups[0].starts_with_line_initial_glyph():
            rand = self.random_number_generator.rand(len(PREFIX_GLYPHS))
            if rand < 3 and groups[0].get_token_count() > 3:
                groups[0] = self.remove_line_initial_glyph(groups[0])

        if groups[0].contains_gallow():
            if not is_paragraph_initial:
                rand = self.random_number_generator.rand(100)
                if rand < PARAGRAPH_STARTS_WITH_GALLOW_PROBABILITY:
                    groups[0] = self.replace_first_line_gallows(groups[0])
            else:
                groups[0] = self.replace_with_first_line_gallows(groups[0])

    def add_line_initial_glyph(self, glyph_group: GlyphGroup) -> GlyphGroup:
        if not glyph_group.starts_with_line_initial_glyph():
            rand = self.random_number_generator.rand(100)
            if rand < 30:
                length = glyph_group.get_token_count()
                glyph = random_line_initial_glyph(self.random_number_generator)
                new_tokens = glyph_group.copy_tokens()
                if (
                    length > 2
                    and glyph in ("o", "y")
                    and glyph_group.get_token(0) in ("d", "ch")
                ):
                    rand = self.random_number_generator.rand(100)
                    if rand < 90:
                        gallow = random_gallow(False, self.random_number_generator)
                        if self.can_follow_each_other_before(gallow, new_tokens[1]):
                            new_tokens[0] = gallow
                new_tokens.insert(0, glyph)
                candidate = GlyphGroup(new_tokens, glyph_group.generate_type)
                # Preserve the released Java call exactly. It passes the newly
                # inserted glyph as both glyphToAdd and group2, which means the
                # CurveLine duplicate-prefix guard usually rejects the candidate.
                if self.can_follow_to_initial_glyph(glyph, new_tokens[0]):
                    return candidate
        return glyph_group

    def combine_glyphgroups(self, source_groups: Sequence[GlyphGroup]) -> GlyphGroup:
        glyph_group1 = source_groups[0]
        glyph_group2 = source_groups[1]
        if (
            glyph_group1.get_token_count() == 2
            or (
                glyph_group1.get_token_count() == 1
                and is_combinable_ligature(glyph_group1.glyph_group)
            )
        ):
            rand = self.random_number_generator.rand(100)
            if rand < 60:
                new_group = self.self_combine(glyph_group1)
                if new_group != glyph_group1:
                    return new_group

        new_tokens2 = self.choose_subgroups(glyph_group2)
        if new_tokens2:
            new_tokens1 = glyph_group1.copy_tokens()
            last_token1 = new_tokens1[-1]
            if is_combinable_ligature(last_token1):
                tokens_to_remove = ALL_COMBINABLE_LIGATURES.get(last_token1)
                if tokens_to_remove is not None:
                    for token in tokens_to_remove:
                        if new_tokens2 and token == new_tokens2[0]:
                            new_tokens2.pop(0)

            new_length = len(new_tokens1) + len(new_tokens2)
            if new_tokens2 and new_length < 9:
                new_tokens1.extend(new_tokens2)
                candidate = GlyphGroup(new_tokens1, GenerateType.COMBINE)
                if (
                    self.can_follow_each_other_before(last_token1, new_tokens2[0])
                    and self.can_follow_each_other_after(new_tokens2[-1], "")
                ):
                    return candidate

        if glyph_group1.get_token_count() <= 2:
            return self.self_combine(glyph_group1)
        return glyph_group1

    def choose_subgroups(self, glyph_group: GlyphGroup) -> list[str]:
        rand = self.random_number_generator.rand(100)
        if rand <= 39:
            result: list[str] = []
            if glyph_group.get_token_count() > 1:
                rand = self.random_number_generator.rand(glyph_group.get_token_count() - 1) + 1
                for index in range(rand):
                    result.append(glyph_group.get_token(index))
            return result
        if rand <= 59:
            new_tokens = glyph_group.copy_tokens()
            new_tokens.pop(glyph_group.get_token_count() - 1)
            return new_tokens
        subgroup = glyph_group.get_first_combinable_ligature()
        if subgroup is not None:
            return [subgroup]
        new_tokens = glyph_group.copy_tokens()
        new_tokens.pop(glyph_group.get_token_count() - 1)
        return new_tokens

    def self_combine(self, glyph_group: GlyphGroup) -> GlyphGroup:
        if glyph_group.get_token_count() > 0:
            first_token = glyph_group.get_token(0)
            last_token = glyph_group.get_token(glyph_group.get_token_count() - 1)
            if self.can_follow_each_other_before(last_token, first_token):
                new_tokens = glyph_group.copy_tokens()
                new_tokens1 = glyph_group.copy_tokens()
                last_token_new = new_tokens[-1]
                ingroup_replacement = replace_ingroup_glyph(last_token_new)
                final_replacement = replace_group_final_glyph(last_token_new)
                if last_token_new != final_replacement:
                    rand = self.random_number_generator.rand(100)
                    if rand < 80:
                        new_tokens[-1] = final_replacement
                if last_token_new != ingroup_replacement:
                    new_tokens1[-1] = ingroup_replacement
                if (
                    self.can_follow_each_other_before(GALLOW_GLYPHS[0], first_token)
                    and self.can_follow_each_other_before(last_token_new, GALLOW_GLYPHS[0])
                ):
                    rand = self.random_number_generator.rand(100)
                    if rand < 30:
                        gallow = random_gallow(False, self.random_number_generator)
                        new_tokens.insert(1, gallow)
                for index, token in enumerate(new_tokens1):
                    new_tokens.insert(index, token)
                return GlyphGroup(new_tokens, GenerateType.COMBINE)
        return glyph_group

    def split_glyphgroup(self, glyph_group: GlyphGroup) -> list[GlyphGroup]:
        result: list[GlyphGroup] = []
        pos = self.calc_split_position(glyph_group)
        if pos >= 0:
            result = self.split_at_position(glyph_group, pos)
        if not result:
            result.append(glyph_group)
        return result

    def calc_split_position(self, glyph_group: GlyphGroup) -> int:
        for index in range(1, glyph_group.get_token_count()):
            last = glyph_group.get_token(index - 1)
            next_token = glyph_group.get_token(index)
            if (
                not self.can_follow_each_other_before(last, next_token)
                or is_combinable_ligature(last)
                or is_gallow(next_token)
            ):
                if index > 1 or len(last) > 1:
                    return index
        return -1

    def try_to_delete_prefix(self, glyph_group: GlyphGroup) -> GlyphGroup:
        group_length = glyph_group.get_token_count()
        start_token = glyph_group.get_token(0)
        new_tokens = glyph_group.copy_tokens()
        new_tokens.pop(0)
        if group_length > 2 and self.can_follow_each_other_before("", start_token):
            first_token = new_tokens[0]
            second_token = new_tokens[1]
            if is_gallow(first_token):
                rand = self.random_number_generator.rand(100)
                if second_token.startswith("a") and rand < 50:
                    new_tokens[0] = "d"
                if (second_token.startswith("e") or second_token.startswith("o")) and rand < 50:
                    new_tokens[0] = "ch"
            return GlyphGroup(new_tokens, GenerateType.DELETE)
        return glyph_group

    def split_at_position(self, glyph_group: GlyphGroup, pos: int) -> list[GlyphGroup]:
        new_tokens = glyph_group.copy_tokens()
        new_tokens1 = new_tokens[:pos]
        new_tokens2 = new_tokens[pos:]
        result: list[GlyphGroup] = []
        group1 = GlyphGroup(new_tokens1, GenerateType.SPLIT)
        if (
            (group1.get_token_count() > 1 or is_combinable_ligature(group1.glyph_group))
            and self.can_follow_each_other_after(
                group1.get_token(group1.get_token_count() - 1), ""
            )
        ):
            result.append(group1)
        group2 = GlyphGroup(new_tokens2, GenerateType.SPLIT)
        if group2.get_token_count() > 1:
            result.append(group2)
        return result

    def replace_first_line_gallows(self, glyph_group: GlyphGroup) -> GlyphGroup:
        new_tokens = glyph_group.copy_tokens()
        for index, token in enumerate(new_tokens):
            if contains_first_line_gallow(token):
                new_gallow = random_gallow(False, self.random_number_generator)
                new_tokens[index] = replace_first_line_gallow(token, new_gallow)
        return GlyphGroup(new_tokens, glyph_group.generate_type)

    def replace_with_first_line_gallows(self, glyph_group: GlyphGroup) -> GlyphGroup:
        new_tokens = glyph_group.copy_tokens()
        for index, token in enumerate(new_tokens):
            if any(gallow in token for gallow in GALLOW_GLYPHS):
                new_gallow = random_gallow(True, self.random_number_generator)
                new_tokens[index] = replace_gallows(token, new_gallow)
        return GlyphGroup(new_tokens, glyph_group.generate_type)

    def reuse_last_morphed_group(
        self,
        is_paragraph_initial: bool,
        groups: list[GlyphGroup],
    ) -> None:
        if len(groups) > 1:
            return
        rand = self.random_number_generator.rand(100)
        if rand < self.reuse_last_probability:
            last_group = groups[0]
            if last_group.contains_combinable_ligature():
                ligature = last_group.get_first_combinable_ligature()
                assert ligature is not None
                group_to_add = GlyphGroup(ligature, GenerateType.SPLIT)
                rand = self.random_number_generator.rand(100)
                if rand < self.morph_reused_probability:
                    group_to_add = self.replace_random_token(group_to_add, is_paragraph_initial)
                groups.append(group_to_add)
            else:
                morphed_group = self.try_to_delete_prefix(last_group)
                if morphed_group != last_group:
                    rand = self.random_number_generator.rand(100)
                    if rand < self.morph_reused_probability:
                        morphed_group = self.replace_random_token(morphed_group, is_paragraph_initial)
                    groups.append(morphed_group)

    def add_gallow(
        self,
        glyph_group: GlyphGroup,
        is_paragraph_initial: bool,
        is_line_initial: bool,
    ) -> GlyphGroup:
        raise NotImplementedError

    def remove_line_initial_glyph(self, glyph_group: GlyphGroup) -> GlyphGroup:
        raise NotImplementedError

    def replace_random_token(
        self,
        glyph_group: GlyphGroup,
        is_paragraph_initial: bool,
    ) -> GlyphGroup:
        raise NotImplementedError
