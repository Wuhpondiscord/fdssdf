from __future__ import annotations

from typing import Sequence

from voynich_lab.selfcitation import GenerateType, GlyphGroup
from voynich_lab.selfcitation_morph import (
    ALL_COMBINABLE_LIGATURES,
    PREFIX_GLYPHS,
    BaseGroupMorpher,
    Substitution,
    is_combinable_ligature,
    is_gallow,
    random_gallow,
)


def _subs(*entries: tuple[Sequence[str] | str, int]) -> tuple[Substitution, ...]:
    result: list[Substitution] = []
    for tokens, probability in entries:
        if isinstance(tokens, str):
            tokens = (tokens,)
        result.append(Substitution(tokens, probability))
    return tuple(result)


# Literal cumulative substitution thresholds from Glyph.java at the pinned
# release commit. These are deliberately not normalized into independent
# weights because SlimGroupMorpher.choose() compares a rand(100) draw against
# each cumulative ``probability`` in source order.
SUBSTITUTION_MAP: dict[str, tuple[Substitution, ...]] = {
    "k": _subs(("t", 77), ("p", 94), ("f", 100)),
    "t": _subs(("k", 84), ("p", 96), ("f", 100)),
    "p": _subs(("k", 59), ("t", 97), ("f", 100)),
    "f": _subs(("k", 56), ("t", 92), ("p", 100)),
    "in": _subs(("n", 8), ("iin", 84), ("iiin", 87), ("ir", 97), ("iir", 98), ("iis", 99), ("il", 100)),
    "iin": _subs(("n", 13), ("in", 70), ("iiin", 74), ("ir", 92), ("iir", 97), ("is", 99), ("il", 100)),
    "iiin": _subs(("n", 6), ("in", 31), ("iin", 90), ("ir", 98), ("iir", 100)),
    "ir": _subs(("r", 6), ("in", 33), ("iin", 95), ("iiin", 97), ("iir", 99), ("iiir", 100)),
    "iir": _subs(("r", 6), ("in", 30), ("iin", 89), ("iiin", 91), ("ir", 99), ("iiir", 100)),
    "iiir": _subs(("ir", 90), ("iir", 100)),
    "is": _subs(("in", 50), ("iis", 100)),
    "iis": _subs(("iin", 50), ("is", 100)),
    "il": _subs(("in", 50), ("iil", 100)),
    "iil": _subs(("iin", 50), ("il", 99), ("iiil", 100)),
    "iiil": _subs(("il", 66), ("iil", 100)),
    "im": _subs(("in", 30), ("iin", 100)),
    "iim": _subs(("in", 30), ("iin", 100)),
    "iiim": _subs(("in", 30), ("iin", 100)),
    "om": _subs(("ol", 45), ("or", 94), ("og", 98), ("omg", 100)),
    "am": _subs(("al", 45), ("ar", 94), ("ag", 98), ("amg", 100)),
    "og": _subs(("or", 30), ("al", 63), ("ar", 100)),
    "ag": _subs(("ol", 48), ("or", 72), ("ar", 100)),
    "ol": _subs(("or", 30), ("al", 63), ("ar", 100)),
    "or": _subs(("ol", 47), ("al", 73), ("ar", 100)),
    "al": _subs(("ol", 48), ("or", 72), ("ar", 100)),
    "ar": _subs(("ol", 49), ("or", 73), ("al", 100)),
    "e": _subs(("e", 50), ("ee", 99), ("eee", 100)),
    "ee": _subs(("ch", 40), (("ch", "e"), 50), ("e", 98), ("eee", 100)),
    "eee": _subs(("ch", 10), (("ch", "e"), 20), ("ee", 65), ("e", 100)),
    "ch": _subs(("ee", 10), (("ch", "e"), 20), ("sh", 90), ("ckh", 97), ("cth", 100)),
    "sh": _subs(("ee", 10), ("ch", 90), ("ckh", 97), ("cth", 100)),
    "ckh": _subs(("cth", 30), (("k", "ch"), 50), (("t", "ch"), 70), ("eke", 72), ("ete", 74), ("cph", 78), ("ch", 100)),
    "cth": _subs(("ckh", 30), (("k", "ch"), 50), (("t", "ch"), 70), ("eke", 72), ("ete", 74), ("cph", 78), ("cfh", 80), ("ch", 100)),
    "cs": _subs(("sh", 100)),
    "ckhh": _subs(("ckh", 100)),
    "cthh": _subs(("cth", 100)),
    "ikh": _subs(("ckh", 100)),
    "ith": _subs(("cth", 100)),
    "iph": _subs(("cph", 100)),
    "ifh": _subs(("cfh", 100)),
    "eke": _subs(("ckh", 30), ("cth", 50), (("k", "ee"), 56), (("t", "ee"), 60), ("ete", 65), ("ee", 100)),
    "ete": _subs(("ckh", 30), ("cth", 50), (("k", "ee"), 56), (("t", "ee"), 60), ("eke", 65), ("ee", 100)),
    "cph": _subs(("ckh", 40), ("cth", 75), ("cfh", 80), ("ch", 100)),
    "cfh": _subs(("ckh", 40), ("cth", 75), ("cph", 80), ("ch", 100)),
    "y": _subs(("o", 100)),
    "o": _subs(("y", 100)),
    "n": _subs(("r", 50), ("in", 63), ("iin", 100)),
    "l": _subs(("r", 100)),
    "r": _subs(("r", 50), ("s", 100)),
    "g": _subs(("m", 100)),
    "s": _subs(("r", 75), ("d", 100)),
    "d": _subs(("d", 90), ("s", 100)),
    "a": _subs(("a", 98), ("o", 100)),
    "qo": _subs(("o", 80), ("y", 100)),
}

FINAL_SUBSTITUTION_MAP: dict[str, tuple[Substitution, ...]] = {
    "om": _subs(("o", 8), ("y", 30), ("ol", 75), ("or", 100)),
    "am": _subs(("o", 4), ("y", 30), ("al", 70), ("ar", 100)),
    "og": _subs(("o", 8), ("y", 30), ("or", 55), ("al", 80), ("ar", 100)),
    "ag": _subs(("o", 4), ("y", 30), ("ol", 65), ("or", 80), ("ar", 100)),
    "ol": _subs(("o", 8), ("y", 30), ("or", 54), ("al", 81), ("ar", 100)),
    "or": _subs(("o", 8), ("y", 30), ("ol", 63), ("al", 81), ("ar", 100)),
    "al": _subs(("o", 4), ("y", 30), ("ol", 64), ("or", 81), ("ar", 100)),
    "ar": _subs(("o", 4), ("y", 30), ("ol", 64), ("or", 82), ("al", 100)),
    "y": _subs(("o", 20), ("ol", 44), ("or", 60), ("al", 75), ("ar", 100)),
    "o": _subs(("y", 20), ("ol", 44), ("or", 60), ("al", 75), ("ar", 100)),
    "d": _subs(("dy", 70), ("d", 100)),
    "dy": _subs(("d", 10), ("dy", 100)),
}


def similar_glyph(token: str, is_final_token: bool) -> tuple[Substitution, ...] | None:
    if is_final_token and token in FINAL_SUBSTITUTION_MAP:
        return FINAL_SUBSTITUTION_MAP[token]
    return SUBSTITUTION_MAP.get(token)


def _tokens_contain_gallow(tokens: Sequence[str]) -> bool:
    return any(any(gallow in token for gallow in ("k", "t", "p", "f")) for token in tokens)


class SlimGroupMorpher(BaseGroupMorpher):
    """Exact CurveLine-mode port of the released Java ``SlimGroupMorpher``."""

    def morph_group(
        self,
        source_groups: Sequence[GlyphGroup],
        previous_group: GlyphGroup | None,
        is_paragraph_initial: bool,
        is_line_initial: bool,
    ) -> list[GlyphGroup]:
        source_group = source_groups[0]
        length = source_group.length()
        result: list[GlyphGroup] = []

        rand = 0 if is_paragraph_initial and is_line_initial else self.random_number_generator.rand(100)
        if rand <= self.add_remove_probability and source_group.generate_type != GenerateType.COMBINE:
            method = "add_remove"
        elif rand <= self.combine_split_probability + self.add_remove_probability:
            method = "combine_split"
        else:
            method = "replace"

        if method == "add_remove":
            morphed = (
                self.add_random_glyph(source_group, previous_group, is_paragraph_initial, is_line_initial)
                if length < 6
                else source_group
            )
            if morphed != source_group:
                result.append(morphed)
            else:
                morphed = self.try_to_delete_prefix(source_group)
                if morphed != source_group:
                    result.append(morphed)

        elif method == "combine_split":
            rand = self.random_number_generator.rand(100)
            if source_group.generate_type != GenerateType.COMBINE:
                if length < 6:
                    combine_groups = length <= 2 or rand < 96
                else:
                    combine_groups = rand < 4 and length <= 8
            else:
                combine_groups = False

            if combine_groups and len(source_groups) > 1:
                morphed = self.combine_glyphgroups(source_groups)
                if morphed != source_group:
                    result.append(morphed)
            else:
                result.extend(self.split_glyphgroup(source_group))

        else:
            rand = self.random_number_generator.rand(100)
            if rand <= 30:
                temp = self.replace_random_token(source_group, is_paragraph_initial)
                if temp == source_group:
                    temp = self.replace_random_token(source_group, is_paragraph_initial)
                if temp != source_group:
                    result.append(temp)
            elif rand <= 40:
                temp = self.replace_random_token(source_group, is_paragraph_initial)
                temp2 = self.replace_random_token(temp, is_paragraph_initial)
                if temp2 == source_group:
                    temp2 = temp
                temp3 = self.replace_random_token(temp2, is_paragraph_initial)
                if temp3 == source_group:
                    temp3 = temp2
                if temp3 != source_group:
                    result.append(temp3)
            else:
                temp = self.replace_random_token(source_group, is_paragraph_initial)
                temp2 = self.replace_random_token(temp, is_paragraph_initial)
                if temp2 == source_group:
                    temp2 = temp
                if temp2 != source_group:
                    result.append(temp2)

        if not result:
            return result

        self.handle_gallows(is_paragraph_initial, is_line_initial, result)
        if self.reuse_last_probability > 0:
            self.reuse_last_morphed_group(is_paragraph_initial, result)
        return result

    def add_random_glyph(
        self,
        glyph_group: GlyphGroup,
        previous_group: GlyphGroup | None,
        is_paragraph_initial: bool,
        is_line_initial: bool,
    ) -> GlyphGroup:
        rand = self.random_number_generator.rand(100)
        if is_paragraph_initial:
            if rand < 80:
                return self.add_gallow(glyph_group, is_paragraph_initial, is_line_initial)
            return self._try_to_add_prefix(glyph_group, previous_group, is_line_initial)
        if rand < 8:
            return self.add_gallow(glyph_group, is_paragraph_initial, is_line_initial)
        return self._try_to_add_prefix(glyph_group, previous_group, is_line_initial)

    def _determine_generate_type(self, suggested_type: GenerateType, glyph_group: GlyphGroup) -> GenerateType:
        if glyph_group.generate_type == GenerateType.COMBINE:
            return glyph_group.generate_type
        return suggested_type

    def _choose_prefix(
        self,
        attempt: int,
        previous_group: GlyphGroup | None,
        prefixes_already_tried: Sequence[str],
    ) -> str:
        rand = self.random_number_generator.rand(100)
        if attempt == 0 and previous_group is not None and previous_group.ends_with_dy_token() and rand < 90:
            return "q"
        if rand < 5 and "l" not in prefixes_already_tried:
            return "l"
        if rand < 34 and "o" not in prefixes_already_tried:
            return "o"
        if rand < 40 and "y" not in prefixes_already_tried:
            return "y"
        if rand < 61 and "ch" not in prefixes_already_tried:
            return "ch"
        if rand < 72 and "sh" not in prefixes_already_tried:
            return "sh"
        if rand < 88 and "q" not in prefixes_already_tried:
            return "q"
        if rand < 99:
            return "d"
        if attempt == 0:
            return "x"
        return "o"

    def _try_to_add_prefix(
        self,
        glyph_group: GlyphGroup,
        previous_group: GlyphGroup | None,
        is_line_initial: bool,
    ) -> GlyphGroup:
        prefixes_already_tried: list[str] = []
        for attempt in range(7):
            prefix = self._choose_prefix(attempt, previous_group, prefixes_already_tried)
            prefixes_already_tried.append(prefix)
            if prefix == "q":
                start_token = glyph_group.get_token(0)
                if start_token in ("o", "y"):
                    new_tokens = glyph_group.copy_tokens()
                    new_tokens[0] = "qo"
                    return GlyphGroup(
                        new_tokens,
                        self._determine_generate_type(GenerateType.ADD, glyph_group),
                    )
            elif prefix == "x":
                allowed_start_glyphs = PREFIX_GLYPHS.get(prefix)
                if allowed_start_glyphs is not None:
                    for glyph in allowed_start_glyphs:
                        if glyph_group.has_prefix(glyph):
                            new_tokens = glyph_group.copy_tokens()
                            new_tokens.insert(0, prefix)
                            return GlyphGroup(
                                new_tokens,
                                self._determine_generate_type(GenerateType.ADD, glyph_group),
                            )
            else:
                allowed_start_glyphs = PREFIX_GLYPHS.get(prefix)
                if allowed_start_glyphs is not None:
                    for glyph in allowed_start_glyphs:
                        if glyph_group.has_prefix(glyph):
                            length = glyph_group.get_token_count()
                            new_tokens = glyph_group.copy_tokens()
                            start_token = glyph_group.get_token(0)
                            if length > 2 and start_token in ("d", "ch", "sh"):
                                gallow = random_gallow(False, self.random_number_generator)
                                if (
                                    self.can_follow_each_other_before(gallow, new_tokens[1])
                                    and self.can_follow_each_other_after(prefix, gallow)
                                ):
                                    rand = self.random_number_generator.rand(100)
                                    if rand < 70:
                                        new_tokens[0] = gallow
                            new_tokens.insert(0, prefix)
                            return GlyphGroup(
                                new_tokens,
                                self._determine_generate_type(GenerateType.ADD, glyph_group),
                            )
        return glyph_group

    def add_gallow(
        self,
        glyph_group: GlyphGroup,
        is_paragraph_initial: bool,
        is_line_initial: bool,
    ) -> GlyphGroup:
        length = glyph_group.get_token_count()
        if is_paragraph_initial and is_line_initial:
            gallow = random_gallow(is_paragraph_initial, self.random_number_generator)
            return_group = self._try_to_place_gallow(gallow, glyph_group, 0)
            if return_group.get_token(0) != gallow:
                new_tokens = return_group.copy_tokens()
                if self.can_follow_each_other_before("o", return_group.glyph_group):
                    new_tokens.insert(0, "o")
                elif self.can_follow_each_other_before("a", return_group.glyph_group):
                    new_tokens.insert(0, "a")
                new_tokens.insert(0, gallow)
                return_group = GlyphGroup(new_tokens, return_group.generate_type)
            return return_group

        if length > 1:
            do_it = True
            if glyph_group.contains_gallow():
                rand = self.random_number_generator.rand(100)
                if rand < 90:
                    do_it = False
            if do_it:
                for _ in range(5):
                    if is_paragraph_initial:
                        pos = self.random_number_generator.rand(length)
                    elif length == 2:
                        pos = 1
                    else:
                        pos = self.random_number_generator.rand(length - 2) + 1
                    gallow = random_gallow(is_paragraph_initial, self.random_number_generator)
                    return_group = self._try_to_place_gallow(gallow, glyph_group, pos)
                    if return_group != glyph_group:
                        return return_group
        return glyph_group

    def _try_to_place_gallow(self, gallow: str, glyph_group: GlyphGroup, pos: int) -> GlyphGroup:
        token = glyph_group.get_token(pos)
        if is_gallow(token):
            new_tokens = glyph_group.copy_tokens()
            new_tokens[pos] = gallow
            return GlyphGroup(new_tokens, glyph_group.generate_type)

        last = "" if pos == 0 else glyph_group.get_token(pos - 1)
        next_token = "" if pos == glyph_group.get_token_count() else glyph_group.get_token(pos)
        last_ok = self.can_follow_each_other_after(last, gallow)
        next_ok = self.can_follow_each_other_before(gallow, next_token)
        if last_ok and next_ok:
            new_tokens = glyph_group.copy_tokens()
            new_tokens.insert(pos, gallow)
            return GlyphGroup(new_tokens, glyph_group.generate_type)

        new_tokens = glyph_group.copy_tokens()
        if last_ok:
            alternative_suffix = glyph_group.get_token(pos + 1) if glyph_group.get_token_count() > pos + 1 else "-"
            if glyph_group.get_token_count() > pos + 1:
                new_tokens[pos] = gallow
                candidate = GlyphGroup(new_tokens, glyph_group.generate_type)
                if self.can_follow_each_other_before(gallow, alternative_suffix):
                    return candidate
                return glyph_group
            return glyph_group
        if next_ok:
            next_to_last_glyph = glyph_group.get_token(pos - 2) if pos - 2 > 0 else "-"
            if pos - 2 > 0:
                new_tokens[pos - 1] = gallow
                candidate = GlyphGroup(new_tokens, glyph_group.generate_type)
                if self.can_follow_each_other_after(next_to_last_glyph, gallow):
                    return candidate
                return glyph_group
            return glyph_group
        return glyph_group

    @staticmethod
    def calc_length(token_list: Sequence[str]) -> int:
        return sum(len(token) for token in token_list)

    def choose_subgroups_for_combine(self, glyph_group: GlyphGroup, is_first_group: bool) -> list[str]:
        token_list: list[str] = []
        if (
            (is_first_group and glyph_group.get_token_count() <= 2)
            or (not is_first_group and glyph_group.get_token_count() <= 3)
        ):
            token_list = glyph_group.copy_tokens()
        else:
            pos = self.calc_split_position(glyph_group)
            if pos >= 0:
                split_list = self.split_at_position(glyph_group, pos)
                if split_list:
                    if is_first_group or len(split_list) == 1:
                        token_list = split_list[0].copy_tokens()
                    else:
                        token_list = split_list[1].copy_tokens()
        return token_list

    def combine_glyphgroups(self, source_groups: Sequence[GlyphGroup]) -> GlyphGroup:
        glyph_group1 = source_groups[0]
        glyph_group2 = source_groups[1]
        new_tokens1 = self.choose_subgroups_for_combine(glyph_group1, True)
        if new_tokens1 and self.calc_length(new_tokens1) > 1:
            last_token1 = new_tokens1[-1]
            new_tokens2 = self.choose_subgroups_for_combine(glyph_group2, False)
            tokens_to_remove = ALL_COMBINABLE_LIGATURES.get(last_token1)
            if tokens_to_remove is not None:
                for token in tokens_to_remove:
                    if new_tokens2 and token == new_tokens2[0]:
                        new_tokens2.pop(0)

            if new_tokens2 and self.calc_length(new_tokens2) > 1:
                new_tokens1.extend(new_tokens2)
                if self.calc_length(new_tokens1) < 9:
                    candidate = GlyphGroup(new_tokens1, GenerateType.COMBINE)
                    if (
                        is_combinable_ligature(last_token1)
                        or self.can_follow_each_other_before(last_token1, new_tokens2[0])
                    ) and self.can_follow_each_other_after(new_tokens2[-1], ""):
                        return candidate
        return super().combine_glyphgroups(source_groups)

    def remove_line_initial_glyph(self, glyph_group: GlyphGroup) -> GlyphGroup:
        length = glyph_group.get_token_count()
        if glyph_group.starts_with_line_initial_glyph() and length > 2:
            new_tokens = glyph_group.copy_tokens()
            new_tokens.pop(0)
            first = new_tokens[0]
            second = new_tokens[1]
            if is_gallow(first):
                if second.startswith("a"):
                    new_tokens[0] = "d"
                if second.startswith("e"):
                    new_tokens[0] = "ch"
            return GlyphGroup(new_tokens, glyph_group.generate_type)
        return glyph_group

    def replace_random_token(self, glyph_group: GlyphGroup, is_paragraph_initial: bool) -> GlyphGroup:
        positions: list[int] = []
        for index in range(glyph_group.get_token_count()):
            positions.append(index)
            positions.append(index)

        attempt = 0
        # Java's for-loop compares j against the *shrinking* list size after
        # every removal, so it performs roughly token_count attempts, not 2x.
        while attempt < len(positions):
            pos_to_try = self.random_number_generator.rand(len(positions)) if len(positions) > 1 else 0
            pos = glyph_group.get_token_count() - 1 if not positions else positions.pop(pos_to_try)
            subgroup = glyph_group.get_token(pos)
            possible = similar_glyph(
                subgroup,
                attempt > 0 and pos == glyph_group.get_token_count() - 1,
            )
            if possible is not None:
                substitute = self._choose_substitution(possible)
                new_tokens = glyph_group.copy_tokens()
                next_token = new_tokens[pos + 1] if len(new_tokens) > pos + 1 else ""
                remove_next = _tokens_contain_gallow(substitute.tokens) and is_gallow(next_token)
                if self.is_replacable(substitute, glyph_group, pos, remove_next):
                    new_tokens.pop(pos)
                    if remove_next:
                        new_tokens.pop(pos)
                    for index, token in enumerate(substitute.tokens):
                        new_tokens.insert(pos + index, token)
                    generate_type = (
                        glyph_group.generate_type
                        if glyph_group.generate_type == GenerateType.COMBINE
                        else GenerateType.REPLACE
                    )
                    candidate = GlyphGroup(new_tokens, generate_type)
                    if candidate != glyph_group:
                        return candidate
            attempt += 1
        return glyph_group

    def _choose_substitution(self, substitutions: Sequence[Substitution]) -> Substitution:
        rand = self.random_number_generator.rand(100)
        for substitute in substitutions:
            if substitute.probability > rand:
                return substitute
        raise RuntimeError("maxProbability < 100")
