from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .selfcitation import (
    GenerateType,
    GlyphGroup,
    JavaRandom,
    SelfCitationConfig,
    canonical_selfcitation_config,
)
from .selfcitation_canfollow import CurveLineCanFollow
from .selfcitation_chooser import PageSourceGroupChooser
from .selfcitation_morph import replace_line_final_glyph
from .selfcitation_slim import SUBSTITUTION_MAP, SlimGroupMorpher, Substitution
from .selfcitation_statistics import StatisticHelper, SuggestMethod


class RandomLike(Protocol):
    def rand(self, maximum: int) -> int: ...


@dataclass(frozen=True)
class SelfCitationGenerationResult:
    """Generated text plus the pinned configuration used to create it."""

    lines: tuple[str, ...]
    config: SelfCitationConfig

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def provenance(self) -> dict[str, object]:
        return self.config.provenance


def search_shorter_glyph(token: str) -> Substitution | None:
    """Port of ``Glyph.searchShorterGlyph``.

    Upstream scans the normal substitution row in source order and keeps the
    first single-token substitution whose character length is strictly shorter
    than every candidate seen so far. It does not use the word-final table.
    """

    glyph_length = len(token)
    return_sub_length = glyph_length
    return_sub: Substitution | None = None
    if glyph_length > 1:
        substitutions = SUBSTITUTION_MAP.get(token)
        if substitutions is not None:
            for substitution in substitutions:
                if substitution.count() == 1:
                    substitution_length = len(substitution.get(0))
                    if substitution_length < return_sub_length:
                        return_sub_length = substitution_length
                        return_sub = substitution
    return return_sub


def _split_initial_lines(value: str) -> tuple[str, ...]:
    """Match the configured ``#`` paragraph-line separator used upstream."""

    # Java String.split drops trailing empty fields when no explicit limit is
    # supplied. Preserve interior empty fields but remove trailing ones.
    parts = value.split("#")
    while parts and parts[-1] == "":
        parts.pop()
    return tuple(parts) if parts else ("",)


class SelfCitationTextGenerator:
    """Released self-citation text generator orchestration port.

    Chooser, morpher and statistics share one RNG object exactly as they do in
    the Java executable. The canonical released configuration uses the pseudo
    generator, for which ``JavaRandom`` provides bit-for-bit ``java.util.Random``
    behavior. A caller can inject another ``rand(max)`` implementation for
    controlled tests or for a separately verified real-random mode.
    """

    def __init__(
        self,
        config: SelfCitationConfig | None = None,
        *,
        random_number_generator: RandomLike | None = None,
        initial_lines: Sequence[str] | None = None,
    ) -> None:
        self.config = config or canonical_selfcitation_config()
        self.config.validate()

        if random_number_generator is None:
            if self.config.random_mode != "pseudo":
                raise NotImplementedError(
                    "the released real-random generator has not been parity-ported; "
                    "inject a verified rand(max) implementation explicitly"
                )
            random_number_generator = JavaRandom(self.config.random_seed)
        self.random_number_generator = random_number_generator

        self.initial_lines = (
            tuple(str(line) for line in initial_lines)
            if initial_lines is not None
            else _split_initial_lines(self.config.initial_line)
        )

        self.can_follow = CurveLineCanFollow(
            use_word_final_substitutions=self.config.use_word_final_substitutions
        )
        self.statistics = StatisticHelper(
            self.config.currier_type,
            self.random_number_generator,
            self.can_follow,
            SuggestMethod(self.config.suggestions),
            self.config.suggestions_probability,
            i_min_percentage=self.config.i_type_min_percentage,
            ol_min_percentage=self.config.ol_type_min_percentage,
            dy_min_percentage=self.config.dy_type_min_percentage,
        )
        self.source_group_chooser = PageSourceGroupChooser(
            self.random_number_generator,
            same_position_probability=self.config.same_position_probability,
        )
        self.group_morpher = SlimGroupMorpher(
            random_number_generator=self.random_number_generator,
            can_follow=self.can_follow,
            method_can_follow_error_rate=self.config.can_follow_error_rate,
            reuse_last_probability=self.config.reuse_last_probability,
            currier_type=self.config.currier_type,
        )
        self.group_morpher.set_probabilities(
            self.config.add_remove_probability,
            self.config.combine_split_probability,
        )

        self.generated_text: list[str] = []
        self.line_arrays: list[list[GlyphGroup]] = []
        self.paragraph_initial_line_arrays: list[list[GlyphGroup]] = []
        self._init_statistics()

    def _init_statistics(self) -> None:
        self.generated_text.clear()
        self.line_arrays.clear()
        self.paragraph_initial_line_arrays.clear()

        word_count = 0
        for line in self.initial_lines:
            text_array = line.split(" ")
            word_count += len(text_array)
            for group in text_array:
                glyph_group = GlyphGroup(group, GenerateType.INITIAL)
                self.statistics.remember(glyph_group)
            if len(line) > self.config.max_line_length:
                self.generated_text.append(self.trim_initial_line(text_array))
            else:
                self.generated_text.append(line)

        if word_count < 6:
            self.statistics.remember(GlyphGroup("daiin", GenerateType.INITIAL))
            self.statistics.remember(GlyphGroup("ol", GenerateType.INITIAL))
            if self.config.currier_type == "A":
                self.statistics.remember(GlyphGroup("cheody", GenerateType.INITIAL))
            else:
                self.statistics.remember(GlyphGroup("chedy", GenerateType.INITIAL))

    def trim_initial_line(self, text_array: Sequence[str]) -> str:
        """Port of the upstream initial-line trimming quirk."""

        new_initial_line = ""
        for group in text_array:
            initial_line_length = len(new_initial_line)
            if initial_line_length < self.config.max_line_length:
                if initial_line_length == 0:
                    new_initial_line = group
                else:
                    new_initial_line = new_initial_line + " " + group
        return new_initial_line

    def generate(self, lines_to_create: int | None = None) -> SelfCitationGenerationResult:
        target = self.config.lines_to_create if lines_to_create is None else int(lines_to_create)
        if target < len(self.generated_text):
            return SelfCitationGenerationResult(tuple(self.generated_text[:target]), self.config)
        if len(self.generated_text) < target:
            self._generate_text(target, len(self.initial_lines))
        return SelfCitationGenerationResult(tuple(self.generated_text[:target]), self.config)

    def _generate_text(self, lines_to_create: int, initial_line_count: int) -> None:
        self.statistics.lines_in_paragraph = len(self.generated_text)
        self.statistics.lines_in_page = len(self.generated_text)

        for line_index in range(self.statistics.lines_in_paragraph, lines_to_create):
            is_paragraph_initial = self.statistics.lines_in_paragraph == 0
            is_paragraph_final = False

            self.statistics.new_line()
            if (
                self.statistics.lines_in_page == self.config.lines_per_page
                or line_index == lines_to_create - 1
            ):
                is_paragraph_final = True
                self.statistics.new_page()
            elif (
                self.statistics.lines_in_page < self.config.lines_per_page - 2
                and self.statistics.lines_in_paragraph > 3
                and line_index < lines_to_create - 2
            ):
                rand = self.random_number_generator.rand(100)
                if rand < self.statistics.lines_in_paragraph * 10:
                    is_paragraph_final = True
                    self.statistics.new_paragraph()

            self.generated_text.append(
                self.generate_line(
                    is_paragraph_initial,
                    is_paragraph_final,
                    initial_line_count,
                )
            )

    def generate_line(
        self,
        is_paragraph_initial: bool,
        is_paragraph_final: bool,
        initial_line_count: int,
    ) -> str:
        is_line_initial = True
        line = ""

        max_line_length = self.config.max_line_length
        if is_paragraph_final:
            max_line_length = self.random_number_generator.rand(max_line_length)
            if max_line_length < self.config.min_line_length:
                max_line_length = self.config.min_line_length

        count = 0
        tries = 0
        glyph_group_list: list[GlyphGroup] = []
        # Upstream declares lastSourceGroups but never assigns to it. Keeping the
        # same inert state preserves the released behavior rather than activating
        # the nearby comment's intended duplicate-source suppression.
        last_source_groups: list[GlyphGroup] | None = None
        last_generated_group: GlyphGroup | None = None

        while len(line) + max(tries - 3, 0) < max_line_length:
            count += 1
            available_place_in_line = max_line_length - len(line)

            source_groups = self.source_group_chooser.choose_source_group(
                self.line_arrays,
                self.paragraph_initial_line_arrays,
                glyph_group_list,
                self.statistics,
                is_paragraph_initial,
                is_line_initial,
                initial_line_count,
            )

            use_source_groups = False
            if source_groups[0].get_token_count() > 5:
                rand = self.random_number_generator.rand(4)
                if source_groups[0].length() < 4 + rand:
                    use_source_groups = True
            else:
                use_source_groups = True

            if (
                use_source_groups
                and last_source_groups is not None
                and last_source_groups[0] == source_groups[0]
            ):
                use_source_groups = False

            if (
                self.config.combined_dismiss_as_source_probability > 0
                and use_source_groups
                and source_groups[0].generate_type == GenerateType.COMBINE
            ):
                if source_groups[1].generate_type == GenerateType.COMBINE:
                    use_source_groups = False
                else:
                    rand = self.random_number_generator.rand(100)
                    if rand < self.config.combined_dismiss_as_source_probability:
                        use_source_groups = False

            force_usage = False
            if ((not use_source_groups and count > 100) or count > 105):
                source_groups = self.statistics.suggest_groups_force(count - 100)
                use_source_groups = True
                if count > 110:
                    force_usage = True
            if count > 130:
                raise RuntimeError(
                    "self-citation generator exceeded the upstream 130-attempt guard"
                )

            if use_source_groups:
                morphed_groups = self.group_morpher.morph_group(
                    source_groups,
                    last_generated_group,
                    is_paragraph_initial,
                    is_line_initial,
                )

                if morphed_groups:
                    first_group = morphed_groups[0]
                    use_morphed_groups = self.can_follow.has_valid_start_glyph(first_group)
                    equal_last_group = bool(glyph_group_list) and (
                        first_group == glyph_group_list[-1]
                    )
                    if equal_last_group:
                        rand = self.random_number_generator.rand(100)
                        use_morphed_groups = rand < 50
                    else:
                        is_type_i = first_group.is_type_i()
                        is_type_dy = first_group.is_type_dy()
                        is_type_ol = first_group.is_type_ol()
                        if (
                            is_type_i
                            and self.statistics.repeated_tokens_i
                            > self.config.max_repeat_count
                        ):
                            use_morphed_groups = False
                        if (
                            is_type_dy
                            and self.statistics.repeated_tokens_dy
                            > self.config.max_repeat_count
                        ):
                            use_morphed_groups = False
                        if (
                            is_type_ol
                            and self.statistics.repeated_tokens_ol
                            > self.config.max_repeat_count
                        ):
                            use_morphed_groups = False
                        if (
                            not is_type_i
                            and not is_type_dy
                            and not is_type_ol
                            and self.statistics.repeated_tokens_unknown
                            > self.config.max_repeat_count
                        ):
                            use_morphed_groups = False

                    if use_morphed_groups or force_usage:
                        count = 0
                        for group_index, morphed_group in enumerate(morphed_groups):
                            if available_place_in_line - morphed_group.length() <= 0:
                                morphed_group = self.try_to_trim(
                                    morphed_group,
                                    available_place_in_line,
                                )

                            if available_place_in_line - morphed_group.length() >= 0:
                                if group_index == 0:
                                    if is_line_initial:
                                        is_line_initial = False
                                    else:
                                        line += " "
                                        available_place_in_line -= 1
                                else:
                                    line += " "
                                    available_place_in_line -= 1

                                glyph_group_list.append(morphed_group)
                                self.statistics.remember(morphed_group)
                                last_generated_group = morphed_group
                                line += morphed_group.glyph_group
                                available_place_in_line -= morphed_group.length()
                                tries = 0
                            else:
                                tries += 1

        self.line_arrays.append(glyph_group_list)
        if is_paragraph_initial:
            self.paragraph_initial_line_arrays.append(glyph_group_list)
        return line

    def try_to_trim(self, glyph_group: GlyphGroup, available_length: int) -> GlyphGroup:
        """Port of ``SelfCitationTextGenerator.tryToTrim``."""

        length = glyph_group.length()
        changed = False
        new_tokens = glyph_group.copy_tokens()

        last_token = new_tokens[-1]
        new_token = replace_line_final_glyph(last_token)
        if last_token != new_token:
            length -= len(last_token)
            length += len(new_token)
            new_tokens[-1] = new_token
            changed = True

        for index in range(len(new_tokens)):
            if available_length < length:
                token = new_tokens[index]
                substitution = search_shorter_glyph(token)
                if substitution is not None:
                    subst = substitution.first()
                    new_tokens[index] = subst
                    length -= len(token)
                    length += len(subst)
                    changed = True

        while available_length < length and len(new_tokens) > 2:
            first_token = new_tokens.pop(0)
            length -= len(first_token)
            changed = True

        if changed:
            return GlyphGroup(new_tokens, GenerateType.SHORTEN)
        return glyph_group


def generate_selfcitation(
    config: SelfCitationConfig | None = None,
    *,
    lines_to_create: int | None = None,
    random_number_generator: RandomLike | None = None,
) -> SelfCitationGenerationResult:
    """Generate released self-citation pseudotext in memory."""

    generator = SelfCitationTextGenerator(
        config,
        random_number_generator=random_number_generator,
    )
    return generator.generate(lines_to_create)


def generate_selfcitation_text(
    config: SelfCitationConfig | None = None,
    *,
    lines_to_create: int | None = None,
    random_number_generator: RandomLike | None = None,
) -> str:
    """Convenience wrapper returning newline-separated generated text."""

    return generate_selfcitation(
        config,
        lines_to_create=lines_to_create,
        random_number_generator=random_number_generator,
    ).text
