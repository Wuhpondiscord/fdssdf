from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .selfcitation import GenerateType, GlyphGroup, _java_hashmap_spread


class RandomLike(Protocol):
    def rand(self, maximum: int) -> int: ...


def _default_hashmap_capacity(size: int) -> int:
    """Final JDK 8+ capacity for a default-constructed HashMap with ``size`` keys."""

    if size <= 0:
        return 0
    capacity = 16
    while size > (capacity * 3) // 4:
        capacity *= 2
    return capacity


class JavaGlyphGroupMap:
    """Small exact model of upstream ``HashMap<GlyphGroup, Long>`` key iteration.

    ``ChooserHelper.chooseRandomly`` copies ``HashMap.keySet()`` to an ArrayList,
    so Python insertion-order dict semantics would change the generated text. For
    ordinary (non-treeified) JDK 8+ bins, key iteration is bucket order followed
    by linked-list order within each bucket. Resizes preserve relative order, so
    recomputing buckets at the current capacity from first-insertion order is
    equivalent.

    The reference generator is parity-tested layer by layer. If a future full
    reference run ever creates a treeified bucket, this class raises rather than
    silently using an ordering that has not yet been proved equivalent.
    """

    def __init__(self) -> None:
        self._insertion_order: list[GlyphGroup] = []
        self._values: dict[str, tuple[GlyphGroup, int]] = {}

    def __len__(self) -> int:
        return len(self._values)

    def contains_key(self, group: GlyphGroup) -> bool:
        return group.glyph_group in self._values

    def get(self, group: GlyphGroup, default: int | None = None) -> int | None:
        item = self._values.get(group.glyph_group)
        return default if item is None else item[1]

    def put(self, group: GlyphGroup, value: int) -> None:
        key = group.glyph_group
        if key not in self._values:
            self._insertion_order.append(group)
            self._values[key] = (group, int(value))
        else:
            original, _ = self._values[key]
            self._values[key] = (original, int(value))

    def increment(self, group: GlyphGroup) -> int:
        current = self.get(group)
        value = 1 if current is None else current + 1
        self.put(group, value)
        return value

    @property
    def capacity(self) -> int:
        return _default_hashmap_capacity(len(self))

    def key_set(self) -> list[GlyphGroup]:
        if not self._insertion_order:
            return []
        capacity = self.capacity
        buckets: list[list[GlyphGroup]] = [[] for _ in range(capacity)]
        for group in self._insertion_order:
            bucket = _java_hashmap_spread(group.glyph_group) & (capacity - 1)
            buckets[bucket].append(group)

        # HashMap treeifies only when a bin grows past eight nodes and the table
        # is already at least 64 slots. The linked-list order for subsequent tree
        # inserts is different enough that it must be ported explicitly if the
        # released reference run ever reaches this state.
        if capacity >= 64 and any(len(bucket) > 8 for bucket in buckets):
            raise NotImplementedError(
                "Java HashMap tree-bin iteration is not ported yet; add it before "
                "using this dictionary state in source selection"
            )
        return [group for bucket in buckets for group in bucket]


@dataclass
class ChooserStatisticsSnapshot:
    """Minimal statistics surface consumed by ``PageSourceGroupChooser``.

    The full ``StatisticHelper`` port will replace this adapter later. Keeping
    this surface small lets the chooser itself be executable-parity tested now.
    """

    valid_group_dictionary: JavaGlyphGroupMap = field(default_factory=JavaGlyphGroupMap)
    lines_in_page: int = 0
    suggestions_probability: int = 40
    suggested_groups: list[GlyphGroup] = field(default_factory=list)

    def has_suggested_groups(self) -> bool:
        return bool(self.suggested_groups)

    def suggest_groups(self) -> list[GlyphGroup]:
        return list(self.suggested_groups)

    def get_suggestions_probability(self) -> int:
        return self.suggestions_probability


def calc_line_position(
    actual_line: Sequence[GlyphGroup], source_line: Sequence[GlyphGroup]
) -> int:
    """Port of ``ChooserHelper.calcLinePosition``."""

    if not source_line:
        raise ValueError("source_line must not be empty")

    writing_pos = sum(group.length() + 1 for group in actual_line)
    source_pos = 0
    for index, group in enumerate(source_line):
        source_pos += group.length() + 1
        if writing_pos <= source_pos:
            return index
    return min(len(source_line) - 1, len(actual_line))


def remove_initial_gallows(groups: Sequence[GlyphGroup]) -> list[GlyphGroup]:
    """Port of ``ChooserHelper.removeInitialGallow`` without mutating the input."""

    result = list(groups)
    for index, group in enumerate(result):
        if group.starts_with_gallow():
            tokens = group.copy_tokens()
            del tokens[0]
            if tokens:
                result[index] = GlyphGroup(tokens, group.generate_type)
    return result


def choose_randomly(
    group_dictionary: JavaGlyphGroupMap, random_number_generator: RandomLike
) -> list[GlyphGroup]:
    """Port of ``ChooserHelper.chooseRandomly`` including two independent draws."""

    keys = group_dictionary.key_set()
    if not keys:
        raise ValueError("group_dictionary must not be empty")
    first = random_number_generator.rand(len(keys))
    second = random_number_generator.rand(len(keys))
    return [keys[first], keys[second]]


class PageSourceGroupChooser:
    """Executable-equivalent port of upstream ``PageSourceGroupChooser``."""

    def __init__(
        self,
        random_number_generator: RandomLike,
        *,
        same_position_probability: int = 28,
        paragraph_initial_probability: int = 70,
    ) -> None:
        self.random_number_generator = random_number_generator
        self.same_position_probability = int(same_position_probability)
        self.line_initial_same_position_probability = max(
            10, self.same_position_probability // 2
        )
        self.paragraph_initial_probability = int(paragraph_initial_probability)

    def set_random_number_generator(self, random_number_generator: RandomLike) -> None:
        self.random_number_generator = random_number_generator

    def choose_source_group(
        self,
        generated_lines: Sequence[Sequence[GlyphGroup]],
        paragraph_initial_lines: Sequence[Sequence[GlyphGroup]],
        actual_line: Sequence[GlyphGroup],
        statistics: ChooserStatisticsSnapshot,
        is_paragraph_initial_line: bool,
        is_line_initial: bool,
        initial_line_count: int,
    ) -> list[GlyphGroup]:
        mode = "LOCALY"

        if len(generated_lines) < initial_line_count:
            mode = "RANDOM"

        # Upstream consumes this draw unconditionally, even when the paragraph
        # condition cannot possibly become true.
        rand = self.random_number_generator.rand(100)
        if (
            is_paragraph_initial_line
            and len(paragraph_initial_lines) > 1
            and (
                is_line_initial
                or (
                    len(actual_line) > 1
                    and rand < self.paragraph_initial_probability
                )
            )
        ):
            mode = "PARAGRAPH_INITIAL"

        if statistics.has_suggested_groups():
            rand = self.random_number_generator.rand(100)
            if rand < statistics.get_suggestions_probability():
                mode = "SUGGESTION"

        if mode == "RANDOM":
            selected = choose_randomly(
                statistics.valid_group_dictionary, self.random_number_generator
            )
        elif mode == "LOCALY":
            selected = self._choose_from_page(
                generated_lines,
                actual_line,
                statistics.lines_in_page,
                is_line_initial,
            )
        elif mode == "PARAGRAPH_INITIAL":
            selected = self._choose_from_paragraph_initial_lines(
                paragraph_initial_lines
            )
        elif mode == "SUGGESTION":
            selected = self._choose_suggestion(statistics)
        else:  # pragma: no cover - mirrors Java's defensive default
            selected = choose_randomly(
                statistics.valid_group_dictionary, self.random_number_generator
            )

        return remove_initial_gallows(selected)

    def _choose_suggestion(
        self, statistics: ChooserStatisticsSnapshot
    ) -> list[GlyphGroup]:
        selected = statistics.suggest_groups()
        if len(selected) == 1:
            random_groups = choose_randomly(
                statistics.valid_group_dictionary, self.random_number_generator
            )
            selected.append(random_groups[0])
        return selected

    def _choose_from_page(
        self,
        generated_lines: Sequence[Sequence[GlyphGroup]],
        actual_line: Sequence[GlyphGroup],
        lines_in_page: int,
        is_line_initial: bool,
    ) -> list[GlyphGroup]:
        if not generated_lines:
            raise ValueError("generated_lines must not be empty in LOCALY mode")

        line = len(generated_lines) - (
            1 + self.random_number_generator.rand(max(2, int(lines_in_page)) - 2)
        )
        if line < 0:
            line = len(generated_lines) - 1
        source_line = generated_lines[line]
        if not source_line:
            raise ValueError("selected source line must not be empty")

        rand = self.random_number_generator.rand(100)
        probability = (
            self.line_initial_same_position_probability
            if is_line_initial
            else self.same_position_probability
        )
        if rand <= probability:
            pos = calc_line_position(actual_line, source_line)
        else:
            pos = self.random_number_generator.rand(len(source_line))

        selected = [source_line[pos]]
        if pos < len(source_line) - 1:
            selected.append(source_line[pos + 1])
        elif pos > 0:
            selected.append(source_line[pos - 1])
        else:
            selected.append(source_line[pos])
        return selected

    def _choose_from_paragraph_initial_lines(
        self, paragraph_initial_lines: Sequence[Sequence[GlyphGroup]]
    ) -> list[GlyphGroup]:
        if not paragraph_initial_lines:
            raise ValueError("paragraph_initial_lines must not be empty")
        line_index = self.random_number_generator.rand(len(paragraph_initial_lines))
        source_line = paragraph_initial_lines[line_index]
        if not source_line:
            raise ValueError("selected paragraph-initial line must not be empty")
        pos = self.random_number_generator.rand(len(source_line))

        selected = [source_line[pos]]
        if pos < len(source_line) - 1:
            selected.append(source_line[pos + 1])
        elif pos > 0:
            selected.append(source_line[pos - 1])
        else:
            selected.append(source_line[pos])
        return selected
