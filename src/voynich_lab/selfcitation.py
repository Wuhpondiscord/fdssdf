from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Sequence


SELFCITATION_UPSTREAM_REPOSITORY = "TorstenTimm/SelfCitationTextgenerator"
SELFCITATION_UPSTREAM_COMMIT = "a6ede2202dd7ad6285ce2c007bf22c2a0e7709b7"
SELFCITATION_FAMILY = "meaningless_pseudotext"
SELFCITATION_LICENSE = "MIT"


class JavaRandom:
    """Bit-for-bit compatible subset of ``java.util.Random`` used upstream.

    Timm/Schinner's pseudo RNG is a thin wrapper around ``Random.nextInt(max)``.
    This class implements the same 48-bit LCG and bounded rejection algorithm so
    later generator layers can preserve the original RNG call sequence exactly.
    """

    _MULTIPLIER = 0x5DEECE66D
    _ADDEND = 0xB
    _MASK = (1 << 48) - 1

    def __init__(self, seed: int):
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        self._seed = (int(seed) ^ self._MULTIPLIER) & self._MASK

    def _next(self, bits: int) -> int:
        if bits < 0 or bits > 32:
            raise ValueError("bits must be in [0, 32]")
        self._seed = (self._seed * self._MULTIPLIER + self._ADDEND) & self._MASK
        return self._seed >> (48 - bits)

    @staticmethod
    def _java_signed_int(value: int) -> int:
        value &= 0xFFFFFFFF
        return value if value < 0x80000000 else value - 0x100000000

    def next_int(self, bound: int | None = None) -> int:
        """Match ``java.util.Random.nextInt()`` / ``nextInt(bound)`` exactly."""

        if bound is None:
            return self._java_signed_int(self._next(32))
        bound = int(bound)
        if bound <= 0:
            raise ValueError("bound must be positive")

        # Java's power-of-two fast path.
        if (bound & -bound) == bound:
            return (bound * self._next(31)) >> 31

        # Java evaluates ``bits - val + (bound - 1)`` as a signed 32-bit int.
        # The overflow-sensitive condition is why a direct Python translation
        # without the signed-int conversion is not fully equivalent.
        while True:
            bits = self._next(31)
            val = bits % bound
            test = self._java_signed_int(bits - val + (bound - 1))
            if test >= 0:
                return val

    def rand(self, maximum: int) -> int:
        """Match the upstream ``I_RandomNumberGenerator.rand(max)`` wrapper."""

        maximum = int(maximum)
        if maximum == 0:
            return 0
        return self.next_int(maximum)


class GenerateType(str, Enum):
    """Upstream ``GlyphGroup.GENERATE_TYPE`` values."""

    INITIAL = "INITIAL"
    ADD = "ADD"
    DELETE = "DELETE"
    REPLACE = "REPLACE"
    COMBINE = "COMBINE"
    SPLIT = "SPLIT"
    SHORTEN = "SHORTEN"


# Literal insertion order from Glyph.java's ``ligatureStrings`` table at the
# pinned upstream commit. ``Glyph.determineStartLigature`` does not iterate this
# array directly: Converter constructs a java.util.HashMap and iterates keySet().
_LIGATURE_INSERTION_ORDER: tuple[str, ...] = (
    "ol",
    "or",
    "al",
    "ar",
    "dy",
    "qo",
    "ch",
    "sh",
    "cs",
    "eee",
    "ee",
    "cth",
    "cthh",
    "ckh",
    "ckhh",
    "cph",
    "cfh",
    "ith",
    "ikh",
    "iph",
    "ifh",
    "eke",
    "ete",
    "in",
    "iin",
    "iiin",
    "ir",
    "iir",
    "iiir",
    "is",
    "iis",
    "iiis",
    "il",
    "iil",
    "iiil",
    "im",
    "iim",
    "iiim",
    "om",
    "am",
    "og",
    "ag",
)

_GALLOW_GLYPHS = ("k", "t", "p", "f")
_LINE_INITIAL_GLYPHS = ("o", "y", "d", "s")
_OL_GLYPHS = ("ol", "al", "or", "ar")
_DY_GLYPH = "dy"


def java_string_hash_code(text: str) -> int:
    """Return Java ``String.hashCode()`` as a signed 32-bit integer."""

    value = 0
    for char in text:
        value = (31 * value + ord(char)) & 0xFFFFFFFF
    return value if value < 0x80000000 else value - 0x100000000


def _java_hashmap_spread(text: str) -> int:
    """Return the JDK 8+ HashMap spread hash for a String key."""

    value = java_string_hash_code(text) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


def _java_hashmap_bucket_scan_order(keys: Sequence[str], capacity: int) -> tuple[str, ...]:
    """Reproduce key iteration for these small, non-treeified upstream maps.

    JDK 8+ ``HashMap`` iteration walks buckets from index 0 upward and follows
    each bucket's linked-list order. The pinned tables here never treeify; their
    final capacities are known from the constructor size and resize threshold.
    """

    if capacity <= 0 or capacity & (capacity - 1):
        raise ValueError("capacity must be a positive power of two")
    buckets: list[list[str]] = [[] for _ in range(capacity)]
    for key in keys:
        buckets[_java_hashmap_spread(key) & (capacity - 1)].append(key)
    return tuple(key for bucket in buckets for key in bucket)


# new HashMap<>(42) allocates a 64-slot table on first insertion and has a
# threshold of 48, so the 42-entry ligature map never resizes.
_LIGATURE_SCAN_ORDER = _java_hashmap_bucket_scan_order(_LIGATURE_INSERTION_ORDER, 64)

# Converter constructs this four-key map with new HashMap<>(4). It resizes once
# from 4 to 8 slots after the fourth insertion; the final JDK 8+ bucket walk is
# therefore ar, al, or, ol (al/or share a bucket and retain insertion order).
_COMBINABLE_LIGATURE_SCAN_ORDER = ("ar", "al", "or", "ol")


def determine_start_ligature(glyph_group: str) -> str | None:
    """Match upstream ``Glyph.determineStartLigature`` exactly."""

    # Upstream special-cases eee so the ee entry cannot capture it first.
    if glyph_group.startswith("eee"):
        return "eee"
    for ligature in _LIGATURE_SCAN_ORDER:
        if glyph_group.startswith(ligature):
            return ligature
    return None


def tokenize_glyph_group(glyph_group: str) -> tuple[str, ...]:
    """Parse one EVA glyph group with the pinned Java ligature semantics."""

    glyph_group = str(glyph_group)
    tokens: list[str] = []
    pos = 0
    while pos < len(glyph_group):
        remainder = glyph_group[pos:]
        ligature = determine_start_ligature(remainder)
        if ligature is not None:
            tokens.append(ligature)
            pos += len(ligature)
        else:
            tokens.append(remainder[0])
            pos += 1
    return tuple(tokens)


class GlyphGroup:
    """Python port of the pinned upstream ``GlyphGroup`` value semantics."""

    def __init__(
        self,
        glyph_group: str | Iterable[str],
        generate_type: GenerateType | str = GenerateType.INITIAL,
    ) -> None:
        if isinstance(glyph_group, str):
            text = glyph_group
        else:
            text = "".join(str(token) for token in glyph_group)
        self.glyph_group = text
        self.generate_type = GenerateType(generate_type)
        # The Java list constructor joins tokens and then calls parse(), so it
        # does not preserve caller-supplied token boundaries either.
        self._tokens = tokenize_glyph_group(text)

    @property
    def tokens(self) -> tuple[str, ...]:
        return self._tokens

    def get_token_count(self) -> int:
        return len(self._tokens)

    def length(self) -> int:
        return len(self.glyph_group)

    def get_token(self, pos: int) -> str:
        return self._tokens[pos]

    def tokens_as_string(self) -> str:
        return "".join(f"{{{token}}}" for token in self._tokens)

    def copy_tokens(self) -> list[str]:
        return list(self._tokens)

    def starts_with_gallow(self) -> bool:
        return self.glyph_group.startswith(_GALLOW_GLYPHS)

    def starts_with_line_initial_glyph(self) -> bool:
        return self.glyph_group.startswith(_LINE_INITIAL_GLYPHS)

    def has_prefix(self, prefix: str) -> bool:
        return self._tokens[0] == prefix

    def contains_combinable_ligature(self) -> bool:
        return any(token in _OL_GLYPHS for token in self._tokens)

    def get_first_combinable_ligature(self) -> str | None:
        # Match the upstream HashMap keySet() scan, not token position order.
        for ligature in _COMBINABLE_LIGATURE_SCAN_ORDER:
            if ligature in self._tokens:
                return ligature
        return None

    def contains(self, search: str) -> bool:
        # Java checks token equality here, not substring containment.
        return search in self._tokens

    def contains_gallow(self) -> bool:
        return any(gallow in self.glyph_group for gallow in _GALLOW_GLYPHS)

    def is_type_i(self) -> bool:
        return "i" in self.glyph_group

    def is_type_ol(self) -> bool:
        return any(token in _OL_GLYPHS for token in self._tokens)

    def is_type_dy(self) -> bool:
        if _DY_GLYPH in self._tokens:
            return True
        last_char = self.glyph_group[-1]
        return last_char in {"y", "d"}

    def ends_with_dy_token(self) -> bool:
        return self._tokens[-1] == _DY_GLYPH

    def __hash__(self) -> int:
        return java_string_hash_code(self.glyph_group)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GlyphGroup) and self.glyph_group == other.glyph_group

    def __str__(self) -> str:
        return self.glyph_group

    def __repr__(self) -> str:
        return f"GlyphGroup({self.glyph_group!r}, {self.generate_type.value})"


@dataclass(frozen=True)
class SelfCitationConfig:
    """Canonical settings from the pinned released ``executable/conf.properties``.

    These values define the released reference generator, and intentionally
    override several fallback/default comments in the Java source where the
    executable configuration is more specific (notably 1,200 lines and the
    20/30/50 morph split).
    """

    lines_to_create: int = 1200
    max_line_length: int = 55
    min_line_length: int = 15
    lines_per_page: int = 29
    max_repeat_count: int = 3
    initial_line: str = (
        "pchal shal shorchdy okeor okain shedy pchedy qotchedy qotar ol lkar"
    )
    random_mode: str = "pseudo"
    random_seed: int = 19
    can_follow: str = "curveline"
    can_follow_error_rate: int = 0
    source_chooser: str = "page"
    same_position_probability: int = 28
    morph: str = "slim"
    add_remove_probability: int = 20
    combine_split_probability: int = 30
    replace_probability: int = 50
    reuse_last_probability: int = 10
    combined_dismiss_as_source_probability: int = 30
    use_word_final_substitutions: bool = True
    suggestions: str = "top"
    suggestions_probability: int = 40
    i_type_min_percentage: float = 0.20
    ol_type_min_percentage: float = 0.25
    dy_type_min_percentage: float = 0.25

    @property
    def currier_type(self) -> str:
        # This mirrors Config.java's released heuristic: an initial line
        # containing "ed" selects Currier B, otherwise A.
        return "B" if "ed" in self.initial_line else "A"

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "family": SELFCITATION_FAMILY,
            "source_repository": SELFCITATION_UPSTREAM_REPOSITORY,
            "upstream_commit": SELFCITATION_UPSTREAM_COMMIT,
            "license": SELFCITATION_LICENSE,
            "reference_config": "executable/conf.properties",
            "config": asdict(self),
        }

    def validate(self) -> None:
        if self.lines_to_create < 1:
            raise ValueError("lines_to_create must be >= 1")
        if self.max_line_length < 40:
            raise ValueError("max_line_length must be >= 40 to match Config.java")
        if self.min_line_length < 5:
            raise ValueError("min_line_length must be >= 5")
        if self.lines_per_page < 1:
            raise ValueError("lines_per_page must be >= 1")
        if not 1 <= self.max_repeat_count <= 5:
            raise ValueError("max_repeat_count must be in [1, 5]")
        if self.random_mode not in {"pseudo", "real"}:
            raise ValueError("random_mode must be 'pseudo' or 'real'")
        if self.can_follow_error_rate not in range(0, 11):
            raise ValueError("can_follow_error_rate must be in [0, 10]")
        for name, value in (
            ("same_position_probability", self.same_position_probability),
            ("reuse_last_probability", self.reuse_last_probability),
            ("combined_dismiss_as_source_probability", self.combined_dismiss_as_source_probability),
            ("suggestions_probability", self.suggestions_probability),
        ):
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100]")
        if self.add_remove_probability + self.combine_split_probability + self.replace_probability != 100:
            raise ValueError("morph probabilities must sum to 100")


def canonical_selfcitation_config() -> SelfCitationConfig:
    """Return the pinned executable reference configuration."""

    config = SelfCitationConfig()
    config.validate()
    return config
