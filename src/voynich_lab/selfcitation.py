from __future__ import annotations

from dataclasses import asdict, dataclass


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
