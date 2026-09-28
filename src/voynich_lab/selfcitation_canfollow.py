from __future__ import annotations

from enum import Enum

from .selfcitation import GenerateType, GlyphGroup


class _Type(Enum):
    GALLOW = "GALLOW"
    CURVE_TYPE = "CURVE_TYPE"
    LINE_TYPE = "LINE_TYPE"
    EMPTY = "EMPTY"
    NONE = "NONE"
    FINAL_TYPE = "FINAL_TYPE"


_START_GLYPHS = ("qo", "a", "o", "y", "c", "s", "d", "k", "t", "p", "f", "x")
_FINAL_GLYPHS = ("y", "n", "l", "r", "s", "d", "m", "g", "x")

_CC_TOKENS = (
    "e", "h", "d", "s", "y", "o", "ch", "sh", "ckh", "cth", "cph", "cfh",
    "al", "ol", "x", "l",
)
_C_FINAL_TOKENS = (
    "d", "g", "o", "y", "dy", "s", "om", "am", "og", "ag", "al", "ar", "ol", "or", "x",
)
_CL_TOKENS = ("a",)

_LC_TOKENS = ("ikh", "ith", "iph", "ifh")
_LL_TOKENS = ("i",)
_L_FINAL_TOKENS = (
    "n", "in", "iin", "iiin", "r", "ir", "iir", "iiir", "m", "im", "iiil", "iil", "il", "iis", "is",
)

_GALLOW_TOKENS = ("k", "t", "p", "f")
_AFTER_GALLOW_TOKENS = ("a", "e", "o", "y", "h", "ch", "sh")
_BEFORE_GALLOW_TOKENS = ("", "a", "e", "o", "l", "y", "h")

_AOY_TOKENS = ("a", "o", "y")
_RMNG_TOKENS = ("r", "m", "n", "g")
_COMBINABLE_LIGATURES = ("ol", "or", "al", "ar")


def _starts_with_any(text: str, candidates: tuple[str, ...]) -> bool:
    return any(text.startswith(candidate) for candidate in candidates)


def _ends_with_any(text: str, candidates: tuple[str, ...]) -> bool:
    return any(text.endswith(candidate) for candidate in candidates)


class CurveLineCanFollow:
    """Exact Python port of the released ``CurveLineCanFollow`` rule.

    ``use_word_final_substitutions=True`` matches the pinned executable config.
    The Java ``resultingGlyphGroup`` argument is intentionally absent from the
    public Python methods because this implementation never reads it upstream.
    """

    def __init__(self, *, use_word_final_substitutions: bool = True) -> None:
        self.use_word_final_substitutions = bool(use_word_final_substitutions)

    @staticmethod
    def _is_combinable_ligature(token: str) -> bool:
        return token in _COMBINABLE_LIGATURES

    @staticmethod
    def _starts_with_combinable_ligature(token: str) -> bool:
        return _starts_with_any(token, _COMBINABLE_LIGATURES)

    @staticmethod
    def _ends_with_combinable_ligature(token: str) -> bool:
        return _ends_with_any(token, _COMBINABLE_LIGATURES)

    @staticmethod
    def _determine_end_token_type(token: str) -> _Type:
        if token == "":
            return _Type.EMPTY
        if _ends_with_any(token, _CL_TOKENS):
            return _Type.LINE_TYPE
        if _ends_with_any(token, _LL_TOKENS):
            return _Type.LINE_TYPE
        if _ends_with_any(token, _CC_TOKENS):
            return _Type.CURVE_TYPE
        if _ends_with_any(token, _LC_TOKENS):
            return _Type.CURVE_TYPE
        if _ends_with_any(token, _FINAL_GLYPHS):
            return _Type.FINAL_TYPE
        if _ends_with_any(token, _GALLOW_TOKENS):
            return _Type.GALLOW
        return _Type.NONE

    @staticmethod
    def _determine_start_token_type(token: str) -> _Type:
        if token == "":
            return _Type.EMPTY
        if _starts_with_any(token, _LL_TOKENS):
            return _Type.LINE_TYPE
        if _starts_with_any(token, _LC_TOKENS):
            return _Type.LINE_TYPE
        if _starts_with_any(token, _L_FINAL_TOKENS):
            return _Type.LINE_TYPE
        if _starts_with_any(token, _CC_TOKENS):
            return _Type.CURVE_TYPE
        if _starts_with_any(token, _CL_TOKENS):
            return _Type.CURVE_TYPE
        if _starts_with_any(token, _C_FINAL_TOKENS):
            return _Type.CURVE_TYPE
        if _starts_with_any(token, _GALLOW_TOKENS):
            return _Type.GALLOW
        return _Type.NONE

    def can_follow_each_other_before(self, glyph_to_add: str, group2: str) -> bool:
        """Return whether ``glyph_to_add`` may be placed immediately before ``group2``."""

        if self._is_combinable_ligature(glyph_to_add):
            if _starts_with_any(group2, _AOY_TOKENS):
                return True
        if self._starts_with_combinable_ligature(group2):
            if _ends_with_any(glyph_to_add, _RMNG_TOKENS):
                return True

        if glyph_to_add != "" and group2.startswith(glyph_to_add):
            return False

        group2_type = self._determine_end_token_type(group2)
        if self.use_word_final_substitutions and group2_type is _Type.EMPTY:
            if glyph_to_add in _L_FINAL_TOKENS:
                return True
            if glyph_to_add in _C_FINAL_TOKENS:
                return True
            return False

        token_type = self._determine_end_token_type(glyph_to_add)
        if token_type is _Type.EMPTY:
            return _starts_with_any(group2, _START_GLYPHS)
        if token_type is _Type.LINE_TYPE:
            return (
                _starts_with_any(group2, _LC_TOKENS)
                or _starts_with_any(group2, _L_FINAL_TOKENS)
                or _starts_with_any(group2, _LL_TOKENS)
            )
        if token_type is _Type.CURVE_TYPE:
            return (
                _starts_with_any(group2, _CC_TOKENS)
                or _starts_with_any(group2, _CL_TOKENS)
                or _starts_with_any(group2, _C_FINAL_TOKENS)
                or _starts_with_any(group2, _GALLOW_TOKENS)
            )
        if token_type is _Type.GALLOW:
            return _starts_with_any(group2, _AFTER_GALLOW_TOKENS)
        if token_type is _Type.FINAL_TYPE:
            return group2 == ""
        if token_type is _Type.NONE:
            return False
        raise RuntimeError(f"unexpected token type: {token_type}")

    def can_follow_each_other_after(self, group1: str, glyph_to_add: str) -> bool:
        """Return whether ``glyph_to_add`` may be placed immediately after ``group1``."""

        if self._is_combinable_ligature(glyph_to_add):
            if _ends_with_any(group1, _RMNG_TOKENS):
                return True
        if self._ends_with_combinable_ligature(group1):
            if _starts_with_any(glyph_to_add, _AOY_TOKENS):
                return True

        if glyph_to_add != "" and group1.endswith(glyph_to_add):
            return False

        group1_type = self._determine_end_token_type(group1)
        if self.use_word_final_substitutions and group1_type is _Type.EMPTY:
            return glyph_to_add in _START_GLYPHS

        token_type = self._determine_start_token_type(glyph_to_add)
        if token_type is _Type.EMPTY:
            return _ends_with_any(group1, _FINAL_GLYPHS)
        if token_type is _Type.LINE_TYPE:
            return _ends_with_any(group1, _LL_TOKENS) or _ends_with_any(group1, _CL_TOKENS)
        if token_type is _Type.CURVE_TYPE:
            return (
                _ends_with_any(group1, _CC_TOKENS)
                or _ends_with_any(group1, _LC_TOKENS)
                or _ends_with_any(group1, _GALLOW_TOKENS)
            )
        if token_type is _Type.GALLOW:
            for candidate in _BEFORE_GALLOW_TOKENS:
                if (candidate == "" and group1 == "") or (
                    candidate != "" and group1.endswith(candidate)
                ):
                    return True
            return False
        if token_type is _Type.NONE:
            return False
        raise RuntimeError(f"unexpected token type: {token_type}")

    def can_follow_to_initial_glyph(self, glyph_to_add: str, group2: str) -> bool:
        return self.can_follow_each_other_before(glyph_to_add, group2)

    def has_valid_start_glyph(self, glyph_group: GlyphGroup) -> bool:
        if glyph_group.generate_type is GenerateType.INITIAL:
            return True
        return glyph_group.get_token_count() > 0 and self.can_follow_each_other_before(
            "", glyph_group.glyph_group
        )

    def is_valid(self, glyph_group: GlyphGroup) -> bool:
        if not self.has_valid_start_glyph(glyph_group):
            return False
        for index in range(1, glyph_group.get_token_count()):
            previous = glyph_group.get_token(index - 1)
            following = glyph_group.get_token(index)
            last_ok = self.can_follow_each_other_after(previous, following)
            next_ok = self.can_follow_each_other_before(previous, following)
            if not last_ok or not next_ok:
                return False
        return True
