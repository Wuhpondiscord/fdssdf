from __future__ import annotations

from dataclasses import replace
import math

from .selfcitation import canonical_selfcitation_config
from .selfcitation_generator import generate_selfcitation_text


SELFCITATION_ADAPTER_NAME = "H1 Timm/Schinner self-citation pseudotext"


def estimate_selfcitation_lines(reference_text: str, *, max_lines: int = 1200) -> int:
    """Choose a self-citation output size comparable to a reference corpus.

    If the reference already has multiple non-empty lines, preserve that line
    count. Otherwise estimate the number of ~45-character lines needed to
    produce a similarly sized sample. The released generator is capped at its
    canonical 1,200-line setting so adversarial-validation runs remain bounded.
    """

    non_empty_lines = [line for line in str(reference_text).splitlines() if line.strip()]
    if len(non_empty_lines) >= 2:
        return max(2, min(int(max_lines), len(non_empty_lines)))

    compact_length = sum(1 for ch in str(reference_text) if not ch.isspace())
    estimated = max(2, math.ceil(compact_length / 45))
    return min(int(max_lines), estimated)


def selfcitation_reference_matched(reference_text: str, seed: int = 0) -> str:
    """Generate length/line-count-matched self-citation pseudotext.

    This adapter exists specifically for the scorecard/discriminator contract:
    ``(reference_text, seed) -> generated_text``. It preserves the released
    self-citation model's own line structure and only changes two configuration
    fields: pseudo-RNG seed and requested line count.
    """

    line_count = estimate_selfcitation_lines(reference_text)
    config = replace(
        canonical_selfcitation_config(),
        random_seed=int(seed),
        lines_to_create=line_count,
    )
    return generate_selfcitation_text(config, lines_to_create=line_count)
