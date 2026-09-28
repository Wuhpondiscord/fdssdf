"""Stratum-preserving generators (N7) and a heterogeneity control.

Every global null in the ladder is fit to the whole manuscript, so it is stationary by construction
while the manuscript is not (herbal vs. astronomical vs. Currier A/B, hand changes, ...). Any
pooled-statistic comparison or block classifier can then "detect" the generator merely because the
generator has no strata. This module makes that confound measurable instead of leaving it implicit:

* ``line_strata`` reads per-line quire / Currier-language labels from a parsed IVTFF document.
* ``stratified_generator`` wraps *any* ``(text, seed) -> text`` generator so it is fit separately inside
  each stratum and the output is reassembled in the original line order (layout-preserving).
* ``heterogeneity_control`` runs the classifier two-sample test against the global generator and its
  stratified twin, and reports how much of the separability was stratum composition.

The wrapped generator is bound to one specific text layout (its labels are per-line). It raises rather
than guessing if it is handed a text with a different number of lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .discriminator import classifier_two_sample_test

_SEED_STRIDE = 1_000_003  # large prime: (replicate, stratum) seed pairs never collide for realistic counts


def line_strata(doc, by: str = "quire") -> list[str]:
    """Per-line stratum label for a parsed IVTFF document ('?' where the label is missing)."""
    if by == "quire":
        return [str(line.quire) if line.quire else "?" for line in doc.lines]
    if by == "currier":
        return [str(line.currier_language) if line.currier_language else "?" for line in doc.lines]
    raise ValueError(f"unknown stratification {by!r}; expected 'quire' or 'currier'")


def stratified_generator(labels: Sequence[str], base_fn: Callable[[str, int], str]) -> Callable[[str, int], str]:
    """Fit ``base_fn`` independently inside each stratum; reassemble in original line order."""
    labels = [str(x) for x in labels]

    def generate(text: str, seed: int = 0) -> str:
        lines = text.split("\n")
        trailing_newline = len(lines) == len(labels) + 1 and lines[-1] == ""
        if trailing_newline:
            lines = lines[:-1]
        if len(lines) != len(labels):
            raise ValueError(
                f"stratified generator was built for {len(labels)} lines but received {len(lines)}; "
                "labels must come from the same document that produced this text"
            )
        members: dict[str, list[int]] = {}
        for i, lab in enumerate(labels):
            members.setdefault(lab, []).append(i)
        out = list(lines)
        for k, lab in enumerate(sorted(members)):
            idxs = members[lab]
            generated = base_fn("\n".join(lines[i] for i in idxs), seed + _SEED_STRIDE * k).split("\n")
            if len(generated) != len(idxs):
                raise ValueError(
                    f"base generator returned {len(generated)} lines for stratum {lab!r}; expected {len(idxs)}"
                )
            for i, g in zip(idxs, generated):
                out[i] = g
        return "\n".join(out) + ("\n" if trailing_newline else "")

    generate.__name__ = f"stratified_{getattr(base_fn, '__name__', 'generator')}"
    return generate


@dataclass
class HeterogeneityControl:
    auc_global: float
    auc_global_ci: tuple[float, float]
    auc_stratified: float
    auc_stratified_ci: tuple[float, float]
    auc_drop: float
    n_strata: int
    stratified_ci_includes_chance: bool
    note: str


def heterogeneity_control(
    text: str,
    labels: Sequence[str],
    base_fn: Callable[[str, int], str],
    *,
    block_size: int = 200,
    n_generated_replicates: int = 3,
    seed: int = 0,
    n_splits: int = 12,
) -> HeterogeneityControl:
    """Classifier two-sample AUC against ``base_fn`` fit globally vs. fit within strata.

    ``auc_drop`` is the separability attributable to stratum composition. Whatever AUC remains above 0.5
    after stratification is *not* explained by the strata used here: it may be within-stratum drift
    (large quires span several sections/hands) or genuine local structure the generator lacks.
    """
    common = dict(block_size=block_size, n_generated_replicates=n_generated_replicates, seed=seed, n_splits=n_splits)
    g = classifier_two_sample_test(text, base_fn, **common)
    s = classifier_two_sample_test(text, stratified_generator(labels, base_fn), **common)
    includes_chance = bool(s.auc_ci_low <= 0.5 <= s.auc_ci_high) if s.auc_ci_low == s.auc_ci_low else False
    note = (
        "auc_drop is the share of block-level separability explained by stratum composition alone. "
        "Residual AUC above 0.5 is unexplained by these strata (within-stratum drift or real local "
        "structure); an AUC near 0.5 still would not certify the generator as correct."
    )
    return HeterogeneityControl(
        auc_global=g.held_out_auc,
        auc_global_ci=(g.auc_ci_low, g.auc_ci_high),
        auc_stratified=s.held_out_auc,
        auc_stratified_ci=(s.auc_ci_low, s.auc_ci_high),
        auc_drop=g.held_out_auc - s.held_out_auc,
        n_strata=len(set(str(x) for x in labels)),
        stratified_ci_includes_chance=includes_chance,
        note=note,
    )
