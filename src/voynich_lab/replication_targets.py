"""Published structural targets to reproduce before trusting any new model.

Per the review: "Do not proceed because the plots look about right.
Predeclare tolerances." These numbers are recorded here, with source and
important caveats about the representation each was computed under, so a
mismatch gets classified rather than glossed over. Treat MISMATCH_REASONS
as a mandatory field, not an afterthought -- an unexplained mismatch is a
finding, not noise to average away.

Sources:
  Rozanova & Temerev (2026), "A Glyph Is Not a Letter, a Token Is Not a
    Word, a Space Is Not a Space" (arXiv:2608.17096) -- character
    conditional entropy, conventional-token succession entropy fraction,
    cross-boundary edge mutual information, BPE unit-scale transition,
    certain- vs uncertain-separator crossing rate.
  Parisel (2026), positional/directional structure paper -- End->Start
    extremity-flow proportion, cross-boundary mutual information (bits),
    generator pass/fail against the four combined targets.

These are reproduction targets for *this project's own reimplementation*,
not a substitute for reading the papers' methods sections. Metric
definitions (representation, collapsing of composite EVA glyphs, line
selection criteria) must match before a comparison is meaningful -- see
MISMATCH_REASONS below.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    name: str
    value: float
    source: str
    note: str = ""


TARGETS: dict[str, Target] = {
    "char_conditional_entropy_bits": Target(
        name="char_conditional_entropy_bits",
        value=2.7,
        source="Rozanova & Temerev 2026",
        note="Composite-collapsed EVA representation; order-1 conditional entropy.",
    ),
    "token_succession_entropy_fraction": Target(
        name="token_succession_entropy_fraction",
        value=0.01,
        source="Rozanova & Temerev 2026",
        note="Upper bound: conventional-token identity explains <~1% of next-token entropy.",
    ),
    "cross_boundary_edge_mi_bits": Target(
        name="cross_boundary_edge_mi_bits",
        value=0.2,
        source="Rozanova & Temerev 2026",
        note="Last-glyph-of-token vs first-glyph-of-next-token mutual information.",
    ),
    "bpe_scale_transition_merges_low": Target(
        name="bpe_scale_transition_merges_low",
        value=32,
        source="Rozanova & Temerev 2026",
        note="Held-out cross-fitted minimum; reported range extends toward ~64.",
    ),
    "uncertain_separator_crossing_rate": Target(
        name="uncertain_separator_crossing_rate",
        value=0.203,
        source="Rozanova & Temerev 2026",
        note="Fraction of learned-unit boundaries crossing transcriber-marked uncertain spaces.",
    ),
    "certain_separator_crossing_rate": Target(
        name="certain_separator_crossing_rate",
        value=0.025,
        source="Rozanova & Temerev 2026",
        note="Same statistic for certain spaces; should be far below the uncertain-space rate.",
    ),
    "end_to_start_flow_proportion": Target(
        name="end_to_start_flow_proportion",
        value=0.806,
        source="Parisel 2026",
        note="Proportion of extremity flow classified End->Start.",
    ),
    "cross_boundary_mutual_information_bits": Target(
        name="cross_boundary_mutual_information_bits",
        value=0.230,
        source="Parisel 2026",
        note="Distinct metric definition from Rozanova's edge-MI figure -- do not conflate the two.",
    ),
}

MISMATCH_REASONS = (
    "DATA_VERSION",
    "TRANSCRIPTION_NORMALIZATION",
    "METRIC_DEFINITION",
    "RANDOM_SEED",
    "IMPLEMENTATION_ERROR",
    "PAPER_AMBIGUITY",
    "UNRESOLVED",
)


def compare_to_targets(
    observed: dict[str, float], tolerance: float = 0.25
) -> list[dict[str, object]]:
    """Relative-error comparison against each target present in `observed`.
    tolerance is a fractional relative-error threshold (0.25 = within 25%
    of the published value counts as reproduced); this is a coarse default
    and should be tightened per-metric once you understand each metric's
    own sampling variance, per the review's delta_s / CI-overlap guidance.
    """
    rows: list[dict[str, object]] = []
    for key, target in TARGETS.items():
        if key not in observed:
            continue
        obs = observed[key]
        denom = max(abs(target.value), 1e-9)
        delta = abs(obs - target.value) / denom
        rows.append(
            {
                "metric": key,
                "observed": obs,
                "published_target": target.value,
                "relative_error": delta,
                "reproduced_within_tolerance": bool(delta <= tolerance),
                "source": target.source,
                "note": target.note,
                "mismatch_reason": "" if delta <= tolerance else "UNRESOLVED",
            }
        )
    return rows
