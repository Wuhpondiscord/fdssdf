"""Verified literature targets for reproduction checks."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    name: str
    value: float
    source: str
    comparison: str = "point"
    note: str = ""
    verified: bool = True


TARGETS: dict[str, Target] = {
    "char_conditional_entropy_bits": Target(
        "char_conditional_entropy_bits", 2.69, "Rozanova & Temerev 2026, arXiv:2608.17096",
        note="Composite-collapsed EVA; first-order conditional entropy after P-locus cleaning and erasing token/line boundaries from the character sequence. Decomposed EVA is reported separately at 2.32 bits.",
    ),
    "token_succession_entropy_fraction": Target(
        "token_succession_entropy_fraction", 0.01, "Rozanova & Temerev 2026, arXiv:2608.17096",
        comparison="upper_bound",
        note="Shuffle-corrected adjacent capped-token MI divided by the full capped-token marginal entropy H(T), using a 2,000-type cap and within-line shuffles. The paper reports under 1%; 0.01 is retained here as an upper bound rather than a point target.",
    ),
    "cross_boundary_edge_mi_bits": Target(
        "cross_boundary_edge_mi_bits", 0.197, "Rozanova & Temerev 2026, arXiv:2608.17096",
        note="Shuffle-corrected last-glyph(token t) to first-glyph(token t+1) MI on composite-collapsed observed-space lines; 100 within-line shuffles in the public reproduction.",
    ),
    "bpe_crossfit_gap_0_bits": Target("bpe_crossfit_gap_0_bits", 1.686, "Rozanova & Temerev 2026, arXiv:2608.17096", note="Glyph-weighted leave-one-quire-out dependence gap H1-H2, 0 within-token BPE merges."),
    "bpe_crossfit_gap_16_bits": Target("bpe_crossfit_gap_16_bits", 1.423, "Rozanova & Temerev 2026, arXiv:2608.17096", note="Glyph-weighted leave-one-quire-out dependence gap H1-H2, 16 within-token BPE merges."),
    "bpe_crossfit_gap_32_bits": Target("bpe_crossfit_gap_32_bits", 1.379, "Rozanova & Temerev 2026, arXiv:2608.17096", note="Glyph-weighted leave-one-quire-out dependence gap H1-H2, 32 within-token BPE merges."),
    "bpe_crossfit_gap_64_bits": Target("bpe_crossfit_gap_64_bits", 1.490, "Rozanova & Temerev 2026, arXiv:2608.17096", note="Glyph-weighted leave-one-quire-out dependence gap H1-H2, 64 within-token BPE merges."),
    "bpe_crossfit_selected_merges": Target(
        "bpe_crossfit_selected_merges", 32, "Rozanova & Temerev 2026, arXiv:2608.17096",
        comparison="exact_integer", note="Held-out minimum among 0/16/32/64 checkpoints. The pooled/in-sample minimum was 64.",
    ),
    "certain_separator_unit_crossing_rate": Target(
        "certain_separator_unit_crossing_rate", 0.025, "Rozanova & Temerev 2026, arXiv:2608.17096",
        note="Share of hidden certain/conventional token boundaries crossed by learned units after spaces are erased; reported as 2.5% for Voynichese in Table 7. The public space-sensitivity driver reports 32/64/128-merge crossing rates, so this target is not auto-scored until the table's specific checkpoint is pinned down.",
    ),
    "end_to_start_flow_proportion": Target(
        "end_to_start_flow_proportion", 0.806, "Parisel 2026, arXiv:2604.19762",
        note="Reported End-to-Start positional-class transition proportion.",
    ),
    "cross_boundary_mutual_information_bits": Target(
        "cross_boundary_mutual_information_bits", 0.230, "Parisel 2026, arXiv:2604.19762",
        note="Parisel total cross-boundary MI. This is not identical to Rozanova & Temerev's edge-MI implementation.",
    ),
}

MISMATCH_REASONS = (
    "DATA_VERSION", "TRANSCRIPTION_NORMALIZATION", "METRIC_DEFINITION", "RANDOM_SEED",
    "IMPLEMENTATION_ERROR", "PAPER_AMBIGUITY", "UNRESOLVED",
)


def compare_to_targets(observed: dict[str, float], tolerance: float = 0.10) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, target in TARGETS.items():
        if key not in observed:
            continue
        obs = float(observed[key])
        if target.comparison == "upper_bound":
            passed = obs <= target.value
            delta = max(0.0, obs - target.value) / max(abs(target.value), 1e-12)
        elif target.comparison == "exact_integer":
            passed = int(round(obs)) == int(round(target.value))
            delta = abs(obs - target.value)
        else:
            delta = abs(obs - target.value) / max(abs(target.value), 1e-12)
            passed = delta <= tolerance
        rows.append({
            "metric": key,
            "observed": obs,
            "published_target": target.value,
            "comparison": target.comparison,
            "error_or_excess": delta,
            "reproduced": bool(passed) if target.verified else None,
            "verified_target": target.verified,
            "source": target.source,
            "note": target.note,
            "mismatch_reason": "" if passed else "UNRESOLVED",
        })
    return rows
