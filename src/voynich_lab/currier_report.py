from __future__ import annotations

import math

from .currier import BOUNDARY_TYPES, currier_ab_bootstrap, currier_boundary_profiles
from .currier_linestart import GLYPHS, line_start_analysis

# The final pinned executable and its bundled STATED constants disagree.
# These constants are retained only so reports can surface that discrepancy;
# they are never used as a computational target for the executable path.
LINESTART_STATED = {
    "jsd": {
        "first": 0.485,
        "other": 0.176,
        "pooled": 0.179,
        "ratio_first_over_other": 2.75,
    },
    "enrichment_first": {
        "p": 23.7,
        "t": 18.2,
        "k": 6.43,
        "f": 4.47,
        "y": 0.18,
        "d": 0.26,
    },
    "enrichment_other": {
        "p": 16.8,
        "t": 4.56,
        "k": 0.43,
        "f": 2.77,
        "y": 4.77,
        "d": 1.98,
    },
}


def _ci_contains_zero(interval: list[float]) -> bool:
    return all(math.isfinite(value) for value in interval) and interval[0] <= 0 <= interval[1]


def _empty_bootstrap() -> dict[str, dict[str, dict[str, object]]]:
    """Return an explicit non-estimable bootstrap result for sparse corpora."""
    output: dict[str, dict[str, dict[str, object]]] = {}
    for scheme in ("independent_strata", "joint_quires"):
        output[scheme] = {}
        for name in BOUNDARY_TYPES:
            output[scheme][name] = {
                "ci90": [float("nan"), float("nan")],
                "ci95": [float("nan"), float("nan")],
                "sd": float("nan"),
                "valid_draws": 0,
            }
    return output


def _profiles_and_bootstrap(
    raw_text: str,
    *,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, object], bool]:
    """Build deterministic profiles and bootstrap only when every class exists.

    Small/custom IVTFF samples can legitimately contain no observations for one
    of the five boundary classes. The paper-scale bootstrap assumes every class
    exists in both Currier strata, so a sparse sample should be reported as
    non-estimable rather than allowed to crash inside percentile calculation.
    """
    boundary = currier_boundary_profiles(raw_text)
    profiles = boundary["profiles"]
    if "A" not in profiles or "B" not in profiles:
        raise ValueError("Currier report requires both native $L=A and $L=B strata.")

    estimable = all(
        profiles[language]["n"][name] > 0
        for language in ("A", "B")
        for name in BOUNDARY_TYPES
    )
    if estimable:
        bootstrap = currier_ab_bootstrap(
            raw_text,
            repetitions=repetitions,
            seed=seed,
        )
    else:
        bootstrap = {
            "repetitions": repetitions,
            "seed": seed,
            **_empty_bootstrap(),
        }
    boundary["bootstrap"] = bootstrap
    return boundary, estimable


def build_currier_report(
    raw_text: str,
    *,
    boundary_bootstrap_repetitions: int = 200,
    line_start_null_draws: int = 0,
    line_start_bootstrap_repetitions: int = 0,
    seed: int = 20260808,
) -> dict[str, object]:
    """Return UI/report-ready Currier and line-start results.

    This function intentionally keeps three concepts separate:
      1. deterministic Currier A/B boundary profiles,
      2. quire-resampled uncertainty around A-B differences, and
      3. line-start executable reproduction, whose pinned final-code output
         differs from the same upstream file's hard-coded STATED constants.
    """
    boundary, bootstrap_estimable = _profiles_and_bootstrap(
        raw_text,
        repetitions=boundary_bootstrap_repetitions,
        seed=seed,
    )
    profiles = boundary["profiles"]

    boundary_rows = []
    for name in BOUNDARY_TYPES:
        boundary_rows.append(
            {
                "boundary_class": name,
                "currier_A_index": profiles["A"]["index"][name],
                "currier_B_index": profiles["B"]["index"][name],
                "A_minus_B": boundary["A_minus_B_index"][name],
                "n_A": profiles["A"]["n"][name],
                "n_B": profiles["B"]["n"][name],
            }
        )

    stratum_rows = []
    for language in ("A", "B"):
        profile = profiles[language]
        stratum_rows.append(
            {
                "currier": language,
                "lines": profile["lines"],
                "pages": profile["pages"],
                "quires": ", ".join(profile["quires"]),
                "uncertain_separator_rate": profile["uncertain_rate"],
                "raw_uncertain_minus_certain_gap_bits": profile["raw_gap_bits"],
                "internal_PMI_anchor_bits": profile["internal_anchor"],
                "random_PMI_anchor_bits": profile["random_anchor"],
            }
        )

    bootstrap_rows = []
    bootstrap = boundary["bootstrap"]
    for scheme in ("independent_strata", "joint_quires"):
        for name in BOUNDARY_TYPES:
            row = bootstrap[scheme][name]
            bootstrap_rows.append(
                {
                    "scheme": scheme,
                    "boundary_class": name,
                    "ci90_low": row["ci90"][0],
                    "ci90_high": row["ci90"][1],
                    "ci95_low": row["ci95"][0],
                    "ci95_high": row["ci95"][1],
                    "sd": row["sd"],
                    "valid_draws": row["valid_draws"],
                    "ci90_contains_zero": _ci_contains_zero(row["ci90"]),
                    "ci95_contains_zero": _ci_contains_zero(row["ci95"]),
                }
            )

    line_start = line_start_analysis(
        raw_text,
        definition="pstart",
        representation="collapsed",
        null_draws=line_start_null_draws,
        bootstrap_repetitions=line_start_bootstrap_repetitions,
        seed=seed,
    )
    jsd_rows = []
    for name in ("first", "other", "pooled", "ratio_first_over_other"):
        executable = line_start["jsd"][name]
        stated = LINESTART_STATED["jsd"][name]
        jsd_rows.append(
            {
                "quantity": name,
                "pinned_executable": executable,
                "bundled_STATED_constant": stated,
                "executable_minus_STATED": executable - stated,
            }
        )

    enrichment_rows = []
    for pool in ("first", "other"):
        executable_map = line_start[f"enrichment_{pool}"]
        stated_map = LINESTART_STATED[f"enrichment_{pool}"]
        for glyph in GLYPHS:
            executable = executable_map[glyph]
            stated = stated_map[glyph]
            enrichment_rows.append(
                {
                    "line_group": pool,
                    "glyph": glyph,
                    "pinned_executable_enrichment": executable,
                    "bundled_STATED_enrichment": stated,
                    "executable_minus_STATED": executable - stated,
                }
            )

    uncertain_independent = bootstrap["independent_strata"]["uncertain"]["ci90"]
    uncertain_joint = bootstrap["joint_quires"]["uncertain"]["ci90"]
    if bootstrap_estimable:
        uncertainty_text = (
            f"For the uncertain-separator A-B index, the independent-strata 90% interval is "
            f"[{uncertain_independent[0]:.4f}, {uncertain_independent[1]:.4f}] and the joint-quire interval is "
            f"[{uncertain_joint[0]:.4f}, {uncertain_joint[1]:.4f}]. "
        )
    else:
        uncertainty_text = (
            "Quire-bootstrap intervals are not estimable for this input because at least one Currier stratum "
            "has zero observations in one or more boundary classes. "
        )
    interpretation = (
        "Currier A/B boundary profiles are computed on stratum-local PMI scales and uncertainty is resampled by quire. "
        + uncertainty_text
        + "Do not interpret a pooled A/B point difference as an independent dialect effect without quire-aware uncertainty. "
        "Line-start values are labeled as pinned-executable results because the authors' final executable output differs from its bundled STATED constants."
    )

    return {
        "boundary_rows": boundary_rows,
        "stratum_rows": stratum_rows,
        "bootstrap_rows": bootstrap_rows,
        "line_start_jsd_rows": jsd_rows,
        "line_start_enrichment_rows": enrichment_rows,
        "line_start_currier": line_start["jsd_by_language"],
        "interpretation": interpretation,
        "provenance": {
            "boundary_method": "stratum-local PMI normalization; native-$Q bootstrap",
            "line_start_method": "pstart + composite-collapsed; executable-parity path",
            "seed": seed,
            "boundary_bootstrap_repetitions": boundary_bootstrap_repetitions,
            "boundary_bootstrap_estimable": bootstrap_estimable,
            "line_start_null_draws": line_start_null_draws,
            "line_start_bootstrap_repetitions": line_start_bootstrap_repetitions,
        },
        "raw_boundary": boundary,
        "raw_line_start": line_start,
    }
