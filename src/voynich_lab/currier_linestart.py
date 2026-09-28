from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

from .currier import parse_currier_boundary_corpus, valid_boundary_token

GLYPHS = tuple("ptkfyd")


def jensen_shannon_divergence(left: Counter, right: Counter) -> float:
    keys = set(left) | set(right)
    left_total = sum(left.values()) or 1
    right_total = sum(right.values()) or 1
    p = {key: left.get(key, 0) / left_total for key in keys}
    q = {key: right.get(key, 0) / right_total for key in keys}
    midpoint = {key: 0.5 * (p[key] + q[key]) for key in keys}

    def kl(a: dict, b: dict) -> float:
        return sum(
            a[key] * math.log2(a[key] / b[key])
            for key in keys
            if a[key] > 0
        )

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def _annotate_line_initials(lines: list[dict], representation: str = "collapsed") -> None:
    if representation not in {"collapsed", "raw"}:
        raise ValueError("representation must be 'collapsed' or 'raw'")
    key = "tokens" if representation == "collapsed" else "raw_tokens"
    for line in lines:
        clean = [token for token in line[key] if valid_boundary_token(token)]
        if not clean:
            line["line_initial"] = ""
            line["mid_initials"] = []
            continue
        line["line_initial"] = clean[0][0]
        line["mid_initials"] = [token[0] for token in clean[1:]]


def _initials(lines: list[dict]) -> tuple[Counter, Counter]:
    first = Counter(line["line_initial"] for line in lines if line.get("line_initial"))
    middle = Counter(glyph for line in lines for glyph in line.get("mid_initials", []))
    return first, middle


def _enrichment(first: Counter, middle: Counter) -> dict[str, float]:
    first_total = sum(first.values())
    middle_total = sum(middle.values())
    return {
        glyph: (
            (first.get(glyph, 0) / first_total) / (middle.get(glyph, 0) / middle_total)
            if first_total and middle_total and middle.get(glyph, 0)
            else float("nan")
        )
        for glyph in GLYPHS
    }


def _paragraph_first(line: dict, definition: str) -> bool:
    if definition == "marker":
        return line["marker"] in "@*"
    if definition != "pstart":
        raise ValueError("definition must be 'pstart' or 'marker'")
    return bool(line["paragraph_start"])


def _shares(counter: Counter) -> dict[str, float]:
    total = sum(counter.values()) or 1
    return {glyph: counter.get(glyph, 0) / total for glyph in GLYPHS}


def _relabel_null(lines: list[dict], draws: int, rng: np.random.Generator) -> float:
    pool = []
    for line in lines:
        pool.append(line["line_initial"])
        pool.extend(line["mid_initials"])
    pool = np.array(pool)
    n_first = len(lines)
    values = []
    for _ in range(draws):
        rng.shuffle(pool)
        values.append(
            jensen_shannon_divergence(
                Counter(pool[:n_first].tolist()),
                Counter(pool[n_first:].tolist()),
            )
        )
    return float(np.mean(values))


def line_start_analysis(
    raw_text: str,
    *,
    definition: str = "pstart",
    representation: str = "collapsed",
    null_draws: int = 0,
    bootstrap_repetitions: int = 0,
    seed: int = 20260808,
) -> dict:
    """Reproduce the Currier/paragraph line-start decomposition.

    The default matches the released primary analysis: composite-collapsed EVA
    and explicit paragraph-start markers. Optional relabel-null and quire
    bootstrap checks implement the public robustness calculations.
    """
    lines, _ = parse_currier_boundary_corpus(raw_text)
    _annotate_line_initials(lines, representation)
    first_lines = [line for line in lines if _paragraph_first(line, definition)]
    other_lines = [line for line in lines if not _paragraph_first(line, definition)]

    pooled_first, pooled_middle = _initials(lines)
    first_first, first_middle = _initials(first_lines)
    other_first, other_middle = _initials(other_lines)
    first_jsd = jensen_shannon_divergence(first_first, first_middle)
    other_jsd = jensen_shannon_divergence(other_first, other_middle)
    pooled_jsd = jensen_shannon_divergence(pooled_first, pooled_middle)

    result = {
        "definition": definition,
        "representation": representation,
        "n_lines": len(lines),
        "n_first": len(first_lines),
        "n_other": len(other_lines),
        "jsd": {
            "pooled": pooled_jsd,
            "first": first_jsd,
            "other": other_jsd,
            "ratio_first_over_other": first_jsd / other_jsd if other_jsd else float("nan"),
        },
        "enrichment_first": _enrichment(first_first, first_middle),
        "enrichment_other": _enrichment(other_first, other_middle),
        "line_initial_share_first": _shares(first_first),
        "line_initial_share_other": _shares(other_first),
        "mid_initial_share_first": _shares(first_middle),
        "mid_initial_share_other": _shares(other_middle),
    }

    by_language = {}
    for language in ("A", "B"):
        language_first = [line for line in first_lines if line["currier"] == language]
        language_other = [line for line in other_lines if line["currier"] == language]
        by_language[language] = {
            "first": jensen_shannon_divergence(*_initials(language_first)),
            "other": jensen_shannon_divergence(*_initials(language_other)),
            "n_first": len(language_first),
            "n_other": len(language_other),
        }
    result["jsd_by_language"] = by_language

    if null_draws:
        rng = np.random.default_rng(seed)
        result["relabel_null"] = {
            "draws": null_draws,
            "pooled": _relabel_null(lines, null_draws, rng),
            "first": _relabel_null(first_lines, null_draws, rng),
            "other": _relabel_null(other_lines, null_draws, rng),
        }

    if bootstrap_repetitions:
        by_quire: dict[str, list[dict]] = defaultdict(list)
        for line in lines:
            by_quire[line["quire"]].append(line)
        quires = sorted(by_quire)
        rng = np.random.default_rng(seed)
        differences = []
        first_values = []
        other_values = []
        for _ in range(bootstrap_repetitions):
            picked = rng.choice(len(quires), len(quires), replace=True)
            sample = [line for index in picked for line in by_quire[quires[index]]]
            sample_first = [line for line in sample if _paragraph_first(line, definition)]
            sample_other = [line for line in sample if not _paragraph_first(line, definition)]
            first_value = jensen_shannon_divergence(*_initials(sample_first))
            other_value = jensen_shannon_divergence(*_initials(sample_other))
            first_values.append(first_value)
            other_values.append(other_value)
            differences.append(first_value - other_value)
        result["quire_bootstrap"] = {
            "draws": bootstrap_repetitions,
            "difference_first_minus_other": first_jsd - other_jsd,
            "difference_ci95": [float(x) for x in np.percentile(differences, [2.5, 97.5])],
            "first_ci95": [float(x) for x in np.percentile(first_values, [2.5, 97.5])],
            "other_ci95": [float(x) for x in np.percentile(other_values, [2.5, 97.5])],
        }
    return result
