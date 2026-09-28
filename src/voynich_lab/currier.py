from __future__ import annotations

from collections import Counter, defaultdict
import re

import numpy as np

# Exact corpus conventions used by reproduce_headlines.py and
# reproduce_dialect_linestart.py for the Currier boundary analysis.
HEADER_RE = re.compile(r"^<(f[0-9]+[rv][0-9]*)>\s+<!\s*([^>]*)>")
LOCUS_RE = re.compile(r"^<(f[0-9rv]+[0-9]*)\.(\d+),([@+*=~])([A-Za-z])[^>]*>")
BAD_BOUNDARY = set("?*<>{}[]()|@;:.,0123456789'")
SUBSTITUTIONS = (
    ("cth", "T"),
    ("ckh", "K"),
    ("cph", "P"),
    ("cfh", "F"),
    ("ch", "C"),
    ("sh", "S"),
    ("iin", "N"),
    ("in", "I"),
    ("ee", "E"),
)
BOUNDARY_TYPES = ("uncertain", "first", "mid", "certain", "linebreak")


def collapse(token: str) -> str:
    out = token
    for old, new in SUBSTITUTIONS:
        out = out.replace(old, new)
    return out


def clean_body(raw: str) -> str:
    text = raw[raw.index(">") + 1 :].strip()
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"\[([^:\]]*):[^\]]*\]", r"\1", text)
    return re.sub(r"\{[^}]*\}", "", text)


def valid_boundary_token(token: str) -> bool:
    return bool(token) and not (set(token) & BAD_BOUNDARY)


def parse_currier_boundary_corpus(raw_text: str) -> tuple[list[dict], list[tuple]]:
    """Parse the exact P-locus corpus used for the published Currier profiles."""
    metadata: dict[str, dict[str, str]] = {}
    lines: list[dict] = []
    for raw in (raw_text or "").splitlines():
        header = HEADER_RE.match(raw)
        if header:
            metadata[header.group(1)] = dict(
                re.findall(r"\$([A-Z])=([A-Za-z0-9]+)", header.group(2))
            )
            continue
        locus = LOCUS_RE.match(raw)
        if not locus or locus.group(4) != "P":
            continue
        page = locus.group(1)
        parts = re.split(r"([.,])", clean_body(raw))
        raw_tokens = [parts[i].strip() for i in range(0, len(parts), 2)]
        tokens = [collapse(token) for token in raw_tokens]
        delimiters = [parts[i] for i in range(1, len(parts), 2)]
        if sum(valid_boundary_token(token) for token in tokens) < 4:
            continue
        meta = metadata.get(page, {})
        line = {
            "page": page,
            "locus": int(locus.group(2)),
            "marker": locus.group(3),
            "quire": meta.get("Q", "?"),
            "bifolio": (meta.get("Q", "?"), meta.get("B", "?")),
            "currier": meta.get("L", "?"),
            "hand": meta.get("H", "?"),
            "tokens": tokens,
            "raw_tokens": raw_tokens,
            "delimiters": delimiters,
            "internal": [],
            "unigrams": [],
            "spaces": [],
            "paragraph_start": "<%>" in raw,
        }
        for token in tokens:
            if not valid_boundary_token(token):
                continue
            line["internal"].extend(zip(token, token[1:]))
            line["unigrams"].extend(token)
        for index, delimiter in enumerate(delimiters):
            if delimiter not in ".,":
                continue
            left, right = tokens[index], tokens[index + 1]
            if not (valid_boundary_token(left) and valid_boundary_token(right)):
                continue
            if index == 0:
                position = "first"
            elif index == len(delimiters) - 1:
                position = "last"
            else:
                position = "mid"
            line["spaces"].append((delimiter, position, left[-1], right[0]))
        lines.append(line)

    linebreaks: list[tuple] = []
    for first, second in zip(lines, lines[1:]):
        if first["page"] != second["page"] or second["marker"] == "@":
            continue
        left = next(
            (token for token in reversed(first["tokens"]) if valid_boundary_token(token)),
            None,
        )
        right = next(
            (token for token in second["tokens"] if valid_boundary_token(token)),
            None,
        )
        if left and right:
            linebreaks.append((first["quire"], left[-1], right[0], first["currier"]))
    return lines, linebreaks


def association_scale(lines: list[dict]) -> dict:
    internal = [pair for line in lines for pair in line["internal"]]
    unigrams = [glyph for line in lines for glyph in line["unigrams"]]
    if not internal or not unigrams:
        raise ValueError("Boundary profile needs within-token bigrams and glyphs.")
    joint = Counter(internal)
    unigram = Counter(unigrams)
    vocabulary = sorted(unigram)
    index = {glyph: i for i, glyph in enumerate(vocabulary)}
    size = len(vocabulary)
    matrix = np.full((size, size), 0.5)
    for (left, right), count in joint.items():
        matrix[index[left], index[right]] += count
    matrix /= sum(joint.values()) + 0.5 * size * size
    marginal = np.array([unigram[glyph] for glyph in vocabulary], dtype=float)
    marginal /= marginal.sum()
    scores = np.log2(matrix / np.outer(marginal, marginal))
    internal_mean = float(np.mean([scores[index[a], index[b]] for a, b in internal]))
    random_mean = float(np.sum(np.outer(marginal, marginal) * scores))
    return {
        "scores": scores,
        "index": index,
        "internal_mean": internal_mean,
        "random_mean": random_mean,
        "denominator": internal_mean - random_mean,
    }


def pair_score(scale: dict, pair: tuple[str, str]) -> float:
    left, right = pair
    index = scale["index"]
    if left not in index or right not in index:
        return float("nan")
    return float(scale["scores"][index[left], index[right]])


def boundary_pools(lines: list[dict], linebreaks: list[tuple]) -> dict[str, list[tuple]]:
    spaces = [space for line in lines for space in line["spaces"]]
    return {
        "uncertain": [(a, b) for delimiter, _, a, b in spaces if delimiter == ","],
        "certain": [(a, b) for delimiter, _, a, b in spaces if delimiter == "."],
        "first": [(a, b) for _, position, a, b in spaces if position == "first"],
        "mid": [(a, b) for _, position, a, b in spaces if position == "mid"],
        "linebreak": [(a, b) for _, a, b in linebreaks],
    }


def boundary_index(lines: list[dict], linebreaks: list[tuple]) -> tuple[dict, dict]:
    scale = association_scale(lines)
    pools = boundary_pools(lines, linebreaks)
    result: dict[str, tuple[float, int]] = {}
    for name, pairs in pools.items():
        values = np.array([pair_score(scale, pair) for pair in pairs], dtype=float)
        values = values[~np.isnan(values)]
        value = (
            float((values.mean() - scale["random_mean"]) / scale["denominator"])
            if len(values)
            else float("nan")
        )
        result[name] = (value, len(values))
    return result, scale


def _stratum_lines(lines: list[dict], language: str) -> list[dict]:
    return [line for line in lines if line["currier"] == language]


def _stratum_breaks(linebreaks: list[tuple], language: str) -> list[tuple]:
    return [(q, a, b) for q, a, b, lang in linebreaks if lang == language]


def stratum_profile(lines: list[dict], breaks: list[tuple]) -> dict:
    estimates, scale = boundary_index(lines, breaks)
    spaces = [space for line in lines for space in line["spaces"]]
    uncertain = [(a, b) for delimiter, _, a, b in spaces if delimiter == ","]
    certain = [(a, b) for delimiter, _, a, b in spaces if delimiter == "."]
    uncertain_scores = np.array([pair_score(scale, pair) for pair in uncertain], dtype=float)
    certain_scores = np.array([pair_score(scale, pair) for pair in certain], dtype=float)
    uncertain_scores = uncertain_scores[~np.isnan(uncertain_scores)]
    certain_scores = certain_scores[~np.isnan(certain_scores)]
    mean_uncertain = float(np.mean(uncertain_scores)) if len(uncertain_scores) else float("nan")
    mean_certain = float(np.mean(certain_scores)) if len(certain_scores) else float("nan")
    return {
        "index": {name: estimates[name][0] for name in BOUNDARY_TYPES},
        "n": {name: estimates[name][1] for name in BOUNDARY_TYPES},
        "lines": len(lines),
        "pages": len({line["page"] for line in lines}),
        "quires": sorted({line["quire"] for line in lines}),
        "uncertain_rate": len(uncertain) / len(spaces) if spaces else float("nan"),
        "raw_gap_bits": mean_uncertain - mean_certain,
        "mean_score_uncertain": mean_uncertain,
        "mean_score_certain": mean_certain,
        "internal_anchor": scale["internal_mean"],
        "random_anchor": scale["random_mean"],
    }


def currier_boundary_profiles(raw_text: str) -> dict:
    """Compute paper-compatible A/B boundary profiles with stratum-local scales."""
    lines, breaks = parse_currier_boundary_corpus(raw_text)
    profiles = {}
    for language in ("A", "B"):
        sub_lines = _stratum_lines(lines, language)
        if not sub_lines:
            continue
        profiles[language] = stratum_profile(
            sub_lines,
            _stratum_breaks(breaks, language),
        )
    differences = {}
    if "A" in profiles and "B" in profiles:
        differences = {
            name: profiles["A"]["index"][name] - profiles["B"]["index"][name]
            for name in BOUNDARY_TYPES
        }
    return {
        "profiles": profiles,
        "A_minus_B_index": differences,
        "corpus_lines": len(lines),
        "linebreaks": len(breaks),
    }


def _index_only(lines: list[dict], breaks: list[tuple]) -> dict[str, float]:
    estimates, _ = boundary_index(lines, breaks)
    return {name: estimates[name][0] for name in BOUNDARY_TYPES}


def currier_ab_bootstrap(
    raw_text: str,
    *,
    repetitions: int = 500,
    seed: int = 20260808,
) -> dict:
    """Bootstrap A-B boundary-index differences by quire under both paper schemes."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    lines, breaks = parse_currier_boundary_corpus(raw_text)
    strata = {}
    for language in ("A", "B"):
        sub = _stratum_lines(lines, language)
        by_quire: dict[str, list[dict]] = defaultdict(list)
        breaks_by_quire: dict[str, list[tuple]] = defaultdict(list)
        for line in sub:
            by_quire[line["quire"]].append(line)
        for quire, left, right, lang in breaks:
            if lang == language:
                breaks_by_quire[quire].append((quire, left, right))
        quires = sorted(by_quire)
        if not quires:
            raise ValueError(f"No Currier {language} quires found.")
        strata[language] = (quires, by_quire, breaks_by_quire)

    rng = np.random.default_rng(seed)
    independent: dict[str, list[float]] = defaultdict(list)
    for _ in range(repetitions):
        draw = {}
        for language in ("A", "B"):
            quires, by_quire, breaks_by_quire = strata[language]
            picked = rng.choice(len(quires), len(quires), replace=True)
            sample_lines = [line for i in picked for line in by_quire[quires[i]]]
            sample_breaks = [item for i in picked for item in breaks_by_quire[quires[i]]]
            draw[language] = _index_only(sample_lines, sample_breaks)
        for name in BOUNDARY_TYPES:
            independent[name].append(draw["A"][name] - draw["B"][name])

    all_quires = sorted({line["quire"] for line in lines})
    lines_by_quire: dict[str, list[dict]] = defaultdict(list)
    breaks_by_quire: dict[str, list[tuple]] = defaultdict(list)
    for line in lines:
        lines_by_quire[line["quire"]].append(line)
    for quire, left, right, language in breaks:
        breaks_by_quire[quire].append((quire, left, right, language))
    rng = np.random.default_rng(seed)
    joint: dict[str, list[float]] = defaultdict(list)
    attempts = 0
    while len(joint["uncertain"]) < repetitions:
        attempts += 1
        if attempts > repetitions * 20:
            raise RuntimeError("Could not obtain enough valid joint-quire bootstrap draws.")
        picked = rng.choice(len(all_quires), len(all_quires), replace=True)
        sample_lines = [line for i in picked for line in lines_by_quire[all_quires[i]]]
        sample_breaks = [item for i in picked for item in breaks_by_quire[all_quires[i]]]
        draw = {}
        valid = True
        for language in ("A", "B"):
            language_lines = _stratum_lines(sample_lines, language)
            if not language_lines:
                valid = False
                break
            try:
                draw[language] = _index_only(
                    language_lines,
                    [
                        (q, a, b)
                        for q, a, b, lang in sample_breaks
                        if lang == language
                    ],
                )
            except ValueError:
                valid = False
                break
        if not valid:
            continue
        for name in BOUNDARY_TYPES:
            joint[name].append(draw["A"][name] - draw["B"][name])

    def summarize(series: dict[str, list[float]]) -> dict:
        output = {}
        for name, values in series.items():
            array = np.array(values, dtype=float)
            array = array[np.isfinite(array)]
            output[name] = {
                "ci90": [float(x) for x in np.percentile(array, [5, 95])],
                "ci95": [float(x) for x in np.percentile(array, [2.5, 97.5])],
                "sd": float(np.std(array)),
                "valid_draws": int(len(array)),
            }
        return output

    return {
        "repetitions": repetitions,
        "seed": seed,
        "independent_strata": summarize(independent),
        "joint_quires": summarize(joint),
    }


def currier_boundary_analysis(
    raw_text: str,
    *,
    bootstrap_repetitions: int = 500,
    seed: int = 20260808,
) -> dict:
    profiles = currier_boundary_profiles(raw_text)
    profiles["bootstrap"] = currier_ab_bootstrap(
        raw_text,
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    return profiles
