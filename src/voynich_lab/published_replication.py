from __future__ import annotations

from collections import Counter, defaultdict
from math import log2
import re

import numpy as np

from .segmentation import cross_fit_bpe_scale_curve

# Public reproduction conventions from lrozanova/voynich-units.
# These are intentionally separate because the paper's headline analyses use
# different corpus-cleaning contracts.
HEADER_RE = re.compile(r"^<([^>.]+)>\s+<!([^\n]+)>", re.M)
LOCUS_RE = re.compile(r"^<([^>.]+)\.(\d+),([@+*=~&])([A-Za-z])[^>]*>\s*(.*)$")
TOKEN_RE = re.compile(r"[a-z]+")
BAD_UNIT = set("?*<>{}[]()|@")
BAD_ENTROPY = set("?*<>{}[]()|")
CAP = 2000
ORDER_SEED = 20260816
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


def collapse_composites(token: str) -> str:
    """Apply the paper's composite-collapsed EVA substitutions in order."""
    out = token
    for old, new in SUBSTITUTIONS:
        out = out.replace(old, new)
    return out


def _strip_common_markup(text: str) -> str:
    """Resolve/remove markup shared by the public reproduction scripts."""
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"\[([^:\]]*):[^\]]*\]", r"\1", text)
    text = re.sub(r"\{[^}]*\}", "", text)
    return text


def _strip_strict_markup(text: str) -> str:
    # plant_crib_attack.strip_markup additionally erases apostrophes.
    return _strip_common_markup(text).replace("'", "")


def _metadata_by_folio(raw_text: str) -> dict[str, dict[str, str]]:
    return {
        match.group(1): dict(re.findall(r"\$([A-Z])=([^ $>]+)", match.group(2)))
        for match in HEADER_RE.finditer(raw_text or "")
    }


def _paragraph_loci(raw_text: str):
    for raw in (raw_text or "").splitlines():
        match = LOCUS_RE.match(raw)
        if match and match.group(4) == "P":
            yield match


def entropy_corpus_tokens(raw_text: str) -> list[str]:
    """Paper-exact token list for the 2.69-bit entropy headline.

    This follows reproduce_headlines.py rather than the stricter BPE cleaner:
    only P loci are used, common IVTFF markup is removed, tokens containing the
    BAD_ENTROPY characters are dropped, but apostrophes are not erased by this
    path. Boundaries are removed only when the final symbol sequence is built.
    """
    tokens: list[str] = []
    for match in _paragraph_loci(raw_text):
        body = _strip_common_markup(match.group(5)).strip()
        tokens.extend(
            word
            for word in re.split(r"[.,]", body)
            if word and not (set(word) & BAD_ENTROPY)
        )
    return tokens


def unit_scale_lines_by_quire(raw_text: str) -> dict[str, list[list[str]]]:
    """Paper-exact P-locus corpus for within-token BPE/cross-fit.

    Mirrors clean_tokens + grouped_voynich_lines in reproduce_unit_scale.py:
    markup and apostrophes are removed, the body is split on punctuation or
    whitespace, invalid pieces are skipped individually, and surviving lines
    are grouped by native $Q metadata (or '?' if absent).
    """
    metadata = _metadata_by_folio(raw_text)
    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for match in _paragraph_loci(raw_text):
        body = _strip_strict_markup(match.group(5))
        tokens: list[str] = []
        for word in re.split(r"[.,\s]+", body):
            word = word.strip().lower()
            if not word or any(ch in BAD_UNIT for ch in word):
                continue
            if not TOKEN_RE.fullmatch(word):
                continue
            tokens.append(word)
        if not tokens:
            continue
        folio = match.group(1)
        quire = metadata.get(folio, {}).get("Q", "?")
        grouped[quire].append(tokens)
    return dict(grouped)


def unit_scale_text_by_quire(raw_text: str) -> dict[str, str]:
    """Render exact unit-scale lines into the segmentation module's interface."""
    return {
        quire: "\n".join(" ".join(line) for line in lines)
        for quire, lines in unit_scale_lines_by_quire(raw_text).items()
        if lines
    }


def strict_space_records(raw_text: str) -> list[dict[str, object]]:
    """Paper-exact strict lines for edge/order and space-sensitivity analyses.

    Mirrors reproduce_space_sensitivity.parse_lines: a malformed token rejects
    the whole line rather than merely dropping that token. Commas mark
    uncertain separators; all other delimiter runs are treated as certain.
    """
    metadata = _metadata_by_folio(raw_text)
    records: list[dict[str, object]] = []
    for match in _paragraph_loci(raw_text):
        cleaned = _strip_strict_markup(match.group(5)).strip().lower()
        parts = re.split(r"([.,\s]+)", cleaned)
        tokens: list[str] = []
        separators: list[str] = []
        usable = True
        for index, part in enumerate(parts):
            if index % 2 == 0:
                if not part:
                    continue
                if not TOKEN_RE.fullmatch(part):
                    usable = False
                    break
                tokens.append(part)
                if len(tokens) >= 2 and len(separators) < len(tokens) - 1:
                    usable = False
                    break
            elif tokens:
                separators.append("u" if "," in part else "c")
        if not usable or len(tokens) < 2 or len(separators) < len(tokens) - 1:
            continue
        folio = match.group(1)
        meta = metadata.get(folio, {})
        records.append(
            {
                "page": folio,
                "quire": meta.get("Q", "?"),
                "currier": meta.get("L", "?"),
                "tokens": tokens,
                "separators": separators[: len(tokens) - 1],
            }
        )
    return records


def conditional_entropy(sequence: list[str]) -> float:
    """Plug-in H(next|current) used by reproduce_headlines.py."""
    if len(sequence) < 2:
        return float("nan")
    bigrams = Counter(zip(sequence, sequence[1:]))
    left = Counter()
    for (current, _), count in bigrams.items():
        left[current] += count
    total = sum(bigrams.values())
    return -sum(
        (count / total) * log2(count / left[current])
        for (current, _), count in bigrams.items()
    )


def paper_character_conditional_entropy(raw_text: str) -> float:
    """Composite-collapsed EVA H(next|current), with boundaries erased."""
    words = [collapse_composites(word) for word in entropy_corpus_tokens(raw_text)]
    symbols = [symbol for word in words for symbol in word]
    return conditional_entropy(symbols)


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts[counts > 0].astype(float)
    if not len(counts):
        return float("nan")
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _pair_mi(x: np.ndarray, y: np.ndarray, ny: int) -> float:
    if len(x) == 0:
        return float("nan")
    _, joint = np.unique(x * ny + y, return_counts=True)
    _, cx = np.unique(x, return_counts=True)
    _, cy = np.unique(y, return_counts=True)
    return _entropy_from_counts(cx) + _entropy_from_counts(cy) - _entropy_from_counts(joint)


def _encode_order_lines(lines: list[list[str]], cap: int = CAP) -> dict[str, object]:
    lines = [line for line in lines if len(line) >= 2]
    flat = [token for line in lines for token in line]
    if not flat:
        return {}
    frequency = Counter(flat)
    retained = {token for token, _ in frequency.most_common(cap)}
    types = sorted(frequency)
    type_index = {token: i for i, token in enumerate(types)}
    token_ids = np.fromiter((type_index[token] for token in flat), dtype=np.int64, count=len(flat))
    line_ids = np.repeat(np.arange(len(lines)), [len(line) for line in lines])

    def code_table(values: list[str]) -> tuple[np.ndarray, int]:
        vocab: dict[str, int] = {}
        codes = np.empty(len(values), dtype=np.int64)
        for i, value in enumerate(values):
            codes[i] = vocab.setdefault(value, len(vocab))
        return codes, len(vocab)

    features: dict[str, np.ndarray] = {}
    sizes: dict[str, int] = {}
    definitions = {
        "last1": [token[-1] for token in types],
        "first1": [token[0] for token in types],
        "ident": [token if token in retained else "<other>" for token in types],
    }
    for name, values in definitions.items():
        codes, size = code_table(values)
        features[name] = codes[token_ids]
        sizes[name] = size
    return {
        "features": features,
        "sizes": sizes,
        "line_ids": line_ids,
        "same_line": line_ids[:-1] == line_ids[1:],
        "n_tokens": len(flat),
    }


def paper_order_metrics(
    raw_text: str,
    *,
    shuffles: int = 100,
    seed: int = ORDER_SEED,
    cap: int = CAP,
) -> dict[str, float | int]:
    """Reproduce the two published within-line shuffle-corrected order targets."""
    records = strict_space_records(raw_text)
    lines = [
        [collapse_composites(token) for token in record["tokens"]]  # type: ignore[index]
        for record in records
    ]
    enc = _encode_order_lines(lines, cap=cap)
    if not enc:
        return {
            "cross_boundary_edge_mi_bits": float("nan"),
            "token_succession_entropy_fraction": float("nan"),
            "order_lines": 0,
            "order_tokens": 0,
        }

    features = enc["features"]  # type: ignore[assignment]
    sizes = enc["sizes"]  # type: ignore[assignment]
    line_ids = enc["line_ids"]  # type: ignore[assignment]
    same = enc["same_line"]  # type: ignore[assignment]
    n = int(enc["n_tokens"])
    positions = np.arange(n)
    obs_left = positions[:-1][same]
    obs_right = positions[1:][same]

    def metric(left: np.ndarray, right: np.ndarray, left_name: str, right_name: str) -> float:
        return _pair_mi(features[left_name][left], features[right_name][right], sizes[right_name])

    edge_observed = metric(obs_left, obs_right, "last1", "first1")
    token_observed = metric(obs_left, obs_right, "ident", "ident")

    ident_counts = np.bincount(features["ident"])
    token_entropy = _entropy_from_counts(ident_counts)
    rng = np.random.default_rng(seed)
    edge_null = np.empty(max(1, int(shuffles)))
    token_null = np.empty(max(1, int(shuffles)))
    for index in range(len(edge_null)):
        keys = rng.random(n)
        perm = np.lexsort((keys, line_ids))
        left = perm[:-1][same]
        right = perm[1:][same]
        edge_null[index] = metric(left, right, "last1", "first1")
        token_null[index] = metric(left, right, "ident", "ident")

    edge_excess = edge_observed - float(edge_null.mean())
    token_excess = token_observed - float(token_null.mean())
    token_share = token_excess / token_entropy if token_entropy > 0 else float("nan")
    return {
        "cross_boundary_edge_mi_bits": float(edge_excess),
        "token_succession_entropy_fraction": float(token_share),
        "edge_observed_mi_bits": float(edge_observed),
        "edge_shuffle_mean_mi_bits": float(edge_null.mean()),
        "token_observed_mi_bits": float(token_observed),
        "token_shuffle_mean_mi_bits": float(token_null.mean()),
        "token_identity_entropy_bits": float(token_entropy),
        "order_lines": len(lines),
        "order_tokens": n,
    }


def automatic_replication_metrics(
    raw_text: str,
    *,
    shuffles: int = 100,
    order_seed: int = ORDER_SEED,
    bpe_checkpoints: tuple[int, ...] = (0, 16, 32, 64),
) -> tuple[dict[str, float], dict[str, object]]:
    """Compute every currently unambiguous Rozanova/Temerev gate target.

    The 2.5% certain-separator crossing target is deliberately not emitted here
    yet: the public space-sensitivity driver reports crossing rates at 32, 64,
    and 128 erased-space BPE merges, while the target citation currently lacks
    an explicit checkpoint. Auto-scoring it before that is resolved would make
    a methodologically ambiguous comparison.
    """
    observed: dict[str, float] = {}
    diagnostics: dict[str, object] = {}

    char_h2 = paper_character_conditional_entropy(raw_text)
    observed["char_conditional_entropy_bits"] = float(char_h2)
    diagnostics["entropy_tokens"] = len(entropy_corpus_tokens(raw_text))

    order = paper_order_metrics(raw_text, shuffles=shuffles, seed=order_seed)
    observed["cross_boundary_edge_mi_bits"] = float(order["cross_boundary_edge_mi_bits"])
    observed["token_succession_entropy_fraction"] = float(order["token_succession_entropy_fraction"])
    diagnostics["order"] = order

    by_quire = unit_scale_text_by_quire(raw_text)
    diagnostics["unit_scale_quires"] = sorted(by_quire)
    if len(by_quire) >= 2:
        rows, selected = cross_fit_bpe_scale_curve(by_quire, checkpoints=bpe_checkpoints)
        diagnostics["bpe_rows"] = rows
        diagnostics["bpe_selected_merges"] = selected
        for row in rows:
            key = f"bpe_crossfit_gap_{int(row['merges'])}_bits"
            observed[key] = float(row["glyph_weighted_dependence_gap_bits"])
        if selected is not None:
            observed["bpe_crossfit_selected_merges"] = float(selected)
    else:
        diagnostics["bpe_rows"] = []
        diagnostics["bpe_selected_merges"] = None

    return observed, diagnostics
