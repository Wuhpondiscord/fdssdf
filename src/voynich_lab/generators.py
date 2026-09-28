from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import random
import unicodedata
from collections import defaultdict
from typing import Mapping


NAIBBE_UPSTREAM_COMMIT = "f2675ec5dd275268bc64dd48ea64fc0e0e9827a2"
NAIBBE_SOURCE_REPOSITORY = "greshko/naibbe-cipher"
NAIBBE_SOURCE_FILE = "naibbe.py"
NAIBBE_CITATION = (
    "Greshko, Michael A. (2025). The Naibbe cipher: a substitution cipher that "
    "encrypts Latin and Italian as Voynich Manuscript-like ciphertext. Cryptologia. "
    "https://doi.org/10.1080/01611194.2025.2566408"
)
NAIBBE_FAMILY = "meaningful_cipher"

ALPHABET = tuple("abcdefghijklmnopqrstuvwxyz")
TABLES = ("alpha", "beta1", "beta2", "beta3", "gamma1", "gamma2")
STATES = ("unigram", "prefix", "suffix")
CARD_WEIGHTS = {
    False: {"alpha": 20, "beta1": 8, "beta2": 8, "beta3": 8, "gamma1": 4, "gamma2": 4},
    True: {"alpha": 28, "beta1": 14, "beta2": 11, "beta3": 11, "gamma1": 7, "gamma2": 7},
}
_DATA_PATH = Path(__file__).with_name("data") / "naibbe_tables.csv"


@dataclass(frozen=True)
class NaibbeConfig:
    """Paper-facing Naibbe defaults from upstream ``naibbe.py``.

    ``use_78_card_deck=False`` is intentional. The upstream ``naibbe_v2.py``
    changes this default and adds a second cross-bigram collision rule; that is
    a distinct generator variant and is not silently folded into this baseline.
    """

    respacing: int = 17
    use_78_card_deck: bool = False
    space_removal_rate: float = 0.03
    unambiguous: bool = True


@dataclass(frozen=True)
class NaibbeResult:
    ciphertext: str
    canonical_ciphertext: str
    respaced_plaintext: str
    ambiguity_retries: int
    seed: int
    config: NaibbeConfig

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "family": NAIBBE_FAMILY,
            "input_kind": "meaningful_plaintext",
            "source_repository": NAIBBE_SOURCE_REPOSITORY,
            "source_file": NAIBBE_SOURCE_FILE,
            "upstream_commit": NAIBBE_UPSTREAM_COMMIT,
            "citation": NAIBBE_CITATION,
            "seed": self.seed,
            "config": asdict(self.config),
        }


def load_naibbe_glyph_map(path: Path | str | None = None) -> dict[str, str]:
    """Load the exact upstream Naibbe substitution table copied under its license."""

    table_path = Path(path) if path is not None else _DATA_PATH
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        mapping = {row["code"]: row["glyphs"] for row in rows}
    if not mapping:
        raise ValueError(f"Naibbe mapping table is empty: {table_path}")
    return mapping


def _placeholder_tables() -> dict[str, dict[tuple[str, str], str]]:
    table: dict[str, dict[tuple[str, str], str]] = defaultdict(dict)
    for table_name in TABLES:
        for state in STATES:
            for letter in ALPHABET:
                table[table_name][(state, letter)] = f"{state}_{table_name}_{letter}"
    return dict(table)


def clean_naibbe_line(text: str) -> str:
    """Normalize plaintext exactly as the paper-facing upstream implementation."""

    normalized = unicodedata.normalize("NFD", text)
    no_diacritics = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    replacements = {
        "æ": "ae",
        "Æ": "ae",
        "œ": "oe",
        "Œ": "oe",
        "ð": "d",
        "Ð": "d",
        "þ": "th",
        "Þ": "th",
        "ł": "l",
        "Ł": "l",
        "ß": "ss",
        "ø": "o",
        "Ø": "o",
    }
    replaced = "".join(replacements.get(c, c) for c in no_diacritics)
    cleaned = "".join(c for c in replaced if c.isalpha()).upper()
    cleaned = cleaned.replace("W", "UU").replace("J", "I").replace("K", "C")
    return cleaned.lower()


def _create_card_deck(rng: random.Random, use_78: bool) -> list[str]:
    deck: list[str] = []
    for table, count in CARD_WEIGHTS[use_78].items():
        deck.extend([table] * count)
    rng.shuffle(deck)
    return deck


def _respace_plaintext(text: str, rng: random.Random, respacing: int) -> list[str]:
    text = text.lower().replace(" ", "")
    i = 0
    output: list[str] = []
    while i < len(text):
        if i == len(text) - 1 or rng.random() < (respacing / 36):
            output.append(text[i])
            i += 1
        else:
            output.append(text[i : i + 2])
            i += 2
    return output


def _encrypt_line(
    plaintext: str,
    *,
    rng: random.Random,
    config: NaibbeConfig,
    tables: Mapping[str, Mapping[tuple[str, str], str]],
    glyph_map: Mapping[str, str],
    unigram_glyphs: set[str],
) -> tuple[list[str], list[str], int]:
    ngrams = _respace_plaintext(plaintext, rng, config.respacing)
    ciphertext: list[str] = []
    deck = _create_card_deck(rng, config.use_78_card_deck)
    deck_index = 0
    ambiguity_retries = 0

    def draw_table() -> str:
        nonlocal deck, deck_index
        if deck_index >= len(deck):
            deck = _create_card_deck(rng, config.use_78_card_deck)
            deck_index = 0
        table = deck[deck_index]
        deck_index += 1
        return table

    for token in ngrams:
        if len(token) == 1:
            table = draw_table()
            code = tables[table][("unigram", token)]
            ciphertext.append(glyph_map.get(code, code))
            continue

        first, second = token[0], token[1]
        if config.unambiguous:
            while True:
                prefix_table = draw_table()
                prefix_code = tables[prefix_table][("prefix", first)]
                prefix_glyph = glyph_map.get(prefix_code, prefix_code)

                suffix_table = draw_table()
                suffix_code = tables[suffix_table][("suffix", second)]
                suffix_glyph = glyph_map.get(suffix_code, suffix_code)

                combined = prefix_glyph + suffix_glyph
                if combined not in unigram_glyphs:
                    ciphertext.append(combined)
                    break
                ambiguity_retries += 1
        else:
            prefix_table = draw_table()
            prefix_code = tables[prefix_table][("prefix", first)]
            prefix_glyph = glyph_map.get(prefix_code, prefix_code)
            suffix_table = draw_table()
            suffix_code = tables[suffix_table][("suffix", second)]
            suffix_glyph = glyph_map.get(suffix_code, suffix_code)
            ciphertext.append(prefix_glyph + suffix_glyph)

    return ngrams, ciphertext, ambiguity_retries


def _remove_output_spaces(line: str, rng: random.Random, drop_rate: float) -> str:
    if drop_rate <= 0:
        return line.strip()
    if drop_rate >= 1:
        return line.replace(" ", "")
    tokens = line.strip().split()
    if len(tokens) < 2:
        return line.strip()
    output = tokens[0]
    for token in tokens[1:]:
        if rng.random() < drop_rate:
            output += token
        else:
            output += " " + token
    return output


def generate_naibbe(
    plaintext: str,
    seed: int = 0,
    config: NaibbeConfig | None = None,
    *,
    glyph_map: Mapping[str, str] | None = None,
) -> NaibbeResult:
    """Encrypt meaningful plaintext with the pinned paper-facing Naibbe baseline.

    Randomness is isolated in ``random.Random(seed)``. For a given seed this
    preserves the upstream order of ``random()`` and ``shuffle()`` calls while
    avoiding changes to Python's module-level RNG state.
    """

    cfg = config or NaibbeConfig()
    if not 0 <= cfg.respacing <= 36:
        raise ValueError("respacing must be in [0, 36]")
    if not 0 <= cfg.space_removal_rate <= 1:
        raise ValueError("space_removal_rate must be in [0, 1]")

    mapping = dict(glyph_map) if glyph_map is not None else load_naibbe_glyph_map()
    tables = _placeholder_tables()
    unigram_glyphs = {
        glyph for code, glyph in mapping.items() if code.startswith("unigram_")
    }
    rng = random.Random(int(seed))

    final_lines: list[str] = []
    canonical_lines: list[str] = []
    respaced_plaintext_lines: list[str] = []
    retries = 0

    # split("\n") rather than splitlines() preserves the upstream file-loop
    # behavior for a final newline by representing the terminal blank line.
    for line in plaintext.split("\n"):
        cleaned = clean_naibbe_line(line)
        if not cleaned:
            final_lines.append("")
            canonical_lines.append("")
            respaced_plaintext_lines.append("")
            continue
        ngrams, tokens, line_retries = _encrypt_line(
            cleaned,
            rng=rng,
            config=cfg,
            tables=tables,
            glyph_map=mapping,
            unigram_glyphs=unigram_glyphs,
        )
        canonical = " ".join(tokens)
        final_lines.append(_remove_output_spaces(canonical, rng, cfg.space_removal_rate))
        canonical_lines.append(canonical)
        respaced_plaintext_lines.append(" ".join(ngrams))
        retries += line_retries

    return NaibbeResult(
        ciphertext="\n".join(final_lines),
        canonical_ciphertext="\n".join(canonical_lines),
        respaced_plaintext="\n".join(respaced_plaintext_lines),
        ambiguity_retries=retries,
        seed=int(seed),
        config=cfg,
    )


def generate_naibbe_text(plaintext: str, seed: int = 0) -> str:
    """Convenience wrapper returning only the paper-style respaced ciphertext."""

    return generate_naibbe(plaintext, seed=seed).ciphertext
