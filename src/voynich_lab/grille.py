from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import random

from .metrics import normalize_text
from .segmentation import apply_token_bpe_rules, learn_token_bpe_rules, text_to_token_lines

RUGG_FAMILY = "table_grille"
RUGG_CITATION = (
    "Rugg, Gordon (2004). An Elegant Hoax? A Possible Solution to the Voynich Manuscript. "
    "Cryptologia 28(1):31-46. https://doi.org/10.1080/0161-110491892755"
)
RUGG_METHOD_NOTE = (
    "Mechanistic fitted implementation of Rugg's published table-and-grille description: "
    "prefix/root/suffix columns, deliberately empty cells, a three-hole grille, and "
    "non-systematic vertical moves between successive three-column groups. It is not an "
    "exact reconstruction of Rugg's private/manual tables or historical production choices."
)


@dataclass(frozen=True)
class RuggGrilleConfig:
    table_rows: int = 40
    column_groups: int = 12
    bpe_merges: int = 16
    n_grilles: int = 8
    max_grille_offset: int = 4
    max_vertical_step: int = 3
    lines_per_grille: int = 12
    empty_retry_limit: int = 8
    avoid_immediate_repeat: bool = False

    def validate(self) -> None:
        if self.table_rows < 4:
            raise ValueError("table_rows must be at least 4")
        if self.column_groups < 2:
            raise ValueError("column_groups must be at least 2")
        if self.bpe_merges < 0:
            raise ValueError("bpe_merges must be non-negative")
        if self.n_grilles < 1:
            raise ValueError("n_grilles must be at least 1")
        if self.max_grille_offset < 0:
            raise ValueError("max_grille_offset must be non-negative")
        choices = 2 * self.max_grille_offset + 1
        capacity = 1 if self.max_grille_offset == 0 else choices ** 3 - choices
        if self.n_grilles > capacity:
            raise ValueError(f"n_grilles exceeds the {capacity} distinct offset patterns available")
        if self.max_vertical_step < 1:
            raise ValueError("max_vertical_step must be at least 1")
        if self.lines_per_grille < 1:
            raise ValueError("lines_per_grille must be at least 1")
        if self.empty_retry_limit < 1:
            raise ValueError("empty_retry_limit must be at least 1")


@dataclass(frozen=True)
class RuggInventory:
    prefixes: tuple[str, ...]
    roots: tuple[str, ...]
    suffixes: tuple[str, ...]
    prefix_occupancy: float
    root_occupancy: float
    suffix_occupancy: float
    bpe_rules: int

    @property
    def unique_counts(self) -> dict[str, int]:
        return {
            "prefixes": len(set(self.prefixes)),
            "roots": len(set(self.roots)),
            "suffixes": len(set(self.suffixes)),
        }


@dataclass(frozen=True)
class RuggGrilleResult:
    text: str
    seed: int
    config: RuggGrilleConfig
    inventory: RuggInventory
    grille_offsets: tuple[tuple[int, int, int], ...]
    fallback_words: int
    repeated_words_kept: int

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "family": RUGG_FAMILY,
            "input_kind": "reference_fitted_pseudotext",
            "citation": RUGG_CITATION,
            "method_note": RUGG_METHOD_NOTE,
            "seed": self.seed,
            "config": asdict(self.config),
            "inventory_unique_counts": self.inventory.unique_counts,
            "component_occupancy": {
                "prefix": self.inventory.prefix_occupancy,
                "root": self.inventory.root_occupancy,
                "suffix": self.inventory.suffix_occupancy,
            },
            "bpe_rules_learned": self.inventory.bpe_rules,
            "grilles": [list(x) for x in self.grille_offsets],
            "fallback_words": self.fallback_words,
            "repeated_words_kept": self.repeated_words_kept,
        }


@lru_cache(maxsize=8)
def _weighted_component_inventory(text: str, bpe_merges: int) -> RuggInventory:
    lines = text_to_token_lines(text)
    if not lines or not any(lines):
        raise ValueError("Rugg grille generator needs at least one conventional token")
    rules = learn_token_bpe_rules(lines, max_merges=int(bpe_merges))
    segmentation = apply_token_bpe_rules(lines, rules)

    prefixes: list[str] = []
    roots: list[str] = []
    suffixes: list[str] = []
    occupied = Counter(prefix=0, root=0, suffix=0)
    total = 0

    # Iterate token occurrences, not types. Repeated fragments therefore occupy
    # proportionally more table cells without a separate frequency-weight parameter.
    for line in lines:
        for token in line:
            units = tuple(segmentation[token])
            if not units:
                continue
            total += 1
            if len(units) == 1:
                roots.append(units[0])
                occupied["root"] += 1
            elif len(units) == 2:
                prefixes.append(units[0])
                suffixes.append(units[1])
                occupied["prefix"] += 1
                occupied["suffix"] += 1
            else:
                prefixes.append(units[0])
                roots.append("".join(units[1:-1]))
                suffixes.append(units[-1])
                occupied["prefix"] += 1
                occupied["root"] += 1
                occupied["suffix"] += 1

    if total == 0:
        raise ValueError("Rugg grille generator could not derive a component inventory")

    # Sparse edge cases should still have a usable value for every column type.
    all_units = [unit for units in segmentation.values() for unit in units if unit]
    if not all_units:
        all_units = [token for line in lines for token in line if token]
    if not prefixes:
        prefixes.extend(all_units)
    if not roots:
        roots.extend(all_units)
    if not suffixes:
        suffixes.extend(all_units)

    return RuggInventory(
        prefixes=tuple(prefixes),
        roots=tuple(roots),
        suffixes=tuple(suffixes),
        prefix_occupancy=occupied["prefix"] / total,
        root_occupancy=occupied["root"] / total,
        suffix_occupancy=occupied["suffix"] / total,
        bpe_rules=len(rules),
    )


def _populate_column(
    rng: random.Random,
    values: tuple[str, ...],
    occupancy: float,
    rows: int,
) -> list[str]:
    out: list[str] = []
    occupancy = min(1.0, max(0.0, float(occupancy)))
    for _ in range(rows):
        out.append(rng.choice(values) if rng.random() < occupancy else "")
    if not any(out):
        out[rng.randrange(rows)] = rng.choice(values)
    return out


def _build_table(
    rng: random.Random,
    inventory: RuggInventory,
    config: RuggGrilleConfig,
) -> list[tuple[list[str], list[str], list[str]]]:
    groups: list[tuple[list[str], list[str], list[str]]] = []
    for _ in range(config.column_groups):
        groups.append(
            (
                _populate_column(rng, inventory.prefixes, inventory.prefix_occupancy, config.table_rows),
                _populate_column(rng, inventory.roots, inventory.root_occupancy, config.table_rows),
                _populate_column(rng, inventory.suffixes, inventory.suffix_occupancy, config.table_rows),
            )
        )
    return groups


def _make_grilles(rng: random.Random, config: RuggGrilleConfig) -> tuple[tuple[int, int, int], ...]:
    choices = list(range(-config.max_grille_offset, config.max_grille_offset + 1))
    grilles: list[tuple[int, int, int]] = []
    while len(grilles) < config.n_grilles:
        offsets = (rng.choice(choices), rng.choice(choices), rng.choice(choices))
        if len(set(offsets)) == 1 and config.max_grille_offset > 0:
            continue
        if offsets not in grilles:
            grilles.append(offsets)
    return tuple(grilles)


def _vertical_step(rng: random.Random, max_step: int) -> int:
    # Rugg explicitly describes haphazard/arbitrary up/down motion rather than a
    # fixed path. A bounded symmetric step is a reproducible computational proxy.
    steps = [i for i in range(-max_step, max_step + 1) if i != 0]
    return rng.choice(steps)


def _word_from_table(
    table: list[tuple[list[str], list[str], list[str]]],
    group: int,
    row: int,
    offsets: tuple[int, int, int],
) -> str:
    rows = len(table[group][0])
    prefix_col, root_col, suffix_col = table[group]
    p = prefix_col[(row + offsets[0]) % rows]
    r = root_col[(row + offsets[1]) % rows]
    s = suffix_col[(row + offsets[2]) % rows]
    return p + r + s


def generate_rugg_grille(
    reference_text: str,
    seed: int = 0,
    config: RuggGrilleConfig | None = None,
) -> RuggGrilleResult:
    """Generate Rugg-style table-and-grille pseudotext fitted to a reference corpus.

    The fit stage derives reusable prefix/root/suffix fragments with within-token
    BPE, estimates how often each role is absent, and then populates a large table
    with those weighted fragments and blanks. Generation traverses successive
    three-column groups with a fixed three-hole grille plus non-systematic vertical
    steps. Output line/token counts are conditioned on the reference so the
    scorecard compares the mechanism rather than trivial document length.
    """

    cfg = config or RuggGrilleConfig()
    cfg.validate()
    clean = normalize_text(reference_text)
    token_lines = [line.split() for line in clean.splitlines() if line.split()]
    if not token_lines:
        raise ValueError("Rugg grille generator needs non-empty reference text")

    inventory = _weighted_component_inventory(clean, cfg.bpe_merges)
    rng = random.Random(int(seed))
    table = _build_table(rng, inventory, cfg)
    grilles = _make_grilles(rng, cfg)

    output_lines: list[str] = []
    fallback_words = 0
    repeated_words_kept = 0
    row = rng.randrange(cfg.table_rows)
    group = 0
    last_word: str | None = None

    for line_index, reference_tokens in enumerate(token_lines):
        grille = grilles[(line_index // cfg.lines_per_grille) % len(grilles)]
        generated_line: list[str] = []
        for _ in reference_tokens:
            candidate = ""
            for _attempt in range(cfg.empty_retry_limit):
                candidate = _word_from_table(table, group, row, grille)
                if candidate and (not cfg.avoid_immediate_repeat or candidate != last_word):
                    break
                row = (row + _vertical_step(rng, cfg.max_vertical_step)) % cfg.table_rows
            if not candidate or (cfg.avoid_immediate_repeat and candidate == last_word):
                alternatives = [root for root in inventory.roots if root != last_word]
                candidate = rng.choice(alternatives or list(inventory.roots))
                fallback_words += 1
            elif candidate == last_word:
                repeated_words_kept += 1

            generated_line.append(candidate)
            last_word = candidate

            group += 1
            if group >= cfg.column_groups:
                group = 0
            row = (row + _vertical_step(rng, cfg.max_vertical_step)) % cfg.table_rows
        output_lines.append(" ".join(generated_line))

    return RuggGrilleResult(
        text="\n".join(output_lines),
        seed=int(seed),
        config=cfg,
        inventory=inventory,
        grille_offsets=grilles,
        fallback_words=fallback_words,
        repeated_words_kept=repeated_words_kept,
    )


def generate_rugg_grille_text(reference_text: str, seed: int = 0) -> str:
    return generate_rugg_grille(reference_text, seed=seed).text
