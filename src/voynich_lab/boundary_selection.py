from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log2
from typing import Mapping, Sequence

from .metrics import normalize_text

_BOS = "\x02"
_EOL = "\n"


@dataclass(frozen=True)
class BoundaryOrderScore:
    order: int
    bits_per_symbol: float
    held_out_symbols: int
    groups: int


@dataclass(frozen=True)
class BoundaryOrderSelection:
    selected_order: int
    scores: tuple[BoundaryOrderScore, ...]
    alpha: float
    note: str


def _symbol_lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in normalize_text(text).splitlines() if line.split()]


def _fit_counts(texts: Sequence[str], order: int):
    tables: dict[int, dict[str, Counter[str]]] = {
        k: defaultdict(Counter) for k in range(0, order + 1)
    }
    alphabet: set[str] = {_EOL}
    for text in texts:
        for line in _symbol_lines(text):
            seq = [_BOS] * order + list(line) + [_EOL]
            alphabet.update(line)
            for i in range(order, len(seq)):
                nxt = seq[i]
                tables[0][""][nxt] += 1
                for k in range(1, order + 1):
                    context = "".join(seq[i - k : i])
                    tables[k][context][nxt] += 1
    return tables, tuple(sorted(alphabet))


def _probability(
    tables: dict[int, dict[str, Counter[str]]],
    alphabet: tuple[str, ...],
    history: list[str],
    nxt: str,
    order: int,
    alpha: float,
) -> float:
    vocab = max(1, len(alphabet))
    for k in range(min(order, len(history)), -1, -1):
        context = "".join(history[-k:]) if k else ""
        counts = tables[k].get(context)
        if counts and sum(counts.values()) > 0:
            total = sum(counts.values())
            return (counts.get(nxt, 0) + alpha) / (total + alpha * vocab)
    return 1.0 / vocab


def _held_out_cross_entropy(
    train_texts: Sequence[str],
    held_text: str,
    *,
    order: int,
    alpha: float,
) -> tuple[float, int]:
    tables, alphabet = _fit_counts(train_texts, order)
    if not alphabet:
        return float("inf"), 0
    loss = 0.0
    symbols = 0
    for line in _symbol_lines(held_text):
        history = [_BOS] * order
        for nxt in list(line) + [_EOL]:
            p = _probability(tables, alphabet, history, nxt, order, alpha)
            loss -= log2(max(p, 1e-300))
            symbols += 1
            history.append(nxt)
    return loss, symbols


def select_boundary_markov_order(
    text_by_group: Mapping[str, str],
    *,
    candidates: Sequence[int] = (1, 2, 3, 4, 5),
    alpha: float = 0.5,
) -> BoundaryOrderSelection:
    """Choose boundary-aware character n-gram order by leave-one-group-out log loss.

    Word spaces are ordinary symbols and each line contributes an explicit EOL.
    The grouping should represent genuinely held-out manuscript structure, e.g.
    native quires. For each candidate order, the model is trained on all other
    groups and scored on the omitted group; the lowest glyph/space/EOL weighted
    bits-per-symbol wins. Add-alpha smoothing is used inside the longest seen
    context, with deterministic backoff when a held-out context was unseen.
    """
    groups = [(str(k), str(v)) for k, v in text_by_group.items() if str(v).strip()]
    if len(groups) < 2:
        raise ValueError("boundary order selection needs at least two non-empty groups")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    orders = sorted({int(x) for x in candidates if int(x) >= 1})
    if not orders:
        raise ValueError("candidates must contain at least one positive order")

    scored: list[BoundaryOrderScore] = []
    for order in orders:
        total_loss = 0.0
        total_symbols = 0
        for held_key, held_text in groups:
            train = [text for key, text in groups if key != held_key]
            loss, symbols = _held_out_cross_entropy(
                train, held_text, order=order, alpha=float(alpha)
            )
            total_loss += loss
            total_symbols += symbols
        bps = total_loss / total_symbols if total_symbols else float("inf")
        scored.append(
            BoundaryOrderScore(
                order=order,
                bits_per_symbol=bps,
                held_out_symbols=total_symbols,
                groups=len(groups),
            )
        )

    winner = min(scored, key=lambda row: (row.bits_per_symbol, row.order))
    return BoundaryOrderSelection(
        selected_order=winner.order,
        scores=tuple(scored),
        alpha=float(alpha),
        note=(
            "Order is selected only by held-out predictive log loss, not by how closely a generated "
            "surrogate matches edge MI, hapax fraction, Zipf slope, or any other target statistic."
        ),
    )
