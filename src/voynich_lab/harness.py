from __future__ import annotations

from functools import partial

import pandas as pd

from .metrics import compute_metrics
from .surrogates import (
    block_shuffle_surrogate,
    iid_glyph_surrogate,
    markov1_surrogate,
    markov_k_surrogate,
    position_conditioned_markov_surrogate,
    token_shuffle_surrogate,
)

METRIC_COLUMNS = ["glyph_entropy_bits", "conditional_entropy_order1_bits", "adjacent_glyph_mi_bits", "cross_token_edge_mi_bits", "hapax_fraction", "mean_token_length", "zipf_loglog_slope"]

# Nested null ladder (N0 - N6; N7 quire-conditioned lives in surrogates.py
# separately since it needs a folio->quire map rather than raw text).
# Ordered from weakest to strongest so summary tables read as a ladder.
DEFAULT_GENERATORS = {
    "N0_iid_glyph": iid_glyph_surrogate,
    "N1_markov1_glyph": markov1_surrogate,
    "N2_markov3_glyph": partial(markov_k_surrogate, order=3),
    "N3_markov5_glyph": partial(markov_k_surrogate, order=5),
    "N4_block_shuffle_3": partial(block_shuffle_surrogate, block_size=3),
    "N5_token_shuffle": token_shuffle_surrogate,
    "N6_position_conditioned_markov": position_conditioned_markov_surrogate,
}


def run_harness(
    text: str,
    replicates: int = 20,
    seed: int = 0,
    generators: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    replicates = max(1, min(int(replicates), 250))
    rows: list[dict[str, object]] = [{"source": "VMS/input", "replicate": 0, **compute_metrics(text)}]
    gens = generators if generators is not None else DEFAULT_GENERATORS
    for name, fn in gens.items():
        for r in range(replicates):
            rows.append({"source": name, "replicate": r + 1, **compute_metrics(fn(text, seed=seed + r))})
    raw = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for source, group in raw.groupby("source", sort=False):
        row: dict[str, object] = {"source": source, "n": len(group)}
        for col in METRIC_COLUMNS:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}__mean"] = vals.mean()
            row[f"{col}__sd"] = vals.std(ddof=1)
        summary_rows.append(row)
    return raw, pd.DataFrame(summary_rows)
