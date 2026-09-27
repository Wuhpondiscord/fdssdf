from __future__ import annotations

import pandas as pd

from .metrics import compute_metrics
from .surrogates import iid_glyph_surrogate, markov1_surrogate, token_shuffle_surrogate

METRIC_COLUMNS = ["glyph_entropy_bits", "conditional_entropy_order1_bits", "adjacent_glyph_mi_bits", "cross_token_edge_mi_bits", "hapax_fraction", "mean_token_length", "zipf_loglog_slope"]


def run_harness(text: str, replicates: int = 20, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    replicates = max(1, min(int(replicates), 250))
    rows: list[dict[str, object]] = [{"source": "VMS/input", "replicate": 0, **compute_metrics(text)}]
    generators = {"iid_glyph": iid_glyph_surrogate, "markov1_glyph": markov1_surrogate, "token_shuffle": token_shuffle_surrogate}
    for name, fn in generators.items():
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
