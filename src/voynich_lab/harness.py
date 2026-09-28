from __future__ import annotations

import pandas as pd

from .metrics import compute_metrics
from .surrogates import NULL_LADDER

METRIC_COLUMNS = ["glyph_entropy_bits", "conditional_entropy_order1_bits", "adjacent_glyph_mi_bits", "cross_token_edge_mi_bits", "hapax_fraction", "mean_token_length", "zipf_loglog_slope"]

# The ladder itself lives in surrogates.NULL_LADDER (importable without pandas); re-exported under its
# historical name so existing callers keep working.
DEFAULT_GENERATORS = NULL_LADDER


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
