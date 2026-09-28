"""Reproduce the null-ladder evaluation on the pinned ZL3b transcription.

    python scripts/evaluate_nulls.py [--corpus ZL3b.txt] [--replicates 10] [--seed 0]

Prints, for every rung of the null ladder (and its per-Currier / per-quire stratified twins for the
boundary-aware chain), the mean +/- sd of four diagnostic features over Monte-Carlo replicates and the
observed manuscript's z-score against them; then the heterogeneity control for the block classifier.

What to look for (values from the pinned corpus; see README "Null-ladder findings"):

* Space-free Markov nulls (N1-N3, N6) and the block shuffle overshoot cross-boundary edge MI by 3-6x,
  because they model boundary transitions as if spaces did not exist; the token shuffle undershoots it.
* The boundary-aware n-gram (N2b/N3b) reproduces hapax fraction / Zipf slope / mean token length that no
  space-free null can, and its edge MI saturates by order 2 -- but stays below the observed value.
* Stratifying that chain by Currier language or quire does NOT close the edge-MI gap.
* The block classifier's separation from global nulls is substantially stratum composition.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.ivtff import parse_ivtff  # noqa: E402
from voynich_lab.metrics import compute_metrics  # noqa: E402
from voynich_lab.stratified import heterogeneity_control, line_strata, stratified_generator  # noqa: E402
from voynich_lab.surrogates import NULL_LADDER as DEFAULT_GENERATORS, bind_generator, boundary_markov_surrogate, token_shuffle_surrogate  # noqa: E402

PINNED_CORPUS_URL = (
    "https://raw.githubusercontent.com/lrozanova/voynich-units/"
    "956a7c4fc39981f4d116fa3f4edfccce6d065571/voynich_decipherment_repro_bundle/"
    "voynich_calibration_sources/ZL3b.txt"
)
FEATURES = {
    "cross_token_edge_mi_bits": "edgeMI",
    "conditional_entropy_order1_bits": "H(1)",
    "hapax_fraction": "hapax",
    "zipf_loglog_slope": "zipf",
}


def _load(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    with urllib.request.urlopen(PINNED_CORPUS_URL, timeout=60) as resp:  # noqa: S310 (pinned https URL)
        return resp.read().decode("utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="local ZL3b.txt (default: download the pinned copy)")
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    doc = parse_ivtff(_load(args.corpus))
    text = doc.to_analysis_text()
    observed = compute_metrics(text)
    print(f"{len(doc.lines):,} loci, {int(observed['glyph_count']):,} glyphs, {int(observed['conventional_token_count']):,} tokens")
    print("observed  " + "  ".join(f"{short}={observed[f]:.3f}" for f, short in FEATURES.items()))

    boundary2 = bind_generator(boundary_markov_surrogate, order=2)
    rungs = dict(DEFAULT_GENERATORS)
    rungs["N7a_boundary_markov2_per_currier"] = stratified_generator(line_strata(doc, "currier"), boundary2)
    rungs["N7b_boundary_markov2_per_quire"] = stratified_generator(line_strata(doc, "quire"), boundary2)

    print(f"\nz = (observed - null mean) / null sd over {args.replicates} replicates (+ means observed is higher)\n")
    header = f"{'null':<36}" + "".join(f"{short:>24}" for short in FEATURES.values())
    print(header + "\n" + "-" * len(header))
    for name, fn in rungs.items():
        start = time.time()
        samples = {f: [] for f in FEATURES}
        for r in range(args.replicates):
            metrics = compute_metrics(fn(text, args.seed + r))
            for f in FEATURES:
                samples[f].append(metrics[f])
        cells = []
        for f in FEATURES:
            arr = np.asarray(samples[f], dtype=float)
            sd = arr.std(ddof=1)
            z = (observed[f] - arr.mean()) / sd if sd > 0 else float("nan")
            cells.append(f"{arr.mean():8.3f}+/-{sd:5.3f} z={z:+7.1f}")
        print(f"{name:<36}" + "".join(f"{c:>24}" for c in cells) + f"   ({time.time() - start:.0f}s)")

    print("\nHeterogeneity control (block classifier, global vs per-quire null):")
    for label, fn in [("N1 order-1 Markov", DEFAULT_GENERATORS["N1_markov1_glyph"]), ("N5 token shuffle", token_shuffle_surrogate)]:
        r = heterogeneity_control(text, line_strata(doc, "quire"), fn, seed=args.seed)
        print(
            f"  {label:<20} global AUC={r.auc_global:.3f}  per-quire AUC={r.auc_stratified:.3f}  "
            f"drop={r.auc_drop:+.3f}  stratified CI includes 0.5: {r.stratified_ci_includes_chance}"
        )


if __name__ == "__main__":
    main()
