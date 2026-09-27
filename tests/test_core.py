from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.metrics import compute_metrics
from voynich_lab.segmentation import discover_bpe_units
from voynich_lab.surrogates import iid_glyph_surrogate, markov1_surrogate

SAMPLE = "qokeedy qokedy\nchedy qokeedy"


def test_metrics_basic():
    m = compute_metrics(SAMPLE)
    assert m["glyph_count"] > 0
    assert m["glyph_types"] > 1
    assert m["conventional_token_count"] == 4


def test_surrogate_lengths():
    raw_len = len("".join(SAMPLE.split()))
    assert len(iid_glyph_surrogate(SAMPLE)) == raw_len
    assert len(markov1_surrogate(SAMPLE)) == raw_len


def test_bpe_reduces_or_preserves_length():
    units, history = discover_bpe_units(SAMPLE, 8)
    assert len(units) <= len("".join(SAMPLE.split()))
    assert len(history) <= 8
