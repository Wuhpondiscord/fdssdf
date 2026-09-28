from collections import Counter
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.currier_linestart import (
    GLYPHS,
    jensen_shannon_divergence,
    line_start_analysis,
)

RAW = """#=IVTFF Eva- 2.0
<f1r> <! $Q=1 $L=A $H=1 >
<f1r.1,@P0;U> <%> peka.toka.kefa.dala
<f1r.2,+P0;U> yala.doka.tara.kala
<f2r> <! $Q=2 $L=A $H=1 >
<f2r.1,@P0;U> <%> teka.pala.fara.dora
<f2r.2,+P0;U> kala.yara.daka.tora
<f3r> <! $Q=3 $L=B $H=2 >
<f3r.1,@P0;U> <%> kora.fala.teda.pora
<f3r.2,+P0;U> dara.kora.yeda.fora
<f4r> <! $Q=4 $L=B $H=2 >
<f4r.1,@P0;U> <%> fora.keda.pora.tyda
<f4r.2,+P0;U> tora.deda.kora.yora
"""


def _same_number(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return left == right


def test_jsd_is_symmetric_and_zero_for_identical_distributions():
    left = Counter("aabbbc")
    right = Counter("abbccc")
    assert jensen_shannon_divergence(left, left) == 0.0
    assert abs(
        jensen_shannon_divergence(left, right)
        - jensen_shannon_divergence(right, left)
    ) < 1e-15


def test_line_start_analysis_separates_paragraph_first_and_other_lines():
    result = line_start_analysis(RAW)
    assert result["definition"] == "pstart"
    assert result["representation"] == "collapsed"
    assert result["n_lines"] == 8
    assert result["n_first"] == 4
    assert result["n_other"] == 4
    assert set(result["enrichment_first"]) == set(GLYPHS)
    assert set(result["enrichment_other"]) == set(GLYPHS)
    assert set(result["jsd_by_language"]) == {"A", "B"}
    assert result["jsd"]["pooled"] == result["jsd"]["pooled"]
    assert result["jsd"]["first"] == result["jsd"]["first"]
    assert result["jsd"]["other"] == result["jsd"]["other"]


def test_marker_definition_is_available_as_sensitivity_analysis():
    pstart = line_start_analysis(RAW, definition="pstart")
    marker = line_start_analysis(RAW, definition="marker")
    assert pstart["n_first"] == 4
    assert marker["n_first"] == 4
    assert marker["n_other"] == 4


def test_line_start_null_and_quire_bootstrap_are_deterministic():
    first = line_start_analysis(
        RAW,
        null_draws=20,
        bootstrap_repetitions=20,
        seed=9,
    )
    second = line_start_analysis(
        RAW,
        null_draws=20,
        bootstrap_repetitions=20,
        seed=9,
    )

    assert first["definition"] == second["definition"]
    assert first["representation"] == second["representation"]
    assert first["n_lines"] == second["n_lines"]
    assert first["n_first"] == second["n_first"]
    assert first["n_other"] == second["n_other"]

    for key in ("pooled", "first", "other", "ratio_first_over_other"):
        assert _same_number(first["jsd"][key], second["jsd"][key])
    for pool in ("enrichment_first", "enrichment_other"):
        for glyph in GLYPHS:
            assert _same_number(first[pool][glyph], second[pool][glyph])
    for key in ("pooled", "first", "other"):
        assert _same_number(first["relabel_null"][key], second["relabel_null"][key])
    for key in ("difference_first_minus_other",):
        assert _same_number(first["quire_bootstrap"][key], second["quire_bootstrap"][key])
    for key in ("difference_ci95", "first_ci95", "other_ci95"):
        assert all(
            _same_number(left, right)
            for left, right in zip(first["quire_bootstrap"][key], second["quire_bootstrap"][key])
        )

    assert first["relabel_null"]["draws"] == 20
    assert first["quire_bootstrap"]["draws"] == 20
    low, high = first["quire_bootstrap"]["difference_ci95"]
    assert low <= high


def test_raw_representation_path_runs():
    result = line_start_analysis(RAW, representation="raw")
    assert result["representation"] == "raw"
    assert result["n_lines"] == 8
