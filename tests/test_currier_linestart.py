from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.currier_linestart import (
    GLYPHS,
    jensen_shannon_divergence,
    line_start_analysis,
)
from collections import Counter

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
    assert first == second
    assert first["relabel_null"]["draws"] == 20
    assert first["quire_bootstrap"]["draws"] == 20
    low, high = first["quire_bootstrap"]["difference_ci95"]
    assert low <= high


def test_raw_representation_path_runs():
    result = line_start_analysis(RAW, representation="raw")
    assert result["representation"] == "raw"
    assert result["n_lines"] == 8
