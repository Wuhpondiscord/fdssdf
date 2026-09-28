from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.currier_report import LINESTART_STATED, build_currier_report

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


def test_report_shapes_and_provenance():
    report = build_currier_report(
        RAW,
        boundary_bootstrap_repetitions=20,
        seed=7,
    )
    assert len(report["boundary_rows"]) == 5
    assert len(report["stratum_rows"]) == 2
    assert len(report["bootstrap_rows"]) == 10
    assert len(report["line_start_jsd_rows"]) == 4
    assert len(report["line_start_enrichment_rows"]) == 12
    assert report["provenance"]["boundary_bootstrap_repetitions"] == 20
    assert report["provenance"]["seed"] == 7
    assert report["provenance"]["boundary_bootstrap_estimable"] is False
    assert "stratum-local PMI" in report["provenance"]["boundary_method"]
    assert "executable-parity" in report["provenance"]["line_start_method"]


def test_sparse_report_marks_bootstrap_non_estimable_instead_of_crashing():
    report = build_currier_report(
        RAW,
        boundary_bootstrap_repetitions=20,
        seed=7,
    )
    assert all(row["valid_draws"] == 0 for row in report["bootstrap_rows"])
    assert all(row["ci90_low"] != row["ci90_low"] for row in report["bootstrap_rows"])
    assert all(row["ci90_high"] != row["ci90_high"] for row in report["bootstrap_rows"])
    assert all(row["ci90_contains_zero"] is False for row in report["bootstrap_rows"])
    assert "not estimable" in report["interpretation"].lower()


def test_report_keeps_executable_and_stated_values_separate():
    report = build_currier_report(
        RAW,
        boundary_bootstrap_repetitions=10,
        seed=2,
    )
    jsd = {row["quantity"]: row for row in report["line_start_jsd_rows"]}
    assert jsd["first"]["bundled_STATED_constant"] == LINESTART_STATED["jsd"]["first"]
    assert jsd["pooled"]["bundled_STATED_constant"] == LINESTART_STATED["jsd"]["pooled"]
    for row in report["line_start_jsd_rows"]:
        assert row["pinned_executable"] == row["pinned_executable"]
        assert row["executable_minus_STATED"] == (
            row["pinned_executable"] - row["bundled_STATED_constant"]
        )


def test_report_interpretation_mentions_quire_uncertainty_and_stated_discrepancy():
    report = build_currier_report(
        RAW,
        boundary_bootstrap_repetitions=10,
        seed=3,
    )
    text = report["interpretation"].lower()
    assert "quire" in text
    assert "pooled a/b" in text
    assert "stated" in text
    assert "pinned-executable" in text
