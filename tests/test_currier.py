from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.currier import (
    currier_ab_bootstrap,
    currier_boundary_profiles,
    parse_currier_boundary_corpus,
)

RAW = """#=IVTFF Eva- 2.0
<f1r> <! $Q=1 $L=A $H=1 >
<f1r.1,@P0;U> qokedy.daiin,chedy.shol
<f1r.2,+P0;U> qokeedy.shedy.daiin,otedy
<f2r> <! $Q=2 $L=A $H=1 >
<f2r.1,@P0;U> chedy.qokedy,ol.daiin
<f2r.2,+P0;U> shedy.otedy.qokeedy,daiin
<f3r> <! $Q=3 $L=B $H=2 >
<f3r.1,@P0;U> daiin,chedy.qokeedy.shol
<f3r.2,+P0;U> otedy.shedy,qokedy.daiin
<f4r> <! $Q=4 $L=B $H=2 >
<f4r.1,@P0;U> qokeedy.daiin.shedy,ol
<f4r.2,+P0;U> shol,qokedy.otedy.chedy
"""


def test_currier_parser_preserves_native_language_and_quire():
    lines, breaks = parse_currier_boundary_corpus(RAW)
    assert len(lines) == 8
    assert len(breaks) == 4
    assert {line["currier"] for line in lines} == {"A", "B"}
    assert {line["quire"] for line in lines} == {"1", "2", "3", "4"}
    assert {item[3] for item in breaks} == {"A", "B"}


def test_currier_profiles_recompute_stratum_local_scales():
    result = currier_boundary_profiles(RAW)
    assert set(result["profiles"]) == {"A", "B"}
    assert result["profiles"]["A"]["lines"] == 4
    assert result["profiles"]["B"]["lines"] == 4
    assert result["profiles"]["A"]["quires"] == ["1", "2"]
    assert result["profiles"]["B"]["quires"] == ["3", "4"]
    assert result["profiles"]["A"]["uncertain_rate"] == 1 / 3
    assert result["profiles"]["B"]["uncertain_rate"] == 1 / 3
    for language in ("A", "B"):
        profile = result["profiles"][language]
        assert profile["internal_anchor"] != profile["random_anchor"]
        assert profile["raw_gap_bits"] == profile["raw_gap_bits"]
        for key, value in profile["index"].items():
            assert value == value, (language, key)


def test_currier_bootstrap_is_deterministic_and_quire_based():
    first = currier_ab_bootstrap(RAW, repetitions=20, seed=3)
    second = currier_ab_bootstrap(RAW, repetitions=20, seed=3)
    assert first == second
    assert first["repetitions"] == 20
    for scheme in ("independent_strata", "joint_quires"):
        assert set(first[scheme]) == {"uncertain", "first", "mid", "certain", "linebreak"}
        for row in first[scheme].values():
            assert row["valid_draws"] == 20
            assert row["ci90"][0] <= row["ci90"][1]
            assert row["ci95"][0] <= row["ci95"][1]
