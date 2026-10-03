from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.full_pipeline import MIN_GLYPHS_FOR_RELIABLE_STAGES, PROFILES, pipeline_summary_markdown, profile_names, run_full_pipeline
from voynich_lab.ivtff import parse_ivtff
from voynich_lab.metrics import normalize_text
from voynich_lab.transcript_catalog import BUNDLED_PLAIN_EVA_SAMPLE, DEFAULT_PRESET, load_preset


def test_pipeline_profiles_are_ordered_for_more_work():
    assert profile_names() == ["Quick", "Standard", "Thorough"]
    assert PROFILES["Quick"].scorecard_replicates < PROFILES["Standard"].scorecard_replicates
    assert PROFILES["Standard"].scorecard_replicates < PROFILES["Thorough"].scorecard_replicates
    assert PROFILES["Quick"].replication_shuffles < PROFILES["Thorough"].replication_shuffles


def test_quick_pipeline_runs_plain_text_and_skips_metadata_only_stages():
    text = "\n".join([BUNDLED_PLAIN_EVA_SAMPLE.strip()] * 4)
    report = run_full_pipeline("", text, profile="Quick", seed=7)

    assert report["input_kind"] == "plain text"
    assert report["metrics"]["corpus"]["glyph_count"] > 0
    assert report["adversarial_candidate"].startswith("boundary-aware order-")
    assert report["ivtff_audit"]["skipped"]
    assert report["replication_gate"]["skipped"]
    assert report["completed_stages"] >= 6
    assert len(report["stages"]) >= 10
    json.dumps(report)  # entire report must be UI/API serializable


def test_pipeline_requires_loaded_text():
    with pytest.raises(ValueError):
        run_full_pipeline("", "", profile="Quick")


def test_corpus_scale_flags_small_input_as_demo_scale():
    text = "\n".join([BUNDLED_PLAIN_EVA_SAMPLE.strip()] * 4)
    report = run_full_pipeline("", text, profile="Quick", seed=7, source_label="unit-test sample")

    scale = report["corpus_scale"]
    assert scale["source_label"] == "unit-test sample"
    assert scale["glyph_count"] < MIN_GLYPHS_FOR_RELIABLE_STAGES
    assert scale["is_demo_scale"] is True
    assert scale["min_glyphs_for_reliable_stages"] == MIN_GLYPHS_FOR_RELIABLE_STAGES
    assert "200,000 glyphs" in scale["note"]

    summary = pipeline_summary_markdown(report)
    assert "Demo-scale input" in summary
    assert "unit-test sample" in summary


def test_corpus_scale_source_label_defaults_to_unspecified():
    text = "\n".join([BUNDLED_PLAIN_EVA_SAMPLE.strip()] * 4)
    report = run_full_pipeline("", text, profile="Quick", seed=7)
    assert report["corpus_scale"]["source_label"] == "unspecified"


def test_corpus_scale_does_not_flag_large_input():
    text = "\n".join([BUNDLED_PLAIN_EVA_SAMPLE.strip()] * 400)
    report = run_full_pipeline("", text, profile="Quick", seed=7, source_label="large synthetic corpus")

    scale = report["corpus_scale"]
    assert scale["glyph_count"] >= MIN_GLYPHS_FOR_RELIABLE_STAGES
    assert scale["is_demo_scale"] is False

    summary = pipeline_summary_markdown(report)
    assert "Demo-scale input" not in summary


def test_default_preset_currier_stage_runs_without_error():
    raw, _preset = load_preset(DEFAULT_PRESET)
    doc = parse_ivtff(raw)
    assert doc is not None

    analysis = normalize_text(raw)
    report = run_full_pipeline(raw, analysis, profile="Quick", seed=0, source_label=DEFAULT_PRESET)

    assert report["error_stages"] == 0
    currier = report.get("currier", {})
    assert "error" not in currier
    json.dumps(report)
