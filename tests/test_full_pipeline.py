from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.full_pipeline import PROFILES, profile_names, run_full_pipeline
from voynich_lab.transcript_catalog import BUNDLED_PLAIN_EVA_SAMPLE


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
