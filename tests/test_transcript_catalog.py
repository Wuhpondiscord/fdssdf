from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.ivtff import is_ivtff, parse_ivtff
from voynich_lab.metrics import normalize_text
from voynich_lab.transcript_catalog import (
    DEFAULT_PRESET,
    IVTFF_EXAMPLE,
    PLAIN_TEXT_EXAMPLE,
    PRESETS,
    load_preset,
    preset_description,
    preset_names,
)


def test_catalog_has_offline_and_full_manuscript_choices():
    names = preset_names()
    assert DEFAULT_PRESET in names
    assert any("full pinned IVTFF manuscript" in name for name in names)
    assert any(not PRESETS[name].is_remote for name in names)


def test_default_bundled_ivtff_is_parseable_and_has_metadata():
    text, preset = load_preset(DEFAULT_PRESET)
    assert preset.is_remote is False
    assert is_ivtff(text)
    doc = parse_ivtff(text)
    assert len(doc.lines) >= 4
    assert set(doc.quires_present()) >= {"A", "B"}
    assert {line.currier_language for line in doc.lines if line.currier_language} >= {"A", "B"}


def test_plain_preset_normalizes_without_ivtff_markup():
    name = "Plain EVA-style — bundled demo"
    text, preset = load_preset(name)
    assert preset.format == "plain text"
    assert not is_ivtff(text)
    assert "qokeedy" in normalize_text(text)


def test_synthetic_preset_is_explicitly_labeled_synthetic():
    name = "Synthetic boundary-coupling stress test"
    text, preset = load_preset(name)
    assert text.strip()
    assert "synthetic" in preset.format
    assert "not a Voynich transcription" in preset.description


def test_format_examples_cover_plain_and_ivtff():
    assert "qokeedy qokedy" in PLAIN_TEXT_EXAMPLE
    assert "#=IVTFF" in IVTFF_EXAMPLE
    assert "$Q=A" in IVTFF_EXAMPLE and "$L=A" in IVTFF_EXAMPLE


def test_preset_description_reports_source_and_format():
    bundled = preset_description(DEFAULT_PRESET)
    remote = preset_description("Voynich ZL3b — full pinned IVTFF manuscript")
    assert "bundled with the app" in bundled
    assert "remote pinned source" in remote
    assert "**Format:**" in bundled
