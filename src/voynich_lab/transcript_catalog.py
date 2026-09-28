from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import urllib.request


PINNED_ZL3B_URL = (
    "https://raw.githubusercontent.com/lrozanova/voynich-units/"
    "956a7c4fc39981f4d116fa3f4edfccce6d065571/"
    "voynich_decipherment_repro_bundle/voynich_calibration_sources/ZL3b.txt"
)

PLAIN_TEXT_EXAMPLE = """qokeedy qokedy chedy
shedy daiin otedy
fachys ykal ar ataiin shol
"""

IVTFF_EXAMPLE = """#=IVTFF Eva- 2.0
<f1r> <! $Q=A $P=A $I=H $L=A $H=1 >
<f1r.1,@P0;U> fachys.ykal.ar,ataiin.shol
<f1r.2,@P0;U> shory.cth[e:o]res.y.k[a:o]l
<f1r.3,@P0;U> qokedy.qokeedy
<f2r> <! $Q=B $P=A $I=T $L=B $H=2 >
<f2r.1,@P0;U> chedy.qokeedy.daiin
"""

# Kept intentionally small so the Space always has an offline IVTFF example even
# if the pinned full-corpus download is unavailable.
BUNDLED_IVTFF_SAMPLE = IVTFF_EXAMPLE

BUNDLED_PLAIN_EVA_SAMPLE = """fachys ykal ar ataiin shol
shory ctheres y kal
qokedy qokeedy
chedy qokeedy daiin
shedy qokedy otedy
ol daiin chedy qokeedy
"""

SYNTHETIC_BOUNDARY_SAMPLE = """xoka yola xeka yela xoka yola
xola yeka xala yela xola yeka
xoka yola xeka yela xoka yola
xola yeka xala yela xola yeka
xoka yola xeka yela xoka yola
xola yeka xala yela xola yeka
"""


@dataclass(frozen=True)
class TranscriptPreset:
    name: str
    description: str
    format: str
    text: str | None = None
    url: str | None = None

    @property
    def is_remote(self) -> bool:
        return self.url is not None


PRESETS: dict[str, TranscriptPreset] = {
    "Voynich ZL3b — full pinned IVTFF manuscript": TranscriptPreset(
        name="Voynich ZL3b — full pinned IVTFF manuscript",
        description=(
            "Full ZL3b IVTFF transcription pinned to the same Rozanova/Temerev reproduction commit "
            "used by scripts/evaluate_nulls.py. Downloads only when selected."
        ),
        format="IVTFF",
        url=PINNED_ZL3B_URL,
    ),
    "Voynich IVTFF — bundled metadata demo": TranscriptPreset(
        name="Voynich IVTFF — bundled metadata demo",
        description=(
            "Small offline IVTFF sample with page headers, quire labels, Currier A/B labels, "
            "certain/uncertain separators, and alternate readings."
        ),
        format="IVTFF",
        text=BUNDLED_IVTFF_SAMPLE,
    ),
    "Plain EVA-style — bundled demo": TranscriptPreset(
        name="Plain EVA-style — bundled demo",
        description=(
            "Small offline plain-text sample. One manuscript line per line; whitespace separates "
            "conventional transcription tokens. No IVTFF metadata is required."
        ),
        format="plain text",
        text=BUNDLED_PLAIN_EVA_SAMPLE,
    ),
    "Synthetic boundary-coupling stress test": TranscriptPreset(
        name="Synthetic boundary-coupling stress test",
        description=(
            "Artificial text for checking boundary-sensitive metrics and null models. This is not a "
            "Voynich transcription and should never be used as manuscript evidence."
        ),
        format="synthetic plain text",
        text=SYNTHETIC_BOUNDARY_SAMPLE,
    ),
}

DEFAULT_PRESET = "Voynich IVTFF — bundled metadata demo"


def preset_names() -> list[str]:
    return list(PRESETS)


def preset_description(name: str) -> str:
    preset = PRESETS[name]
    location = "remote pinned source" if preset.is_remote else "bundled with the app"
    return f"**Format:** {preset.format}  \n**Source:** {location}  \n{preset.description}"


@lru_cache(maxsize=2)
def _download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "voynich-structure-lab/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310 - fixed HTTPS preset URL
        payload = response.read(12_000_001)
    if len(payload) > 12_000_000:
        raise ValueError("built-in transcript exceeds the 12 MB safety limit")
    return payload.decode("utf-8", errors="replace")


def load_preset(name: str) -> tuple[str, TranscriptPreset]:
    if name not in PRESETS:
        raise KeyError(f"unknown transcript preset: {name}")
    preset = PRESETS[name]
    if preset.text is not None:
        return preset.text, preset
    if preset.url is None:
        raise ValueError(f"preset {name!r} has neither bundled text nor a source URL")
    text = _download_text(preset.url)
    if not text.strip():
        raise ValueError(f"preset {name!r} downloaded an empty transcript")
    return text, preset
