from __future__ import annotations

from pathlib import Path

import gradio as gr

import legacy_app as legacy
from voynich_lab.transcript_catalog import (
    DEFAULT_PRESET,
    IVTFF_EXAMPLE,
    PLAIN_TEXT_EXAMPLE,
    load_preset,
    preset_description,
    preset_names,
)

APP_TITLE = legacy.APP_TITLE


def _fan_out_loaded(result):
    raw, analysis, status, preview = result
    return raw, analysis, status, preview, status, preview


def load_builtin_ui(name: str):
    try:
        raw, preset = load_preset(name)
    except Exception as exc:
        raise gr.Error(f"Could not load built-in transcript {name!r}: {exc}") from exc
    result = legacy.load_uploaded(None, raw)
    raw_text, analysis, status, preview = result
    status = status.replace("pasted text", f"built-in preset '{preset.name}'")
    if preset.is_remote:
        status += " Source is pinned to the project's reproduction commit."
    return _fan_out_loaded((raw_text, analysis, status, preview))


def load_custom_ui(file_path: str | None, pasted: str):
    if not (pasted and pasted.strip()) and not file_path:
        raise gr.Error("Paste a transcription or choose a .txt, .eva, or .ivtff file first.")
    return _fan_out_loaded(legacy.load_uploaded(file_path, pasted))


def preset_info_ui(name: str):
    return preset_description(name)


FORMAT_HELP = f"""
### Accepted transcript formats

**Plain EVA-style / plain text**

Use one manuscript line per line. Separate conventional transcription tokens with whitespace. This is the simplest format when you do not need quire, Currier-language, hand, folio, or separator-certainty metadata.

```text
{PLAIN_TEXT_EXAMPLE.rstrip()}
```

**IVTFF**

IVTFF is preferred for analyses that need native manuscript metadata. Page headers can contain variables such as `$Q` (quire), `$L` (Currier language), `$H` (hand), and `$I` (illustration type). Locus lines carry the transcription. In ZL-style input, `.` and `,` can preserve certain versus uncertain separators, and alternate readings such as `[e:o]` are accepted by the parser.

```text
{IVTFF_EXAMPLE.rstrip()}
```

The loader detects IVTFF automatically. Metadata is excluded from glyph statistics; plain text is analyzed directly after whitespace normalization.
"""


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(
        "Start with a built-in transcript or provide your own. The bundled IVTFF demo is loaded automatically "
        "when the app opens, so the analysis tabs are immediately usable."
    )

    with gr.Group():
        gr.Markdown("## Transcript source")
        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(
                    choices=preset_names(),
                    value=DEFAULT_PRESET,
                    label="Built-in transcript",
                )
                preset_info = gr.Markdown(preset_description(DEFAULT_PRESET))
                load_builtin = gr.Button("Load selected built-in transcript", variant="primary")
            with gr.Column(scale=1):
                pasted = gr.Textbox(
                    label="Or paste your own transcription",
                    lines=8,
                    placeholder="Paste EVA-style plain text or IVTFF here…",
                )
                upload = gr.File(
                    label="Or upload your own transcription",
                    file_types=[".txt", ".eva", ".ivtff"],
                    type="filepath",
                )
                gr.Markdown("If both are supplied, pasted text takes precedence over the uploaded file.")
                load_custom = gr.Button("Load pasted / uploaded transcript")

        with gr.Accordion("Format examples and input rules", open=False):
            gr.Markdown(FORMAT_HELP)

        source_status = gr.Textbox(label="Loaded source", lines=4, interactive=False)
        source_preview = gr.Textbox(
            label="Analysis-stream preview (metadata removed)",
            lines=8,
            interactive=False,
        )

    gr.Markdown("---\n## Analysis workspace")
    legacy.demo.render()

    source_outputs = [
        legacy.raw_state,
        legacy.text_state,
        source_status,
        source_preview,
        legacy.load_status,
        legacy.analysis_preview,
    ]
    load_builtin.click(load_builtin_ui, preset, source_outputs)
    load_custom.click(load_custom_ui, [upload, pasted], source_outputs)
    preset.change(preset_info_ui, preset, preset_info, queue=False)
    demo.load(load_builtin_ui, preset, source_outputs, queue=False)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
