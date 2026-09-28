from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

import legacy_app as legacy
from voynich_lab.full_pipeline import (
    pipeline_summary_markdown,
    profile_names,
    run_full_pipeline,
)
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


def run_full_pipeline_ui(raw_text: str, analysis_text: str, profile: str, seed: int):
    if not analysis_text.strip():
        raise gr.Error("Load a transcript before running the full pipeline.")
    try:
        report = run_full_pipeline(
            raw_text,
            analysis_text,
            profile=str(profile),
            seed=int(seed),
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    rows = [
        [row["stage"], row["status"], row["seconds"], row["detail"]]
        for row in report.get("stages", [])
    ]
    status = (
        f"Pipeline finished: {report.get('completed_stages', 0)} completed, "
        f"{report.get('skipped_stages', 0)} skipped, {report.get('error_stages', 0)} errors."
    )
    return status, rows, pipeline_summary_markdown(report), json.dumps(report, indent=2, sort_keys=True)


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

    with gr.Group():
        gr.Markdown("## Run the full pipeline")
        gr.Markdown(
            "One click runs the main analysis stack in sequence. IVTFF-only stages are included when metadata is "
            "available and skipped cleanly for plain text. Individual stage failures do not stop later stages. "
            "ZeroGPU, generator playgrounds, and Hugging Face sync remain manual by design."
        )
        with gr.Row():
            pipeline_profile = gr.Dropdown(
                choices=profile_names(),
                value="Standard",
                label="Pipeline profile",
                info="Quick is best for previews; Standard is the default; Thorough increases Monte Carlo/bootstrap depth.",
            )
            pipeline_seed = gr.Number(value=0, precision=0, label="Pipeline seed")
            pipeline_btn = gr.Button("Run full pipeline", variant="primary")
        pipeline_status = gr.Textbox(label="Pipeline status", interactive=False)
        pipeline_summary = gr.Markdown()
        pipeline_stages = gr.Dataframe(
            headers=["stage", "status", "seconds", "detail"],
            datatype=["str", "str", "number", "str"],
            label="Stage status",
            interactive=False,
        )
        with gr.Accordion("Full pipeline JSON report", open=False):
            pipeline_json = gr.Code(language="json", label="Detailed results")

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
    pipeline_btn.click(
        run_full_pipeline_ui,
        [legacy.raw_state, legacy.text_state, pipeline_profile, pipeline_seed],
        [pipeline_status, pipeline_stages, pipeline_summary, pipeline_json],
    )
    demo.load(load_builtin_ui, preset, source_outputs, queue=False)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
