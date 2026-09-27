from __future__ import annotations

import json
from pathlib import Path
import sys

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voynich_lab.gpu_probe import gpu_cooccurrence_probe
from voynich_lab.harness import run_harness
from voynich_lab.hf_sync import configured_space, push_snapshot, sync_status
from voynich_lab.metrics import compute_metrics, normalize_text, positional_entropy_by_decile
from voynich_lab.segmentation import discover_bpe_units

APP_TITLE = "Voynich Structure Lab"


def load_uploaded(file_path: str | None, pasted: str) -> tuple[str, str]:
    if pasted and pasted.strip():
        text = normalize_text(pasted)
        return text, f"Loaded pasted text: {len(text):,} characters including layout separators."
    if not file_path:
        return "", "No input loaded."
    path = Path(file_path)
    text = normalize_text(path.read_text(encoding="utf-8", errors="replace"))
    return text, f"Loaded {path.name}: {len(text):,} characters including layout separators."


def run_metrics_ui(text: str):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    return pd.DataFrame([{"metric": k, "value": v} for k, v in compute_metrics(text).items()]), pd.DataFrame([{"position_bin": k, "glyph_entropy_bits": v} for k, v in positional_entropy_by_decile(text).items()])


def run_harness_ui(text: str, replicates: int, seed: int):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    raw, summary = run_harness(text, int(replicates), int(seed))
    return summary, raw


def run_bpe_ui(text: str, merges: int):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    units, history = discover_bpe_units(text, int(merges))
    return pd.DataFrame(history), " · ".join(units[:300]), len(units), len(set(units))


def run_gpu_ui(text: str, window: int, max_symbols: int):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    result = gpu_cooccurrence_probe(text, int(window), int(max_symbols))
    table = pd.DataFrame(result.pop("glyph_table"))
    return json.dumps(result, indent=2), table


def push_ui(repo_id: str):
    try:
        return push_snapshot(repo_id)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown("Falsification-first structural analysis. Whitespace-delimited strings are treated as **conventional transcription tokens**, not assumed linguistic words.")
    text_state = gr.State("")

    with gr.Tab("1 · Data"):
        pasted = gr.Textbox(label="Paste transcription", lines=14, placeholder="Paste EVA/IVTFF-derived text here…")
        upload = gr.File(label="Upload transcription", file_types=[".txt", ".ivtff"], type="filepath")
        load_btn = gr.Button("Load transcription", variant="primary")
        load_status = gr.Textbox(label="Status", interactive=False)
        load_btn.click(load_uploaded, [upload, pasted], [text_state, load_status])

    with gr.Tab("2 · Phase 2 metrics"):
        gr.Markdown("Core metrics run entirely on CPU and do **not** consume ZeroGPU quota.")
        metrics_btn = gr.Button("Compute descriptive metrics", variant="primary")
        metrics_table = gr.Dataframe(label="Corpus metrics", interactive=False)
        positional_table = gr.Dataframe(label="Within-line positional entropy", interactive=False)
        metrics_btn.click(run_metrics_ui, text_state, [metrics_table, positional_table])
        with gr.Row():
            replicates = gr.Slider(1, 100, value=20, step=1, label="Replicates per null")
            seed = gr.Number(value=0, precision=0, label="Seed")
        harness_btn = gr.Button("Run CPU null-model harness")
        harness_summary = gr.Dataframe(label="Summary", interactive=False)
        harness_raw = gr.Dataframe(label="Raw replicates", interactive=False)
        harness_btn.click(run_harness_ui, [text_state, replicates, seed], [harness_summary, harness_raw])

    with gr.Tab("3 · Unit discovery"):
        gr.Markdown("BPE-style merges are learned on the continuous glyph stream with whitespace erased. This is a baseline, not a claim that BPE units are linguistic units.")
        merges = gr.Slider(0, 128, value=32, step=1, label="Merge operations")
        bpe_btn = gr.Button("Discover units")
        bpe_history = gr.Dataframe(label="Merge history", interactive=False)
        bpe_preview = gr.Textbox(label="Segmented preview", lines=8, interactive=False)
        with gr.Row():
            bpe_len = gr.Number(label="Resulting sequence length", interactive=False)
            bpe_types = gr.Number(label="Distinct discovered symbols/units", interactive=False)
        bpe_btn.click(run_bpe_ui, [text_state, merges], [bpe_history, bpe_preview, bpe_len, bpe_types])

    with gr.Tab("4 · Optional ZeroGPU probe"):
        gr.Markdown("This tab is the **only** path that requests ZeroGPU. It is deliberately not called automatically.")
        with gr.Row():
            gpu_window = gr.Slider(1, 32, value=5, step=1, label="Context window")
            gpu_max = gr.Slider(1000, 100000, value=50000, step=1000, label="Max glyphs")
        gpu_btn = gr.Button("Run GPU probe")
        gpu_json = gr.Code(label="GPU probe summary", language="json")
        gpu_table = gr.Dataframe(label="Glyph frequencies", interactive=False)
        gpu_btn.click(run_gpu_ui, [text_state, gpu_window, gpu_max], [gpu_json, gpu_table])

    with gr.Tab("5 · Hugging Face sync"):
        gr.Markdown("GitHub `main` is the source of truth. The workflow mirrors pushes to Hugging Face. This button is a recovery path.")
        hf_repo = gr.Textbox(label="HF Space repo", value=configured_space())
        sync_check = gr.Button("Check sync configuration")
        sync_info = gr.Textbox(label="Configuration", lines=4, interactive=False)
        sync_check.click(sync_status, outputs=sync_info)
        push_btn = gr.Button("Push current snapshot to Hugging Face")
        push_result = gr.Textbox(label="Push result", lines=4, interactive=False)
        push_btn.click(push_ui, hf_repo, push_result)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
