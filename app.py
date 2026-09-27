from __future__ import annotations

import json
from pathlib import Path
import sys

import gradio as gr
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voynich_lab.discriminator import classifier_two_sample_test
from voynich_lab.gpu_probe import gpu_cooccurrence_probe
from voynich_lab.harness import run_harness
from voynich_lab.hf_sync import configured_space_value, push_snapshot, sync_status
from voynich_lab.ivtff import parse_ivtff
from voynich_lab.longrange import block_entropy, dfa_fluctuation, lagged_mutual_information
from voynich_lab.metrics import compute_metrics, normalize_text, positional_entropy_by_decile
from voynich_lab.replication_targets import compare_to_targets
from voynich_lab.scorecard import scorecard_to_rows, run_scorecard
from voynich_lab.segmentation import discover_bpe_units
from voynich_lab.surrogates import iid_glyph_surrogate, markov1_surrogate, markov_k_surrogate

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


def run_ivtff_ui(text: str):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    doc = parse_ivtff(text)
    if not doc.lines:
        return (
            "No IVTFF locus lines recognized (expected lines like `<f1r.1,@P0;U> word.word`). "
            "This tab requires real IVTFF-formatted input, not plain text.",
            pd.DataFrame(),
        )
    n_certain = sum(1 for l in doc.lines for s in l.separators if s == "certain")
    n_uncertain = sum(1 for l in doc.lines for s in l.separators if s == "uncertain")
    folios = sorted({l.folio for l in doc.lines})
    summary = (
        f"Parsed {len(doc.lines)} loci across {len(folios)} folios. "
        f"Separators: {n_certain} certain, {n_uncertain} uncertain "
        f"({n_uncertain / max(1, n_certain + n_uncertain):.1%} uncertain). "
        "Quire membership is not in IVTFF itself -- load a folio,quire CSV via "
        "voynich_lab.ivtff.load_quire_map for quire-held-out splits."
    )
    preview = pd.DataFrame(
        [
            {"folio": l.folio, "line": l.line_no, "tokens": ".".join(t.text for t in l.tokens)}
            for l in doc.lines[:50]
        ]
    )
    return summary, preview


GENERATOR_CHOICES = {
    "N0 i.i.d. glyph": iid_glyph_surrogate,
    "N1 order-1 Markov": markov1_surrogate,
    "N2 order-3 Markov": lambda t, seed: markov_k_surrogate(t, order=3, seed=seed),
}


def run_longrange_ui(text: str, max_lag: int, dfa_min: int, dfa_max: int, dfa_steps: int):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    stream = "".join(ch for ch in text if not ch.isspace())
    mi = lagged_mutual_information(stream, max_lag=int(max_lag))
    be = block_entropy(stream)
    box_sizes = tuple(sorted(set(int(x) for x in np.linspace(dfa_min, dfa_max, int(dfa_steps)))))
    fluct, alpha = dfa_fluctuation(text, box_sizes=box_sizes)
    mi_df = pd.DataFrame([{"lag": k, "mutual_information_bits": v} for k, v in mi.items()])
    be_df = pd.DataFrame([{"block_size": k, "per_symbol_entropy_bits": v} for k, v in be.items()])
    fluct_df = pd.DataFrame([{"box_size": k, "rms_fluctuation": v} for k, v in fluct.items()])
    return mi_df, be_df, fluct_df, f"Estimated DFA scaling exponent alpha = {alpha:.3f} (0.5 = no long-range correlation)"


def run_scorecard_ui(text: str, generator_name: str, n_replicates: int, seed: int):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    fn = GENERATOR_CHOICES[generator_name]
    result = run_scorecard(text, fn, n_replicates=int(n_replicates), seed=int(seed))
    rows = scorecard_to_rows(result)
    verdict = (
        f"D_infinity (worst |z-score|) = {result.d_infinity:.2f} on feature '{result.worst_feature}'. "
        "Rule of thumb only, not a formal p-value: values above ~2-3 indicate the generator "
        "misses at least one axis badly enough to fail layer 1, regardless of how well it "
        "matches the other features."
    )
    return pd.DataFrame(rows), verdict


def run_discriminator_ui(text: str, generator_name: str, block_size: int, n_gen_replicates: int, seed: int):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    fn = GENERATOR_CHOICES[generator_name]
    result = classifier_two_sample_test(
        text, fn, block_size=int(block_size), n_generated_replicates=int(n_gen_replicates), seed=int(seed)
    )
    summary = (
        f"Held-out AUC = {result.held_out_auc:.3f} "
        f"({result.n_real_blocks} real blocks vs {result.n_generated_blocks} generated blocks, "
        f"block_size={result.block_size}).\n{result.note}"
    )
    return summary


def run_replication_ui(observed_json: str, tolerance: float):
    try:
        observed = json.loads(observed_json) if observed_json.strip() else {}
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Invalid JSON: {exc}") from exc
    if not observed:
        raise gr.Error("Paste a JSON object mapping metric names to your computed values.")
    rows = compare_to_targets(observed, tolerance=float(tolerance))
    if not rows:
        raise gr.Error("None of the provided metric names match a known published target. See replication_targets.py for the expected keys.")
    return pd.DataFrame(rows)


def push_ui(target: str):
    try:
        return push_snapshot(target)
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

    with gr.Tab("3b · IVTFF parsing"):
        gr.Markdown(
            "Requires real IVTFF-formatted input (e.g. `<f1r.1,@P0;U> fachys.ykal.ar,ataiin`), "
            "not plain whitespace text. Period = certain word space, comma = uncertain word space, "
            "per Zandbergen's IVTFF 2.0 spec -- this is the distinction the current Phase 2 metrics "
            "tab cannot see because it treats all whitespace as equivalent."
        )
        ivtff_btn = gr.Button("Parse as IVTFF")
        ivtff_summary = gr.Textbox(label="Summary", lines=4, interactive=False)
        ivtff_preview = gr.Dataframe(label="Parsed loci (first 50)", interactive=False)
        ivtff_btn.click(run_ivtff_ui, text_state, [ivtff_summary, ivtff_preview])

    with gr.Tab("3c · Long-range structure"):
        gr.Markdown(
            "Three independent measures, per the review: a generator should not get credit for "
            "'long-range structure' on the strength of one estimator's favorable slope."
        )
        with gr.Row():
            max_lag = gr.Slider(2, 50, value=20, step=1, label="Max lag (mutual information)")
        with gr.Row():
            dfa_min = gr.Number(value=10, precision=0, label="DFA min box size")
            dfa_max = gr.Number(value=320, precision=0, label="DFA max box size")
            dfa_steps = gr.Slider(2, 12, value=6, step=1, label="DFA box-size steps")
        longrange_btn = gr.Button("Compute long-range statistics")
        mi_table = gr.Dataframe(label="Lagged mutual information I(k)", interactive=False)
        be_table = gr.Dataframe(label="Block entropy (per symbol)", interactive=False)
        fluct_table = gr.Dataframe(label="DFA-style fluctuation by box size", interactive=False)
        alpha_text = gr.Textbox(label="Scaling exponent", interactive=False)
        longrange_btn.click(
            run_longrange_ui,
            [text_state, max_lag, dfa_min, dfa_max, dfa_steps],
            [mi_table, be_table, fluct_table, alpha_text],
        )

    with gr.Tab("6 · Adversarial validation"):
        gr.Markdown(
            "Layer 1 (locked scorecard): standardized discrepancy per feature against a generator's "
            "own Monte Carlo distribution. Layer 2 (classifier two-sample test): a black-box check for "
            "residual structure. Layer 1 should be read first -- an AUC near 0.5 in layer 2 does not "
            "certify a generator as historically correct, only that this classifier found no difference."
        )
        gen_choice = gr.Dropdown(list(GENERATOR_CHOICES), value="N1 order-1 Markov", label="Candidate generator")
        with gr.Row():
            sc_replicates = gr.Slider(20, 500, value=150, step=10, label="Scorecard Monte Carlo replicates")
            sc_seed = gr.Number(value=0, precision=0, label="Seed")
        scorecard_btn = gr.Button("Run layer 1: locked scorecard", variant="primary")
        scorecard_table = gr.Dataframe(label="Per-feature z-scores", interactive=False)
        scorecard_verdict = gr.Textbox(label="D_infinity verdict", lines=3, interactive=False)
        scorecard_btn.click(
            run_scorecard_ui, [text_state, gen_choice, sc_replicates, sc_seed], [scorecard_table, scorecard_verdict]
        )
        with gr.Row():
            disc_block = gr.Slider(20, 1000, value=200, step=10, label="Block size (glyphs)")
            disc_reps = gr.Slider(1, 20, value=5, step=1, label="Generator replicates")
            disc_seed = gr.Number(value=0, precision=0, label="Seed")
        discriminator_btn = gr.Button("Run layer 2: classifier two-sample test")
        discriminator_result = gr.Textbox(label="Held-out AUC", lines=4, interactive=False)
        discriminator_btn.click(
            run_discriminator_ui,
            [text_state, gen_choice, disc_block, disc_reps, disc_seed],
            discriminator_result,
        )

    with gr.Tab("7 · Replication gate"):
        gr.Markdown(
            "Paste your own computed metrics as a JSON object to check against published 2025-2026 "
            "targets (see `replication_targets.py` for source citations and metric-definition caveats). "
            'Example: `{"char_conditional_entropy_bits": 2.65, "cross_boundary_mutual_information_bits": 0.21}`'
        )
        observed_json = gr.Textbox(label="Observed metrics (JSON)", lines=4)
        tolerance = gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="Relative-error tolerance")
        replication_btn = gr.Button("Compare to published targets")
        replication_table = gr.Dataframe(label="Replication gate", interactive=False)
        replication_btn.click(run_replication_ui, [observed_json, tolerance], replication_table)

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
        hf_target = gr.Textbox(label="HF Space URL or repo", value=configured_space_value())
        sync_check = gr.Button("Check sync configuration")
        sync_info = gr.Textbox(label="Configuration", lines=5, interactive=False)
        sync_check.click(sync_status, outputs=sync_info)
        push_btn = gr.Button("Push current snapshot to Hugging Face")
        push_result = gr.Textbox(label="Push result", lines=4, interactive=False)
        push_btn.click(push_ui, hf_target, push_result)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
