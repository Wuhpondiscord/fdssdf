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

from voynich_lab.currier_report import build_currier_report
from voynich_lab.discriminator import classifier_two_sample_test
from voynich_lab.gpu_probe import gpu_cooccurrence_probe
from voynich_lab.harness import run_harness
from voynich_lab.hf_sync import configured_space_value, push_snapshot, sync_status
from voynich_lab.ivtff import is_ivtff, parse_ivtff
from voynich_lab.longrange import block_entropy, dfa_fluctuation, lagged_mi_with_shuffle_baseline
from voynich_lab.metrics import compute_metrics, normalize_text, positional_entropy_by_decile
from voynich_lab.published_replication import automatic_replication_metrics, unit_scale_text_by_quire
from voynich_lab.replication_targets import compare_to_targets
from voynich_lab.scorecard import scorecard_to_rows, run_scorecard
from voynich_lab.segmentation import cross_fit_bpe_scale_curve, discover_bpe_units
from voynich_lab.surrogates import (
    block_shuffle_surrogate,
    iid_glyph_surrogate,
    markov1_surrogate,
    markov_k_surrogate,
    position_conditioned_markov_surrogate,
    token_shuffle_surrogate,
)

APP_TITLE = "Voynich Structure Lab"


def _source_text(file_path: str | None, pasted: str) -> tuple[str, str]:
    if pasted and pasted.strip():
        return pasted, "pasted text"
    if not file_path:
        return "", ""
    path = Path(file_path)
    return path.read_text(encoding="utf-8", errors="replace"), path.name


def load_uploaded(file_path: str | None, pasted: str):
    raw, source = _source_text(file_path, pasted)
    if not raw.strip():
        return "", "", "No input loaded.", ""
    analysis = normalize_text(raw)
    if is_ivtff(raw):
        doc = parse_ivtff(raw)
        if not doc.lines:
            return raw, "", "Input resembles IVTFF, but no locus text could be parsed.", ""
        quires = doc.quires_present()
        langs = sorted({line.currier_language for line in doc.lines if line.currier_language})
        status = (
            f"Loaded {source} as IVTFF: {len(doc.lines):,} loci, {len(doc.pages):,} page headers. "
            f"Native quires: {', '.join(quires) if quires else 'none found'}; "
            f"Currier labels: {', '.join(langs) if langs else 'none found'}. "
            f"Parser warnings: {len(doc.warnings)}. Metadata is excluded from glyph statistics."
        )
    else:
        status = f"Loaded {source} as plain text: {len(analysis):,} normalized characters including separators."
    return raw, analysis, status, analysis[:5000]


def run_metrics_ui(text: str):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    metrics = pd.DataFrame([{"metric": key, "value": value} for key, value in compute_metrics(text).items()])
    positional = pd.DataFrame(
        [{"position_bin": key, "glyph_entropy_bits": value} for key, value in positional_entropy_by_decile(text).items()]
    )
    return metrics, positional


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


def run_bpe_crossfit_ui(raw_text: str):
    if not raw_text.strip() or not is_ivtff(raw_text):
        raise gr.Error("Cross-fit BPE needs IVTFF input with native quire metadata ($Q).")
    by_quire = unit_scale_text_by_quire(raw_text)
    if len(by_quire) < 2:
        raise gr.Error("Fewer than two unit-scale quire groups were found after paper-exact P-locus cleaning.")
    rows, selected = cross_fit_bpe_scale_curve(by_quire, checkpoints=(0, 16, 32, 64))
    table_rows = [
        {
            "merges": row["merges"],
            "mean_unit_length": row["mean_unit_length"],
            "H1_bits": row["H1_bits"],
            "H2_conditional_bits": row["H2_conditional_bits"],
            "bits_per_glyph": row["bits_per_glyph"],
            "glyph_weighted_dependence_gap_bits": row["glyph_weighted_dependence_gap_bits"],
            "glyphs": row["glyphs"],
            "folds": row["folds"],
            "per_quire_gaps": json.dumps(row["fold_values"], sort_keys=True),
        }
        for row in rows
    ]
    return pd.DataFrame(table_rows), f"Held-out minimum among 0/16/32/64 merges: {selected}"


def run_gpu_ui(text: str, window: int, max_symbols: int):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    result = gpu_cooccurrence_probe(text, int(window), int(max_symbols))
    table = pd.DataFrame(result.pop("glyph_table"))
    return json.dumps(result, indent=2), table


def run_ivtff_ui(raw_text: str):
    if not raw_text.strip():
        raise gr.Error("Load a transcription first.")
    if not is_ivtff(raw_text):
        return "The loaded input is plain text, not IVTFF.", pd.DataFrame(), pd.DataFrame()
    doc = parse_ivtff(raw_text)
    counts = {label: 0 for label in ("certain", "uncertain", "drawing", "vertical")}
    for line in doc.lines:
        for sep in line.separators:
            counts[sep] = counts.get(sep, 0) + 1
    folios = sorted({line.folio for line in doc.lines})
    summary = (
        f"Parsed {len(doc.lines)} loci across {len(folios)} folios. Boundary observations: {counts}. "
        f"Native $Q metadata: {sum(1 for line in doc.lines if line.quire)}/{len(doc.lines)} loci. "
        f"Warnings: {len(doc.warnings)}."
    )
    preview = pd.DataFrame(
        [
            {
                "folio": line.folio,
                "line": line.line_no,
                "quire": line.quire,
                "Currier_L": line.currier_language,
                "hand": line.hand,
                "illustration": line.illustration_type,
                "locus_code": line.locus_code,
                "tokens": " | ".join(token.text for token in line.tokens),
            }
            for line in doc.lines[:100]
        ]
    )
    page_meta = pd.DataFrame(
        [
            {"folio": folio, **{f"${key}": value for key, value in page.variables.items()}}
            for folio, page in sorted(doc.pages.items())
        ]
    )
    return summary, preview, page_meta


def run_currier_ui(raw_text: str, bootstrap_repetitions: int, seed: int):
    if not raw_text.strip() or not is_ivtff(raw_text):
        raise gr.Error("Currier analysis requires IVTFF input with native $L and $Q metadata.")
    try:
        report = build_currier_report(
            raw_text,
            boundary_bootstrap_repetitions=int(bootstrap_repetitions),
            seed=int(seed),
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return (
        pd.DataFrame(report["boundary_rows"]),
        pd.DataFrame(report["stratum_rows"]),
        pd.DataFrame(report["bootstrap_rows"]),
        report["interpretation"],
        pd.DataFrame(report["line_start_jsd_rows"]),
        pd.DataFrame(report["line_start_enrichment_rows"]),
        json.dumps(report["provenance"], indent=2, sort_keys=True),
    )


# Every generator exposed to scorecard/discriminator obeys the same
# (text, seed) -> generated_text contract. Wrappers are explicit here because
# markov_k/block_shuffle/position_conditioned otherwise have a second
# positional argument that is not the random seed.
GENERATOR_CHOICES = {
    "N0 i.i.d. glyph (layout fixed)": iid_glyph_surrogate,
    "N1 order-1 Markov (layout fixed)": markov1_surrogate,
    "N2 order-3 Markov with backoff": lambda t, seed: markov_k_surrogate(t, order=3, seed=seed),
    "N3 order-5 Markov with backoff": lambda t, seed: markov_k_surrogate(t, order=5, seed=seed),
    "N4 glyph-block shuffle, size 3": lambda t, seed: block_shuffle_surrogate(t, block_size=3, seed=seed),
    "N5 conventional-token shuffle": token_shuffle_surrogate,
    "N6 position-conditioned Markov": lambda t, seed: position_conditioned_markov_surrogate(t, n_bins=10, seed=seed),
}


def run_longrange_ui(text: str, max_lag: int, permutations: int, dfa_min: int, dfa_max: int, dfa_steps: int):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    mi_rows = lagged_mi_with_shuffle_baseline(
        text, max_lag=int(max_lag), n_permutations=int(permutations), seed=0
    )
    be = block_entropy(text)
    box_sizes = tuple(sorted(set(int(x) for x in np.linspace(int(dfa_min), int(dfa_max), int(dfa_steps)))))
    fluct, alpha = dfa_fluctuation(text, box_sizes=box_sizes)
    return (
        pd.DataFrame(mi_rows),
        pd.DataFrame([{"block_size": key, "per_symbol_entropy_bits": value} for key, value in be.items()]),
        pd.DataFrame([{"box_size": key, "rms_fluctuation": value} for key, value in fluct.items()]),
        f"Descriptive DFA-style scaling exponent α = {alpha:.3f}. Interpret relative to matched nulls; do not use α=0.5 as a stand-alone rule.",
    )


def run_scorecard_ui(text: str, generator_name: str, n_replicates: int, seed: int):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    result = run_scorecard(
        text,
        GENERATOR_CHOICES[generator_name],
        n_replicates=int(n_replicates),
        seed=int(seed),
    )
    verdict = (
        f"D∞ = {result.d_infinity:.3f}; worst feature = {result.worst_feature}; "
        f"global leave-one-replicate-out Monte Carlo p = {result.global_p_value:.4f}. "
        "Use a preregistered alpha threshold; feature p-values are Holm-adjusted in the table."
    )
    return pd.DataFrame(scorecard_to_rows(result)), verdict


def run_discriminator_ui(text: str, generator_name: str, block_size: int, n_gen_replicates: int, seed: int):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    result = classifier_two_sample_test(
        text,
        GENERATOR_CHOICES[generator_name],
        block_size=int(block_size),
        n_generated_replicates=int(n_gen_replicates),
        seed=int(seed),
    )
    return (
        f"Group-aware held-out AUC = {result.held_out_auc:.3f} ± {result.auc_sd:.3f}; "
        f"split interval [{result.auc_ci_low:.3f}, {result.auc_ci_high:.3f}] across {result.n_splits} valid splits. "
        f"{result.n_real_blocks} real blocks vs {result.n_generated_blocks} generated blocks.\n{result.note}"
    )


def run_replication_ui(observed_json: str, tolerance: float):
    try:
        observed = json.loads(observed_json) if observed_json.strip() else {}
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Invalid JSON: {exc}") from exc
    if not observed:
        raise gr.Error("Paste a JSON object mapping metric names to computed values.")
    rows = compare_to_targets(observed, tolerance=float(tolerance))
    if not rows:
        raise gr.Error("None of the supplied metric names match a verified target.")
    return pd.DataFrame(rows)


def run_automatic_replication_ui(raw_text: str, tolerance: float):
    if not raw_text.strip() or not is_ivtff(raw_text):
        raise gr.Error("Automatic literature reproduction requires the original IVTFF transcription.")
    observed, diagnostics = automatic_replication_metrics(raw_text, shuffles=100)
    rows = compare_to_targets(observed, tolerance=float(tolerance))
    crossing = diagnostics["erased_space_crossing"]
    crossing_rows = []
    for key, label in (
        ("certain", "ZL certain separators"),
        ("uncertain", "ZL uncertain separators"),
        ("all_positions", "all eligible intra-line positions"),
    ):
        values = crossing[key]
        crossing_rows.append(
            {
                "boundary_pool": label,
                "bpe_merges": crossing["merges"],
                "crossed": values["crossed"],
                "total": values["total"],
                "crossing_rate": values["rate"],
            }
        )
    note = {
        "method": "Rozanova & Temerev public reproduction conventions",
        "order_shuffles": 100,
        "edge_order_rng_seed": 20260816,
        "token_order_rng_seed": 20260810,
        "separator_crossing_bpe_merges": 64,
        "auto_scored_targets": sorted(observed),
        "diagnostics": diagnostics,
    }
    return (
        json.dumps(observed, indent=2, sort_keys=True),
        json.dumps(note, indent=2, sort_keys=True),
        pd.DataFrame(rows),
        pd.DataFrame(crossing_rows),
    )


def push_ui(target: str):
    try:
        return push_snapshot(target)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(
        "Falsification-first structural analysis. IVTFF markup is parsed as metadata and excluded from manuscript glyph statistics. "
        "Conventional transcription tokens remain observations, not assumed linguistic words."
    )
    raw_state = gr.State("")
    text_state = gr.State("")

    with gr.Tab("1 · Data"):
        pasted = gr.Textbox(label="Paste transcription", lines=12, placeholder="Paste EVA / IVTFF / plain transcription here…")
        upload = gr.File(label="Upload transcription", file_types=[".txt", ".ivtff"], type="filepath")
        load_btn = gr.Button("Load transcription", variant="primary")
        load_status = gr.Textbox(label="Status", lines=4, interactive=False)
        analysis_preview = gr.Textbox(label="Analysis stream preview (metadata removed)", lines=8, interactive=False)
        load_btn.click(load_uploaded, [upload, pasted], [raw_state, text_state, load_status, analysis_preview])

    with gr.Tab("2 · Phase 2 metrics & nulls"):
        gr.Markdown("CPU-only. Glyph-level nulls preserve observed whitespace/layout so token-edge metrics remain defined under a fixed-boundary null.")
        metrics_btn = gr.Button("Compute descriptive metrics", variant="primary")
        metrics_table = gr.Dataframe(label="Corpus metrics", interactive=False)
        positional_table = gr.Dataframe(label="Within-line positional entropy", interactive=False)
        metrics_btn.click(run_metrics_ui, text_state, [metrics_table, positional_table])
        with gr.Row():
            replicates = gr.Slider(1, 100, value=20, step=1, label="Replicates per null")
            seed = gr.Number(value=0, precision=0, label="Seed")
        harness_btn = gr.Button("Run null-model suite")
        harness_summary = gr.Dataframe(label="Summary", interactive=False)
        harness_raw = gr.Dataframe(label="Raw replicates", interactive=False)
        harness_btn.click(run_harness_ui, [text_state, replicates, seed], [harness_summary, harness_raw])

    with gr.Tab("3 · Unit discovery"):
        gr.Markdown(
            "Exploratory in-sample stream BPE is shown separately from the reportable leave-one-quire-out curve. "
            "The cross-fit path uses the paper's P-locus cleaner and learns merges only inside conventional tokens."
        )
        merges = gr.Slider(0, 128, value=32, step=1, label="Exploratory merge operations")
        bpe_btn = gr.Button("Run exploratory BPE")
        bpe_history = gr.Dataframe(label="Merge history", interactive=False)
        bpe_preview = gr.Textbox(label="Segmented preview", lines=6, interactive=False)
        with gr.Row():
            bpe_len = gr.Number(label="Sequence length", interactive=False)
            bpe_types = gr.Number(label="Distinct units", interactive=False)
        bpe_btn.click(run_bpe_ui, [text_state, merges], [bpe_history, bpe_preview, bpe_len, bpe_types])
        gr.Markdown("#### Published-method leave-one-quire-out BPE scale selection (0/16/32/64 merges)")
        crossfit_btn = gr.Button("Run paper-exact cross-fit BPE scale curve")
        crossfit_table = gr.Dataframe(label="Held-out unit-scale statistics", interactive=False)
        crossfit_selected = gr.Textbox(label="Selected held-out scale", interactive=False)
        crossfit_btn.click(run_bpe_crossfit_ui, raw_state, [crossfit_table, crossfit_selected])

    with gr.Tab("4 · IVTFF audit"):
        gr.Markdown("Native IVTFF page variables such as `$Q`, `$L`, `$H`, and `$I` are parsed directly when present. External maps are only a fallback.")
        ivtff_btn = gr.Button("Audit IVTFF metadata")
        ivtff_summary = gr.Textbox(label="Summary", lines=4, interactive=False)
        ivtff_preview = gr.Dataframe(label="Parsed loci (first 100)", interactive=False)
        page_meta = gr.Dataframe(label="Page metadata", interactive=False)
        ivtff_btn.click(run_ivtff_ui, raw_state, [ivtff_summary, ivtff_preview, page_meta])

    with gr.Tab("5 · Currier A/B"):
        gr.Markdown(
            "Currier A/B boundary profiles use separate within-stratum PMI scales. "
            "A−B uncertainty is resampled by native quire under both independent-stratum and joint-quire schemes; "
            "do not treat a pooled A/B point difference as an independent dialect effect."
        )
        with gr.Row():
            currier_bootstrap = gr.Slider(20, 1500, value=200, step=20, label="Quire bootstrap draws")
            currier_seed = gr.Number(value=20260808, precision=0, label="Seed")
        currier_btn = gr.Button("Run Currier A/B analysis", variant="primary")
        currier_boundary = gr.Dataframe(label="Boundary profiles: stratum-local indices", interactive=False)
        currier_strata = gr.Dataframe(label="Currier stratum summary", interactive=False)
        currier_bootstrap_table = gr.Dataframe(label="A−B quire-bootstrap intervals", interactive=False)
        currier_interpretation = gr.Textbox(label="Interpretation", lines=5, interactive=False)
        gr.Markdown(
            "#### Line-start executable reproduction vs bundled STATED constants\n"
            "These tables reproduce the authors' pinned final executable. They deliberately show its output next to the "
            "same upstream file's older hard-coded STATED constants rather than forcing one to match the other."
        )
        currier_jsd = gr.Dataframe(label="Line-start JSD: executable vs STATED", interactive=False)
        currier_enrichment = gr.Dataframe(label="Line-start enrichment: executable vs STATED", interactive=False)
        currier_provenance = gr.Code(label="Method provenance", language="json")
        currier_btn.click(
            run_currier_ui,
            [raw_state, currier_bootstrap, currier_seed],
            [
                currier_boundary,
                currier_strata,
                currier_bootstrap_table,
                currier_interpretation,
                currier_jsd,
                currier_enrichment,
                currier_provenance,
            ],
        )

    with gr.Tab("6 · Long-range structure"):
        gr.Markdown("Raw lagged MI is shown with a shuffled finite-sample baseline; the DFA-style exponent is descriptive and should be judged against matched nulls.")
        with gr.Row():
            max_lag = gr.Slider(2, 50, value=20, step=1, label="Max lag")
            mi_perms = gr.Slider(5, 50, value=20, step=5, label="MI shuffle baselines")
        with gr.Row():
            dfa_min = gr.Number(value=10, precision=0, label="DFA min box")
            dfa_max = gr.Number(value=320, precision=0, label="DFA max box")
            dfa_steps = gr.Slider(2, 12, value=6, step=1, label="DFA steps")
        longrange_btn = gr.Button("Compute long-range statistics")
        mi_table = gr.Dataframe(label="Lagged MI with bias baseline", interactive=False)
        be_table = gr.Dataframe(label="Block entropy", interactive=False)
        fluct_table = gr.Dataframe(label="DFA-style fluctuation", interactive=False)
        alpha_text = gr.Textbox(label="DFA note", interactive=False)
        longrange_btn.click(
            run_longrange_ui,
            [text_state, max_lag, mi_perms, dfa_min, dfa_max, dfa_steps],
            [mi_table, be_table, fluct_table, alpha_text],
        )

    with gr.Tab("7 · Adversarial validation"):
        gr.Markdown("Layer 1 uses an empirically calibrated global max-statistic plus feature-wise Holm correction. Layer 2 uses contiguous group holdouts rather than random neighboring-block splits.")
        gen_choice = gr.Dropdown(list(GENERATOR_CHOICES), value="N1 order-1 Markov (layout fixed)", label="Candidate generator")
        with gr.Row():
            sc_replicates = gr.Slider(20, 500, value=150, step=10, label="Scorecard Monte Carlo replicates")
            sc_seed = gr.Number(value=0, precision=0, label="Seed")
        scorecard_btn = gr.Button("Run layer 1 scorecard", variant="primary")
        scorecard_table = gr.Dataframe(label="Per-feature results", interactive=False)
        scorecard_verdict = gr.Textbox(label="Global score", lines=3, interactive=False)
        scorecard_btn.click(run_scorecard_ui, [text_state, gen_choice, sc_replicates, sc_seed], [scorecard_table, scorecard_verdict])
        with gr.Row():
            disc_block = gr.Slider(20, 1000, value=200, step=10, label="Block size")
            disc_reps = gr.Slider(1, 20, value=5, step=1, label="Generator replicates")
            disc_seed = gr.Number(value=0, precision=0, label="Seed")
        discriminator_btn = gr.Button("Run layer 2 grouped classifier test")
        discriminator_result = gr.Textbox(label="Group-aware held-out AUC", lines=6, interactive=False)
        discriminator_btn.click(
            run_discriminator_ui,
            [text_state, gen_choice, disc_block, disc_reps, disc_seed],
            discriminator_result,
        )

    with gr.Tab("8 · Replication gate"):
        gr.Markdown(
            "The automatic path reproduces the published preprocessing rather than reusing the generic descriptive stream. "
            "Entropy, within-token cross-fit BPE, edge/order statistics, and the 64-merge erased-space separator-crossing analysis each use their own paper-specific corpus/null contract."
        )
        tolerance = gr.Slider(0.01, 0.50, value=0.10, step=0.01, label="Relative tolerance for point targets")
        auto_replication_btn = gr.Button("Run automatic Rozanova/Temerev replication", variant="primary")
        auto_observed = gr.Code(label="Automatically computed observed metrics", language="json")
        auto_replication_table = gr.Dataframe(label="Automatic replication gate", interactive=False)
        auto_crossing_table = gr.Dataframe(label="64-merge erased-space boundary crossing", interactive=False)
        auto_diagnostics = gr.Code(label="Full method diagnostics", language="json")
        auto_replication_btn.click(
            run_automatic_replication_ui,
            [raw_state, tolerance],
            [auto_observed, auto_diagnostics, auto_replication_table, auto_crossing_table],
        )
        gr.Markdown("#### Manual target comparison\nUse this for independently computed metrics or literature targets not implemented by the automatic Rozanova/Temerev path, including the separate Parisel statistics.")
        observed_json = gr.Textbox(
            label="Observed metrics (JSON)",
            lines=5,
            placeholder='{"end_to_start_flow_proportion": 0.806}',
        )
        replication_btn = gr.Button("Compare supplied values to verified targets")
        replication_table = gr.Dataframe(label="Manual replication gate", interactive=False)
        replication_btn.click(run_replication_ui, [observed_json, tolerance], replication_table)

    with gr.Tab("9 · Optional ZeroGPU probe"):
        gr.Markdown("This remains the only path that requests ZeroGPU. It is never invoked by loading, parsing, null generation, BPE, or the standard statistical tests.")
        with gr.Row():
            gpu_window = gr.Slider(1, 32, value=5, step=1, label="Context window")
            gpu_max = gr.Slider(1000, 100000, value=50000, step=1000, label="Max glyphs")
        gpu_btn = gr.Button("Run GPU probe")
        gpu_json = gr.Code(label="GPU probe summary", language="json")
        gpu_table = gr.Dataframe(label="Glyph frequencies", interactive=False)
        gpu_btn.click(run_gpu_ui, [text_state, gpu_window, gpu_max], [gpu_json, gpu_table])

    with gr.Tab("10 · Hugging Face sync"):
        gr.Markdown("GitHub `main` remains the source of truth. The workflow mirrors pushes to Hugging Face; this button is a recovery path.")
        hf_target = gr.Textbox(label="HF Space URL or repo", value=configured_space_value())
        sync_check = gr.Button("Check sync configuration")
        sync_info = gr.Textbox(label="Configuration", lines=5, interactive=False)
        sync_check.click(sync_status, outputs=sync_info)
        push_btn = gr.Button("Push current snapshot to Hugging Face")
        push_result = gr.Textbox(label="Push result", lines=4, interactive=False)
        push_btn.click(push_ui, hf_target, push_result)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
