from pathlib import Path

app_path = Path("app.py")
app = app_path.read_text(encoding="utf-8")

import_marker = "from voynich_lab.segmentation import cross_fit_bpe_scale_curve, discover_bpe_units\n"
import_replacement = (
    import_marker
    + "from voynich_lab.unigram_segmentation import segmentation_consensus\n"
)
assert app.count(import_marker) == 1, "segmentation import marker drifted"
app = app.replace(import_marker, import_replacement, 1)

function_marker = "def run_gpu_ui(text: str, window: int, max_symbols: int):\n"
function_block = '''def run_segmentation_consensus_ui(
    text: str,
    bpe_merges: int,
    max_unit_length: int,
    min_count: int,
    max_vocab: int,
    iterations: int,
    length_penalty_bits: float,
):
    if not text.strip():
        raise gr.Error("Load a transcription first.")
    try:
        result = segmentation_consensus(
            text,
            bpe_merges=int(bpe_merges),
            max_unit_length=int(max_unit_length),
            min_count=int(min_count),
            max_vocab=int(max_vocab),
            iterations=int(iterations),
            length_penalty_bits=float(length_penalty_bits),
            preview_lines=12,
        )
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc

    summary = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in result.summary.items()]
    )
    lines = pd.DataFrame(result.line_rows)
    units = pd.DataFrame(result.unit_rows)
    note = (
        f"Exploratory consensus only: unigram latent model vocabulary={result.model.vocabulary_size}, "
        f"EM iterations={result.model.iterations_run}; line-bounded BPE rules={len(result.bpe_rules)}. "
        "Both models see the same space-erased glyph lines. Transcription spaces are withheld from fitting "
        "and used only afterward as boundary observations. Shared boundaries are candidates for follow-up, "
        "not identified linguistic morphemes or words."
    )
    return summary, lines, units, result.preview, note


'''
assert app.count(function_marker) == 1, "function marker drifted"
app = app.replace(function_marker, function_block + function_marker, 1)

ui_marker = '''        crossfit_btn.click(run_bpe_crossfit_ui, raw_state, [crossfit_table, crossfit_selected])

    with gr.Tab("4 · IVTFF audit"):
'''
ui_block = '''        crossfit_btn.click(run_bpe_crossfit_ui, raw_state, [crossfit_table, crossfit_selected])

        gr.Markdown(
            "#### Independent boundary consensus (exploratory)\\n"
            "This path does **not** use spaces as hard word boundaries. A finite unigram latent-unit model "
            "and a separate line-bounded BPE are fit to the same space-erased glyph lines; observed spaces "
            "are compared to inferred boundaries only after fitting."
        )
        with gr.Row():
            consensus_bpe_merges = gr.Slider(0, 128, value=32, step=1, label="Comparator BPE merges")
            consensus_max_len = gr.Slider(2, 12, value=8, step=1, label="Max unigram unit length")
            consensus_min_count = gr.Slider(1, 10, value=2, step=1, label="Min recurring substring count")
        with gr.Row():
            consensus_vocab = gr.Slider(64, 2048, value=512, step=64, label="Unigram candidate vocabulary cap")
            consensus_iterations = gr.Slider(2, 30, value=10, step=1, label="EM iterations")
            consensus_length_penalty = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Long-unit penalty (bits per extra glyph)")
        consensus_btn = gr.Button("Run independent segmentation consensus")
        consensus_summary = gr.Dataframe(label="Boundary-consensus summary", interactive=False)
        consensus_lines = gr.Dataframe(label="Per-line boundary agreement", interactive=False)
        consensus_units = gr.Dataframe(label="Top units by method", interactive=False)
        consensus_preview = gr.Textbox(label="Segmentation preview", lines=16, interactive=False)
        consensus_note = gr.Textbox(label="Interpretation guardrail", lines=5, interactive=False)
        consensus_btn.click(
            run_segmentation_consensus_ui,
            [
                text_state,
                consensus_bpe_merges,
                consensus_max_len,
                consensus_min_count,
                consensus_vocab,
                consensus_iterations,
                consensus_length_penalty,
            ],
            [
                consensus_summary,
                consensus_lines,
                consensus_units,
                consensus_preview,
                consensus_note,
            ],
        )

    with gr.Tab("4 · IVTFF audit"):
'''
assert app.count(ui_marker) == 1, "unit-discovery UI marker drifted"
app = app.replace(ui_marker, ui_block, 1)
app_path.write_text(app, encoding="utf-8")

readme_path = Path("README.md")
readme = readme_path.read_text(encoding="utf-8")
old_gap = "2. **Consensus segmentation.** BPE is implemented, including paper-exact held-out BPE, but an independent segmental HSMM and motif-discovery path are not yet implemented. A later consensus analysis should report where independent methods agree rather than treating BPE output as identified linguistic units.\n"
new_gap = "2. **Consensus segmentation.** BPE now has an independent finite-unigram latent segmentation comparator with boundary-consensus diagnostics that withhold transcription spaces from fitting. Remaining work is a third independent family (segmental HSMM and/or motif discovery), quire-held-out stability for the non-BPE model, and explicit certain-vs-uncertain IVTFF boundary stratification in the consensus layer.\n"
assert readme.count(old_gap) == 1, "README segmentation-gap marker drifted"
readme = readme.replace(old_gap, new_gap, 1)
readme_path.write_text(readme, encoding="utf-8")

init_path = Path("src/voynich_lab/__init__.py")
init_text = init_path.read_text(encoding="utf-8")
init_marker = '    "segmentation",\n'
assert init_text.count(init_marker) == 1, "package export marker drifted"
if '    "unigram_segmentation",\n' not in init_text:
    init_text = init_text.replace(
        init_marker,
        init_marker + '    "unigram_segmentation",\n',
        1,
    )
init_path.write_text(init_text, encoding="utf-8")
