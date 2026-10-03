from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import math
from statistics import mean, stdev
from time import perf_counter
from typing import Any, Callable

from .boundary_selection import select_boundary_markov_order
from .currier_report import build_currier_report
from .discriminator import classifier_two_sample_test
from .ivtff import is_ivtff, parse_ivtff
from .longrange import block_entropy, dfa_fluctuation, lagged_mi_with_shuffle_baseline
from .metrics import compute_metrics, conventional_tokens, glyph_stream, positional_entropy_by_decile
from .published_replication import automatic_replication_metrics, unit_scale_text_by_quire
from .replication_targets import compare_to_targets
from .scorecard import run_scorecard, scorecard_to_rows
from .segmentation import cross_fit_bpe_scale_curve, discover_bpe_units
from .stratified import heterogeneity_control, line_strata
from .surrogates import NULL_LADDER, bind_generator, boundary_markov_surrogate
from .unigram_segmentation import segmentation_consensus


@dataclass(frozen=True)
class PipelineProfile:
    label: str
    null_replicates: int
    bpe_merges: int
    consensus_bpe_merges: int
    consensus_vocab: int
    consensus_iterations: int
    max_lag: int
    mi_permutations: int
    scorecard_replicates: int
    discriminator_replicates: int
    discriminator_splits: int
    currier_bootstrap: int
    replication_shuffles: int


PROFILES: dict[str, PipelineProfile] = {
    "Quick": PipelineProfile("Quick", 3, 16, 16, 256, 4, 10, 5, 20, 2, 6, 40, 20),
    "Standard": PipelineProfile("Standard", 10, 32, 32, 512, 8, 20, 10, 50, 3, 8, 100, 50),
    "Thorough": PipelineProfile("Thorough", 20, 64, 64, 1024, 12, 30, 20, 150, 5, 12, 300, 100),
}

DIAGNOSTIC_FEATURES = (
    "conditional_entropy_order1_bits",
    "adjacent_glyph_mi_bits",
    "cross_token_edge_mi_bits",
    "hapax_fraction",
    "mean_token_length",
    "zipf_loglog_slope",
)

# The published Rozanova & Temerev / Parisel targets this pipeline reproduces against were fit on
# roughly 200,000 glyphs / 33,000 conventional tokens (the full ZL3b corpus). Below this many glyphs,
# the size-sensitive stages (replication_gate, discriminator, heterogeneity_control, and to a lesser
# extent scorecard) cannot carry evidential weight: there is no amount of Monte Carlo replication that
# fixes too little underlying text. This threshold is intentionally generous (an order of magnitude
# below the real corpus) so it only fires for genuinely toy-scale input such as the bundled offline demo.
MIN_GLYPHS_FOR_RELIABLE_STAGES = 5_000


def _corpus_scale(analysis_text: str, source_label: str) -> dict[str, Any]:
    glyph_count = len(glyph_stream(analysis_text))
    token_count = len(conventional_tokens(analysis_text))
    is_demo_scale = glyph_count < MIN_GLYPHS_FOR_RELIABLE_STAGES
    note = (
        f"{glyph_count:,} glyphs / {token_count:,} conventional tokens is far below the published "
        f"reproduction corpus (~200,000 glyphs / ~33,000 tokens). Treat every stage below as a wiring "
        "smoke test, not a manuscript finding: replication_gate mismatches, null discriminator AUCs, and "
        "degenerate scorecard z-scores are expected at this scale and do not indicate anything about the "
        "Voynich Manuscript. Select the full pinned ZL3b transcript before drawing any conclusion."
        if is_demo_scale
        else f"{glyph_count:,} glyphs / {token_count:,} conventional tokens."
    )
    return {
        "source_label": source_label or "unspecified",
        "glyph_count": glyph_count,
        "conventional_token_count": token_count,
        "min_glyphs_for_reliable_stages": MIN_GLYPHS_FOR_RELIABLE_STAGES,
        "is_demo_scale": is_demo_scale,
        "note": note,
    }


def profile_names() -> list[str]:
    return list(PROFILES)


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _contiguous_line_groups(text: str, target_groups: int = 5) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return {}
    groups = min(max(2, target_groups), len(lines))
    out: dict[str, list[str]] = {f"fold_{i + 1}": [] for i in range(groups)}
    for i, line in enumerate(lines):
        bucket = min(groups - 1, (i * groups) // len(lines))
        out[f"fold_{bucket + 1}"].append(line)
    return {key: "\n".join(value) for key, value in out.items() if value}


def _null_ladder(text: str, replicates: int, seed: int) -> list[dict[str, Any]]:
    observed = compute_metrics(text)
    rows: list[dict[str, Any]] = []
    for name, generator in NULL_LADDER.items():
        samples = {feature: [] for feature in DIAGNOSTIC_FEATURES}
        for r in range(max(1, int(replicates))):
            metrics = compute_metrics(generator(text, seed + r))
            for feature in DIAGNOSTIC_FEATURES:
                value = float(metrics[feature])
                if math.isfinite(value):
                    samples[feature].append(value)
        for feature in DIAGNOSTIC_FEATURES:
            values = samples[feature]
            mu = mean(values) if values else float("nan")
            sd = stdev(values) if len(values) > 1 else 0.0
            obs = float(observed[feature])
            z = (obs - mu) / sd if sd > 0 and math.isfinite(obs) and math.isfinite(mu) else float("nan")
            rows.append(
                {
                    "null": name,
                    "feature": feature,
                    "observed": obs,
                    "null_mean": mu,
                    "null_sd": sd,
                    "z": z,
                    "replicates": len(values),
                }
            )
    return rows


def _run_stage(
    stages: list[dict[str, Any]],
    report: dict[str, Any],
    key: str,
    label: str,
    fn: Callable[[], Any],
) -> Any:
    started = perf_counter()
    try:
        value = fn()
    except Exception as exc:
        stages.append(
            {
                "stage": label,
                "status": "error",
                "seconds": round(perf_counter() - started, 3),
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        report[key] = {"error": f"{type(exc).__name__}: {exc}"}
        return None
    stages.append(
        {
            "stage": label,
            "status": "completed",
            "seconds": round(perf_counter() - started, 3),
            "detail": "",
        }
    )
    report[key] = _plain(value)
    return value


def _skip(stages: list[dict[str, Any]], report: dict[str, Any], key: str, label: str, reason: str) -> None:
    stages.append({"stage": label, "status": "skipped", "seconds": 0.0, "detail": reason})
    report[key] = {"skipped": reason}


def run_full_pipeline(
    raw_text: str,
    analysis_text: str,
    *,
    profile: str = "Standard",
    seed: int = 0,
    source_label: str = "",
) -> dict[str, Any]:
    """Run the main analysis stack in one call while isolating per-stage failures.

    IVTFF-only stages are skipped for plain text instead of aborting the run. The
    optional ZeroGPU probe, generator playgrounds, and Hugging Face sync are not
    part of this pipeline because they are resource/deployment actions rather
    than manuscript analyses.

    ``source_label`` identifies which transcript produced this report (a built-in
    preset name, "pasted text", or an uploaded filename). It is carried into the
    report's ``corpus_scale`` field purely so the JSON is self-describing; callers
    that don't track a label can omit it.
    """
    raw_text = str(raw_text or "")
    analysis_text = str(analysis_text or "")
    if not analysis_text.strip():
        raise ValueError("load a transcript before running the full pipeline")
    if profile not in PROFILES:
        raise ValueError(f"unknown pipeline profile {profile!r}")
    cfg = PROFILES[profile]

    stages: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "profile": _plain(cfg),
        "seed": int(seed),
        "input_kind": "IVTFF" if raw_text.strip() and is_ivtff(raw_text) else "plain text",
        "corpus_scale": _corpus_scale(analysis_text, source_label),
        "exclusions": [
            "Optional ZeroGPU probe is not run automatically.",
            "Generator playground output (Naibbe/self-citation/Rugg) is not generated automatically.",
            "Hugging Face sync is a deployment action and is not part of analysis.",
        ],
    }

    _run_stage(stages, report, "metrics", "Descriptive metrics", lambda: {
        "corpus": compute_metrics(analysis_text),
        "positional_entropy": positional_entropy_by_decile(analysis_text),
    })

    _run_stage(
        stages,
        report,
        "null_ladder",
        "Null-model ladder",
        lambda: _null_ladder(analysis_text, cfg.null_replicates, int(seed)),
    )

    def exploratory_bpe():
        units, history = discover_bpe_units(analysis_text, cfg.bpe_merges)
        return {
            "requested_merges": cfg.bpe_merges,
            "merges_completed": len(history),
            "sequence_units": len(units),
            "distinct_units": len(set(units)),
            "merge_history": history,
            "preview": " · ".join(units[:300]),
        }

    _run_stage(stages, report, "exploratory_bpe", "Exploratory BPE", exploratory_bpe)

    _run_stage(
        stages,
        report,
        "segmentation_consensus",
        "Independent segmentation consensus",
        lambda: segmentation_consensus(
            analysis_text,
            bpe_merges=cfg.consensus_bpe_merges,
            max_unit_length=8,
            min_count=2,
            max_vocab=cfg.consensus_vocab,
            iterations=cfg.consensus_iterations,
            length_penalty_bits=0.0,
            preview_lines=12,
        ),
    )

    def long_range():
        box_sizes = (10, 20, 40, 80, 160, 320)
        fluct, alpha = dfa_fluctuation(analysis_text, box_sizes=box_sizes)
        return {
            "lagged_mi": lagged_mi_with_shuffle_baseline(
                analysis_text,
                max_lag=cfg.max_lag,
                n_permutations=cfg.mi_permutations,
                seed=int(seed),
            ),
            "block_entropy": block_entropy(analysis_text),
            "dfa_fluctuation": fluct,
            "dfa_alpha": alpha,
        }

    _run_stage(stages, report, "long_range", "Long-range structure", long_range)

    ivtff = bool(raw_text.strip() and is_ivtff(raw_text))
    doc = None
    if ivtff:
        doc = _run_stage(stages, report, "ivtff_audit", "IVTFF audit", lambda: parse_ivtff(raw_text))
    else:
        _skip(stages, report, "ivtff_audit", "IVTFF audit", "plain-text input has no IVTFF metadata")

    groups: dict[str, str] = {}
    group_source = "contiguous plain-text folds"
    if doc is not None:
        try:
            by_quire = doc.text_by_quire()
        except Exception:
            by_quire = {}
        if len(by_quire) >= 2:
            groups = by_quire
            group_source = "native IVTFF quires"
    if len(groups) < 2:
        groups = _contiguous_line_groups(analysis_text)

    selection = _run_stage(
        stages,
        report,
        "boundary_order_selection",
        "Held-out boundary n-gram order selection",
        lambda: {
            "group_source": group_source,
            "selection": select_boundary_markov_order(groups, candidates=(1, 2, 3, 4, 5)),
        },
    ) if len(groups) >= 2 else None

    selected_order = 2
    if selection is not None:
        try:
            selected_order = int(selection["selection"].selected_order)
        except Exception:
            pass
    candidate = bind_generator(boundary_markov_surrogate, order=selected_order)
    report["adversarial_candidate"] = f"boundary-aware order-{selected_order} n-gram"

    _run_stage(
        stages,
        report,
        "scorecard",
        "Adversarial layer 1 scorecard",
        lambda: {
            "order": selected_order,
            "result": run_scorecard(
                analysis_text,
                candidate,
                n_replicates=cfg.scorecard_replicates,
                seed=int(seed),
            ),
        },
    )

    _run_stage(
        stages,
        report,
        "discriminator",
        "Adversarial layer 2 grouped classifier",
        lambda: classifier_two_sample_test(
            analysis_text,
            candidate,
            block_size=200,
            n_generated_replicates=cfg.discriminator_replicates,
            seed=int(seed),
            n_splits=cfg.discriminator_splits,
        ),
    )

    if doc is not None:
        _run_stage(
            stages,
            report,
            "crossfit_bpe",
            "Published leave-one-quire-out BPE",
            lambda: cross_fit_bpe_scale_curve(
                unit_scale_text_by_quire(raw_text), checkpoints=(0, 16, 32, 64)
            ),
        )

        langs = {line.currier_language for line in doc.lines if line.currier_language}
        if {"A", "B"}.issubset(langs):
            _run_stage(
                stages,
                report,
                "currier",
                "Currier A/B analysis",
                lambda: build_currier_report(
                    raw_text,
                    boundary_bootstrap_repetitions=cfg.currier_bootstrap,
                    seed=int(seed),
                ),
            )
        else:
            _skip(stages, report, "currier", "Currier A/B analysis", "both Currier A and B labels are not present")

        labels = line_strata(doc, "quire")
        if len(set(labels)) >= 2:
            _run_stage(
                stages,
                report,
                "heterogeneity_control",
                "Adversarial heterogeneity control",
                lambda: heterogeneity_control(
                    analysis_text,
                    labels,
                    candidate,
                    block_size=200,
                    n_generated_replicates=cfg.discriminator_replicates,
                    seed=int(seed),
                    n_splits=cfg.discriminator_splits,
                ),
            )
        else:
            _skip(stages, report, "heterogeneity_control", "Adversarial heterogeneity control", "fewer than two quire strata")

        def replication():
            observed, diagnostics = automatic_replication_metrics(
                raw_text, shuffles=cfg.replication_shuffles
            )
            return {
                "observed": observed,
                "target_comparison": compare_to_targets(observed, tolerance=0.10),
                "diagnostics": diagnostics,
            }

        _run_stage(stages, report, "replication_gate", "Automatic literature replication gate", replication)
    else:
        for key, label in (
            ("crossfit_bpe", "Published leave-one-quire-out BPE"),
            ("currier", "Currier A/B analysis"),
            ("heterogeneity_control", "Adversarial heterogeneity control"),
            ("replication_gate", "Automatic literature replication gate"),
        ):
            _skip(stages, report, key, label, "requires IVTFF metadata")

    report["stages"] = stages
    report["completed_stages"] = sum(row["status"] == "completed" for row in stages)
    report["error_stages"] = sum(row["status"] == "error" for row in stages)
    report["skipped_stages"] = sum(row["status"] == "skipped" for row in stages)
    return _plain(report)


def pipeline_summary_markdown(report: dict[str, Any]) -> str:
    metrics = report.get("metrics", {}).get("corpus", {})
    score = report.get("scorecard", {}).get("result", {})
    disc = report.get("discriminator", {})
    scale = report.get("corpus_scale", {})
    lines = ["### Full-pipeline summary"]
    if scale.get("is_demo_scale"):
        lines.append(
            f"> ⚠️ **Demo-scale input ({scale.get('source_label', 'unspecified')}): "
            f"{scale.get('glyph_count', '?'):,} glyphs.** {scale.get('note', '')}"
            if isinstance(scale.get("glyph_count"), int)
            else f"> ⚠️ **Demo-scale input.** {scale.get('note', '')}"
        )
    lines.append(f"- Source: **{scale.get('source_label', 'unspecified')}**")
    lines.extend(
        [
            f"- Profile: **{report.get('profile', {}).get('label', '?')}**",
            f"- Input: **{report.get('input_kind', '?')}**",
            f"- Stages: **{report.get('completed_stages', 0)} completed**, "
            f"**{report.get('skipped_stages', 0)} skipped**, **{report.get('error_stages', 0)} errors**",
        ]
    )
    if metrics:
        lines.append(
            f"- Corpus: {metrics.get('glyph_count', '?')} glyphs, "
            f"{metrics.get('conventional_token_count', '?')} conventional tokens; "
            f"edge MI={metrics.get('cross_token_edge_mi_bits', '?')}"
        )
    lines.append(f"- Adversarial candidate: {report.get('adversarial_candidate', '?')}")
    if score and not score.get("error"):
        lines.append(
            f"- Scorecard: D∞={score.get('d_infinity', '?')}, "
            f"global p={score.get('global_p_value', '?')}, worst feature={score.get('worst_feature', '?')}"
        )
    if disc and not disc.get("error"):
        lines.append(
            f"- Grouped discriminator AUC: {disc.get('held_out_auc', '?')} "
            f"[{disc.get('auc_ci_low', '?')}, {disc.get('auc_ci_high', '?')}]"
        )
    lines.append("\nDetailed outputs for every stage are available in the JSON report below.")
    return "\n".join(lines)
