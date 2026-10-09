# -*- coding: utf-8 -*-
"""
Post-run analysis for encoder-level method experiments.

AUROC bar chart, per-subject deltas vs baseline, confusion counts, report update.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

PYTORCH_ROOT = Path(__file__).resolve().parents[1]
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))

from post_training.config import (  # noqa: E402
    INFER_THRESHOLD,
    METHOD_BASELINE_INF,
    METHOD_HOLDOUT_PKL,
    METHOD_LARGE_SHOT_COUNTS,
    METHOD_RUNS_DIR,
    METHOD_SHOT_BASELINE_INF,
    METHOD_SHOT_COUNTS,
    METHOD_SHOT_HOLDOUT_PKL,
    METHOD_SHOT_RUNS_DIR,
    head_only_inference_metrics,
    unified_head_only_inference_metrics,
    unified_partial_inference_metrics,
)
from post_training.shot_analysis import (  # noqa: E402
    _confusion_cell,
    binary_label_from_record,
    compute_prediction_deltas,
    load_json,
)


def _fmt_float(val: Any, digits: int = 4) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return str(val)


def _metrics_row(metrics: Dict[str, Any], *, delta_auc: Optional[float] = None) -> Dict[str, Any]:
    return {
        "auc_roc": metrics.get("auc_roc"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "tp": metrics.get("tp"),
        "tn": metrics.get("tn"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "delta_auc_roc": delta_auc,
    }


def plot_shot_auroc_comparison(
    shot_summaries: List[Dict[str, Any]],
    baseline_auc: float,
    output_path: Path,
) -> None:
    """Grouped bar chart: per shot, baseline vs head-only vs partial unfreeze."""
    shots = [s["shots_per_class"] for s in shot_summaries]
    n = len(shots)
    if n == 0:
        return

    x = np.arange(n)
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(7, n * 2.5), 4.5))

    baseline_aucs = [baseline_auc] * n
    head_aucs = [float(s["head_only"]["auc_roc"]) for s in shot_summaries]
    partial_aucs = [float(s["partial_unfreeze"]["auc_roc"]) for s in shot_summaries]

    ax.bar(x - width, baseline_aucs, width, label="Deploy baseline", color="#4C72B0")
    ax.bar(x, head_aucs, width, label="Head-only", color="#55A868")
    ax.bar(x + width, partial_aucs, width, label="Partial unfreeze", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}-shot" for k in shots])
    ax.set_ylabel("Holdout AUROC")
    ax.set_title("AUROC: baseline vs head-only vs partial unfreeze (shot_runs holdout)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(max(0, min(baseline_aucs + head_aucs + partial_aucs) - 0.05), 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_shot_per_subject_csv(
    baseline_records: List[Dict[str, Any]],
    shot_records: Dict[int, Dict[str, List[Dict[str, Any]]]],
    output_path: Path,
) -> None:
    """CSV with baseline + head-only + partial per shot."""
    fieldnames = [
        "subject_id",
        "source_file",
        "label",
        "baseline_prob",
        "baseline_pred",
        "baseline_cell",
    ]
    for k in sorted(shot_records):
        for model in ("head_only", "partial_unfreeze"):
            prefix = f"shot{k}_{model}"
            fieldnames.extend(
                [f"{prefix}_prob", f"{prefix}_pred", f"{prefix}_cell", f"flip_{prefix}"]
            )

    by_shot_model: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for k, models in shot_records.items():
        by_shot_model[k] = {}
        for model_name, records in models.items():
            by_shot_model[k][model_name] = {r["source_file"]: r for r in records}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(baseline_records, key=lambda x: x.get("subject_id", "")):
            src = r["source_file"]
            label_bin = binary_label_from_record(r)
            b_pred = int(r.get("prediction", 0))
            row: Dict[str, Any] = {
                "subject_id": r.get("subject_id"),
                "source_file": src,
                "label": label_bin,
                "baseline_prob": f"{float(r['probability']):.4f}",
                "baseline_pred": b_pred,
                "baseline_cell": _confusion_cell(label_bin, b_pred),
            }
            for k in sorted(shot_records):
                for model_name in ("head_only", "partial_unfreeze"):
                    rec = by_shot_model[k][model_name].get(src)
                    prefix = f"shot{k}_{model_name}"
                    if rec is None:
                        continue
                    m_pred = int(rec.get("prediction", 0))
                    row[f"{prefix}_prob"] = f"{float(rec['probability']):.4f}"
                    row[f"{prefix}_pred"] = m_pred
                    row[f"{prefix}_cell"] = _confusion_cell(label_bin, m_pred)
                    row[f"flip_{prefix}"] = "Y" if b_pred != m_pred else ""
            writer.writerow(row)


def write_shot_method_comparison_report(
    runs_dir: Path,
    *,
    shot_summaries: List[Dict[str, Any]],
    baseline_metrics: Dict[str, Any],
    prediction_deltas: Dict[int, Dict[str, Any]],
) -> Path:
    report_path = runs_dir / "SHOT_METHOD_COMPARISON_REPORT.md"
    baseline_auc = float(baseline_metrics["auc_roc"])

    lines: List[str] = [
        "# Shot-Method Comparison: Deploy vs Head-Only vs Partial Unfreeze",
        "",
        f"Holdout cohort: **`shot_runs/holdout_pkl`** ({int(baseline_metrics.get('n_labeled', 0))} labeled patches)",
        "",
        "> **Not comparable** to [`MODEL_COMPARISON_REPORT.md`](MODEL_COMPARISON_REPORT.md) "
        "(15-shot partial unfreeze on `shot_runs_large` holdout, 65 patches).",
        "",
        f"**Deploy baseline AUROC:** {_fmt_float(baseline_auc)}",
        "",
        "## Metrics overview",
        "",
        "| Shot | Model | AUROC | Δ vs baseline | Accuracy | Precision | Recall | F1 |",
        "|------|-------|-------|---------------|----------|-----------|--------|-----|",
    ]

    for summary in shot_summaries:
        k = summary["shots_per_class"]
        for model_key, label in [
            ("baseline", "Deploy baseline"),
            ("head_only", "Head-only"),
            ("partial_unfreeze", "Partial unfreeze"),
        ]:
            m = summary[model_key]
            dauc = m.get("delta_auc_roc")
            dauc_str = "—" if model_key == "baseline" else _fmt_float(dauc, 4)
            lines.append(
                f"| {k} | {label} | {_fmt_float(m.get('auc_roc'))} | {dauc_str} | "
                f"{_fmt_float(m.get('accuracy'))} | {_fmt_float(m.get('precision'))} | "
                f"{_fmt_float(m.get('recall'))} | {_fmt_float(m.get('f1'))} |"
            )

    for summary in shot_summaries:
        k = summary["shots_per_class"]
        lines.extend(
            [
                "",
                f"## {k}-shot confusion matrices (threshold=0.5)",
                "",
                "From `metrics.json`. Rows=actual, cols=predicted.",
                "",
            ]
        )
        for model_key, label in [
            ("baseline", "Deploy baseline"),
            ("head_only", "Head-only"),
            ("partial_unfreeze", "Partial unfreeze"),
        ]:
            m = summary[model_key]
            tp, tn, fp, fn = int(m["tp"]), int(m["tn"]), int(m["fp"]), int(m["fn"])
            lines.extend(
                [
                    f"### {label}",
                    "",
                    "|  | Pred 0 | Pred 1 |",
                    "|--|--------|--------|",
                    f"| **Actual 0** | TN {tn} | FP {fp} |",
                    f"| **Actual 1** | FN {fn} | TP {tp} |",
                    "",
                ]
            )

        deltas = prediction_deltas.get(k, {})
        ho = deltas.get("head_only", {})
        pu = deltas.get("partial_unfreeze", {})
        lines.extend(
            [
                f"### {k}-shot prediction flips vs baseline",
                "",
                f"- **Head-only:** {ho.get('n_flipped', 0)} flipped "
                f"(mean |Δprob|={ho.get('mean_abs_prob_delta', 0):.4f})",
                f"- **Partial unfreeze:** {pu.get('n_flipped', 0)} flipped "
                f"(mean |Δprob|={pu.get('mean_abs_prob_delta', 0):.4f})",
            ]
        )
        if pu.get("flip_breakdown"):
            lines.append(
                f"  - Partial flip transitions: "
                + ", ".join(f"{kk}={vv}" for kk, vv in pu["flip_breakdown"].items())
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Leaderboard: `{runs_dir / 'shot_method_leaderboard.json'}`",
            f"- Per-shot partial unfreeze: `{runs_dir}/shot{{K}}/partial_unfreeze/`",
            f"- AUROC chart: `{runs_dir / 'analysis' / 'shot_auroc_comparison.png'}`",
            f"- Per-subject CSV: `{runs_dir / 'analysis' / 'shot_holdout_per_subject_results.csv'}`",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved shot method comparison report: {report_path}")
    return report_path


def run_shot_method_comparison(
    runs_dir: Path,
    *,
    shots: Optional[List[int]] = None,
    shot_runs_dir: Path = METHOD_SHOT_RUNS_DIR,
) -> Dict[str, Any]:
    runs_dir = runs_dir.resolve()
    shots = sorted(set(shots or METHOD_SHOT_COUNTS))
    analysis_dir = runs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics_path = METHOD_SHOT_BASELINE_INF / "metrics.json"
    baseline_results_path = METHOD_SHOT_BASELINE_INF / "inference_results.json"
    if not baseline_metrics_path.is_file():
        raise FileNotFoundError(f"Baseline metrics not found: {baseline_metrics_path}")
    baseline_metrics = load_json(baseline_metrics_path)
    baseline_records = load_json(baseline_results_path)
    baseline_auc = float(baseline_metrics["auc_roc"])

    shot_summaries: List[Dict[str, Any]] = []
    prediction_deltas: Dict[int, Dict[str, Any]] = {}
    shot_records: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}

    for k in shots:
        head_metrics_path = head_only_inference_metrics(k, shot_runs_dir=shot_runs_dir)
        partial_metrics_path = runs_dir / f"shot{k}" / "partial_unfreeze" / "inference" / "metrics.json"
        partial_results_path = runs_dir / f"shot{k}" / "partial_unfreeze" / "inference" / "inference_results.json"
        head_results_path = shot_runs_dir / f"shot{k}" / "inference" / "inference_results.json"

        if not partial_metrics_path.is_file():
            print(f"Skipping shot{k}: partial unfreeze metrics not found at {partial_metrics_path}")
            continue
        if not head_metrics_path.is_file():
            raise FileNotFoundError(f"Head-only metrics not found: {head_metrics_path}")

        head_metrics = load_json(head_metrics_path)
        partial_metrics = load_json(partial_metrics_path)
        head_auc = float(head_metrics["auc_roc"])
        partial_auc = float(partial_metrics["auc_roc"])

        summary = {
            "shots_per_class": k,
            "baseline": _metrics_row(baseline_metrics),
            "head_only": _metrics_row(head_metrics, delta_auc=head_auc - baseline_auc),
            "partial_unfreeze": _metrics_row(partial_metrics, delta_auc=partial_auc - baseline_auc),
        }
        shot_summaries.append(summary)

        head_records = load_json(head_results_path)
        partial_records = load_json(partial_results_path)
        shot_records[k] = {"head_only": head_records, "partial_unfreeze": partial_records}
        prediction_deltas[k] = {
            "head_only": compute_prediction_deltas(baseline_records, head_records),
            "partial_unfreeze": compute_prediction_deltas(baseline_records, partial_records),
        }

    plot_shot_auroc_comparison(
        shot_summaries, baseline_auc, analysis_dir / "shot_auroc_comparison.png"
    )
    if shot_records:
        export_shot_per_subject_csv(
            baseline_records, shot_records, analysis_dir / "shot_holdout_per_subject_results.csv"
        )
        print(f"Saved per-subject CSV: {analysis_dir / 'shot_holdout_per_subject_results.csv'}")

    write_shot_method_comparison_report(
        runs_dir,
        shot_summaries=shot_summaries,
        baseline_metrics=baseline_metrics,
        prediction_deltas=prediction_deltas,
    )

    summary = {
        "runs_dir": str(runs_dir),
        "shots_evaluated": shots,
        "holdout_pkl_dir": str(METHOD_SHOT_HOLDOUT_PKL),
        "baseline_auc_roc": baseline_auc,
        "shot_summaries": shot_summaries,
        "prediction_deltas": prediction_deltas,
    }
    summary_path = analysis_dir / "shot_method_comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved analysis summary: {summary_path}")
    return summary


def plot_unified_shot_scaling(
    shot_summaries: List[Dict[str, Any]],
    baseline_auc: float,
    output_path: Path,
) -> None:
    """Line chart: AUROC vs support subjects for baseline, head-only, partial unfreeze."""
    if not shot_summaries:
        return

    support_ns = [s["n_support_subjects"] for s in shot_summaries]
    baseline_aucs = [baseline_auc] * len(support_ns)
    head_aucs = [float(s["head_only"]["auc_roc"]) for s in shot_summaries]
    partial_aucs = [float(s["partial_unfreeze"]["auc_roc"]) for s in shot_summaries]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(support_ns, baseline_aucs, "o--", label="Deploy baseline", color="#4C72B0", linewidth=2)
    ax.plot(support_ns, head_aucs, "o-", label="Head-only", color="#55A868", linewidth=2)
    ax.plot(support_ns, partial_aucs, "o-", label="Partial unfreeze", color="#DD8452", linewidth=2)

    ax.set_xlabel("Support subjects (pos + neg)")
    ax.set_ylabel("Holdout AUROC")
    ax.set_title("AUROC scaling on unified holdout (shot_runs_large)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    all_aucs = baseline_aucs + head_aucs + partial_aucs
    ax.set_ylim(max(0, min(all_aucs) - 0.05), min(1.0, max(all_aucs) + 0.02))
    ax.set_xticks(support_ns)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_unified_shot_scaling_report(
    runs_dir: Path,
    *,
    shot_summaries: List[Dict[str, Any]],
    baseline_metrics: Dict[str, Any],
    prediction_deltas: Dict[int, Dict[str, Any]],
) -> Path:
    report_path = runs_dir / "UNIFIED_SHOT_SCALING_REPORT.md"
    baseline_auc = float(baseline_metrics["auc_roc"])

    lines: List[str] = [
        "# Unified Shot Scaling: Deploy vs Head-Only vs Partial Unfreeze",
        "",
        f"Holdout cohort: **`shot_runs_large/holdout_pkl`** "
        f"({int(baseline_metrics.get('n_labeled', 0))} labeled patches)",
        "",
        "> **Comparable across all K.** Supersedes cross-cohort numbers in "
        "[`SHOT_METHOD_COMPARISON_REPORT.md`](SHOT_METHOD_COMPARISON_REPORT.md) "
        "(1/5-shot on `shot_runs` holdout, 85 patches).",
        "",
        f"**Deploy baseline AUROC:** {_fmt_float(baseline_auc)} (same for all K)",
        "",
        "## Scaling overview",
        "",
        "| Shot | Support n | Baseline AUROC | Head-only | Partial unfreeze | Δ partial vs HO |",
        "|------|-----------|----------------|-----------|------------------|-----------------|",
    ]

    for summary in shot_summaries:
        k = summary["shots_per_class"]
        n_subj = summary["n_support_subjects"]
        ho = summary["head_only"]
        pu = summary["partial_unfreeze"]
        ho_auc = float(ho["auc_roc"])
        pu_auc = float(pu["auc_roc"])
        delta_pu_ho = pu_auc - ho_auc
        lines.append(
            f"| {k} | {n_subj} | {_fmt_float(baseline_auc)} | {_fmt_float(ho_auc)} | "
            f"{_fmt_float(pu_auc)} | {delta_pu_ho:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Full metrics",
            "",
            "| Shot | Model | AUROC | Δ vs baseline | Accuracy | Precision | Recall | F1 |",
            "|------|-------|-------|---------------|----------|-----------|--------|-----|",
        ]
    )

    for summary in shot_summaries:
        k = summary["shots_per_class"]
        for model_key, label in [
            ("baseline", "Deploy baseline"),
            ("head_only", "Head-only"),
            ("partial_unfreeze", "Partial unfreeze"),
        ]:
            m = summary[model_key]
            dauc = m.get("delta_auc_roc")
            dauc_str = "—" if model_key == "baseline" else _fmt_float(dauc, 4)
            lines.append(
                f"| {k} | {label} | {_fmt_float(m.get('auc_roc'))} | {dauc_str} | "
                f"{_fmt_float(m.get('accuracy'))} | {_fmt_float(m.get('precision'))} | "
                f"{_fmt_float(m.get('recall'))} | {_fmt_float(m.get('f1'))} |"
            )

    for summary in shot_summaries:
        k = summary["shots_per_class"]
        lines.extend(
            [
                "",
                f"## {k}-shot confusion matrices (threshold=0.5)",
                "",
                "From `metrics.json`. Rows=actual, cols=predicted.",
                "",
            ]
        )
        for model_key, label in [
            ("baseline", "Deploy baseline"),
            ("head_only", "Head-only"),
            ("partial_unfreeze", "Partial unfreeze"),
        ]:
            m = summary[model_key]
            tp, tn, fp, fn = int(m["tp"]), int(m["tn"]), int(m["fp"]), int(m["fn"])
            lines.extend(
                [
                    f"### {label}",
                    "",
                    "|  | Pred 0 | Pred 1 |",
                    "|--|--------|--------|",
                    f"| **Actual 0** | TN {tn} | FP {fp} |",
                    f"| **Actual 1** | FN {fn} | TP {tp} |",
                    "",
                ]
            )

        deltas = prediction_deltas.get(k, {})
        ho = deltas.get("head_only", {})
        pu = deltas.get("partial_unfreeze", {})
        lines.extend(
            [
                f"### {k}-shot prediction flips vs baseline",
                "",
                f"- **Head-only:** {ho.get('n_flipped', 0)} flipped "
                f"(mean |Δprob|={ho.get('mean_abs_prob_delta', 0):.4f})",
                f"- **Partial unfreeze:** {pu.get('n_flipped', 0)} flipped "
                f"(mean |Δprob|={pu.get('mean_abs_prob_delta', 0):.4f})",
            ]
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Leaderboard: `{runs_dir / 'unified_shot_scaling_leaderboard.json'}`",
            f"- Head-only re-infer: `{runs_dir}/shot{{K}}/head_only/inference/`",
            f"- Partial unfreeze: `{runs_dir}/shot{{K}}/partial_unfreeze/inference/`",
            f"- AUROC scaling plot: `{runs_dir / 'analysis' / 'unified_shot_scaling_auroc.png'}`",
            f"- Per-subject CSV: `{runs_dir / 'analysis' / 'unified_shot_holdout_per_subject_results.csv'}`",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved unified shot scaling report: {report_path}")
    return report_path


def run_unified_shot_scaling_comparison(
    runs_dir: Path,
    *,
    shots: Optional[List[int]] = None,
) -> Dict[str, Any]:
    runs_dir = runs_dir.resolve()
    shots = sorted(set(shots or METHOD_LARGE_SHOT_COUNTS))
    analysis_dir = runs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics_path = METHOD_BASELINE_INF / "metrics.json"
    baseline_results_path = METHOD_BASELINE_INF / "inference_results.json"
    if not baseline_metrics_path.is_file():
        raise FileNotFoundError(f"Baseline metrics not found: {baseline_metrics_path}")
    baseline_metrics = load_json(baseline_metrics_path)
    baseline_records = load_json(baseline_results_path)
    baseline_auc = float(baseline_metrics["auc_roc"])

    shot_summaries: List[Dict[str, Any]] = []
    prediction_deltas: Dict[int, Dict[str, Any]] = {}
    shot_records: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}

    for k in shots:
        head_metrics_path = unified_head_only_inference_metrics(k, method_runs_dir=runs_dir)
        partial_metrics_path = unified_partial_inference_metrics(k, method_runs_dir=runs_dir)
        head_results_path = runs_dir / f"shot{k}" / "head_only" / "inference" / "inference_results.json"
        partial_results_path = (
            runs_dir / f"shot{k}" / "partial_unfreeze" / "inference" / "inference_results.json"
        )

        if not partial_metrics_path.is_file():
            print(f"Skipping shot{k}: partial unfreeze metrics not found at {partial_metrics_path}")
            continue
        if not head_metrics_path.is_file():
            raise FileNotFoundError(f"Head-only metrics not found: {head_metrics_path}")

        head_metrics = load_json(head_metrics_path)
        partial_metrics = load_json(partial_metrics_path)
        head_auc = float(head_metrics["auc_roc"])
        partial_auc = float(partial_metrics["auc_roc"])

        summary = {
            "shots_per_class": k,
            "n_support_subjects": k * 2,
            "baseline": _metrics_row(baseline_metrics),
            "head_only": _metrics_row(head_metrics, delta_auc=head_auc - baseline_auc),
            "partial_unfreeze": _metrics_row(partial_metrics, delta_auc=partial_auc - baseline_auc),
            "delta_partial_vs_head_only": partial_auc - head_auc,
        }
        shot_summaries.append(summary)

        head_records = load_json(head_results_path)
        partial_records = load_json(partial_results_path)
        shot_records[k] = {"head_only": head_records, "partial_unfreeze": partial_records}
        prediction_deltas[k] = {
            "head_only": compute_prediction_deltas(baseline_records, head_records),
            "partial_unfreeze": compute_prediction_deltas(baseline_records, partial_records),
        }

    plot_unified_shot_scaling(
        shot_summaries, baseline_auc, analysis_dir / "unified_shot_scaling_auroc.png"
    )
    print(f"Saved AUROC scaling plot: {analysis_dir / 'unified_shot_scaling_auroc.png'}")

    if shot_records:
        export_shot_per_subject_csv(
            baseline_records,
            shot_records,
            analysis_dir / "unified_shot_holdout_per_subject_results.csv",
        )
        print(
            f"Saved per-subject CSV: {analysis_dir / 'unified_shot_holdout_per_subject_results.csv'}"
        )

    write_unified_shot_scaling_report(
        runs_dir,
        shot_summaries=shot_summaries,
        baseline_metrics=baseline_metrics,
        prediction_deltas=prediction_deltas,
    )

    summary = {
        "runs_dir": str(runs_dir),
        "shots_evaluated": shots,
        "holdout_pkl_dir": str(METHOD_HOLDOUT_PKL),
        "baseline_auc_roc": baseline_auc,
        "shot_summaries": shot_summaries,
        "prediction_deltas": prediction_deltas,
    }
    summary_path = analysis_dir / "unified_shot_scaling_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved analysis summary: {summary_path}")
    return summary


def plot_auroc_comparison(payload: Dict[str, Any], output_path: Path) -> None:
    baseline_auc = float(payload.get("baseline", {}).get("auc_roc", 0))
    ranked = payload.get("ranked", [])
    head_ref = payload.get("head_only_reference", {})

    names = ["baseline"]
    aucs = [baseline_auc]
    colors = ["#4C72B0"]

    for row in ranked:
        names.append(row.get("name", "?"))
        aucs.append(float(row.get("holdout_auc_roc", 0)))
        colors.append("#DD8452")

    if head_ref.get("holdout_auc_roc") is not None:
        names.append("head_only_15shot")
        aucs.append(float(head_ref["holdout_auc_roc"]))
        colors.append("#55A868")

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4.5))
    x = np.arange(len(names))
    bars = ax.bar(x, aucs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(baseline_auc, color="gray", linestyle="--", linewidth=1, label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Holdout AUROC")
    ax.set_title("AUROC: baseline vs encoder methods")
    ax.set_ylim(max(0, min(aucs) - 0.05), min(1.0, max(aucs) + 0.02))
    for bar, val in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def confusion_counts_from_metrics(metrics_path: Path) -> Dict[str, int]:
    """Authoritative TP/TN/FP/FN from 03_Infer.py metrics.json."""
    m = load_json(metrics_path)
    return {"TP": int(m["tp"]), "TN": int(m["tn"]), "FP": int(m["fp"]), "FN": int(m["fn"])}


def confusion_counts(records: List[Dict[str, Any]], threshold: float = INFER_THRESHOLD) -> Dict[str, int]:
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for r in records:
        actual = binary_label_from_record(r)
        pred = int(r.get("prediction", int(float(r.get("probability", 0)) >= threshold)))
        cell = _confusion_cell(actual, pred)
        counts[cell] = counts.get(cell, 0) + 1
    return counts


def export_per_subject_csv(
    baseline_records: List[Dict[str, Any]],
    method_records: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
) -> None:
    method_by_source: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method_name, records in method_records.items():
        method_by_source[method_name] = {r["source_file"]: r for r in records}

    fieldnames = [
        "subject_id",
        "source_file",
        "label",
        "baseline_prob",
        "baseline_pred",
        "baseline_cell",
    ]
    for method_name in sorted(method_records):
        fieldnames.extend(
            [
                f"{method_name}_prob",
                f"{method_name}_pred",
                f"{method_name}_cell",
                f"{method_name}_delta_prob",
                f"flip_{method_name}",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(baseline_records, key=lambda x: x.get("subject_id", x.get("source_file", ""))):
            src = r["source_file"]
            label_bin = binary_label_from_record(r)
            b_pred = int(r.get("prediction", 0))
            row: Dict[str, Any] = {
                "subject_id": r.get("subject_id"),
                "source_file": src,
                "label": label_bin,
                "baseline_prob": f"{float(r['probability']):.4f}",
                "baseline_pred": b_pred,
                "baseline_cell": _confusion_cell(label_bin, b_pred),
            }
            for method_name in sorted(method_records):
                mrec = method_by_source[method_name].get(src)
                if mrec is None:
                    continue
                m_pred = int(mrec.get("prediction", 0))
                m_prob = float(mrec["probability"])
                row[f"{method_name}_prob"] = f"{m_prob:.4f}"
                row[f"{method_name}_pred"] = m_pred
                row[f"{method_name}_cell"] = _confusion_cell(label_bin, m_pred)
                row[f"{method_name}_delta_prob"] = f"{m_prob - float(r['probability']):+.4f}"
                row[f"flip_{method_name}"] = "Y" if b_pred != m_pred else ""
            writer.writerow(row)


def append_analysis_to_report(
    runs_dir: Path,
    analysis_summary: Dict[str, Any],
) -> None:
    report_path = runs_dir / "METHOD_EXPERIMENT_REPORT.md"
    if not report_path.is_file():
        return

    text = report_path.read_text(encoding="utf-8").rstrip()
    if "## Analysis (auto-generated)" in text:
        text = text.split("## Analysis (auto-generated)")[0].rstrip()
    lines = [text]
    deltas = analysis_summary.get("prediction_deltas", {})
    confusion = analysis_summary.get("confusion_counts", {})
    training = analysis_summary.get("training_summary", [])

    lines.extend(["", "## Analysis (auto-generated)", ""])

    lines.append("### Prediction changes vs baseline")
    lines.append("")
    if not deltas:
        lines.append("No prediction delta data available.")
    else:
        lines.append("| Method | Paired | Flipped | Mean Δprob | Mean |Δprob| |")
        lines.append("|--------|--------|---------|------------|-------------|")
        for method_name, stats in sorted(deltas.items()):
            lines.append(
                f"| {method_name} | {stats['n_paired']} | {stats['n_flipped']} | "
                f"{stats.get('mean_prob_delta', 0):+.4f} | {stats.get('mean_abs_prob_delta', 0):.4f} |"
            )
        lines.append("")
        for method_name, stats in sorted(deltas.items()):
            if stats.get("flip_breakdown"):
                lines.append(
                    f"- **{method_name}** flip transitions: "
                    + ", ".join(f"{k}={v}" for k, v in stats["flip_breakdown"].items())
                )

    lines.extend(
        [
            "",
            "### Confusion matrix counts (threshold=0.5)",
            "",
            "From `metrics.json` (same source as `MODEL_COMPARISON_REPORT.md`).",
            "",
        ]
    )
    lines.append("| Model | TP | TN | FP | FN |")
    lines.append("|-------|----|----|----|-----|")
    for model_name, counts in sorted(confusion.items()):
        lines.append(
            f"| {model_name} | {counts.get('TP', 0)} | {counts.get('TN', 0)} | "
            f"{counts.get('FP', 0)} | {counts.get('FN', 0)} |"
        )

    lines.extend(["", "### Training convergence", ""])
    if training:
        lines.append("| Method | Best epoch | Best train loss | Trainable params |")
        lines.append("|--------|------------|-----------------|------------------|")
        for row in training:
            fs = row.get("freeze_stats") or {}
            lines.append(
                f"| {row['method']} | {row.get('best_epoch')} | {row.get('best_loss', 'N/A')} | "
                f"{fs.get('trainable_params', 'N/A')} |"
            )
    else:
        lines.append("No training manifests found.")

    lines.extend(
        [
            "",
            "### Cross-reference: head-only 15-shot",
            "",
            "Head-only post-training at 30 support subjects (`shot_runs_large/shot15`) "
            "achieved AUROC **0.7466** (Δ **-0.0010** vs baseline **0.7476**). "
            "Encoder methods should beat this to justify representation-level FT.",
            "",
            f"Per-subject CSV: `{runs_dir / 'analysis' / 'holdout_per_subject_results.csv'}`",
            f"AUROC chart: `{runs_dir / 'analysis' / 'auroc_comparison.png'}`",
            f"Analysis JSON: `{runs_dir / 'analysis' / 'analysis_summary.json'}`",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated report with analysis: {report_path}")


def load_training_summary(runs_dir: Path, method_name: str) -> Optional[Dict[str, Any]]:
    manifest_path = runs_dir / method_name / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = load_json(manifest_path)
    training = manifest.get("training", {})
    return {
        "method": method_name,
        "best_epoch": training.get("best_epoch"),
        "best_loss": training.get("best_loss"),
        "freeze_stats": training.get("freeze_stats"),
        "label_counts": training.get("label_counts"),
    }


def run_method_analysis(runs_dir: Path) -> Dict[str, Any]:
    runs_dir = runs_dir.resolve()
    leaderboard_path = runs_dir / "leaderboard.json"
    if not leaderboard_path.is_file():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_path}")

    payload = load_json(leaderboard_path)
    analysis_dir = runs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    holdout_pkl = Path(payload.get("holdout_pkl_dir", METHOD_HOLDOUT_PKL))
    baseline_results_path = holdout_pkl / "inference_baseline" / "inference_results.json"
    if not baseline_results_path.is_file():
        baseline_results_path = runs_dir.parent / "shot_runs_large" / "holdout_pkl" / "inference_baseline" / "inference_results.json"
    if not baseline_results_path.is_file():
        raise FileNotFoundError(f"Baseline inference not found: {baseline_results_path}")
    baseline_records = load_json(baseline_results_path)

    prediction_deltas: Dict[str, Any] = {}
    baseline_metrics_path = baseline_results_path.parent / "metrics.json"
    confusion_counts_map: Dict[str, Any] = {
        "baseline": confusion_counts_from_metrics(baseline_metrics_path),
    }
    training_summary: List[Dict[str, Any]] = []
    method_records: Dict[str, List[Dict[str, Any]]] = {}

    ranked = payload.get("ranked", [])
    for row in ranked:
        method_name = row.get("name")
        if not method_name:
            continue
        post_results_path = runs_dir / method_name / "inference" / "inference_results.json"
        if post_results_path.is_file():
            post_records = load_json(post_results_path)
            method_records[method_name] = post_records
            prediction_deltas[method_name] = compute_prediction_deltas(baseline_records, post_records)
            method_metrics_path = runs_dir / method_name / "inference" / "metrics.json"
            if method_metrics_path.is_file():
                confusion_counts_map[method_name] = confusion_counts_from_metrics(method_metrics_path)
            else:
                confusion_counts_map[method_name] = confusion_counts(post_records)

        train_row = load_training_summary(runs_dir, method_name)
        if train_row:
            training_summary.append(train_row)

    plot_auroc_comparison(payload, analysis_dir / "auroc_comparison.png")

    csv_path = analysis_dir / "holdout_per_subject_results.csv"
    if method_records:
        export_per_subject_csv(baseline_records, method_records, csv_path)
        print(f"Saved per-subject CSV: {csv_path}")

    summary = {
        "runs_dir": str(runs_dir),
        "experiment_id": payload.get("experiment_id"),
        "baseline_auc_roc": payload.get("baseline", {}).get("auc_roc"),
        "best_method": payload.get("best", {}).get("name") if payload.get("best") else None,
        "prediction_deltas": prediction_deltas,
        "confusion_counts": confusion_counts_map,
        "training_summary": training_summary,
        "head_only_reference": payload.get("head_only_reference"),
    }

    summary_path = analysis_dir / "analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved analysis summary: {summary_path}")

    append_analysis_to_report(runs_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze encoder method experiment results")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Method runs directory containing leaderboard.json",
    )
    parser.add_argument(
        "--shot-compare",
        action="store_true",
        help="Run 1/5-shot 3-way comparison (baseline vs head-only vs partial unfreeze)",
    )
    parser.add_argument(
        "--unified-compare",
        action="store_true",
        help="Run unified 1/5/10/15-shot scaling comparison (shot_runs_large holdout)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=None,
        help="Shot counts for --shot-compare or --unified-compare",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir or METHOD_RUNS_DIR
    if args.unified_compare:
        run_unified_shot_scaling_comparison(runs_dir, shots=args.shots)
    elif args.shot_compare:
        run_shot_method_comparison(runs_dir, shots=args.shots)
    else:
        run_method_analysis(runs_dir)


if __name__ == "__main__":
    main()
