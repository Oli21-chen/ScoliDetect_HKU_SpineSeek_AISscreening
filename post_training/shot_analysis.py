# -*- coding: utf-8 -*-
"""
Post-run analysis for few-shot / large-support shot experiments.

Reads leaderboard.json and per-shot inference manifests; writes plots and
summary tables under runs_dir/analysis/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PYTORCH_ROOT = Path(__file__).resolve().parents[1]
CODE_VIDEO_ROOT = PYTORCH_ROOT.parent
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))

from post_training.config import (  # noqa: E402
    INFER_THRESHOLD,
    SCOLI_ROOT,
    SHOT_RUNS_DIR,
    STEP_04,
    TABLE_DIR,
    python_cmd,
)
from post_training.compare_models import (  # noqa: E402
    pair_records_by_source,
    run_baseline_vs_posttrain_comparison,
)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_label(label: Any) -> float:
    if label is None:
        return 0.0
    if isinstance(label, (list, tuple)):
        if not label:
            return 0.0
        return float(label[0])
    return float(label)


def binary_label_from_record(record: Dict[str, Any]) -> int:
    """Binary ground-truth class (0/1) from an inference_results.json record."""
    if record.get("label") is not None:
        return int(float(record["label"]) >= 0.5)
    lv = record.get("label_value")
    if isinstance(lv, (list, tuple)) and lv:
        from post_training.config import BINARY_THRESHOLD

        return int(max(float(x) for x in lv) >= BINARY_THRESHOLD)
    if lv is not None:
        return int(float(lv) >= 0.5)
    return 0


def _confusion_cell(label: Any, pred: int) -> str:
    if isinstance(label, dict):
        actual = binary_label_from_record(label)
    else:
        actual = int(float(label) >= 0.5)
    if actual == 1 and pred == 1:
        return "TP"
    if actual == 0 and pred == 0:
        return "TN"
    if actual == 0 and pred == 1:
        return "FP"
    return "FN"


def compute_prediction_deltas(
    baseline_records: List[Dict[str, Any]],
    posttrain_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    paired = pair_records_by_source(baseline_records, posttrain_records)
    if not paired:
        return {
            "n_paired": 0,
            "n_flipped": 0,
            "mean_prob_delta": None,
            "mean_abs_prob_delta": None,
            "flip_breakdown": {},
            "flipped_subjects": [],
        }

    deltas = [p["posttrain_prob"] - p["baseline_prob"] for p in paired]
    flipped = [p for p in paired if p["baseline_pred"] != p["posttrain_pred"]]
    flip_breakdown: Dict[str, int] = {}
    flipped_subjects: List[Dict[str, Any]] = []

    for p in flipped:
        label_rec = {
            "label": p.get("label"),
            "label_value": p.get("label_value"),
        }
        base_cell = _confusion_cell(label_rec, p["baseline_pred"])
        post_cell = _confusion_cell(label_rec, p["posttrain_pred"])
        key = f"{base_cell}->{post_cell}"
        flip_breakdown[key] = flip_breakdown.get(key, 0) + 1
        flipped_subjects.append(
            {
                "source_file": p["source_file"],
                "label": binary_label_from_record(
                    {"label": p.get("label"), "label_value": p.get("label_value")}
                ),
                "baseline_prob": p["baseline_prob"],
                "posttrain_prob": p["posttrain_prob"],
                "transition": key,
            }
        )

    return {
        "n_paired": len(paired),
        "n_flipped": len(flipped),
        "mean_prob_delta": float(np.mean(deltas)),
        "mean_abs_prob_delta": float(np.mean(np.abs(deltas))),
        "max_abs_prob_delta": float(np.max(np.abs(deltas))),
        "flip_breakdown": flip_breakdown,
        "flipped_subjects": flipped_subjects,
    }


def plot_auroc_scaling(
    payload: Dict[str, Any],
    output_path: Path,
) -> None:
    baseline_auc = float(payload.get("baseline", {}).get("auc_roc", 0))
    ranked = payload.get("ranked", [])
    if not ranked:
        return

    xs = [0]
    ys = [baseline_auc]
    labels = ["baseline"]

    for row in sorted(ranked, key=lambda r: r.get("shots_per_class", 0)):
        xs.append(row.get("n_support_subjects", 0))
        ys.append(row.get("holdout_auc_roc", baseline_auc))
        labels.append(f"{row.get('shots_per_class')}-shot")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(baseline_auc, color="gray", linestyle="--", linewidth=1, label="Baseline AUROC")
    ax.plot(xs[1:], ys[1:], marker="o", color="#DD8452", linewidth=2, label="Post-trained")
    ax.scatter([0], [baseline_auc], marker="s", color="#4C72B0", s=60, zorder=5, label="Deploy baseline")
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Support subjects (pos + neg)")
    ax.set_ylabel("Holdout AUROC")
    ax.set_title("AUROC vs support size")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_per_shot_analysis(
    runs_dir: Path,
    shots_per_class: int,
    analysis_dir: Path,
    *,
    skip_symmetry: bool = True,
) -> None:
    shot_dir = runs_dir / f"shot{shots_per_class}"
    results_json = shot_dir / "inference" / "inference_results.json"
    if not results_json.is_file():
        print(f"Skipping 04 analysis for shot{shots_per_class}: missing {results_json}")
        return

    out_parent = analysis_dir / f"shot{shots_per_class}"
    out_parent.mkdir(parents=True, exist_ok=True)
    cmd = python_cmd(
        [
            str(STEP_04),
            "--results-json",
            str(results_json),
            "--pkl-dir",
            str(runs_dir / "holdout_pkl"),
            "--table-dir",
            str(TABLE_DIR),
            "--scoli-root",
            str(SCOLI_ROOT),
        ]
    )
    if skip_symmetry:
        cmd.append("--skip-symmetry")
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(CODE_VIDEO_ROOT))


def load_training_summary(runs_dir: Path, shots_per_class: int) -> Optional[Dict[str, Any]]:
    manifest_path = runs_dir / f"shot{shots_per_class}" / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = load_json(manifest_path)
    training = manifest.get("training", {})
    return {
        "shots_per_class": shots_per_class,
        "n_support_subjects": manifest.get("n_support_subjects"),
        "best_epoch": training.get("best_epoch"),
        "best_loss": training.get("best_loss"),
        "label_counts": training.get("label_counts"),
        "final_epoch_loss": (
            training.get("history", [])[-1].get("loss")
            if training.get("history")
            else None
        ),
    }


def load_prior_shot_runs_summary() -> Optional[Dict[str, Any]]:
    prior_lb = SHOT_RUNS_DIR / "leaderboard.json"
    if not prior_lb.is_file():
        return None
    return load_json(prior_lb)


def append_analysis_to_report(
    runs_dir: Path,
    analysis_summary: Dict[str, Any],
) -> None:
    report_path = runs_dir / "SHOT_EXPERIMENT_REPORT.md"
    if not report_path.is_file():
        return

    lines = [report_path.read_text(encoding="utf-8").rstrip()]
    prior = analysis_summary.get("prior_shot_runs")
    deltas = analysis_summary.get("prediction_deltas", {})
    training = analysis_summary.get("training_summary", [])
    best_shot = analysis_summary.get("best_shot")

    lines.extend(["", "## Analysis (auto-generated)", ""])

    lines.append("### Prediction changes vs baseline")
    lines.append("")
    if not deltas:
        lines.append("No prediction delta data available.")
    else:
        lines.append("| Shot | Paired | Flipped | Mean Δprob | Mean |Δprob| |")
        lines.append("|------|--------|---------|------------|-------------|")
        for shot_key, stats in sorted(deltas.items(), key=lambda x: int(x[0].replace("shot", ""))):
            lines.append(
                f"| {shot_key.replace('shot', '')} | {stats['n_paired']} | {stats['n_flipped']} | "
                f"{stats.get('mean_prob_delta', 0):+.4f} | {stats.get('mean_abs_prob_delta', 0):.4f} |"
            )
        lines.append("")
        for shot_key, stats in sorted(deltas.items(), key=lambda x: int(x[0].replace("shot", ""))):
            if stats.get("flip_breakdown"):
                lines.append(
                    f"- **{shot_key}** flip transitions: "
                    + ", ".join(f"{k}={v}" for k, v in stats["flip_breakdown"].items())
                )

    lines.extend(["", "### Training convergence", ""])
    if training:
        lines.append("| Shot | Support n | Best epoch | Best train loss | Final epoch loss |")
        lines.append("|------|-----------|------------|-----------------|------------------|")
        for row in training:
            lines.append(
                f"| {row['shots_per_class']} | {row.get('n_support_subjects')} | "
                f"{row.get('best_epoch')} | {row.get('best_loss', 'N/A')} | "
                f"{row.get('final_epoch_loss', 'N/A')} |"
            )
    else:
        lines.append("No training manifests found.")

    if best_shot is not None:
        lines.extend(
            [
                "",
                f"### Best-shot comparison artifacts",
                "",
                f"Baseline vs **{best_shot}-shot** comparison plots: "
                f"`{runs_dir / 'analysis' / 'comparison_best'}/`",
            ]
        )

    lines.extend(["", "### Cross-experiment reference (`shot_runs/` 1/3/5-shot)", ""])
    if prior:
        prior_baseline = prior.get("baseline", {}).get("auc_roc")
        prior_holdout_n = prior.get("fixed_holdout_n_subjects")
        lines.append(
            f"Previous small-support experiment (`shot_runs/`, holdout n={prior_holdout_n}) "
            f"baseline AUROC **{prior_baseline:.4f}** with best delta **+0.0006** at 3/5-shot. "
            "**Do not compare absolute AUROC** across directories — holdout cohorts differ."
        )
        lines.append("")
        lines.append("| Prior shot | AUROC | Δ vs prior baseline |")
        lines.append("|------------|-------|---------------------|")
        prior_base = float(prior_baseline or 0)
        for row in sorted(prior.get("ranked", []), key=lambda r: r.get("shots_per_class", 0)):
            lines.append(
                f"| {row.get('shots_per_class')}-shot | "
                f"{row.get('holdout_auc_roc', 0):.4f} | "
                f"{(row.get('holdout_auc_roc', 0) or 0) - prior_base:+.4f} |"
            )
    else:
        lines.append("Prior `shot_runs/leaderboard.json` not found.")

    lines.extend(
        [
            "",
            "### Interpretation",
            "",
            "- **Δ AUROC > 0.01:** meaningful head-only gain on this holdout",
            "- **Δ AUROC 0.001–0.01:** marginal (similar to 3/5-shot plateau)",
            "- **Δ AUROC ≈ 0 with zero flips:** head-only capacity exhausted at this support scale",
            "- **Train loss ↓ but holdout flat:** support overfitting",
            "",
            f"Full analysis JSON: `{runs_dir / 'analysis' / 'analysis_summary.json'}`",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated report with analysis: {report_path}")


def run_shot_analysis(
    runs_dir: Path,
    *,
    skip_symmetry: bool = True,
) -> Dict[str, Any]:
    runs_dir = runs_dir.resolve()
    leaderboard_path = runs_dir / "leaderboard.json"
    if not leaderboard_path.is_file():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_path}")

    payload = load_json(leaderboard_path)
    analysis_dir = runs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    baseline_results_path = runs_dir / "holdout_pkl" / "inference_baseline" / "inference_results.json"
    if not baseline_results_path.is_file():
        raise FileNotFoundError(f"Baseline inference not found: {baseline_results_path}")
    baseline_records = load_json(baseline_results_path)

    prediction_deltas: Dict[str, Any] = {}
    training_summary: List[Dict[str, Any]] = []

    ranked = payload.get("ranked", [])
    for row in ranked:
        shots = row.get("shots_per_class")
        if shots is None:
            continue
        shot_key = f"shot{shots}"
        post_results_path = runs_dir / shot_key / "inference" / "inference_results.json"
        if post_results_path.is_file():
            post_records = load_json(post_results_path)
            prediction_deltas[shot_key] = compute_prediction_deltas(baseline_records, post_records)

        train_row = load_training_summary(runs_dir, shots)
        if train_row:
            training_summary.append(train_row)

        run_per_shot_analysis(runs_dir, shots, analysis_dir, skip_symmetry=skip_symmetry)

    plot_auroc_scaling(payload, analysis_dir / "auroc_vs_shots.png")

    best = payload.get("best")
    best_shot = best.get("shots_per_class") if best else None
    if best_shot is not None:
        comparison_dir = analysis_dir / "comparison_best"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        run_baseline_vs_posttrain_comparison(
            baseline_inference_dir=str(runs_dir / "holdout_pkl" / "inference_baseline"),
            posttrain_inference_dir=str(runs_dir / f"shot{best_shot}" / "inference"),
            pkl_dir=str(runs_dir / "holdout_pkl"),
            table_dir=str(TABLE_DIR),
            scoli_root=str(SCOLI_ROOT),
            output_dir=str(comparison_dir),
            skip_symmetry=skip_symmetry,
            threshold=INFER_THRESHOLD,
        )

    summary = {
        "runs_dir": str(runs_dir),
        "experiment_id": payload.get("experiment_id"),
        "baseline_auc_roc": payload.get("baseline", {}).get("auc_roc"),
        "best_shot": best_shot,
        "prediction_deltas": prediction_deltas,
        "training_summary": training_summary,
        "prior_shot_runs": load_prior_shot_runs_summary(),
    }

    summary_path = analysis_dir / "analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved analysis summary: {summary_path}")

    append_analysis_to_report(runs_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze shot experiment results")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Experiment runs directory containing leaderboard.json",
    )
    parser.add_argument("--skip-symmetry", action="store_true", default=True)
    args = parser.parse_args()
    run_shot_analysis(args.runs_dir, skip_symmetry=args.skip_symmetry)


if __name__ == "__main__":
    main()
