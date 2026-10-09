# -*- coding: utf-8 -*-
"""
Encoder-level post-training method comparison (partial unfreeze vs full encoder FT).

Supports:
- Large 15-shot cohort (shot_runs_large holdout) — default flat method_runs/{method}/
- Small 1/5-shot cohort (shot_runs holdout) — method_runs/shot{K}/{method}/
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PYTORCH_ROOT = Path(__file__).resolve().parents[1]
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))

from post_training.config import (  # noqa: E402
    BASE_CHECKPOINT,
    HeadFinetuneConfig,
    LARGE_MAX_SHOTS_PER_CLASS,
    METHOD_BASELINE_INF,
    METHOD_CONFIGS,
    METHOD_HOLDOUT_PKL,
    METHOD_LARGE_SHOT_COUNTS,
    METHOD_LARGE_SHOT_RUNS_DIR,
    METHOD_RUNS_DIR,
    METHOD_SHOT_BASELINE_INF,
    METHOD_SHOT_COUNTS,
    METHOD_SHOT_HOLDOUT_PKL,
    METHOD_SHOT_RUNS_DIR,
    METHOD_SUPPORT_PKL,
    TABLE_DIR,
    VIDEO_DIR,
    head_only_checkpoint_for_shot,
    method_support_pkl,
    method_support_pkl_large,
)
from post_training.head_finetune import run_support_sampling, train_posttrain  # noqa: E402
from post_training.shot_experiment import load_json, run_holdout_inference  # noqa: E402
from post_training.support_split import flatten_shot_support, resolve_shot_support  # noqa: E402


@dataclass
class MethodRunsLayout:
    runs_dir: Path
    holdout_pkl_override: Optional[Path] = None
    baseline_inf_override: Optional[Path] = None
    shot_mode: bool = False
    large_shot_mode: bool = False

    @property
    def holdout_pkl_dir(self) -> Path:
        return self.holdout_pkl_override or METHOD_HOLDOUT_PKL

    @property
    def baseline_inf_dir(self) -> Path:
        return self.baseline_inf_override or METHOD_BASELINE_INF

    @property
    def baseline_metrics_path(self) -> Path:
        return self.baseline_inf_dir / "metrics.json"

    @property
    def leaderboard_json(self) -> Path:
        if self.large_shot_mode:
            return self.runs_dir / "unified_shot_scaling_leaderboard.json"
        if self.shot_mode:
            return self.runs_dir / "shot_method_leaderboard.json"
        return self.runs_dir / "leaderboard.json"

    @property
    def leaderboard_csv(self) -> Path:
        if self.large_shot_mode:
            return self.runs_dir / "unified_shot_scaling_leaderboard.csv"
        if self.shot_mode:
            return self.runs_dir / "shot_method_leaderboard.csv"
        return self.runs_dir / "leaderboard.csv"

    @property
    def report_md(self) -> Path:
        return self.runs_dir / "METHOD_EXPERIMENT_REPORT.md"

    def head_only_dir(self, shots_per_class: int) -> Path:
        return self.runs_dir / f"shot{shots_per_class}" / "head_only"

    def method_dir(self, method_name: str, shots_per_class: Optional[int] = None) -> Path:
        if shots_per_class is not None:
            return self.runs_dir / f"shot{shots_per_class}" / method_name
        return self.runs_dir / method_name


def ensure_large_support_pkl(shots_k: int) -> Path:
    support_dir = method_support_pkl_large(shots_k)
    if support_dir.joinpath("patch_metadata.pkl").is_file():
        return support_dir
    support = resolve_shot_support(
        shots_k,
        max_shots_per_class=LARGE_MAX_SHOTS_PER_CLASS,
    )
    indices = flatten_shot_support(support)
    print(f"Sampling {shots_k}-shot support ({len(indices)} subjects) -> {support_dir}")
    run_support_sampling(
        indices,
        support_dir,
        table_dir=TABLE_DIR,
        video_dir=VIDEO_DIR,
    )
    return support_dir


def ensure_shot15_partial_checkpoint(layout: MethodRunsLayout) -> Path:
    """Reuse existing 15-shot partial_unfreeze checkpoint under shot15/ layout."""
    dest = layout.method_dir("partial_unfreeze", 15) / "checkpoint_posttrain.pth"
    if dest.is_file():
        return dest
    legacy = layout.runs_dir / "partial_unfreeze" / "checkpoint_posttrain.pth"
    if legacy.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, dest)
        print(f"Copied 15-shot checkpoint: {legacy} -> {dest}")
        return dest
    return dest


def reinfer_head_only_on_holdout(
    shots: List[int],
    layout: MethodRunsLayout,
) -> None:
    for shots_k in sorted(set(shots)):
        checkpoint = head_only_checkpoint_for_shot(shots_k)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Head-only checkpoint not found: {checkpoint}")
        inference_dir = layout.head_only_dir(shots_k) / "inference"
        print(f"\nRe-infer head-only {shots_k}-shot on unified holdout...")
        run_holdout_inference(layout.holdout_pkl_dir, checkpoint, inference_dir)


def method_finetune_config(method_name: str, *, verbose: bool = False) -> HeadFinetuneConfig:
    if method_name not in METHOD_CONFIGS:
        raise KeyError(f"Unknown method: {method_name}. Available: {list(METHOD_CONFIGS)}")
    hp = METHOD_CONFIGS[method_name]
    return HeadFinetuneConfig(
        epochs=hp["epochs"],
        lr=hp["lr"],
        batch_size=hp["batch_size"],
        weight_decay=hp["weight_decay"],
        km_gaussian_noise_std=hp["km_gaussian_noise_std"],
        loss_type=hp["loss_type"],
        verbose=verbose,
    )


def load_baseline_metrics(layout: MethodRunsLayout) -> Dict[str, Any]:
    if not layout.baseline_metrics_path.is_file():
        raise FileNotFoundError(
            f"Baseline metrics not found: {layout.baseline_metrics_path}. "
            "Run baseline inference on the holdout first."
        )
    return load_json(layout.baseline_metrics_path)


def run_method(
    method_name: str,
    layout: MethodRunsLayout,
    *,
    support_pkl_dir: Path,
    skip_train: bool = False,
    baseline_auc: float,
    shots_per_class: Optional[int] = None,
) -> Dict[str, Any]:
    if method_name not in METHOD_CONFIGS:
        raise ValueError(f"Unknown method: {method_name}")

    hp = METHOD_CONFIGS[method_name]
    run_dir = layout.method_dir(method_name, shots_per_class=shots_per_class)
    checkpoint = run_dir / "checkpoint_posttrain.pth"
    manifest_path = run_dir / "manifest.json"
    inference_dir = run_dir / "inference"

    result: Dict[str, Any] = {
        "name": method_name,
        "shots_per_class": shots_per_class,
        "freeze_policy": hp["freeze_policy"],
        "hyperparameters": dict(hp),
        "support_pkl_dir": str(support_pkl_dir),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "status": "started",
    }

    try:
        if not skip_train:
            label = f"{shots_per_class}-shot " if shots_per_class else ""
            print(f"\n{'=' * 60}\n{label}Method: {method_name}\n{'=' * 60}")
            if not support_pkl_dir.joinpath("patch_metadata.pkl").is_file():
                raise FileNotFoundError(f"Support PKL not found: {support_pkl_dir}")

            cfg = method_finetune_config(method_name, verbose=False)
            train_result = train_posttrain(
                support_pkl_dir=support_pkl_dir,
                base_checkpoint=BASE_CHECKPOINT,
                output_checkpoint=checkpoint,
                cfg=cfg,
                freeze_policy=hp["freeze_policy"],
                encoder_lr=hp.get("encoder_lr"),
                head_lr=hp.get("head_lr"),
                unfreeze_n_blocks=hp.get("unfreeze_n_blocks", 2),
            )
            manifest = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "method": method_name,
                "shots_per_class": shots_per_class,
                "freeze_policy": hp["freeze_policy"],
                "support_pkl_dir": str(support_pkl_dir),
                "hyperparameters": dict(hp),
                "training": train_result,
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            result["training"] = {
                "best_epoch": train_result.get("best_epoch"),
                "best_loss": train_result.get("best_loss"),
                "label_counts": train_result.get("label_counts"),
                "freeze_stats": train_result.get("freeze_stats"),
            }
        elif not checkpoint.is_file():
            result["status"] = "skipped"
            result["error"] = f"Checkpoint not found: {checkpoint}"
            return result

        holdout_metrics = run_holdout_inference(
            layout.holdout_pkl_dir, checkpoint, inference_dir
        )
        result["holdout_metrics"] = holdout_metrics
        result["holdout_auc_roc"] = holdout_metrics.get("auc_roc")
        result["holdout_accuracy"] = holdout_metrics.get("accuracy")
        result["holdout_f1"] = holdout_metrics.get("f1")
        result["baseline_auc_roc"] = baseline_auc
        result["delta_auc_roc"] = (
            float(holdout_metrics["auc_roc"]) - baseline_auc
            if holdout_metrics.get("auc_roc") is not None
            else None
        )
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        print(f"{method_name} failed: {exc}")

    return result


def rank_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    completed = [r for r in results if r.get("status") == "completed"]

    def sort_key(r: Dict[str, Any]):
        auc = r.get("holdout_auc_roc") or -1.0
        shots = r.get("shots_per_class") or 0
        return (-auc, -shots)

    ranked = sorted(completed, key=sort_key)
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def _fmt_float(val: Any, digits: int = 4) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return str(val)


def write_leaderboard(
    layout: MethodRunsLayout,
    results: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    *,
    methods: List[str],
    shots: Optional[List[int]] = None,
    experiment_id: str = "partial_unfreeze_vs_full_encoder_lowlr",
) -> None:
    layout.runs_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "encoder_method_comparison",
        "experiment_id": experiment_id,
        "runs_dir": str(layout.runs_dir),
        "methods_evaluated": methods,
        "shots_evaluated": shots,
        "holdout_pkl_dir": str(layout.holdout_pkl_dir),
        "baseline": baseline,
        "n_completed": sum(1 for r in results if r.get("status") == "completed"),
        "n_failed": sum(1 for r in results if r.get("status") == "failed"),
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
        "all_results": results,
    }
    if shots:
        if layout.large_shot_mode:
            payload["experiment"] = "unified_shot_scaling"
            payload["shot_runs_dir"] = str(METHOD_LARGE_SHOT_RUNS_DIR)
            payload["note"] = (
                "All shots evaluated on shot_runs_large holdout; "
                "supersedes cross-cohort numbers in SHOT_METHOD_COMPARISON_REPORT.md"
            )
        else:
            payload["experiment"] = "shot_encoder_method_comparison"
            payload["shot_runs_dir"] = str(METHOD_SHOT_RUNS_DIR)
    else:
        payload["support_pkl_dir"] = str(METHOD_SUPPORT_PKL)
        payload["head_only_reference"] = {
            "source": "shot_runs_large/shot15",
            "holdout_auc_roc": 0.7466,
            "delta_vs_baseline": -0.0010,
            "note": "15-shot head-only; same holdout cohort",
        }

    with open(layout.leaderboard_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved leaderboard: {layout.leaderboard_json}")

    if ranked:
        fieldnames = [
            "rank",
            "shots_per_class",
            "name",
            "freeze_policy",
            "holdout_auc_roc",
            "holdout_accuracy",
            "holdout_f1",
            "delta_auc_roc",
            "encoder_lr",
            "head_lr",
            "epochs",
            "best_loss",
            "best_epoch",
        ]
        with open(layout.leaderboard_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in ranked:
                hp = row.get("hyperparameters", {})
                training = row.get("training", {})
                writer.writerow(
                    {
                        "rank": row.get("rank"),
                        "shots_per_class": row.get("shots_per_class"),
                        "name": row.get("name"),
                        "freeze_policy": row.get("freeze_policy"),
                        "holdout_auc_roc": row.get("holdout_auc_roc"),
                        "holdout_accuracy": row.get("holdout_accuracy"),
                        "holdout_f1": row.get("holdout_f1"),
                        "delta_auc_roc": row.get("delta_auc_roc"),
                        "encoder_lr": hp.get("encoder_lr"),
                        "head_lr": hp.get("head_lr"),
                        "epochs": hp.get("epochs"),
                        "best_loss": training.get("best_loss"),
                        "best_epoch": training.get("best_epoch"),
                    }
                )
        print(f"Saved leaderboard CSV: {layout.leaderboard_csv}")


def write_report_md(layout: MethodRunsLayout, payload: Dict[str, Any]) -> Path:
    layout.runs_dir.mkdir(parents=True, exist_ok=True)
    baseline = payload.get("baseline", {})
    baseline_auc = float(baseline.get("auc_roc", 0))
    all_results: List[Dict[str, Any]] = payload.get("all_results", [])
    best = payload.get("best")
    head_ref = payload.get("head_only_reference", {})
    shots = payload.get("shots_evaluated")

    lines: List[str] = [
        "# Encoder Method Experiment Report",
        "",
        f"Generated: {payload.get('created_at', '')}",
        f"Runs directory: `{layout.runs_dir}`",
        "",
        "## Executive Summary",
        "",
        f"- **Baseline holdout AUROC:** {_fmt_float(baseline_auc)}",
        f"- **Holdout:** `{layout.holdout_pkl_dir}`",
        f"- **Completed:** {payload.get('n_completed', 0)} | **Failed:** {payload.get('n_failed', 0)}",
    ]

    if shots:
        lines.append(f"- **Shots evaluated:** {shots} (shot_runs cohort)")
    else:
        lines.extend(
            [
                f"- **Support:** 15-shot (30 subjects) — `{METHOD_SUPPORT_PKL}`",
                f"- **Head-only reference (15-shot):** AUROC {_fmt_float(head_ref.get('holdout_auc_roc'))} "
                f"(Δ {_fmt_float(head_ref.get('delta_vs_baseline'), 4)})",
            ]
        )

    if best:
        shot_label = f"{best.get('shots_per_class')}-shot " if best.get("shots_per_class") else ""
        lines.append(
            f"- **Best:** {shot_label}`{best.get('name')}` "
            f"(AUROC {_fmt_float(best.get('holdout_auc_roc'))}, "
            f"Δ {_fmt_float(best.get('delta_auc_roc'), 4)})"
        )

    lines.extend(
        [
            "",
            "## Method Comparison (primary result)",
            "",
            "| Shot | Method | Freeze policy | Holdout AUROC | Δ vs baseline | Accuracy | F1 |"
            if shots
            else "| Method | Freeze policy | Holdout AUROC | Δ vs baseline | Accuracy | F1 |",
            "|------|--------|---------------|---------------|---------------|----------|-----|"
            if shots
            else "|--------|---------------|---------------|---------------|----------|-----|",
        ]
    )
    for row in all_results:
        hp = row.get("hyperparameters", {})
        if shots:
            lines.append(
                "| {shot} | {name} | {policy} | {auc} | {dauc} | {acc} | {f1} |".format(
                    shot=row.get("shots_per_class", ""),
                    name=row.get("name"),
                    policy=row.get("freeze_policy"),
                    auc=_fmt_float(row.get("holdout_auc_roc")),
                    dauc=_fmt_float(row.get("delta_auc_roc"), 4),
                    acc=_fmt_float(row.get("holdout_accuracy")),
                    f1=_fmt_float(row.get("holdout_f1")),
                )
            )
        else:
            lines.append(
                "| {name} | {policy} | {auc} | {dauc} | {acc} | {f1} |".format(
                    name=row.get("name"),
                    policy=row.get("freeze_policy"),
                    auc=_fmt_float(row.get("holdout_auc_roc")),
                    dauc=_fmt_float(row.get("delta_auc_roc"), 4),
                    acc=_fmt_float(row.get("holdout_accuracy")),
                    f1=_fmt_float(row.get("holdout_f1")),
                )
            )
        enc_lr = hp.get("encoder_lr")
        hd_lr = hp.get("head_lr")
        if enc_lr is not None:
            lines.append(
                f"  - HP: encoder_lr={enc_lr}, head_lr={hd_lr}, "
                f"epochs={hp.get('epochs')}, wd={hp.get('weight_decay')}"
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Leaderboard: `{layout.leaderboard_json}`",
            f"- Per-method dirs: `{layout.runs_dir}/`",
            f"- Analysis: `{layout.runs_dir}/analysis/`",
        ]
    )

    layout.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {layout.report_md}")
    return layout.report_md


def print_summary(ranked: List[Dict[str, Any]], baseline_auc: float) -> None:
    print(f"\n{'=' * 60}\nMethod experiment summary\n{'=' * 60}")
    print(f"Baseline holdout AUROC: {baseline_auc:.4f}")
    if not ranked:
        print("No completed method runs.")
        return
    for row in ranked:
        shot_label = f"{row.get('shots_per_class')}-shot " if row.get("shots_per_class") else ""
        print(
            f"  {shot_label}{row.get('name')}: "
            f"AUROC={row.get('holdout_auc_roc'):.4f} "
            f"(delta {row.get('delta_auc_roc'):+.4f})"
        )
    best = ranked[0]
    shot_label = f"{best.get('shots_per_class')}-shot " if best.get("shots_per_class") else ""
    print(f"\nBest: {shot_label}{best.get('name')} (AUROC={best.get('holdout_auc_roc'):.4f})")


def run_shot_methods(
    shots: List[int],
    methods: List[str],
    layout: MethodRunsLayout,
    *,
    skip_train: bool = False,
) -> List[Dict[str, Any]]:
    baseline = load_baseline_metrics(layout)
    baseline_auc = float(baseline["auc_roc"])
    print(f"Baseline holdout AUROC: {baseline_auc:.4f}")

    results: List[Dict[str, Any]] = []
    for shots_k in sorted(set(shots)):
        support_pkl = method_support_pkl(shots_k)
        print(f"\n--- {shots_k}-shot support: {support_pkl} ---")
        for method_name in methods:
            results.append(
                run_method(
                    method_name,
                    layout,
                    support_pkl_dir=support_pkl,
                    skip_train=skip_train,
                    baseline_auc=baseline_auc,
                    shots_per_class=shots_k,
                )
            )
    return results


def run_large_shot_methods(
    shots: List[int],
    methods: List[str],
    layout: MethodRunsLayout,
    *,
    skip_train: bool = False,
    skip_head_reinfer: bool = False,
) -> List[Dict[str, Any]]:
    baseline = load_baseline_metrics(layout)
    baseline_auc = float(baseline["auc_roc"])
    print(f"Baseline holdout AUROC: {baseline_auc:.4f}")

    results: List[Dict[str, Any]] = []
    for shots_k in sorted(set(shots)):
        support_pkl = ensure_large_support_pkl(shots_k)
        print(f"\n--- {shots_k}-shot support (large cohort): {support_pkl} ---")
        for method_name in methods:
            do_skip_train = skip_train
            if shots_k == 15 and not skip_train:
                ckpt = ensure_shot15_partial_checkpoint(layout)
                if ckpt.is_file():
                    do_skip_train = True
            results.append(
                run_method(
                    method_name,
                    layout,
                    support_pkl_dir=support_pkl,
                    skip_train=do_skip_train,
                    baseline_auc=baseline_auc,
                    shots_per_class=shots_k,
                )
            )

    if not skip_head_reinfer:
        reinfer_head_only_on_holdout(shots, layout)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Encoder-level method comparison experiment")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_CONFIGS.keys()),
        help=f"Methods to run (default: {list(METHOD_CONFIGS.keys())})",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=METHOD_RUNS_DIR,
        help=f"Output root (default: {METHOD_RUNS_DIR})",
    )
    parser.add_argument(
        "--support-pkl-dir",
        type=Path,
        default=None,
        help="Support PKL (default: METHOD_SUPPORT_PKL or per-shot when --shots set)",
    )
    parser.add_argument(
        "--holdout-pkl-dir",
        type=Path,
        default=None,
        help="Holdout PKL (default: shot_runs_large or shot_runs with --shot-runs)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=None,
        help=f"Shot counts for small-support mode (e.g. 1 5; default: {METHOD_SHOT_COUNTS})",
    )
    parser.add_argument(
        "--shot-runs",
        action="store_true",
        help="Use shot_runs/ holdout and per-shot support PKLs (required with --shots)",
    )
    parser.add_argument(
        "--large-shot-runs",
        action="store_true",
        help="Unified 1/5/10/15-shot on shot_runs_large holdout (same holdout for all K)",
    )
    parser.add_argument(
        "--skip-head-reinfer",
        action="store_true",
        help="Skip head-only re-inference on unified holdout (large-shot mode)",
    )
    parser.add_argument(
        "--unified-compare-only",
        action="store_true",
        help="Only run unified shot scaling comparison report (no train/infer)",
    )
    parser.add_argument("--skip-train", action="store_true", help="Re-infer existing checkpoints only")
    parser.add_argument("--analyze", action="store_true", help="Run post-experiment analysis")
    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis")
    parser.add_argument(
        "--shot-compare-only",
        action="store_true",
        help="Only run 1/5-shot 3-way comparison report (no train/infer)",
    )
    parser.add_argument(
        "--write-report-only",
        action="store_true",
        help="Regenerate METHOD_EXPERIMENT_REPORT.md from leaderboard.json",
    )
    args = parser.parse_args()

    if args.large_shot_runs and args.shot_runs:
        raise SystemExit("Cannot combine --large-shot-runs with --shot-runs")

    large_shot_mode = args.large_shot_runs
    shot_mode = args.shot_runs or large_shot_mode or (args.shots is not None and not large_shot_mode)

    if large_shot_mode:
        shots = sorted(set(args.shots or METHOD_LARGE_SHOT_COUNTS))
        holdout_pkl = (
            args.holdout_pkl_dir.resolve() if args.holdout_pkl_dir else METHOD_HOLDOUT_PKL
        )
        baseline_inf = METHOD_BASELINE_INF
    elif shot_mode:
        shots = sorted(set(args.shots or METHOD_SHOT_COUNTS))
        holdout_pkl = (
            args.holdout_pkl_dir.resolve()
            if args.holdout_pkl_dir
            else METHOD_SHOT_HOLDOUT_PKL
        )
        baseline_inf = METHOD_SHOT_BASELINE_INF
    else:
        shots = None
        holdout_pkl = args.holdout_pkl_dir.resolve() if args.holdout_pkl_dir else METHOD_HOLDOUT_PKL
        baseline_inf = METHOD_BASELINE_INF

    layout = MethodRunsLayout(
        runs_dir=args.runs_dir.resolve(),
        holdout_pkl_override=holdout_pkl,
        baseline_inf_override=baseline_inf,
        shot_mode=shot_mode,
        large_shot_mode=large_shot_mode,
    )

    if args.unified_compare_only:
        from post_training.method_analysis import run_unified_shot_scaling_comparison  # noqa: E402

        run_unified_shot_scaling_comparison(layout.runs_dir, shots=shots)
        return

    if args.shot_compare_only:
        from post_training.method_analysis import run_shot_method_comparison  # noqa: E402

        run_shot_method_comparison(layout.runs_dir, shots=shots)
        return

    if args.analyze_only:
        if large_shot_mode:
            from post_training.method_analysis import run_unified_shot_scaling_comparison  # noqa: E402

            run_unified_shot_scaling_comparison(layout.runs_dir, shots=shots)
        elif shot_mode:
            from post_training.method_analysis import run_shot_method_comparison  # noqa: E402

            run_shot_method_comparison(layout.runs_dir, shots=shots)
        else:
            if not layout.leaderboard_json.is_file():
                raise SystemExit(f"Leaderboard not found: {layout.leaderboard_json}")
            from post_training.method_analysis import run_method_analysis  # noqa: E402

            run_method_analysis(layout.runs_dir)
        return

    if args.write_report_only:
        lb = layout.leaderboard_json
        if not lb.is_file():
            lb = layout.runs_dir / "leaderboard.json"
        if not lb.is_file():
            raise SystemExit(f"Leaderboard not found: {lb}")
        write_report_md(layout, load_json(lb))
        return

    methods = [m for m in args.methods if m in METHOD_CONFIGS]
    if not methods:
        raise SystemExit(f"No valid methods in {args.methods}")

    print(f"Runs dir: {layout.runs_dir}")
    print(f"Holdout PKL: {layout.holdout_pkl_dir}")

    if large_shot_mode:
        results = run_large_shot_methods(
            shots,
            methods,
            layout,
            skip_train=args.skip_train,
            skip_head_reinfer=args.skip_head_reinfer,
        )
        baseline = load_baseline_metrics(layout)
        ranked = rank_results(results)
        write_leaderboard(
            layout,
            results,
            ranked,
            baseline,
            methods=methods,
            shots=shots,
            experiment_id="unified_shot_scaling_1_5_10_15",
        )
        print_summary(ranked, float(baseline["auc_roc"]))
        if args.analyze:
            from post_training.method_analysis import run_unified_shot_scaling_comparison  # noqa: E402

            run_unified_shot_scaling_comparison(layout.runs_dir, shots=shots)
        return

    if shot_mode:
        results = run_shot_methods(shots, methods, layout, skip_train=args.skip_train)
        baseline = load_baseline_metrics(layout)
        ranked = rank_results(results)
        write_leaderboard(
            layout,
            results,
            ranked,
            baseline,
            methods=methods,
            shots=shots,
            experiment_id="partial_unfreeze_shot_1_5",
        )
        print_summary(ranked, float(baseline["auc_roc"]))
        if args.analyze:
            from post_training.method_analysis import run_shot_method_comparison  # noqa: E402

            run_shot_method_comparison(layout.runs_dir, shots=shots)
        return

    support_pkl = args.support_pkl_dir.resolve() if args.support_pkl_dir else METHOD_SUPPORT_PKL
    print(f"Support PKL: {support_pkl}")

    baseline = load_baseline_metrics(layout)
    baseline_auc = float(baseline["auc_roc"])
    print(f"Baseline holdout AUROC: {baseline_auc:.4f}")

    results: List[Dict[str, Any]] = []
    for method_name in methods:
        results.append(
            run_method(
                method_name,
                layout,
                support_pkl_dir=support_pkl,
                skip_train=args.skip_train,
                baseline_auc=baseline_auc,
            )
        )

    ranked = rank_results(results)
    write_leaderboard(layout, results, ranked, baseline, methods=methods)
    print_summary(ranked, baseline_auc)
    write_report_md(layout, load_json(layout.leaderboard_json))

    if args.analyze:
        from post_training.method_analysis import run_method_analysis  # noqa: E402

        run_method_analysis(layout.runs_dir)


if __name__ == "__main__":
    main()
