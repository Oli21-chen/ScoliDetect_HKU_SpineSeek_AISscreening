# -*- coding: utf-8 -*-
"""
Few-shot post-training experiment orchestrator.

Trains head-only models at increasing shot counts using fixed best HP,
evaluates all on the same fixed holdout, and writes a comparison leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PYTORCH_ROOT = Path(__file__).resolve().parents[1]
CODE_VIDEO_ROOT = PYTORCH_ROOT.parent
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))

from post_training.config import (  # noqa: E402
    BASE_CHECKPOINT,
    BEST_HEAD_FINETUNE_HP,
    CSV_SUFFIX,
    HeadFinetuneConfig,
    INFER_THRESHOLD,
    KM_PROFILE,
    KM_TIMESTEPS,
    LABEL_EXCEL,
    LARGE_MAX_SHOTS_PER_CLASS,
    LARGE_SHOT_COUNTS,
    MAX_AGE,
    MAX_SHOTS_PER_CLASS,
    MIN_FILE_INDEX,
    SCOLI_ROOT,
    SHOT_COUNTS,
    SHOT_RUNS_DIR,
    STEP_02,
    STEP_03,
    TABLE_DIR,
    VIDEO_DIR,
    VIDEO_TIMESTEPS,
    python_cmd,
)
from post_training.head_finetune import run_support_sampling, train_head  # noqa: E402
from post_training.support_split import (  # noqa: E402
    flatten_shot_support,
    resolve_fixed_holdout,
    resolve_shot_support,
)


@dataclass(frozen=True)
class ShotRunsLayout:
    runs_dir: Path

    @property
    def holdout_pkl_dir(self) -> Path:
        return self.runs_dir / "holdout_pkl"

    @property
    def baseline_inf_dir(self) -> Path:
        return self.holdout_pkl_dir / "inference_baseline"

    @property
    def baseline_metrics_path(self) -> Path:
        return self.baseline_inf_dir / "metrics.json"

    @property
    def leaderboard_json(self) -> Path:
        return self.runs_dir / "leaderboard.json"

    @property
    def leaderboard_csv(self) -> Path:
        return self.runs_dir / "leaderboard.csv"

    @property
    def report_md(self) -> Path:
        return self.runs_dir / "SHOT_EXPERIMENT_REPORT.md"

    def shot_dir(self, shots_per_class: int) -> Path:
        return self.runs_dir / f"shot{shots_per_class}"


def run_cmd(cmd: List[str], *, env: Optional[dict] = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(CODE_VIDEO_ROOT), env=merged)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def infer_experiment_id(runs_dir: Path, shots: List[int]) -> str:
    name = runs_dir.name
    if name == "shot_runs_large":
        return "large_support_10_15"
    shot_str = "_".join(str(s) for s in sorted(shots))
    return f"few_shot_{shot_str}"


def best_finetune_config(*, verbose: bool = False) -> HeadFinetuneConfig:
    return HeadFinetuneConfig(
        epochs=BEST_HEAD_FINETUNE_HP["epochs"],
        lr=BEST_HEAD_FINETUNE_HP["lr"],
        batch_size=BEST_HEAD_FINETUNE_HP["batch_size"],
        weight_decay=BEST_HEAD_FINETUNE_HP["weight_decay"],
        km_gaussian_noise_std=BEST_HEAD_FINETUNE_HP["km_gaussian_noise_std"],
        loss_type=BEST_HEAD_FINETUNE_HP["loss_type"],
        verbose=verbose,
    )


def run_holdout_sampling(holdout_indices: List[int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = python_cmd(
        [
            str(STEP_02),
            "--table-path",
            str(TABLE_DIR),
            "--video-path",
            str(VIDEO_DIR),
            "--output-dir",
            str(output_dir),
            "--csv-suffix",
            CSV_SUFFIX,
            "--min-file-index",
            str(MIN_FILE_INDEX),
            "--km-timesteps",
            str(KM_TIMESTEPS),
            "--video-timesteps",
            str(VIDEO_TIMESTEPS),
            "--indices",
            *[str(i) for i in holdout_indices],
        ]
    )
    run_cmd(cmd, env={"SCOLI_KM_PROFILE": KM_PROFILE})


def run_holdout_inference(
    holdout_pkl_dir: Path,
    checkpoint: Optional[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = python_cmd(
        [
            str(STEP_03),
            "--pkl-dir",
            str(holdout_pkl_dir),
            "--output-dir",
            str(output_dir),
            "--scoli-root",
            str(SCOLI_ROOT),
            "--threshold",
            str(INFER_THRESHOLD),
            "--max-age",
            str(MAX_AGE),
            "--label-excel",
            str(LABEL_EXCEL),
        ]
    )
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])
    run_cmd(cmd)
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Inference metrics missing: {metrics_path}")
    return load_json(metrics_path)


def ensure_baseline(
    layout: ShotRunsLayout,
    holdout_indices: List[int],
    *,
    skip_sample: bool = False,
    skip_infer: bool = False,
) -> Dict[str, Any]:
    if not skip_sample or not layout.holdout_pkl_dir.joinpath("patch_metadata.pkl").is_file():
        print(f"\nSampling fixed holdout ({len(holdout_indices)} subjects)...")
        run_holdout_sampling(holdout_indices, layout.holdout_pkl_dir)
    elif skip_sample:
        print(f"Reusing holdout PKL at {layout.holdout_pkl_dir}")

    if skip_infer and layout.baseline_metrics_path.is_file():
        print(f"Reusing baseline metrics at {layout.baseline_metrics_path}")
        return load_json(layout.baseline_metrics_path)

    print("\nRunning baseline inference on fixed holdout...")
    return run_holdout_inference(
        layout.holdout_pkl_dir,
        checkpoint=None,
        output_dir=layout.baseline_inf_dir,
    )


def run_shot(
    shots_per_class: int,
    layout: ShotRunsLayout,
    *,
    skip_train: bool = False,
    skip_sample: bool = False,
    baseline_auc: float,
    max_shots_per_class: int = MAX_SHOTS_PER_CLASS,
) -> Dict[str, Any]:
    name = f"shot{shots_per_class}"
    run_dir = layout.shot_dir(shots_per_class)
    support_pkl_dir = run_dir / "support_pkl"
    checkpoint = run_dir / "checkpoint_posttrain_head.pth"
    manifest_path = run_dir / "manifest.json"
    inference_dir = run_dir / "inference"

    support = resolve_shot_support(
        shots_per_class,
        max_shots_per_class=max_shots_per_class,
        table_path=TABLE_DIR,
        video_path=VIDEO_DIR,
    )
    support_indices = flatten_shot_support(support)
    n_subjects = len(support_indices)

    result: Dict[str, Any] = {
        "name": name,
        "shots_per_class": shots_per_class,
        "support_subjects": support,
        "support_indices": support_indices,
        "n_support_subjects": n_subjects,
        "n_support_patches": n_subjects,
        "hyperparameters": dict(BEST_HEAD_FINETUNE_HP),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "status": "started",
    }

    try:
        if not skip_train:
            print(f"\n{'=' * 60}\n{name}: {shots_per_class}-shot ({n_subjects} subjects)\n{'=' * 60}")
            if not skip_sample or not support_pkl_dir.joinpath("patch_metadata.pkl").is_file():
                run_support_sampling(
                    support_indices,
                    support_pkl_dir,
                    table_dir=TABLE_DIR,
                    video_dir=VIDEO_DIR,
                )
            cfg = best_finetune_config(verbose=False)
            train_result = train_head(
                support_pkl_dir=support_pkl_dir,
                base_checkpoint=BASE_CHECKPOINT,
                output_checkpoint=checkpoint,
                cfg=cfg,
            )
            manifest = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "shots_per_class": shots_per_class,
                "support_subjects": support,
                "support_indices": support_indices,
                "n_support_subjects": n_subjects,
                "hyperparameters": dict(BEST_HEAD_FINETUNE_HP),
                "training": train_result,
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            result["training"] = {
                "best_epoch": train_result.get("best_epoch"),
                "best_loss": train_result.get("best_loss"),
                "label_counts": train_result.get("label_counts"),
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
        print(f"{name} failed: {exc}")

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


def write_leaderboard(
    layout: ShotRunsLayout,
    results: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    *,
    holdout_indices: List[int],
    max_pool: Dict[str, List[int]],
    max_shots_per_class: int,
    experiment_id: str,
) -> None:
    layout.runs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "few_shot_post_training",
        "experiment_id": experiment_id,
        "runs_dir": str(layout.runs_dir),
        "shots_evaluated": [r.get("shots_per_class") for r in results],
        "max_shots_per_class": max_shots_per_class,
        "max_support_pool": max_pool,
        "fixed_holdout_n_subjects": len(holdout_indices),
        "fixed_holdout_indices": holdout_indices,
        "hyperparameters": dict(BEST_HEAD_FINETUNE_HP),
        "baseline": baseline,
        "n_completed": sum(1 for r in results if r.get("status") == "completed"),
        "n_failed": sum(1 for r in results if r.get("status") == "failed"),
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
        "all_results": results,
    }
    with open(layout.leaderboard_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved leaderboard: {layout.leaderboard_json}")

    if ranked:
        fieldnames = [
            "rank",
            "shots_per_class",
            "name",
            "n_support_subjects",
            "n_support_patches",
            "holdout_auc_roc",
            "holdout_accuracy",
            "holdout_f1",
            "delta_auc_roc",
            "best_loss",
            "best_epoch",
            "support_pos",
            "support_neg",
        ]
        with open(layout.leaderboard_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in ranked:
                support = row.get("support_subjects", {})
                training = row.get("training", {})
                writer.writerow(
                    {
                        "rank": row.get("rank"),
                        "shots_per_class": row.get("shots_per_class"),
                        "name": row.get("name"),
                        "n_support_subjects": row.get("n_support_subjects"),
                        "n_support_patches": row.get("n_support_patches"),
                        "holdout_auc_roc": row.get("holdout_auc_roc"),
                        "holdout_accuracy": row.get("holdout_accuracy"),
                        "holdout_f1": row.get("holdout_f1"),
                        "delta_auc_roc": row.get("delta_auc_roc"),
                        "best_loss": training.get("best_loss"),
                        "best_epoch": training.get("best_epoch"),
                        "support_pos": ",".join(str(i) for i in support.get("pos", [])),
                        "support_neg": ",".join(str(i) for i in support.get("neg", [])),
                    }
                )
        print(f"Saved leaderboard CSV: {layout.leaderboard_csv}")


def _fmt_float(val: Any, digits: int = 4) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return str(val)


def write_report_md(layout: ShotRunsLayout, payload: Dict[str, Any]) -> Path:
    layout.runs_dir.mkdir(parents=True, exist_ok=True)
    baseline = payload.get("baseline", {})
    baseline_auc = float(baseline.get("auc_roc", 0))
    ranked: List[Dict[str, Any]] = payload.get("ranked", [])
    all_results: List[Dict[str, Any]] = payload.get("all_results", [])
    best = payload.get("best")
    max_pool = payload.get("max_support_pool", {})
    max_shots = payload.get("max_shots_per_class", MAX_SHOTS_PER_CLASS)
    hp = payload.get("hyperparameters", {})
    experiment_id = payload.get("experiment_id", "few_shot_post_training")

    lines: List[str] = [
        "# Few-Shot Post-Training Experiment Report",
        "",
        f"Generated: {payload.get('created_at', '')}",
        f"Experiment ID: `{experiment_id}`",
        f"Runs directory: `{layout.runs_dir}`",
        "",
        "## Executive Summary",
        "",
        f"- **Baseline holdout AUROC (fixed holdout):** {_fmt_float(baseline_auc)}",
        f"- **Fixed holdout subjects:** {payload.get('fixed_holdout_n_subjects', 'N/A')}",
        f"- **Hyperparameters:** lr={hp.get('lr')}, epochs={hp.get('epochs')}, "
        f"wd={hp.get('weight_decay')}, loss={hp.get('loss_type')}, "
        f"noise={hp.get('km_gaussian_noise_std')}",
        f"- **Completed:** {payload.get('n_completed', 0)} | **Failed:** {payload.get('n_failed', 0)}",
    ]

    if best:
        lines.extend(
            [
                f"- **Best shot count:** {best.get('shots_per_class')}-shot "
                f"(AUROC {_fmt_float(best.get('holdout_auc_roc'))}, "
                f"delta {_fmt_float(best.get('delta_auc_roc'), 4)})",
            ]
        )

    lines.extend(
        [
            "",
            "## Shot Comparison (primary result)",
            "",
            "| Shot | Support subjects | Holdout AUROC | Δ vs baseline | Accuracy | F1 |",
            "|------|-----------------|---------------|---------------|----------|-----|",
        ]
    )
    for row in sorted(all_results, key=lambda r: r.get("shots_per_class", 0)):
        support = row.get("support_subjects", {})
        pos_str = ",".join(str(i) for i in support.get("pos", []))
        neg_str = ",".join(str(i) for i in support.get("neg", []))
        lines.append(
            "| {shot} | pos=[{pos}] neg=[{neg}] | {auc} | {dauc} | {acc} | {f1} |".format(
                shot=row.get("shots_per_class"),
                pos=pos_str,
                neg=neg_str,
                auc=_fmt_float(row.get("holdout_auc_roc")),
                dauc=_fmt_float(row.get("delta_auc_roc"), 4),
                acc=_fmt_float(row.get("holdout_accuracy")),
                f1=_fmt_float(row.get("holdout_f1")),
            )
        )

    lines.extend(
        [
            "",
            "## Experimental Setup",
            "",
            "- **Shot definition:** K-shot = K positive + K negative subjects (nested)",
            f"- **Max support pool ({max_shots}-shot):** pos={max_pool.get('pos', [])}, neg={max_pool.get('neg', [])}",
            "- **Holdout protocol:** fixed — same holdout for all shot counts in this run",
            "- **No HP sweep** — all runs use best settings from `hp_runs/leaderboard.json`",
            "",
            "## Holdout comparability",
            "",
            "AUROC values are only directly comparable **within this runs directory** "
            f"(`{layout.runs_dir.name}/`). Different `max_shots_per_class` settings "
            "exclude different support pools and yield different holdout cohorts.",
            "",
            "## Artifacts",
            "",
            f"- Leaderboard JSON: `{layout.leaderboard_json}`",
            f"- Leaderboard CSV: `{layout.leaderboard_csv}`",
            f"- Per-shot dirs: `{layout.runs_dir}/shot{{K}}/`",
            f"- Shared holdout: `{layout.holdout_pkl_dir}`",
            f"- Analysis: `{layout.runs_dir}/analysis/`",
        ]
    )

    layout.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {layout.report_md}")
    return layout.report_md


def print_dry_run(
    shots: List[int],
    max_shots_per_class: int = MAX_SHOTS_PER_CLASS,
) -> None:
    max_pool = resolve_shot_support(max_shots_per_class, max_shots_per_class=max_shots_per_class)
    holdout = resolve_fixed_holdout(max_shots_per_class)
    print(f"Max support pool ({max_shots_per_class}-shot):")
    print(f"  pos ({len(max_pool['pos'])}): {max_pool['pos']}")
    print(f"  neg ({len(max_pool['neg'])}): {max_pool['neg']}")
    print(f"Fixed holdout: {len(holdout)} subjects")
    print(f"Hyperparameters: {BEST_HEAD_FINETUNE_HP}")
    print()
    for k in sorted(shots):
        support = resolve_shot_support(k, max_shots_per_class=max_shots_per_class)
        indices = flatten_shot_support(support)
        print(f"  {k}-shot: {len(indices)} subjects — pos={support['pos']}, neg={support['neg']}")


def print_summary(ranked: List[Dict[str, Any]], baseline_auc: float) -> None:
    print(f"\n{'=' * 60}\nFew-shot experiment summary\n{'=' * 60}")
    print(f"Baseline holdout AUROC: {baseline_auc:.4f}")
    if not ranked:
        print("No completed shot runs.")
        return
    print("\nResults by shot count:")
    for row in sorted(ranked, key=lambda r: r.get("shots_per_class", 0)):
        print(
            f"  {row.get('shots_per_class')}-shot: "
            f"AUROC={row.get('holdout_auc_roc'):.4f} "
            f"(delta {row.get('delta_auc_roc'):+.4f})"
        )
    best = ranked[0]
    print(
        f"\nBest: {best.get('shots_per_class')}-shot "
        f"(AUROC={best.get('holdout_auc_roc'):.4f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot post-training experiment")
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=SHOT_COUNTS,
        help=f"Shot counts to run (default: {SHOT_COUNTS})",
    )
    parser.add_argument("--max-shots-per-class", type=int, default=MAX_SHOTS_PER_CLASS)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=SHOT_RUNS_DIR,
        help=f"Experiment output root (default: {SHOT_RUNS_DIR})",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Identifier stored in leaderboard JSON (auto-inferred if omitted)",
    )
    parser.add_argument("--skip-train", action="store_true", help="Re-infer existing checkpoints only")
    parser.add_argument("--skip-sample", action="store_true", help="Reuse existing PKL dirs")
    parser.add_argument("--skip-baseline", action="store_true", help="Reuse existing baseline metrics")
    parser.add_argument("--dry-run", action="store_true", help="Print planned splits and exit")
    parser.add_argument(
        "--write-report-only",
        action="store_true",
        help="Regenerate SHOT_EXPERIMENT_REPORT.md from existing leaderboard.json",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run post-experiment analysis (requires completed leaderboard)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis on existing leaderboard (skip train/infer)",
    )
    args = parser.parse_args()

    layout = ShotRunsLayout(runs_dir=args.runs_dir.resolve())
    experiment_id = args.experiment_id or infer_experiment_id(layout.runs_dir, args.shots)

    if args.write_report_only:
        if not layout.leaderboard_json.is_file():
            raise SystemExit(f"Leaderboard not found: {layout.leaderboard_json}")
        write_report_md(layout, load_json(layout.leaderboard_json))
        return

    if args.dry_run:
        print_dry_run(args.shots, args.max_shots_per_class)
        return

    if args.analyze_only:
        if not layout.leaderboard_json.is_file():
            raise SystemExit(f"Leaderboard not found: {layout.leaderboard_json}")
        if not layout.report_md.is_file():
            write_report_md(layout, load_json(layout.leaderboard_json))
        from post_training.shot_analysis import run_shot_analysis  # noqa: E402

        run_shot_analysis(layout.runs_dir)
        return

    max_pool = resolve_shot_support(
        args.max_shots_per_class,
        max_shots_per_class=args.max_shots_per_class,
    )
    holdout_indices = resolve_fixed_holdout(args.max_shots_per_class)
    print(f"Runs dir: {layout.runs_dir}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Max support pool: pos={max_pool['pos']}, neg={max_pool['neg']}")
    print(f"Fixed holdout: {len(holdout_indices)} subjects")

    baseline = ensure_baseline(
        layout,
        holdout_indices,
        skip_sample=args.skip_sample,
        skip_infer=args.skip_baseline or args.skip_train,
    )
    baseline_auc = float(baseline["auc_roc"])
    print(f"Baseline holdout AUROC: {baseline_auc:.4f}")

    results: List[Dict[str, Any]] = []
    for shots in sorted(set(args.shots)):
        results.append(
            run_shot(
                shots,
                layout,
                skip_train=args.skip_train,
                skip_sample=args.skip_sample,
                baseline_auc=baseline_auc,
                max_shots_per_class=args.max_shots_per_class,
            )
        )

    ranked = rank_results(results)
    write_leaderboard(
        layout,
        results,
        ranked,
        baseline,
        holdout_indices=holdout_indices,
        max_pool=max_pool,
        max_shots_per_class=args.max_shots_per_class,
        experiment_id=experiment_id,
    )
    print_summary(ranked, baseline_auc)
    write_report_md(layout, load_json(layout.leaderboard_json))

    if args.analyze:
        from post_training.shot_analysis import run_shot_analysis  # noqa: E402

        run_shot_analysis(layout.runs_dir)


if __name__ == "__main__":
    main()
