# -*- coding: utf-8 -*-
"""
Orchestrate holdout evaluation: 02 sampling -> dual 03 inference -> 04 analysis -> comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PYTORCH_ROOT = Path(__file__).resolve().parents[1]
CODE_VIDEO_ROOT = PYTORCH_ROOT.parent
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))

from post_training.config import (  # noqa: E402
    BASE_CHECKPOINT,
    COMPARISON_DIR,
    CSV_SUFFIX,
    HOLDOUT_PKL_DIR,
    INFER_THRESHOLD,
    KM_PROFILE,
    KM_TIMESTEPS,
    LABEL_EXCEL,
    MAX_AGE,
    MIN_FILE_INDEX,
    POSTTRAIN_CHECKPOINT,
    SCOLI_ROOT,
    STEP_02,
    STEP_03,
    STEP_04,
    SUPPORT_PKL_DIR,
    TABLE_DIR,
    VIDEO_DIR,
    VIDEO_TIMESTEPS,
    python_cmd,
)
from post_training.support_split import resolve_holdout_indices, resolve_support_indices  # noqa: E402

HEAD_FINETUNE = PYTORCH_ROOT / "post_training" / "head_finetune.py"


def run_cmd(cmd: List[str], *, env: Optional[dict] = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(CODE_VIDEO_ROOT), env=merged)


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


def run_inference(
    pkl_dir: Path,
    output_dir: Path,
    checkpoint: Optional[Path] = None,
) -> None:
    cmd = python_cmd(
        [
            str(STEP_03),
            "--pkl-dir",
            str(pkl_dir),
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


def run_analysis(
    results_json: Path,
    pkl_dir: Path,
    output_parent: Path,
    *,
    skip_symmetry: bool = False,
) -> None:
    cmd = python_cmd(
        [
            str(STEP_04),
            "--results-json",
            str(results_json),
            "--pkl-dir",
            str(pkl_dir),
            "--table-dir",
            str(TABLE_DIR),
            "--scoli-root",
            str(SCOLI_ROOT),
        ]
    )
    if skip_symmetry:
        cmd.append("--skip-symmetry")
    run_cmd(cmd)


def run_comparison(
    baseline_inference_dir: Path,
    posttrain_inference_dir: Path,
    pkl_dir: Path,
    output_dir: Path,
    *,
    skip_symmetry: bool = False,
) -> None:
    cmd = python_cmd(
        [
            str(STEP_04),
            "--compare-baseline-vs-posttrain",
            "--baseline-inference-dir",
            str(baseline_inference_dir),
            "--posttrain-inference-dir",
            str(posttrain_inference_dir),
            "--pkl-dir",
            str(pkl_dir),
            "--table-dir",
            str(TABLE_DIR),
            "--scoli-root",
            str(SCOLI_ROOT),
            "--compare-output-dir",
            str(output_dir),
        ]
    )
    if skip_symmetry:
        cmd.append("--skip-symmetry")
    run_cmd(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training holdout evaluation pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--finetune", action="store_true", help="Run head fine-tune only")
    parser.add_argument("--sample-holdout", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--skip-symmetry", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--holdout-pkl-dir", type=Path, default=HOLDOUT_PKL_DIR)
    parser.add_argument("--posttrain-checkpoint", type=Path, default=POSTTRAIN_CHECKPOINT)
    args = parser.parse_args()

    if not any(
        [
            args.all,
            args.finetune,
            args.sample_holdout,
            args.infer,
            args.analyze,
            args.compare,
        ]
    ):
        args.all = True

    support = resolve_support_indices(TABLE_DIR, VIDEO_DIR)
    holdout_indices = resolve_holdout_indices(support)
    print(f"Support: {support}")
    print(f"Holdout subjects: {len(holdout_indices)}")

    baseline_inf = args.holdout_pkl_dir / "inference_baseline"
    posttrain_inf = args.holdout_pkl_dir / "inference_posttrain"

    if args.all or args.finetune:
        if not args.skip_finetune:
            finetune_cmd = python_cmd([str(HEAD_FINETUNE)])
            if (SUPPORT_PKL_DIR / "patch_metadata.pkl").is_file():
                finetune_cmd.append("--skip-sample")
            run_cmd(finetune_cmd, env={"SCOLI_KM_PROFILE": KM_PROFILE})
        elif args.finetune:
            print("Skipping fine-tune (--skip-finetune)")

    if args.all or args.sample_holdout:
        run_holdout_sampling(holdout_indices, args.holdout_pkl_dir)

    if args.all or args.infer:
        run_inference(args.holdout_pkl_dir, baseline_inf, checkpoint=None)
        if not args.posttrain_checkpoint.is_file():
            raise FileNotFoundError(f"Post-trained checkpoint not found: {args.posttrain_checkpoint}")
        run_inference(
            args.holdout_pkl_dir,
            posttrain_inf,
            checkpoint=args.posttrain_checkpoint,
        )

    if args.all or args.analyze:
        baseline_results = baseline_inf / "inference_results.json"
        posttrain_results = posttrain_inf / "inference_results.json"
        if baseline_results.is_file():
            run_analysis(
                baseline_results,
                args.holdout_pkl_dir,
                baseline_inf,
                skip_symmetry=args.skip_symmetry,
            )
        if posttrain_results.is_file():
            run_analysis(
                posttrain_results,
                args.holdout_pkl_dir,
                posttrain_inf,
                skip_symmetry=args.skip_symmetry,
            )

    if args.all or args.compare:
        run_comparison(
            baseline_inf,
            posttrain_inf,
            args.holdout_pkl_dir,
            COMPARISON_DIR,
            skip_symmetry=args.skip_symmetry,
        )

    summary_path = COMPARISON_DIR / "pipeline_summary.json"
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "support": support,
        "holdout_n_subjects": len(holdout_indices),
        "holdout_pkl_dir": str(args.holdout_pkl_dir),
        "baseline_inference_dir": str(baseline_inf),
        "posttrain_inference_dir": str(posttrain_inf),
        "posttrain_checkpoint": str(args.posttrain_checkpoint),
        "base_checkpoint": str(BASE_CHECKPOINT),
    }
    for name, inf_dir in (("baseline", baseline_inf), ("posttrain", posttrain_inf)):
        metrics_path = inf_dir / "metrics.json"
        if metrics_path.is_file():
            summary[f"{name}_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Pipeline summary: {summary_path}")


if __name__ == "__main__":
    main()
