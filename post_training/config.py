# -*- coding: utf-8 -*-
"""Default paths and hyperparameters for head-only post-training."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

DESKTOP = Path(r"C:\Users\Olive\Desktop")
CODE_VIDEO = DESKTOP / "Nature_Style" / "code_video"
EXPERIMENTS_ROOT = DESKTOP / "experiments"
SCOLI_ROOT = DESKTOP / "ScoliDetect_deployversion"

POST_TRAINING_ROOT = CODE_VIDEO / "pytorch" / "post_training"
SUPPORT_PKL_DIR = POST_TRAINING_ROOT / "support_pkl"
HOLDOUT_PKL_DIR = POST_TRAINING_ROOT / "holdout_pkl"
CHECKPOINT_DIR = POST_TRAINING_ROOT / "checkpoints"
COMPARISON_DIR = POST_TRAINING_ROOT / "comparison"
HP_RUNS_DIR = POST_TRAINING_ROOT / "hp_runs"
SHOT_RUNS_DIR = POST_TRAINING_ROOT / "shot_runs"

BASE_CHECKPOINT = (
    SCOLI_ROOT
    / "checkpoints"
    / "kfold5_kvt_vivit_pretrained_cobb11"
    / "fold_1"
    / "checkpoint_best.pth"
)
POSTTRAIN_CHECKPOINT = CHECKPOINT_DIR / "checkpoint_posttrain_head.pth"
POSTTRAIN_MANIFEST = POST_TRAINING_ROOT / "posttrain_manifest.json"

TABLE_DIR = EXPERIMENTS_ROOT / "table"
VIDEO_DIR = EXPERIMENTS_ROOT / "video"
if not VIDEO_DIR.is_dir():
    VIDEO_DIR = DESKTOP / "video"

LABEL_EXCEL = DESKTOP / "video_retrival" / "video_retrival" / "Label_SZpart2.xlsx"

PROMPTS_PATH = CODE_VIDEO / "pytorch" / "data" / "general_gait_prompts_from_report.json"
PROMPT_SELECTION = "concise_prompts"  # same as deploy checkpoint config

STEP_02 = CODE_VIDEO / "main" / "02_Sampling.py"
STEP_03 = CODE_VIDEO / "main" / "03_Infer.py"
STEP_04 = CODE_VIDEO / "main" / "04_results_analyze.py"

MIN_FILE_INDEX = 884
BINARY_THRESHOLD = 11.0
KM_PROFILE = "deploy_pose_size_no_zscore"
KM_TIMESTEPS = 96
VIDEO_TIMESTEPS = 32
CSV_SUFFIX = "_step_1.csv"

INFER_THRESHOLD = 0.5
MAX_AGE = 100.0

CONDA_ENV = "PytorchCuda11.8"
CONDA_EXE = Path(r"C:\Users\Olive\anaconda3\Scripts\conda.exe")
CONDA_PYTHON = Path(r"C:\Users\Olive\anaconda3\envs\PytorchCuda11.8\python.exe")


def python_cmd(script_args: List[str]) -> List[str]:
    if CONDA_PYTHON.is_file():
        return [str(CONDA_PYTHON), *script_args]
    if CONDA_EXE.is_file():
        return [str(CONDA_EXE), "run", "-n", CONDA_ENV, "python", *script_args]
    return [sys.executable, *script_args]


def resolve_prompts_path(ckpt_config: dict | None = None) -> Path:
    """Pick a local prompts JSON (deploy ckpt paths are often Linux-only)."""
    candidates: list[Path] = []
    if ckpt_config:
        raw = ckpt_config.get("prompts_path")
        if raw:
            candidates.append(Path(str(raw)))
    candidates.extend(
        [
            PROMPTS_PATH,
            SCOLI_ROOT / "checkpoints" / "general_gait_prompts_from_report.json",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Prompts JSON not found. Expected at "
        f"{PROMPTS_PATH} or deploy checkpoints copy."
    )


def resolve_prompt_selection(ckpt_config: dict | None = None) -> str:
    if ckpt_config and ckpt_config.get("prompt_selection"):
        return str(ckpt_config["prompt_selection"])
    return PROMPT_SELECTION


MAX_SHOTS_PER_CLASS = 5
SHOT_COUNTS = [1, 3, 5]

SHOT_RUNS_LARGE_DIR = POST_TRAINING_ROOT / "shot_runs_large"
LARGE_MAX_SHOTS_PER_CLASS = 15
LARGE_SHOT_COUNTS = [10, 15]

METHOD_RUNS_DIR = POST_TRAINING_ROOT / "method_runs"
METHOD_SUPPORT_PKL = SHOT_RUNS_LARGE_DIR / "shot15" / "support_pkl"
METHOD_HOLDOUT_PKL = SHOT_RUNS_LARGE_DIR / "holdout_pkl"
METHOD_BASELINE_INF = METHOD_HOLDOUT_PKL / "inference_baseline"

METHOD_SHOT_RUNS_DIR = SHOT_RUNS_DIR
METHOD_SHOT_HOLDOUT_PKL = SHOT_RUNS_DIR / "holdout_pkl"
METHOD_SHOT_BASELINE_INF = METHOD_SHOT_HOLDOUT_PKL / "inference_baseline"
METHOD_SHOT_COUNTS = [1, 5]

METHOD_LARGE_SHOT_COUNTS = [1, 5, 10, 15]
METHOD_LARGE_SHOT_RUNS_DIR = SHOT_RUNS_LARGE_DIR


def method_support_pkl(shots_per_class: int, *, shot_runs_dir: Path = SHOT_RUNS_DIR) -> Path:
    """Support PKL for K-shot partial-unfreeze (shot_runs/shot{K}/support_pkl)."""
    return shot_runs_dir / f"shot{shots_per_class}" / "support_pkl"


def method_support_pkl_large(shots_per_class: int) -> Path:
    """Support PKL on 15-max-pool cohort (shot_runs_large/shot{K}/support_pkl)."""
    return SHOT_RUNS_LARGE_DIR / f"shot{shots_per_class}" / "support_pkl"


def head_only_checkpoint_for_shot(shots_per_class: int) -> Path:
    """Head-only checkpoint path (1/5 in shot_runs; 10/15 in shot_runs_large)."""
    if shots_per_class <= MAX_SHOTS_PER_CLASS:
        return SHOT_RUNS_DIR / f"shot{shots_per_class}" / "checkpoint_posttrain_head.pth"
    return SHOT_RUNS_LARGE_DIR / f"shot{shots_per_class}" / "checkpoint_posttrain_head.pth"


def head_only_checkpoint_large(shots_per_class: int) -> Path:
    """Alias for unified large-cohort head-only checkpoint lookup."""
    return head_only_checkpoint_for_shot(shots_per_class)


def head_only_inference_metrics(shots_per_class: int, *, shot_runs_dir: Path = SHOT_RUNS_DIR) -> Path:
    return shot_runs_dir / f"shot{shots_per_class}" / "inference" / "metrics.json"


def unified_head_only_inference_metrics(
    shots_per_class: int, *, method_runs_dir: Path = METHOD_RUNS_DIR
) -> Path:
    return method_runs_dir / f"shot{shots_per_class}" / "head_only" / "inference" / "metrics.json"


def unified_partial_inference_metrics(
    shots_per_class: int, *, method_runs_dir: Path = METHOD_RUNS_DIR
) -> Path:
    return method_runs_dir / f"shot{shots_per_class}" / "partial_unfreeze" / "inference" / "metrics.json"

METHOD_CONFIGS = {
    "partial_unfreeze": {
        "freeze_policy": "partial_unfreeze",
        "epochs": 8,
        "encoder_lr": 1e-7,
        "head_lr": 1e-5,
        "lr": 1e-7,
        "batch_size": 1,
        "weight_decay": 1e-3,
        "km_gaussian_noise_std": 0.01,
        "loss_type": "bce",
        "unfreeze_n_blocks": 2,
    },
    "full_encoder_lowlr": {
        "freeze_policy": "full_encoder",
        "epochs": 5,
        "encoder_lr": 3e-8,
        "head_lr": 3e-8,
        "lr": 3e-8,
        "batch_size": 1,
        "weight_decay": 1e-3,
        "km_gaussian_noise_std": 0.01,
        "loss_type": "bce",
        "unfreeze_n_blocks": 0,
    },
}

# Winning HP from hp_runs/leaderboard.json rank 1 (lr1e-7_ep10_wd1e-3)
BEST_HEAD_FINETUNE_HP = {
    "epochs": 10,
    "lr": 1e-7,
    "batch_size": 1,
    "weight_decay": 1e-3,
    "km_gaussian_noise_std": 0.01,
    "loss_type": "bce",
}


@dataclass(frozen=True)
class HeadFinetuneConfig:
    epochs: int = BEST_HEAD_FINETUNE_HP["epochs"]
    lr: float = BEST_HEAD_FINETUNE_HP["lr"]
    batch_size: int = BEST_HEAD_FINETUNE_HP["batch_size"]
    weight_decay: float = BEST_HEAD_FINETUNE_HP["weight_decay"]
    km_gaussian_noise_std: float = BEST_HEAD_FINETUNE_HP["km_gaussian_noise_std"]
    loss_type: str = BEST_HEAD_FINETUNE_HP["loss_type"]
    num_workers: int = 0
    verbose: bool = True
