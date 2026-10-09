# -*- coding: utf-8 -*-
"""
Head-only post-training on support subjects (first pos + first neg).

Freezes video/km/text encoders; trains fusion modules + regressor.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

PYTORCH_ROOT = Path(__file__).resolve().parents[1]
CODE_VIDEO_ROOT = PYTORCH_ROOT.parent
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))

from post_training.config import (  # noqa: E402
    BASE_CHECKPOINT,
    CHECKPOINT_DIR,
    CSV_SUFFIX,
    HeadFinetuneConfig,
    KM_PROFILE,
    KM_TIMESTEPS,
    LABEL_EXCEL,
    MIN_FILE_INDEX,
    POSTTRAIN_CHECKPOINT,
    POSTTRAIN_MANIFEST,
    STEP_02,
    SUPPORT_PKL_DIR,
    TABLE_DIR,
    VIDEO_DIR,
    VIDEO_TIMESTEPS,
    python_cmd,
    resolve_prompt_selection,
    resolve_prompts_path,
)
from post_training.freeze import (  # noqa: E402
    FreezePolicy,
    apply_freeze_policy,
    build_param_groups,
    count_trainable,
)
from post_training.model_io import load_deploy_checkpoint_model  # noqa: E402
from post_training.support_split import (  # noqa: E402
    flatten_shot_support,
    resolve_shot_support,
)
from utils.data_sampler import SigLIPFullGaitDatasetPKL, fullgait_collate_fn  # noqa: E402
from utils.sft_utils import (  # noqa: E402
    pick_criterion,
    resolve_device,
    save_checkpoint,
    train_one_epoch,
)


def run_support_sampling(
    support_indices: List[int],
    output_dir: Path,
    *,
    table_dir: Path,
    video_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = python_cmd(
        [
            str(STEP_02),
            "--table-path",
            str(table_dir),
            "--video-path",
            str(video_dir),
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
            *[str(i) for i in support_indices],
        ]
    )
    env = os.environ.copy()
    env["SCOLI_KM_PROFILE"] = KM_PROFILE
    print("Running support sampling:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(CODE_VIDEO_ROOT), env=env)


def count_patch_labels(dataset: SigLIPFullGaitDatasetPKL) -> Dict[str, int]:
    pos = neg = 0
    for entry in dataset.patch_metadata:
        label = entry.get("label")
        if label is None:
            continue
        if float(label) >= 0.5:
            pos += 1
        else:
            neg += 1
    return {"pos": pos, "neg": neg}


def compute_pos_weight(dataset: SigLIPFullGaitDatasetPKL) -> float:
    counts = count_patch_labels(dataset)
    pos = max(1, counts["pos"])
    neg = max(0, counts["neg"])
    return neg / float(pos)


def train_posttrain(
    *,
    support_pkl_dir: Path,
    base_checkpoint: Path,
    output_checkpoint: Path,
    cfg: HeadFinetuneConfig,
    freeze_policy: FreezePolicy = "head_only",
    encoder_lr: Optional[float] = None,
    head_lr: Optional[float] = None,
    unfreeze_n_blocks: int = 2,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    device = device or resolve_device(None)
    print(f"Device: {device}")

    model, ckpt_config, _ = load_deploy_checkpoint_model(base_checkpoint, device)
    freeze_stats = apply_freeze_policy(
        model,
        freeze_policy,
        unfreeze_n_blocks=unfreeze_n_blocks,
    )
    trainable, total = count_trainable(model)
    print(
        f"Freeze policy={freeze_policy}: frozen={freeze_stats.get('frozen_params', 0):,}, "
        f"trainable={trainable:,} / {total:,}"
    )

    prompts_path = resolve_prompts_path(ckpt_config)
    prompt_selection = resolve_prompt_selection(ckpt_config)
    print(f"Prompts: {prompts_path} (selection={prompt_selection})")

    dataset = SigLIPFullGaitDatasetPKL(
        pkl_data_dir=str(support_pkl_dir),
        km_gaussian_noise_std=cfg.km_gaussian_noise_std,
        mode="train",
        prompts_path=str(prompts_path),
        prompt_selection=prompt_selection,
    )
    label_counts = count_patch_labels(dataset)
    print(f"Support patches: {len(dataset)} (pos={label_counts['pos']}, neg={label_counts['neg']})")
    if len(dataset) == 0:
        raise ValueError(f"No patches found in {support_pkl_dir}")

    pos_weight = compute_pos_weight(dataset)
    loader = DataLoader(
        dataset,
        batch_size=min(cfg.batch_size, len(dataset)),
        shuffle=True,
        collate_fn=fullgait_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    enc_lr = encoder_lr if encoder_lr is not None else cfg.lr
    hd_lr = head_lr if head_lr is not None else cfg.lr
    param_groups = build_param_groups(
        model,
        encoder_lr=enc_lr,
        head_lr=hd_lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    label_dim = int(ckpt_config.get("label_dim", 1))
    criterion = pick_criterion(
        torch.tensor(0.0),
        label_dim=label_dim,
        loss_type=cfg.loss_type,
        pos_weight=pos_weight,
        device=device,
    )

    train_config = dict(ckpt_config)
    train_config.update(
        {
            "posttrain": True,
            "freeze_policy": freeze_policy,
            "head_only": freeze_policy == "head_only",
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "encoder_lr": enc_lr,
            "head_lr": hd_lr,
            "unfreeze_n_blocks": unfreeze_n_blocks,
            "pos_weight": pos_weight,
            "verbose": cfg.verbose,
            "prompts_path": str(prompts_path),
            "prompt_selection": prompt_selection,
            "support_pkl_dir": str(support_pkl_dir),
            "base_checkpoint": str(base_checkpoint),
        }
    )

    best_loss = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        avg_loss, metrics, global_step = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device,
            scheduler=None,
            writer=None,
            global_step=global_step,
            config=train_config,
        )
        history.append({"epoch": epoch, "loss": avg_loss, **metrics})
        if cfg.verbose:
            print(
                f"Epoch {epoch}/{cfg.epochs}: loss={avg_loss:.4f}, "
                f"acc={metrics.get('accuracy', 0):.4f}, "
                f"auc={metrics.get('auc_roc', 0):.4f}"
            )
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                model,
                optimizer,
                epoch,
                avg_loss,
                avg_loss,
                str(output_checkpoint),
                train_config,
                best_acc=metrics.get("accuracy"),
                best_auc=metrics.get("auc_roc"),
            )

    return {
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "label_counts": label_counts,
        "pos_weight": pos_weight,
        "freeze_stats": freeze_stats,
        "freeze_policy": freeze_policy,
        "encoder_lr": enc_lr,
        "head_lr": hd_lr,
        "unfreeze_n_blocks": unfreeze_n_blocks,
        "history": history,
        "output_checkpoint": str(output_checkpoint),
    }


def train_head(
    *,
    support_pkl_dir: Path,
    base_checkpoint: Path,
    output_checkpoint: Path,
    cfg: HeadFinetuneConfig,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    return train_posttrain(
        support_pkl_dir=support_pkl_dir,
        base_checkpoint=base_checkpoint,
        output_checkpoint=output_checkpoint,
        cfg=cfg,
        freeze_policy="head_only",
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-only post-training on support set")
    parser.add_argument("--support-pkl-dir", type=Path, default=SUPPORT_PKL_DIR)
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--output-checkpoint", type=Path, default=POSTTRAIN_CHECKPOINT)
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--epochs", type=int, default=HeadFinetuneConfig.epochs)
    parser.add_argument("--lr", type=float, default=HeadFinetuneConfig.lr)
    parser.add_argument("--batch-size", type=int, default=HeadFinetuneConfig.batch_size)
    parser.add_argument("--weight-decay", type=float, default=HeadFinetuneConfig.weight_decay)
    parser.add_argument(
        "--km-gaussian-noise-std",
        type=float,
        default=HeadFinetuneConfig.km_gaussian_noise_std,
    )
    parser.add_argument(
        "--loss-type",
        choices=["bce", "focal"],
        default=HeadFinetuneConfig.loss_type,
    )
    parser.add_argument("--quiet", action="store_true", help="Disable per-batch training logs")
    parser.add_argument(
        "--shots-per-class",
        type=int,
        default=1,
        help="Number of positive and negative support subjects per class (few-shot K)",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=POSTTRAIN_MANIFEST,
        help="Where to write posttrain_manifest.json",
    )
    args = parser.parse_args()

    support = resolve_shot_support(args.shots_per_class, table_path=args.table_dir, video_dir=args.video_dir)
    support_indices = flatten_shot_support(support)
    print(
        f"Support ({args.shots_per_class}-shot): "
        f"pos={support['pos']}, neg={support['neg']} "
        f"({len(support_indices)} subjects)"
    )

    if not args.skip_sample:
        run_support_sampling(
            support_indices,
            args.support_pkl_dir,
            table_dir=args.table_dir,
            video_dir=args.video_dir,
        )

    cfg = HeadFinetuneConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        km_gaussian_noise_std=args.km_gaussian_noise_std,
        loss_type=args.loss_type,
        verbose=not args.quiet,
    )
    train_result = train_head(
        support_pkl_dir=args.support_pkl_dir,
        base_checkpoint=args.base_checkpoint,
        output_checkpoint=args.output_checkpoint,
        cfg=cfg,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shots_per_class": args.shots_per_class,
        "support_subjects": support,
        "support_indices": support_indices,
        "n_support_subjects": len(support_indices),
        "base_checkpoint": str(args.base_checkpoint),
        "output_checkpoint": str(args.output_checkpoint),
        "support_pkl_dir": str(args.support_pkl_dir),
        "km_profile": KM_PROFILE,
        "label_excel": str(LABEL_EXCEL),
        "prompts_path": str(resolve_prompts_path()),
        "prompt_selection": resolve_prompt_selection(),
        "min_file_index": MIN_FILE_INDEX,
        "training": train_result,
        "hyperparameters": {
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "weight_decay": cfg.weight_decay,
            "km_gaussian_noise_std": cfg.km_gaussian_noise_std,
            "loss_type": cfg.loss_type,
            "verbose": cfg.verbose,
        },
    }
    manifest_path = args.manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved checkpoint: {args.output_checkpoint}")


if __name__ == "__main__":
    main()
