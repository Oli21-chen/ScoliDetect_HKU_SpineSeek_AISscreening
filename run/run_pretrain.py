# -*- coding: utf-8 -*-
"""
SigLIP trimodal pretraining — fair comparison of preprocessed datasets (raw vs peak-anchored).

Same CLIP-style objective and hyperparameters; only ``pkl_data_dir`` differs via ``--condition``.

After each run, compare:
  - best validation trimodal loss (checkpoint ``siglip_checkpoint_best.pth``)
  - batch-level TopK on val (``inbatch_retrieval_at_1/5``, batch size = training batch size)

Examples:
  python run_pretrain.py --condition raw
  python run_pretrain.py --condition peak
  python tools/compare_pretrain_conditions.py \\
    checkpoints/pretrain_align_raw_* checkpoints/pretrain_align_peak_*

# Raw PKLs
python run_pretrain.py --condition raw
# or
python DCU_train.py --gpu-ids 0,1 --script run_pretrain.py --condition raw

# Peak-anchored PKLs (stage3)
python run_pretrain.py --condition peak

"""

import argparse
import os
import sys
import math
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models import SigLIPBaseline
from utils.pre_utils import (
    save_checkpoint,
    load_checkpoint,
    print_gpu_info,
    unwrap_model,
    trimodality_contrastive_loss,
    append_epoch_metrics,
    save_pretrain_metrics,
    make_train_val_subsets,
)
from utils.data_sampler import SigLIPPretrainDataset, SigLIPPretrainDatasetPKL, siglip_collate_fn
from utils.training import evaluate, train_one_epoch, _extract_text
from utils.alignment_metrics import evaluate_val_alignment
from utils.sft_utils import resolve_device, setup_multi_gpu
from utils.utils import plot_pretrain_loss_curves

CONDITION_PRESETS = {
    "raw": {
        "run_name": "pretrain_align_raw",
        "pkl_data_dir": ["./data/train_sz_pkl^1", "./data/test_sz_pkl^1"],
    },
    "peak": {
        "run_name": "pretrain_align_peak",
        "pkl_data_dir": ["./data/sz_pkl_stage3"],
    },
}


def build_default_config() -> dict:
    """Shared pretrain config for raw vs peak (only dataset path differs)."""
    return {
        "run_name": "pretrain_align_raw",
        "table_dir": None,
        "video_dir": None,
        "directions": ["going_backward", "going_forward"],
        "video_target_size": (224, 224),
        "patch_size": 96,
        "video_frame_count": 32,
        "batch_size": 4,
        "gradient_accumulation_steps": 16,
        "num_epochs": 200,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.05,
        "save_dir": "./checkpoints",
        "log_dir": "./logs",
        "num_workers": 4,
        "val_split": 0.2,
        "random_seed": 42,
        "prompts_path": "./data/general_gait_prompts_from_report.json",
        "prompt_selection": "concise_prompts",
        "pkl_data_dir": ["./data/train_sz_pkl^1", "./data/test_sz_pkl^1"],
        "km_gaussian_noise_std": 0.2,
        "val_split_mode": "patient",
        "text_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "text_max_length": 128,
        "text_trainable": False,
        "use_amp": True,
        "max_grad_norm": 1.0,
        "empty_cache_frequency": 5,
        "gpu_ids": [0, 1],
        "use_distributed": False,
        "video_encoder_type": "vivit",
        "km_encoder_type": "vit",
        "hidden_dim": 512,
        "km_feature_dim": 238,
        "video_encoder_kwargs": {
            "img_size": 224,
            "patch_size": 16,
            "temporal_size": 32,
            "in_channels": 3,
            "depth": 4,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
        },
        "km_encoder_kwargs": {
            "depth": 4,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "drop": 0.0,
            "attn_drop": 0.0,
        },
        "resume_from_checkpoint": None,
        "plot_loss_curves": True,
    }


def parse_args():
    ap = argparse.ArgumentParser(
        description="SigLIP pretrain: same CLIP objective, compare raw vs peak PKL corpora"
    )
    ap.add_argument(
        "--condition",
        choices=["raw", "peak"],
        required=True,
        help="raw: train+test PKLs; peak: sz_pkl_stage3 (peak-anchored)",
    )
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--pkl-data-dir", nargs="+", default=None)
    ap.add_argument("--num-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--val-split-mode", choices=["patient", "random"], default=None)
    ap.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    return ap.parse_args()


def merge_config_from_args(config: dict, args) -> dict:
    preset = CONDITION_PRESETS[args.condition]
    config.update(preset)
    config["condition"] = args.condition
    if args.run_name:
        config["run_name"] = args.run_name
    if args.pkl_data_dir:
        config["pkl_data_dir"] = args.pkl_data_dir
    if args.num_epochs is not None:
        config["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        config["gradient_accumulation_steps"] = args.grad_accum
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.val_split_mode:
        config["val_split_mode"] = args.val_split_mode
    if args.resume:
        config["resume_from_checkpoint"] = args.resume
    return config


def _log_alignment(writer, metrics: dict, epoch: int) -> None:
    if writer is None:
        return
    writer.add_scalar("Alignment/val_cosine_gap", metrics["cosine_gap"], epoch)
    writer.add_scalar("Alignment/val_inbatch_r1", metrics.get("inbatch_retrieval_at_1", 0.0), epoch)
    writer.add_scalar("Alignment/val_inbatch_r5", metrics.get("inbatch_retrieval_at_5", 0.0), epoch)
    writer.add_scalar("Alignment/val_gallery_r1", metrics.get("retrieval_at_1", 0.0), epoch)


def _print_alignment(metrics: dict, batch_size: int) -> None:
    ib1 = metrics.get("inbatch_retrieval_at_1", float("nan"))
    ib5 = metrics.get("inbatch_retrieval_at_5", float("nan"))
    g1 = metrics.get("retrieval_at_1", float("nan"))
    print(
        f"   Alignment:  gap={metrics['cosine_gap']:.4f} | "
        f"in-batch R@1={ib1:.1%} (chance {1/batch_size:.0%}) | "
        f"in-batch R@5={ib5:.1%} | gallery R@1={g1:.1%}"
    )


def save_pretrain_summary(run_ckpt_dir: str, summary: dict) -> str:
    path = os.path.join(run_ckpt_dir, "pretrain_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def main():
    args = parse_args()
    config = merge_config_from_args(build_default_config(), args)

    visible_gpus = print_gpu_info()
    gpu_ids = config.get("gpu_ids")
    device = resolve_device(gpu_ids)

    print("=" * 50)
    print("SigLIP Pretraining — dataset comparison")
    print("=" * 50)
    print(f"Condition:     {config['condition']}")
    print(f"PKL dirs:      {config['pkl_data_dir']}")
    print(f"Batch size:    {config['batch_size']} (in-batch TopK uses this)")
    print(f"Grad accum:    {config.get('gradient_accumulation_steps', 1)}")
    print(f"Val split:     {config.get('val_split_mode', 'patient')}")
    print(f"Best ckpt by:  val_loss (trimodal)")
    print("=" * 50)

    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    print("\nLoading dataset...")
    pkl_cfg = config.get("pkl_data_dir")
    if not pkl_cfg:
        raise ValueError("pkl_data_dir is required")
    dirs = pkl_cfg if isinstance(pkl_cfg, (list, tuple)) else [pkl_cfg]
    datasets = []
    for d in dirs:
        pkl_abs = os.path.abspath(os.path.normpath(str(d)))
        if not os.path.isdir(pkl_abs):
            raise FileNotFoundError(f"pkl_data_dir not found: {pkl_abs}")
        print(f"  - {pkl_abs}")
        datasets.append(
            SigLIPPretrainDatasetPKL(
                pkl_data_dir=pkl_abs,
                prompts_path=config.get("prompts_path"),
                prompt_selection=config.get("prompt_selection", "concise_prompts"),
                km_gaussian_noise_std=config.get("km_gaussian_noise_std"),
            )
        )
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    print(f"Dataset loaded: {len(dataset)} patches")

    train_dataset, val_dataset = make_train_val_subsets(dataset, config)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    workers = config["num_workers"] if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=siglip_collate_fn,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=siglip_collate_fn,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config['run_name']}_{timestamp}"
    run_log_dir = os.path.join(config["log_dir"], run_name)
    run_ckpt_dir = os.path.join(config["save_dir"], run_name)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    config_path = os.path.join(run_ckpt_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: (str(v) if isinstance(v, (list, tuple)) else v) for k, v in config.items()},
            f,
            indent=2,
            default=str,
        )
    print(f"Config saved:   {config_path}")
    print(f"Checkpoint dir: {run_ckpt_dir}")

    video_kwargs = config.get("video_encoder_kwargs") if config["video_encoder_type"] != "conv3d" else None
    km_kwargs = config.get("km_encoder_kwargs") if config["km_encoder_type"] != "baseline" else None

    base_model = SigLIPBaseline(
        km_input_dim=config.get("km_feature_dim", 238),
        hidden_dim=config.get("hidden_dim", 512),
        projection_dim=512,
        text_model_name=config["text_model_name"],
        text_max_length=config["text_max_length"],
        text_trainable=config["text_trainable"],
        video_encoder_type=config["video_encoder_type"],
        video_encoder_kwargs=video_kwargs,
        km_encoder_type=config["km_encoder_type"],
        km_encoder_kwargs=km_kwargs,
    ).to(device)

    model = setup_multi_gpu(base_model, gpu_ids=gpu_ids, use_distributed=False)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    batches_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
    total_steps = config["num_epochs"] * steps_per_epoch
    warmup_steps = max(1, int(total_steps * config.get("warmup_ratio", 0.05)))
    cosine_steps = max(1, total_steps - warmup_steps)
    base_lr = config["learning_rate"]
    eta_min = 1e-6

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )

    def cosine_lambda(step):
        relative_step = max(0, step - warmup_steps)
        progress = min(relative_step / max(cosine_steps - 1, 1), 1.0)
        cosine_val = (1 + math.cos(math.pi * progress)) / 2
        return eta_min / base_lr + (1.0 - eta_min / base_lr) * cosine_val

    cosine_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lambda)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    scaler = torch.amp.GradScaler("cuda") if config["use_amp"] and device.type == "cuda" else None
    writer = SummaryWriter(log_dir=run_log_dir)
    loss_fn = trimodality_contrastive_loss

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = 0
    best_alignment: dict = {}
    loss_history = []

    if config.get("resume_from_checkpoint") and os.path.exists(config["resume_from_checkpoint"]):
        start_epoch, global_step, best_val_loss = load_checkpoint(
            checkpoint_path=config["resume_from_checkpoint"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

    print("\nStarting training...")
    for epoch in range(start_epoch, config["num_epochs"] + 1):
        train_loss, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            scheduler=scheduler,
            use_amp=config["use_amp"] and device.type == "cuda",
            epoch=epoch,
            global_step=global_step,
            verbose=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            writer=writer,
            max_grad_norm=config.get("max_grad_norm"),
        )

        if device.type == "cuda" and config.get("empty_cache_frequency", 0) > 0:
            if epoch % config["empty_cache_frequency"] == 0:
                torch.cuda.empty_cache()

        print(f"\n{'='*80}\nEpoch {epoch} - Validation\n{'='*80}")
        val_loss, val_loss_details = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            writer=writer,
            global_step=global_step,
        )
        val_alignment = evaluate_val_alignment(
            model,
            val_loader,
            device,
            _extract_text,
            batch_size=int(config["batch_size"]),
        )
        _log_alignment(writer, val_alignment, epoch)

        append_epoch_metrics(
            loss_history,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=scheduler.get_last_lr()[0],
            val_details=val_loss_details,
            val_alignment=val_alignment,
        )
        save_pretrain_metrics(loss_history, run_ckpt_dir)

        print(f"\n{'='*80}\nEpoch {epoch}/{config['num_epochs']} Summary:")
        print(f"   Train loss: {train_loss:.6f}")
        print(f"   Val loss:   {val_loss:.6f}")
        if val_loss_details:
            print(
                f"   Components: km-t={val_loss_details['loss_km_text']:.4f} "
                f"v-t={val_loss_details['loss_video_text']:.4f} "
                f"v-k={val_loss_details['loss_video_km']:.4f}"
            )
        _print_alignment(val_alignment, config["batch_size"])
        print(f"{'='*80}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_alignment = dict(val_alignment)
            best_path = os.path.join(run_ckpt_dir, "siglip_checkpoint_best.pth")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                val_loss=val_loss,
                config=config,
                path=best_path,
            )
            print(f"🏆 New best val_loss={best_val_loss:.6f} → {best_path}\n")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            val_loss=val_loss,
            config=config,
            path=os.path.join(run_ckpt_dir, "siglip_checkpoint_latest.pth"),
        )

    summary = {
        "condition": config["condition"],
        "run_name": run_name,
        "pkl_data_dir": config["pkl_data_dir"],
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "batch_size": config["batch_size"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_alignment": best_alignment,
        "checkpoint_best": os.path.join(run_ckpt_dir, "siglip_checkpoint_best.pth"),
    }
    summary_path = save_pretrain_summary(run_ckpt_dir, summary)

    metric_paths = save_pretrain_metrics(loss_history, run_ckpt_dir)
    if config.get("plot_loss_curves", True) and loss_history:
        curve_path = os.path.join(run_ckpt_dir, "loss_curves.png")
        plot_pretrain_loss_curves(loss_history, curve_path, title=config["run_name"])
        print(f"Loss curves: {curve_path}")

    print(f"\n{'='*80}")
    print("Training complete")
    print(f"  Best val loss:     {best_val_loss:.6f} (epoch {best_epoch})")
    if best_alignment:
        _print_alignment(best_alignment, config["batch_size"])
    print(f"  Summary JSON:    {summary_path}")
    print(f"  Loss history:    {metric_paths['json']}")
    print(f"\nCompare conditions:")
    print(f"  python tools/compare_pretrain_conditions.py <raw_run_dir> <peak_run_dir>")
    print(f"{'='*80}")
    writer.close()


if __name__ == "__main__":
    main()
