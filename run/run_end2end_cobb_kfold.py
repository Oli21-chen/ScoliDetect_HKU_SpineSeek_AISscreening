"""
K-Fold cross-validation entrypoint for multimodal SFTRegressor.
Subject-level stratified splitting to prevent data leakage.

Produces per-fold metrics and aggregated mean +/- std with 95% CI,
suitable for reporting in academic manuscripts (e.g. Nature Medicine).
"""

import os
import sys
import time
import json
import warnings
import multiprocessing
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

warnings.filterwarnings("ignore", message="PyTorch is not compiled with NCCL support")

if sys.platform == 'win32':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold

from utils.data_sampler import SigLIPFullGaitDatasetPKL, fullgait_collate_fn
from utils.utils import build_subject_map
from utils.sft_utils import (
    resolve_device,
    setup_multi_gpu,
    pick_criterion,
    train_one_epoch,
    eval_epoch,
    save_checkpoint,
    print_label_distribution,
    load_pretrained_checkpoint,
    print_model_info,
    create_optimizer_and_scheduler,
)
from models.sft_regressorv2 import SFTRegressor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config: Dict[str, Any] = {
        # ---- K-fold settings ----
        "n_folds": 5,
        "start_fold": 1,  # Set to N to skip folds 1..N-1 and resume from fold N (1 = run all)
        "resume_ckpt_base": None,
        "random_seed": 42,

        # ---- Data ----
        "use_preprocessed_pkl": True,
        "pkl_data_dir": ["./data/train_sz_pkl^1","./data/test_sz_pkl^1"],
        "external_test_dir": ["./data/test_dk_pkl^1"],  # External cohort for independent validation (evaluated after each fold)
        "prompts_path": r"C:\Users\Administrator\project\data\sz_general_gait_prompts.json",
        "prompt_selection": "concise_prompts",
        "km_feature_dim": 238,
        "binary_threshold": 15.0,
        "km_gaussian_noise_std": 0.2,

        # ---- Training ----
        "batch_size": 8,
        "num_epochs": 100,
        "learning_rate": 8e-6,
        "warmup_ratio": 0.1,
        "num_workers": 2,
        "weight_decay": 0.1,
        "optimizer": "adamw",
        "loss_type": "focal",
        "max_grad_norm": None,

        # ---- Model ----
        "hidden_dim": 512,
        "label_dim": 1,
        "video_target_size": (224, 224),
        "patch_size": 96,
        "video_frame_count": 32,
        "video_encoder_type": "vivit",
        "video_encoder_kwargs": {
            "img_size": 224,
            "patch_size": 16,
            "temporal_size": 32,
            "in_channels": 3,
            "depth": 4,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "drop_rate": 0.2,
            "attn_drop_rate": 0.2,
        },
        "km_encoder_type": "patch_vit",
        "km_encoder_kwargs": {
            "depth": 4,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "drop": 0.2,
            "attn_drop": 0.2,
        },
        "text_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "text_max_length": 128,
        "text_trainable": False,
        "use_text": True,
        "use_latent_pooling": True,
        "latent_pool_size": 1,
        "regressor_dropout": 0.2,
        "use_km_video_cross_attn": True,
        "cross_attn_num_heads": 8,
        "cross_attn_drop": 0.1,
        "use_gated_token_pooling": True,

        "pretrained_checkpoint_path": None,
        "gpu_ids": [0, 1],
        "use_distributed": False,

        "run_name": "kfold5_latentpool_kvt_4layer8_512^1_15d_focal_adamw",
        "save_dir": "./checkpoints",
        "log_dir": "./logs",
        "debug_print_labels": False,
    }

    # ------------------------------------------------------------------
    # Device setup
    # ------------------------------------------------------------------
    gpu_ids = config.get("gpu_ids")
    device = resolve_device(gpu_ids)
    print(f"Primary device: {device}")

    if gpu_ids is not None and len(gpu_ids) > 1:
        num_gpus = len(gpu_ids)
    elif gpu_ids is None and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1
    effective_batch_size = config["batch_size"] * num_gpus
    print(f"Effective batch size: {effective_batch_size} ({config['batch_size']} x {num_gpus} GPU(s))")

    # ------------------------------------------------------------------
    # Load full dataset
    # ------------------------------------------------------------------
    pkl_entries = config["pkl_data_dir"]
    if isinstance(pkl_entries, (str, bytes)):
        pkl_entries = [pkl_entries]

    datasets = []
    total_patches = 0
    for idx, entry in enumerate(pkl_entries):
        pkl_data_dir = os.path.abspath(os.path.normpath(entry))
        print(f"  PKL dir [{idx}]: {pkl_data_dir}")
        ds = SigLIPFullGaitDatasetPKL(
            pkl_data_dir=pkl_data_dir,
            km_gaussian_noise_std=config.get("km_gaussian_noise_std", 0.2),
            mode="train",
            prompts_path=config.get("prompts_path"),
            prompt_selection=config.get("prompt_selection", "concise_prompts"),
            binary_threshold=config.get("binary_threshold"),
        )
        print(f"    -> {len(ds)} patches")
        datasets.append(ds)
        total_patches += len(ds)

    full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"Total patches loaded: {total_patches}")

    # ------------------------------------------------------------------
    # Load external test dataset (DK cohort)
    # ------------------------------------------------------------------
    ext_test_entries = config.get("external_test_dir")
    ext_test_dataset = None
    if ext_test_entries is not None:
        if isinstance(ext_test_entries, (str, bytes)):
            ext_test_entries = [ext_test_entries]
        ext_datasets = []
        ext_patches = 0
        for idx, entry in enumerate(ext_test_entries):
            edir = os.path.abspath(os.path.normpath(entry))
            print(f"  External test dir [{idx}]: {edir}")
            ds = SigLIPFullGaitDatasetPKL(
                pkl_data_dir=edir,
                km_gaussian_noise_std=None,
                mode="test",
                prompts_path=config.get("prompts_path"),
                prompt_selection=config.get("prompt_selection", "concise_prompts"),
                binary_threshold=config.get("binary_threshold"),
            )
            print(f"    -> {len(ds)} patches")
            ext_datasets.append(ds)
            ext_patches += len(ds)
        ext_test_dataset = ConcatDataset(ext_datasets) if len(ext_datasets) > 1 else ext_datasets[0]
        print(f"External test dataset: {ext_patches} patches")

    # ------------------------------------------------------------------
    # Build subject-level map for stratified k-fold
    # ------------------------------------------------------------------
    binary_threshold = config.get("binary_threshold", 15.0)
    print("\nBuilding subject-level map for stratified k-fold...")
    subject_ids, subject_labels, subject_to_indices = build_subject_map(
        full_dataset, binary_threshold
    )
    n_subjects = len(subject_ids)
    n_pos = int(subject_labels.sum())
    n_neg = n_subjects - n_pos
    print(f"  Unique subjects: {n_subjects}")
    print(f"  Positive subjects (label=1): {n_pos}")
    print(f"  Negative subjects (label=0): {n_neg}")

    # ------------------------------------------------------------------
    # K-fold loop
    # ------------------------------------------------------------------
    n_folds = config["n_folds"]
    start_fold = config.get("start_fold", 1)
    seed = config["random_seed"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # When resuming, reuse existing run directory; otherwise create new one
    resume_ckpt_base = config.get("resume_ckpt_base")  # e.g. "./checkpoints/kfold5_..._20260309_120000"
    if resume_ckpt_base and start_fold > 1:
        base_ckpt_dir = resume_ckpt_base
        base_log_dir = resume_ckpt_base.replace(config["save_dir"], config.get("log_dir", "./logs"), 1)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = config.get("run_name", "kfold")
        base_ckpt_dir = os.path.join(config["save_dir"], f"{run_name}_{timestamp}")
        base_log_dir = os.path.join(config.get("log_dir", "./logs"), f"{run_name}_{timestamp}")
    os.makedirs(base_ckpt_dir, exist_ok=True)
    os.makedirs(base_log_dir, exist_ok=True)
    print(f"\nCheckpoint base: {base_ckpt_dir}")
    print(f"Log base:        {base_log_dir}")
    if start_fold > 1:
        print(f"Resuming from fold {start_fold} (skipping folds 1..{start_fold - 1})")

    all_fold_results: List[Dict[str, Any]] = []

    subject_ids_arr = np.array(subject_ids)

    for fold_idx, (train_subj_indices, val_subj_indices) in enumerate(
        skf.split(subject_ids_arr, subject_labels)
    ):
        fold_num = fold_idx + 1

        train_subjects = set(subject_ids_arr[train_subj_indices])
        val_subjects = set(subject_ids_arr[val_subj_indices])
        train_patch_indices = []
        val_patch_indices = []
        for sid in train_subjects:
            train_patch_indices.extend(subject_to_indices[sid])
        for sid in val_subjects:
            val_patch_indices.extend(subject_to_indices[sid])

        # --- Skip completed folds: load results from saved checkpoint ---
        if fold_num < start_fold:
            fold_ckpt_dir = os.path.join(base_ckpt_dir, f"fold_{fold_num}")
            best_ckpt = os.path.join(fold_ckpt_dir, "checkpoint_best_auc.pth")
            if os.path.exists(best_ckpt):
                ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
                prev_best_acc = ckpt.get("best_acc", 0.0)
                prev_best_auc = ckpt.get("best_auc", 0.0)
                print(f"  [Fold {fold_num}] Skipped (already done) — best_acc={prev_best_acc:.4f}, best_auc={prev_best_auc:.4f}")
                fold_result = {
                    "fold": fold_num,
                    "best_val_acc": prev_best_acc,
                    "best_val_auc": prev_best_auc,
                    "best_val_loss": ckpt.get("val_loss", 0.0),
                    "train_subjects": len(train_subjects),
                    "val_subjects": len(val_subjects),
                    "train_patches": len(train_patch_indices),
                    "val_patches": len(val_patch_indices),
                    "skipped": True,
                }
                all_fold_results.append(fold_result)
            else:
                print(f"  [Fold {fold_num}] Skipped but no checkpoint found at {best_ckpt} — no results recorded")
            continue

        print(f"\n{'='*70}")
        print(f"  FOLD {fold_num}/{n_folds}")
        print(f"{'='*70}")
        print(f"  Train subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)}")
        print(f"  Train patches:  {len(train_patch_indices)} | Val patches: {len(val_patch_indices)}")

        train_dataset = Subset(full_dataset, train_patch_indices)
        val_dataset = Subset(full_dataset, val_patch_indices)

        # Label distribution
        label_stats = print_label_distribution(train_dataset, val_dataset, binary_threshold)
        train_pos = max(1, label_stats.get("train_pos", 0))
        train_neg = max(0, label_stats.get("train_neg", 0))
        pos_weight = train_neg / float(train_pos)
        fold_config = {**config, "pos_weight": pos_weight, "fold": fold_num}

        # DataLoaders
        num_workers = config["num_workers"] if sys.platform != 'win32' else min(config["num_workers"], 8)
        pin_memory = device.type == "cuda" and sys.platform != 'win32'

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=fullgait_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=fullgait_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        # Fresh model per fold
        model = SFTRegressor(
            km_feature_dim=config["km_feature_dim"],
            hidden_dim=config["hidden_dim"],
            label_dim=config["label_dim"],
            video_encoder_type=config.get("video_encoder_type", "vivit"),
            video_encoder_kwargs=config.get("video_encoder_kwargs"),
            km_encoder_type=config.get("km_encoder_type", "baseline"),
            km_encoder_kwargs=config.get("km_encoder_kwargs"),
            text_model_name=config.get("text_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            text_max_length=config.get("text_max_length", 128),
            text_trainable=config.get("text_trainable", True),
            use_text=config.get("use_text", True),
            use_latent_pooling=config.get("use_latent_pooling", False),
            latent_pool_size=config.get("latent_pool_size", 1),
            regressor_dropout=config.get("regressor_dropout", 0.1),
            use_km_video_cross_attn=config.get("use_km_video_cross_attn", True),
            cross_attn_num_heads=config.get("cross_attn_num_heads", 8),
            cross_attn_drop=config.get("cross_attn_drop", 0.1),
            use_gated_token_pooling=config.get("use_gated_token_pooling", True),
        )

        if config.get("pretrained_checkpoint_path"):
            model = load_pretrained_checkpoint(
                model, config["pretrained_checkpoint_path"], device,
                text_trainable=config.get("text_trainable", True),
            )

        model = setup_multi_gpu(model, gpu_ids=gpu_ids, use_distributed=config.get("use_distributed", False))
        if fold_idx == 0:
            print_model_info(model, config)

        # Criterion
        sample_batch = next(iter(train_loader))
        sample_label = sample_batch.get("label")
        criterion = pick_criterion(
            sample_label,
            label_dim=config["label_dim"],
            loss_type=config.get("loss_type", "bce"),
            pos_weight=pos_weight,
            device=device,
        )

        # Optimizer & scheduler
        steps_per_epoch = max(1, len(train_loader))
        optimizer, scheduler = create_optimizer_and_scheduler(model, fold_config, steps_per_epoch)

        # Directories for this fold
        fold_ckpt_dir = os.path.join(base_ckpt_dir, f"fold_{fold_num}")
        fold_log_dir = os.path.join(base_log_dir, f"fold_{fold_num}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)
        os.makedirs(fold_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=fold_log_dir)

        best_val_loss = float('inf')
        best_acc = 0.0
        best_auc = 0.0
        global_step = 0

        for epoch in range(1, config["num_epochs"] + 1):
            epoch_start = time.time()

            train_loss, train_metrics, global_step = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                scheduler, writer, global_step, config=fold_config,
            )

            val_loss, val_metrics = eval_epoch(
                model, val_loader, criterion, device,
                writer, global_step, config=fold_config,
            )

            val_acc = val_metrics.get('accuracy', 0.0)
            val_auc = val_metrics.get('auc_roc', 0.0)
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - epoch_start

            print(
                f"[Fold {fold_num} Epoch {epoch:03d}] "
                f"TrainLoss: {train_loss:.6f} | TrainAcc: {train_metrics.get('accuracy',0):.4f} | "
                f"ValLoss: {val_loss:.6f} | ValAcc: {val_acc:.4f} | ValAUC: {val_auc:.4f} | "
                f"BestAcc: {best_acc:.4f} | BestAUC: {best_auc:.4f} | "
                f"LR: {current_lr:.2e} | {elapsed:.1f}s"
            )

            if val_acc >= best_acc:
                best_acc = val_acc
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    os.path.join(fold_ckpt_dir, "checkpoint_best.pth"),
                    fold_config, best_acc=best_acc, best_auc=best_auc,
                )

            if val_auc >= best_auc:
                best_auc = val_auc
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    os.path.join(fold_ckpt_dir, "checkpoint_best_auc.pth"),
                    fold_config, best_acc=best_acc, best_auc=best_auc,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Save last checkpoint each epoch (for potential resume)
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                os.path.join(fold_ckpt_dir, "checkpoint_last.pth"),
                fold_config, best_acc=best_acc, best_auc=best_auc,
            )

        writer.close()

        # ----------------------------------------------------------
        # External evaluation: load best-AUC checkpoint, eval on DK
        # ----------------------------------------------------------
        ext_metrics_result: Dict[str, float] = {}
        if ext_test_dataset is not None:
            print(f"\n  Evaluating fold {fold_num} best model on external test set...")
            best_auc_ckpt = os.path.join(fold_ckpt_dir, "checkpoint_best_auc.pth")
            ckpt = torch.load(best_auc_ckpt, map_location=device, weights_only=False)
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt["model_state_dict"])

            ext_loader = DataLoader(
                ext_test_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                collate_fn=fullgait_collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
            )
            ext_loss, ext_metrics = eval_epoch(
                model, ext_loader, criterion, device,
                writer=None, global_step=0, verbose=False, config=fold_config,
            )
            ext_metrics_result = {
                "ext_acc": ext_metrics.get('accuracy', 0.0),
                "ext_auc": ext_metrics.get('auc_roc', 0.0),
                "ext_sensitivity": ext_metrics.get('sensitivity', 0.0),
                "ext_specificity": ext_metrics.get('specificity', 0.0),
                "ext_f1": ext_metrics.get('f1', 0.0),
                "ext_loss": ext_loss,
            }
            print(
                f"  External test — Acc: {ext_metrics_result['ext_acc']:.4f} | "
                f"AUC: {ext_metrics_result['ext_auc']:.4f} | "
                f"Sens: {ext_metrics_result['ext_sensitivity']:.4f} | "
                f"Spec: {ext_metrics_result['ext_specificity']:.4f}"
            )

        fold_result = {
            "fold": fold_num,
            "best_val_acc": best_acc,
            "best_val_auc": best_auc,
            "best_val_loss": best_val_loss,
            "train_subjects": len(train_subjects),
            "val_subjects": len(val_subjects),
            "train_patches": len(train_patch_indices),
            "val_patches": len(val_patch_indices),
            "final_train_loss": train_loss,
            "final_train_acc": train_metrics.get('accuracy', 0.0),
            "final_val_loss": val_loss,
            "final_val_acc": val_acc,
            "final_val_auc": val_auc,
            "final_val_sensitivity": val_metrics.get('sensitivity', 0.0),
            "final_val_specificity": val_metrics.get('specificity', 0.0),
            "final_val_f1": val_metrics.get('f1', 0.0),
            **ext_metrics_result,
        }
        all_fold_results.append(fold_result)
        print(f"\n  Fold {fold_num} done: best_acc={best_acc:.4f}, best_auc={best_auc:.4f}")

    # ------------------------------------------------------------------
    # Aggregate results across folds
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  K-FOLD CROSS-VALIDATION SUMMARY ({n_folds} folds)")
    print(f"{'='*70}\n")

    # Internal validation metrics
    internal_metric_keys = [
        ("best_val_acc", "Val Accuracy (best)"),
        ("best_val_auc", "Val AUC-ROC (best)"),
        ("final_val_sensitivity", "Val Sensitivity"),
        ("final_val_specificity", "Val Specificity"),
        ("final_val_f1", "Val F1 Score"),
    ]

    def _collect_metric(key_name: str) -> List[float]:
        vals: List[float] = []
        for r in all_fold_results:
            if key_name in r and r[key_name] is not None:
                try:
                    vals.append(float(r[key_name]))
                except (TypeError, ValueError):
                    continue
        return vals

    # External validation metrics (only if external test was run)
    has_external = any(("ext_acc" in r) for r in all_fold_results)
    external_metric_keys = [
        ("ext_acc", "Ext Accuracy"),
        ("ext_auc", "Ext AUC-ROC"),
        ("ext_sensitivity", "Ext Sensitivity"),
        ("ext_specificity", "Ext Specificity"),
        ("ext_f1", "Ext F1 Score"),
    ] if has_external else []

    summary: Dict[str, Any] = {"n_folds": n_folds, "n_subjects": n_subjects, "folds": all_fold_results}

    print("--- Internal Validation (SZ cohort, 5-fold CV) ---")
    for key, display_name in internal_metric_keys:
        values = _collect_metric(key)
        if len(values) == 0:
            print(f"  {display_name:30s}: (missing)")
            continue
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci95 = 1.96 * std_val / np.sqrt(len(values)) if len(values) > 1 else 0.0
        summary[f"{key}_mean"] = mean_val
        summary[f"{key}_std"] = std_val
        summary[f"{key}_ci95"] = ci95
        print(f"  {display_name:30s}: {mean_val:.4f} +/- {std_val:.4f}  (95% CI: {mean_val-ci95:.4f} - {mean_val+ci95:.4f})")

    if external_metric_keys:
        print("\n--- External Validation (DK cohort) ---")
        for key, display_name in external_metric_keys:
            values = _collect_metric(key)
            if len(values) == 0:
                print(f"  {display_name:30s}: (missing)")
                continue
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci95 = 1.96 * std_val / np.sqrt(len(values)) if len(values) > 1 else 0.0
            summary[f"{key}_mean"] = mean_val
            summary[f"{key}_std"] = std_val
            summary[f"{key}_ci95"] = ci95
            print(f"  {display_name:30s}: {mean_val:.4f} +/- {std_val:.4f}  (95% CI: {mean_val-ci95:.4f} - {mean_val+ci95:.4f})")

    print()
    header = f"  {'Fold':>4s}  {'ValAcc':>8s}  {'ValAUC':>8s}  {'Sens':>8s}  {'Spec':>8s}  {'F1':>8s}"
    if has_external:
        header += f"  {'ExtAcc':>8s}  {'ExtAUC':>8s}  {'ExtSens':>8s}  {'ExtSpec':>8s}"
    header += f"  {'#Train':>6s}  {'#Val':>6s}"
    print("Per-fold results:")
    print(header)
    for r in all_fold_results:
        line = (
            f"  {r['fold']:4d}  {r['best_val_acc']:8.4f}  {r['best_val_auc']:8.4f}  "
            f"{r.get('final_val_sensitivity', 0.0):8.4f}  {r.get('final_val_specificity', 0.0):8.4f}  "
            f"{r.get('final_val_f1', 0.0):8.4f}"
        )
        if has_external:
            line += (
                f"  {r.get('ext_acc',0):8.4f}  {r.get('ext_auc',0):8.4f}  "
                f"{r.get('ext_sensitivity',0):8.4f}  {r.get('ext_specificity',0):8.4f}"
            )
        line += f"  {r['train_patches']:6d}  {r['val_patches']:6d}"
        print(line)

    # Save summary JSON
    summary_path = os.path.join(base_ckpt_dir, "kfold_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Save config
    config_path = os.path.join(base_ckpt_dir, "config.json")
    config_serializable = {}
    for k, v in config.items():
        try:
            json.dumps(v)
            config_serializable[k] = v
        except (TypeError, ValueError):
            config_serializable[k] = str(v)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_serializable, f, indent=2)

    print(f"\nFinished {n_folds}-fold cross-validation.")


def train_one_epoch_regression(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    writer: Optional[Any] = None,
    global_step: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float], int]:
    """
    Train for one epoch on continuous max_cobb targets.
    Keeps original model/logging intact; does not affect classification training.
    """
    model.train()
    total_loss = 0.0
    steps = 0
    total_batches = len(dataloader)
    print(f"  [Regression] Starting training: {total_batches} batches")

    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()

        video = batch["video"].to(device)
        knowledge_map = batch["knowledge_map"].to(device)
        texts = batch.get("texts", None)
        km_indices = batch.get("km_indices", None)
        video_indices = batch.get("video_indices", None)
        if km_indices is not None:
            km_indices = km_indices.to(device)
        if video_indices is not None:
            video_indices = video_indices.to(device)

        # Continuous target: max_cobb (added in data_sampler)
        if "max_cobb" not in batch:
            print(f"  [Regression] Batch {batch_idx + 1}: skipped (no max_cobb in batch)")
            continue
        targets = batch["max_cobb"].to(device).view(-1, 1)

        preds = model(
            video,
            knowledge_map,
            texts,
            km_indices=km_indices,
            video_indices=video_indices,
        )

        if torch.isnan(preds).any():
            nan_count = torch.isnan(preds).sum().item()
            print(f"  [Regression] Batch {batch_idx + 1}: skipping {nan_count} NaN predictions")
            continue

        # Ensure shapes are compatible
        if preds.dim() == 1:
            preds = preds.view(-1, 1)

        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        steps += 1
        global_step += 1

        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

        # Batch-level regression metrics
        with torch.no_grad():
            mse = F.mse_loss(preds, targets).item()
            mae = F.l1_loss(preds, targets).item()

        elapsed = time.time() - batch_start
        current_avg_loss = total_loss / max(steps, 1)
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else 0.0
        print(
            f"  [Regression] Batch {batch_idx + 1}/{total_batches} | "
            f"Loss: {loss.item():.6f} | AvgLoss: {current_avg_loss:.6f} | "
            f"MSE: {mse:.6f} | MAE: {mae:.6f} | LR: {current_lr:.6e} | Time: {elapsed:.3f}s"
        )

        if writer is not None:
            writer.add_scalar("Loss/train_reg", loss.item(), global_step)
            writer.add_scalar("Metrics/train_reg/mse_batch", mse, global_step)
            writer.add_scalar("Metrics/train_reg/mae_batch", mae, global_step)

    metrics: Dict[str, float] = {}
    if all_preds:
        all_preds_t = torch.cat(all_preds, dim=0)
        all_targets_t = torch.cat(all_targets, dim=0)
        with torch.no_grad():
            mse = F.mse_loss(all_preds_t, all_targets_t).item()
            mae = F.l1_loss(all_preds_t, all_targets_t).item()
        metrics = {"mse": mse, "mae": mae}
        if writer is not None:
            writer.add_scalar("Metrics/train_reg/mse_epoch", mse, global_step)
            writer.add_scalar("Metrics/train_reg/mae_epoch", mae, global_step)

    avg_loss = total_loss / max(steps, 1)
    if writer is not None:
        writer.add_scalar("Loss/train_reg_epoch", avg_loss, global_step)

    print("  [Regression] Training Summary:")
    print(f"    Batches processed: {steps}/{total_batches}")
    print(f"    Avg Loss: {avg_loss:.6f}")
    if metrics:
        print(f"    MSE: {metrics['mse']:.6f} | MAE: {metrics['mae']:.6f}")

    return avg_loss, metrics, global_step


@torch.no_grad()
def eval_epoch_regression(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    writer: Optional[Any] = None,
    global_step: int = 0,
    verbose: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluation loop for continuous max_cobb regression.
    Does not affect existing classification eval.
    """
    model.eval()
    total_loss = 0.0
    steps = 0
    total_batches = len(dataloader)
    if verbose:
        print(f"  [Regression] Starting evaluation: {total_batches} batches")

    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()

        video = batch["video"].to(device)
        knowledge_map = batch["knowledge_map"].to(device)
        texts = batch.get("texts", None)
        km_indices = batch.get("km_indices", None)
        video_indices = batch.get("video_indices", None)
        if km_indices is not None:
            km_indices = km_indices.to(device)
        if video_indices is not None:
            video_indices = video_indices.to(device)

        if "max_cobb" not in batch:
            if verbose:
                print(f"  [Regression] Batch {batch_idx + 1}: skipped (no max_cobb in batch)")
            continue
        targets = batch["max_cobb"].to(device).view(-1, 1)

        preds = model(
            video,
            knowledge_map,
            texts,
            km_indices=km_indices,
            video_indices=video_indices,
        )

        if torch.isnan(preds).any():
            nan_count = torch.isnan(preds).sum().item()
            if verbose:
                print(f"  [Regression] Batch {batch_idx + 1}: skipping {nan_count} NaN predictions")
            continue

        if preds.dim() == 1:
            preds = preds.view(-1, 1)

        loss = criterion(preds, targets)
        total_loss += loss.item()
        steps += 1

        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

        with torch.no_grad():
            mse = F.mse_loss(preds, targets).item()
            mae = F.l1_loss(preds, targets).item()

        if verbose:
            elapsed = time.time() - batch_start
            print(
                f"  [Regression] Batch {batch_idx + 1}/{total_batches} | "
                f"Loss: {loss.item():.6f} | MSE: {mse:.6f} | MAE: {mae:.6f} | Time: {elapsed:.3f}s"
            )

    metrics: Dict[str, float] = {}
    if all_preds:
        all_preds_t = torch.cat(all_preds, dim=0)
        all_targets_t = torch.cat(all_targets, dim=0)
        with torch.no_grad():
            mse = F.mse_loss(all_preds_t, all_targets_t).item()
            mae = F.l1_loss(all_preds_t, all_targets_t).item()
        metrics = {"mse": mse, "mae": mae}
        if writer is not None:
            writer.add_scalar("Metrics/val_reg/mse_epoch", mse, global_step)
            writer.add_scalar("Metrics/val_reg/mae_epoch", mae, global_step)

    avg_loss = total_loss / max(steps, 1)
    if writer is not None:
        writer.add_scalar("Loss/val_reg_epoch", avg_loss, global_step)

    if verbose:
        print("  [Regression] Evaluation Summary:")
        print(f"    Batches processed: {steps}/{total_batches}")
        print(f"    Avg Loss: {avg_loss:.6f}")
        if metrics:
            print(f"    MSE: {metrics['mse']:.6f} | MAE: {metrics['mae']:.6f}")

    return avg_loss, metrics


def main_regression():
    """
    K-fold training entrypoint for continuous Cobb angle regression.
    Reuses the same model and data pipeline but trains on max_cobb instead of binary labels.
    """
    config: Dict[str, Any] = {
        **{
            "n_folds": 5,
            "start_fold": 1,
            "resume_ckpt_base": None,
            "random_seed": 42,
            "use_preprocessed_pkl": True,
            "pkl_data_dir": ["./data/train_sz_pkl^1", "./data/test_sz_pkl^1"],
            "external_test_dir": ["./data/test_dk_pkl^1"],
            "prompts_path": r"C:\Users\Administrator\project\data\sz_general_gait_prompts.json",
            "prompt_selection": "concise_prompts",
            "km_feature_dim": 238,
            "km_gaussian_noise_std": 0.2,
            "batch_size": 8,
            "num_epochs": 100,
            "learning_rate": 1e-4,
            "warmup_ratio": 0.1,
            "num_workers": 2,
            "weight_decay": 0.1,
            "optimizer": "adamw",
            "loss_type": "mse",  # informational only for regression
            "max_grad_norm": None,
            "hidden_dim": 512,
            "label_dim": 1,
            "video_target_size": (224, 224),
            "patch_size": 96,
            "video_frame_count": 32,
            "video_encoder_type": "vivit",
            "video_encoder_kwargs": {
                "img_size": 224,
                "patch_size": 16,
                "temporal_size": 32,
                "in_channels": 3,
                "depth": 4,
                "num_heads": 8,
                "mlp_ratio": 4.0,
                "drop_rate": 0.2,
                "attn_drop_rate": 0.2,
            },
            "km_encoder_type": "patch_vit",
            "km_encoder_kwargs": {
                "depth": 4,
                "num_heads": 8,
                "mlp_ratio": 4.0,
                "drop": 0.2,
                "attn_drop": 0.2,
            },
            "text_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "text_max_length": 128,
            "text_trainable": False,
            "use_text": True,
            "use_latent_pooling": False,
            "latent_pool_size": 1,
            "regressor_dropout": 0.2,
            "use_km_video_cross_attn": True,
            "cross_attn_num_heads": 8,
            "cross_attn_drop": 0.1,
            "use_gated_token_pooling": True,
            "pretrained_checkpoint_path": None,
            "gpu_ids": [0, 1],
            "use_distributed": False,
            "run_name": "kfold5Cobb_kvt_4layer8_512_1e4",
            "save_dir": "./checkpoints",
            "log_dir": "./logs",
            "debug_print_labels": False,
        }
    }

    gpu_ids = config.get("gpu_ids")
    device = resolve_device(gpu_ids)
    print(f"[Regression] Primary device: {device}")

    if gpu_ids is not None and len(gpu_ids) > 1:
        num_gpus = len(gpu_ids)
    elif gpu_ids is None and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1
    effective_batch_size = config["batch_size"] * num_gpus
    print(f"[Regression] Effective batch size: {effective_batch_size} ({config['batch_size']} x {num_gpus} GPU(s))")

    # Load datasets (same as classification main)
    pkl_entries = config["pkl_data_dir"]
    if isinstance(pkl_entries, (str, bytes)):
        pkl_entries = [pkl_entries]

    datasets = []
    total_patches = 0
    for idx, entry in enumerate(pkl_entries):
        pkl_data_dir = os.path.abspath(os.path.normpath(entry))
        print(f"  [Regression] PKL dir [{idx}]: {pkl_data_dir}")
        ds = SigLIPFullGaitDatasetPKL(
            pkl_data_dir=pkl_data_dir,
            km_gaussian_noise_std=config.get("km_gaussian_noise_std", 0.2),
            mode="train",
            prompts_path=config.get("prompts_path"),
            prompt_selection=config.get("prompt_selection", "concise_prompts"),
            binary_threshold=None,  # do not recompute binary labels
        )
        print(f"    -> {len(ds)} patches")
        datasets.append(ds)
        total_patches += len(ds)

    full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[Regression] Total patches loaded: {total_patches}")

    # No external test set by default for regression (can be added analogously if desired)
    ext_test_dataset = None

    # Build subject-level map using binary labels from existing helper (for stratification only)
    binary_threshold = 15.0
    print("\n[Regression] Building subject-level map for stratified k-fold...")
    subject_ids, subject_labels, subject_to_indices = build_subject_map(
        full_dataset, binary_threshold
    )
    n_subjects = len(subject_ids)
    n_pos = int(subject_labels.sum())
    n_neg = n_subjects - n_pos
    print(f"  Unique subjects: {n_subjects}")
    print(f"  Positive subjects (label=1): {n_pos}")
    print(f"  Negative subjects (label=0): {n_neg}")

    n_folds = config["n_folds"]
    start_fold = config.get("start_fold", 1)
    seed = config["random_seed"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = config.get("run_name", "kfold_regression")
    base_ckpt_dir = os.path.join(config["save_dir"], f"{run_name}_{timestamp}")
    base_log_dir = os.path.join(config.get("log_dir", "./logs"), f"{run_name}_{timestamp}")
    os.makedirs(base_ckpt_dir, exist_ok=True)
    os.makedirs(base_log_dir, exist_ok=True)
    print(f"\n[Regression] Checkpoint base: {base_ckpt_dir}")
    print(f"[Regression] Log base:        {base_log_dir}")

    all_fold_results: List[Dict[str, Any]] = []
    subject_ids_arr = np.array(subject_ids)

    for fold_idx, (train_subj_indices, val_subj_indices) in enumerate(
        skf.split(subject_ids_arr, subject_labels)
    ):
        fold_num = fold_idx + 1
        if fold_num < start_fold:
            print(f"[Regression] Skipping fold {fold_num} (< start_fold={start_fold})")
            continue

        train_subjects = set(subject_ids_arr[train_subj_indices])
        val_subjects = set(subject_ids_arr[val_subj_indices])
        train_patch_indices = []
        val_patch_indices = []
        for sid in train_subjects:
            train_patch_indices.extend(subject_to_indices[sid])
        for sid in val_subjects:
            val_patch_indices.extend(subject_to_indices[sid])

        print(f"\n{'='*70}")
        print(f"[Regression]  FOLD {fold_num}/{n_folds}")
        print(f"{'='*70}")
        print(f"  Train subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)}")
        print(f"  Train patches:  {len(train_patch_indices)} | Val patches: {len(val_patch_indices)}")

        train_dataset = Subset(full_dataset, train_patch_indices)
        val_dataset = Subset(full_dataset, val_patch_indices)

        # DataLoaders
        num_workers = config["num_workers"] if sys.platform != 'win32' else min(config["num_workers"], 8)
        pin_memory = device.type == "cuda" and sys.platform != 'win32'

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=fullgait_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=fullgait_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        # Model and optimizer/scheduler
        model = SFTRegressor(
            km_feature_dim=config["km_feature_dim"],
            hidden_dim=config["hidden_dim"],
            label_dim=config["label_dim"],
            video_encoder_type=config.get("video_encoder_type", "vivit"),
            video_encoder_kwargs=config.get("video_encoder_kwargs"),
            km_encoder_type=config.get("km_encoder_type", "baseline"),
            km_encoder_kwargs=config.get("km_encoder_kwargs"),
            text_model_name=config.get("text_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            text_max_length=config.get("text_max_length", 128),
            text_trainable=config.get("text_trainable", True),
            use_text=config.get("use_text", True),
            use_latent_pooling=config.get("use_latent_pooling", False),
            latent_pool_size=config.get("latent_pool_size", 1),
            regressor_dropout=config.get("regressor_dropout", 0.1),
            use_km_video_cross_attn=config.get("use_km_video_cross_attn", True),
            cross_attn_num_heads=config.get("cross_attn_num_heads", 8),
            cross_attn_drop=config.get("cross_attn_drop", 0.1),
            use_gated_token_pooling=config.get("use_gated_token_pooling", True),
        )

        if config.get("pretrained_checkpoint_path"):
            model = load_pretrained_checkpoint(
                model, config["pretrained_checkpoint_path"], device,
                text_trainable=config.get("text_trainable", True),
            )

        model = setup_multi_gpu(model, gpu_ids=gpu_ids, use_distributed=config.get("use_distributed", False))
        if fold_idx == 0:
            print_model_info(model, config)

        # Regression criterion
        criterion = nn.MSELoss()

        steps_per_epoch = max(1, len(train_loader))
        optimizer, scheduler = create_optimizer_and_scheduler(model, config, steps_per_epoch)

        fold_ckpt_dir = os.path.join(base_ckpt_dir, f"fold_{fold_num}")
        fold_log_dir = os.path.join(base_log_dir, f"fold_{fold_num}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)
        os.makedirs(fold_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=fold_log_dir)

        best_val_rmse = float("inf")
        best_val_mae = float("inf")
        global_step = 0

        for epoch in range(1, config["num_epochs"] + 1):
            epoch_start = time.time()

            train_loss, train_metrics, global_step = train_one_epoch_regression(
                model, train_loader, optimizer, criterion, device,
                scheduler, writer, global_step, config=config,
            )

            val_loss, val_metrics = eval_epoch_regression(
                model, val_loader, criterion, device,
                writer, global_step, verbose=True, config=config,
            )

            val_rmse = float(np.sqrt(val_metrics.get("mse", val_loss))) if val_metrics else float("inf")
            val_mae = float(val_metrics.get("mae", val_loss)) if val_metrics else float("inf")
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - epoch_start

            print(
                f"[Regression Fold {fold_num} Epoch {epoch:03d}] "
                f"TrainLoss: {train_loss:.6f} | ValLoss: {val_loss:.6f} | "
                f"ValRMSE: {val_rmse:.6f} | ValMAE: {val_mae:.6f} | "
                f"BestRMSE: {best_val_rmse:.6f} | BestMAE: {best_val_mae:.6f} | "
                f"LR: {current_lr:.2e} | {elapsed:.1f}s"
            )

            improved = False
            if val_rmse <= best_val_rmse:
                best_val_rmse = val_rmse
                improved = True
            if val_mae <= best_val_mae:
                best_val_mae = val_mae
                improved = True

            if improved:
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    os.path.join(fold_ckpt_dir, "checkpoint_best_regression.pth"),
                    config, best_acc=0.0, best_auc=0.0,
                )

            # Save last checkpoint each epoch
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                os.path.join(fold_ckpt_dir, "checkpoint_last_regression.pth"),
                config, best_acc=0.0, best_auc=0.0,
            )

        writer.close()

        fold_result = {
            "fold": fold_num,
            "best_val_rmse": best_val_rmse,
            "best_val_mae": best_val_mae,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "train_subjects": len(train_subjects),
            "val_subjects": len(val_subjects),
            "train_patches": len(train_patch_indices),
            "val_patches": len(val_patch_indices),
        }
        all_fold_results.append(fold_result)
        print(f"\n  [Regression] Fold {fold_num} done: best_rmse={best_val_rmse:.6f}, best_mae={best_val_mae:.6f}")

    # Aggregate regression results across folds
    print(f"\n{'='*70}")
    print(f"  REGRESSION K-FOLD SUMMARY ({n_folds} folds)")
    print(f"{'='*70}\n")

    def _collect_metric_reg(key_name: str) -> List[float]:
        vals: List[float] = []
        for r in all_fold_results:
            if key_name in r and r[key_name] is not None:
                try:
                    vals.append(float(r[key_name]))
                except (TypeError, ValueError):
                    continue
        return vals

    summary: Dict[str, Any] = {"n_folds": n_folds, "n_subjects": n_subjects, "folds": all_fold_results}
    for key, display_name in [
        ("best_val_rmse", "Val RMSE (best)"),
        ("best_val_mae", "Val MAE (best)"),
    ]:
        values = _collect_metric_reg(key)
        if not values:
            print(f"  {display_name:25s}: (missing)")
            continue
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci95 = 1.96 * std_val / np.sqrt(len(values)) if len(values) > 1 else 0.0
        summary[f"{key}_mean"] = mean_val
        summary[f"{key}_std"] = std_val
        summary[f"{key}_ci95"] = ci95
        print(f"  {display_name:25s}: {mean_val:.4f} +/- {std_val:.4f}  "
              f"(95% CI: {mean_val-ci95:.4f} - {mean_val+ci95:.4f})")

    # Per-fold table
    print("\nPer-fold regression results:")
    header = f"  {'Fold':>4s}  {'BestRMSE':>10s}  {'BestMAE':>10s}  {'#Train':>6s}  {'#Val':>6s}"
    print(header)
    for r in all_fold_results:
        print(
            f"  {r['fold']:4d}  {r['best_val_rmse']:10.4f}  {r['best_val_mae']:10.4f}  "
            f"{r['train_patches']:6d}  {r['val_patches']:6d}"
        )

    summary_path = os.path.join(base_ckpt_dir, "kfold_regression_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Regression] Summary saved to: {summary_path}")

    config_path = os.path.join(base_ckpt_dir, "config_regression.json")
    config_serializable = {}
    for k, v in config.items():
        try:
            json.dumps(v)
            config_serializable[k] = v
        except (TypeError, ValueError):
            config_serializable[k] = str(v)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_serializable, f, indent=2)

    print(f"\n[Regression] Finished {n_folds}-fold regression training.")


if __name__ == "__main__":
    # Default behaviour: keep existing binary classification.
    # To run regression on max_cobb, set environment variable COBB_TASK=regression.
    task = "regression" # os.environ.get("COBB_TASK", "classification").lower()
    if task == "regression":
        main_regression()
    else:
        main()
