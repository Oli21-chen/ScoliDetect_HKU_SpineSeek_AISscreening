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
from typing import Dict, List, Any, Optional

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
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold

from utils.data_sampler import (
    SigLIPFullGaitDatasetPKL,
    fullgait_collate_fn,
    make_fullgait_train_collate_with_km_noise,
)
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
from models.sft_regressor import SFTRegressor #SFTRegressor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config: Dict[str, Any] = {
        # ---- K-fold settings ----
        "n_folds": 10,
        "start_fold": 1,  # Set to N to skip folds 1..N-1 and resume from fold N (1 = run all)
        # Inclusive upper bound. Use start_fold=3 and end_fold=3 to retrain only fold 3.
        # None means run through n_folds (default full CV).
        "end_fold": 1,
        "resume_ckpt_base": None,
        "random_seed": 42,

        # ---- Data ----
        "use_preprocessed_pkl": True,
        "pkl_data_dir": ["./data/train_sz_pkl^1","./data/test_sz_pkl^1"],
        "external_test_dir": ["./data/test_dk_pkl^1"],  # External cohort for independent validation (evaluated after each fold)
        "prompts_path": r"/root/private_data/Dong_project/data/general_gait_prompts_from_report.json",
        "prompt_selection": "concise_prompts",
        "km_feature_dim": 238,
        "binary_threshold": 11.0,
        # Applied only in **train** DataLoader collate (clean KM in dataset for CV val).
        "km_gaussian_noise_std": 0.1,
        # Train-only video augmentation (no flip): brightness/contrast jitter and random crop.
        "video_aug": True,
        "video_brightness_contrast_jitter": 0.1,
        "video_random_crop_scale": (1.0, 1.0),

        # ---- Training ----
        "batch_size": 8,
        "num_epochs": 100,
        "patience": 50,  # early stop if val AUC does not improve for N epochs
        "learning_rate": 5e-6,
        "warmup_ratio": 0.1,
        "num_workers": 4,
        "weight_decay": 0.1,
        "optimizer": "adamw",
        "loss_type": "focal",
        "focal_gamma": 2.0,
        "focal_alpha": 0.25,
        "focal_label_smoothing": 0.05,
        "focal_logit_reg": 1e-4,
        "focal_use_pos_weight": True,
        "max_grad_norm": None,

        # ---- Model ----
        "hidden_dim": 512,
        "label_dim": 1,
        "video_target_size": (224, 224),
        "patch_size": 96,
        "video_frame_count": 32,
        # Publishable 3D-CNN baseline (ResNet3D family) with parameter matching.
        # - torchvision `r3d_18` is ~33M params; this scaled ResNet3D keeps the same design but narrower.
        # - `base_channels=48` gives ~18.9M params, close to TimeSformer-512 depth-4 (~17.2M).
        "video_encoder_type": "r2plus1d_18",#"r2plus1d_18","r3d_18"
        "video_encoder_kwargs": {
            "pretrained": True,
        },
        
        # "video_encoder_type": "resnet3d_scaled",
        # "video_encoder_kwargs": {
        #     "base_channels": 48,
        # },
        "km_encoder_type": "vit",
        "km_encoder_kwargs": {
            "depth": 4,
            "num_heads": 8,
            "mlp_ratio": 2.0,
            "drop": 0.2,
            "attn_drop": 0.2,
        },
        "text_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "text_max_length": 128,
        "text_trainable": False,
        "use_text": True,
        "use_latent_pooling": False,
        "latent_pool_size": 1,
        "regressor_dropout": 0.5,
        "use_km_video_cross_attn": True,
        "cross_attn_num_heads": 8,
        "cross_attn_drop": 0.1,
        "cross_attn_num_layers": 2,
        "temporal_attn_bias_strength": 1.0,
        "modality_dropout_prob": 0.1,
        "use_bottleneck_fusion": True,
        "bottleneck_tokens": 32,
        "bottleneck_layers": 1,
        "align_loss_weight": 0.05,
        "align_loss_temperature": 0.07,
        # Auxiliary multimodal fusion loss:
        #   "infonce" (existing), "barlow", "vicreg", "hybrid" (InfoNCE + VICReg)
        "aux_loss_type": "infonce",
        # Projection dim before auxiliary loss (used by all aux types).
        "aux_proj_dim": 256,
        # Barlow Twins hyperparameter (off-diagonal weight).
        "barlow_lambda": 5e-3,
        # VICReg hyperparameters.
        "vicreg_sim_coeff": 25.0,
        "vicreg_var_coeff": 25.0,
        "vicreg_cov_coeff": 1.0,
        "vicreg_var_target": 1.0,
        "vicreg_eps": 1e-4,
        "use_gated_token_pooling": True,

        "pretrained_checkpoint_path": None,
        # When loading pretrained ckpt: do not copy text_encoder.* (keep sentence-transformer defaults).
        "load_text_encoder_weights_from_pretrained": True,
        # If True: load text_encoder.* from ckpt. text_proj + text_norm are still skipped unless True below.
        "load_text_proj_from_pretrained": False,
        "gpu_ids": [0, 1],
        "use_distributed": False,

        "run_name": "kfold5_kvt_r2plus1d",
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
    # Load full dataset (no KM Gaussian noise here — val fold is a Subset of this pool)
    # ------------------------------------------------------------------
    pkl_entries = config["pkl_data_dir"]
    if isinstance(pkl_entries, (str, bytes)):
        pkl_entries = [pkl_entries]

    km_noise_train = config.get("km_gaussian_noise_std")
    use_video_aug = bool(config.get("video_aug", False))
    train_collate_fn = make_fullgait_train_collate_with_km_noise(
        km_gaussian_noise_std=km_noise_train,
        video_brightness_contrast_jitter=(
            config.get("video_brightness_contrast_jitter", 0.0) if use_video_aug else 0.0
        ),
        video_random_crop_scale=(
            config.get("video_random_crop_scale") if use_video_aug else None
        ),
    )

    datasets = []
    total_patches = 0
    for idx, entry in enumerate(pkl_entries):
        pkl_data_dir = os.path.abspath(os.path.normpath(entry))
        print(f"  PKL dir [{idx}]: {pkl_data_dir}")
        ds = SigLIPFullGaitDatasetPKL(
            pkl_data_dir=pkl_data_dir,
            km_gaussian_noise_std=None,  # clean KM; noise added in train collate only
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
    print(
        f"  K-fold: KM Gaussian noise disabled in dataset; "
        f"train batches use collate std = {km_noise_train!r}"
    )

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
    start_fold = int(config.get("start_fold", 1))
    end_fold = config.get("end_fold", None)
    if end_fold is None:
        end_fold = n_folds
    else:
        end_fold = int(end_fold)
    if start_fold < 1 or start_fold > n_folds:
        raise ValueError(f"start_fold must be in [1, {n_folds}], got {start_fold}")
    if end_fold < 1 or end_fold > n_folds:
        raise ValueError(f"end_fold must be in [1, {n_folds}], got {end_fold}")
    if start_fold > end_fold:
        raise ValueError(f"start_fold ({start_fold}) must be <= end_fold ({end_fold})")
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
    if start_fold > 1 or end_fold < n_folds:
        print(
            f"Fold range: training folds {start_fold}..{end_fold} "
            f"(n_folds={n_folds}; folds outside this range load from checkpoint if present)"
        )

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

        # --- Skip folds outside [start_fold, end_fold]: load results from saved checkpoint if any ---
        if fold_num < start_fold or fold_num > end_fold:
            fold_ckpt_dir = os.path.join(base_ckpt_dir, f"fold_{fold_num}")
            best_ckpt = os.path.join(fold_ckpt_dir, "checkpoint_best_auc.pth")
            if os.path.exists(best_ckpt):
                ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
                prev_best_acc = ckpt.get("best_acc", 0.0)
                prev_best_auc = ckpt.get("best_auc", 0.0)
                reason = "before start_fold" if fold_num < start_fold else "after end_fold (not retraining)"
                print(
                    f"  [Fold {fold_num}] Skipped ({reason}) — loaded best_acc={prev_best_acc:.4f}, best_auc={prev_best_auc:.4f}"
                )
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
                print(
                    f"  [Fold {fold_num}] Skipped (outside {start_fold}..{end_fold}) — "
                    f"no checkpoint at {best_ckpt} — no results recorded"
                )
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
            collate_fn=train_collate_fn,
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
            cross_attn_num_layers=config.get("cross_attn_num_layers", 2),
            temporal_attn_bias_strength=config.get("temporal_attn_bias_strength", 0.0),
            modality_dropout_prob=config.get("modality_dropout_prob", 0.0),
            use_bottleneck_fusion=config.get("use_bottleneck_fusion", False),
            bottleneck_tokens=config.get("bottleneck_tokens", 16),
            bottleneck_layers=config.get("bottleneck_layers", 1),
            align_loss_weight=config.get("align_loss_weight", 0.0),
            align_loss_temperature=config.get("align_loss_temperature", 0.07),
            aux_loss_type=config.get("aux_loss_type", "infonce"),
            aux_proj_dim=config.get("aux_proj_dim", 256),
            barlow_lambda=config.get("barlow_lambda", 5e-3),
            vicreg_sim_coeff=config.get("vicreg_sim_coeff", 25.0),
            vicreg_var_coeff=config.get("vicreg_var_coeff", 25.0),
            vicreg_cov_coeff=config.get("vicreg_cov_coeff", 1.0),
            vicreg_var_target=config.get("vicreg_var_target", 1.0),
            vicreg_eps=config.get("vicreg_eps", 1e-4),
            use_gated_token_pooling=config.get("use_gated_token_pooling", True),
        )

        if config.get("pretrained_checkpoint_path"):
            model = load_pretrained_checkpoint(
                model, config["pretrained_checkpoint_path"], device,
                text_trainable=config.get("text_trainable", True),
                load_text_encoder_weights=config.get("load_text_encoder_weights_from_pretrained", False),
                load_text_proj_from_pretrained=config.get("load_text_proj_from_pretrained", False),
            )

        model = setup_multi_gpu(model, gpu_ids=gpu_ids, use_distributed=config.get("use_distributed", False))
        # Print architecture once on the first fold we actually train (not necessarily fold 1)
        if fold_num == start_fold:
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
            focal_gamma=config.get("focal_gamma", 2.0),
            focal_alpha=config.get("focal_alpha", 0.25),
            focal_label_smoothing=config.get("focal_label_smoothing", 0.05),
            focal_logit_reg=config.get("focal_logit_reg", 1e-4),
            focal_use_pos_weight=config.get("focal_use_pos_weight", False),
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
        # Support both "patience" and legacy alias "early_stop_patience"
        early_stop_patience = int(config.get("patience", config.get("early_stop_patience", 0)) or 0)
        epochs_no_improve = 0
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
                epochs_no_improve = 0
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    os.path.join(fold_ckpt_dir, "checkpoint_best_auc.pth"),
                    fold_config, best_acc=best_acc, best_auc=best_auc,
                )
            else:
                epochs_no_improve += 1

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Save last checkpoint each epoch (for potential resume)
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                os.path.join(fold_ckpt_dir, "checkpoint_last.pth"),
                fold_config, best_acc=best_acc, best_auc=best_auc,
            )

            if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                print(
                    f"  Early stopping at epoch {epoch}: "
                    f"val AUC has not improved for {epochs_no_improve} epochs "
                    f"(patience={early_stop_patience})."
                )
                break

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


if __name__ == "__main__":
    main()
