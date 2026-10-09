"""
Supervised fine-tuning entrypoint for multimodal SigLIP using fullgait data.
Training script for sz_table_refinedyolo and sz_video_refinedyolo datasets.
"""

import os
import sys
import time
import warnings
import multiprocessing
from datetime import datetime
from typing import Dict

# Suppress NCCL warning on Windows (NCCL not supported, but we use DataParallel/gloo instead)
warnings.filterwarnings("ignore", message="PyTorch is not compiled with NCCL support")

# Fix Windows multiprocessing: use 'spawn' instead of 'fork'
if sys.platform == 'win32':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from utils.data_sampler import SigLIPFullGaitDataset_v2, SigLIPFullGaitDatasetPKL, fullgait_collate_fn
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
    setup_directories,
    load_resume_checkpoint,
)
from models.sft_regressorv2 import SFTRegressor


def main():
    # Configuration for fullgait training
    config: Dict = {
        # Data loading options
        "use_preprocessed_pkl": True,  # Set to True to use preprocessed pickle files, False to load on-the-fly
        "pkl_data_dir": ["./data/train_sz_pkl^1","./data/test_dk_pkl","./data/test_sz_pkl^1"],  # Train: directory/ies for preprocessed pickle files (if use_preprocessed_pkl=True)
        "val_data_path": None,  # If set (str or list of paths), use as separate validation set; train on full pkl_data_dir. If None, use val_split on combined data.
        "test_data_path":None,  # If set (str or list of paths), use as external test set; evaluate each epoch and save best_test_acc / best_test_auc checkpoints. If None, skip.
        "table_dir": r"C:\Users\Administrator\project\data\sz_table_refinedyolo",  # Knowledge map CSV files directory (if use_preprocessed_pkl=False)
        "video_dir": r"C:\Users\Administrator\project\data\sz_video_refinedyolo",  # Video files directory (if use_preprocessed_pkl=False)
        "label_json_path": r"data\train_indices.json",  # Path to train/test indices JSON
        "split": "train",  # "train" or "test" - only needed if label_json_path is provided
        # Note: fullgait dataset has flat structure, no direction subdirectories
        "video_target_size": (224, 224),
        "patch_size": 96,
        "video_frame_count": 32,
        "batch_size": 8,
        "num_epochs": 100,
        "learning_rate": 8e-6,  # Lower than pretrain (1e-4) for fine-tuning stability
        "warmup_ratio": 0.1,  # 10% of total steps for warmup
        "num_workers": 2,  # Can use more workers with preprocessed data (if use_preprocessed_pkl=True)
        "train_ratio": 0.7,  # Train : val : test = 7 : 2 : 1
        "val_ratio": 0.2,
        "test_ratio": 0.1,
        "weight_decay": 0.1,  # Used by AdamW; for Adam often 0 or small (e.g. 1e-5)
        "optimizer": "adamw",  # "adamw" (default) or "adam"
        "random_seed": 42,  # Random seed for train/val split reproducibility
        "prompts_path": r"C:\Users\Administrator\project\data\sz_general_gait_prompts.json",  # Path to prompts JSON (general or individual)
        "prompt_selection": "concise_prompts",  # Options: 'concise_prompts' (general), 'top_feature_prompts', 'forward', 'backward', 'both', 'auto'
        "km_feature_dim": 238,  # knowledge_map feature dimension
        "hidden_dim": 512,
        "label_dim": 1,  # Binary classification (0 or 1)
        "binary_threshold": 15.0,  # Threshold for binary classification: if max(label) >= threshold, label = 1, else 0
        "km_gaussian_noise_std": 0.2,  # Gaussian noise std for augmentation (applied during loading, not baked into pkl)
        "run_name": "e2ev4_kvt_4layer8-6RopeLat_512^1_dk_15d_gnstd0.2_Flatent_focal_adamw_moredrop",  # Prefix for checkpoint and log directory names (e.g., "sft", "run", "exp1")
        "save_dir": "./checkpoints",
        "log_dir": "./logs",  # TensorBoard log directory
        # Encoder configurations (matching pretrain.py)
        "video_encoder_type": "vivit",  # Options: "conv3d", "vivit", "timesformer", etc.
        # Video encoder kwargs (matching pretrain.py)
        "video_encoder_kwargs": {
            "img_size": 224,  # Match video_target_size
            "patch_size": 16,
            "temporal_size": 32,  # Match video_frame_count
            "in_channels": 3,
            "depth": 4,
            "num_heads": 8,  
            "mlp_ratio": 4.0,
            "drop_rate": 0.2,
            "attn_drop_rate": 0.2,
        },
        "km_encoder_type": "vit",  # Options: "baseline", "vit", "patch_vit"
        # Knowledge encoder kwargs (matching pretrain.py)
        "km_encoder_kwargs": {
            "depth": 4,
            "num_heads": 8, 
            "mlp_ratio": 4.0,
            "drop": 0.2,
            "attn_drop": 0.2,
        },
        "text_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "text_max_length": 128, 
        "text_trainable": True,  # Freeze text encoder; only text_proj (projection to hidden_dim) stays trainable
        "use_text":True,  # Set to False to disable text encoder
        "use_latent_pooling": False,  # Set to False to disable latent pooling
        "latent_pool_size": 1,  # Number of latent tokens for attention pooling (when use_latent_pooling=True)
        "regressor_dropout": 0.2,  # Dropout in regressor head to reduce overfitting (0 to disable)
        "use_km_video_cross_attn": True,  # Cross attention between km and video modalities
        "cross_attn_num_heads": 8,
        "cross_attn_drop": 0.1,
        "use_gated_token_pooling": False,  # Gated (sigmoid-weighted) pooling over temporal tokens
        # Pretrained checkpoint path (optional); loads video_encoder, km_encoder, and text_encoder/text_proj when present
        "pretrained_checkpoint_path":None,
        # r"C:\Users\Administrator\project\checkpoints\pretrain_kvt_512_20260205_174659\siglip_checkpoint_best.pth",
        # Resume from fine-tuning checkpoint (optional)
        # Can use: checkpoint_last.pth (saved every epoch), checkpoint_epoch_N.pth (saved every 10 epochs), or checkpoint_best.pth
        "resume_from_checkpoint":None,
        "gpu_ids": [0,1],  # List of GPU IDs to use (e.g., [0, 1, 2, 3] or None for all GPUs)
        "use_distributed": False,  # Set to True for DistributedDataParallel (requires torch.distributed setup)

        "loss_type": "focal",  # "bce" or "focal" (focal + label smoothing helps imbalanced binary)
        "max_grad_norm": 1.0,  # Gradient clipping; None to disable (helps stabilize training)
        "debug_print_labels": False,  # Set True to print labels for each batch during training
    }

    # Setup device and multi-GPU
    gpu_ids = config.get("gpu_ids")
    if gpu_ids is not None:
        print(f"Using GPUs: {gpu_ids}")
    else:
        print("Using all available GPUs")
    
    device = resolve_device(gpu_ids)
    os.makedirs(config["save_dir"], exist_ok=True)

    print(f"Primary device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    # Load dataset (either from preprocessed pickle files or on-the-fly)
    use_preprocessed = config.get("use_preprocessed_pkl", False)
    val_data_path = config.get("val_data_path")  # None or path(s) for separate validation set

    if use_preprocessed:
        print(f"Loading preprocessed dataset from pickle files...")

        # Train: allow either a single directory (str) or multiple directories (list/tuple)
        pkl_entries = config["pkl_data_dir"]
        if isinstance(pkl_entries, (str, bytes)):
            pkl_entries = [pkl_entries]

        train_datasets = []
        train_patches = 0
        for idx, entry in enumerate(pkl_entries):
            pkl_data_dir = os.path.abspath(os.path.normpath(entry))
            print(f"  Train PKL dir [{idx}]: {pkl_data_dir}")

            ds = SigLIPFullGaitDatasetPKL(
                pkl_data_dir=pkl_data_dir,
                km_gaussian_noise_std=config.get("km_gaussian_noise_std", 0.2),
                mode=config.get("split", "train"),
                prompts_path=config.get("prompts_path"),
                prompt_selection=config.get("prompt_selection", "concise_prompts"),
                binary_threshold=config.get("binary_threshold"),
            )
            print(f"    ↳ Loaded {len(ds)} patches")
            train_datasets.append(ds)
            train_patches += len(ds)

        if len(train_datasets) == 1:
            full_dataset = train_datasets[0]
        else:
            full_dataset = ConcatDataset(train_datasets)
        print(f"✅ Loaded {train_patches} patches from {len(train_datasets)} PKL directory(ies)")

        # Split into train : val : test = train_ratio : val_ratio : test_ratio (default 7 : 2 : 1)
        train_ratio = config.get("train_ratio", 0.7)
        val_ratio = config.get("val_ratio", 0.2)
        test_ratio = config.get("test_ratio", 0.1)
        total = len(full_dataset)
        train_size = max(0, int(total * train_ratio))
        val_size = max(0, int(total * val_ratio))
        test_size = total - train_size - val_size  # remainder so sizes sum exactly to total
        if test_size < 0:
            test_size = 0
            val_size = total - train_size
        gen = torch.Generator().manual_seed(config.get("random_seed", 42))
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=gen
        )
        print(f"✅ Train/val/test split (7:2:1): {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    else:
        print(f"Loading fullgait dataset on-the-fly...")
        print(f"  Table dir: {config['table_dir']}")
        print(f"  Video dir: {config['video_dir']}")
        
        dataset = SigLIPFullGaitDataset_v2(
            table_dir=config["table_dir"],
            video_dir=config["video_dir"],
            label_json_path=config.get("label_json_path"),
            split=config.get("split") if config.get("label_json_path") is not None else None,
            directions=None,  # Fullgait dataset has flat structure, no directions
            patch_size=config["patch_size"],
            video_frame_count=config["video_frame_count"],
            video_target_size=config["video_target_size"],
            prompts_path=config.get("prompts_path"),
            prompt_selection=config.get("prompt_selection", "top_feature_prompts"),
            binary_threshold=config.get("binary_threshold", 11.0),
            km_gaussian_noise_std=config.get("km_gaussian_noise_std", 0.2),
            mode=config.get("split", "train"),  # Use "train" mode to split videos into 96-frame chunks, "test" mode uses first 96 frames only
        )
        print(f"✅ Dataset loaded: {len(dataset)} patches")

        # On-the-fly: train : val : test = 7 : 2 : 1
        train_ratio = config.get("train_ratio", 0.7)
        val_ratio = config.get("val_ratio", 0.2)
        test_ratio = config.get("test_ratio", 0.1)
        total = len(dataset)
        train_size = max(0, int(total * train_ratio))
        val_size = max(0, int(total * val_ratio))
        test_size = total - train_size - val_size
        if test_size < 0:
            test_size = 0
            val_size = total - train_size
        gen = torch.Generator().manual_seed(config.get("random_seed", 42))
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=gen
        )
        print(f"✅ Train/val/test split (7:2:1): {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")
    
    # Print label distribution and derive positive-class weight for BCE, if desired
    label_stats = print_label_distribution(
        train_dataset,
        val_dataset,
        config.get("binary_threshold", 11.0),
    )
    train_pos = max(1, label_stats.get("train_pos", 0))
    train_neg = max(0, label_stats.get("train_neg", 0))
    pos_weight = train_neg / float(train_pos)
    config["pos_weight"] = pos_weight
    print(f"Computed pos_weight for BCE: {pos_weight:.4f} (neg/pos in train set)")

    # Determine number of GPUs being used
    if gpu_ids is not None and len(gpu_ids) > 1:
        num_gpus = len(gpu_ids)
    elif gpu_ids is None and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1
    
    # Note: With DataParallel, effective batch size = batch_size * num_gpus
    effective_batch_size = config["batch_size"] * num_gpus
    print(f"Batch size per GPU: {config['batch_size']}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Effective batch size: {effective_batch_size}")

    # Windows compatibility: reduce workers and disable pin_memory if issues occur
    # With preprocessed data, can use more workers safely
    if use_preprocessed:
        num_workers = config["num_workers"] if sys.platform != 'win32' else min(config["num_workers"], 8)
    else:
        num_workers = config["num_workers"] if sys.platform != 'win32' else min(config["num_workers"], 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=fullgait_collate_fn,
        num_workers=num_workers,
        pin_memory=device.type == "cuda" and sys.platform != 'win32',  # Disable on Windows to avoid sharing issues
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,  # Reduced for Windows
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=fullgait_collate_fn,
        num_workers=num_workers,
        pin_memory=device.type == "cuda" and sys.platform != 'win32',  # Disable on Windows to avoid sharing issues
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,  # Reduced for Windows
    )

    # Test loader: from 7:2:1 split (test_ratio) when test_dataset has samples; else optional external test_data_path
    test_loader = None
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=fullgait_collate_fn,
            num_workers=num_workers,
            pin_memory=device.type == "cuda" and sys.platform != 'win32',
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        print(f"✅ Test loader: {len(test_dataset)} samples (from train:val:test split)")
    test_data_path = config.get("test_data_path")
    if test_loader is None and test_data_path is not None and use_preprocessed:
        test_entries = [test_data_path] if isinstance(test_data_path, (str, bytes)) else list(test_data_path)
        test_datasets = []
        test_patches = 0
        for idx, entry in enumerate(test_entries):
            tdir = os.path.abspath(os.path.normpath(entry))
            print(f"  Test PKL dir [{idx}]: {tdir}")
            ds = SigLIPFullGaitDatasetPKL(
                pkl_data_dir=tdir,
                km_gaussian_noise_std=None,  # No augmentation for test
                mode="test",
                prompts_path=config.get("prompts_path"),
                prompt_selection=config.get("prompt_selection", "concise_prompts"),
                binary_threshold=config.get("binary_threshold"),
            )
            print(f"    ↳ Loaded {len(ds)} patches")
            test_datasets.append(ds)
            test_patches += len(ds)
        test_dataset = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]
        print(f"✅ Test dataset: {test_patches} patches from {len(test_datasets)} PKL directory(ies)")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=fullgait_collate_fn,
            num_workers=num_workers,
            pin_memory=device.type == "cuda" and sys.platform != 'win32',
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    elif test_data_path is not None and not use_preprocessed:
        print("⚠️ test_data_path set but use_preprocessed_pkl=False; external test set requires PKL. Skipping test loader.")

    # Create model
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
        text_trainable=config.get("text_trainable", False),
        use_text=config.get("use_text", True),
        use_latent_pooling=config.get("use_latent_pooling", False),
        latent_pool_size=config.get("latent_pool_size", 1),
        regressor_dropout=config.get("regressor_dropout", 0.1),
        use_km_video_cross_attn=config.get("use_km_video_cross_attn", True),
        cross_attn_num_heads=config.get("cross_attn_num_heads", 8),
        cross_attn_drop=config.get("cross_attn_drop", 0.1),
        use_gated_token_pooling=config.get("use_gated_token_pooling", True),
    )
    
    # Load pretrained checkpoint if specified
    pretrained_checkpoint_path = config.get("pretrained_checkpoint_path")
    if pretrained_checkpoint_path:
        model = load_pretrained_checkpoint(
            model, pretrained_checkpoint_path, device,
            text_trainable=config.get("text_trainable", True),
        )
    
    # Setup multi-GPU if specified
    use_distributed = config.get("use_distributed", False)
    model = setup_multi_gpu(model, gpu_ids=gpu_ids, use_distributed=use_distributed)
    
    # Print model info
    print_model_info(model, config)

    # Peek one batch to choose loss (criterion). Note: this consumes the first batch of
    # train_loader, so the first training epoch will process (total_batches - 1) batches.
    sample_batch = next(iter(train_loader))
    sample_label = sample_batch.get("label")
    criterion = pick_criterion(
        sample_label,
        label_dim=config["label_dim"],
        loss_type=config.get("loss_type", "bce"),
        pos_weight=config.get("pos_weight"),
        device=device,
    )
    
    if criterion is None:
        print("Warning: No labels found. Training will skip batches without labels.")
    else:
        print(f"Using criterion: {criterion.__class__.__name__}")

    # Create optimizer and scheduler
    steps_per_epoch = max(1, len(train_loader))
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, steps_per_epoch)

    # Setup directories
    ckpt_dir, log_dir = setup_directories(config)
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize training state and load resume checkpoint if specified
    resume_checkpoint_path = config.get("resume_from_checkpoint")
    if resume_checkpoint_path:
        start_epoch, global_step, best_val_loss, best_acc, best_auc, best_test_acc, best_test_auc = load_resume_checkpoint(
            model, optimizer, scheduler, resume_checkpoint_path, device, steps_per_epoch,
            text_trainable=config.get("text_trainable", True),
        )
    else:
        start_epoch = 1
        global_step = 0
        best_val_loss = float('inf')
        best_acc = 0.0  # Track best validation accuracy
        best_auc = 0.0  # Track best validation AUC-ROC
        best_test_acc = 0.0  # Track best external test accuracy (when test_data_path is set)
        best_test_auc = 0.0  # Track best external test AUC-ROC (when test_data_path is set)
    
    for epoch in range(start_epoch, config["num_epochs"] + 1):
        epoch_start_time = time.time()
        
        # Training phase
        train_start_time = time.time()
        train_loss, train_metrics, global_step = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device, 
            scheduler, 
            writer, 
            global_step, 
            config=config
        )
        train_duration = time.time() - train_start_time
        
        # Validation phase
        val_start_time = time.time()
        val_loss, val_metrics = eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            writer,
            global_step,
            config=config,
        )
        val_duration = time.time() - val_start_time

        # Optional: evaluate on external test set
        test_acc, test_auc = None, None
        if test_loader is not None:
            test_loss, test_metrics = eval_epoch(
                model, test_loader, criterion, device, writer, global_step,
                verbose=False, config=config,
            )
            test_acc = test_metrics.get('accuracy', 0.0)
            test_auc = test_metrics.get('auc_roc', 0.0)
            if writer:
                writer.add_scalar("Metrics/test/accuracy", test_acc, global_step)
                writer.add_scalar("Metrics/test/auc_roc", test_auc, global_step)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        current_lr = scheduler.get_last_lr()[0]
        
        # Get validation metrics for tracking best model
        val_acc = val_metrics.get('accuracy', 0.0)
        val_auc = val_metrics.get('auc_roc', 0.0)
        
        # Print metrics with timing (emphasize AUC instead of F1/Recall)
        print_line = (
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Train Acc: {train_metrics.get('accuracy', 0.0):.4f} | "
            f"Train AUC: {train_metrics.get('auc_roc', 0.0):.4f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Best Acc: {best_acc:.4f} | "
            f"Best AUC: {best_auc:.4f}"
        )
        if test_loader is not None and test_acc is not None and test_auc is not None:
            print_line += f" | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f} | Best Test Acc: {best_test_acc:.4f} | Best Test AUC: {best_test_auc:.4f}"
        print_line += f" | LR: {current_lr:.6e} | Time: {epoch_duration:.2f}s (Train: {train_duration:.2f}s, Val: {val_duration:.2f}s)"
        print(print_line)
        
        # Log epoch times to TensorBoard
        if writer:
            writer.add_scalar("Time/epoch_duration", epoch_duration, epoch)
            writer.add_scalar("Time/train_duration", train_duration, epoch)
            writer.add_scalar("Time/val_duration", val_duration, epoch)

        # Update best accuracy and save best model based on accuracy
        is_best = val_acc >= best_acc
        if is_best:
            previous_best_acc = best_acc
            best_acc = val_acc
            best_ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                best_ckpt_path,
                config,
                best_acc=best_acc,
                best_auc=best_auc,
                best_test_acc=best_test_acc,
                best_test_auc=best_test_auc,
            )
            print(f"🏆 New best accuracy! Saved: {best_ckpt_path}")
            print(f"   Previous best Acc: {previous_best_acc:.4f} → New best Acc: {val_acc:.4f}")
            print(f"   Validation Loss: {val_loss:.6f}")

        # Update best AUC-ROC and save separate best-AUC checkpoint
        is_best_auc = val_auc >= best_auc
        if is_best_auc:
            previous_best_auc = best_auc
            best_auc = val_auc
            best_auc_ckpt_path = os.path.join(ckpt_dir, "checkpoint_best_auc.pth")
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                best_auc_ckpt_path,
                config,
                best_acc=best_acc,
                best_auc=best_auc,
                best_test_acc=best_test_acc,
                best_test_auc=best_test_auc,
            )
            print(f"🏆 New best AUC! Saved: {best_auc_ckpt_path}")
            print(f"   Previous best AUC: {previous_best_auc:.4f} → New best AUC: {val_auc:.4f}")
            print(f"   Validation Loss: {val_loss:.6f}")

        # Update best external test metrics and save checkpoints (only when test_data_path is set)
        if test_loader is not None and test_acc is not None and test_auc is not None:
            if test_acc >= best_test_acc:
                prev_test_acc = best_test_acc
                best_test_acc = test_acc
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    os.path.join(ckpt_dir, "checkpoint_best_test_acc.pth"),
                    config, best_acc=best_acc, best_auc=best_auc,
                    best_test_acc=best_test_acc, best_test_auc=best_test_auc,
                )
                print(f"🏆 New best test accuracy! Saved: checkpoint_best_test_acc.pth ({prev_test_acc:.4f} → {best_test_acc:.4f})")
            if test_auc >= best_test_auc:
                prev_test_auc = best_test_auc
                best_test_auc = test_auc
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    os.path.join(ckpt_dir, "checkpoint_best_test_auc.pth"),
                    config, best_acc=best_acc, best_auc=best_auc,
                    best_test_acc=best_test_acc, best_test_auc=best_test_auc,
                )
                print(f"🏆 New best test AUC! Saved: checkpoint_best_test_auc.pth ({prev_test_auc:.4f} → {best_test_auc:.4f})")

        # Save last checkpoint for resume (overwrites each epoch)
        last_ckpt_path = os.path.join(ckpt_dir, "checkpoint_last.pth")
        save_checkpoint(
            model,
            optimizer,
            epoch,
            train_loss,
            val_loss,
            last_ckpt_path,
            config,
            best_acc=best_acc,
            best_auc=best_auc,
            best_test_acc=best_test_acc,
            best_test_auc=best_test_auc,
        )

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            epoch_ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                epoch_ckpt_path,
                config,
                best_acc=best_acc,
                best_auc=best_auc,
                best_test_acc=best_test_acc,
                best_test_auc=best_test_auc,
            )
            print(f"Checkpoint saved: {epoch_ckpt_path}")
        
        # Save checkpoint at the last epoch
        if epoch == config["num_epochs"]:
            final_ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                final_ckpt_path,
                config,
                best_acc=best_acc,
                best_auc=best_auc,
                best_test_acc=best_test_acc,
                best_test_auc=best_test_auc,
            )
            print(f"Final checkpoint saved: {final_ckpt_path}")
    
    writer.close()

    print("Finished fullgait training.")


if __name__ == "__main__":
    main()

