"""
Test script for evaluating trained model checkpoints on test set.
Uses test_indices.json for test data.
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

from utils.data_sampler import (
    SigLIPFullGaitDataset_v2,
    SigLIPFullGaitDatasetPKL,
    fullgait_collate_fn,
)
from utils.sft_utils import (
    resolve_device,
    setup_multi_gpu,
    pick_criterion,
    eval_epoch,
    _compute_binary_metrics_from_logits
)
from utils.utils import (
    get_km_attention_map,
    plot_attention_maps,
    get_km_attention_x_grad_map,
    plot_attention_x_grad_maps,
)
from models.sft_regressor import SFTRegressor #SFTRegressor
from utils.subgroup_dk_indices import (
    build_dk_strata_indices,
    load_subgroup_dk_rows,
    warn_oob_subgroup_rows,
)
from utils.km_interpretability_core6 import (
    save_km_interpretability_core3_full_stratum,
    write_subgroup_interpretability_markdown,
)
from utils.plot_style_nature import apply_nature_journal_mpl_style

# JSON strata evaluated in ``run_test_visualization`` (order for logs and KM plots)
STRATUM_ORDER_DK = (
    ("general_cobb_gt10", "General (Cobb>10)"),
    ("single_thoracic", "Single thoracic"),
    ("single_lumbar", "Single lumbar"),
    ("multi", "Multi-curve"),
)


def _plot_first_batch_km_attn_xgrad(
    *,
    raw_model,
    stratum_loader: DataLoader,
    device,
    log_dir: str,
    attn_batch_size: int,
    prefix_attn: str,
    prefix_xgrad: str,
    batch: Optional[Dict[str, Any]] = None,
) -> None:
    """First batch only: KM attention and attention×gradient maps (optional per-stratum)."""
    if batch is None:
        for batch in stratum_loader:
            if "label" not in batch:
                continue
            break
        else:
            return

    with torch.no_grad():
        video = batch["video"].to(device)[:attn_batch_size]
        knowledge_map = batch["knowledge_map"].to(device)[:attn_batch_size]
        texts = batch.get("texts", None)
        if isinstance(texts, list):
            texts = texts[:attn_batch_size]
        km_indices = batch.get("km_indices", None)
        video_indices = batch.get("video_indices", None)
        if km_indices is not None:
            km_indices = km_indices[:attn_batch_size].to(device)
        if video_indices is not None:
            video_indices = video_indices[:attn_batch_size].to(device)
        _ = raw_model(
            video,
            knowledge_map,
            texts=texts,
            km_indices=km_indices,
            video_indices=video_indices,
        )
        attn_map = get_km_attention_map(raw_model, knowledge_map, use_last_layer=True, layer_index=-1)
        if attn_map is not None:
            plot_attention_maps(
                attn_map,
                knowledge_map,
                save_dir=log_dir,
                max_samples=attn_batch_size,
                prefix=prefix_attn,
            )

    video = batch["video"].to(device)[:attn_batch_size]
    knowledge_map = batch["knowledge_map"].to(device)[:attn_batch_size]
    texts = batch.get("texts", None)
    if isinstance(texts, list):
        texts = texts[:attn_batch_size]
    km_indices = batch.get("km_indices", None)
    video_indices = batch.get("video_indices", None)
    if km_indices is not None:
        km_indices = km_indices[:attn_batch_size].to(device)
    if video_indices is not None:
        video_indices = video_indices[:attn_batch_size].to(device)

    axg_map, topk_info, logits_np = get_km_attention_x_grad_map(
        raw_model=raw_model,
        video=video,
        knowledge_map=knowledge_map,
        texts=texts,
        km_indices=km_indices,
        video_indices=video_indices,
        use_last_layer=True,
        layer_index=-1,
    )
    if axg_map is not None:
        plot_attention_x_grad_maps(
            heatmap_maps=axg_map,
            knowledge_map=knowledge_map,
            save_dir=log_dir,
            topk_info=topk_info,
            logits=logits_np,
            max_samples=attn_batch_size,
            prefix=prefix_xgrad,
        )


def main():
    # Test-specific configuration (dataset paths, etc.)
    # Model config will be loaded from checkpoint
    checkpoint_path = r"C:\Users\Olive\Desktop\Nature_Style\code_video\pytorch\checkpoints\kfold5_kvt_vivit_pretrained_cobb11\fold_1\checkpoint_best.pth"
    
    # Optional: Explicitly specify test pkl directory if different from default
    # If None, will try to auto-detect or use checkpoint config
    # You can also provide a list of directories to concatenate.
    test_pkl_data_dir_override = [
        os.path.join(current_dir, "dataset", "test_dk_pkl^1"),
    ]
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Please set checkpoint_path in the script.")
        return
    
    # Load checkpoint first to get model config
    print("Loading checkpoint...")
    device = resolve_device(None)  # Use default device initially
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    
    if not checkpoint_config:
        print("Error: Checkpoint does not contain config. Cannot proceed.")
        return
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Train loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    
    # Setup device and multi-GPU from checkpoint config
    gpu_ids = checkpoint_config.get("gpu_ids")
    if gpu_ids is not None:
        print(f"Using GPUs: {gpu_ids}")
    device = resolve_device(gpu_ids)
    print(f"Primary device: {device}")
    
    # Create test dataset (use checkpoint config with test-specific overrides)
    print("\n" + "="*60)
    print("Loading test dataset...")
    print("="*60)
    
    # test_binary_threshold = checkpoint_config.get("binary_threshold", 11.0)  # Use checkpoint's threshold
    test_binary_threshold = 15.0
    print(f"Using binary_threshold: {test_binary_threshold}")
    print(f"  (Labels with max(label) >= {test_binary_threshold} will be classified as positive)")
    
    # Check if preprocessed pkl files are available (prefer from checkpoint config, or use default test pkl dir)
    use_preprocessed_pkl = checkpoint_config.get("use_preprocessed_pkl", False)
    
    # Use override if provided, otherwise try checkpoint config, then auto-detect
    if test_pkl_data_dir_override is not None:
        test_pkl_data_dir = test_pkl_data_dir_override
        use_preprocessed_pkl = True
        print(f"Using explicit test pkl directory: {test_pkl_data_dir}")
    else:
        test_pkl_data_dir = checkpoint_config.get("pkl_data_dir", None)
        
        # If pkl_data_dir is not in config, try to infer test pkl directory
        # Common pattern: if training pkl is in "./data/preprocessed_pkl", test might be in "./data/preprocessed_pkl_test"
        # if not use_preprocessed_pkl and test_pkl_data_dir is None:
        #     # Try to find test pkl directory
        #     possible_test_pkl_dirs = [
        #         "./data/test_sz_pkl",
        #         checkpoint_config.get("pkl_data_dir", "./data/preprocessed_pkl") + "_test",
        #     ]
        #     for test_dir in possible_test_pkl_dirs:
        #         metadata_path = os.path.join(test_dir, "patch_metadata.pkl")
        #         if os.path.exists(metadata_path):
        #             test_pkl_data_dir = test_dir
        #             use_preprocessed_pkl = True
        #             print(f"Auto-detected test pkl directory: {test_pkl_data_dir}")
        #             break

    # use_preprocessed_pkl=False######oliver####
    # Load dataset (either from preprocessed pickle files or on-the-fly)
    if use_preprocessed_pkl:  # and test_pkl_data_dir is not None:
        print(f"Loading preprocessed test dataset from pickle files...")

        # Support either a single directory or a list of directories.
        if isinstance(test_pkl_data_dir, (list, tuple)):
            test_datasets = []
            total_patches = 0
            for d in test_pkl_data_dir:
                d_abs = os.path.abspath(os.path.normpath(d))
                print(f"  PKL data dir: {d_abs}")
                ds = SigLIPFullGaitDatasetPKL(
                    pkl_data_dir=d_abs,
                    metadata_path=None,
                    km_gaussian_noise_std=None,  # No augmentation for testing
                    mode="test",  # Test mode: uses first 96 frames only (consistent evaluation)
                    prompts_path=checkpoint_config.get("prompts_path"),
                    prompt_selection=checkpoint_config.get("prompt_selection", "top_feature_prompts"),
                    binary_threshold=test_binary_threshold,
                )
                print(f"    -> {len(ds)} patches")
                test_datasets.append(ds)
                total_patches += len(ds)

            if len(test_datasets) == 1:
                test_dataset = test_datasets[0]
            else:
                test_dataset = ConcatDataset(test_datasets)
            print(f"Test dataset loaded: {total_patches} samples from {len(test_datasets)} pickle directories")
        else:
            print(f"  PKL data dir: {test_pkl_data_dir}")

            # When an explicit override directory is provided, we want to be sure
            # we read PKL files from that directory, even if there is an old
            # metadata file with stale absolute paths.
            #
            # Strategy:
            # - If test_pkl_data_dir_override is set, point metadata_path to a
            #   dedicated metadata file inside that directory. If it does not
            #   exist, SigLIPFullGaitDatasetPKL will rebuild it from
            #   <pkl_data_dir>/patches, using correct paths.
            # - If override is None, fall back to the default behavior (let the
            #   dataset decide which metadata to use).
            if test_pkl_data_dir_override is not None:
                metadata_path = os.path.join(
                    test_pkl_data_dir_override, "patch_metadata_test_override.pkl"
                )
            else:
                metadata_path = None

            test_dataset = SigLIPFullGaitDatasetPKL(
                pkl_data_dir=test_pkl_data_dir,
                metadata_path=metadata_path,
                km_gaussian_noise_std=None,  # No augmentation for testing
                mode="test",  # Test mode: uses first 96 frames only (consistent evaluation)
                prompts_path=checkpoint_config.get("prompts_path"),
                prompt_selection=checkpoint_config.get("prompt_selection", "top_feature_prompts"),
                binary_threshold=test_binary_threshold,
            )
            print(f"Test dataset loaded: {len(test_dataset)} samples from pickle files")
    else:
        print(f"Loading test dataset on-the-fly...")
        # Extract default paths to avoid backslash issues in f-strings
        default_table_dir = r"C:\Users\Administrator\project\data\sz_table_refinedyolo"
        default_video_dir = r"C:\Users\Administrator\project\data\sz_video_refinedyolo"
        table_dir = checkpoint_config.get("table_dir", default_table_dir)
        video_dir = checkpoint_config.get("video_dir", default_video_dir)
        print(f"  Table dir: {table_dir}")
        print(f"  Video dir: {video_dir}")
        
        test_dataset = SigLIPFullGaitDataset_v2(
            table_dir=table_dir,
            video_dir=video_dir,
            label_json_path=r"data\test_indices_dk.json",  # Test-specific
            split="test",  # Test-specific
            patch_size=96,  # Not used for patching, but kept for compatibility
            video_frame_count=checkpoint_config.get("video_frame_count", 32),
            video_target_size=checkpoint_config.get("video_target_size", (224, 224)),
            prompts_path=checkpoint_config.get("prompts_path"),
            prompt_selection=checkpoint_config.get("prompt_selection", "top_feature_prompts"),
            binary_threshold=test_binary_threshold,
            mode="test",  # Test mode: uses first 96 frames only (consistent evaluation)
        )
        print(f"Test dataset loaded: {len(test_dataset)} samples")
    ##############################################################
    # Use more workers for preprocessed pkl files (faster loading)
    num_workers = checkpoint_config.get("num_workers", 4)
    if use_preprocessed_pkl:
        # Can use more workers with preprocessed data (faster I/O)
        num_workers = max(num_workers, 4)
        print(f"Using {num_workers} workers for faster pkl loading")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=checkpoint_config.get("batch_size", 32),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=fullgait_collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else 2,
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model using checkpoint config
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    
    model = SFTRegressor(
        km_feature_dim=checkpoint_config.get("km_feature_dim", 238),
        hidden_dim=checkpoint_config.get("hidden_dim", 256),
        label_dim=checkpoint_config.get("label_dim", 1),
        video_encoder_type=checkpoint_config.get("video_encoder_type", "vivit"),
        video_encoder_kwargs=checkpoint_config.get("video_encoder_kwargs", {}),
        km_encoder_type=checkpoint_config.get("km_encoder_type", "vit"),
        km_encoder_kwargs=checkpoint_config.get("km_encoder_kwargs", {}),
        text_model_name=checkpoint_config.get("text_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        text_max_length=checkpoint_config.get("text_max_length", 128),
        text_trainable=checkpoint_config.get("text_trainable", False),
        use_text=checkpoint_config.get("use_text", True),
        use_latent_pooling=checkpoint_config.get("use_latent_pooling", False),
        latent_pool_size=checkpoint_config.get("latent_pool_size", 1),
        regressor_dropout=checkpoint_config.get("regressor_dropout", 0.1),
        use_km_video_cross_attn=checkpoint_config.get("use_km_video_cross_attn", False),
        cross_attn_num_heads=checkpoint_config.get("cross_attn_num_heads", 8),
        cross_attn_drop=checkpoint_config.get("cross_attn_drop", 0.1),
        use_gated_token_pooling=checkpoint_config.get("use_gated_token_pooling", False),
    )
    
    # Setup multi-GPU if specified
    model = setup_multi_gpu(model, gpu_ids, checkpoint_config.get("use_distributed", False))
    model = model.to(device)
    model.eval()
    
    # Load model weights (strict=False so old checkpoints without OGM unimodal heads load correctly)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print("Model weights loaded successfully.")

    # Count total and trainable parameters
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")

    # Setup loss criterion
    criterion = pick_criterion(checkpoint_config.get("label_dim", 1))
    
    # Setup log directory for results file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    log_dir = os.path.join(checkpoint_config.get("log_dir", "./logs"), f"test_{checkpoint_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Run evaluation
    print("\n" + "="*60)
    print("Running test evaluation...")
    print("="*60)
    
    test_loss, test_metrics = eval_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        writer=None,  # No TensorBoard logging
        global_step=0,
        verbose=False,  # Suppress detailed batch-by-batch output
        config=None,
    )
    
    # Calculate macro-averaged metrics
    positive_precision = test_metrics.get('precision', 0.0)
    positive_recall = test_metrics.get('recall', 0.0)
    positive_f1 = test_metrics.get('f1', 0.0)
    
    negative_precision = test_metrics.get('negative_precision', 0.0)
    negative_recall = test_metrics.get('negative_recall', 0.0)
    # Calculate negative F1
    negative_f1 = 2 * (negative_precision * negative_recall) / (negative_precision + negative_recall + 1e-8) if (negative_precision + negative_recall) > 0 else 0.0
    
    macro_avg_precision = (positive_precision + negative_precision) / 2.0
    macro_avg_recall = (positive_recall + negative_recall) / 2.0
    macro_avg_f1 = (positive_f1 + negative_f1) / 2.0
    
    # For AUC metrics, we'll use the overall AUC-ROC as macro-averaged AUC (OVR)
    # Per-class AUC would require separate calculation, but for binary classification,
    # the overall AUC-ROC is typically what we report
    macro_auc_ovr = test_metrics.get('auc_roc', 0.0)
    positive_auc = macro_auc_ovr  # For binary classification, positive class AUC = overall AUC
    negative_auc = macro_auc_ovr  # For binary classification, negative class AUC = overall AUC
    
    # Print results in the requested format
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Total Accuracy: {test_metrics.get('accuracy', 0.0) * 100:.2f}%")
    print(f"Macro-avg Precision: {macro_avg_precision * 100:.2f}%")
    print(f"Macro-avg Recall: {macro_avg_recall * 100:.2f}%")
    print(f"Macro-avg F1 Score: {macro_avg_f1 * 100:.2f}%")
    print(f"=== Positive Class Metrics ===")
    print(f"Positive Precision: {positive_precision * 100:.2f}%")
    print(f"Positive Recall: {positive_recall * 100:.2f}%")
    print(f"Positive F1: {positive_f1 * 100:.2f}%")
    print(f"=== Negative Class Metrics ===")
    print(f"Negative Precision: {negative_precision * 100:.2f}%")
    print(f"Negative Recall: {negative_recall * 100:.2f}%")
    print(f"Negative F1: {negative_f1 * 100:.2f}%")
    print(f"=== AUC Metrics ===")
    print(f"Macro-averaged AUC (OVR): {macro_auc_ovr * 100:.2f}%")
    print(f"Positive Class AUC: {positive_auc * 100:.2f}%")
    print(f"Negative Class AUC: {negative_auc * 100:.2f}%")
    print("="*70)

    # Confusion matrix (TP, TN, FP, FN from last evaluation)
    tp = int(test_metrics.get("tp", 0))
    tn = int(test_metrics.get("tn", 0))
    fp = int(test_metrics.get("fp", 0))
    fn = int(test_metrics.get("fn", 0))
    print("\nConfusion matrix (last evaluation):")
    print("                 Predicted")
    print("                 Neg    Pos")
    print(f"  Actual Neg   {tn:5d}   {fp:5d}   (TN, FP)")
    print(f"  Actual Pos   {fn:5d}   {tp:5d}   (FN, TP)")
    print("="*70)

    # Save base results to file
    results_file = os.path.join(log_dir, "test_results.txt")
    with open(results_file, "w") as f:
        f.write("TEST RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint.get('epoch', 'N/A')}\n")
        f.write(f"Binary Threshold: {test_binary_threshold}\n")
        f.write(f"Loss: {test_loss:.6f}\n\n")
        
        f.write(f"Total Accuracy: {test_metrics.get('accuracy', 0.0) * 100:.2f}%\n")
        f.write(f"Macro-avg Precision: {macro_avg_precision * 100:.2f}%\n")
        f.write(f"Macro-avg Recall: {macro_avg_recall * 100:.2f}%\n")
        f.write(f"Macro-avg F1 Score: {macro_avg_f1 * 100:.2f}%\n")
        f.write(f"=== Positive Class Metrics ===\n")
        f.write(f"Positive Precision: {positive_precision * 100:.2f}%\n")
        f.write(f"Positive Recall: {positive_recall * 100:.2f}%\n")
        f.write(f"Positive F1: {positive_f1 * 100:.2f}%\n")
        f.write(f"=== Negative Class Metrics ===\n")
        f.write(f"Negative Precision: {negative_precision * 100:.2f}%\n")
        f.write(f"Negative Recall: {negative_recall * 100:.2f}%\n")
        f.write(f"Negative F1: {negative_f1 * 100:.2f}%\n")
        f.write(f"=== AUC Metrics ===\n")
        f.write(f"Macro-averaged AUC (OVR): {macro_auc_ovr * 100:.2f}%\n")
        f.write(f"Positive Class AUC: {positive_auc * 100:.2f}%\n")
        f.write(f"Negative Class AUC: {negative_auc * 100:.2f}%\n")
        f.write("="*70 + "\n")
        f.write("\nConfusion matrix (last evaluation):\n")
        f.write("                 Predicted\n")
        f.write("                 Neg    Pos\n")
        f.write(f"  Actual Neg   {tn:5d}   {fp:5d}   (TN, FP)\n")
        f.write(f"  Actual Pos   {fn:5d}   {tp:5d}   (FN, TP)\n")
        f.write("="*70 + "\n")

    # ------------------------------------------------------------------
    # Optional: threshold sweep to find better operating point
    # ------------------------------------------------------------------
    print("\nSweeping decision thresholds on test set (based on logits)...")
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            if "label" not in batch:
                continue
            video = batch["video"].to(device)
            knowledge_map = batch["knowledge_map"].to(device)
            texts = batch.get("texts", None)
            km_indices = batch.get("km_indices", None)
            video_indices = batch.get("video_indices", None)
            if km_indices is not None:
                km_indices = km_indices.to(device)
            if video_indices is not None:
                video_indices = video_indices.to(device)

            logits = model(
                video,
                knowledge_map,
                texts,
                km_indices=km_indices,
                video_indices=video_indices,
            )
            labels = batch["label"].to(device)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if all_logits:
        all_logits_tensor = torch.cat(all_logits, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)

        thresholds = [round(t, 2) for t in [i / 20.0 for i in range(1, 20)]]  # 0.05 ... 0.95
        sweep_results = []
        for t in thresholds:
            metrics_t = _compute_binary_metrics_from_logits(all_logits_tensor, all_labels_tensor, t)
            sweep_results.append(metrics_t)

        # Find best F1 threshold
        best_by_f1 = max(sweep_results, key=lambda m: m["f1"])

        print("\nThreshold sweep (test set):")
        print("thresh | acc    prec   rec    f1    auc")
        print("----------------------------------------------")
        for m in sweep_results:
            print(
                f"{m['threshold']:.2f}  | "
                f"{m['accuracy']*100:5.1f}% "
                f"{m['precision']*100:5.1f}% "
                f"{m['recall']*100:5.1f}% "
                f"{m['f1']*100:5.1f}% "
                f"{m['auc_roc']:.3f}"
            )

        print("\nBest F1 threshold on test set:")
        print(
            f"  t = {best_by_f1['threshold']:.2f} "
            f"(acc={best_by_f1['accuracy']*100:.2f}%, "
            f"prec={best_by_f1['precision']*100:.2f}%, "
            f"rec={best_by_f1['recall']*100:.2f}%, "
            f"f1={best_by_f1['f1']*100:.2f}%, "
            f"auc={best_by_f1['auc_roc']:.3f})"
        )

        # Append threshold sweep summary to results file
        with open(results_file, "a") as f:
            f.write("\nThreshold sweep (test set):\n")
            f.write("thresh | acc    prec   rec    f1    auc\n")
            f.write("----------------------------------------------\n")
            for m in sweep_results:
                f.write(
                    f"{m['threshold']:.2f}  | "
                    f"{m['accuracy']*100:5.1f}% "
                    f"{m['precision']*100:5.1f}% "
                    f"{m['recall']*100:5.1f}% "
                    f"{m['f1']*100:5.1f}% "
                    f"{m['auc_roc']:.3f}\n"
                )
            f.write("\nBest F1 threshold on test set:\n")
            f.write(
                f"t = {best_by_f1['threshold']:.2f} "
                f"(acc={best_by_f1['accuracy']*100:.2f}%, "
                f"prec={best_by_f1['precision']*100:.2f}%, "
                f"rec={best_by_f1['recall']*100:.2f}%, "
                f"f1={best_by_f1['f1']*100:.2f}%, "
                f"auc={best_by_f1['auc_roc']:.3f})\n"
            )

    # ------------------------------------------------------------------
    # DK subgroup_dk.json strata: eval_epoch per stratum (DK PKL only)
    # ------------------------------------------------------------------
    subgroup_json_path = os.path.join(current_dir, "data", "subgroup_dk.json")
    subgroup_eval: Dict[str, Any] = {}
    if use_preprocessed_pkl and os.path.isfile(subgroup_json_path):
        n_all = len(test_dataset)
        rows = load_subgroup_dk_rows(subgroup_json_path)
        warn_oob_subgroup_rows(rows, n_all)
        strata_indices, strata_meta = build_dk_strata_indices(rows, n_all)
        subgroup_eval["strata_meta"] = strata_meta
        eval_bs = int(checkpoint_config.get("batch_size", 32))
        print("\n" + "=" * 60)
        print("Subgroup evaluation (data/subgroup_dk.json, DK patches only)")
        print("=" * 60)
        with open(results_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write("SUBGROUP EVALUATION (subgroup_dk.json, DK only)\n")
            f.write("=" * 70 + "\n")

        for key, title in STRATUM_ORDER_DK:
            idx_list = strata_indices.get(key, [])
            entry: Dict[str, Any] = {"title": title, "n_patches": len(idx_list)}
            if not idx_list:
                entry["skipped"] = True
                subgroup_eval[key] = entry
                print(f"  [{key}] n=0 (skipped)")
                with open(results_file, "a", encoding="utf-8") as f:
                    f.write(f"\n[{title}] n_patches=0 (skipped)\n")
                continue
            sub_ds = Subset(test_dataset, idx_list)
            sub_loader = DataLoader(
                sub_ds,
                batch_size=eval_bs,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=fullgait_collate_fn,
                pin_memory=device.type == "cuda",
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else 2,
            )
            s_loss, s_metrics = eval_epoch(
                model=model,
                dataloader=sub_loader,
                criterion=criterion,
                device=device,
                writer=None,
                global_step=0,
                verbose=False,
                config=None,
            )
            entry["loss"] = float(s_loss)
            entry["metrics"] = {k: float(v) for k, v in s_metrics.items() if isinstance(v, (int, float, np.floating))}
            subgroup_eval[key] = entry
            print(f"  [{key}] n={len(idx_list)} loss={s_loss:.6f} auc={s_metrics.get('auc_roc', 0):.4f}")
            with open(results_file, "a", encoding="utf-8") as f:
                f.write(f"\n[{title}] n_patches={len(idx_list)}\n")
                f.write(f"  Loss: {s_loss:.6f}\n")
                for mk in sorted(entry["metrics"].keys()):
                    f.write(f"  {mk}: {entry['metrics'][mk]:.6f}\n")

        subgroup_json_out = os.path.join(log_dir, "subgroup_eval.json")

        def _json_sanitize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _json_sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_json_sanitize(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            return obj

        with open(subgroup_json_out, "w", encoding="utf-8") as jf:
            json.dump(_json_sanitize(subgroup_eval), jf, indent=2)
        print(f"Subgroup metrics JSON: {subgroup_json_out}")
    elif use_preprocessed_pkl:
        print(f"\nSubgroup JSON not found (skipping strata): {subgroup_json_path}")

    # ------------------------------------------------------------------
    # KM attention / attention×grad: first batch per JSON stratum (prefix)
    # ------------------------------------------------------------------
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    if hasattr(raw_model, "km_encoder") and hasattr(raw_model.km_encoder, "get_block_attention"):
        apply_nature_journal_mpl_style()
        attn_batch_size = min(8, checkpoint_config.get("batch_size", 32))
        if use_preprocessed_pkl and os.path.isfile(subgroup_json_path):
            print("\nExtracting KM attention / attention×grad and subgroup core panels (all patches per stratum)...")
            eval_bs = int(checkpoint_config.get("batch_size", 32))
            n_all = len(test_dataset)
            rows = load_subgroup_dk_rows(subgroup_json_path)
            strata_indices, _ = build_dk_strata_indices(rows, n_all)
            core_stats_collected: List[Dict[str, Any]] = []
            for key, title in STRATUM_ORDER_DK:
                idx_list = strata_indices.get(key, [])
                if not idx_list:
                    continue
                sub_ds = Subset(test_dataset, idx_list)
                sub_loader = DataLoader(
                    sub_ds,
                    batch_size=eval_bs,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=fullgait_collate_fn,
                    pin_memory=device.type == "cuda",
                    persistent_workers=num_workers > 0,
                    prefetch_factor=4 if num_workers > 0 else 2,
                )
                # Keep per-sample diagnostic plots: first batch only
                first_batch = None
                for batch in sub_loader:
                    if "label" not in batch:
                        continue
                    first_batch = batch
                    break
                if first_batch is not None:
                    print(f"  First-batch diagnostic plots: {title} (prefix km_attn_{key})")
                    _plot_first_batch_km_attn_xgrad(
                        raw_model=raw_model,
                        stratum_loader=sub_loader,
                        device=device,
                        log_dir=log_dir,
                        attn_batch_size=attn_batch_size,
                        prefix_attn=f"km_attn_{key}",
                        prefix_xgrad=f"km_attn_xgrad_{key}",
                        batch=first_batch,
                    )

                print(f"  Subgroup core panels (all patches): {title} (core3)")
                st = save_km_interpretability_core3_full_stratum(
                    raw_model=raw_model,
                    stratum_key=key,
                    stratum_title=title,
                    dataloader=sub_loader,
                    device=device,
                    out_dir=log_dir,
                )
                if st is not None:
                    core_stats_collected.append(st)
            if core_stats_collected:
                write_subgroup_interpretability_markdown(
                    out_dir=log_dir,
                    strata_stats=core_stats_collected,
                    log_tag=os.path.basename(log_dir),
                )
        else:
            print("\nSkipping stratum KM plots (no subgroup JSON or PKL mode off).")
    else:
        print("\nSkipping attention maps (model has no ViT/PatchViT KM encoder).")

    print(f"\nResults saved to: {results_file}")
    
    print("\nTest evaluation completed!")


if __name__ == "__main__":
    main()

