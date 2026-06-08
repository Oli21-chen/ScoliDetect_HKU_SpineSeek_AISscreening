"""
Test script for evaluating trained model checkpoints on test set.
Uses test_indices.json for test data.
"""

import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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
    save_rows_csv,
    safe_filename_stem,
)
from models.sft_regressor import SFTRegressor #SFTRegressor
from utils.subgroup_dk_indices import (
    build_dk_strata_indices,
    filter_strata_indices_ais_only,
    load_dk_control_patch_ids,
    load_subgroup_dk_rows,
    warn_oob_subgroup_rows,
)
from utils.km_interpretability_core6 import (
    _aggregate_km_interpretability_from_dataloader,
    save_km_interpretability_core3_from_agg,
    save_fig5_dk_composite_interpretability,
)

STRATUM_ORDER_DK: Tuple[Tuple[str, str], ...] = (
    ("general_cobb_gt10", "General (Cobb>10)"),
    ("single_thoracic", "Single thoracic"),
    ("single_lumbar", "Single lumbar"),
    ("multi", "Multi-curve"),
)


def _resolve_patch_metadata(test_dataset) -> Optional[List[Dict[str, Any]]]:
    if hasattr(test_dataset, "patch_metadata"):
        return getattr(test_dataset, "patch_metadata")
    if isinstance(test_dataset, ConcatDataset):
        for ds in test_dataset.datasets:
            if hasattr(ds, "patch_metadata"):
                return getattr(ds, "patch_metadata")
    return None


def _resolve_n_dk(test_dataset) -> int:
    if isinstance(test_dataset, ConcatDataset):
        return len(test_dataset.datasets[0])
    return len(test_dataset)


def _resolve_existing_pkl_dir(path_str: str) -> str:
    """
    Resolve a PKL data directory robustly.
    - Accepts relative or absolute paths.
    - Strips trailing footnote-like suffixes (e.g., '^1') often introduced by copy/paste.
    - For relative paths, tries both current working directory and this script directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw = str(path_str).strip()
    candidates: List[str] = [raw]

    # Common copy/paste artifact from markdown footnotes.
    without_footnote = re.sub(r"\^\d+$", "", raw)
    if without_footnote != raw:
        candidates.append(without_footnote)

    resolved_candidates: List[str] = []
    for c in candidates:
        if os.path.isabs(c):
            resolved_candidates.append(os.path.abspath(os.path.normpath(c)))
        else:
            resolved_candidates.append(os.path.abspath(os.path.normpath(c)))
            resolved_candidates.append(os.path.abspath(os.path.normpath(os.path.join(script_dir, c))))

    # Preserve order while removing duplicates.
    seen = set()
    uniq_candidates = []
    for c in resolved_candidates:
        if c not in seen:
            uniq_candidates.append(c)
            seen.add(c)

    for c in uniq_candidates:
        if os.path.isdir(c):
            return c

    hint = (
        f"Could not find PKL directory from '{path_str}'. Tried:\n  - "
        + "\n  - ".join(uniq_candidates)
        + "\nTip: if your path came from markdown/docs, remove trailing '^<number>' footnotes."
    )
    raise FileNotFoundError(hint)


def _resolve_existing_file(
    path_str: Optional[str],
    *,
    script_dir: str,
    preferred_dirs: Optional[List[str]] = None,
    file_desc: str = "file",
) -> Optional[str]:
    """
    Resolve a file path robustly across machines.
    - Accepts absolute/relative paths.
    - Strips trailing footnote-like suffixes (e.g., '^1').
    - If direct path is stale (e.g., old absolute path from another server),
      falls back by filename into preferred directories (such as local ./data).
    """
    if path_str is None:
        return None

    raw = str(path_str).strip()
    candidates: List[str] = [raw]
    without_footnote = re.sub(r"\^\d+$", "", raw)
    if without_footnote != raw:
        candidates.append(without_footnote)

    resolved_candidates: List[str] = []
    for c in candidates:
        if os.path.isabs(c):
            resolved_candidates.append(os.path.abspath(os.path.normpath(c)))
        else:
            resolved_candidates.append(os.path.abspath(os.path.normpath(c)))
            resolved_candidates.append(os.path.abspath(os.path.normpath(os.path.join(script_dir, c))))

    basename = os.path.basename(without_footnote or raw)
    search_dirs = preferred_dirs or []
    for d in search_dirs:
        resolved_candidates.append(os.path.abspath(os.path.normpath(os.path.join(d, basename))))

    seen = set()
    uniq_candidates: List[str] = []
    for c in resolved_candidates:
        if c not in seen:
            uniq_candidates.append(c)
            seen.add(c)

    for c in uniq_candidates:
        if os.path.isfile(c):
            return c

    hint = (
        f"Could not find {file_desc} from '{path_str}'. Tried:\n  - "
        + "\n  - ".join(uniq_candidates)
    )
    raise FileNotFoundError(hint)


def _resolve_log_dir_root(config_log_dir: Optional[str], *, script_dir: str) -> str:
    """
    Keep test logs inside this project folder for cross-server portability.
    - If config path exists locally, use it.
    - Otherwise, remap to <script_dir>/logs.
    """
    default_root = os.path.join(script_dir, "logs")
    if not config_log_dir:
        return default_root

    candidate = str(config_log_dir).strip()
    if not candidate:
        return default_root

    if not os.path.isabs(candidate):
        rel_candidate = os.path.abspath(os.path.normpath(os.path.join(script_dir, candidate)))
        return rel_candidate

    abs_candidate = os.path.abspath(os.path.normpath(candidate))
    if os.path.isdir(abs_candidate):
        return abs_candidate

    # Stale absolute path from another server -> remap to local project logs.
    return default_root


def _run_dk_km_fig5_interpretability(
    *,
    raw_model,
    test_dataset,
    device: torch.device,
    log_dir: str,
    checkpoint_config: Dict[str, Any],
    use_preprocessed_pkl: bool,
    num_workers: int,
    subgroup_json_path: str,
    test_indices_dk_path: str,
    test_binary_threshold: float,
    cobb_general_threshold: float = 10.0,
    exclude_screening_negative: bool = False,
) -> None:
    """Fig. 5 composite + per-stratum core3 KM interpretability (DK subgroups)."""
    if not use_preprocessed_pkl:
        print("\nSkipping Fig. 5 KM interpretability (requires preprocessed PKL test data).")
        return
    if not os.path.isfile(subgroup_json_path):
        print(f"\nSkipping Fig. 5 KM interpretability (subgroup JSON not found): {subgroup_json_path}")
        return
    if not (hasattr(raw_model, "km_encoder") and hasattr(raw_model.km_encoder, "get_block_attention")):
        print("\nSkipping Fig. 5 KM interpretability (model has no ViT/PatchViT KM encoder).")
        return

    n_all = len(test_dataset)
    n_dk = _resolve_n_dk(test_dataset)
    rows = load_subgroup_dk_rows(subgroup_json_path)
    patch_metadata = _resolve_patch_metadata(test_dataset)
    warn_oob_subgroup_rows(rows, n_dk, patch_metadata=patch_metadata)
    control_patch_ids = load_dk_control_patch_ids(test_indices_dk_path)
    strata_indices, strata_meta = build_dk_strata_indices(
        rows,
        n_dk,
        patch_metadata=patch_metadata,
        cobb_general_threshold=cobb_general_threshold,
        control_patch_ids=control_patch_ids,
        ais_only=True,
    )
    strata_indices, ais_filter_stats = filter_strata_indices_ais_only(
        strata_indices,
        full_dataset=test_dataset,
        n_dk=n_dk,
        binary_threshold=test_binary_threshold,
        control_patch_ids=control_patch_ids,
        exclude_screening_negative=exclude_screening_negative,
    )
    print("\n" + "=" * 60)
    print("Fig. 5 KM interpretability (AIS-only DK cohort, subgroup_indices.json)")
    print("=" * 60)
    print(
        f"  Controls excluded (max Cobb=0 in {os.path.basename(test_indices_dk_path)}): "
        f"{len(control_patch_ids)} patch ids"
    )
    print(
        f"  AIS filter: Cobb>{cobb_general_threshold}°, label1 single/multi, "
        f"screening-negative excluded={exclude_screening_negative} "
        f"(binary threshold={test_binary_threshold})"
    )
    print(f"  Stratum counts (AIS-only): {ais_filter_stats.get('per_stratum', {})}")
    print(f"  Union AIS interpretability cohort: n={ais_filter_stats.get('ais_union_n', '?')}")

    if isinstance(test_dataset, ConcatDataset) and len(test_dataset.datasets) > 1:
        print(
            "  Note: test set concatenates multiple PKL roots; subgroup indices resolve by DK patch_id. "
            "For manuscript n counts, use DK-only test_pkl_data_dir_override."
        )

    eval_bs = int(checkpoint_config.get("batch_size", 32))
    attn_batch_size = min(8, eval_bs)
    axg_chunk_size = int(checkpoint_config.get("attn_batch_size_xgrad", 1))
    axg_chunk_size = max(1, min(attn_batch_size, axg_chunk_size))
    # x-grad backward is heavy; avoid multiprocessing pin_memory crashes after OOM.
    fig5_num_workers = 0
    print(
        f"  Fig. 5 loader: batch_size={eval_bs}, axg_chunk_size={axg_chunk_size}, "
        f"num_workers={fig5_num_workers}"
    )
    aggs: Dict[str, Any] = {}
    core_stats_collected: List[Dict[str, Any]] = []

    for key, title in STRATUM_ORDER_DK:
        idx_list = strata_indices.get(key, [])
        if not idx_list:
            print(f"  [{key}] n=0 (skipped)")
            continue
        sub_ds = Subset(test_dataset, idx_list)
        sub_loader = DataLoader(
            sub_ds,
            batch_size=eval_bs,
            shuffle=False,
            num_workers=fig5_num_workers,
            collate_fn=fullgait_collate_fn,
            pin_memory=False,
            persistent_workers=False,
        )
        print(f"  Aggregating: {title} (n={len(idx_list)} patches)...")
        agg = _aggregate_km_interpretability_from_dataloader(
            raw_model=raw_model,
            dataloader=sub_loader,
            device=device,
            axg_chunk_size=axg_chunk_size,
            attn_chunk_size=attn_batch_size,
        )
        if agg is None:
            print(f"  [{key}] aggregation failed (skipped)")
            continue
        aggs[key] = (title, agg)
        st = save_km_interpretability_core3_from_agg(
            agg,
            stratum_key=key,
            stratum_title=title,
            out_dir=log_dir,
        )
        if st is not None:
            core_stats_collected.append(st)

    general_entry = aggs.get("general_cobb_gt10")
    stratum_entries = [
        (key, aggs[key][0], aggs[key][1])
        for key, _title in STRATUM_ORDER_DK
        if key in aggs
    ]
    if general_entry is not None and len(stratum_entries) == len(STRATUM_ORDER_DK):
        save_fig5_dk_composite_interpretability(
            general_agg=general_entry[1],
            stratum_aggs=stratum_entries,
            out_dir=log_dir,
        )
    else:
        print(
            "  Warning: Fig. 5 composite skipped (need all four strata aggregated, "
            f"including general_cobb_gt10; have keys={list(aggs.keys())})."
        )

    # Markdown report is intentionally disabled for this test pipeline.


def main():
    # Test-specific configuration (dataset paths, etc.)
    # Model config will be loaded from checkpoint
    checkpoint_path = r"/root/private_data/NMI_Project_Val/checkpoints/kfold5_kvt_vivit_pretrained_cobb11/fold_1/checkpoint_best.pth"
    test_binary_threshold = 11.0
    
    # Optional: Explicitly specify test pkl directory if different from default
    # If None, will try to auto-detect or use checkpoint config
    # You can also provide a list of directories to concatenate.
    # test_pkl_data_dir_override = [
    #     "./data/dk_pkl_stage3",
    #     "./data/pk_pkl_stage3",
    # ]
    test_pkl_data_dir_override = [
        "./data/test_dk_pkl^1",
        "./data/test_pk_pkl^1",
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

    # Resolve prompts_path for portability across servers.
    # Checkpoint/PKL metadata may store stale absolute paths from another machine.
    prompts_path_from_ckpt = checkpoint_config.get("prompts_path")
    resolved_prompts_path = _resolve_existing_file(
        prompts_path_from_ckpt,
        script_dir=current_dir,
        preferred_dirs=[
            os.path.join(current_dir, "data"),
            os.path.join(os.getcwd(), "data"),
            current_dir,
            os.getcwd(),
        ],
        file_desc="prompts JSON",
    )
    checkpoint_config["prompts_path"] = resolved_prompts_path
    if prompts_path_from_ckpt != resolved_prompts_path:
        print("Remapped prompts_path for current server:")
        print(f"  from: {prompts_path_from_ckpt}")
        print(f"  to:   {resolved_prompts_path}")
    
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
        
       
    # use_preprocessed_pkl=False######oliver####
    # Load dataset (either from preprocessed pickle files or on-the-fly)
    if use_preprocessed_pkl:  # and test_pkl_data_dir is not None:
        print(f"Loading preprocessed test dataset from pickle files...")

        # Support either a single directory or a list of directories.
        if isinstance(test_pkl_data_dir, (list, tuple)):
            test_datasets = []
            total_patches = 0
            for d in test_pkl_data_dir:
                d_abs = _resolve_existing_pkl_dir(d)
                print(f"  PKL data dir: {d_abs}")
                ds = SigLIPFullGaitDatasetPKL(
                    pkl_data_dir=d_abs,
                    metadata_path=None,
                    km_gaussian_noise_std=None,  # No augmentation for testing
                    mode="test",  # Test mode: uses first 96 frames only (consistent evaluation)
                    prompts_path=resolved_prompts_path,
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
            print(f"✅ Test dataset loaded: {total_patches} samples from {len(test_datasets)} pickle directories")
        else:
            test_pkl_data_dir = _resolve_existing_pkl_dir(test_pkl_data_dir)
            print(f"  PKL data dir: {test_pkl_data_dir}")

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
                prompts_path=resolved_prompts_path,
                prompt_selection=checkpoint_config.get("prompt_selection", "top_feature_prompts"),
                binary_threshold=test_binary_threshold,
            )
            print(f"✅ Test dataset loaded: {len(test_dataset)} samples from pickle files")
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
            label_json_path=os.path.join(current_dir, "data", "test_indices_dk.json"),  # Test-specific
            split="test",  # Test-specific
            patch_size=96,  # Not used for patching, but kept for compatibility
            video_frame_count=checkpoint_config.get("video_frame_count", 32),
            video_target_size=checkpoint_config.get("video_target_size", (224, 224)),
            prompts_path=resolved_prompts_path,
            prompt_selection=checkpoint_config.get("prompt_selection", "top_feature_prompts"),
            binary_threshold=test_binary_threshold,
            mode="test",  # Test mode: uses first 96 frames only (consistent evaluation)
        )
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
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
        cross_attn_num_layers=checkpoint_config.get("cross_attn_num_layers", 2),
        temporal_attn_bias_strength=checkpoint_config.get("temporal_attn_bias_strength", 0.0),
        modality_dropout_prob=checkpoint_config.get("modality_dropout_prob", 0.0),
        use_bottleneck_fusion=checkpoint_config.get("use_bottleneck_fusion", False),
        bottleneck_tokens=checkpoint_config.get("bottleneck_tokens", 16),
        bottleneck_layers=checkpoint_config.get("bottleneck_layers", 1),
        align_loss_weight=checkpoint_config.get("align_loss_weight", 0.0),
        align_loss_temperature=checkpoint_config.get("align_loss_temperature", 0.07),
        aux_loss_type=checkpoint_config.get("aux_loss_type", "infonce"),
        aux_proj_dim=checkpoint_config.get("aux_proj_dim", 256),
        barlow_lambda=checkpoint_config.get("barlow_lambda", 5e-3),
        vicreg_sim_coeff=checkpoint_config.get("vicreg_sim_coeff", 25.0),
        vicreg_var_coeff=checkpoint_config.get("vicreg_var_coeff", 25.0),
        vicreg_cov_coeff=checkpoint_config.get("vicreg_cov_coeff", 1.0),
        vicreg_var_target=checkpoint_config.get("vicreg_var_target", 1.0),
        vicreg_eps=checkpoint_config.get("vicreg_eps", 1e-4),
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
    
    # Setup log directory for results file (portable across servers)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    log_root = _resolve_log_dir_root(checkpoint_config.get("log_dir"), script_dir=current_dir)
    log_dir = os.path.join(log_root, f"test_{checkpoint_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_config["log_dir"] = log_root
    print(f"Log root: {log_root}")
    print(f"Run log dir: {log_dir}")
    created_at_iso = datetime.now().isoformat()
    classification_prob_threshold = float(checkpoint_config.get("classification_prob_threshold", 0.5))
    experiment = safe_filename_stem(
        checkpoint_config.get("run_name")
        or checkpoint_config.get("experiment_name")
        or checkpoint_config.get("model_name")
        or checkpoint_name
    )
    
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
    # Prediction CSV (single "fold") + threshold sweep
    # ------------------------------------------------------------------
    print("\nCollecting logits/labels for CSV export and threshold sweep...")
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

        # --------------------------------------------------------------
        # Save predictions CSV (Nature-style AUC-ROC input)
        # Same schema as run_prediction_kfold.py, but single fold.
        # --------------------------------------------------------------
        probs_tensor = torch.sigmoid(all_logits_tensor.float()).reshape(-1)
        labels_tensor = all_labels_tensor.float().reshape(-1)
        prediction_rows = []
        n = int(probs_tensor.numel())
        for i in range(n):
            prediction_rows.append(
                {
                    "created_at": created_at_iso,
                    "experiment": experiment,
                    "fold": 1,
                    "split": "test",
                    "sample_index": i,
                    "probability": float(probs_tensor[i].item()),
                    "label": float(labels_tensor[i].item()),
                    "binary_threshold_cobb": float(test_binary_threshold),
                    "classification_prob_threshold": float(classification_prob_threshold),
                    "checkpoint_path": checkpoint_path,
                }
            )

        output_base = safe_filename_stem(f"{experiment}_{timestamp}")
        pred_csv = os.path.join(log_dir, f"{output_base}_predictions.csv")
        try:
            if prediction_rows:
                save_rows_csv(prediction_rows, pred_csv)
                print(f"Prediction probabilities CSV (for AUC-ROC): {pred_csv}")
        except Exception as e:
            print(f"Warning: could not save prediction CSV: {e}")

        # Save a one-row metrics CSV for quick ingestion
        metrics_csv = os.path.join(log_dir, f"{output_base}_metrics.csv")
        try:
            save_rows_csv(
                [
                    {
                        "created_at": created_at_iso,
                        "experiment": experiment,
                        "fold": 1,
                        "split": "test",
                        "checkpoint_path": checkpoint_path,
                        "binary_threshold_cobb": float(test_binary_threshold),
                        "classification_prob_threshold": float(classification_prob_threshold),
                        "loss": float(test_loss),
                        "accuracy": float(test_metrics.get("accuracy", 0.0)),
                        "auc_roc": float(test_metrics.get("auc_roc", 0.0)),
                        "sensitivity": float(test_metrics.get("sensitivity", 0.0)),
                        "specificity": float(test_metrics.get("specificity", 0.0)),
                        "ppv": float(test_metrics.get("ppv", test_metrics.get("precision", 0.0))),
                        "npv": float(test_metrics.get("npv", 0.0)),
                        "f1": float(test_metrics.get("f1", 0.0)),
                        "tp": int(test_metrics.get("tp", 0)),
                        "tn": int(test_metrics.get("tn", 0)),
                        "fp": int(test_metrics.get("fp", 0)),
                        "fn": int(test_metrics.get("fn", 0)),
                        "n_samples": int(n),
                    }
                ],
                metrics_csv,
            )
            print(f"Metrics CSV: {metrics_csv}")
        except Exception as e:
            print(f"Warning: could not save metrics CSV: {e}")

        # --------------------------------------------------------------
        # Optional: threshold sweep to find better operating point
        # --------------------------------------------------------------
        print("\nSweeping decision thresholds on test set (based on logits)...")
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

        # Save threshold sweep table CSV for plotting/inspection
        sweep_csv = os.path.join(log_dir, f"{output_base}_threshold_sweep.csv")
        try:
            save_rows_csv(sweep_results, sweep_csv)
            print(f"Threshold sweep CSV: {sweep_csv}")
        except Exception as e:
            print(f"Warning: could not save threshold sweep CSV: {e}")

    # ------------------------------------------------------------------
    # Extract and plot KM attention maps (ViT or PatchViT encoder)
    # ------------------------------------------------------------------
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    if hasattr(raw_model, "km_encoder") and hasattr(raw_model.km_encoder, "get_block_attention"):
        print("\nExtracting KM attention maps and plotting...")
        attn_batch_size = min(8, checkpoint_config.get("batch_size", 32))
        # x-grad is much more memory-intensive than plain attention extraction.
        attn_batch_size_xgrad = int(checkpoint_config.get("attn_batch_size_xgrad", 1))
        attn_batch_size_xgrad = max(1, min(attn_batch_size, attn_batch_size_xgrad))
        with torch.no_grad():
            for batch in test_loader:
                if "label" not in batch:
                    continue
                # Slice on CPU first, then move only the needed samples to GPU.
                video = batch["video"][:attn_batch_size].to(device)
                knowledge_map = batch["knowledge_map"][:attn_batch_size].to(device)
                texts = batch.get("texts", None)
                if texts is not None:
                    texts = texts[:attn_batch_size]
                km_indices = batch.get("km_indices", None)
                video_indices = batch.get("video_indices", None)
                if km_indices is not None:
                    km_indices = km_indices[:attn_batch_size].to(device)
                if video_indices is not None:
                    video_indices = video_indices[:attn_batch_size].to(device)
                # Single-device forward so block attention is for this full batch
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
                        prefix="km_attn",
                    )
                break

        # Fallback explanation: temporal attention x input-gradient saliency
        print("Extracting KM attention x gradient maps and plotting...")
        torch.cuda.empty_cache()
        for batch in test_loader:
            if "label" not in batch:
                continue
            # Keep x-grad memory bounded by processing tiny chunks.
            n_samples = min(attn_batch_size, batch["video"].shape[0])
            all_maps = []
            all_topk = []
            all_logits = []
            km_for_plot = []
            for start in range(0, n_samples, attn_batch_size_xgrad):
                end = min(start + attn_batch_size_xgrad, n_samples)
                video = batch["video"][start:end].to(device)
                knowledge_map = batch["knowledge_map"][start:end].to(device)
                texts = batch.get("texts", None)
                if texts is not None:
                    texts = texts[start:end]
                km_indices = batch.get("km_indices", None)
                video_indices = batch.get("video_indices", None)
                if km_indices is not None:
                    km_indices = km_indices[start:end].to(device)
                if video_indices is not None:
                    video_indices = video_indices[start:end].to(device)

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
                if axg_map is None:
                    continue
                all_maps.append(axg_map)
                all_topk.extend(topk_info)
                all_logits.append(logits_np)
                km_for_plot.append(knowledge_map.detach().cpu())
                torch.cuda.empty_cache()

            if len(all_maps) > 0:
                axg_map_full = np.concatenate(all_maps, axis=0)
                logits_full = np.concatenate(all_logits, axis=0)
                km_plot_tensor = torch.cat(km_for_plot, dim=0)
                plot_attention_x_grad_maps(
                    heatmap_maps=axg_map_full,
                    knowledge_map=km_plot_tensor,
                    save_dir=log_dir,
                    topk_info=all_topk,
                    logits=logits_full,
                    max_samples=min(attn_batch_size, axg_map_full.shape[0]),
                    prefix="km_attn_xgrad",
                )
            else:
                print("Skipping KM attention x grad maps (attention cache unavailable or OOM-safe chunking yielded no maps).")
            break
    else:
        print("\nSkipping attention maps (model has no ViT/PatchViT KM encoder).")

    run_fig5_km_interpretability = True
    subgroup_json_path = os.path.join(current_dir, "data", "subgroup_indices.json")
    test_indices_dk_path = os.path.join(current_dir, "data", "test_indices_dk.json")
    if run_fig5_km_interpretability:
        _run_dk_km_fig5_interpretability(
            raw_model=raw_model,
            test_dataset=test_dataset,
            device=device,
            log_dir=log_dir,
            checkpoint_config=checkpoint_config,
            use_preprocessed_pkl=use_preprocessed_pkl,
            num_workers=num_workers,
            subgroup_json_path=subgroup_json_path,
            test_indices_dk_path=test_indices_dk_path,
            test_binary_threshold=test_binary_threshold,
            cobb_general_threshold=10.0,
            exclude_screening_negative=True,
        )

    print(f"\nResults saved to: {results_file}")
    
    print("\nTest evaluation completed!")


if __name__ == "__main__":
    main()

