"""
K-fold evaluation script *after* Cobb-angle **regression** training, using existing checkpoints.

Mirrors `run_end2end_cobb_kfold.py` (COBB_TASK=regression): predicts continuous **max_cobb**
from the same PKL pipeline (`max_cobb` + `fullgait_collate_fn`).

Given a regression k-fold run directory (e.g. checkpoints from `main_regression()`), this script:

1. Rebuilds the same subject-level stratified 5-fold split on SZ PKL data
   (`pkl_data_dir` from config, default train_sz + test_sz) using `random_seed` from config.
2. For each fold N:
   - Load `fold_N/checkpoint_best_regression.pth` (override via `ckpt_filename`)
   - Evaluate on that fold's **validation subset** (MSE, RMSE, MAE, R², Pearson r)
   - Evaluate on **external** cohorts (`external_dirs` in `main()` — same as before)
3. Saves a JSON report under the k-fold checkpoint directory.
4. Optional: scatter plots predicted vs. true Cobb (val / test per fold).
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from utils.data_sampler import (
    SigLIPFullGaitDatasetPKL,
    fullgait_collate_fn,
)
from utils.sft_utils import resolve_device, setup_multi_gpu
from utils.utils import build_subject_map
from models.sft_regressorv2 import SFTRegressor


def build_model_from_config(cfg: Dict[str, Any]) -> SFTRegressor:
    return SFTRegressor(
        km_feature_dim=cfg.get("km_feature_dim", 238),
        hidden_dim=cfg.get("hidden_dim", 256),
        label_dim=cfg.get("label_dim", 1),
        video_encoder_type=cfg.get("video_encoder_type", "vivit"),
        video_encoder_kwargs=cfg.get("video_encoder_kwargs", {}),
        km_encoder_type=cfg.get("km_encoder_type", "patch_vit"),
        km_encoder_kwargs=cfg.get("km_encoder_kwargs", {}),
        text_model_name=cfg.get("text_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        text_max_length=cfg.get("text_max_length", 128),
        text_trainable=cfg.get("text_trainable", False),
        use_text=cfg.get("use_text", True),
        use_latent_pooling=cfg.get("use_latent_pooling", False),
        latent_pool_size=cfg.get("latent_pool_size", 1),
        regressor_dropout=cfg.get("regressor_dropout", 0.1),
        use_km_video_cross_attn=cfg.get("use_km_video_cross_attn", True),
        cross_attn_num_heads=cfg.get("cross_attn_num_heads", 8),
        cross_attn_drop=cfg.get("cross_attn_drop", 0.1),
        use_gated_token_pooling=cfg.get("use_gated_token_pooling", True),
    )


def compute_classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute accuracy, AUC, sensitivity, specificity, PPV, NPV, confusion matrix."""
    preds = (probs >= threshold).astype(np.float32)
    labels = labels.astype(np.float32)

    tp = float(((preds == 1) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / (total + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)  # recall
    specificity = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)  # precision
    npv = tn / (tn + fn + 1e-8)

    try:
        if len(np.unique(labels)) > 1:
            auc = float(roc_auc_score(labels, probs))
        else:
            auc = 0.0
    except Exception:
        auc = 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "auc_roc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def eval_dataset_for_checkpoint(
    ckpt_path: str,
    dataloader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    prob_threshold: float = 0.5,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate one checkpoint on one dataloader and return metrics + (probs, labels)."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = checkpoint.get("config", cfg)

    # Build and load model
    gpu_ids = ckpt_cfg.get("gpu_ids")
    device = resolve_device(gpu_ids) if device is None else device
    model = build_model_from_config(ckpt_cfg)
    model = setup_multi_gpu(
        model,
        gpu_ids=gpu_ids,
        use_distributed=ckpt_cfg.get("use_distributed", False),
    )
    model = model.to(device)
    model.eval()

    state_dict = checkpoint["model_state_dict"]
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
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
            )  # (B, 1)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

            labels = batch.get("label", None)
            if labels is None:
                # No labels → cannot compute metrics
                continue

            # Robustly convert labels to 1-D numpy array; skip empty/invalid
            try:
                labels_np = labels.detach().cpu().numpy()
                labels_np = labels_np.reshape(-1)
                if labels_np.size == 0:
                    continue
            except Exception:
                continue

            if probs.shape[0] != labels_np.shape[0]:
                # Mismatched batch; skip to avoid shape errors
                continue

            all_probs.append(probs)
            all_labels.append(labels_np)

    if not all_probs or not all_labels:
        return {"has_labels": False}, np.array([]), np.array([])

    probs_concat = np.concatenate(all_probs, axis=0).reshape(-1)
    labels_concat = np.concatenate(all_labels, axis=0).reshape(-1)

    metrics = compute_classification_metrics(
        probs_concat,
        labels_concat,
        threshold=prob_threshold,
    )
    metrics["has_labels"] = True
    metrics["checkpoint_path"] = ckpt_path
    metrics["epoch"] = checkpoint.get("epoch", None)

    return metrics, probs_concat, labels_concat


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """MSE, RMSE, MAE, R², Pearson r for Cobb-angle regression."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {"has_labels": False}

    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    if y_true.size > 1 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        r_pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r_pearson = 0.0

    return {
        "has_labels": True,
        "n_samples": int(y_true.size),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": r_pearson,
    }


def eval_dataset_for_checkpoint_regression(
    ckpt_path: str,
    dataloader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluate one regression checkpoint on one dataloader.
    Targets come from batch['max_cobb']; predictions are raw model outputs (degrees).
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = checkpoint.get("config", cfg)

    gpu_ids = ckpt_cfg.get("gpu_ids")
    device = resolve_device(gpu_ids) if device is None else device
    model = build_model_from_config(ckpt_cfg)
    model = setup_multi_gpu(
        model,
        gpu_ids=gpu_ids,
        use_distributed=ckpt_cfg.get("use_distributed", False),
    )
    model = model.to(device)
    model.eval()

    state_dict = checkpoint["model_state_dict"]
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
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
                continue

            out = model(
                video,
                knowledge_map,
                texts,
                km_indices=km_indices,
                video_indices=video_indices,
            )
            preds = out.squeeze(-1).cpu().numpy()

            targets = batch["max_cobb"].detach().cpu().numpy().reshape(-1)

            if preds.ndim > 1:
                preds = preds.reshape(-1)
            if preds.shape[0] != targets.shape[0]:
                continue

            all_preds.append(preds)
            all_targets.append(targets)

    if not all_preds or not all_targets:
        return {"has_labels": False}, np.array([]), np.array([])

    pred_concat = np.concatenate(all_preds, axis=0).reshape(-1)
    target_concat = np.concatenate(all_targets, axis=0).reshape(-1)

    metrics = compute_regression_metrics(target_concat, pred_concat)
    metrics["checkpoint_path"] = ckpt_path
    metrics["epoch"] = checkpoint.get("epoch", None)

    return metrics, pred_concat, target_concat


def main():
    # ------------------------------------------------------------------
    # Configuration (edit if needed)
    # ------------------------------------------------------------------
    ckpt_base_dir = r"C:\Users\Administrator\project\checkpoints\kfold5Cobb_kvt_4layer8_512_1e4_20260319_181946"
    ckpt_filename = "checkpoint_best_regression.pth"

    cfg = None
    for config_name in ("config_regression.json", "config.json"):
        config_path = os.path.join(ckpt_base_dir, config_name)
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            print(f"Loaded config: {config_path}")
            break
    if cfg is None:
        print(f"Error: neither config_regression.json nor config.json found in {ckpt_base_dir}")
        return

    n_folds = int(cfg.get("n_folds", 5))
    seed = int(cfg.get("random_seed", 42))
    pkl_data_dirs = cfg.get("pkl_data_dir", ["./data/train_sz_pkl^1", "./data/test_sz_pkl^1"])
    # Same external layout as before (edit list here if paths change)
    external_dirs = ["./data/test_dk_pkl^1", "./data/test_pk_pkl^1"]
    binary_threshold = float(cfg.get("binary_threshold", 15.0))
    batch_size = int(cfg.get("batch_size", 8))

    # Save JSON **inside** k-fold checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = os.path.join(ckpt_base_dir, f"kfold_regression_eval_val_test_{timestamp}.json")

    # ------------------------------------------------------------------
    # Build SZ full dataset (train_sz_pkl^1 + test_sz_pkl^1)
    # ------------------------------------------------------------------
    print("Building SZ full dataset for internal validation metrics...")
    sz_datasets = []
    total_patches_sz = 0
    for d in pkl_data_dirs:
        d_abs = os.path.abspath(os.path.normpath(d))
        if not os.path.exists(d_abs):
            raise FileNotFoundError(f"PKL data dir not found: {d_abs}")
        print(f"  SZ PKL dir: {d_abs}")
        ds = SigLIPFullGaitDatasetPKL(
            pkl_data_dir=d_abs,
            km_gaussian_noise_std=None,
            mode="test",
            prompts_path=cfg.get("prompts_path"),
            prompt_selection=cfg.get("prompt_selection", "concise_prompts"),
            binary_threshold=None,
        )
        print(f"    -> {len(ds)} patches")
        sz_datasets.append(ds)
        total_patches_sz += len(ds)

    if len(sz_datasets) == 1:
        full_sz_dataset = sz_datasets[0]
    else:
        full_sz_dataset = ConcatDataset(sz_datasets)
    print(f"Total SZ patches: {total_patches_sz}")

    # Subject-level map and labels
    subject_ids, subject_labels, subject_to_indices = build_subject_map(
        full_sz_dataset, binary_threshold
    )
    print(f"Unique SZ subjects: {len(subject_ids)}")

    # ------------------------------------------------------------------
    # Build external DK dataset
    # ------------------------------------------------------------------
    print("\nBuilding DK external dataset for test metrics...")
    dk_datasets = []
    total_patches_dk = 0
    for d in external_dirs:
        d_abs = os.path.abspath(os.path.normpath(d))
        if not os.path.exists(d_abs):
            raise FileNotFoundError(f"External PKL data dir not found: {d_abs}")
        print(f"  DK PKL dir: {d_abs}")
        ds = SigLIPFullGaitDatasetPKL(
            pkl_data_dir=d_abs,
            km_gaussian_noise_std=None,
            mode="test",
            prompts_path=cfg.get("prompts_path"),
            prompt_selection=cfg.get("prompt_selection", "concise_prompts"),
            binary_threshold=None,
        )
        print(f"    -> {len(ds)} patches")
        dk_datasets.append(ds)
        total_patches_dk += len(ds)

    if len(dk_datasets) == 1:
        dk_dataset = dk_datasets[0]
    else:
        dk_dataset = ConcatDataset(dk_datasets)
    print(f"Total DK patches: {total_patches_dk}")

    device = resolve_device(cfg.get("gpu_ids"))
    print(f"\nUsing device: {device}")

    # DataLoader for DK (same for all folds)
    dk_loader = DataLoader(
        dk_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        collate_fn=fullgait_collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    # ------------------------------------------------------------------
    # Rebuild k-fold subject-level splits
    # ------------------------------------------------------------------
    print("\nRebuilding subject-level stratified k-fold splits...")
    subject_ids_arr = np.array(subject_ids)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    per_fold_results: List[Dict[str, Any]] = []
    val_scatter_series: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
    test_scatter_series: List[Tuple[int, np.ndarray, np.ndarray, float]] = []

    for fold_idx, (train_sub_idx, val_sub_idx) in enumerate(
        skf.split(subject_ids_arr, subject_labels), start=1
    ):
        fold_num = fold_idx
        print(f"\n{'='*70}\n  FOLD {fold_num}/{n_folds}\n{'='*70}")

        train_subjects = set(subject_ids_arr[train_sub_idx])
        val_subjects = set(subject_ids_arr[val_sub_idx])
        print(f"  Train subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)}")

        val_indices: List[int] = []
        for sid in val_subjects:
            val_indices.extend(subject_to_indices[sid])
        print(f"  Val patches: {len(val_indices)}")

        val_dataset = Subset(full_sz_dataset, val_indices)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.get("num_workers", 2),
            collate_fn=fullgait_collate_fn,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )

        # Checkpoint path for this fold
        ckpt_path = os.path.join(ckpt_base_dir, f"fold_{fold_num}", ckpt_filename)
        if not os.path.exists(ckpt_path):
            print(f"  Warning: checkpoint not found for fold {fold_num}: {ckpt_path}")
            continue
        print(f"  Using checkpoint: {ckpt_path}")

        val_metrics, val_pred, val_true = eval_dataset_for_checkpoint_regression(
            ckpt_path, val_loader, cfg, device
        )
        test_metrics, test_pred, test_true = eval_dataset_for_checkpoint_regression(
            ckpt_path, dk_loader, cfg, device
        )

        if val_metrics.get("has_labels") and val_true.size > 0:
            val_scatter_series.append(
                (fold_num, val_true, val_pred, float(val_metrics.get("pearson_r", 0.0)))
            )
        if test_metrics.get("has_labels") and test_true.size > 0:
            test_scatter_series.append(
                (fold_num, test_true, test_pred, float(test_metrics.get("pearson_r", 0.0)))
            )

        per_fold_results.append(
            {
                "fold": fold_num,
                "val": val_metrics,
                "test": test_metrics,
                "num_val_patches": len(val_indices),
                "num_test_patches": total_patches_dk,
            }
        )

        print(
            f"  Val  - RMSE={val_metrics.get('rmse',0):.4f}, "
            f"MAE={val_metrics.get('mae',0):.4f}, "
            f"R²={val_metrics.get('r2',0):.4f}, "
            f"r={val_metrics.get('pearson_r',0):.4f}"
        )
        print(
            f"  Test - RMSE={test_metrics.get('rmse',0):.4f}, "
            f"MAE={test_metrics.get('mae',0):.4f}, "
            f"R²={test_metrics.get('r2',0):.4f}, "
            f"r={test_metrics.get('pearson_r',0):.4f}"
        )

    # ------------------------------------------------------------------
    # Aggregate metrics across folds
    # ------------------------------------------------------------------
    def _agg(metric_name: str, split: str) -> Dict[str, float]:
        vals = [
            float(fr[split].get(metric_name))
            for fr in per_fold_results
            if fr[split].get("has_labels") and metric_name in fr[split]
        ]
        if not vals:
            return {}
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return {"mean": mean_v, "std": std_v, "n": len(vals)}

    aggregate = {
        "val": {
            "mse": _agg("mse", "val"),
            "rmse": _agg("rmse", "val"),
            "mae": _agg("mae", "val"),
            "r2": _agg("r2", "val"),
            "pearson_r": _agg("pearson_r", "val"),
        },
        "test": {
            "mse": _agg("mse", "test"),
            "rmse": _agg("rmse", "test"),
            "mae": _agg("mae", "test"),
            "r2": _agg("r2", "test"),
            "pearson_r": _agg("pearson_r", "test"),
        },
    }

    output_payload: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "ckpt_base_dir": ckpt_base_dir,
        "ckpt_filename": ckpt_filename,
        "n_folds": n_folds,
        "random_seed": seed,
        "pkl_data_dir": pkl_data_dirs,
        "external_test_dir": external_dirs,
        "binary_threshold": binary_threshold,
        "task": "regression_max_cobb",
        "per_fold_results": per_fold_results,
        "aggregate": aggregate,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nJSON results saved to: {output_json_path}")

    # ------------------------------------------------------------------
    # Visualizations: predicted vs. true Cobb (regression)
    # ------------------------------------------------------------------
    def _plot_scatter(
        series: List[Tuple[int, np.ndarray, np.ndarray, float]],
        title: str,
        out_name: str,
    ) -> None:
        if not series:
            return
        plt.figure(figsize=(6, 6))
        for fold_num, yt, yp, r in series:
            plt.scatter(yt, yp, s=8, alpha=0.5, label=f"Fold {fold_num} (r={r:.3f})")
        # Identity line
        all_y = np.concatenate([np.r_[s[1], s[2]] for s in series])
        lo, hi = float(np.min(all_y)), float(np.max(all_y))
        pad = max(1.0, (hi - lo) * 0.05)
        plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.6, label="Identity")
        plt.xlabel("True max Cobb (°)")
        plt.ylabel("Predicted max Cobb (°)")
        plt.title(title)
        plt.legend(loc="upper left", fontsize=7)
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        out_path = os.path.join(ckpt_base_dir, out_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Figure saved to: {out_path}")

    _plot_scatter(val_scatter_series, "Validation: pred vs. true (SZ)", "kfold_regression_val_scatter.png")
    _plot_scatter(test_scatter_series, "External: pred vs. true", "kfold_regression_test_scatter.png")

    print("\nK-fold regression evaluation completed.")


if __name__ == "__main__":
    main()

