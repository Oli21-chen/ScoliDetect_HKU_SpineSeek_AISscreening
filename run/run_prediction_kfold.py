"""
K-fold evaluation script *after* training, using existing checkpoints.

Given a k-fold run directory (e.g. ``checkpoints/kfold5sft_...``) this script will:

1. Rebuild the same subject-level stratified k-fold split on SZ PKL data (config paths + seed).
2. For each fold: load the fold checkpoint, evaluate on the validation subset and external cohort.
3. Print per-fold and aggregate metrics (accuracy, AUC-ROC, sensitivity, specificity, PPV, NPV).
4. Save **model prediction probabilities + labels** in ``plots/<model_name>/`` where
   ``model_name`` is derived from the k-fold run folder (e.g. ``kfold5_kvt_no_gated_token_pooling``).
5. Save ROC figures and any tables in that same subfolder using Nature-style defaults (``plots/rules.text``).
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from utils.data_sampler import (
    SigLIPFullGaitDatasetPKL,
    fullgait_collate_fn,
)
from utils.sft_utils import resolve_device, setup_multi_gpu
from utils.utils import (
    build_subject_map,
    safe_filename_stem,
    model_name_from_run_dir,
    apply_nature_style_mpl,
    save_rows_csv,
    eval_sft_checkpoint_on_dataloader,
)


def main():
    # ------------------------------------------------------------------
    # Configuration (edit if needed)
    # ------------------------------------------------------------------
    ckpt_base_dir = r"/root/private_data/Dong_project/checkpoints/ablation_vivit_kvt_peak_20260522_155514"
    ckpt_filename = "checkpoint_best.pth"
    # Set to e.g. [1] to evaluate/plot only the first fold.
    target_folds: Optional[List[int]] = [1]
    # If True, skip external dataset evaluation/plotting and only process validation folds.
    eval_external = False
    # Outputs: ``<repo>/plots/<model_name>/`` (e.g. plots/kfold5_kvt_no_gated_token_pooling/)
    project_root = current_dir
    plots_dir = os.path.abspath(os.path.join(project_root, "plots"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = safe_filename_stem(os.path.basename(ckpt_base_dir.rstrip("/\\")))
    model_name = model_name_from_run_dir(ckpt_base_dir)
    plots_out_dir = os.path.join(plots_dir, model_name)
    os.makedirs(plots_out_dir, exist_ok=True)
    print(f"Plot & CSV output directory: {plots_out_dir}")
    # File basename inside subfolder: k-fold run folder + timestamp (unique per evaluation)
    output_base = safe_filename_stem(f"{experiment}_{timestamp}")
    classification_prob_threshold = 0.5

    config_path = os.path.join(ckpt_base_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"Loaded config from: {config_path}")
    else:
        fallback_ckpt_path = os.path.join(ckpt_base_dir, "fold_1", ckpt_filename)
        if not os.path.exists(fallback_ckpt_path):
            print(
                f"Error: config.json not found in {ckpt_base_dir}, and fallback checkpoint "
                f"config is unavailable at {fallback_ckpt_path}"
            )
            return
        ckpt_obj = torch.load(fallback_ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt_obj.get("config", {})
        if not cfg:
            print(f"Error: fallback checkpoint has no embedded config: {fallback_ckpt_path}")
            return
        print(f"Loaded embedded config from: {fallback_ckpt_path}")

    n_folds = int(cfg.get("n_folds", 5))
    seed = int(cfg.get("random_seed", 42))
    pkl_data_dirs = cfg.get("pkl_data_dir", ["./data/train_sz_pkl^1", "./data/test_sz_pkl^1"])
    external_dirs = ["./data/test_dk_pkl^1", "./data/test_pk_pkl^1"]
    # external_dirs = cfg.get(
    #     "external_test_dir",
    #     ["./data/test_dk_pkl^1", "./data/test_pk_pkl^1"],
    # )
    binary_threshold = float(cfg.get("binary_threshold", 15.0))
    batch_size = int(cfg.get("batch_size", 8))
    created_at_iso = datetime.now().isoformat()

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
            binary_threshold=binary_threshold,
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
    total_patches_dk = 0
    dk_dataset = None
    if eval_external:
        print("\nBuilding DK external dataset for test metrics...")
        dk_datasets = []
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
                binary_threshold=binary_threshold,
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
    dk_loader = None
    if eval_external and dk_dataset is not None:
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
    val_roc_data = []   # list of (fold_num, fpr, tpr, auc)
    test_roc_data = []  # list of (fold_num, fpr, tpr, auc)
    prediction_rows: List[Dict[str, Any]] = []

    for fold_idx, (train_sub_idx, val_sub_idx) in enumerate(
        skf.split(subject_ids_arr, subject_labels), start=1
    ):
        fold_num = fold_idx
        if target_folds is not None and fold_num not in target_folds:
            continue
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

        # Evaluate on validation fold
        val_metrics, val_probs, val_labels = eval_sft_checkpoint_on_dataloader(
            ckpt_path, val_loader, cfg, device, prob_threshold=classification_prob_threshold
        )
        # Evaluate on external DK
        test_metrics = {}
        test_probs = np.array([])
        test_labels = np.array([])
        if eval_external and dk_loader is not None:
            test_metrics, test_probs, test_labels = eval_sft_checkpoint_on_dataloader(
                ckpt_path, dk_loader, cfg, device, prob_threshold=classification_prob_threshold
            )

        if val_metrics.get("has_labels") and val_probs.size > 0:
            for i in range(len(val_probs)):
                prediction_rows.append(
                    {
                        "created_at": created_at_iso,
                        "experiment": experiment,
                        "fold": fold_num,
                        "split": "val",
                        "sample_index": i,
                        "probability": float(val_probs[i]),
                        "label": float(val_labels[i]),
                        "binary_threshold_cobb": binary_threshold,
                        "classification_prob_threshold": classification_prob_threshold,
                        "checkpoint_path": ckpt_path,
                    }
                )
        if eval_external and test_metrics.get("has_labels") and test_probs.size > 0:
            for i in range(len(test_probs)):
                prediction_rows.append(
                    {
                        "created_at": created_at_iso,
                        "experiment": experiment,
                        "fold": fold_num,
                        "split": "external",
                        "sample_index": i,
                        "probability": float(test_probs[i]),
                        "label": float(test_labels[i]),
                        "binary_threshold_cobb": binary_threshold,
                        "classification_prob_threshold": classification_prob_threshold,
                        "checkpoint_path": ckpt_path,
                    }
                )

        # ROC curves
        if val_metrics.get("has_labels"):
            try:
                fpr_v, tpr_v, _ = roc_curve(val_labels, val_probs)
                val_roc_data.append((fold_num, fpr_v, tpr_v, val_metrics.get("auc_roc", 0.0)))
            except Exception:
                pass
        if eval_external and test_metrics.get("has_labels"):
            try:
                fpr_t, tpr_t, _ = roc_curve(test_labels, test_probs)
                test_roc_data.append((fold_num, fpr_t, tpr_t, test_metrics.get("auc_roc", 0.0)))
            except Exception:
                pass

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
            f"  Val  - Acc={val_metrics.get('accuracy',0):.4f}, "
            f"AUC={val_metrics.get('auc_roc',0):.4f}, "
            f"Sens={val_metrics.get('sensitivity',0):.4f}, "
            f"Spec={val_metrics.get('specificity',0):.4f}, "
            f"PPV={val_metrics.get('ppv',0):.4f}, "
            f"NPV={val_metrics.get('npv',0):.4f}"
        )
        if eval_external:
            print(
                f"  Test - Acc={test_metrics.get('accuracy',0):.4f}, "
                f"AUC={test_metrics.get('auc_roc',0):.4f}, "
                f"Sens={test_metrics.get('sensitivity',0):.4f}, "
                f"Spec={test_metrics.get('specificity',0):.4f}, "
                f"PPV={test_metrics.get('ppv',0):.4f}, "
                f"NPV={test_metrics.get('npv',0):.4f}"
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
            "accuracy": _agg("accuracy", "val"),
            "auc_roc": _agg("auc_roc", "val"),
            "sensitivity": _agg("sensitivity", "val"),
            "specificity": _agg("specificity", "val"),
            "ppv": _agg("ppv", "val"),
            "npv": _agg("npv", "val"),
        },
        "test": {
            "accuracy": _agg("accuracy", "test"),
            "auc_roc": _agg("auc_roc", "test"),
            "sensitivity": _agg("sensitivity", "test"),
            "specificity": _agg("specificity", "test"),
            "ppv": _agg("ppv", "test"),
            "npv": _agg("npv", "test"),
        },
    }

    print("\n--- Aggregate (mean ± std over folds) ---")
    split_iter = [("val", "val")] + ([("external", "test")] if eval_external else [])
    for split_name, key in split_iter:
        print(f"  [{split_name}]")
        for metric in ("accuracy", "auc_roc", "sensitivity", "specificity", "ppv", "npv"):
            a = aggregate[key].get(metric, {})
            if not a:
                continue
            print(
                f"    {metric}: {a.get('mean', 0):.4f} ± {a.get('std', 0):.4f} (n={a.get('n', 0)})"
            )

    # ------------------------------------------------------------------
    # Single CSV: prediction probabilities for AUC-ROC (plots/rules.text)
    # ------------------------------------------------------------------
    pred_csv = os.path.join(plots_out_dir, f"{output_base}_predictions.csv")
    try:
        if prediction_rows:
            save_rows_csv(prediction_rows, pred_csv)
            print(f"\nPrediction probabilities CSV (for AUC-ROC): {pred_csv}")
        else:
            print("\nWarning: no prediction rows to save (no successful fold evaluations).")
    except Exception as e:
        print(f"\nWarning: could not save prediction CSV: {e}")

    # ------------------------------------------------------------------
    # ROC figures (Nature-style); same subfolder as CSV
    # ------------------------------------------------------------------
    apply_nature_style_mpl()
    # ~Nature single-column width (~88 mm)
    fig_inches = 3.46

    if val_roc_data:
        plt.figure(figsize=(fig_inches, fig_inches))
        for fold_num, fpr, tpr, auc in val_roc_data:
            plt.plot(fpr, tpr, label=f"Fold {fold_num} (AUC={auc:.3f})", alpha=0.9)
        plt.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Validation ROC")
        plt.legend(loc="lower right", frameon=False)
        plt.grid(True, alpha=0.25, lw=0.4)
        val_roc_path = os.path.join(plots_out_dir, f"{output_base}_val_roc.png")
        plt.savefig(val_roc_path)
        plt.close()
        print(f"Validation ROC figure: {val_roc_path}")

    if eval_external and test_roc_data:
        plt.figure(figsize=(fig_inches, fig_inches))
        for fold_num, fpr, tpr, auc in test_roc_data:
            plt.plot(fpr, tpr, label=f"Fold {fold_num} (AUC={auc:.3f})", alpha=0.9)
        plt.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("External test ROC")
        plt.legend(loc="lower right", frameon=False)
        plt.grid(True, alpha=0.25, lw=0.4)
        test_roc_path = os.path.join(plots_out_dir, f"{output_base}_test_roc.png")
        plt.savefig(test_roc_path)
        plt.close()
        print(f"External ROC figure: {test_roc_path}")

    if target_folds is not None:
        print(f"\nCompleted selected folds: {target_folds}")
    print("\nK-fold evaluation completed.")


if __name__ == "__main__":
    main()

