"""
Subgroup inference + DK stratification plots (Nature-style ROC / confusion).

Mirrors ``run_prediction_kfold.py``:

1. Load ``config.json`` from a k-fold run directory (e.g. ``checkpoints/kfold5sft_vivit``).
2. Load checkpoint(s): either **one fold** or **all folds** (``RUN['use_all_folds']``).
3. Build **DK-only** ``SigLIPFullGaitDatasetPKL`` from ``RUN['dk_pkl_dir']``.
4. Stratify by ``data/subgroup_dk.json``; run ``eval_sft_checkpoint_on_dataloader`` per stratum.
5. ROC: if all folds — **mean TPR(FPR) ± s.d.** across folds + **mean AUC ± s.d.**; else single ROC per stratum.
6. Confusion: if all folds — **ensemble** (mean probability across folds, then threshold); else one fold.

Edit ``RUN`` below, then::

  python tools/subgroup_analyze.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
from torch.utils.data import DataLoader, Subset

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_sampler import SigLIPFullGaitDatasetPKL, fullgait_collate_fn
from utils.sft_utils import resolve_device
from utils.subgroup_dk_indices import (
    build_dk_strata_indices,
    load_subgroup_dk_rows,
    warn_oob_subgroup_rows,
)
from utils.utils import eval_sft_checkpoint_on_dataloader

PROJECT_ROOT = parent_dir
DEFAULT_CHECKPOINT_RUN_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "kfold5_kvt_vivit_20260408_172609")
DEFAULT_DK_PKL_DIR = os.path.join(PROJECT_ROOT, "data", "test_dk_pkl^1")
DEFAULT_SUBGROUP_JSON = os.path.join(PROJECT_ROOT, "data", "subgroup_dk.json")
DEFAULT_OUT_DIR = os.path.join(current_dir, "subgroup_plots", "kfold5_kvt_vivit")

# Nature-family matplotlib defaults (single place for figure typography / spines)
NATURE_JOURNAL_RCPARAMS: Dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.2,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
}


def apply_subgroup_nature_style() -> None:
    plt.rcParams.update(NATURE_JOURNAL_RCPARAMS)


# --- Edit this block ---
RUN: Dict[str, Any] = {
    "checkpoint_run_dir": DEFAULT_CHECKPOINT_RUN_DIR,
    "use_all_folds": True,  # False → only ``fold`` below
    "fold": 1,
    "ckpt_filename": "checkpoint_best.pth",
    "dk_pkl_dir": DEFAULT_DK_PKL_DIR,
    "subgroup_json": DEFAULT_SUBGROUP_JSON,
    "out_dir": DEFAULT_OUT_DIR,
    "classification_prob_threshold": 0.5,
    "batch_size": None,  # None → use config.json batch_size
}

# Okabe–Ito (colourblind-safe)
_ROC_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")
_ROC_SINGLE_COL_W = 3.425
_ROC_FPR_GRID = np.linspace(0.0, 1.0, 101)


def _roc_safe(
    y_true: np.ndarray, y_score: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, float(auc(fpr, tpr))


def _interp_tpr_on_grid(fpr: np.ndarray, tpr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """TPR as a function of FPR on ``grid`` (for aggregating ROCs across folds)."""
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    if fpr.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    return np.interp(grid, fpr, tpr, left=0.0, right=1.0)


def plot_roc_curves(
    series: List[Tuple[str, np.ndarray, np.ndarray, float]],
    title: str,
    out_path: str,
) -> None:
    """One ROC line per stratum (single-fold mode)."""
    w = _ROC_SINGLE_COL_W
    fig, ax = plt.subplots(figsize=(w, w), facecolor="white")
    for i, (name, fpr, tpr, a) in enumerate(series):
        c = _ROC_COLORS[i % len(_ROC_COLORS)]
        ax.plot(
            fpr,
            tpr,
            color=c,
            solid_capstyle="round",
            label=f"{name}, AUC = {a:.2f}",
        )
    ax.plot(
        [0, 1],
        [0, 1],
        color="#BBBBBB",
        ls="--",
        lw=0.75,
        dashes=(3, 2),
        label="Chance",
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontweight="normal", pad=5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="both", which="major", length=3.2, width=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22, linewidth=0.4, linestyle="-")
    ax.legend(loc="lower right", handlelength=1.8, borderaxespad=0.5)
    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def plot_roc_mean_std(
    strata: List[Dict[str, Any]],
    title: str,
    out_path: str,
) -> None:
    """
    Each stratum dict: name, fpr_grid, mean_tpr, std_tpr, auc_mean, auc_std, n_folds.
    Shaded band = mean_tpr ± std_tpr across folds at each FPR grid point.
    """
    w = _ROC_SINGLE_COL_W
    fig, ax = plt.subplots(figsize=(w, w), facecolor="white")
    for i, s in enumerate(strata):
        c = _ROC_COLORS[i % len(_ROC_COLORS)]
        g = s["fpr_grid"]
        m = s["mean_tpr"]
        sd = s["std_tpr"]
        am = s["auc_mean"]
        astd = s["auc_std"]
        nf = int(s["n_folds"])
        label = f"{s['name']}, AUC = {am:.2f} ± {astd:.2f} (n={nf})"
        ax.fill_between(
            g,
            np.clip(m - sd, 0, 1),
            np.clip(m + sd, 0, 1),
            color=c,
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(g, m, color=c, solid_capstyle="round", label=label)
    ax.plot(
        [0, 1],
        [0, 1],
        color="#BBBBBB",
        ls="--",
        lw=0.75,
        dashes=(3, 2),
        label="Chance",
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontweight="normal", pad=5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="both", which="major", length=3.2, width=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22, linewidth=0.4, linestyle="-")
    ax.legend(loc="lower right", handlelength=1.8, borderaxespad=0.5)
    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def plot_confusion_pair(
    matrices: List[Tuple[str, np.ndarray]],
    out_path: str,
    suptitle: str,
) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.2 * len(matrices), 1.65))
    if len(matrices) == 1:
        axes = [axes]
    vmax = max(float(m[1].max()) for m in matrices) if matrices else 1.0
    vmax = max(vmax, 1.0)
    for ax, (name, cm) in zip(axes, matrices):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for (j, i), v in np.ndenumerate(cm):
            ax.text(i, j, int(v), ha="center", va="center", color="black", fontsize=7)
        ax.set_title(name, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle(suptitle, fontsize=7, y=1.08, fontweight="normal")
    plt.tight_layout()
    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def plot_confusion_grid(
    matrices: List[Tuple[str, np.ndarray]],
    out_path: str,
    suptitle: str,
) -> None:
    """2×2 (or partial) confusion subplots for up to four named strata."""
    n = len(matrices)
    if n == 0:
        return
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 1.65 * nrows), facecolor="white")
    axes_flat = np.atleast_1d(axes).ravel()
    vmax = max(float(m[1].max()) for m in matrices) if matrices else 1.0
    vmax = max(vmax, 1.0)
    for k in range(nrows * ncols):
        ax = axes_flat[k]
        if k < n:
            name, cm = matrices[k]
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticklabels(["True 0", "True 1"])
            for (j, i), v in np.ndenumerate(cm):
                ax.text(i, j, int(v), ha="center", va="center", color="black", fontsize=7)
            ax.set_title(name, fontsize=7)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        else:
            ax.axis("off")
    fig.suptitle(suptitle, fontsize=7, y=1.02, fontweight="normal")
    plt.tight_layout()
    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def _binary_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = cm.ravel()
    return {
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "n": float(tn + fp + fn + tp),
    }


def _eval_stratum(
    name: str,
    ckpt_path: str,
    dk_dataset: Any,
    indices: List[int],
    cfg: Dict[str, Any],
    device,
    prob_threshold: float,
    batch_size: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    if not indices:
        return {"has_labels": False}, np.array([]), np.array([])
    sub = Subset(dk_dataset, indices)
    loader = DataLoader(
        sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=fullgait_collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    print(f"  [{name}] patches={len(indices)} ← {os.path.basename(os.path.dirname(ckpt_path))}")
    return eval_sft_checkpoint_on_dataloader(
        ckpt_path, loader, cfg, device, prob_threshold=prob_threshold
    )


def _run_stratum_all_folds(
    stratum_key: str,
    ckpt_root: str,
    ckpt_name: str,
    fold_nums: List[int],
    dk_dataset: Any,
    indices: List[int],
    cfg: Dict[str, Any],
    device,
    prob_threshold: float,
    batch_size: int,
) -> Dict[str, Any]:
    """Collect probs per fold, interpolate TPR on grid, ensemble mean prob for confusion."""
    fold_probs: List[np.ndarray] = []
    tpr_rows: List[np.ndarray] = []
    aucs: List[float] = []
    y_true: Optional[np.ndarray] = None
    used_folds: List[int] = []

    for fold in fold_nums:
        ckpt_path = os.path.join(ckpt_root, f"fold_{fold}", ckpt_name)
        if not os.path.isfile(ckpt_path):
            print(f"  Warning: missing checkpoint, skip fold {fold}: {ckpt_path}")
            continue
        m, p, y = _eval_stratum(
            stratum_key,
            ckpt_path,
            dk_dataset,
            indices,
            cfg,
            device,
            prob_threshold,
            batch_size,
        )
        if p.size == 0:
            continue
        if y_true is None:
            y_true = y
        elif not np.array_equal(np.asarray(y_true).astype(int), np.asarray(y).astype(int)):
            raise RuntimeError(f"Label mismatch across folds for {stratum_key}")
        fold_probs.append(p.reshape(-1))
        used_folds.append(fold)
        roc = _roc_safe(y, p)
        if roc is not None:
            fpr, tpr, a = roc
            tpr_rows.append(_interp_tpr_on_grid(fpr, tpr, _ROC_FPR_GRID))
            aucs.append(a)

    if not fold_probs or y_true is None:
        return {
            "has_data": False,
            "y_true": np.array([]),
            "fold_probs": [],
            "mean_tpr": None,
            "std_tpr": None,
            "auc_mean": None,
            "auc_std": None,
            "n_folds_used": 0,
        }

    if not tpr_rows:
        mean_tpr, std_tpr = None, None
    else:
        tpr_mat = np.stack(tpr_rows, axis=0)
        if tpr_mat.shape[0] == 1:
            mean_tpr = tpr_mat[0].astype(np.float64)
            std_tpr = np.zeros(len(_ROC_FPR_GRID), dtype=np.float64)
        else:
            mean_tpr = tpr_mat.mean(axis=0)
            std_tpr = tpr_mat.std(axis=0, ddof=1)
    auc_mean = float(np.mean(aucs)) if aucs else None
    auc_std = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0

    return {
        "has_data": True,
        "y_true": np.asarray(y_true).reshape(-1),
        "fold_probs": fold_probs,
        "fpr_grid": _ROC_FPR_GRID,
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "aucs_per_fold": aucs,
        "folds_used": used_folds,
        "n_folds_used": len(fold_probs),
    }


def main() -> None:
    cfg_run = RUN
    ckpt_root = os.path.abspath(os.path.normpath(str(cfg_run["checkpoint_run_dir"])))
    use_all_folds = bool(cfg_run.get("use_all_folds", False))
    fold_single = int(cfg_run.get("fold", 1))
    ckpt_name = str(cfg_run["ckpt_filename"])
    dk_dir = os.path.abspath(os.path.normpath(str(cfg_run["dk_pkl_dir"])))
    subgroup_path = os.path.abspath(os.path.normpath(str(cfg_run["subgroup_json"])))
    out_dir = os.path.abspath(os.path.normpath(str(cfg_run["out_dir"])))
    prob_thr = float(cfg_run.get("classification_prob_threshold", 0.5))

    config_path = os.path.join(ckpt_root, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    batch_size = cfg_run.get("batch_size")
    if batch_size is None:
        batch_size = int(cfg.get("batch_size", 8))
    else:
        batch_size = int(batch_size)

    n_folds_cfg = int(cfg.get("n_folds", 5))
    fold_nums = list(range(1, n_folds_cfg + 1)) if use_all_folds else [fold_single]

    apply_subgroup_nature_style()
    os.makedirs(out_dir, exist_ok=True)

    device = resolve_device(cfg.get("gpu_ids"))
    binary_threshold = float(cfg.get("binary_threshold", 15.0))

    print(f"Checkpoint root: {ckpt_root}")
    print(f"Folds: {fold_nums} ({'all' if use_all_folds else 'single-fold'})")
    print(f"Device: {device}")
    print(f"DK PKL dir: {dk_dir}")
    print(f"Subgroup JSON: {subgroup_path}")
    print(f"Output: {out_dir}")

    dk_dataset = SigLIPFullGaitDatasetPKL(
        pkl_data_dir=dk_dir,
        km_gaussian_noise_std=None,
        mode="test",
        prompts_path=cfg.get("prompts_path"),
        prompt_selection=cfg.get("prompt_selection", "concise_prompts"),
        binary_threshold=binary_threshold,
    )
    n = len(dk_dataset)
    print(f"DK patches in dataset: {n}")

    subgroup_rows = load_subgroup_dk_rows(subgroup_path)
    warn_oob_subgroup_rows(subgroup_rows, n)
    indices, strata_meta = build_dk_strata_indices(subgroup_rows, n)

    idx_gen = indices["general_cobb_gt10"]
    idx_single = indices["single"]
    idx_multi = indices["multi"]
    idx_th = indices["single_thoracic"]
    idx_lb = indices["single_lumbar"]

    summary: Dict[str, Any] = {
        "checkpoint_run_dir": ckpt_root,
        "use_all_folds": use_all_folds,
        "folds_evaluated": fold_nums,
        "dk_pkl_dir": dk_dir,
        "n_dk_patches": n,
        "classification_prob_threshold": prob_thr,
        "binary_threshold_cobb": binary_threshold,
        "strata_patch_counts": {
            "general_cobb_gt10": len(idx_gen),
            "single": len(idx_single),
            "multi": len(idx_multi),
            "single_thoracic": len(idx_th),
            "single_lumbar": len(idx_lb),
        },
        "subgroup_strata_meta": strata_meta,
    }

    if use_all_folds:
        agg_s = _run_stratum_all_folds(
            "single", ckpt_root, ckpt_name, fold_nums, dk_dataset, idx_single, cfg, device, prob_thr, batch_size
        )
        agg_m = _run_stratum_all_folds(
            "multi", ckpt_root, ckpt_name, fold_nums, dk_dataset, idx_multi, cfg, device, prob_thr, batch_size
        )
        agg_th = _run_stratum_all_folds(
            "single/thoracic",
            ckpt_root,
            ckpt_name,
            fold_nums,
            dk_dataset,
            idx_th,
            cfg,
            device,
            prob_thr,
            batch_size,
        )
        agg_lb = _run_stratum_all_folds(
            "single/lumbar",
            ckpt_root,
            ckpt_name,
            fold_nums,
            dk_dataset,
            idx_lb,
            cfg,
            device,
            prob_thr,
            batch_size,
        )
        agg_gen = _run_stratum_all_folds(
            "general_cobb_gt10",
            ckpt_root,
            ckpt_name,
            fold_nums,
            dk_dataset,
            idx_gen,
            cfg,
            device,
            prob_thr,
            batch_size,
        )

        # ----- Chart 1 ROC (mean ± s.d. TPR; AUC mean ± s.d.) -----
        roc1_strata: List[Dict[str, Any]] = []
        for label, agg in (("Single curve", agg_s), ("Multi curve", agg_m)):
            if not agg["has_data"] or agg["mean_tpr"] is None or agg["auc_mean"] is None:
                continue
            roc1_strata.append(
                {
                    "name": label,
                    "fpr_grid": agg["fpr_grid"],
                    "mean_tpr": agg["mean_tpr"],
                    "std_tpr": agg["std_tpr"] if agg["std_tpr"] is not None else np.zeros_like(agg["mean_tpr"]),
                    "auc_mean": agg["auc_mean"],
                    "auc_std": agg["auc_std"],
                    "n_folds": agg["n_folds_used"],
                }
            )
            key = f"chart1_{label.lower().replace(' ', '_')}"
            summary[key] = {
                "n_folds_used": agg["n_folds_used"],
                "folds_used": agg["folds_used"],
                "auc_mean": agg["auc_mean"],
                "auc_std": agg["auc_std"],
                "aucs_per_fold": agg["aucs_per_fold"],
            }

        if roc1_strata:
            plot_roc_mean_std(
                roc1_strata,
                "ROC by curve type (mean ± s.d. over folds)",
                os.path.join(out_dir, "subgroup_chart1_single_multi_roc.png"),
            )

        # Chart 1 confusion: ensemble mean probability
        cm1: List[Tuple[str, np.ndarray]] = []
        for label, agg in (("Single curve", agg_s), ("Multi curve", agg_m)):
            if not agg["has_data"] or not agg["fold_probs"]:
                continue
            key = f"chart1_{label.lower().replace(' ', '_')}"
            p_bar = np.mean(np.stack(agg["fold_probs"], axis=0), axis=0)
            yt = agg["y_true"].astype(int)
            y_pred = (p_bar >= prob_thr).astype(int)
            cm = confusion_matrix(yt, y_pred, labels=[0, 1])
            cm1.append((label, cm))
            summary[key]["confusion_ensemble_mean_prob"] = _binary_metrics_from_cm(cm)

        if cm1:
            plot_confusion_pair(
                cm1,
                os.path.join(out_dir, "subgroup_chart1_single_multi_confusion.png"),
                f"Confusion @ prob ≥ {prob_thr:.3f} (ensemble mean over folds)",
            )

        # ----- Chart 2 -----
        roc2_strata: List[Dict[str, Any]] = []
        for label, agg in (("Thoracic", agg_th), ("Lumbar", agg_lb)):
            if not agg["has_data"] or agg["mean_tpr"] is None or agg["auc_mean"] is None:
                continue
            roc2_strata.append(
                {
                    "name": f"Single / {label}",
                    "fpr_grid": agg["fpr_grid"],
                    "mean_tpr": agg["mean_tpr"],
                    "std_tpr": agg["std_tpr"] if agg["std_tpr"] is not None else np.zeros_like(agg["mean_tpr"]),
                    "auc_mean": agg["auc_mean"],
                    "auc_std": agg["auc_std"],
                    "n_folds": agg["n_folds_used"],
                }
            )
            key = f"chart2_{label.lower()}"
            summary[key] = {
                "n_folds_used": agg["n_folds_used"],
                "folds_used": agg["folds_used"],
                "auc_mean": agg["auc_mean"],
                "auc_std": agg["auc_std"],
                "aucs_per_fold": agg["aucs_per_fold"],
            }

        if roc2_strata:
            plot_roc_mean_std(
                roc2_strata,
                "ROC: thoracic vs lumbar (mean ± s.d. over folds)",
                os.path.join(out_dir, "subgroup_chart2_single_thoracic_lumbar_roc.png"),
            )

        cm2: List[Tuple[str, np.ndarray]] = []
        for label, agg in (("Thoracic", agg_th), ("Lumbar", agg_lb)):
            if not agg["has_data"] or not agg["fold_probs"]:
                continue
            key = f"chart2_{label.lower()}"
            p_bar = np.mean(np.stack(agg["fold_probs"], axis=0), axis=0)
            yt = agg["y_true"].astype(int)
            y_pred = (p_bar >= prob_thr).astype(int)
            cm = confusion_matrix(yt, y_pred, labels=[0, 1])
            cm2.append((f"Single / {label}", cm))
            summary[key]["confusion_ensemble_mean_prob"] = _binary_metrics_from_cm(cm)

        if cm2:
            plot_confusion_pair(
                cm2,
                os.path.join(out_dir, "subgroup_chart2_single_thoracic_lumbar_confusion.png"),
                f"Confusion @ prob ≥ {prob_thr:.3f} (single only, ensemble mean)",
            )

        # ----- Chart 3: four clinical strata (general / single-th / single-lb / multi) -----
        roc3_strata: List[Dict[str, Any]] = []
        chart3_defs = (
            ("General (Cobb>10)", agg_gen),
            ("Single thoracic", agg_th),
            ("Single lumbar", agg_lb),
            ("Multi-curve", agg_m),
        )
        for label, agg in chart3_defs:
            if not agg["has_data"] or agg["mean_tpr"] is None or agg["auc_mean"] is None:
                continue
            key_slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("°", "")
            roc3_strata.append(
                {
                    "name": label,
                    "fpr_grid": agg["fpr_grid"],
                    "mean_tpr": agg["mean_tpr"],
                    "std_tpr": agg["std_tpr"] if agg["std_tpr"] is not None else np.zeros_like(agg["mean_tpr"]),
                    "auc_mean": agg["auc_mean"],
                    "auc_std": agg["auc_std"],
                    "n_folds": agg["n_folds_used"],
                }
            )
            summary[f"chart3_{key_slug}"] = {
                "n_folds_used": agg["n_folds_used"],
                "folds_used": agg["folds_used"],
                "auc_mean": agg["auc_mean"],
                "auc_std": agg["auc_std"],
                "aucs_per_fold": agg["aucs_per_fold"],
            }

        if roc3_strata:
            plot_roc_mean_std(
                roc3_strata,
                "ROC: DK JSON strata (mean ± s.d. over folds)",
                os.path.join(out_dir, "subgroup_chart3_four_strata_roc.png"),
            )

        cm3: List[Tuple[str, np.ndarray]] = []
        for label, agg in chart3_defs:
            if not agg["has_data"] or not agg["fold_probs"]:
                continue
            key_slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("°", "")
            chart3_key = f"chart3_{key_slug}"
            if chart3_key not in summary:
                summary[chart3_key] = {}
            p_bar = np.mean(np.stack(agg["fold_probs"], axis=0), axis=0)
            yt = agg["y_true"].astype(int)
            y_pred = (p_bar >= prob_thr).astype(int)
            cm = confusion_matrix(yt, y_pred, labels=[0, 1])
            cm3.append((label, cm))
            summary[chart3_key]["confusion_ensemble_mean_prob"] = _binary_metrics_from_cm(cm)

        if cm3:
            plot_confusion_grid(
                cm3,
                os.path.join(out_dir, "subgroup_chart3_four_strata_confusion.png"),
                f"Confusion @ prob ≥ {prob_thr:.3f} (four strata, ensemble mean)",
            )

    else:
        ckpt_path = os.path.join(ckpt_root, f"fold_{fold_single}", ckpt_name)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        summary["checkpoint_path"] = ckpt_path

        m_s, p_s, y_s = _eval_stratum(
            "single", ckpt_path, dk_dataset, idx_single, cfg, device, prob_thr, batch_size
        )
        m_m, p_m, y_m = _eval_stratum(
            "multi", ckpt_path, dk_dataset, idx_multi, cfg, device, prob_thr, batch_size
        )
        m_th, p_th, y_th = _eval_stratum(
            "single/thoracic", ckpt_path, dk_dataset, idx_th, cfg, device, prob_thr, batch_size
        )
        m_lb, p_lb, y_lb = _eval_stratum(
            "single/lumbar", ckpt_path, dk_dataset, idx_lb, cfg, device, prob_thr, batch_size
        )
        m_gen, p_gen, y_gen = _eval_stratum(
            "general_cobb_gt10", ckpt_path, dk_dataset, idx_gen, cfg, device, prob_thr, batch_size
        )

        roc1: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
        cm1: List[Tuple[str, np.ndarray]] = []
        for label, probs, labels, mets in (
            ("Single curve", p_s, y_s, m_s),
            ("Multi curve", p_m, y_m, m_m),
        ):
            key = f"chart1_{label.lower().replace(' ', '_')}"
            summary[key] = {"n_predictions": int(probs.size)}
            if mets.get("has_labels"):
                summary[key]["auc_roc"] = mets.get("auc_roc")
                summary[key]["accuracy"] = mets.get("accuracy")
            roc = _roc_safe(labels, probs) if probs.size else None
            if roc is not None:
                fpr, tpr, a = roc
                roc1.append((label, fpr, tpr, a))
            if probs.size:
                y_pred = (probs >= prob_thr).astype(int)
                yt = labels.astype(int)
                cm = confusion_matrix(yt, y_pred, labels=[0, 1])
                cm1.append((label, cm))
                summary[key].update(_binary_metrics_from_cm(cm))

        if roc1:
            plot_roc_curves(
                roc1,
                "ROC by curve type (DK subgroup)",
                os.path.join(out_dir, "subgroup_chart1_single_multi_roc.png"),
            )
        if cm1:
            plot_confusion_pair(
                cm1,
                os.path.join(out_dir, "subgroup_chart1_single_multi_confusion.png"),
                f"Confusion @ prob ≥ {prob_thr:.3f}",
            )

        roc2: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
        cm2: List[Tuple[str, np.ndarray]] = []
        for label, probs, labels, mets in (
            ("Thoracic", p_th, y_th, m_th),
            ("Lumbar", p_lb, y_lb, m_lb),
        ):
            key = f"chart2_{label.lower()}"
            summary[key] = {"n_predictions": int(probs.size)}
            if mets.get("has_labels"):
                summary[key]["auc_roc"] = mets.get("auc_roc")
            roc = _roc_safe(labels, probs) if probs.size else None
            if roc is not None:
                fpr, tpr, a = roc
                roc2.append((f"Single / {label}", fpr, tpr, a))
            if probs.size:
                y_pred = (probs >= prob_thr).astype(int)
                yt = labels.astype(int)
                cm = confusion_matrix(yt, y_pred, labels=[0, 1])
                cm2.append((f"Single / {label}", cm))
                summary[key].update(_binary_metrics_from_cm(cm))

        if roc2:
            plot_roc_curves(
                roc2,
                "ROC: single-curve thoracic vs lumbar",
                os.path.join(out_dir, "subgroup_chart2_single_thoracic_lumbar_roc.png"),
            )
        if cm2:
            plot_confusion_pair(
                cm2,
                os.path.join(out_dir, "subgroup_chart2_single_thoracic_lumbar_confusion.png"),
                f"Confusion @ prob ≥ {prob_thr:.3f} (single only)",
            )

        chart3_defs_sf = (
            ("General (Cobb>10)", p_gen, y_gen, m_gen),
            ("Single thoracic", p_th, y_th, m_th),
            ("Single lumbar", p_lb, y_lb, m_lb),
            ("Multi-curve", p_m, y_m, m_m),
        )
        roc3: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
        cm3_sf: List[Tuple[str, np.ndarray]] = []
        for label, probs, labels, mets in chart3_defs_sf:
            key_slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("°", "")
            chart3_key = f"chart3_{key_slug}"
            summary[chart3_key] = {"n_predictions": int(probs.size)}
            if mets.get("has_labels"):
                summary[chart3_key]["auc_roc"] = mets.get("auc_roc")
            roc = _roc_safe(labels, probs) if probs.size else None
            if roc is not None:
                fpr, tpr, a = roc
                roc3.append((label, fpr, tpr, a))
            if probs.size:
                y_pred = (probs >= prob_thr).astype(int)
                yt = labels.astype(int)
                cm = confusion_matrix(yt, y_pred, labels=[0, 1])
                cm3_sf.append((label, cm))
                summary[chart3_key].update(_binary_metrics_from_cm(cm))

        if roc3:
            plot_roc_curves(
                roc3,
                "ROC: DK JSON strata",
                os.path.join(out_dir, "subgroup_chart3_four_strata_roc.png"),
            )
        if cm3_sf:
            plot_confusion_grid(
                cm3_sf,
                os.path.join(out_dir, "subgroup_chart3_four_strata_confusion.png"),
                f"Confusion @ prob ≥ {prob_thr:.3f} (four strata)",
            )

    out_json = os.path.join(out_dir, "subgroup_analysis_summary.json")

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

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(summary), f, indent=2)

    print(json.dumps(_json_sanitize(summary), indent=2))
    print(f"\nSaved figures and {out_json}")


if __name__ == "__main__":
    main()
