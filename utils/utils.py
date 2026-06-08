"""
Shared helpers for k-fold style scripts (subject-level stratified splits).

Used by training/eval entrypoints that build a ``ConcatDataset`` or single
PKL dataset and need subject IDs + binary labels for ``StratifiedKFold``.

Also includes knowledge-map attention / attention×gradient visualization
helpers used by ``run_test.py`` and similar eval scripts.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset

from utils.plot_style_nature import mm_to_inch, style_axis_nature


def _get_subject_id_for_index(full_dataset, global_idx: int) -> str:
    """Resolve global index -> subject_id string for Dataset or ConcatDataset."""
    if isinstance(full_dataset, ConcatDataset):
        offset = 0
        for ds in full_dataset.datasets:
            if global_idx < offset + len(ds):
                local_idx = global_idx - offset
                entry = ds.patch_metadata[local_idx]
                sid = entry.get("subject_id")
                return str(int(float(sid))) if sid is not None else f"unk_{global_idx}"
            offset += len(ds)
        raise IndexError(f"Index {global_idx} out of range")
    entry = full_dataset.patch_metadata[global_idx]
    sid = entry.get("subject_id")
    return str(int(float(sid))) if sid is not None else f"unk_{global_idx}"


def _get_label_for_index(full_dataset, global_idx: int, binary_threshold: float) -> int:
    """Resolve global index -> binary label (0 or 1) for stratification."""
    if isinstance(full_dataset, ConcatDataset):
        offset = 0
        for ds in full_dataset.datasets:
            if global_idx < offset + len(ds):
                local_idx = global_idx - offset
                entry = ds.patch_metadata[local_idx]
                break
            offset += len(ds)
        else:
            raise IndexError(f"Index {global_idx} out of range")
    else:
        entry = full_dataset.patch_metadata[global_idx]

    # Keep extraction consistent with utils.sft_utils.count_dataset_pos_neg:
    # prefer label_value, fall back to label only when label_value is missing.
    label_value = entry.get("label_value")
    if label_value is None:
        label_value = entry.get("label")
    if label_value is None:
        return 0
    if isinstance(label_value, list):
        return 1 if max(label_value) >= binary_threshold else 0
    return 1 if float(label_value) >= binary_threshold else 0


def build_subject_map(
    full_dataset,
    binary_threshold: float,
) -> Tuple[List[str], np.ndarray, Dict[str, List[int]]]:
    """Build subject-level structures for stratified k-fold.

    Returns:
        subject_ids: sorted list of unique subject IDs
        subject_labels: binary label per subject (for stratification)
        subject_to_indices: mapping subject_id -> list of global dataset indices
    """
    subject_to_indices: Dict[str, List[int]] = defaultdict(list)
    subject_label_votes: Dict[str, List[int]] = defaultdict(list)

    total = len(full_dataset)
    for idx in range(total):
        sid = _get_subject_id_for_index(full_dataset, idx)
        lbl = _get_label_for_index(full_dataset, idx, binary_threshold)
        subject_to_indices[sid].append(idx)
        subject_label_votes[sid].append(lbl)

    subject_ids = sorted(subject_to_indices.keys(), key=lambda x: int(x) if x.isdigit() else x)
    subject_labels = np.array(
        [int(np.round(np.mean(subject_label_votes[sid]))) for sid in subject_ids]
    )
    return subject_ids, subject_labels, dict(subject_to_indices)


# ---------------------------------------------------------------------------
# Knowledge-map attention & attention × gradient explanation (test / viz)
# ---------------------------------------------------------------------------


def get_km_attention_map(raw_model, knowledge_map, use_last_layer=True, layer_index=-1):
    """
    Get KM attention map with same shape as input knowledge_map (B, T, F).
    Uses block-level attention from the last forward pass (run model forward first).
    Delegates to encoder.get_attention_map_2d(T, F, ...) for ViT and PatchViT encoders (same API).
    Returns None if encoder does not support it or no attention available.
    """
    _, T_km, F = knowledge_map.shape
    km_encoder = getattr(raw_model, "km_encoder", None)
    if km_encoder is None:
        return None
    if not hasattr(km_encoder, "get_attention_map_2d"):
        return None
    out = km_encoder.get_attention_map_2d(
        T_km,
        F,
        use_last_layer=use_last_layer,
        layer_index=layer_index,
    )
    if out is not None:
        return out.detach().cpu().float().numpy()
    return None


def plot_attention_maps(attention_maps, knowledge_map, save_dir, max_samples=8, prefix="km_attn"):
    """
    Plot attention heatmaps (same shape as KM) and save to save_dir.
    attention_maps: (B, T, F) numpy
    knowledge_map: (B, T, F) tensor or numpy (for optional side-by-side KM plot)
    """
    os.makedirs(save_dir, exist_ok=True)
    B, T, F = attention_maps.shape
    n_plot = min(B, max_samples)
    for i in range(n_plot):
        attn = attention_maps[i]  # (T, F)
        km = knowledge_map[i].detach().cpu().numpy() if torch.is_tensor(knowledge_map) else knowledge_map[i]
        fig, axes = plt.subplots(1, 2, figsize=(mm_to_inch(180), mm_to_inch(52)), facecolor="white")
        axes[0].imshow(attn, aspect="auto", cmap="viridis")
        axes[0].set_title("Attention (KM grid)")
        axes[0].set_xlabel("Feature index")
        axes[0].set_ylabel("Time index")
        style_axis_nature(axes[0])
        axes[1].imshow(km, aspect="auto", cmap="RdBu_r")
        axes[1].set_title("Knowledge map")
        axes[1].set_xlabel("Feature index")
        axes[1].set_ylabel("Time index")
        style_axis_nature(axes[1])
        plt.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_sample_{i}.png")
        plt.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()
    print(f"Saved {n_plot} attention maps to {save_dir} (prefix={prefix})")


def _extract_temporal_attention(raw_model, T_km, use_last_layer=True, layer_index=-1):
    """
    Extract temporal attention A_t from KnowledgeViT-style block attention cache.
    Returns:
        A_t: (B, T_km) torch.Tensor on current device, or None.
    """
    km_encoder = getattr(raw_model, "km_encoder", None)
    if km_encoder is None or not hasattr(km_encoder, "get_block_attention"):
        return None
    block_attns = km_encoder.get_block_attention()
    valid = [a for a in block_attns if a is not None]
    if not valid:
        return None
    attn = valid[layer_index] if use_last_layer else torch.stack(valid, dim=0).mean(dim=0)  # (B, H, N, N)
    N = attn.shape[-1]
    if N == T_km + 1:
        # CLS + temporal tokens
        a_t = attn[:, :, 0, 1:].mean(dim=1)  # (B, T)
    elif N == T_km:
        # Temporal-only tokens
        a_t = attn.mean(dim=2).mean(dim=1)  # (B, T)
    else:
        return None
    # Normalize each sample to sum=1 for stable fusion
    a_t = a_t / (a_t.sum(dim=1, keepdim=True) + 1e-8)
    return a_t


def get_km_attention_x_grad_map(
    raw_model,
    video,
    knowledge_map,
    texts=None,
    km_indices=None,
    video_indices=None,
    use_last_layer=True,
    layer_index=-1,
):
    """
    Fallback explanation map for KM branch:
      heatmap(T,F) = temporal_attention(T) * |d logit / d knowledge_map(T,F)|
    Returns:
      heatmap_np: (B, T, F) numpy in [0,1]
      topk_info: list[dict] with top-k factor indices per sample
      logits_np: (B,) numpy logits
    """
    raw_model.eval()
    # Clone input so we can compute input gradients without side effects
    km = knowledge_map.detach().clone().requires_grad_(True)

    # Forward with grad enabled
    with torch.enable_grad():
        preds = raw_model(
            video,
            km,
            texts=texts,
            km_indices=km_indices,
            video_indices=video_indices,
        )
        if preds.ndim == 2 and preds.shape[1] == 1:
            logits = preds[:, 0]
        elif preds.ndim == 2 and preds.shape[1] > 1:
            # Positive class logit for binary classification with 2 outputs
            logits = preds[:, 1]
        else:
            raise ValueError(f"Unexpected prediction shape: {tuple(preds.shape)}")

        raw_model.zero_grad(set_to_none=True)
        if km.grad is not None:
            km.grad.zero_()
        logits.sum().backward()

        grad_tf = km.grad.detach().abs()  # (B, T, F)
        _, T_km, _ = km.shape
        a_t = _extract_temporal_attention(
            raw_model,
            T_km=T_km,
            use_last_layer=use_last_layer,
            layer_index=layer_index,
        )
        if a_t is None:
            return None, None, None
        heatmap = a_t.unsqueeze(-1) * grad_tf  # (B, T, F)

    # Normalize each sample to [0,1] for plotting
    B = heatmap.shape[0]
    flat = heatmap.view(B, -1)
    h_min = flat.min(dim=1, keepdim=True).values
    h_max = flat.max(dim=1, keepdim=True).values
    heatmap_norm = ((flat - h_min) / (h_max - h_min + 1e-8)).view_as(heatmap)

    # Top-k factors per sample by aggregating over time
    factor_scores = heatmap_norm.mean(dim=1)  # (B, F)
    k = min(10, factor_scores.shape[1])
    top_vals, top_idx = torch.topk(factor_scores, k=k, dim=1)
    topk_info = []
    for i in range(B):
        topk_info.append(
            {
                "factor_indices": top_idx[i].detach().cpu().tolist(),
                "scores": [float(v) for v in top_vals[i].detach().cpu().tolist()],
            }
        )

    return (
        heatmap_norm.detach().cpu().float().numpy(),
        topk_info,
        logits.detach().cpu().float().numpy(),
    )


def plot_attention_x_grad_maps(
    heatmap_maps,
    knowledge_map,
    save_dir,
    topk_info=None,
    logits=None,
    max_samples=8,
    prefix="km_attn_xgrad",
):
    """
    Plot attention x gradient heatmaps and save top-k factor summaries.
    """
    os.makedirs(save_dir, exist_ok=True)
    B, _, _ = heatmap_maps.shape
    n_plot = min(B, max_samples)
    for i in range(n_plot):
        hmap = heatmap_maps[i]  # (T, F)
        km = knowledge_map[i].detach().cpu().numpy() if torch.is_tensor(knowledge_map) else knowledge_map[i]
        fig, axes = plt.subplots(1, 2, figsize=(mm_to_inch(180), mm_to_inch(52)), facecolor="white")
        axes[0].imshow(hmap, aspect="auto", cmap="magma")
        axes[0].set_title("Attention × input-gradient")
        axes[0].set_xlabel("Feature index")
        axes[0].set_ylabel("Time index")
        style_axis_nature(axes[0])
        axes[1].imshow(km, aspect="auto", cmap="RdBu_r")
        axes[1].set_title("Knowledge map")
        axes[1].set_xlabel("Feature index")
        axes[1].set_ylabel("Time index")
        style_axis_nature(axes[1])
        plt.tight_layout()
        path = os.path.join(save_dir, f"{prefix}_sample_{i}.png")
        plt.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()

    # Save textual top-k factor report
    if topk_info is not None:
        topk_path = os.path.join(save_dir, f"{prefix}_topk_factors.txt")
        with open(topk_path, "w") as f:
            for i in range(n_plot):
                logit_str = ""
                if logits is not None and i < len(logits):
                    logit_str = f" | logit={float(logits[i]):.6f}"
                f.write(f"sample_{i}{logit_str}\n")
                idxs = topk_info[i]["factor_indices"]
                vals = topk_info[i]["scores"]
                for rank, (idx, val) in enumerate(zip(idxs, vals), start=1):
                    f.write(f"  top{rank:02d}: factor={idx}, score={val:.6f}\n")
                f.write("\n")

    print(f"Saved {n_plot} attention x grad maps to {save_dir} (prefix={prefix})")


# ---------------------------------------------------------------------------
# Generic evaluation + output helpers (shared by eval scripts)
# ---------------------------------------------------------------------------


def safe_filename_stem(name: str, max_len: int = 120) -> str:
    """File stem safe on Windows; keep alphanumerics, dot, underscore, hyphen."""
    s = re.sub(r"[^\w.\-]+", "_", str(name), flags=re.UNICODE)
    s = s.strip("._-") or "run"
    return s[:max_len]


def model_name_from_run_dir(run_dir: str) -> str:
    """
    Extract a stable model name from a k-fold run folder name.

    Example:
      kfold5_kvt_no_gated_token_pooling_20260404_114906 -> kfold5_kvt_no_gated_token_pooling
    """
    base = os.path.basename(str(run_dir).rstrip("/\\"))
    base = safe_filename_stem(base)
    # Strip common trailing timestamp pattern: _YYYYMMDD_HHMMSS
    base = re.sub(r"_\d{8}_\d{6}$", "", base)
    return base or "run"


def apply_nature_style_mpl() -> None:
    """Matplotlib defaults aligned with ``plots/rules.text`` (Nature-style)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica Neue",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_pretrain_loss_curves(
    history: Sequence[Dict[str, Any]],
    out_path: str,
    title: Optional[str] = None,
) -> str:
    """
    Plot pretrain train/val total loss and trimodal val components vs epoch.

    Args:
        history: rows from loss_history.json (epoch, train_loss, val_loss, ...).
        out_path: PNG path (parent dirs created).
        title: optional figure suptitle.

    Returns:
        Absolute path to saved PNG.
    """
    if not history:
        raise ValueError("history is empty; nothing to plot")

    def _f(row: Dict[str, Any], key: str) -> Optional[float]:
        v = row.get(key)
        if v is None or v == "":
            return None
        return float(v)

    epochs = [_f(r, "epoch") for r in history]
    train_loss = [_f(r, "train_loss") for r in history]
    val_loss = [_f(r, "val_loss") for r in history]
    has_components = any(
        _f(r, "val_loss_km_text") is not None for r in history
    )

    apply_nature_style_mpl()
    if has_components:
        fig, axes = plt.subplots(1, 2, figsize=(mm_to_inch(180), mm_to_inch(70)))
        ax_total, ax_comp = axes
    else:
        fig, ax_total = plt.subplots(figsize=(mm_to_inch(89), mm_to_inch(70)))
        ax_comp = None

    ax_total.plot(epochs, train_loss, label="Train", color="#2166ac", linewidth=1.2)
    ax_total.plot(epochs, val_loss, label="Val", color="#b2182b", linewidth=1.2)
    ax_total.set_xlabel("Epoch")
    ax_total.set_ylabel("Trimodal contrastive loss")
    ax_total.legend(frameon=False, loc="best")
    style_axis_nature(ax_total)

    if ax_comp is not None:
        km = [_f(r, "val_loss_km_text") for r in history]
        vt = [_f(r, "val_loss_video_text") for r in history]
        vk = [_f(r, "val_loss_video_km") for r in history]
        ax_comp.plot(epochs, km, label="KM–text", linewidth=1.0)
        ax_comp.plot(epochs, vt, label="Video–text", linewidth=1.0)
        ax_comp.plot(epochs, vk, label="Video–KM", linewidth=1.0)
        ax_comp.set_xlabel("Epoch")
        ax_comp.set_ylabel("Val pair loss")
        ax_comp.legend(frameon=False, loc="best", fontsize=6)
        style_axis_nature(ax_comp)

    if title:
        fig.suptitle(title, fontsize=8, y=1.02)
    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


def plot_pretrain_loss_compare(
    runs: Sequence[Dict[str, Any]],
    out_path: str,
    metric: str = "val_loss",
    title: Optional[str] = None,
) -> str:
    """
    Overlay one metric from multiple pretrain runs (e.g. A/B/C alignment ladder).

    Args:
        runs: list of {"label": str, "history": list[dict]} or {"label", "csv"/"json"}.
        out_path: PNG path.
        metric: column to plot (train_loss, val_loss, val_loss_km_text, ...).
        title: optional figure title.

    Returns:
        Absolute path to saved PNG.
    """
    from utils.pre_utils import load_pretrain_metrics

    apply_nature_style_mpl()
    fig, ax = plt.subplots(figsize=(mm_to_inch(89), mm_to_inch(70)))
    colors = ["#2166ac", "#4daf4a", "#b2182b", "#984ea3", "#ff7f00"]

    for idx, run in enumerate(runs):
        label = run.get("label", f"run_{idx}")
        if "history" in run:
            history = run["history"]
        else:
            path = run.get("csv") or run.get("json")
            if not path:
                raise ValueError(f"run {label}: provide history or csv/json path")
            history = load_pretrain_metrics(path)
        epochs = [float(r["epoch"]) for r in history]
        ys = [float(r[metric]) for r in history if r.get(metric) not in (None, "")]
        if len(ys) != len(epochs):
            epochs = epochs[: len(ys)]
        ax.plot(
            epochs,
            ys,
            label=label,
            color=colors[idx % len(colors)],
            linewidth=1.2,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " "))
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    style_axis_nature(ax)
    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


def save_rows_csv(rows: Sequence[Dict[str, Any]], csv_path: str) -> str:
    """Save list[dict] to CSV, creating parent directory."""
    import pandas as pd

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def compute_binary_classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute accuracy, AUC, sensitivity, specificity, PPV, NPV, confusion counts."""
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(np.float32)
    preds = (probs >= float(threshold)).astype(np.float32)

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
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
    except Exception:
        auc = 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "auc_roc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def build_sft_regressor_from_config(cfg: Dict[str, Any]):
    """
    Build SFTRegressor from a config dict.

    Import is local to avoid hard dependency when utils are used elsewhere.
    """
    from models.sft_regressor import SFTRegressor

    return SFTRegressor(
        km_feature_dim=cfg.get("km_feature_dim", 238),
        hidden_dim=cfg.get("hidden_dim", 256),
        label_dim=cfg.get("label_dim", 1),
        video_encoder_type=cfg.get("video_encoder_type", "vivit"),
        video_encoder_kwargs=cfg.get("video_encoder_kwargs", {}),
        km_encoder_type=cfg.get("km_encoder_type", "vit"),
        km_encoder_kwargs=cfg.get("km_encoder_kwargs", {}),
        text_model_name=cfg.get(
            "text_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        ),
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


def eval_sft_checkpoint_on_dataloader(
    ckpt_path: str,
    dataloader,
    cfg: Dict[str, Any],
    device,
    prob_threshold: float = 0.5,
):
    """
    Evaluate one SFT checkpoint on one dataloader and return (metrics, probs, labels).

    Notes:
    - Uses `utils.sft_utils.resolve_device/setup_multi_gpu` if needed.
    - Expects batch dict keys: video, knowledge_map, optional texts/km_indices/video_indices, and label.
    """
    import torch
    import numpy as np

    from utils.sft_utils import resolve_device, setup_multi_gpu

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = checkpoint.get("config", cfg)

    gpu_ids = ckpt_cfg.get("gpu_ids")
    device = resolve_device(gpu_ids) if device is None else device

    model = build_sft_regressor_from_config(ckpt_cfg)
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
                continue

            try:
                labels_np = labels.detach().cpu().numpy().reshape(-1)
                if labels_np.size == 0:
                    continue
            except Exception:
                continue

            if probs.shape[0] != labels_np.shape[0]:
                continue

            all_probs.append(probs)
            all_labels.append(labels_np)

    if not all_probs or not all_labels:
        return {"has_labels": False}, np.array([]), np.array([])

    probs_concat = np.concatenate(all_probs, axis=0).reshape(-1)
    labels_concat = np.concatenate(all_labels, axis=0).reshape(-1)

    metrics = compute_binary_classification_metrics(
        probs_concat,
        labels_concat,
        threshold=prob_threshold,
    )
    metrics["has_labels"] = True
    metrics["checkpoint_path"] = ckpt_path
    metrics["epoch"] = checkpoint.get("epoch", None)
    return metrics, probs_concat, labels_concat

__all__ = [
    "build_subject_map",
    "get_km_attention_map",
    "plot_attention_maps",
    "get_km_attention_x_grad_map",
    "plot_attention_x_grad_maps",
    "safe_filename_stem",
    "model_name_from_run_dir",
    "apply_nature_style_mpl",
    "save_rows_csv",
    "compute_binary_classification_metrics",
    "build_sft_regressor_from_config",
    "eval_sft_checkpoint_on_dataloader",
]
