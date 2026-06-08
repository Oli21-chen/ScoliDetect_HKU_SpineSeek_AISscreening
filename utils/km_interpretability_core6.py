"""
Six-panel KM interpretability dashboard (motion / skeleton / signal domains).

Domain index ranges match ``utils/knowledge_map.py`` construction:
  motion:  [0, 34), skeleton: [34, 172), signal: [172, 238).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils.plot_style_nature import (
    apply_nature_journal_mpl_style,
    mm_to_inch,
    style_axis_nature,
)
from utils.utils import get_km_attention_map, get_km_attention_x_grad_map

DOMAIN_SLICES = {
    "motion": (0, 34),
    "skeleton": (34, 172),
    "signal": (172, 238),
}
DOMAIN_COLORS = {
    "motion": "#0072B2",
    "skeleton": "#009E73",
    "signal": "#E69F00",
}
F_TOTAL = 238


def domain_name_for_feature(fi: int) -> str:
    for name, (a, b) in DOMAIN_SLICES.items():
        if a <= fi < b:
            return name
    return "out_of_range"


def _trim_batch(batch: Dict[str, Any], n: int, device: torch.device) -> Dict[str, Any]:
    out = {}
    video = batch["video"][:n].to(device)
    km = batch["knowledge_map"][:n].to(device)
    out["video"] = video
    out["knowledge_map"] = km
    texts = batch.get("texts", None)
    if isinstance(texts, list):
        out["texts"] = texts[:n]
    else:
        out["texts"] = texts
    km_ix = batch.get("km_indices", None)
    out["km_indices"] = km_ix[:n].to(device) if km_ix is not None else None
    vid_ix = batch.get("video_indices", None)
    out["video_indices"] = vid_ix[:n].to(device) if vid_ix is not None else None
    if "label" in batch and batch["label"] is not None:
        out["label"] = batch["label"][:n].to(device)
    return out


def _logits_from_preds(preds: torch.Tensor) -> torch.Tensor:
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds[:, 0]
    if preds.ndim == 2 and preds.shape[1] > 1:
        return preds[:, 1]
    return preds.reshape(-1)


def _forward_logits_np(
    raw_model: torch.nn.Module,
    video: torch.Tensor,
    km: torch.Tensor,
    texts: Any,
    km_indices: Optional[torch.Tensor],
    video_indices: Optional[torch.Tensor],
) -> np.ndarray:
    with torch.no_grad():
        preds = raw_model(
            video,
            km,
            texts,
            km_indices=km_indices,
            video_indices=video_indices,
        )
    return _logits_from_preds(preds).detach().cpu().numpy()


def _safe_auc(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    y = np.asarray(y).astype(int).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def _bootstrap_factor_se(axg_tf: np.ndarray, n_boot: int = 200, seed: int = 0) -> np.ndarray:
    """axg_tf: (B, T, F) -> SE vector length F (bootstrap over subjects of mean over time)."""
    rng = np.random.default_rng(seed)
    B, _T, F = axg_tf.shape
    if B < 2:
        return np.zeros(F, dtype=np.float64)
    means: List[np.ndarray] = []
    for _ in range(n_boot):
        idx = rng.integers(0, B, size=B)
        m = axg_tf[idx].mean(axis=(0, 1))
        means.append(m)
    stack = np.stack(means, axis=0)
    return stack.std(axis=0, ddof=1)


def _bootstrap_se_from_rows(x: np.ndarray, n_boot: int = 500, seed: int = 0) -> np.ndarray:
    """x: (N, D) -> bootstrap SE of mean per dimension."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    n, d = x.shape
    if n < 2:
        return np.zeros(d, dtype=np.float64)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(x[idx].mean(axis=0))
    stack = np.stack(means, axis=0)
    return stack.std(axis=0, ddof=1)


@dataclass
class AggregatedKmInterpretability:
    """Cohort-level KM interpretability tensors from a full dataloader pass."""

    temporal_mat: np.ndarray  # (N, T)
    factor_mat: np.ndarray  # (N, F)
    frac_stack: np.ndarray  # (3, T)
    time_pct: np.ndarray
    n_total: int
    label_counts: Dict[str, int]
    km_shape: List[int]


def _collect_axg_maps_chunked(
    *,
    raw_model: torch.nn.Module,
    video: torch.Tensor,
    knowledge_map: torch.Tensor,
    texts: Any,
    km_indices: Optional[torch.Tensor],
    video_indices: Optional[torch.Tensor],
    axg_chunk_size: int,
    device: torch.device,
) -> Optional[np.ndarray]:
    """
    Attention×gradient maps with bounded GPU memory.

    ``get_km_attention_x_grad_map`` runs a backward pass; large batch sizes OOM on ViT+video.
    Process ``axg_chunk_size`` samples at a time (default 1, matching ``run_test.py``).
    """
    chunk_size = max(1, int(axg_chunk_size))
    B = int(video.shape[0])
    chunk_maps: List[np.ndarray] = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        texts_chunk = texts
        if isinstance(texts, list):
            texts_chunk = texts[start:end]
        km_ix = km_indices[start:end] if km_indices is not None else None
        vid_ix = video_indices[start:end] if video_indices is not None else None
        axg_map, _topk, _logits_np = get_km_attention_x_grad_map(
            raw_model=raw_model,
            video=video[start:end],
            knowledge_map=knowledge_map[start:end],
            texts=texts_chunk,
            km_indices=km_ix,
            video_indices=vid_ix,
            use_last_layer=True,
            layer_index=-1,
        )
        if axg_map is not None:
            chunk_maps.append(np.asarray(axg_map, dtype=np.float64))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not chunk_maps:
        return None
    return np.concatenate(chunk_maps, axis=0)


def _aggregate_km_interpretability_from_dataloader(
    *,
    raw_model: torch.nn.Module,
    dataloader,
    device: torch.device,
    axg_chunk_size: int = 1,
    attn_chunk_size: Optional[int] = None,
) -> Optional[AggregatedKmInterpretability]:
    """Single pass: temporal attention, factor attribution, domain mass fractions."""
    raw_model.eval()
    temporal_rows: List[np.ndarray] = []
    factor_rows: List[np.ndarray] = []
    dom_mass_sum: Dict[str, Optional[np.ndarray]] = {k: None for k in DOMAIN_SLICES}
    total_mass_sum: Optional[np.ndarray] = None
    n_total = 0
    km_shape: Optional[List[int]] = None
    label_counts: Dict[str, int] = {}
    attn_chunk = max(1, int(attn_chunk_size)) if attn_chunk_size is not None else max(1, int(axg_chunk_size))

    for batch in dataloader:
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
        labels = batch["label"].detach().cpu().numpy().astype(int).ravel()
        for v in labels:
            label_counts[str(int(v))] = label_counts.get(str(int(v)), 0) + 1

        B = int(video.shape[0])
        attn_chunks: List[np.ndarray] = []
        for start in range(0, B, attn_chunk):
            end = min(start + attn_chunk, B)
            texts_chunk = texts
            if isinstance(texts, list):
                texts_chunk = texts[start:end]
            km_ix = km_indices[start:end] if km_indices is not None else None
            vid_ix = video_indices[start:end] if video_indices is not None else None
            with torch.no_grad():
                _ = raw_model(
                    video[start:end],
                    knowledge_map[start:end],
                    texts=texts_chunk,
                    km_indices=km_ix,
                    video_indices=vid_ix,
                )
            attn_map = get_km_attention_map(
                raw_model, knowledge_map[start:end], use_last_layer=True, layer_index=-1
            )
            if attn_map is not None:
                temporal = np.asarray(attn_map, dtype=np.float64).mean(axis=2)  # (B,T)
                attn_chunks.append(temporal)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if attn_chunks:
            temporal_rows.append(np.concatenate(attn_chunks, axis=0))

        axg_map = _collect_axg_maps_chunked(
            raw_model=raw_model,
            video=video,
            knowledge_map=knowledge_map,
            texts=texts,
            km_indices=km_indices,
            video_indices=video_indices,
            axg_chunk_size=axg_chunk_size,
            device=device,
        )
        if axg_map is None:
            continue

        B, T, Fdim = axg_map.shape
        if Fdim != F_TOTAL:
            raise ValueError(f"Expected F={F_TOTAL}, got {Fdim}")
        if km_shape is None:
            km_shape = [int(T), int(Fdim)]

        factor_scores = axg_map.mean(axis=1)  # (B,F)
        factor_rows.append(np.asarray(factor_scores, dtype=np.float64))

        if total_mass_sum is None:
            total_mass_sum = np.zeros(T, dtype=np.float64)
        total_mass_sum += np.asarray(axg_map, dtype=np.float64).sum(axis=2).sum(axis=0)
        for dname, (lo, hi) in DOMAIN_SLICES.items():
            seg = np.asarray(axg_map[:, :, lo:hi], dtype=np.float64).sum(axis=2).sum(axis=0)
            if dom_mass_sum[dname] is None:
                dom_mass_sum[dname] = np.zeros(T, dtype=np.float64)
            dom_mass_sum[dname] = dom_mass_sum[dname] + seg

        n_total += int(B)

    if n_total == 0 or km_shape is None:
        return None

    temporal_mat = np.concatenate(temporal_rows, axis=0) if temporal_rows else None
    factor_mat = np.concatenate(factor_rows, axis=0) if factor_rows else None
    if temporal_mat is None or factor_mat is None:
        return None

    T = temporal_mat.shape[1]
    time_pct = np.linspace(0.0, 100.0, T)
    assert total_mass_sum is not None
    frac_curves = []
    for dname in DOMAIN_SLICES.keys():
        dm = dom_mass_sum[dname]
        if dm is None:
            dm = np.zeros_like(total_mass_sum)
        frac_curves.append(dm / (total_mass_sum + 1e-8))
    frac_stack = np.stack(frac_curves, axis=0)

    return AggregatedKmInterpretability(
        temporal_mat=temporal_mat,
        factor_mat=factor_mat,
        frac_stack=frac_stack,
        time_pct=time_pct,
        n_total=int(n_total),
        label_counts=label_counts,
        km_shape=km_shape,
    )


def _derived_factor_stats(
    agg: AggregatedKmInterpretability,
    *,
    top_k: int = 20,
    bootstrap_rounds: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    mean_factor = agg.factor_mat.mean(axis=0)
    se_factor = _bootstrap_se_from_rows(agg.factor_mat, n_boot=bootstrap_rounds)
    order = np.argsort(-mean_factor)
    k = max(1, min(int(top_k), int(len(order))))
    top_idx = order[:k]
    return mean_factor, se_factor, order, k, top_idx


def _fig5_subgroup_panel_title(stratum_key: str, stratum_title: str, n: int) -> str:
    if stratum_key == "general_cobb_gt10":
        return f"General (Cobb > 10°, n = {n})"
    return f"{stratum_title} (n = {n})"


def save_km_interpretability_core3_from_agg(
    agg: AggregatedKmInterpretability,
    *,
    stratum_key: str,
    stratum_title: str,
    out_dir: str,
    top_k: int = 20,
    bootstrap_rounds: int = 500,
) -> Optional[Dict[str, Any]]:
    """Save per-stratum 1×3 core3 figure and top-factors CSV from pre-aggregated tensors."""
    mean_temporal = agg.temporal_mat.mean(axis=0)
    std_temporal = (
        agg.temporal_mat.std(axis=0, ddof=1)
        if agg.temporal_mat.shape[0] > 1
        else np.zeros_like(mean_temporal)
    )
    mean_factor, se_factor, order, k, top_idx = _derived_factor_stats(
        agg, top_k=top_k, bootstrap_rounds=bootstrap_rounds
    )

    apply_nature_journal_mpl_style()
    fig_w = mm_to_inch(300.0)
    fig_h = mm_to_inch(72.0)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(fig_w, fig_h),
        facecolor="white",
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]},
    )
    ax_a, ax_b, ax_d = axes[0], axes[1], axes[2]

    ax_a.plot(agg.time_pct, mean_temporal, color="#222222", linewidth=1.0)
    ax_a.fill_between(
        agg.time_pct,
        mean_temporal - std_temporal,
        mean_temporal + std_temporal,
        alpha=0.18,
        color="#222222",
    )
    ax_a.set_xlabel("Time (% sequence)")
    ax_a.set_ylabel("Mean attention (over KM features)")
    ax_a.set_title("Temporal attention")
    ax_a.grid(True, alpha=0.22, linewidth=0.4, linestyle="-")
    style_axis_nature(ax_a)

    y = np.arange(k)
    ax_b.barh(
        y,
        mean_factor[top_idx][::-1],
        xerr=se_factor[top_idx][::-1],
        color=[DOMAIN_COLORS[domain_name_for_feature(int(i))] for i in top_idx][::-1],
        ecolor="#666666",
        capsize=1.5,
        linewidth=0,
        edgecolor="none",
        height=0.82,
    )
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([str(int(i)) for i in top_idx[::-1]])
    ax_b.set_xlabel("Attribution (mean ± bootstrap SE)")
    ax_b.set_title(
        _fig5_subgroup_panel_title(stratum_key, stratum_title, agg.n_total),
        fontsize=8,
    )
    ax_b.grid(True, axis="x", alpha=0.22, linewidth=0.4, linestyle="-")
    style_axis_nature(ax_b)

    ax_d.stackplot(
        agg.time_pct,
        agg.frac_stack[0],
        agg.frac_stack[1],
        agg.frac_stack[2],
        labels=list(DOMAIN_SLICES.keys()),
        colors=[DOMAIN_COLORS[dk] for dk in DOMAIN_SLICES],
        alpha=0.88,
        linewidth=0.0,
    )
    ax_d.set_xlim(agg.time_pct[0], agg.time_pct[-1])
    ax_d.set_ylim(0.0, 1.0)
    ax_d.set_xlabel("Time (% sequence)")
    ax_d.set_ylabel("Attribution mass fraction")
    ax_d.set_title("Domain dynamics")
    ax_d.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax_d.grid(True, alpha=0.22, linewidth=0.4, linestyle="-")
    style_axis_nature(ax_d)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"km_interpretability_core3_{stratum_key}.png")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    csv_path = os.path.join(out_dir, f"km_interpretability_top_factors_{stratum_key}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["rank", "factor_index", "domain", "mean_attribution", "bootstrap_se"])
        for r, fi in enumerate(order[:50], start=1):
            fi = int(fi)
            w.writerow(
                [
                    r,
                    fi,
                    domain_name_for_feature(fi),
                    f"{float(mean_factor[fi]):.8f}",
                    f"{float(se_factor[fi]):.8f}",
                ]
            )

    stats: Dict[str, Any] = {
        "stratum_key": stratum_key,
        "stratum_title": stratum_title,
        "n_samples": int(agg.n_total),
        "km_shape": agg.km_shape,
        "label_counts": agg.label_counts,
        "top_k": int(k),
        "figure_path": png_path,
        "csv_path": csv_path,
    }
    json_path = os.path.join(out_dir, f"km_interpretability_core3_{stratum_key}_stats.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(stats, jf, indent=2)
    stats["stats_json_path"] = json_path
    print(f"Saved subgroup core3 interpretability: {png_path}")
    return stats


def save_fig5_dk_composite_interpretability(
    *,
    general_agg: AggregatedKmInterpretability,
    stratum_aggs: Sequence[Tuple[str, str, AggregatedKmInterpretability]],
    out_dir: str,
    top_k: int = 20,
    bootstrap_rounds: int = 500,
    ci_z: float = 1.96,
) -> str:
    """
    Fig. 5 composite (2×3): temporal attention + domain dynamics (general stratum),
    then top-K factors for each clinical stratum (panels c–f).
    """
    apply_nature_journal_mpl_style()
    fig_w = mm_to_inch(180.0)
    fig_h = mm_to_inch(120.0)
    fig, axes = plt.subplots(2, 3, figsize=(fig_w, fig_h), facecolor="white")
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_cf = axes[1, :]

    mean_temporal = general_agg.temporal_mat.mean(axis=0)
    se_temporal = _bootstrap_se_from_rows(general_agg.temporal_mat, n_boot=bootstrap_rounds)
    ci_half = ci_z * se_temporal
    ax_a.plot(general_agg.time_pct, mean_temporal, color="#222222", linewidth=1.0)
    ax_a.fill_between(
        general_agg.time_pct,
        mean_temporal - ci_half,
        mean_temporal + ci_half,
        alpha=0.18,
        color="#222222",
    )
    ax_a.set_xlabel("Time (% sequence)")
    ax_a.set_ylabel("Mean KKM temporal attention")
    ax_a.set_title("Temporal attention")
    ax_a.grid(True, alpha=0.22, linewidth=0.4, linestyle="-")
    style_axis_nature(ax_a)

    ax_b.stackplot(
        general_agg.time_pct,
        general_agg.frac_stack[0],
        general_agg.frac_stack[1],
        general_agg.frac_stack[2],
        labels=list(DOMAIN_SLICES.keys()),
        colors=[DOMAIN_COLORS[dk] for dk in DOMAIN_SLICES],
        alpha=0.88,
        linewidth=0.0,
    )
    ax_b.set_xlim(general_agg.time_pct[0], general_agg.time_pct[-1])
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_xlabel("Time (% sequence)")
    ax_b.set_ylabel("Relative attribution mass")
    ax_b.set_title("Domain dynamics")
    ax_b.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6)
    ax_b.grid(True, alpha=0.22, linewidth=0.4, linestyle="-")
    style_axis_nature(ax_b)

    axes[0, 2].set_visible(False)

    for ax, (stratum_key, stratum_title, agg) in zip(ax_cf, stratum_aggs):
        mean_factor, se_factor, _order, k, top_idx = _derived_factor_stats(
            agg, top_k=top_k, bootstrap_rounds=bootstrap_rounds
        )
        y = np.arange(k)
        ax.barh(
            y,
            mean_factor[top_idx][::-1],
            xerr=se_factor[top_idx][::-1],
            color=[DOMAIN_COLORS[domain_name_for_feature(int(i))] for i in top_idx][::-1],
            ecolor="#666666",
            capsize=1.5,
            linewidth=0,
            edgecolor="none",
            height=0.82,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([str(int(i)) for i in top_idx[::-1]], fontsize=6)
        ax.set_xlabel("Attribution (mean ± bootstrap SE)")
        ax.set_title(_fig5_subgroup_panel_title(stratum_key, stratum_title, agg.n_total), fontsize=7)
        ax.grid(True, axis="x", alpha=0.22, linewidth=0.4, linestyle="-")
        style_axis_nature(ax)

    fig.suptitle(
        "Fig. 5 | Factor-level interpretability across gait phase and AIS subgroups",
        fontsize=8,
        fontweight="normal",
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "fig5_km_interpretability_dk.png")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved Fig. 5 composite: {png_path}")
    return png_path


def save_km_interpretability_core3_full_stratum(
    *,
    raw_model: torch.nn.Module,
    stratum_key: str,
    stratum_title: str,
    dataloader,
    device: torch.device,
    out_dir: str,
    top_k: int = 20,
    bootstrap_rounds: int = 500,
    axg_chunk_size: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Cohort-style subgroup interpretability summary (Panels A/B/D only).

    Aggregates over *all* samples in the provided dataloader, then saves core3 artifacts.
    """
    agg = _aggregate_km_interpretability_from_dataloader(
        raw_model=raw_model,
        dataloader=dataloader,
        device=device,
        axg_chunk_size=axg_chunk_size,
    )
    if agg is None:
        return None
    return save_km_interpretability_core3_from_agg(
        agg,
        stratum_key=stratum_key,
        stratum_title=stratum_title,
        out_dir=out_dir,
        top_k=top_k,
        bootstrap_rounds=bootstrap_rounds,
    )


def save_km_interpretability_core6(
    raw_model: torch.nn.Module,
    stratum_key: str,
    stratum_title: str,
    batch: Dict[str, Any],
    device: torch.device,
    out_dir: str,
    max_samples: int = 8,
    bootstrap_rounds: int = 200,
) -> Optional[Dict[str, Any]]:
    """
    Build and save ``km_interpretability_core6_{stratum_key}.png`` plus JSON/CSV sidecars.

    Returns a stats dict (JSON-serializable) or None if attention×grad is unavailable.
    """
    if "label" not in batch:
        return None
    bsz = min(int(batch["video"].shape[0]), max_samples)
    if bsz < 1:
        return None
    b = _trim_batch(batch, bsz, device)
    video = b["video"]
    knowledge_map = b["knowledge_map"]
    texts = b["texts"]
    km_indices = b["km_indices"]
    video_indices = b["video_indices"]
    labels = b["label"].detach().cpu().numpy().astype(int).ravel()

    raw_model.eval()
    with torch.no_grad():
        _ = raw_model(
            video,
            knowledge_map,
            texts=texts,
            km_indices=km_indices,
            video_indices=video_indices,
        )
    attn_map = get_km_attention_map(raw_model, knowledge_map, use_last_layer=True, layer_index=-1)
    if attn_map is None:
        return None

    axg_map, _topk, logits_np = get_km_attention_x_grad_map(
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
        return None

    B, T, Fdim = axg_map.shape
    assert Fdim == F_TOTAL

    # Temporal attention (mean over features) for panel A
    attn_tf = np.asarray(attn_map, dtype=np.float64)
    temporal = attn_tf.mean(axis=2)  # (B, T)
    time_pct = np.linspace(0.0, 100.0, T)

    # Panel F: domain ablation AUC proxy
    probs_full = 1.0 / (1.0 + np.exp(-logits_np))
    auc_full = _safe_auc(labels, probs_full)
    ablation: Dict[str, Any] = {"full": auc_full}
    km_base = knowledge_map.detach()
    for dname, (lo, hi) in DOMAIN_SLICES.items():
        km_z = km_base.clone()
        km_z[:, :, lo:hi] = 0.0
        z_logits = _forward_logits_np(raw_model, video, km_z, texts, km_indices, video_indices)
        z_prob = 1.0 / (1.0 + np.exp(-z_logits))
        ablation[f"{dname}_zeroed"] = _safe_auc(labels, z_prob)

    # Mean attribution per factor (normalized heatmap)
    mean_attr_f = axg_map.mean(axis=(0, 1))  # (F,)
    top_order = np.argsort(-mean_attr_f)
    top30 = top_order[:30]
    se_f = _bootstrap_factor_se(axg_map, n_boot=bootstrap_rounds)

    # Representative sample: strongest absolute logit
    rep_i = int(np.argmax(np.abs(logits_np)))

    # --- Figure ---
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), facecolor="white")
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]
    ax_e, ax_f = axes[2, 0], axes[2, 1]

    # A: temporal attention by outcome
    for cls in (0, 1):
        mask = labels == cls
        if not np.any(mask):
            continue
        m = temporal[mask]
        mean_t = m.mean(axis=0)
        std_t = m.std(axis=0, ddof=1) if m.shape[0] > 1 else np.zeros_like(mean_t)
        ax_a.plot(time_pct, mean_t, label=f"label={cls} (n={int(mask.sum())})")
        ax_a.fill_between(time_pct, mean_t - std_t, mean_t + std_t, alpha=0.2)
    ax_a.set_xlabel("Time (% of sequence)")
    ax_a.set_ylabel("Mean attention (over features)")
    ax_a.set_title("A. Temporal attention by outcome")
    ax_a.legend(frameon=False, fontsize=8)
    ax_a.grid(True, alpha=0.3)

    # B: top-k factors (global mean attribution)
    kbar = min(12, Fdim)
    topk_idx = top_order[:kbar]
    colors_b = [DOMAIN_COLORS[domain_name_for_feature(int(i))] for i in topk_idx]
    ax_b.bar(range(kbar), mean_attr_f[topk_idx], color=colors_b, edgecolor="none")
    ax_b.set_xticks(range(kbar))
    ax_b.set_xticklabels([str(int(i)) for i in topk_idx], rotation=45, ha="right", fontsize=7)
    ax_b.set_ylabel("Mean attribution")
    ax_b.set_title("B. Top factors (attention×grad, mean over time & batch)")
    ax_b.grid(True, axis="y", alpha=0.3)

    # C: representative heatmap
    ax_c.imshow(axg_map[rep_i], aspect="auto", cmap="magma")
    ax_c.set_title(f"C. Representative sample (batch idx {rep_i}, logit={float(logits_np[rep_i]):.3f})")
    ax_c.set_xlabel("Feature")
    ax_c.set_ylabel("Time")

    # D: domain dynamics (fraction of |axg| mass per time)
    dom_curves = []
    for dname, (lo, hi) in DOMAIN_SLICES.items():
        seg = axg_map[:, :, lo:hi].sum(axis=2)  # (B, T)
        dom_curves.append(seg.mean(axis=0))
    dom_stack = np.stack(dom_curves, axis=0)  # (3, T)
    dom_stack = np.maximum(dom_stack, 0)
    row_sum = dom_stack.sum(axis=0, keepdims=True) + 1e-8
    dom_frac = dom_stack / row_sum
    ax_d.stackplot(
        time_pct,
        dom_frac[0],
        dom_frac[1],
        dom_frac[2],
        labels=list(DOMAIN_SLICES.keys()),
        colors=[DOMAIN_COLORS[k] for k in DOMAIN_SLICES],
        alpha=0.85,
    )
    ax_d.set_xlim(time_pct[0], time_pct[-1])
    ax_d.set_ylim(0.0, 1.0)
    ax_d.set_xlabel("Time (% of sequence)")
    ax_d.set_ylabel("Fraction of attribution mass")
    ax_d.set_title("D. Domain importance dynamics")
    ax_d.legend(loc="upper right", frameon=False, fontsize=8)
    ax_d.grid(True, alpha=0.2)

    # E: top 30 factors with SE
    ranks = np.arange(1, 31)
    ax_e.barh(
        ranks[::-1],
        mean_attr_f[top30],
        xerr=se_f[top30],
        color=[DOMAIN_COLORS[domain_name_for_feature(int(i))] for i in top30][::-1],
        ecolor="gray",
        capsize=2,
    )
    ax_e.set_yticks(ranks)
    ax_e.set_yticklabels([str(int(i)) for i in top30[::-1]], fontsize=6)
    ax_e.set_xlabel("Mean attribution ± bootstrap SE")
    ax_e.set_title("E. Top 30 factors (stability)")
    ax_e.grid(True, axis="x", alpha=0.3)

    # F: ablation AUC bar
    names_f = ["full", "motion\nzeroed", "skeleton\nzeroed", "signal\nzeroed"]
    keys_f = ["full", "motion_zeroed", "skeleton_zeroed", "signal_zeroed"]
    vals = [ablation.get(k) for k in keys_f]
    vals_plot = [v if v is not None else 0.0 for v in vals]
    colors_f = ["#333333", DOMAIN_COLORS["motion"], DOMAIN_COLORS["skeleton"], DOMAIN_COLORS["signal"]]
    bars = ax_f.bar(range(4), vals_plot, color=colors_f, edgecolor="none")
    ax_f.set_xticks(range(4))
    ax_f.set_xticklabels(names_f, fontsize=8)
    ax_f.set_ylabel("AUC (ranking proxy)")
    ax_f.set_title("F. Domain ablation proxy (batch)")
    ax_f.set_ylim(0.0, 1.05)
    ax_f.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        if v is None:
            ax_f.text(i, 0.02, "n/a", ha="center", fontsize=8)
        else:
            ax_f.text(i, min(v + 0.02, 1.0), f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle(
        f"KM interpretability (core 6): {stratum_title}\n"
        f"n={B}, KM={T}×{Fdim}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    png_name = f"km_interpretability_core6_{stratum_key}.png"
    png_path = os.path.join(out_dir, png_name)
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # CSV top factors
    csv_path = os.path.join(out_dir, f"km_interpretability_top_factors_{stratum_key}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["rank", "factor_index", "domain", "mean_attribution", "bootstrap_se"])
        for r, fi in enumerate(top_order[:50], start=1):
            fi = int(fi)
            w.writerow(
                [
                    r,
                    fi,
                    domain_name_for_feature(fi),
                    f"{float(mean_attr_f[fi]):.8f}",
                    f"{float(se_f[fi]):.8f}",
                ]
            )

    top1 = int(top_order[0])
    stats: Dict[str, Any] = {
        "stratum_key": stratum_key,
        "stratum_title": stratum_title,
        "n_samples": int(B),
        "km_shape": [int(T), int(Fdim)],
        "label_counts": {str(int(k)): int(np.sum(labels == k)) for k in np.unique(labels)},
        "representative_batch_index": rep_i,
        "representative_logit": float(logits_np[rep_i]),
        "panel_f_ablation_auc": {k: (float(v) if v is not None else None) for k, v in ablation.items()},
        "top_factor": {
            "index": top1,
            "domain": domain_name_for_feature(top1),
            "mean_attribution": float(mean_attr_f[top1]),
            "bootstrap_se": float(se_f[top1]),
        },
        "figure_path": png_path,
        "csv_path": csv_path,
    }
    json_path = os.path.join(out_dir, f"km_interpretability_core6_{stratum_key}_stats.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(stats, jf, indent=2)
    stats["stats_json_path"] = json_path
    print(f"Saved core-6 interpretability: {png_path}")
    return stats


def write_subgroup_interpretability_markdown(
    out_dir: str,
    strata_stats: List[Dict[str, Any]],
    log_tag: str,
) -> str:
    """
    Write ``km_interpretability_dk_subgroups.md`` under ``out_dir`` with run-specific numbers.
    """
    path = os.path.join(out_dir, "km_interpretability_dk_subgroups.md")
    lines: List[str] = [
        "## DK subgroup knowledge-map interpretability",
        "",
        f"Run log directory: `{log_tag}`",
        "",
        "For each clinical stratum defined in `data/subgroup_indices.json`, we generated a subgroup-level "
        "KM interpretability summary aggregated over **all patches in the stratum**. Domains: motion "
        "`[0:34)`, skeleton `[34:172)`, signal cross-correlation `[172:238)`.",
        "",
        "Artifacts (in this log folder):",
        "",
        "- `fig5_km_interpretability_dk.png` — Fig. 5 composite (2×3: temporal attention, domain dynamics, top factors per stratum)",
        "- `km_interpretability_core3_<stratum>.png` — per-stratum 1×3 figure (temporal / top factors / domain dynamics)",
        "- `km_interpretability_core3_<stratum>_stats.json` — counts and paths",
        "- `km_interpretability_top_factors_<stratum>.csv` — ranked factors",
        "",
        "### Stratum summaries",
        "",
    ]
    for s in strata_stats:
        key = s.get("stratum_key", "?")
        title = s.get("stratum_title", key)
        lines.append(f"#### {title} (`{key}`)")
        lines.append("")
        lines.append(f"- Analysis batch size: **{s.get('n_samples', '?')}**")
        lc = s.get("label_counts", {})
        lines.append(f"- Label counts (binary): {lc}")
        lines.append(
            f"- Top factors: see `km_interpretability_top_factors_{key}.csv` "
            f"(mean attribution ± bootstrap SE; Option 1 combined B/E)."
        )
        lines.append("")
    lines.append(
        "### Interpretation notes\n\n"
        "These subgroup figures remove single-sample and ablation proxy panels (no C/F). "
        "Panels A/B/D are aggregated over **all stratum patches**, supporting subgroup comparisons "
        "with reduced sensitivity to batch selection. For predictive performance, use `subgroup_eval.json` "
        "and `test_results.txt` from the same run."
    )
    lines.append("")
    with open(path, "w", encoding="utf-8") as mf:
        mf.write("\n".join(lines))
    print(f"Wrote subgroup interpretability markdown: {path}")
    return path
