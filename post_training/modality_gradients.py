# -*- coding: utf-8 -*-
"""Per-modality gradient attribution during inference (video / KM / text)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils.data_sampler import SigLIPFullGaitDatasetPKL, fullgait_collate_fn
from utils.km_interpretability_core6 import DOMAIN_COLORS, DOMAIN_SLICES
from utils.plot_style_nature import apply_nature_journal_mpl_style, mm_to_inch, style_axis_nature

MODALITY_LABELS = ("video", "km", "text")
MODALITY_COLORS = {"video": "#D55E00", "km": "#009E73", "text": "#0072B2"}


@dataclass
class ModalityGradResult:
    logit: float
    probability: float
    checkpoint_path: str
    checkpoint_md5: str
    head_token_norms: Dict[str, float]
    head_token_pcts: Dict[str, float]
    video_temporal_grad: np.ndarray
    km_grad_heatmap: np.ndarray
    text_topk_dims: List[int]
    text_topk_vals: List[float]
    # Backward-compatible aliases (head-only attribution)
    token_norms: Dict[str, float] = field(init=False)
    token_pcts: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.token_norms = self.head_token_norms
        self.token_pcts = self.head_token_pcts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logit": self.logit,
            "probability": self.probability,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_md5": self.checkpoint_md5,
            "head_token_norms": self.head_token_norms,
            "head_token_pcts": self.head_token_pcts,
            "token_norms": self.head_token_norms,
            "token_pcts": self.head_token_pcts,
            "video_temporal_grad": self.video_temporal_grad.tolist(),
            "km_grad_heatmap_shape": list(self.km_grad_heatmap.shape),
            "text_topk_dims": self.text_topk_dims,
            "text_topk_vals": self.text_topk_vals,
        }


def checkpoint_fingerprint(path: Path) -> Dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "md5": digest.hexdigest(),
        "size_bytes": str(path.stat().st_size),
    }


@dataclass
class ExampleRecord:
    key: str
    case_tag: str
    record: Dict[str, Any]

    @property
    def subject_id(self) -> int:
        return int(self.record["subject_id"])

    @property
    def source_file(self) -> str:
        return str(self.record.get("source_file", ""))

    @property
    def pkl_path(self) -> str:
        return str(self.record["pkl_path"])


def _case_tag(rec: Dict[str, Any]) -> str:
    label = float(rec.get("label", 0))
    pred = int(rec.get("prediction", 0))
    if label >= 0.5 and pred == 1:
        return "TP"
    if label < 0.5 and pred == 0:
        return "TN"
    if label >= 0.5 and pred == 0:
        return "FN"
    return "FP"


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _l2_norm(grad: Optional[torch.Tensor]) -> float:
    if grad is None:
        return 0.0
    return float(grad.detach().norm().item())


def _pct_breakdown(norms: Dict[str, float]) -> Dict[str, float]:
    total = sum(norms.values())
    if total <= 0:
        n = len(norms)
        return {k: 100.0 / n for k in norms}
    return {k: 100.0 * v / total for k, v in norms.items()}


def load_inference_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def select_example_records(
    records: List[Dict[str, Any]],
    *,
    subject_ids: Optional[List[int]] = None,
) -> List[ExampleRecord]:
    """Pick TP, TN, and error/borderline examples."""
    if subject_ids:
        by_subject = {int(r["subject_id"]): r for r in records}
        tags = ["TP", "TN", "ERR"]
        out: List[ExampleRecord] = []
        for sid, tag in zip(subject_ids, tags):
            if sid not in by_subject:
                raise ValueError(f"Subject {sid} not found in inference results")
            rec = by_subject[sid]
            out.append(ExampleRecord(key=f"custom_{sid}", case_tag=tag, record=rec))
        return out

    labeled = [r for r in records if r.get("has_ground_truth", True)]
    tps = [r for r in labeled if _case_tag(r) == "TP"]
    tns = [r for r in labeled if _case_tag(r) == "TN"]
    errors = [r for r in labeled if _case_tag(r) in ("FN", "FP")]

    if not tps or not tns:
        raise ValueError("Could not find both TP and TN examples in inference results")

    tp_rec = max(tps, key=lambda r: float(r["probability"]))
    tn_rec = min(tns, key=lambda r: float(r["probability"]))
    if errors:
        err_rec = min(errors, key=lambda r: abs(float(r["probability"]) - 0.5))
        err_tag = _case_tag(err_rec)
    else:
        err_rec = min(labeled, key=lambda r: abs(float(r["probability"]) - 0.5))
        err_tag = "borderline"

    return [
        ExampleRecord(key="A", case_tag="TP", record=tp_rec),
        ExampleRecord(key="B", case_tag="TN", record=tn_rec),
        ExampleRecord(
            key="C",
            case_tag=err_tag,
            record=err_rec,
        ),
    ]


def load_single_patch_batch(
    *,
    pkl_dir: Path,
    pkl_path: Path,
    prompts_path: Path,
    prompt_selection: str,
) -> Dict[str, Any]:
    dataset = SigLIPFullGaitDatasetPKL(
        pkl_data_dir=str(pkl_dir),
        mode="test",
        km_gaussian_noise_std=None,
        prompts_path=str(prompts_path),
        prompt_selection=prompt_selection,
    )
    target = str(pkl_path.resolve())
    idx = None
    for i, meta in enumerate(dataset.patch_metadata):
        if str(Path(meta["pkl_path"]).resolve()) == target:
            idx = i
            break
    if idx is None:
        raise FileNotFoundError(f"Patch not in dataset metadata: {pkl_path}")
    return fullgait_collate_fn([dataset[idx]])


def _extract_logit(preds: torch.Tensor) -> torch.Tensor:
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds[0, 0]
    if preds.ndim == 2 and preds.shape[1] > 1:
        return preds[0, 1]
    return preds.reshape(-1)[0]


def _encode_modality_tokens(
    raw_model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run frozen encoders once; return detached modality tokens."""
    video = batch["video"].to(device)
    km = batch["knowledge_map"].to(device)
    texts = batch.get("texts")
    km_indices = batch.get("km_indices")
    video_indices = batch.get("video_indices")
    if km_indices is not None:
        km_indices = km_indices.to(device)
    if video_indices is not None:
        video_indices = video_indices.to(device)

    text_capture: Dict[str, torch.Tensor] = {}
    text_module = getattr(raw_model, "text_norm", None) or getattr(raw_model, "text_proj", None)
    hook_handle = None

    def _text_hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            text_capture["tensor"] = output.detach()

    if text_module is not None:
        hook_handle = text_module.register_forward_hook(_text_hook)

    try:
        with torch.no_grad():
            out = raw_model(
                video,
                km,
                texts=texts,
                km_indices=km_indices,
                video_indices=video_indices,
                return_embeddings=True,
            )
            video_token = out["video_emb"].detach()
            km_token = out["km_emb"].detach()
            if "tensor" not in text_capture:
                hidden = int(getattr(raw_model, "hidden_dim", video_token.shape[-1]))
                text_token = torch.zeros(1, hidden, device=device, dtype=video_token.dtype)
            else:
                text_token = text_capture["tensor"]
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return video_token, km_token, text_token


def _forward_head_from_tokens(
    raw_model: nn.Module,
    video_token: torch.Tensor,
    km_token: torch.Tensor,
    text_token: torch.Tensor,
) -> torch.Tensor:
    batch_size = video_token.shape[0]
    stacked = torch.cat(
        [video_token.unsqueeze(1), km_token.unsqueeze(1), text_token.unsqueeze(1)],
        dim=1,
    )
    if getattr(raw_model, "use_latent_pooling", False):
        latents = raw_model.latent_query.expand(batch_size, -1, -1)
        latent_out, _ = raw_model.latent_attention(latents, stacked, stacked)
        latent_out = raw_model.latent_norm(latent_out)
        fused = latent_out.mean(dim=1)
    else:
        fused = stacked.reshape(batch_size, -1)
    return raw_model.regressor(fused)


def _input_level_grad_maps(
    raw_model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Input gradients through full model (dominated by shared frozen encoders)."""
    video = batch["video"].to(device).detach().clone().requires_grad_(True)
    km = batch["knowledge_map"].to(device).detach().clone().requires_grad_(True)
    texts = batch.get("texts")
    km_indices = batch.get("km_indices")
    video_indices = batch.get("video_indices")
    if km_indices is not None:
        km_indices = km_indices.to(device)
    if video_indices is not None:
        video_indices = video_indices.to(device)

    with torch.enable_grad():
        preds = raw_model(
            video,
            km,
            texts=texts,
            km_indices=km_indices,
            video_indices=video_indices,
        )
        logit = _extract_logit(preds)
        raw_model.zero_grad(set_to_none=True)
        logit.backward()

    if video.grad is not None:
        g = video.grad[0].abs()
        video_temporal = (
            g.detach().cpu().numpy()
            if g.ndim == 1
            else g.mean(dim=tuple(range(1, g.ndim))).detach().cpu().numpy()
        )
    else:
        video_temporal = np.zeros(int(video.shape[1]))

    if km.grad is not None:
        km_heat = km.grad[0].abs().detach().cpu().numpy()
    else:
        km_heat = np.zeros(km.shape[1:])

    return video_temporal, km_heat


def compute_modality_gradients(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    *,
    checkpoint_path: Optional[Path] = None,
    text_topk: int = 10,
) -> ModalityGradResult:
    """
    Head-only modality token gradients (fusion + regressor) plus input-level maps.

    Modality % bars use head-only ∂logit/∂token so post-trained checkpoints differ
    from baseline. Input heatmaps still reflect shared frozen encoders.
    """
    raw_model = _unwrap(model)
    raw_model.eval()

    ckpt = checkpoint_fingerprint(checkpoint_path) if checkpoint_path else {
        "path": "",
        "md5": "",
    }

    video_token, km_token, text_token = _encode_modality_tokens(raw_model, batch, device)
    v = video_token.detach().clone().requires_grad_(True)
    k = km_token.detach().clone().requires_grad_(True)
    t = text_token.detach().clone().requires_grad_(True)

    with torch.enable_grad():
        preds = _forward_head_from_tokens(raw_model, v, k, t)
        logit = _extract_logit(preds)
        raw_model.zero_grad(set_to_none=True)
        logit.backward()

        head_token_norms = {
            "video": _l2_norm(v.grad),
            "km": _l2_norm(k.grad),
            "text": _l2_norm(t.grad),
        }
        head_token_pcts = _pct_breakdown(head_token_norms)

        text_topk_dims: List[int] = []
        text_topk_vals: List[float] = []
        if t.grad is not None:
            g = t.grad[0].abs()
            k_top = min(text_topk, g.numel())
            vals, idxs = torch.topk(g, k=k_top)
            text_topk_dims = idxs.detach().cpu().tolist()
            text_topk_vals = vals.detach().cpu().tolist()

    video_temporal, km_heat = _input_level_grad_maps(raw_model, batch, device)
    prob = float(torch.sigmoid(logit).item())

    return ModalityGradResult(
        logit=float(logit.item()),
        probability=prob,
        checkpoint_path=ckpt["path"],
        checkpoint_md5=ckpt["md5"],
        head_token_norms=head_token_norms,
        head_token_pcts=head_token_pcts,
        video_temporal_grad=video_temporal,
        km_grad_heatmap=km_heat,
        text_topk_dims=text_topk_dims,
        text_topk_vals=text_topk_vals,
    )


def _plot_modality_bars(ax, result: ModalityGradResult, title: str) -> None:
    vals = [result.head_token_pcts[m] for m in MODALITY_LABELS]
    colors = [MODALITY_COLORS[m] for m in MODALITY_LABELS]
    ax.bar(MODALITY_LABELS, vals, color=colors, width=0.55)
    ax.set_ylabel("Head grad mass (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=7)
    style_axis_nature(ax)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{v:.2f}%", ha="center", va="bottom", fontsize=5.5)


def _plot_logit_prob(ax, result: ModalityGradResult, title: str) -> None:
    ax.bar(["logit", "prob"], [result.logit, result.probability], color=["#666666", "#CC79A7"], width=0.45)
    ax.axhline(0.5, color="0.7", linewidth=0.5, linestyle=":")
    ax.set_title(title, fontsize=7)
    style_axis_nature(ax)
    for i, (label, val) in enumerate(zip(["logit", "prob"], [result.logit, result.probability])):
        ax.text(i, val, f"{val:.4f}", ha="center", va="bottom", fontsize=5.5)


def _plot_km_delta_heatmap(
    ax,
    heatmap: np.ndarray,
    title: str,
    *,
    symmetric: bool = False,
) -> None:
    if symmetric:
        vmax = max(float(np.max(np.abs(heatmap))), 1e-12)
        im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(heatmap, aspect="auto", cmap="magma", origin="lower")
    _add_km_domain_bands(ax)
    ax.set_xlabel("KM feature")
    ax.set_ylabel("Time")
    ax.set_title(title, fontsize=7)
    style_axis_nature(ax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _plot_video_temporal(ax, result: ModalityGradResult, title: str) -> None:
    t = np.arange(len(result.video_temporal_grad))
    ax.plot(t, result.video_temporal_grad, color=MODALITY_COLORS["video"], linewidth=1.0)
    ax.set_xlabel("Video frame")
    ax.set_ylabel("Mean |grad|")
    ax.set_title(title)
    style_axis_nature(ax)


def _add_km_domain_bands(ax) -> None:
    for name, (start, end) in DOMAIN_SLICES.items():
        ax.axvspan(start, end, color=DOMAIN_COLORS[name], alpha=0.08, lw=0)
    ax.axvline(34, color="0.5", linewidth=0.4, linestyle="--")
    ax.axvline(172, color="0.5", linewidth=0.4, linestyle="--")


def _plot_km_heatmap(ax, result: ModalityGradResult, title: str) -> None:
    heat = result.km_grad_heatmap
    im = ax.imshow(heat, aspect="auto", cmap="magma", origin="lower")
    _add_km_domain_bands(ax)
    ax.set_xlabel("KM feature")
    ax.set_ylabel("Time")
    ax.set_title(title)
    style_axis_nature(ax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _plot_text_topk(ax, result: ModalityGradResult, title: str) -> None:
    if not result.text_topk_dims:
        ax.text(0.5, 0.5, "No text gradient", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    dims = [str(d) for d in result.text_topk_dims]
    ax.barh(dims[::-1], result.text_topk_vals[::-1], color=MODALITY_COLORS["text"])
    ax.set_xlabel("|grad|")
    ax.set_ylabel("Embedding dim")
    ax.set_title(title)
    style_axis_nature(ax)


def plot_example_figure(
    example: ExampleRecord,
    model_results: List[Tuple[str, ModalityGradResult]],
    out_path: Path,
) -> None:
    apply_nature_journal_mpl_style()
    n_cols = len(model_results)
    baseline_km = model_results[0][1].km_grad_heatmap
    fig, axes = plt.subplots(
        5,
        n_cols,
        figsize=(mm_to_inch(92 * n_cols), mm_to_inch(230)),
        facecolor="white",
        gridspec_kw={"height_ratios": [1.0, 0.8, 1.0, 1.2, 0.9]},
    )
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    rec = example.record
    label = float(rec.get("label", 0))
    prob_parts = " | ".join(
        f"{mlabel.split()[0].lower()} p={result.probability:.4f}"
        for mlabel, result in model_results
    )
    title = f"{example.source_file} | {example.case_tag} | label={int(label)} | {prob_parts}"
    fig.suptitle(title, fontsize=8, y=0.995)

    for col, (mlabel, result) in enumerate(model_results):
        ck_short = result.checkpoint_md5[:8] if result.checkpoint_md5 else "?"
        col_title = f"{mlabel}\n{Path(result.checkpoint_path).name}\nmd5={ck_short}"
        _plot_modality_bars(axes[0, col], result, f"{col_title}\nhead ∂logit/∂token")
        _plot_logit_prob(axes[1, col], result, "Prediction")
        _plot_video_temporal(
            axes[2, col],
            result,
            f"{mlabel}: input video |grad| (shared encoders)",
        )
        if col == 0:
            _plot_km_delta_heatmap(axes[3, col], result.km_grad_heatmap, f"{mlabel}: KM input |grad|")
        else:
            delta = result.km_grad_heatmap - baseline_km
            _plot_km_delta_heatmap(
                axes[3, col],
                delta,
                f"{mlabel}: ΔKM |grad| vs baseline",
                symmetric=True,
            )
        _plot_text_topk(axes[4, col], result, f"{mlabel}: text token top-|grad|")

    logit_parts = " | ".join(
        f"{mlabel.split()[0].lower()}={result.logit:.4f}" for mlabel, result in model_results
    )
    fig.text(
        0.5,
        0.01,
        (
            f"Subject {example.subject_id} | pred={rec.get('prediction')} | logits: {logit_parts} | "
            "Row1=head-only (post-train differs); rows3-4=input grad (encoders frozen → nearly identical)"
        ),
        ha="center",
        fontsize=5.5,
        color="0.35",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def example_filename(example: ExampleRecord) -> str:
    src = example.source_file.replace(".mp4", "").replace("_step1", "")
    return f"example_{example.key}_{example.case_tag}_{src}.png"
