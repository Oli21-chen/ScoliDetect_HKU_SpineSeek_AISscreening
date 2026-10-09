# -*- coding: utf-8 -*-
"""Freeze policies for head-only and encoder-level post-training."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

import torch.nn as nn

ENCODER_PREFIXES = ("video_encoder.", "km_encoder.", "text_encoder.")
VIDEO_KM_PREFIXES = ("video_encoder.", "km_encoder.")

FreezePolicy = Literal["head_only", "partial_unfreeze", "full_encoder"]


def _set_requires_grad_by_prefix(model: nn.Module, prefixes: Tuple[str, ...], requires_grad: bool) -> None:
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad = requires_grad


def _set_requires_grad_for_all(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def _count_params(model: nn.Module) -> Tuple[int, int]:
    frozen = trainable = 0
    for param in model.parameters():
        n = param.numel()
        if param.requires_grad:
            trainable += n
        else:
            frozen += n
    return frozen, trainable


def _unfreeze_last_n_blocks(module: nn.Module, n_blocks: int) -> int:
    """Unfreeze the last N blocks in a ViT/ViViT-style encoder. Returns blocks unfrozen."""
    if n_blocks <= 0:
        return 0
    candidates: List[nn.ModuleList] = []
    for attr in ("blocks", "layers"):
        if hasattr(module, attr):
            obj = getattr(module, attr)
            if isinstance(obj, nn.ModuleList):
                candidates.append(obj)
    if hasattr(module, "transformer"):
        transformer = getattr(module, "transformer")
        for attr in ("blocks", "layers"):
            if hasattr(transformer, attr):
                obj = getattr(transformer, attr)
                if isinstance(obj, nn.ModuleList):
                    candidates.append(obj)

    if not candidates:
        return 0
    blocks = candidates[0]
    n = min(len(blocks), n_blocks)
    for block in list(blocks)[-n:]:
        for p in block.parameters():
            p.requires_grad = True
    return n


def apply_head_only_policy(model: nn.Module) -> Dict[str, int]:
    """
    Freeze video/km/text encoders; enable gradients on fusion + regressor modules.

    Returns counts of frozen and trainable parameters.
    """
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if name.startswith(ENCODER_PREFIXES):
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()
    return {"frozen_params": frozen, "trainable_params": trainable, "policy": "head_only"}


def apply_partial_unfreeze_policy(
    model: nn.Module,
    *,
    n_blocks: int = 2,
) -> Dict[str, Any]:
    """
    Freeze all encoders, then unfreeze last N blocks of video/km encoders.
    Fusion + regressor remain trainable. text_encoder stays frozen.
    """
    _set_requires_grad_for_all(model, True)
    _set_requires_grad_by_prefix(model, ENCODER_PREFIXES, False)

    video_unfrozen = _unfreeze_last_n_blocks(model.video_encoder, n_blocks)
    km_unfrozen = _unfreeze_last_n_blocks(model.km_encoder, n_blocks)

    if video_unfrozen == 0:
        print("Warning: partial_unfreeze found 0 video blocks; unfreezing full video_encoder")
        _set_requires_grad_by_prefix(model, ("video_encoder.",), True)
        video_unfrozen = -1
    if km_unfrozen == 0:
        print("Warning: partial_unfreeze found 0 km blocks; unfreezing full km_encoder")
        _set_requires_grad_by_prefix(model, ("km_encoder.",), True)
        km_unfrozen = -1

    frozen, trainable = _count_params(model)
    return {
        "frozen_params": frozen,
        "trainable_params": trainable,
        "policy": "partial_unfreeze",
        "unfreeze_n_blocks": n_blocks,
        "video_blocks_unfrozen": video_unfrozen,
        "km_blocks_unfrozen": km_unfrozen,
    }


def apply_full_encoder_policy(model: nn.Module) -> Dict[str, int]:
    """Unfreeze video+km encoders and fusion head; keep text_encoder frozen."""
    _set_requires_grad_for_all(model, True)
    _set_requires_grad_by_prefix(model, ("text_encoder.",), False)
    frozen, trainable = _count_params(model)
    return {
        "frozen_params": frozen,
        "trainable_params": trainable,
        "policy": "full_encoder",
    }


def apply_freeze_policy(
    model: nn.Module,
    policy: FreezePolicy,
    *,
    unfreeze_n_blocks: int = 2,
) -> Dict[str, Any]:
    if policy == "head_only":
        return apply_head_only_policy(model)
    if policy == "partial_unfreeze":
        return apply_partial_unfreeze_policy(model, n_blocks=unfreeze_n_blocks)
    if policy == "full_encoder":
        return apply_full_encoder_policy(model)
    raise ValueError(f"Unknown freeze policy: {policy}")


def is_encoder_param(name: str) -> bool:
    return name.startswith(VIDEO_KM_PREFIXES)


def build_param_groups(
    model: nn.Module,
    *,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """AdamW param groups: encoder (video+km, requires_grad) vs fusion/regressor."""
    encoder_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_encoder_param(name):
            encoder_params.append(param)
        else:
            head_params.append(param)

    groups: List[Dict[str, Any]] = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    return groups


def trainable_parameters(model: nn.Module):
    """Yield parameters with requires_grad=True."""
    return (p for p in model.parameters() if p.requires_grad)


def count_trainable(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
