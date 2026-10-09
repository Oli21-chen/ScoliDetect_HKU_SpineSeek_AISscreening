# -*- coding: utf-8 -*-
"""Load deploy-compatible SFTRegressor checkpoints for post-training."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from post_training.config import SCOLI_ROOT

PYTORCH_ROOT = Path(__file__).resolve().parents[1]


def _ensure_scoli_root() -> None:
    root = str(SCOLI_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """Instantiate deploy SFTRegressor (same class used by 03_Infer)."""
    _ensure_scoli_root()
    from models.sft_regressor import SFTRegressor

    sig = inspect.signature(SFTRegressor.__init__)
    valid_keys = set(sig.parameters) - {"self"}
    kwargs = {k: v for k, v in config.items() if k in valid_keys}
    defaults = {
        "km_feature_dim": 238,
        "hidden_dim": 256,
        "label_dim": 1,
        "video_encoder_type": "vivit",
        "km_encoder_type": "vit",
        "use_text": True,
    }
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    return SFTRegressor(**kwargs)


def load_deploy_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Build model from checkpoint config and load full state dict."""
    checkpoint_path = str(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    if not config:
        raise ValueError(f"Checkpoint has no 'config' dict: {checkpoint_path}")

    model = build_model_from_config(config)
    load_info = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if load_info.missing_keys:
        print(f"Warning: {len(load_info.missing_keys)} missing keys when loading checkpoint")
    if load_info.unexpected_keys:
        print(f"Warning: {len(load_info.unexpected_keys)} unexpected keys in checkpoint")
    model.to(device)
    return model, config, checkpoint
