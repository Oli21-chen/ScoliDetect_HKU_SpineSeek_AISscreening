# -*- coding: utf-8 -*-
"""Hyperparameter grids for post-training search."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, fields
from typing import Any, Dict, Iterator, List

from post_training.config import HeadFinetuneConfig

SMALL_GRID: Dict[str, List[Any]] = {
    "lr": [1e-8, 1e-7, 1e-6, 1e-5],
    "epochs": [5, 10, 20],
}

MEDIUM_GRID: Dict[str, List[Any]] = {
    **SMALL_GRID,
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
}

PHASE_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "small": SMALL_GRID,
    "medium": MEDIUM_GRID,
}

PHASE2_ANCHOR: Dict[str, Any] = {
    "lr": 1e-7,
    "epochs": 10,
    "weight_decay": HeadFinetuneConfig.weight_decay,
    "km_gaussian_noise_std": HeadFinetuneConfig.km_gaussian_noise_std,
    "loss_type": HeadFinetuneConfig.loss_type,
}


@dataclass(frozen=True)
class HpVariant:
    name: str
    lr: float
    epochs: int
    batch_size: int = HeadFinetuneConfig.batch_size
    weight_decay: float = HeadFinetuneConfig.weight_decay
    km_gaussian_noise_std: float = HeadFinetuneConfig.km_gaussian_noise_std
    loss_type: str = HeadFinetuneConfig.loss_type
    num_workers: int = HeadFinetuneConfig.num_workers
    verbose: bool = False

    def to_finetune_config(self) -> HeadFinetuneConfig:
        return HeadFinetuneConfig(
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            km_gaussian_noise_std=self.km_gaussian_noise_std,
            loss_type=self.loss_type,
            num_workers=self.num_workers,
            verbose=self.verbose,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "name"}


def _format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _format_wd(wd: float) -> str:
    if wd == 0.0:
        return "0"
    return _format_lr(wd)


def _format_noise(noise: float) -> str:
    if noise == 0.0:
        return "0"
    text = f"{noise:g}"
    return text


def variant_name(hp: Dict[str, Any]) -> str:
    """e.g. lr1e-07_ep10_wd0_focal_noise0.05"""
    name = f"lr{_format_lr(float(hp['lr']))}_ep{int(hp['epochs'])}"
    default_wd = HeadFinetuneConfig.weight_decay
    default_noise = HeadFinetuneConfig.km_gaussian_noise_std
    default_loss = HeadFinetuneConfig.loss_type

    wd = float(hp.get("weight_decay", default_wd))
    noise = float(hp.get("km_gaussian_noise_std", default_noise))
    loss = str(hp.get("loss_type", default_loss))

    if wd != default_wd:
        name += f"_wd{_format_wd(wd)}"
    if loss != default_loss:
        name += "_focal"
    if noise != default_noise:
        name += f"_noise{_format_noise(noise)}"
    return name


def _make_phase2_variant(**overrides: Any) -> HpVariant:
    hp = {**PHASE2_ANCHOR, **overrides}
    name = variant_name(hp)
    return HpVariant(
        name=name,
        lr=float(hp["lr"]),
        epochs=int(hp["epochs"]),
        weight_decay=float(hp["weight_decay"]),
        km_gaussian_noise_std=float(hp["km_gaussian_noise_std"]),
        loss_type=str(hp["loss_type"]),
        verbose=False,
    )


PHASE2_VARIANTS: List[HpVariant] = [
    # refined_lr (epochs=10, wd=1e-4, bce, noise=0.01)
    _make_phase2_variant(lr=3e-8),
    _make_phase2_variant(lr=3e-7),
    _make_phase2_variant(lr=5e-7),
    _make_phase2_variant(lr=8e-7),
    # weight_decay (lr=1e-7, epochs=10, bce, noise=0.01)
    _make_phase2_variant(weight_decay=0.0),
    _make_phase2_variant(weight_decay=1e-5),
    _make_phase2_variant(weight_decay=1e-3),
    _make_phase2_variant(weight_decay=1e-2),
    # loss_noise (lr=1e-7, epochs=10, wd=1e-4)
    _make_phase2_variant(km_gaussian_noise_std=0.0),
    _make_phase2_variant(loss_type="focal"),
    _make_phase2_variant(loss_type="focal", km_gaussian_noise_std=0.0),
    _make_phase2_variant(km_gaussian_noise_std=0.05),
]


def iter_variants(phase: str = "small") -> Iterator[HpVariant]:
    if phase == "phase2":
        yield from PHASE2_VARIANTS
        return

    grid = PHASE_GRIDS.get(phase)
    if grid is None:
        raise ValueError(
            f"Unknown phase {phase!r}; choose from {list(PHASE_GRIDS) + ['phase2']}"
        )

    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        hp = dict(zip(keys, values))
        name = variant_name(hp)
        defaults = HpVariant(name=name, lr=1e-7, epochs=1)
        merged = {**defaults.as_dict(), **hp, "name": name}
        yield HpVariant(**merged)
