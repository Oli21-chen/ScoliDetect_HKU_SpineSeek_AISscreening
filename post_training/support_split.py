# -*- coding: utf-8 -*-
"""Resolve support (first pos + first neg) and holdout subject indices."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from post_training.config import (
    BINARY_THRESHOLD,
    CSV_SUFFIX,
    LABEL_EXCEL,
    MIN_FILE_INDEX,
    TABLE_DIR,
    VIDEO_DIR,
)


def discover_paired_indices(
    table_path: Path,
    video_path: Path,
    para_name: str = "sz",
    csv_suffix: str = CSV_SUFFIX,
) -> List[int]:
    """Return sorted indices with both pose CSV and trimmed video present."""
    prefix = f"{para_name}_"
    indices: List[int] = []
    table_path = Path(table_path)
    video_path = Path(video_path)
    if not table_path.is_dir():
        return indices
    for fname in os.listdir(table_path):
        if not fname.startswith(prefix) or not fname.endswith(csv_suffix):
            continue
        mid = fname[len(prefix) : -len(csv_suffix)]
        try:
            index = int(mid)
        except ValueError:
            continue
        table_name = table_path / fname
        video_name = video_path / f"{para_name}_{index}_step1.mp4"
        if table_name.is_file() and video_name.is_file():
            indices.append(index)
    return sorted(indices)


def load_labeled_subjects(
    label_excel: Path = LABEL_EXCEL,
    min_file_index: int = MIN_FILE_INDEX,
    binary_threshold: float = BINARY_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (subject_indices, binary_labels) for subjects >= min_file_index."""
    dictionary = pd.read_excel(
        label_excel,
        usecols=["File No.", "age", "sex", "M_cobb_l", "M_cobb_r"],
    ).dropna()
    dictionary_val = dictionary.values
    starting_index = np.where(dictionary_val[:, 0] == min_file_index)[0][0]
    sub_index = dictionary_val[starting_index:, 0].astype(int)
    sub_label = dictionary_val[starting_index:, 3:5]
    labels = np.array(
        [1.0 if max(row) >= binary_threshold else 0.0 for row in sub_label],
        dtype=float,
    )
    return sub_index, labels


def _collect_paired_pos_neg_lists(
    table_path: Path = TABLE_DIR,
    video_path: Path = VIDEO_DIR,
    label_excel: Path = LABEL_EXCEL,
    min_file_index: int = MIN_FILE_INDEX,
    binary_threshold: float = BINARY_THRESHOLD,
    *,
    max_per_class: int | None = None,
) -> Dict[str, List[int]]:
    """Collect paired positive/negative subject indices in label-Excel order."""
    paired = set(discover_paired_indices(table_path, video_path))
    sub_index, labels = load_labeled_subjects(label_excel, min_file_index, binary_threshold)

    pos_list: List[int] = []
    neg_list: List[int] = []
    for idx, label in zip(sub_index, labels):
        if int(idx) not in paired:
            continue
        if label >= 0.5:
            pos_list.append(int(idx))
        else:
            neg_list.append(int(idx))
        if max_per_class is not None and len(pos_list) >= max_per_class and len(neg_list) >= max_per_class:
            break

    return {"pos": pos_list, "neg": neg_list}


def resolve_shot_support(
    shots_per_class: int,
    *,
    max_shots_per_class: int = 5,
    table_path: Path = TABLE_DIR,
    video_path: Path = VIDEO_DIR,
    label_excel: Path = LABEL_EXCEL,
    min_file_index: int = MIN_FILE_INDEX,
    binary_threshold: float = BINARY_THRESHOLD,
) -> Dict[str, List[int]]:
    """
    Return first K positive and first K negative paired subjects per class.

    Nested: 1-shot support is a subset of 3-shot, which is a subset of 5-shot.
    """
    if shots_per_class < 1:
        raise ValueError(f"shots_per_class must be >= 1, got {shots_per_class}")
    if shots_per_class > max_shots_per_class:
        raise ValueError(
            f"shots_per_class ({shots_per_class}) exceeds max_shots_per_class ({max_shots_per_class})"
        )

    pools = _collect_paired_pos_neg_lists(
        table_path,
        video_path,
        label_excel,
        min_file_index,
        binary_threshold,
        max_per_class=max_shots_per_class,
    )
    pos = pools["pos"][:shots_per_class]
    neg = pools["neg"][:shots_per_class]

    if len(pos) < shots_per_class or len(neg) < shots_per_class:
        raise ValueError(
            f"Insufficient paired subjects for {shots_per_class}-shot "
            f"(pos={len(pos)}/{shots_per_class}, neg={len(neg)}/{shots_per_class})"
        )
    return {"pos": pos, "neg": neg}


def resolve_fixed_holdout(
    max_shots_per_class: int = 5,
    table_path: Path = TABLE_DIR,
    video_path: Path = VIDEO_DIR,
    label_excel: Path = LABEL_EXCEL,
    min_file_index: int = MIN_FILE_INDEX,
    binary_threshold: float = BINARY_THRESHOLD,
) -> List[int]:
    """All paired subjects outside the max-shot support pool."""
    pools = resolve_shot_support(
        max_shots_per_class,
        max_shots_per_class=max_shots_per_class,
        table_path=table_path,
        video_path=video_path,
        label_excel=label_excel,
        min_file_index=min_file_index,
        binary_threshold=binary_threshold,
    )
    support_set = set(pools["pos"]) | set(pools["neg"])
    paired = discover_paired_indices(table_path, video_path)
    return [idx for idx in paired if idx not in support_set]


def flatten_shot_support(support: Dict[str, List[int]]) -> List[int]:
    """Flatten {'pos': [...], 'neg': [...]} into a single index list."""
    return list(support["pos"]) + list(support["neg"])


def resolve_support_indices(
    table_path: Path = TABLE_DIR,
    video_path: Path = VIDEO_DIR,
    label_excel: Path = LABEL_EXCEL,
    min_file_index: int = MIN_FILE_INDEX,
    binary_threshold: float = BINARY_THRESHOLD,
) -> Dict[str, int]:
    """
    Return first positive and first negative subject indices among paired clips.

    Falls back to {pos: 884, neg: 885} when they are present in paired indices.
    """
    shot = resolve_shot_support(
        1,
        table_path=table_path,
        video_path=video_path,
        label_excel=label_excel,
        min_file_index=min_file_index,
        binary_threshold=binary_threshold,
    )
    return {"pos": shot["pos"][0], "neg": shot["neg"][0]}


def resolve_holdout_indices(
    support: Dict[str, int] | None = None,
    table_path: Path = TABLE_DIR,
    video_path: Path = VIDEO_DIR,
    label_excel: Path = LABEL_EXCEL,
    min_file_index: int = MIN_FILE_INDEX,
    binary_threshold: float = BINARY_THRESHOLD,
) -> List[int]:
    """All paired indices except support subjects."""
    if support is None:
        support = resolve_support_indices(
            table_path, video_path, label_excel, min_file_index, binary_threshold
        )
    support_set = {int(support["pos"]), int(support["neg"])}
    paired = discover_paired_indices(table_path, video_path)
    return [idx for idx in paired if idx not in support_set]
