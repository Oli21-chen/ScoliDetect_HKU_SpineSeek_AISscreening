# -*- coding: utf-8 -*-
"""
# DATA PREPARATION

Step 2 — peak-centered gait sampling (StraightTurningV3 / gait_sampling_v3).

Batch-processes paired pose CSV + trimmed video clips and writes per-sample PKL
files compatible with ScoliDetect predict_video / predict_from_arrays.

Default sampling (matches deploy inference):
  - KM: 96 timesteps
  - Video: 32 frames at 224x224 RGB

This script is used to sample the gait data from the CSV and video files.

@author: Olive
"""

from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

current_dir = os.path.dirname(os.path.abspath(__file__))
pytorch_dir = os.path.join(os.path.dirname(current_dir), "pytorch")
sys.path.insert(0, pytorch_dir)

from utils.gait_sampling_v3 import process_clip_v3_style  # noqa: E402


def process_one_clip(
    csv_path: str,
    video_path: str,
    *,
    km_timesteps: int = 96,
    video_timesteps: int = 32,
    video_target_size: Tuple[int, int] = (224, 224),
    high_fps_threshold: int = 300,
    from_start: bool = True,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, int]:
    """Run process_clip_v3_style on one CSV/video pair."""
    return process_clip_v3_style(
        csv_path=csv_path,
        video_path=video_path,
        start_idx=start_idx,
        end_idx=end_idx,
        km_timesteps=km_timesteps,
        video_timesteps=video_timesteps,
        video_target_size=video_target_size,
        high_fps_threshold=high_fps_threshold,
        from_start=from_start,
    )


def compute_binary_label(
    label_value: Any,
    binary_threshold: float = 11.0,
) -> Optional[float]:
    """Derive binary label from Cobb angles or scalar label_value."""
    if label_value is None:
        return None
    if isinstance(label_value, (list, tuple, np.ndarray)):
        max_val = max(label_value)
        return 1.0 if max_val >= binary_threshold else 0.0
    if isinstance(label_value, (int, float)):
        return 1.0 if label_value >= binary_threshold else 0.0
    try:
        val = float(label_value)
        return 1.0 if val >= binary_threshold else 0.0
    except (ValueError, TypeError):
        return 0.0


def save_patch_pkl(
    output_dir: str,
    patch_id_counter: int,
    km_clip: np.ndarray,
    video_clip: np.ndarray,
    peak_idx: int,
    km_indices: np.ndarray,
    video_indices: np.ndarray,
    downsample_factor: int,
    source_file: str,
    subject_id: Union[int, str],
    *,
    label_value: Any = None,
    fixed_label: Optional[float] = None,
    binary_threshold: float = 11.0,
    split: str = "all",
) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    """Write one patch PKL and return updated counter plus metadata entries."""
    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    if fixed_label is not None:
        label = fixed_label
    else:
        label = compute_binary_label(label_value, binary_threshold)
    patch_id = f"{patch_id_counter:08d}"
    patch_path = os.path.join(patches_dir, f"patch_{patch_id}.pkl")

    patch_data = {
        "knowledge_map": km_clip,
        "video": video_clip,
        "source_file": source_file,
        "prompts": [],
        "km_indices": km_indices,
        "video_indices": video_indices,
        "subject_id": subject_id,
        "direction": None,
        "start_idx": 0,
        "end_idx": None,
        "peak_idx": peak_idx,
        "downsample_factor": downsample_factor,
        "sampling_method": "stage3_peak_centered",
        "label": label,
        "label_value": label_value,
        "binary_threshold": binary_threshold,
    }

    with open(patch_path, "wb") as f:
        pickle.dump(patch_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta_entry = {
        "patch_id": patch_id,
        "pkl_path": patch_path,
        "subject_id": subject_id,
        "direction": None,
        "source_file": source_file,
        "start_idx": 0,
        "end_idx": None,
        "knowledge_map_shape": km_clip.shape,
        "video_shape": video_clip.shape,
        "num_prompts": 0,
        "has_label": label is not None,
        "label": label,
        "label_value": label_value,
        "binary_threshold": binary_threshold,
        "peak_idx": peak_idx,
        "downsample_factor": downsample_factor,
        "sampling_method": "stage3_peak_centered",
        "split": split,
    }

    return patch_id_counter + 1, patch_data, meta_entry


def save_patch_metadata(
    output_dir: str,
    patch_metadata: List[Dict[str, Any]],
    *,
    table_path: str,
    video_path: str,
    km_timesteps: int,
    video_timesteps: int,
    video_target_size: Tuple[int, int],
    binary_threshold: float,
    high_fps_threshold: int,
    from_start: bool,
    para_name: str,
) -> str:
    """Persist combined patch_metadata.pkl with config block."""
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "patch_metadata.pkl")
    config = {
        "patch_size": km_timesteps,
        "video_frame_count": video_timesteps,
        "video_target_size": video_target_size,
        "mode": "batch_folder",
        "sampling_method": "stage3_peak_centered",
        "high_fps_threshold": high_fps_threshold,
        "binary_threshold": binary_threshold,
        "from_start": from_start,
        "para_name": para_name,
        "created_at": datetime.now().isoformat(),
        "table_dir": table_path,
        "video_dir": video_path,
    }
    with open(metadata_path, "wb") as f:
        pickle.dump({"patch_metadata": patch_metadata, "config": config}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return metadata_path


def _print_summary(patch_metadata: List[Dict[str, Any]], metadata_path: str) -> None:
    print(f"\nSaved {len(patch_metadata)} patches")
    print(f"Metadata: {metadata_path}")
    if patch_metadata:
        print(f"  knowledge_map shape: {patch_metadata[0]['knowledge_map_shape']}")
        print(f"  video shape: {patch_metadata[0]['video_shape']}")
        labels = [p["label"] for p in patch_metadata if p.get("has_label")]
        if labels:
            print(f"  label distribution: {{0.0: {labels.count(0.0)}, 1.0: {labels.count(1.0)}}}")


def discover_paired_indices(
    table_path: str,
    video_path: str,
    para_name: str = "pk",
    csv_suffix: str = "_step_1.csv",
) -> List[int]:
    """Return sorted indices with both pose CSV and trimmed video present."""
    prefix = f"{para_name}_"
    indices: List[int] = []

    if not os.path.isdir(table_path):
        return indices

    for fname in os.listdir(table_path):
        if not fname.startswith(prefix) or not fname.endswith(csv_suffix):
            continue
        mid = fname[len(prefix) : -len(csv_suffix)]
        try:
            index = int(mid)
        except ValueError:
            continue
        table_name = os.path.join(table_path, fname)
        video_name = os.path.join(video_path, f"{para_name}_{index}_step1.mp4")
        if os.path.isfile(table_name) and os.path.isfile(video_name):
            indices.append(index)

    return sorted(indices)


def run_batch(
    para_name: str,
    table_path: str,
    video_path: str,
    output_dir: str,
    indices: List[Union[int, float]],
    labels: Optional[List[Any]] = None,
    *,
    fixed_label: Optional[float] = None,
    km_timesteps: int = 96,
    video_timesteps: int = 32,
    video_target_size: Tuple[int, int] = (224, 224),
    high_fps_threshold: int = 300,
    from_start: bool = True,
    binary_threshold: float = 11.0,
    split: str = "all",
    csv_suffix: str = "_step_1.csv",
) -> str:
    """Process paired CSV/video files for each index and write PKL patches."""
    patch_metadata: List[Dict[str, Any]] = []
    patch_id_counter = 0

    for i, index in enumerate(indices):
        label_value = None if fixed_label is not None else (labels[i] if labels is not None else None)
        print(f"para_name, index: {para_name}, {index}")

        video_name = os.path.join(video_path, f"{para_name}_{int(index)}_step1.mp4")
        table_name = os.path.join(table_path, f"{para_name}_{int(index)}{csv_suffix}")
        if not os.path.exists(video_name) or not os.path.exists(table_name):
            continue

        source_file = f"{para_name}_{int(index)}_step1"
        try:
            km_clip, video_clip, peak_idx, km_indices, video_indices, downsample_factor = (
                process_one_clip(
                    table_name,
                    video_name,
                    km_timesteps=km_timesteps,
                    video_timesteps=video_timesteps,
                    video_target_size=video_target_size,
                    high_fps_threshold=high_fps_threshold,
                    from_start=from_start,
                )
            )
        except Exception as exc:
            print(f"  skipped {source_file}: {exc}")
            continue

        patch_id_counter, _, meta_entry = save_patch_pkl(
            output_dir,
            patch_id_counter,
            km_clip,
            video_clip,
            peak_idx,
            km_indices,
            video_indices,
            downsample_factor,
            source_file,
            int(index),
            label_value=label_value,
            fixed_label=fixed_label,
            binary_threshold=binary_threshold,
            split=split,
        )
        patch_metadata.append(meta_entry)
        print(f"  saved patch {meta_entry['patch_id']}: km={km_clip.shape}, video={video_clip.shape}")

    metadata_path = save_patch_metadata(
        output_dir,
        patch_metadata,
        table_path=table_path,
        video_path=video_path,
        km_timesteps=km_timesteps,
        video_timesteps=video_timesteps,
        video_target_size=video_target_size,
        binary_threshold=binary_threshold,
        high_fps_threshold=high_fps_threshold,
        from_start=from_start,
        para_name=para_name,
    )
    _print_summary(patch_metadata, metadata_path)
    return metadata_path


def run_sz(
    table_path: str,
    video_path: str,
    output_dir: str,
    *,
    km_timesteps: int = 96,
    video_timesteps: int = 32,
    video_target_size: Tuple[int, int] = (224, 224),
    high_fps_threshold: int = 300,
    from_start: bool = True,
    binary_threshold: float = 11.0,
    min_file_index: int = 1,
    label_excel: str = r"C:\Users\Olive\Desktop\video_retrival\video_retrival\Label_SZpart2.xlsx",
    csv_suffix: str = "_step_1.csv",
    indices: Optional[List[int]] = None,
) -> str:
    dictionary = pd.read_excel(
        label_excel,
        usecols=["File No.", "age", "sex", "M_cobb_l", "M_cobb_r"],
    ).dropna()
    dictionary_val = dictionary.values
    starting_index = np.where(dictionary_val[:, 0] == min_file_index)[0][0]
    print("starting_index", starting_index)
    sub_label = dictionary_val[starting_index:, 3:5]
    sub_index = dictionary_val[starting_index:, 0]
    if indices is not None:
        index_set = {int(i) for i in indices}
        mask = [int(i) in index_set for i in sub_index]
        sub_index = sub_index[mask]
        sub_label = sub_label[mask]
    labels = [row.tolist() for row in sub_label]
    return run_batch(
        "sz",
        table_path,
        video_path,
        output_dir,
        sub_index.tolist(),
        labels,
        km_timesteps=km_timesteps,
        video_timesteps=video_timesteps,
        video_target_size=video_target_size,
        high_fps_threshold=high_fps_threshold,
        from_start=from_start,
        binary_threshold=binary_threshold,
        csv_suffix=csv_suffix,
    )

def run_unlabeled(
    table_path: str,
    video_path: str,
    output_dir: str,
    *,
    para_name: str = "pk",
    fixed_label: float = 0.0,
    km_timesteps: int = 96,
    video_timesteps: int = 32,
    video_target_size: Tuple[int, int] = (224, 224),
    high_fps_threshold: int = 300,
    from_start: bool = True,
    binary_threshold: float = 11.0,
    csv_suffix: str = "_step_1.csv",
) -> str:
    """Sample paired clips without a label Excel; assign fixed_label to every patch."""
    indices = discover_paired_indices(table_path, video_path, para_name, csv_suffix=csv_suffix)
    print(f"Discovered {len(indices)} paired {para_name} clips (label={fixed_label})")
    return run_batch(
        para_name,
        table_path,
        video_path,
        output_dir,
        indices,
        labels=None,
        fixed_label=fixed_label,
        km_timesteps=km_timesteps,
        video_timesteps=video_timesteps,
        video_target_size=video_target_size,
        high_fps_threshold=high_fps_threshold,
        from_start=from_start,
        binary_threshold=binary_threshold,
        csv_suffix=csv_suffix,
    )




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 02 gait sampling")
    parser.add_argument("--table-path", default=r"C:\Users\Olive\Desktop\table")
    parser.add_argument("--video-path", default=r"C:\Users\Olive\Desktop\video")
    parser.add_argument("--output-dir", default=r"C:\Users\Olive\Desktop\infer_data_11")
    parser.add_argument("--csv-suffix", default="_step_1.csv")
    parser.add_argument("--min-file-index", type=int, default=884)
    parser.add_argument("--km-timesteps", type=int, default=96)
    parser.add_argument("--video-timesteps", type=int, default=32)
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional subject indices subset (pilot runs)",
    )
    args = parser.parse_args()

    run_sz(
        table_path=args.table_path,
        video_path=args.video_path,
        output_dir=args.output_dir,
        km_timesteps=args.km_timesteps,
        video_timesteps=args.video_timesteps,
        video_target_size=(224, 224),
        min_file_index=args.min_file_index,
        csv_suffix=args.csv_suffix,
        indices=args.indices,
    )
