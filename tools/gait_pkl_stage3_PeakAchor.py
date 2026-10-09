"""
Preprocessing script (stage 3): peak-centered gait sampling → PKL.

Processes gait clips using StraightTurningV3-style alignment (peak shift + FRAME_APART KM + symmetric video).

Usage:
    1. Configure paths in main()
    2. Run: python gait_pkl_stage3_PeakAchor.py
    3. Pickle files are written to output_dir/patches/
    4. With train/test index JSON paths: filter subjects by split metadata
    5. With both index paths None: process every paired CSV/video in table_dir and video_dir
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
pytorch_dir = os.path.dirname(current_dir)
sys.path.insert(0, pytorch_dir)

from utils.data_sampler import SigLIPFullGaitDataset_v2
from utils.gait_sampling_v3 import process_clip_v3_style


def _label_value_comparable(val: Any) -> Optional[Tuple]:
    """Normalize label_value to a comparable form for consistency check."""
    if val is None:
        return None
    if isinstance(val, list):
        return tuple(sorted((x if isinstance(x, (int, float)) else float(x) for x in val)))
    if isinstance(val, (int, float)):
        return (float(val),)
    try:
        return (float(val),)
    except (ValueError, TypeError):
        return None


def check_train_split_label_consistency(
    patch_metadata: List[Dict],
    mode: str,
) -> Tuple[bool, List[str]]:
    """Verify that all patches from the same source file have the same label and label_value."""
    errors: List[str] = []
    if mode != "train" or not patch_metadata:
        return True, []

    from collections import defaultdict
    by_source: Dict[str, List[Dict]] = defaultdict(list)
    for p in patch_metadata:
        src = p.get("source_file") or p.get("subject_id") or "unknown"
        by_source[src].append(p)

    for source_file, patches in by_source.items():
        if len(patches) <= 1:
            continue
        first = patches[0]
        first_label = first.get("label")
        first_lv = _label_value_comparable(first.get("label_value"))
        for i, p in enumerate(patches[1:], start=1):
            if p.get("label") != first_label:
                errors.append(
                    f"Source '{source_file}': patch index {i} has label {p.get('label')} "
                    f"(expected {first_label})"
                )
            p_lv = _label_value_comparable(p.get("label_value"))
            if first_lv != p_lv:
                errors.append(
                    f"Source '{source_file}': patch index {i} has label_value {p.get('label_value')} "
                    f"(expected {first.get('label_value')})"
                )

    return len(errors) == 0, errors


def process_split_and_append(
    table_dir: str,
    video_dir: str,
    output_dir: str,
    label_json_path: Optional[str],
    split: str,
    mode: str,
    patch_id_counter: int,
    patch_metadata: List[Dict],
    patch_size: int = 96,
    video_frame_count: int = 32,
    video_target_size: Optional[Tuple[int, int]] = None,
    prompts_path: Optional[str] = None,
    prompt_selection: str = "concise_prompts",
    binary_threshold: float = 11.0,
    km_gaussian_noise_std: Optional[float] = None,
    chunk_size: int = 1000,
    high_fps_threshold: int = 300,
    from_start: bool = True,
) -> Tuple[int, List[Dict]]:
    """Process one split and append to patch_metadata."""
    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    if label_json_path is None:
        print(f"\n=== Processing {split.upper()} ({mode} mode): all paired files in folders ===\n")
    else:
        print(f"\n=== Processing {split.upper()} split ({mode} mode) from {label_json_path} ===\n")

    dataset = SigLIPFullGaitDataset_v2(
        table_dir=table_dir,
        video_dir=video_dir,
        label_json_path=label_json_path,
        split=split,
        directions=None,
        patch_size=patch_size,
        video_frame_count=video_frame_count,
        max_samples_per_file=None,
        video_target_size=video_target_size,
        prompts_path=prompts_path,
        prompt_selection=prompt_selection,
        binary_threshold=binary_threshold,
        km_gaussian_noise_std=None,
        mode=mode,
    )

    print(f"✅ Found {len(dataset)} samples to process for {split}")

    progress_bar = tqdm(
        range(len(dataset)),
        desc=f"Processing {split}",
        unit="sample",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for idx in progress_bar:
        sample_info = dataset.sample_indices[idx]
        csv_path = sample_info["csv_path"]
        video_path = sample_info["video_path"]
        start_idx = sample_info["start_idx"]
        end_idx = sample_info["end_idx"]

        progress_bar.set_postfix({
            "file": (
                os.path.basename(csv_path)[:30] + "..."
                if len(os.path.basename(csv_path)) > 30
                else os.path.basename(csv_path)
            ),
            "patches": patch_id_counter,
        })

        try:
            km_clip, video_clip, peak_idx, km_indices, video_indices, downsample_factor = (
                process_clip_v3_style(
                    csv_path=csv_path,
                    video_path=video_path,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    km_timesteps=patch_size,
                    video_timesteps=video_frame_count,
                    video_target_size=video_target_size,
                    high_fps_threshold=high_fps_threshold,
                    from_start=from_start,
                )
            )

            prompts: List[str] = []

            label_value = sample_info.get("label")
            label = None
            if label_value is not None:
                if isinstance(label_value, list):
                    max_val = max(label_value)
                    label = 1.0 if max_val >= binary_threshold else 0.0
                elif isinstance(label_value, (int, float)):
                    label = 1.0 if label_value >= binary_threshold else 0.0
                else:
                    try:
                        val = float(label_value)
                        label = 1.0 if val >= binary_threshold else 0.0
                    except (ValueError, TypeError):
                        label = 0.0

            patch_id = f"{patch_id_counter:08d}"
            patch_filename = f"patch_{patch_id}.pkl"
            patch_path = os.path.join(patches_dir, patch_filename)

            patch_data = {
                "knowledge_map": km_clip,
                "video": video_clip,
                "source_file": sample_info["source_file"],
                "prompts": prompts,
                "km_indices": km_indices,
                "video_indices": video_indices,
                "subject_id": sample_info.get("subject_id"),
                "direction": sample_info.get("direction"),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "peak_idx": peak_idx,
                "downsample_factor": downsample_factor,
                "sampling_method": "stage3_peak_centered",
                "label": label,
                "label_value": label_value,
                "binary_threshold": binary_threshold,
            }

            with open(patch_path, "wb") as f:
                pickle.dump(patch_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            patch_metadata.append({
                "patch_id": patch_id,
                "pkl_path": patch_path,
                "subject_id": sample_info.get("subject_id"),
                "direction": sample_info.get("direction"),
                "source_file": sample_info["source_file"],
                "start_idx": start_idx,
                "end_idx": end_idx,
                "knowledge_map_shape": km_clip.shape,
                "video_shape": video_clip.shape,
                "num_prompts": len(prompts),
                "has_label": label is not None,
                "label": label,
                "label_value": label_value,
                "binary_threshold": binary_threshold,
                "peak_idx": peak_idx,
                "downsample_factor": downsample_factor,
                "sampling_method": "stage3_peak_centered",
                "split": split,
            })

            patch_id_counter += 1

            if len(patch_metadata) % chunk_size == 0:
                progress_bar.write(
                    f"💾 Checkpoint: {patch_id_counter} total patches saved so far"
                )

        except Exception as e:
            progress_bar.write(
                f"❌ Error processing {split} sample {idx} "
                f"({sample_info.get('source_file', 'unknown')}): {e}"
            )
            continue

    progress_bar.close()

    if mode == "train" and patch_metadata:
        split_metadata = [p for p in patch_metadata if p.get("split") == split]
        if split_metadata:
            ok, errs = check_train_split_label_consistency(split_metadata, mode)
            if ok:
                print(
                    f"\n✅ {split.upper()} split label check: "
                    "all clips from the same source have the same label values."
                )
            else:
                print(f"\n⚠️ {split.upper()} split label check FAILED:")
                for e in errs[:20]:
                    print(f"   {e}")
                if len(errs) > 20:
                    print(f"   ... and {len(errs) - 20} more.")

    return patch_id_counter, patch_metadata


def process_all_indices(
    table_dir: str,
    video_dir: str,
    output_dir: str,
    train_indices_path: Optional[str] = None,
    test_indices_path: Optional[str] = None,
    patch_size: int = 96,
    video_frame_count: int = 32,
    video_target_size: Optional[Tuple[int, int]] = None,
    prompts_path: Optional[str] = None,
    prompt_selection: str = "concise_prompts",
    binary_threshold: float = 11.0,
    km_gaussian_noise_std: Optional[float] = None,
    chunk_size: int = 1000,
    high_fps_threshold: int = 300,
    from_start: bool = True,
) -> str:
    """Process clips with stage-3 peak-centered sampling (index JSON optional)."""
    print("=" * 60)
    print("Gait PKL Stage 3 — peak-centered sampling")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Table dir: {table_dir}")
    print(f"Video dir: {video_dir}")
    print(f"Train indices: {train_indices_path or '(none — use folder scan)'}")
    print(f"Test indices: {test_indices_path or '(none — use folder scan)'}")

    os.makedirs(output_dir, exist_ok=True)

    patch_metadata: List[Dict] = []
    patch_id_counter = 0
    train_count = 0
    test_count = 0
    all_count = 0

    if train_indices_path is None and test_indices_path is None:
        patch_id_counter, patch_metadata = process_split_and_append(
            table_dir=table_dir,
            video_dir=video_dir,
            output_dir=output_dir,
            label_json_path=None,
            split="all",
            mode="test",
            patch_id_counter=patch_id_counter,
            patch_metadata=patch_metadata,
            patch_size=patch_size,
            video_frame_count=video_frame_count,
            video_target_size=video_target_size,
            prompts_path=prompts_path,
            prompt_selection=prompt_selection,
            binary_threshold=binary_threshold,
            km_gaussian_noise_std=km_gaussian_noise_std,
            chunk_size=chunk_size,
            high_fps_threshold=high_fps_threshold,
            from_start=from_start,
        )
        all_count = len(patch_metadata)
    else:
        if train_indices_path is not None:
            patch_id_counter, patch_metadata = process_split_and_append(
                table_dir=table_dir,
                video_dir=video_dir,
                output_dir=output_dir,
                label_json_path=train_indices_path,
                split="train",
                mode="test",
                patch_id_counter=patch_id_counter,
                patch_metadata=patch_metadata,
                patch_size=patch_size,
                video_frame_count=video_frame_count,
                video_target_size=video_target_size,
                prompts_path=prompts_path,
                prompt_selection=prompt_selection,
                binary_threshold=binary_threshold,
                km_gaussian_noise_std=km_gaussian_noise_std,
                chunk_size=chunk_size,
                high_fps_threshold=high_fps_threshold,
                from_start=from_start,
            )
            train_count = len(patch_metadata)

        if test_indices_path is not None:
            patch_id_counter, patch_metadata = process_split_and_append(
                table_dir=table_dir,
                video_dir=video_dir,
                output_dir=output_dir,
                label_json_path=test_indices_path,
                split="test",
                mode="test",
                patch_id_counter=patch_id_counter,
                patch_metadata=patch_metadata,
                patch_size=patch_size,
                video_frame_count=video_frame_count,
                video_target_size=video_target_size,
                prompts_path=prompts_path,
                prompt_selection=prompt_selection,
                binary_threshold=binary_threshold,
                km_gaussian_noise_std=km_gaussian_noise_std,
                chunk_size=chunk_size,
                high_fps_threshold=high_fps_threshold,
                from_start=from_start,
            )
            test_count = len(patch_metadata) - train_count

    print("\n💾 Saving combined metadata...")
    metadata_path = os.path.join(output_dir, "patch_metadata.pkl")
    config = {
        "patch_size": patch_size,
        "video_frame_count": video_frame_count,
        "video_target_size": video_target_size,
        "mode": (
            "all_folder"
            if train_indices_path is None and test_indices_path is None
            else "train+test"
        ),
        "sampling_method": "stage3_peak_centered",
        "high_fps_threshold": high_fps_threshold,
        "binary_threshold": binary_threshold,
        "km_gaussian_noise_std": km_gaussian_noise_std,
        "prompt_selection": prompt_selection,
        "created_at": datetime.now().isoformat(),
        "table_dir": table_dir,
        "video_dir": video_dir,
        "train_indices_path": train_indices_path,
        "test_indices_path": test_indices_path,
        "prompts_path": prompts_path,
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(
            {"patch_metadata": patch_metadata, "config": config},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    patches_dir = os.path.join(output_dir, "patches")
    print("\n✅ Processing complete!")
    print(f"  📦 Total patches: {len(patch_metadata)}")
    if all_count:
        print(f"  📂 All-folder patches: {all_count}")
    else:
        print(f"  📂 Train patches: {train_count}")
        print(f"  📂 Test patches: {test_count}")
    print(f"  📁 Patches directory: {patches_dir}")
    print(f"  📄 Metadata file: {metadata_path}")

    if patch_metadata:
        labels = [p.get("label") for p in patch_metadata if p.get("has_label")]
        if labels:
            label_counts = {0.0: labels.count(0.0), 1.0: labels.count(1.0)}
            print(f"  📊 Label distribution (all): {label_counts}")
        train_labels = [
            p.get("label")
            for p in patch_metadata
            if p.get("has_label") and p.get("split") == "train"
        ]
        test_labels = [
            p.get("label")
            for p in patch_metadata
            if p.get("has_label") and p.get("split") == "test"
        ]
        if train_labels:
            tc = {0.0: train_labels.count(0.0), 1.0: train_labels.count(1.0)}
            print(f"  📊 Train label distribution: {tc}")
        if test_labels:
            tc = {0.0: test_labels.count(0.0), 1.0: test_labels.count(1.0)}
            print(f"  📊 Test label distribution: {tc}")
        print(f"  📏 Knowledge map shape: {patch_metadata[0].get('knowledge_map_shape', 'unknown')}")
        print(f"  🎥 Video shape: {patch_metadata[0].get('video_shape', 'unknown')}")

    return metadata_path


def main():
    project_root = Path(__file__).resolve().parent.parent
    config = {
        "table_dir": str(project_root / "data" / "pk_testpart_table"),
        "video_dir": str(project_root / "data" / "pk_testpart_video"),
        "train_indices_path": str(project_root / "dataset" / "train_indices.json"),
        "test_indices_path":  str(project_root / "dataset" / "test_indices.json"),
        "patch_size": 96,
        "video_frame_count": 32,
        "video_target_size": (224, 224),
        "prompts_path": str(project_root / "dataset" / "sz_general_gait_prompts.json"),
        "prompt_selection": "concise_prompts",
        "binary_threshold": 11.0,
        "km_gaussian_noise_std": 0.0,
        "output_dir": str(project_root / "data" / "pk_pkl_stage3"),
        "high_fps_threshold": 300,
        "from_start": True,
    }

    metadata_path = process_all_indices(
        table_dir=config["table_dir"],
        video_dir=config["video_dir"],
        output_dir=config["output_dir"],
        train_indices_path=config["train_indices_path"],
        test_indices_path=config["test_indices_path"],
        patch_size=config["patch_size"],
        video_frame_count=config["video_frame_count"],
        video_target_size=config["video_target_size"],
        prompts_path=config.get("prompts_path"),
        prompt_selection=config.get("prompt_selection", "concise_prompts"),
        binary_threshold=config.get("binary_threshold", 11.0),
        km_gaussian_noise_std=config.get("km_gaussian_noise_std"),
        high_fps_threshold=config.get("high_fps_threshold", 300),
        from_start=config.get("from_start", True),
    )

    print(f"\nPreprocessing complete! Metadata saved to: {metadata_path}")
    if config["train_indices_path"] is None and config["test_indices_path"] is None:
        print("All paired files processed; metadata entries have split='all'")
    else:
        print("Filter by split: metadata['patch_metadata'] entries have 'split' = 'train' or 'test'")


if __name__ == "__main__":
    main()
