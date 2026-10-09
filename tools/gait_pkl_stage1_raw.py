"""
Preprocessing script v2: Process ALL indices from train_indices.json and test_indices.json.
Saves all pkl files in ONE folder with combined metadata.

Usage:
    1. Configure the config dictionary in main() function with your paths
    2. Run: python gaitcycle_to_pkl_v2.py
    3. All pickle files will be created in the output directory (patches subfolder)
    4. Metadata includes a "split" field ("train" or "test") for each sample

Example:
    After preprocessing, filter by split when loading:
    
    metadata = pickle.load(open('patch_metadata.pkl', 'rb'))
    train_entries = [e for e in metadata['patch_metadata'] if e.get('split') == 'train']
    test_entries = [e for e in metadata['patch_metadata'] if e.get('split') == 'test']
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
pytorch_dir = os.path.dirname(current_dir)
sys.path.insert(0, pytorch_dir)

from utils.data_sampler import (
    SigLIPFullGaitDataset_v2,
    load_knowledge_map,
    load_video_frames_range,
)


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
    label_json_path: str,
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
) -> Tuple[int, List[Dict]]:
    """
    Process one split (train or test) and append to existing patch_metadata.
    Returns (new_patch_id_counter, updated_patch_metadata).
    """
    patches_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)

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
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for idx in progress_bar:
        sample_info = dataset.sample_indices[idx]
        csv_path = sample_info["csv_path"]
        video_path = sample_info["video_path"]
        start_idx = sample_info["start_idx"]
        end_idx = sample_info["end_idx"]

        progress_bar.set_postfix({
            'file': os.path.basename(csv_path)[:30] + '...' if len(os.path.basename(csv_path)) > 30 else os.path.basename(csv_path),
            'patches': patch_id_counter
        })

        try:
            km_data = load_knowledge_map(csv_path)
            video_clip = load_video_frames_range(
                video_path,
                start_idx,
                end_idx,
                target_size = video_target_size,
            )

            km_clip = km_data[start_idx:end_idx]
            min_len = min(len(km_clip), len(video_clip))
            original_length = min_len

            km_target_timesteps = patch_size
            video_target_timesteps = video_frame_count

            if original_length > km_target_timesteps:
                km_sampled_idx = np.linspace(0, original_length - 1, km_target_timesteps, dtype=int)
                km_clip = np.take(km_clip, km_sampled_idx, axis=0)
            elif original_length < km_target_timesteps:
                pad_len = km_target_timesteps - original_length
                km_clip = np.pad(km_clip[:original_length], ((0, pad_len), (0, 0)), mode='constant')

            if original_length > video_target_timesteps:
                video_sampled_idx = np.linspace(0, original_length - 1, video_target_timesteps, dtype=int)
                video_clip = np.take(video_clip, video_sampled_idx, axis=0)
            elif original_length < video_target_timesteps:
                pad_len = video_target_timesteps - original_length
                video_clip = np.pad(
                    video_clip[:original_length],
                    ((0, pad_len), (0, 0), (0, 0), (0, 0)),
                    mode='constant'
                )
            _OUT_PATH = os.path.join(r"C:\Users\Olive\Desktop\Nature_Style\code_video", "temp.png")
            import matplotlib.pyplot as plt
            a = np.asarray(km_clip)
            curve = np.sum(a, axis=1)
            plt.figure()
            plt.plot(curve)
            plt.xlabel("Feature index")
            plt.ylabel("Sum over time")
            plt.savefig(_OUT_PATH, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved: {_OUT_PATH}")


            prompts: List[str] = []

            km_indices = np.arange(0, km_target_timesteps, dtype=int)
            video_indices = np.linspace(
                0, km_target_timesteps - 1, video_target_timesteps, dtype=int
            )

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
                'knowledge_map': km_clip,
                'video': video_clip,
                'source_file': sample_info["source_file"],
                'prompts': prompts,
                'km_indices': km_indices,
                'video_indices': video_indices,
                'subject_id': sample_info.get("subject_id"),
                'direction': sample_info.get("direction"),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'label': label,
                'label_value': label_value,
                'binary_threshold': binary_threshold,
            }

            with open(patch_path, 'wb') as f:
                pickle.dump(patch_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            patch_metadata.append({
                'patch_id': patch_id,
                'pkl_path': patch_path,
                'subject_id': sample_info.get("subject_id"),
                'direction': sample_info.get("direction"),
                'source_file': sample_info["source_file"],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'knowledge_map_shape': km_clip.shape,
                'video_shape': video_clip.shape,
                'num_prompts': len(prompts),
                'has_label': label is not None,
                'label': label,
                'label_value': label_value,
                'binary_threshold': binary_threshold,
                'split': split,
            })

            patch_id_counter += 1

            if (len(patch_metadata)) % chunk_size == 0:
                progress_bar.write(
                    f"💾 Checkpoint: {patch_id_counter} total patches saved so far"
                )

        except Exception as e:
            progress_bar.write(
                f"❌ Error processing {split} sample {idx} ({sample_info.get('source_file', 'unknown')}): {e}"
            )
            continue

    progress_bar.close()

    if mode == "train" and patch_metadata:
        split_metadata = [p for p in patch_metadata if p.get('split') == split]
        if split_metadata:
            ok, errs = check_train_split_label_consistency(split_metadata, mode)
            if ok:
                print(f"\n✅ {split.upper()} split label check: all clips from the same source have the same label values.")
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
    train_indices_path: str,
    test_indices_path: str,
    patch_size: int = 96,
    video_frame_count: int = 32,
    video_target_size: Optional[Tuple[int, int]] = None,
    prompts_path: Optional[str] = None,
    prompt_selection: str = "concise_prompts",
    binary_threshold: float = 11.0,
    km_gaussian_noise_std: Optional[float] = None,
    chunk_size: int = 1000,
) -> str:
    """
    Process ALL indices from train_indices.json and test_indices.json.
    Saves all pkl files in ONE folder (output_dir/patches/).
    
    Train indices: processed in train mode (chunked into 96-frame patches)
    Test indices: processed in test mode (one sample per video, full length)
    
    Each metadata entry includes a "split" field ("train" or "test").
    
    Returns:
        Path to the combined metadata file
    """
    print("=" * 60)
    print("GaitCycle to PKL v2 - Process ALL indices (train + test)")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Train indices: {train_indices_path}")
    print(f"Test indices: {test_indices_path}")

    os.makedirs(output_dir, exist_ok=True)

    patch_metadata: List[Dict] = []
    patch_id_counter = 0

    # Process train split first
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
    )

    train_count = len(patch_metadata)

    # Process test split (append to same folder)
    patch_id_counter, patch_metadata = process_split_and_append(
        table_dir=table_dir,
        video_dir=video_dir,
        output_dir=output_dir,
        label_json_path=test_indices_path,
        split="test",
        #test mode → 1 PKL patch (whole walk, then V3 peak shift + resample to 96/32)
        #train mode → 3 PKL patches (three separate 96-frame segments, each processed the same way)
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
    )

    test_count = len(patch_metadata) - train_count

    # Save combined metadata
    print("\n💾 Saving combined metadata...")
    metadata_path = os.path.join(output_dir, 'patch_metadata.pkl')
    config = {
        'patch_size': patch_size,
        'video_frame_count': video_frame_count,
        'video_target_size': video_target_size,
        'mode': 'train+test',
        'binary_threshold': binary_threshold,
        'km_gaussian_noise_std': km_gaussian_noise_std,
        'prompt_selection': prompt_selection,
        'created_at': datetime.now().isoformat(),
        'table_dir': table_dir,
        'video_dir': video_dir,
        'train_indices_path': train_indices_path,
        'test_indices_path': test_indices_path,
        'prompts_path': prompts_path,
    }

    with open(metadata_path, 'wb') as f:
        pickle.dump({'patch_metadata': patch_metadata, 'config': config}, f, protocol=pickle.HIGHEST_PROTOCOL)

    patches_dir = os.path.join(output_dir, 'patches')
    print(f"\n✅ Processing complete!")
    print(f"  📦 Total patches: {len(patch_metadata)}")
    print(f"  📂 Train patches: {train_count}")
    print(f"  📂 Test patches: {test_count}")
    print(f"  📁 Patches directory: {patches_dir}")
    print(f"  📄 Metadata file: {metadata_path}")

    if patch_metadata:
        labels = [p.get('label') for p in patch_metadata if p.get('has_label')]
        if labels:
            label_counts = {0.0: labels.count(0.0), 1.0: labels.count(1.0)}
            print(f"  📊 Label distribution (all): {label_counts}")
        train_labels = [p.get('label') for p in patch_metadata if p.get('has_label') and p.get('split') == 'train']
        test_labels = [p.get('label') for p in patch_metadata if p.get('has_label') and p.get('split') == 'test']
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
    """Main function with configuration."""
    project_root = Path(__file__).resolve().parent.parent
    config = {
        "table_dir": str(project_root / "data" / "sz_table_refinedyolo"),
        "video_dir": str(project_root / "data" / "sz_video_refinedyolo"),
        "train_indices_path": str(project_root / "dataset"  / "train_indices.json"),
        "test_indices_path": str(project_root / "dataset"  / "test_indices.json"),
        "patch_size": 96,
        "video_frame_count": 32,
        "video_target_size": (224, 224),
        "prompts_path": str(project_root / "dataset" / "sz_general_gait_prompts.json"),
        "prompt_selection": "concise_prompts",
        "binary_threshold": 11.0,
        "km_gaussian_noise_std": 0.0,
        "output_dir": str(project_root / "data" / "sz_pkl_raw"),
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
    )

    print(f"\nPreprocessing complete! Metadata saved to: {metadata_path}")
    print(f"Filter by split: metadata['patch_metadata'] entries have 'split' = 'train' or 'test'")


if __name__ == "__main__":
    main()
