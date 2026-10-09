"""
Preprocessing script to convert dataset to pickle format for faster loading.
Processes data using the same logic as data_sampler.py but saves as pkl files.

Usage:
    1. Configure the config dictionary in main() function with your paths
    2. Run: python preprocess_data_to_pkl.py
    3. The script will create pickle files in the output directory
    4. Use SigLIPFullGaitDatasetPKL to load the preprocessed data

Example:
    After preprocessing, you can use the dataset like this:
    
    from utils.data_sampler import SigLIPFullGaitDatasetPKL, fullgait_collate_fn
    from torch.utils.data import DataLoader
    
    dataset = SigLIPFullGaitDatasetPKL(
        pkl_data_dir="./data/preprocessed_pkl",
        prompts_path="./data/sz_general_gait_prompts.json",  # Optional: prompts loaded by dataset class
        prompt_selection="concise_prompts"  # Optional: can change without reprocessing
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=fullgait_collate_fn)
    
Benefits:
    - Much faster data loading (no on-the-fly processing)
    - Can use more DataLoader workers
    - Consistent preprocessing across runs
    - Reduced memory usage during training (data already processed)
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
pytorch_dir = os.path.dirname(current_dir)  # pytorch directory
sys.path.insert(0, pytorch_dir)

from utils.data_sampler import (
    SigLIPFullGaitDataset_v2,
    SigLIPPretrainDataset,
    load_knowledge_map,
    load_video_frames_range,
    extract_subject_id_from_filename,
    augment_knowledge_map_gaussian_noise,
    pad_to_multiple,
    create_non_overlapping_patches,
    sample_frames_evenly,
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
    """
    When mode is 'train', verify that all patches from the same source file
    have the same label and label_value (i.e. the 3 clips from one video share one label).
    Returns (all_ok, list of error messages).
    """
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


def process_and_save_fullgait_v2(
    table_dir: str,
    video_dir: str,
    output_dir: str,
    label_json_path: Optional[str] = None,
    split: Optional[str] = None,
    patch_size: int = 96,
    video_frame_count: int = 32,
    video_target_size: Optional[Tuple[int, int]] = None,
    prompts_path: Optional[str] = None,
    prompt_selection: str = "concise_prompts",
    binary_threshold: float = 11.0,
    km_gaussian_noise_std: Optional[float] = None,
    mode: str = "train",
    num_workers: int = 1,
    chunk_size: int = 1000,
) -> str:
    """
    Process and save SigLIPFullGaitDataset_v2 data to pickle files.
    
    Args:
        table_dir: Directory containing knowledge map CSV files
        video_dir: Directory containing video files
        output_dir: Directory to save pickle files
        label_json_path: Path to label JSON file (optional)
        split: Split name ('train' or 'test')
        patch_size: Knowledge map patch size (default 96)
        video_frame_count: Number of video frames (default 32)
        video_target_size: Target size for video frames (width, height)
        prompts_path: Path to prompts JSON file (optional)
        prompt_selection: Prompt selection method
        binary_threshold: Threshold for binary classification
        km_gaussian_noise_std: Standard deviation for Gaussian noise augmentation (None to disable)
        mode: 'train' or 'test' mode
        num_workers: Number of parallel workers (currently not used, processes sequentially)
        chunk_size: Number of samples to process before saving intermediate metadata
    
    Returns:
        Path to the metadata file
    """
    print(f"=== Processing data in {mode} mode... ===\n")
    print(f"Output directory: {output_dir}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    patches_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)
    
    # Create dataset to get sample indices (but we'll process manually)
    print("📋 Creating dataset to get sample indices...")
    with tqdm(total=1, desc="Initializing dataset", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
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
            km_gaussian_noise_std=None,  # We'll apply augmentation during preprocessing if needed
            mode=mode,
        )
        pbar.update(1)
    
    print(f"✅ Found {len(dataset)} samples to process")
    
    # Note: Prompts will be loaded and processed by the dataset class during training
    # We only store prompts_path in metadata for reference
    
    # Process samples
    patch_metadata = []
    patch_id_counter = 0
    
    print("Processing samples...")
    # Create progress bar with more details
    progress_bar = tqdm(
        range(len(dataset)), 
        desc="Processing samples",
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
        
        # Update progress bar description with current file
        progress_bar.set_postfix({
            'file': os.path.basename(csv_path)[:30] + '...' if len(os.path.basename(csv_path)) > 30 else os.path.basename(csv_path),
            'patches': patch_id_counter
        })
        
        try:
            # Load knowledge map and video for the specific range
            km_data = load_knowledge_map(csv_path)
            video_clip = load_video_frames_range(
                video_path,
                start_idx,
                end_idx,
                target_size=video_target_size,
            )
            
            # Get the clip for the specified range
            km_clip = km_data[start_idx:end_idx]
        
            
            # Ensure same length
            min_len = min(len(km_clip), len(video_clip))
            original_length = min_len

            # Target timesteps
            km_target_timesteps = patch_size  # 96
            video_target_timesteps = video_frame_count  # 32
            
            # Sample knowledge map to exactly 96 timesteps (evenly spaced)
            # Use np.pad for more efficient padding (consistent with optimized dataset class)
            if original_length > km_target_timesteps:
                km_sampled_idx = np.linspace(0, original_length - 1, km_target_timesteps, dtype=int)
                km_clip = np.take(km_clip, km_sampled_idx, axis=0)
            elif original_length < km_target_timesteps:
                # Pad if shorter - use np.pad instead of concatenate (more efficient)
                pad_len = km_target_timesteps - original_length
                km_clip = np.pad(km_clip[:original_length], ((0, pad_len), (0, 0)), mode='constant')
            
            # Sample video to exactly 32 timesteps (evenly spaced)
            # Use np.pad for more efficient padding (consistent with optimized dataset class)
            if original_length > video_target_timesteps:
                video_sampled_idx = np.linspace(0, original_length - 1, video_target_timesteps, dtype=int)
                video_clip = np.take(video_clip, video_sampled_idx, axis=0)
            elif original_length < video_target_timesteps:
                # Pad if shorter - use np.pad instead of concatenate (more efficient)
                pad_len = video_target_timesteps - original_length
                video_clip = np.pad(
                    video_clip[:original_length],
                    ((0, pad_len), (0, 0), (0, 0), (0, 0)),
                    mode='constant'
                )
            
            
            # Note: Prompts are NOT processed here - they will be loaded by the dataset class
            # This allows flexibility to change prompts without reprocessing data
            prompts: List[str] = []  # Empty - will be populated by dataset class
            
            
            # Indices: keep same meaning as SigLIPFullGaitDataset_v2.
            # Use precomputed global indices when available, otherwise derive from this window.
            # km_indices = sample_info.get("km_indices")
            # video_indices = sample_info.get("video_indices")
            
            # make sure each clip has the same indices
            
            km_indices = np.arange(
                0,
                km_target_timesteps,
                dtype=int,
            )
            video_indices =  np.linspace(
                0, 
                km_target_timesteps-1, 
                video_target_timesteps, 
                dtype=int
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
            
            # Save patch data
            patch_id = f"{patch_id_counter:08d}"
            patch_filename = f"patch_{patch_id}.pkl"
            patch_path = os.path.join(patches_dir, patch_filename)
            
            

            patch_data = {
                'knowledge_map': km_clip,  # numpy array, shape (96, km_features)
                'video': video_clip,  # numpy array, shape (32, H, W, C)
                'source_file': sample_info["source_file"],
                'prompts': prompts,
                'km_indices': km_indices,  # numpy array
                'video_indices': video_indices,  # numpy array
                'subject_id': sample_info.get("subject_id"),
                'direction': sample_info.get("direction"),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'label': label,  # float (0.0 or 1.0) or None
                'label_value': label_value,  # raw label (list/int/float) from JSON
                'binary_threshold': binary_threshold,  # threshold used to derive label
            }
            
            # Save patch file
            with open(patch_path, 'wb') as f:
                pickle.dump(patch_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Add to metadata
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
            })
            
            patch_id_counter += 1
            
            # Save intermediate metadata periodically
            if (idx + 1) % chunk_size == 0:
                metadata_path = os.path.join(output_dir, 'patch_metadata.pkl')
                config = {
                    'patch_size': patch_size,
                    'video_frame_count': video_frame_count,
                    'video_target_size': video_target_size,
                    'mode': mode,
                    'binary_threshold': binary_threshold,
                    # Note: km_gaussian_noise_std is stored but not applied (for reference)
                    # Noise will be applied during dataloader loading
                    'km_gaussian_noise_std': km_gaussian_noise_std,
                    'prompt_selection': prompt_selection,
                    'created_at': datetime.now().isoformat(),
                }
                with open(metadata_path, 'wb') as f:
                    pickle.dump({'patch_metadata': patch_metadata, 'config': config}, f, protocol=pickle.HIGHEST_PROTOCOL)
                progress_bar.write(f"💾 Saved intermediate metadata ({idx + 1}/{len(dataset)} samples processed, {patch_id_counter} patches saved)")
        
        except Exception as e:
            progress_bar.write(f"❌ Error processing sample {idx} ({sample_info.get('source_file', 'unknown')}): {e}")
            continue
    
    # Close progress bar
    progress_bar.close()

    # In train mode, verify that all clips from the same source file have the same label/label_value
    if mode == "train" and patch_metadata:
        ok, errs = check_train_split_label_consistency(patch_metadata, mode)
        if ok:
            print("\n✅ Train split label check: all clips from the same source have the same label values.")
        else:
            print("\n⚠️ Train split label check FAILED (same source has different labels):")
            for e in errs[:20]:
                print(f"   {e}")
            if len(errs) > 20:
                print(f"   ... and {len(errs) - 20} more.")
    
    # Save final metadata
    print("\n💾 Saving final metadata...")
    metadata_path = os.path.join(output_dir, 'patch_metadata.pkl')
    config = {
        'patch_size': patch_size,
        'video_frame_count': video_frame_count,
        'video_target_size': video_target_size,
        'mode': mode,
        'binary_threshold': binary_threshold,
        'km_gaussian_noise_std': km_gaussian_noise_std,
        'prompt_selection': prompt_selection,
        'created_at': datetime.now().isoformat(),
        'table_dir': table_dir,
        'video_dir': video_dir,
        'label_json_path': label_json_path,
        'split': split,
        'prompts_path': prompts_path,
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump({'patch_metadata': patch_metadata, 'config': config}, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n✅ Processing complete!")
    print(f"  📦 Total patches: {len(patch_metadata)}")
    print(f"  📁 Patches directory: {patches_dir}")
    print(f"  📄 Metadata file: {metadata_path}")
    
    # Print summary statistics
    if patch_metadata:
        labels = [p.get('label') for p in patch_metadata if p.get('has_label')]
        if labels:
            label_counts = {0.0: labels.count(0.0), 1.0: labels.count(1.0)}
            print(f"  📊 Label distribution: {label_counts}")
        print(f"  📏 Knowledge map shape: {patch_metadata[0].get('knowledge_map_shape', 'unknown')}")
        print(f"  🎥 Video shape: {patch_metadata[0].get('video_shape', 'unknown')}")
    
    return metadata_path


def main():
    """Main function with example configuration."""
    # Configuration (similar to run_end2end.py)
    config = {
        "table_dir": r"C:\Users\Administrator\project\data\pk_exter_table",
        "video_dir": r"C:\Users\Administrator\project\data\pk_exter_video",
        "label_json_path": r"data\test_indices_pk.json", #r"data\test_indices.json",
        "split": "test",  # "train" or "test"
        "patch_size": 96,
        "video_frame_count": 32,
        "video_target_size": (224, 224),
        "prompts_path": r"C:\Users\Administrator\project\data\sz_general_gait_prompts.json",
        "prompt_selection": "concise_prompts",
        "binary_threshold": 15.0,
        "km_gaussian_noise_std": 0.,  # Set to None to disable augmentation
        "mode": "test",  # "train" or "test"
        "output_dir": "./data/test_pk_pkl^1",  # Output directory for pickle files
    }
    
    # Process and save
    metadata_path = process_and_save_fullgait_v2(
        table_dir=config["table_dir"],
        video_dir=config["video_dir"],
        output_dir=config["output_dir"],
        label_json_path=config.get("label_json_path"),
        split=config.get("split"),
        patch_size=config["patch_size"],
        video_frame_count=config["video_frame_count"],
        video_target_size=config["video_target_size"],
        prompts_path=config.get("prompts_path"),
        prompt_selection=config.get("prompt_selection", "top_feature_prompts"),
        binary_threshold=config.get("binary_threshold", 11.0),
        km_gaussian_noise_std=config.get("km_gaussian_noise_std"),
        mode=config.get("mode", "train"),
    )
    
    print(f"\nPreprocessing complete! Metadata saved to: {metadata_path}")
    print(f"You can now use SigLIPFullGaitDatasetPKL (to be created) or modify existing dataset classes to load from: {config['output_dir']}")


if __name__ == "__main__":
    main()

