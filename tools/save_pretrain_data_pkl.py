# -*- coding: utf-8 -*-
"""
Script to preprocess and save SigLIP pretraining data as pkl files.
This speeds up data loading by avoiding on-the-fly processing.
"""

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
pytorch_dir = os.path.dirname(current_dir)  # pytorch directory
sys.path.insert(0, pytorch_dir)

# Import from utils.data_sampler (relative to pytorch directory)
from utils.data_sampler import (
    SigLIPPretrainDataset,
    extract_subject_id_from_filename,
    load_prompts_json,
    get_prompts_for_sample
)


def save_patches_as_pkl(
    table_dir: str,
    video_dir: str,
    directions: List[str],
    output_dir: str,
    patch_size: int = 96,
    video_frame_count: int = 32,
    pad_mode: str = 'zero',
    video_target_size: Optional[Tuple[int, int]] = (224, 224),
    prompts_path: Optional[str] = None,
    prompt_selection: str = 'top_feature_prompts',
    max_samples_per_file: Optional[int] = None,
    num_workers: int = 0
):
    """
    Preprocess and save all patches as pkl files.
    
    Args:
        table_dir: Base directory containing knowledge map CSV files
        video_dir: Base directory containing video files
        directions: List of direction subdirectories to include
        output_dir: Directory to save pkl files
        patch_size: Size of each patch for knowledge map
        video_frame_count: Number of frames to sample from video patch
        pad_mode: Padding mode - 'repeat' or 'zero'
        video_target_size: Optional (width, height) to resize video frames
        prompts_path: Path to individual_gait_prompts.json file (optional)
        prompt_selection: Selection key for prompts
        max_samples_per_file: Maximum number of patches per file (None for all)
        num_workers: Number of workers for parallel processing (0 for sequential)
    """
    print("=" * 50)
    print("Preprocessing and Saving Data as PKL Files")
    print("=" * 50)
    print(f"Table directory: {table_dir}")
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Patch size: {patch_size}")
    print(f"Video frame count: {video_frame_count}")
    print(f"Video target size: {video_target_size}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    patches_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)
    
    # Load prompts if provided
    prompts_data = None
    if prompts_path is not None:
        try:
            prompts_data = load_prompts_json(prompts_path)
            print(f"\nLoaded prompts from: {prompts_path}")
            print(f"  Prompt selection: {prompt_selection}")
        except Exception as e:
            print(f"Warning: Could not load prompts: {e}")
            prompts_data = None
    
    # Create dataset to get all patch indices
    print("\nCreating dataset to enumerate patches...")
    with tqdm(desc="Initializing dataset", unit="step", ncols=100, total=1) as init_pbar:
        dataset = SigLIPPretrainDataset(
            table_dir=table_dir,
            video_dir=video_dir,
            directions=directions,
            patch_size=patch_size,
            video_frame_count=video_frame_count,
            pad_mode=pad_mode,
            max_samples_per_file=max_samples_per_file,
            video_target_size=video_target_size,
            prompts_path=prompts_path,
            prompt_selection=prompt_selection
        )
        init_pbar.update(1)
    
    total_patches = len(dataset)
    print(f"Total patches to process: {total_patches}")
    
    # Process and save each patch
    patch_metadata = []
    failed_patches = []
    
    print("\nProcessing and saving patches...")
    # Enhanced progress bar with more details
    with tqdm(total=total_patches, desc="Processing patches", 
              unit="patch", ncols=100, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for idx in range(total_patches):
            try:
                # Get patch data
                patch_data = dataset[idx]
                
                # Extract metadata
                patch_info = dataset.patch_indices[idx]
                subject_id = patch_info.get('subject_id', 'unknown')
                direction = patch_info.get('direction', 'unknown')
                source_file = patch_info.get('source_file', 'unknown')
                start_idx = patch_info.get('start_idx', 0)
                end_idx = patch_info.get('end_idx', 0)
                
                # Create unique patch ID
                patch_id = f"{idx:06d}"
                pkl_filename = f"patch_{patch_id}.pkl"
                pkl_path = os.path.join(patches_dir, pkl_filename)
                
                # Prepare data to save
                save_data = {
                    'knowledge_map': patch_data['knowledge_map'].numpy(),  # Convert tensor to numpy
                    'video': patch_data['video'].numpy(),  # Convert tensor to numpy
                    'prompts': patch_data.get('prompts', []),
                    'subject_id': subject_id,
                    'direction': direction,
                    'source_file': source_file,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'patch_id': patch_id
                }
                
                # Save patch as pkl
                with open(pkl_path, 'wb') as f:
                    pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Store metadata
                patch_metadata.append({
                    'patch_id': patch_id,
                    'pkl_path': pkl_path,
                    'subject_id': subject_id,
                    'direction': direction,
                    'source_file': source_file,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'knowledge_map_shape': patch_data['knowledge_map'].shape,
                    'video_shape': patch_data['video'].shape,
                    'num_prompts': len(patch_data.get('prompts', []))
                })
                
                # Update progress bar with additional info
                pbar.set_postfix({
                    'success': len(patch_metadata),
                    'failed': len(failed_patches),
                    'subject': subject_id[:10] if len(str(subject_id)) > 10 else subject_id
                })
                pbar.update(1)
                
            except Exception as e:
                failed_patches.append({
                    'patch_idx': idx,
                    'error': str(e)
                })
                # Update progress bar even on failure
                pbar.set_postfix({
                    'success': len(patch_metadata),
                    'failed': len(failed_patches),
                    'error': 'Yes'
                })
                pbar.update(1)
                continue
    
    # Save metadata index
    print("\nSaving metadata index...")
    metadata_path = os.path.join(output_dir, 'patch_metadata.pkl')
    with tqdm(total=1, desc="Saving metadata", unit="file", ncols=100) as meta_pbar:
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'patch_metadata': patch_metadata,
                'failed_patches': failed_patches,
                'config': {
                    'table_dir': table_dir,
                    'video_dir': video_dir,
                    'directions': directions,
                    'patch_size': patch_size,
                    'video_frame_count': video_frame_count,
                    'pad_mode': pad_mode,
                    'video_target_size': video_target_size,
                    'prompts_path': prompts_path,
                    'prompt_selection': prompt_selection
                },
                'total_patches': len(patch_metadata),
                'failed_count': len(failed_patches)
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta_pbar.update(1)
    
    print("\n" + "=" * 50)
    print("Preprocessing Complete!")
    print("=" * 50)
    print(f"Total patches saved: {len(patch_metadata)}")
    print(f"Failed patches: {len(failed_patches)}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata file: {metadata_path}")
    print("=" * 50)
    
    return output_dir, metadata_path


def main():
    """Main function to preprocess and save data."""
    
    # Configuration - should match run.py config
    config = {
        'table_dir': r'C:\Users\Olive\.spyder-py3\sz621_table',
        'video_dir': r'C:\Users\Olive\.spyder-py3\sz621_video',
        'directions': ['going_backward', 'going_forward'],
        'video_target_size': (224, 224),
        'patch_size': 96,
        'video_frame_count': 32,
        'pad_mode': 'zero',
        'output_dir': r'./preprocessed_data',  # Directory to save pkl files
        'prompts_path': r'C:\Users\Olive\Desktop\Nature_Communication\code_video\individual_prompts\individual_gait_prompts.json',
        'prompt_selection': 'top_feature_prompts',
        'max_samples_per_file': None,  # None for all patches
        'num_workers': 0  # 0 for sequential processing
    }
    
    # Run preprocessing
    output_dir, metadata_path = save_patches_as_pkl(
        table_dir=config['table_dir'],
        video_dir=config['video_dir'],
        directions=config['directions'],
        output_dir=config['output_dir'],
        patch_size=config['patch_size'],
        video_frame_count=config['video_frame_count'],
        pad_mode=config['pad_mode'],
        video_target_size=config['video_target_size'],
        prompts_path=config.get('prompts_path'),
        prompt_selection=config.get('prompt_selection', 'top_feature_prompts'),
        max_samples_per_file=config.get('max_samples_per_file'),
        num_workers=config.get('num_workers', 0)
    )
    
    print(f"\nPreprocessed data saved to: {output_dir}")
    print(f"Use this path in run.py: 'pkl_data_dir': '{output_dir}'")


if __name__ == '__main__':
    main()
