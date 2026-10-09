"""
Generate train/test indices JSON files based on the same 9:1 split used in run_end2end.py.

Reproduces the exact random_split from run_end2end.py (same pkl dirs, same seed),
then extracts unique subject_ids + labels for each split and saves as JSON.

Output:
  data/train_indices_split.json  (90% train+val subjects)
  data/test_indices_split.json   (10% test subjects)
"""

import os
import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import ConcatDataset, random_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from utils.data_sampler import SigLIPFullGaitDatasetPKL


def get_patch_metadata_entry(full_dataset, global_idx):
    """Resolve global index -> patch_metadata entry for Dataset or ConcatDataset."""
    if isinstance(full_dataset, ConcatDataset):
        offset = 0
        for ds in full_dataset.datasets:
            if global_idx < offset + len(ds):
                local_idx = global_idx - offset
                return ds.patch_metadata[local_idx]
            offset += len(ds)
        raise IndexError(f"Index {global_idx} out of range")
    else:
        return full_dataset.patch_metadata[global_idx]


def collect_subjects(full_dataset, indices):
    """Collect unique (subject_id, label_value) pairs from split indices."""
    subjects = {}
    for idx in indices:
        entry = get_patch_metadata_entry(full_dataset, idx)
        sid = entry.get("subject_id")
        if sid is None:
            continue
        sid_key = str(int(float(sid)))
        if sid_key not in subjects:
            subjects[sid_key] = entry.get("label_value")
    return subjects


def main():
    # Must match run_end2end.py config exactly
    pkl_data_dirs = ["./data/train_sz_pkl^1", "./data/test_sz_pkl^1"]
    random_seed = 42
    train_val_ratio = 0.8  # train+val
    test_ratio = 0.2
    binary_threshold = 15.0

    print("Loading PKL datasets (same as run_end2end.py)...")
    datasets = []
    total_patches = 0
    for idx, entry in enumerate(pkl_data_dirs):
        pkl_data_dir = os.path.abspath(os.path.normpath(entry))
        print(f"  PKL dir [{idx}]: {pkl_data_dir}")
        ds = SigLIPFullGaitDatasetPKL(
            pkl_data_dir=pkl_data_dir,
            km_gaussian_noise_std=None,
            mode="train",
            prompts_path=os.path.join(project_root, "data", "sz_general_gait_prompts.json"),
            prompt_selection="concise_prompts",
            binary_threshold=binary_threshold,
        )
        print(f"    Loaded {len(ds)} patches")
        datasets.append(ds)
        total_patches += len(ds)

    if len(datasets) == 1:
        full_dataset = datasets[0]
    else:
        full_dataset = ConcatDataset(datasets)
    print(f"\nTotal patches: {total_patches}")

    # Reproduce the same split as run_end2end.py
    total = len(full_dataset)
    train_val_size = max(0, int(total * train_val_ratio))
    test_size = total - train_val_size
    print(f"Splitting: train+val={train_val_size}, test={test_size} (seed={random_seed})")

    gen = torch.Generator().manual_seed(random_seed)
    train_val_subset, test_subset = random_split(
        full_dataset, [train_val_size, test_size], generator=gen
    )

    # Collect unique subjects from each split
    train_val_subjects = collect_subjects(full_dataset, train_val_subset.indices)
    test_subjects = collect_subjects(full_dataset, test_subset.indices)

    # Check for overlap (subjects appearing in both splits due to patch-level splitting)
    overlap = set(train_val_subjects.keys()) & set(test_subjects.keys())
    if overlap:
        print(f"\n⚠️  {len(overlap)} subjects appear in BOTH splits (patch-level split, not subject-level):")
        print(f"   Overlap subject IDs: {sorted(overlap, key=int)[:20]}{'...' if len(overlap) > 20 else ''}")
        print(f"   These subjects had patches in both train+val and test.")

    # Build JSON in same format as train_indices.json / test_indices.json
    def build_entries(subjects_dict):
        entries = []
        for sid in sorted(subjects_dict.keys(), key=lambda x: int(x)):
            label = subjects_dict[sid]
            entries.append({"index": int(sid), "label": label})
        return entries

    train_json = {"train": build_entries(train_val_subjects)}
    test_json = {"test": build_entries(test_subjects)}

    out_dir = os.path.join(project_root, "data")
    train_path = os.path.join(out_dir, "train_indices_split.json")
    test_path = os.path.join(out_dir, "test_indices_split.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_json, f, indent=4)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_json, f, indent=4)

    print(f"\n✅ Saved:")
    print(f"  Train+val: {train_path} ({len(train_val_subjects)} subjects, {train_val_size} patches)")
    print(f"  Test:      {test_path} ({len(test_subjects)} subjects, {test_size} patches)")

    # Summary
    all_subjects = set(train_val_subjects.keys()) | set(test_subjects.keys())
    print(f"\n  Total unique subjects: {len(all_subjects)}")
    print(f"  Train+val only: {len(train_val_subjects) - len(overlap)}")
    print(f"  Test only:      {len(test_subjects) - len(overlap)}")
    print(f"  In both:        {len(overlap)}")


if __name__ == "__main__":
    main()
