"""
Script to create scoliosis1k_plus dataset from all folder data.

This script copies silhouette data from normal, double, single_lumbar, and single_thoracic
folders and reorganizes it into the same structure as Scoliosis1K-sil-raw dataset:
- Structure: {sample_id}/{label}/{angle}/
- Example: 00000/negative/000_180/ (for normal)
- Example: 00050/positive/000_180/ (for double, single_lumbar, single_thoracic)
- Images renamed from frame_000000.png to 00000.png format
"""

import shutil
from pathlib import Path
from tqdm import tqdm
import re
import csv


def extract_frame_number(filename):
    """Extract frame number from filename like 'frame_000000.png' -> 0"""
    match = re.search(r'frame_(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return None


def copy_and_rename_silhouettes(source_dir, target_dir, sample_id, label="negative", angle="000_180"):
    """
    Copy silhouette images from source to target directory with renamed format.
    
    Args:
        source_dir: Source directory containing frame_*.png files
        target_dir: Target directory where renamed files will be copied
        sample_id: Sample ID (e.g., "00000")
        label: Label folder name ("negative" or "positive")
        angle: Angle folder name (default: "000_180")
    """
    # Create target directory structure
    target_path = Path(target_dir) / sample_id / label / angle
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get all silhouette files
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Warning: Source directory does not exist: {source_dir}")
        return 0
    
    silhouette_files = sorted(source_path.glob("frame_*.png"))
    
    if len(silhouette_files) == 0:
        print(f"Warning: No silhouette files found in {source_dir}")
        return 0
    
    # Copy and rename files
    copied_count = 0
    for source_file in silhouette_files:
        # Extract frame number
        frame_num = extract_frame_number(source_file.name)
        if frame_num is None:
            print(f"Warning: Could not extract frame number from {source_file.name}")
            continue
        
        # Create new filename: 00000.png, 00001.png, etc.
        new_filename = f"{frame_num:05d}.png"
        target_file = target_path / new_filename
        
        # Copy file
        shutil.copy2(source_file, target_file)
        copied_count += 1
    
    return copied_count


def generate_label_file(target_root, label_ranges):
    """
    Generate label mapping files (text and CSV) documenting sample ID ranges.
    
    Args:
        target_root: Root directory where label files will be saved
        label_ranges: List of dictionaries with category, label, start_id, end_id, count
    """
    target_path = Path(target_root)
    
    # Generate text file
    txt_file = target_path / "label_mapping.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Scoliosis1K-Plus Dataset Label Mapping\n")
        f.write("=" * 80 + "\n\n")
        f.write("This file documents the sample ID ranges for each category.\n\n")
        
        f.write("Category Mapping:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Category':<20} {'Label':<12} {'Start ID':<12} {'End ID':<12} {'Count':<8}\n")
        f.write("-" * 80 + "\n")
        
        for info in label_ranges:
            f.write(f"{info['category']:<20} {info['label']:<12} {info['start_id']:<12} {info['end_id']:<12} {info['count']:<8}\n")
        
        f.write("-" * 80 + "\n\n")
        
        f.write("Detailed Ranges:\n")
        f.write("-" * 80 + "\n")
        for info in label_ranges:
            f.write(f"\n{info['category'].upper()} ({info['label'].upper()}):\n")
            f.write(f"  Sample IDs: {info['start_id']} to {info['end_id']}\n")
            f.write(f"  Total samples: {info['count']}\n")
            f.write(f"  Range: {info['start_id']}-{info['end_id']}\n")
    
    # Generate CSV file
    csv_file = target_path / "label_mapping.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Label', 'Start_ID', 'End_ID', 'Count', 'Range'])
        for info in label_ranges:
            writer.writerow([
                info['category'],
                info['label'],
                info['start_id'],
                info['end_id'],
                info['count'],
                f"{info['start_id']}-{info['end_id']}"
            ])


def main():
    """Main function to process all samples from all folders."""
    # Paths
    script_dir = Path(__file__).parent
    source_base_dir = script_dir / "scoliosis1k-plus"
    target_root = Path(r"C:\Users\Olive\Desktop\scoliosis1K\scoliosis1k_plus")
    
    # Create target root directory
    target_root.mkdir(parents=True, exist_ok=True)
    
    # Define folder configurations: (folder_name, label)
    folder_configs = [
        ("normal", "negative"),
        ("double", "positive"),
        ("single_lumbar", "positive"),
        ("single_thoracic", "positive"),
    ]
    
    print(f"Target: {target_root}")
    print("-" * 80)
    
    # Process all folders sequentially
    total_copied = 0
    total_samples = 0
    sample_counter = 0  # Global counter for sample IDs
    
    # Track label ranges for documentation
    label_ranges = []
    
    for folder_name, label in folder_configs:
        source_dir = source_base_dir / folder_name
        
        if not source_dir.exists():
            print(f"Warning: Folder does not exist: {source_dir}")
            continue
        
        # Get all sample folders in this directory
        sample_folders = sorted([d for d in source_dir.iterdir() if d.is_dir()])
        
        if len(sample_folders) == 0:
            print(f"Warning: No sample folders found in {source_dir}")
            continue
        
        print(f"\nProcessing {folder_name} folder ({label}): {len(sample_folders)} samples")
        
        # Record start ID for this category
        start_id = sample_counter
        
        # Process each sample
        for sample_folder in tqdm(sample_folders, desc=f"Processing {folder_name}"):
            # Generate sample ID (continues from previous folders)
            sample_id = f"{sample_counter:05d}"
            
            # Path to silhouette folder
            silhouette_dir = sample_folder / "silhouette"
            
            # Copy and rename silhouettes
            copied = copy_and_rename_silhouettes(
                silhouette_dir,
                target_root,
                sample_id,
                label=label,
                angle="000_180"
            )
            
            total_copied += copied
            total_samples += 1
            sample_counter += 1
            
            if copied == 0:
                print(f"Warning: No files copied for sample {sample_folder.name} (ID: {sample_id})")
        
        # Record end ID for this category
        end_id = sample_counter - 1
        label_ranges.append({
            'category': folder_name,
            'label': label,
            'start_id': f"{start_id:05d}",
            'end_id': f"{end_id:05d}",
            'count': len(sample_folders)
        })
    
    # Generate label file
    generate_label_file(target_root, label_ranges)
    
    print("-" * 80)
    print(f"Processing complete!")
    print(f"Total samples processed: {total_samples}")
    print(f"Total images copied: {total_copied}")
    print(f"Output directory: {target_root}")
    print(f"Label file created: {target_root / 'label_mapping.txt'}")
    print(f"Label CSV created: {target_root / 'label_mapping.csv'}")


if __name__ == "__main__":
    main()

