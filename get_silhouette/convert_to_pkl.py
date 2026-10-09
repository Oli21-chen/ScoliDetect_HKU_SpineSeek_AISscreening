"""
Script to convert raw silhouette PNG images to PKL format.

This script processes silhouette images from scoliosis1k_plus_raw and converts them
to PKL format matching the structure of Scoliosis1K-sil-pkl dataset.

Structure:
- Input: {sample_id}/{label}/{angle}/*.png
- Output: {sample_id}/{label}/{angle}/{angle}.pkl
- PKL contains numpy array of shape (num_frames, height, width) with dtype uint8
"""

import cv2
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm


def load_images_from_folder(folder_path, target_size=(64, 64)):
    """
    Load all PNG images from a folder, resize them, and return as numpy array.
    
    Args:
        folder_path: Path to folder containing PNG images
        target_size: Target size (height, width) for resizing images
        
    Returns:
        numpy array of shape (num_frames, height, width) with dtype uint8
    """
    folder = Path(folder_path)
    if not folder.exists():
        return None
    
    # Get all PNG files and sort them
    image_files = sorted(folder.glob("*.png"))
    
    if len(image_files) == 0:
        return None
    
    # Load and resize images
    images = []
    target_h, target_w = target_size
    
    for img_file in image_files:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {img_file}")
            continue
        
        # Resize image to target size if needed
        if img.shape != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        images.append(img)
    
    if len(images) == 0:
        return None
    
    # Stack images into numpy array
    # Shape: (num_frames, height, width)
    images_array = np.stack(images, axis=0).astype(np.uint8)
    
    return images_array


def process_sample(sample_dir, output_root, angle="000_180"):
    """
    Process a single sample directory and convert to PKL format.
    
    Args:
        sample_dir: Path to sample directory (e.g., 00000)
        output_root: Root output directory
        angle: Angle folder name (default: "000_180")
        
    Returns:
        True if successful, False otherwise
    """
    sample_dir = Path(sample_dir)
    sample_id = sample_dir.name
    
    # Find label folder (negative or positive)
    label_folders = [d for d in sample_dir.iterdir() if d.is_dir() and d.name in ['negative', 'positive']]
    
    if len(label_folders) == 0:
        print(f"Warning: No label folder found in {sample_dir}")
        return False
    
    # Process each label folder
    for label_folder in label_folders:
        label = label_folder.name
        angle_folder = label_folder / angle
        
        if not angle_folder.exists():
            print(f"Warning: Angle folder does not exist: {angle_folder}")
            continue
        
        # Load images (resize to 64x64 to match reference format)
        images_array = load_images_from_folder(angle_folder, target_size=(64, 64))
        
        if images_array is None:
            print(f"Warning: No images loaded from {angle_folder}")
            continue
        
        # Create output directory structure
        output_dir = Path(output_root) / sample_id / label / angle
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as PKL file
        pkl_file = output_dir / f"{angle}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(images_array, f)
        
        return True
    
    return False


def main():
    """Main function to process all samples."""
    # Paths
    input_root = Path(r"C:\Users\Olive\Desktop\scoliosis1K\scoliosis1k_plus_raw")
    output_root = Path(r"C:\Users\Olive\Desktop\scoliosis1K\scoliosis1k_plus_pkl")
    angle = "000_180"
    
    # Create output root directory
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Get all sample directories
    sample_dirs = sorted([d for d in input_root.iterdir() if d.is_dir() and d.name.isdigit()])
    
    if len(sample_dirs) == 0:
        print(f"Error: No sample directories found in {input_root}")
        return
    
    print(f"Found {len(sample_dirs)} samples to process")
    print(f"Input: {input_root}")
    print(f"Output: {output_root}")
    print("-" * 80)
    
    # Process each sample
    successful = 0
    failed = 0
    
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        if process_sample(sample_dir, output_root, angle):
            successful += 1
        else:
            failed += 1
            print(f"Failed to process: {sample_dir.name}")
    
    print("-" * 80)
    print(f"Processing complete!")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_root}")


if __name__ == "__main__":
    main()

