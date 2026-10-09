"""
Script to create scoliosis1k_plusv2 dataset from JSON indices and save as PKL files.

This script:
1. Reads test_indices.json to map indices to labels (thoracic, lumbar angles)
2. Uses a threshold parameter to categorize samples as "positive" or "negative"
3. Generates silhouette images from videos in sz_video_refinedyolo using YOLO segmentation
4. Saves data as PKL files in the structure: {sample_id}/{label}/{angle}/{angle}.pkl

Structure:
- Input: Video files from sz_video_refinedyolo
- Output: {sample_id}/{label}/{angle}/{angle}.pkl
- PKL contains numpy array of shape (num_frames, height, width) with dtype uint8
"""

import json
import cv2
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import re
import csv
import argparse
from ultralytics import YOLO
import tempfile
import shutil


def extract_index_from_filename(filename):
    """Extract index number from filename like 'sz_1_step1.mp4' -> 1"""
    match = re.search(r'sz_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None


def determine_label(thoracic_angle, lumbar_angle, threshold=10.0):
    """
    Determine if sample is positive (has scoliosis) or negative (normal).
    
    Args:
        thoracic_angle: Thoracic Cobb angle
        lumbar_angle: Lumbar Cobb angle
        threshold: Threshold value (if either angle >= threshold, it's positive)
        
    Returns:
        "positive" if either angle >= threshold, "negative" otherwise
    """
    if thoracic_angle >= threshold or lumbar_angle >= threshold:
        return "positive"
    return "negative"


def find_video_file(video_dir, index):
    """
    Find video file for a given index.
    
    Args:
        video_dir: Directory containing video files
        index: Sample index number
        
    Returns:
        Path to video file, or None if not found
    """
    video_dir = Path(video_dir)
    
    # Try different possible video file patterns
    possible_patterns = [
        f"sz_{index}_step1.mp4",
        f"sz_{index}_step_1.mp4",
        f"sz_{index}.mp4",
    ]
    
    for pattern in possible_patterns:
        video_path = video_dir / pattern
        if video_path.exists():
            return video_path
    
    # Try to find by pattern matching
    for video_file in video_dir.glob(f"sz_{index}*.mp4"):
        return video_file
    
    return None


def normalize_silhouette(mask, target_size=(64, 64), padding_ratio=0.05):
    """
    Normalize a silhouette mask to a fixed size while preserving aspect ratio.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 255
        target_size: Output size (height, width)
        padding_ratio: Padding around the silhouette (0.05 = 5% on each side)
        
    Returns:
        Normalized silhouette
    """
    # Find bounding box of the silhouette
    coords = np.where(mask > 127)
    if len(coords[0]) == 0:
        # No silhouette found, return empty
        return np.zeros(target_size, dtype=np.uint8)
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Crop the silhouette region
    cropped = mask[y_min:y_max+1, x_min:x_max+1]
    
    # Calculate target dimensions with padding
    target_h, target_w = target_size
    available_h = int(target_h * (1 - 2 * padding_ratio))
    available_w = int(target_w * (1 - 2 * padding_ratio))
    
    # Calculate scale to fit while preserving aspect ratio
    h, w = cropped.shape
    scale = min(available_h / h, available_w / w)
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize the cropped silhouette
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Create output canvas and center the silhouette
    output = np.zeros(target_size, dtype=np.uint8)
    
    # Calculate centering offsets
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # Place the silhouette in the center
    output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return output


def extract_silhouettes_from_video(video_path, output_dir, seg_model, target_size=(64, 64), 
                                   conf_threshold=0.5, padding_ratio=0.05):
    """
    Extract silhouettes from a video file using YOLO segmentation.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save silhouette images
        seg_model: YOLO segmentation model
        target_size: Target size (height, width) for silhouettes
        conf_threshold: Confidence threshold for detection
        padding_ratio: Padding ratio for normalization
        
    Returns:
        Number of frames extracted, or 0 if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return 0
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO segmentation
        results = seg_model(frame, conf=conf_threshold, verbose=False)
        
        # Create empty silhouette mask
        silhouette_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Process results
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            
            # Find largest person mask
            person_masks = []
            for mask, box in zip(masks, boxes):
                if int(box.cls) == 0:  # person class
                    mask_resized = cv2.resize(mask, (width, height))
                    mask_binary = (mask_resized * 255).astype(np.uint8)
                    area = np.sum(mask_binary > 127)
                    person_masks.append((mask_binary, area))
            
            if person_masks:
                # Select largest person
                person_masks.sort(key=lambda x: x[1], reverse=True)
                silhouette_mask = person_masks[0][0]
        
        # Normalize silhouette
        normalized_mask = normalize_silhouette(
            silhouette_mask,
            target_size=target_size,
            padding_ratio=padding_ratio
        )
        
        # Save silhouette frame (white on black)
        silhouette_frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        silhouette_frame[normalized_mask > 127] = [255, 255, 255]
        
        # Save frame
        cv2.imwrite(str(output_dir / f"frame_{saved_count:06d}.png"), silhouette_frame)
        
        saved_count += 1
        frame_idx += 1
    
    cap.release()
    return saved_count


def load_images_from_folder(folder_path, target_size=(64, 64)):
    """
    Load all PNG images from a folder, resize them, and return as numpy array.
    
    Args:
        folder_path: Path to folder containing PNG images
        target_size: Target size (height, width) for resizing images
        
    Returns:
        numpy array of shape (num_frames, height, width) with dtype uint8, or None
    """
    folder = Path(folder_path)
    if not folder.exists():
        return None
    
    # Get all PNG files and sort them
    # Try different naming patterns
    image_files = []
    for pattern in ["frame_*.png", "*.png", "sz_*.png"]:
        image_files = sorted(folder.glob(pattern))
        if len(image_files) > 0:
            break
    
    if len(image_files) == 0:
        return None
    
    # Load and resize images
    images = []
    target_h, target_w = target_size
    
    for img_file in image_files:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
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


def process_sample(index, label_values, video_dir, table_dir, output_root, threshold, 
                  seg_model, angle="000_180", target_size=(64, 64), temp_dir=None):
    """
    Process a single sample: generate silhouettes from video and save as PKL file.
    
    Args:
        index: Sample index from JSON
        label_values: List [thoracic_angle, lumbar_angle]
        video_dir: Directory containing video files
        table_dir: Directory containing CSV data (not used currently)
        output_root: Root output directory
        threshold: Threshold for positive/negative classification
        seg_model: YOLO segmentation model
        angle: Angle folder name (default: "000_180")
        target_size: Target image size (height, width)
        temp_dir: Temporary directory for storing silhouettes (optional)
        
    Returns:
        True if successful, False otherwise
    """
    thoracic_angle, lumbar_angle = label_values[0], label_values[1]
    
    # Determine label based on threshold
    label = determine_label(thoracic_angle, lumbar_angle, threshold)
    
    # Find video file
    video_path = find_video_file(video_dir, index)
    
    if video_path is None:
        print(f"Warning: Could not find video file for index {index}")
        return False
    
    # Create temporary directory for silhouettes if not provided
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
        cleanup_temp = True
    else:
        temp_dir = Path(temp_dir)
        cleanup_temp = False
    
    try:
        # Extract silhouettes from video
        silhouette_dir = temp_dir / f"sz_{index}" / "silhouette"
        num_frames = extract_silhouettes_from_video(
            video_path=video_path,
            output_dir=silhouette_dir,
            seg_model=seg_model,
            target_size=target_size,
            conf_threshold=0.5,
            padding_ratio=0.05
        )
        
        if num_frames == 0:
            print(f"Warning: No frames extracted for index {index}")
            return False
        
        # Load images
        images_array = load_images_from_folder(silhouette_dir, target_size=target_size)
        
        if images_array is None:
            print(f"Warning: No images loaded for index {index}")
            return False
        
        # Generate sample ID (use index, zero-padded to 5 digits)
        sample_id = f"{index:05d}"
        
        # Create output directory structure
        output_dir = Path(output_root) / sample_id / label / angle
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as PKL file
        pkl_file = output_dir / f"{angle}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(images_array, f)
        
        return True
    
    finally:
        # Clean up temporary directory if we created it
        if cleanup_temp and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def generate_label_file(target_root, processed_samples, threshold):
    """
    Generate label mapping files (text and CSV) documenting processed samples.
    
    Args:
        target_root: Root directory where label files will be saved
        processed_samples: List of dictionaries with sample information
        threshold: Threshold value used for classification
    """
    target_path = Path(target_root)
    
    # Generate text file
    txt_file = target_path / "label_mapping.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Scoliosis1K-PlusV2 Dataset Label Mapping\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Threshold used for classification: {threshold}\n")
        f.write("(Samples with thoracic or lumbar angle >= threshold are classified as 'positive')\n\n")
        
        f.write("Summary:\n")
        f.write("-" * 80 + "\n")
        positive_count = sum(1 for s in processed_samples if s['label'] == 'positive')
        negative_count = sum(1 for s in processed_samples if s['label'] == 'negative')
        f.write(f"Total samples: {len(processed_samples)}\n")
        f.write(f"Positive samples: {positive_count}\n")
        f.write(f"Negative samples: {negative_count}\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("Sample Details:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Sample ID':<12} {'Index':<8} {'Label':<12} {'Thoracic':<12} {'Lumbar':<12} {'Frames':<8}\n")
        f.write("-" * 80 + "\n")
        
        for sample in processed_samples:
            f.write(f"{sample['sample_id']:<12} {sample['index']:<8} {sample['label']:<12} "
                   f"{sample['thoracic']:<12.1f} {sample['lumbar']:<12.1f} {sample['frames']:<8}\n")
    
    # Generate CSV file
    csv_file = target_path / "label_mapping.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_ID', 'Index', 'Label', 'Thoracic_Angle', 'Lumbar_Angle', 'Num_Frames', 'Threshold'])
        for sample in processed_samples:
            writer.writerow([
                sample['sample_id'],
                sample['index'],
                sample['label'],
                sample['thoracic'],
                sample['lumbar'],
                sample['frames'],
                threshold
            ])


def main():
    """Main function to process all samples from JSON indices."""
    parser = argparse.ArgumentParser(description='Create scoliosis1k_plusv2 dataset from JSON indices')
    parser.add_argument('--threshold', type=float, default=15.0,
                       help='Threshold for positive/negative classification (default: 10.0)')
    parser.add_argument('--target_size', type=int, nargs=2, default=[64, 64],
                       help='Target image size as height width (default: 64 64)')
    parser.add_argument('--angle', type=str, default='000_180',
                       help='Angle folder name (default: 000_180)')
    parser.add_argument('--yolo_model', type=str, default='yolov8n-seg.pt',
                       help='Path to YOLO segmentation model (default: yolov8n-seg.pt)')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold for YOLO detection (default: 0.5)')
    parser.add_argument('--temp_dir', type=str, default=None,
                       help='Temporary directory for storing silhouettes (default: auto-create)')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    json_file = script_dir / "test_indices.json"
    video_dir = Path(r"C:\Users\Olive\Desktop\video_retrival\video_retrival\sz_video_refinedyolo")
    table_dir = Path(r"C:\Users\Olive\Desktop\video_retrival\video_retrival\sz_table_refinedyolo")
    target_root = Path(r"C:\Users\Olive\Desktop\Nature_Communication\code_video\pytorch\get_silhouette\scoliosis1k-plusv2")
    
    # Create target root directory
    target_root.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO model
    print(f"Loading YOLO model: {args.yolo_model}")
    try:
        seg_model = YOLO(args.yolo_model)
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please make sure ultralytics is installed: pip install ultralytics")
        return
    
    # Load JSON file
    if not json_file.exists():
        print(f"Error: JSON file not found: {json_file}")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get test samples
    test_samples = data.get('test', [])
    
    if len(test_samples) == 0:
        print("Error: No test samples found in JSON file")
        return
    
    print(f"Loaded {len(test_samples)} samples from JSON file")
    print(f"Threshold: {args.threshold}")
    print(f"Target size: {args.target_size}")
    print(f"Video directory: {video_dir}")
    print(f"Table directory: {table_dir}")
    print(f"Output directory: {target_root}")
    print("-" * 80)
    
    # Create temporary directory if specified
    temp_dir = None
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each sample
    successful = 0
    failed = 0
    processed_samples = []
    
    for sample_data in tqdm(test_samples, desc="Processing samples"):
        index = sample_data.get('index')
        label_values = sample_data.get('label', [0.0, 0.0])
        
        if index is None:
            print(f"Warning: Sample missing index: {sample_data}")
            failed += 1
            continue
        
        if len(label_values) < 2:
            print(f"Warning: Sample {index} has invalid label values: {label_values}")
            failed += 1
            continue
        
        # Process sample
        success = process_sample(
            index=index,
            label_values=label_values,
            video_dir=video_dir,
            table_dir=table_dir,
            output_root=target_root,
            threshold=args.threshold,
            seg_model=seg_model,
            angle=args.angle,
            target_size=tuple(args.target_size),
            temp_dir=temp_dir
        )
        
        if success:
            # Get number of frames from the PKL file
            sample_id = f"{index:05d}"
            thoracic_angle, lumbar_angle = label_values[0], label_values[1]
            label = determine_label(thoracic_angle, lumbar_angle, args.threshold)
            pkl_file = target_root / sample_id / label / args.angle / f"{args.angle}.pkl"
            
            num_frames = 0
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    images_array = pickle.load(f)
                    num_frames = images_array.shape[0]
            
            processed_samples.append({
                'sample_id': sample_id,
                'index': index,
                'label': label,
                'thoracic': thoracic_angle,
                'lumbar': lumbar_angle,
                'frames': num_frames
            })
            successful += 1
        else:
            failed += 1
            print(f"Failed to process sample index {index}")
    
    # Generate label file
    if len(processed_samples) > 0:
        generate_label_file(target_root, processed_samples, args.threshold)
    
    print("-" * 80)
    print(f"Processing complete!")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {target_root}")
    print(f"Label file created: {target_root / 'label_mapping.txt'}")
    print(f"Label CSV created: {target_root / 'label_mapping.csv'}")


if __name__ == "__main__":
    main()
