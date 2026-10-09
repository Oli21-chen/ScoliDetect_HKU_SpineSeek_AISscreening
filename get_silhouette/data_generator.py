"""
Date: November 28, 2025 (Refined for new data)
This script is used to generate the data for the silhouette and pose estimation.

Purpose:
1. Get the silhouette and pose estimation data for different classes of data
2. Add white patch on face and neck area of RGB video and save frames (privacy protection)
3. Save the data into organized folder structure

Features:
- Configurable data paths and folder structures
- Flexible video/CSV naming patterns
- Support for different video resolutions
- Better error handling and validation
- Progress tracking and statistics
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import os
import shutil
from typing import Optional, Tuple, List, Dict, Callable
from tqdm import tqdm
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration - Can be overridden via command line arguments
# ============================================================================
# Default data root - can be set via --data_root argument
DEFAULT_DATA_ROOT = Path(r"C:/Users/Olive/scoliosis_predictive_factors_pose_estimation/scoligait_kinetfactors")
DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "scoliosis1k-plus"

# Default class definitions: (class_name, video_folder, table_folder)
# Can be customized via config file or command line
DEFAULT_CLASS_DEFINITIONS = [
    ("normal", "nv", "nt"),
    # ("single_thoracic", "new_XAI_single_thoracic_video50", "new_XAI_single_thoracic_table50"),
    # ("double", "new_XAI_double_video50", "new_XAI_double_table50"),
    # ("single_lumbar", "new_XAI_single_lumbar_video50", "new_XAI_single_lumbar_table50"),
]

# Output settings
DEFAULT_SILHOUETTE_SIZE = (128, 64)  # (height, width) - Standard gait recognition size
DEFAULT_MASKED_RGB_SIZE = (128, 64)  # Same size for masked RGB frames (head covered with white patch)
DEFAULT_PADDING_RATIO = 0.05  # 5% padding around silhouette

# Video/CSV naming patterns - can be customized
DEFAULT_VIDEO_PATTERNS = {
    'video_extensions': ['.mp4', '.avi', '.mov', '.mkv'],
    'csv_extensions': ['.csv'],
    'video_to_csv_mapper': lambda video_name: video_name.replace('step1', 'step_1') + '.csv',
}

# Default recorded video resolution for keypoint scaling
DEFAULT_RECORDED_SIZE = (1080, 1920)  # (width, height)

# YOLOv8 keypoint indices for face/neck region
FACE_NECK_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
}


# ============================================================================
# Configuration Management Functions
# ============================================================================
def load_config(config_path: Optional[Path] = None) -> Dict:
    """
    Load configuration from JSON file or return defaults.
    
    Args:
        config_path: Path to JSON config file (optional)
    
    Returns:
        Configuration dictionary
    """
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
    
    return {
        'data_root': str(DEFAULT_DATA_ROOT),
        'output_root': str(DEFAULT_OUTPUT_ROOT),
        'class_definitions': DEFAULT_CLASS_DEFINITIONS,
        'silhouette_size': DEFAULT_SILHOUETTE_SIZE,
        'masked_rgb_size': DEFAULT_MASKED_RGB_SIZE,
        'padding_ratio': DEFAULT_PADDING_RATIO,
        'recorded_size': DEFAULT_RECORDED_SIZE,
        'video_patterns': DEFAULT_VIDEO_PATTERNS,
    }


def find_csv_for_video(
    video_path: Path,
    table_dir: Path,
    video_to_csv_mapper: Optional[Callable] = None,
    csv_extensions: List[str] = None
) -> Optional[Path]:
    """
    Find corresponding CSV file for a video using flexible matching.
    
    Args:
        video_path: Path to video file
        table_dir: Directory containing CSV files
        video_to_csv_mapper: Function to map video name to CSV name (optional)
        csv_extensions: List of CSV file extensions to try
    
    Returns:
        Path to CSV file or None if not found
    """
    if csv_extensions is None:
        csv_extensions = ['.csv']
    
    video_name = video_path.stem
    
    # Try custom mapper first
    if video_to_csv_mapper:
        try:
            csv_name = video_to_csv_mapper(video_name)
            csv_path = table_dir / csv_name
            if csv_path.exists():
                return csv_path
        except Exception as e:
            logger.debug(f"Custom mapper failed: {e}")
    
    # Try common patterns
    patterns = [
        video_name.replace('step1', 'step_1') + '.csv',
        video_name.replace('step1', 'step1') + '.csv',
        video_name + '.csv',
        video_name.replace('_step1', '_step_1') + '.csv',
        video_name.replace('_step_1', '_step1') + '.csv',
    ]
    
    for pattern in patterns:
        csv_path = table_dir / pattern
        if csv_path.exists():
            return csv_path
    
    # Try finding by video ID (extract base name)
    video_id = video_name.rsplit('_', 1)[0] if '_' in video_name else video_name
    for ext in csv_extensions:
        # Try various patterns with video ID
        patterns = [
            f"{video_id}_step_1{ext}",
            f"{video_id}_step1{ext}",
            f"{video_id}{ext}",
        ]
        for pattern in patterns:
            csv_path = table_dir / pattern
            if csv_path.exists():
                return csv_path
    
    return None


def extract_video_id(video_path: Path, naming_pattern: str = "auto") -> str:
    """
    Extract video ID from video path using flexible patterns.
    
    Args:
        video_path: Path to video file
        naming_pattern: Pattern to use ('auto', 'last_underscore', 'first_underscore', 'stem')
    
    Returns:
        Video ID string
    """
    video_name = video_path.stem
    
    if naming_pattern == "auto":
        # Try to detect pattern
        if '_step' in video_name.lower():
            # Pattern: sz_116_step1 -> sz_116
            parts = video_name.rsplit('_step', 1)
            if len(parts) > 1:
                return parts[0]
        elif '_' in video_name:
            # Pattern: sz_116_step_1 -> sz_116
            parts = video_name.rsplit('_', 1)
            if len(parts) > 1 and parts[1].isdigit():
                return parts[0]
            # Pattern: video_name_part1_part2 -> video_name_part1
            return video_name.rsplit('_', 1)[0]
        else:
            return video_name
    elif naming_pattern == "last_underscore":
        parts = video_name.rsplit('_', 1)
        return parts[0] if len(parts) > 1 else video_name
    elif naming_pattern == "first_underscore":
        parts = video_name.split('_', 1)
        return parts[0] if len(parts) > 1 else video_name
    elif naming_pattern == "stem":
        return video_name
    else:
        return video_name


# ============================================================================
# Silhouette Extraction Functions (from process.py)
# ============================================================================
def normalize_silhouette(
    mask: np.ndarray,
    target_size: tuple = (64, 64),
    padding_ratio: float = 0.1,
) -> Tuple[np.ndarray, dict]:
    """
    Normalize a silhouette mask to a fixed size while preserving aspect ratio.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 255
        target_size: Output size (height, width)
        padding_ratio: Padding around the silhouette (0.1 = 10% on each side)
    
    Returns:
        Tuple of (normalized silhouette, transform_info dict)
    """
    transform_info = {
        'bbox': None,
        'scale': 1.0,
        'offset': (0, 0),
        'valid': False
    }
    
    # Find bounding box of the silhouette
    coords = np.where(mask > 127)
    if len(coords[0]) == 0:
        # No silhouette found, return empty
        return np.zeros(target_size, dtype=np.uint8), transform_info
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    transform_info['bbox'] = (x_min, y_min, x_max, y_max)
    
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
    
    transform_info['scale'] = scale
    
    # Resize the cropped silhouette
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Create output canvas and center the silhouette
    output = np.zeros(target_size, dtype=np.uint8)
    
    # Calculate centering offsets
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    transform_info['offset'] = (x_offset, y_offset)
    transform_info['valid'] = True
    
    # Place the silhouette in the center
    output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return output, transform_info


def apply_transform_to_rgb(
    rgb_frame: np.ndarray,
    transform_info: dict,
    target_size: tuple = (128, 64),
) -> np.ndarray:
    """
    Apply the same transform used for silhouette to an RGB frame.
    
    Args:
        rgb_frame: RGB frame (H, W, 3)
        transform_info: Transform info from normalize_silhouette
        target_size: Output size (height, width)
    
    Returns:
        Transformed RGB frame
    """
    if not transform_info['valid']:
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    x_min, y_min, x_max, y_max = transform_info['bbox']
    scale = transform_info['scale']
    x_offset, y_offset = transform_info['offset']
    
    # Crop the same region
    cropped = rgb_frame[y_min:y_max+1, x_min:x_max+1]
    
    # Calculate new size
    h, w = cropped.shape[:2]
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create output canvas
    output = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # Place in center
    output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return output


def transform_keypoints(
    keypoints: np.ndarray,
    transform_info: dict,
    target_size: tuple = (128, 64),
) -> np.ndarray:
    """
    Transform keypoints using the same transform as silhouette.
    
    Args:
        keypoints: Array of keypoints (N, 2) with (x, y) coordinates
        transform_info: Transform info from normalize_silhouette
        target_size: Output size (height, width)
    
    Returns:
        Transformed keypoints (N, 2)
    """
    if not transform_info['valid']:
        return np.zeros_like(keypoints)
    
    x_min, y_min, x_max, y_max = transform_info['bbox']
    scale = transform_info['scale']
    x_offset, y_offset = transform_info['offset']
    
    # Apply transform: translate, scale, offset
    transformed = keypoints.copy().astype(float)
    transformed[:, 0] = (transformed[:, 0] - x_min) * scale + x_offset
    transformed[:, 1] = (transformed[:, 1] - y_min) * scale + y_offset
    
    return transformed


# ============================================================================
# Head Masking Functions (White Patch for Privacy)
# ============================================================================
def apply_head_white_patch(
    frame: np.ndarray,
    keypoints: np.ndarray,
    expand_ratio: float = 1.5,
) -> np.ndarray:
    """
    Apply white patch to cover the whole head area using pose keypoints.
    Only covers head region, NOT the body.
    
    Args:
        frame: RGB frame (H, W, 3)
        keypoints: Array of 17 keypoints (17, 2) with (x, y) coordinates
        expand_ratio: Ratio to expand the head region
    
    Returns:
        Frame with white patch covering the head
    """
    frame_out = frame.copy()
    h, w = frame.shape[:2]
    
    if keypoints is None or len(keypoints) < 7:
        return frame_out
    
    # Get head keypoints only (indices 0-4: nose, eyes, ears)
    # Do NOT include shoulders (indices 5-6) to avoid covering body
    head_kpts = keypoints[:5]  # nose, left_eye, right_eye, left_ear, right_ear
    
    # Filter out invalid keypoints - must be within frame bounds
    valid_mask = (head_kpts[:, 0] > 0) & (head_kpts[:, 1] > 0) & \
                 (head_kpts[:, 0] < w) & (head_kpts[:, 1] < h)
    valid_kpts = head_kpts[valid_mask]
    
    if len(valid_kpts) < 1:
        # Fallback: try using nose only (index 0) if available
        nose = keypoints[0]
        if nose[0] > 0 and nose[1] > 0 and nose[0] < w and nose[1] < h:
            valid_kpts = np.array([nose])
        else:
            return frame_out
    
    # Calculate bounding box for head region
    x_min = valid_kpts[:, 0].min()
    x_max = valid_kpts[:, 0].max()
    y_min = valid_kpts[:, 1].min()
    y_max = valid_kpts[:, 1].max()
    
    # If only one keypoint, estimate head size based on typical proportions
    if len(valid_kpts) == 1:
        # Estimate head size as ~15% of frame height
        estimated_head_size = h * 0.12
        x_min = valid_kpts[0, 0] - estimated_head_size / 2
        x_max = valid_kpts[0, 0] + estimated_head_size / 2
        y_min = valid_kpts[0, 1] - estimated_head_size * 0.8
        y_max = valid_kpts[0, 1] + estimated_head_size * 0.5
    
    # Calculate head dimensions
    head_width = max(x_max - x_min, 1)
    head_height = max(y_max - y_min, 1)
    
    # Use shoulder position to limit bottom boundary (don't go below neck)
    shoulder_y = None
    if len(keypoints) >= 7:
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        valid_shoulders = []
        if 0 < left_shoulder[1] < h:
            valid_shoulders.append(left_shoulder[1])
        if 0 < right_shoulder[1] < h:
            valid_shoulders.append(right_shoulder[1])
        if valid_shoulders:
            shoulder_y = min(valid_shoulders)
    
    # Expand the region for full head coverage
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # Use larger dimension for consistent coverage
    head_size = max(head_width, head_height)
    half_w = head_size / 2 * expand_ratio
    half_h = head_size / 2 * expand_ratio
    
    # Calculate final box coordinates
    box_x1 = max(0, int(center_x - half_w))
    box_x2 = min(w, int(center_x + half_w))
    
    # Top: extend upward to cover hair/top of head
    box_y1 = max(0, int(center_y - half_h * 1.5))
    
    # Bottom boundary: use shoulder position or expanded head region (whichever is higher/smaller y)
    if shoulder_y is not None and shoulder_y > y_max:
        # Stop at neck level (slightly above shoulders)
        neck_y = y_max + (shoulder_y - y_max) * 0.3
        box_y2 = min(h, int(neck_y))
    else:
        box_y2 = min(h, int(center_y + half_h * 0.8))
    
    # Ensure valid box dimensions
    if box_x2 <= box_x1:
        box_x1 = max(0, int(center_x - 20))
        box_x2 = min(w, int(center_x + 20))
    if box_y2 <= box_y1:
        box_y1 = max(0, int(center_y - 30))
        box_y2 = min(h, int(center_y + 20))
    
    # Apply white patch (RGB: 255, 255, 255)
    if box_x2 > box_x1 and box_y2 > box_y1:
        frame_out[box_y1:box_y2, box_x1:box_x2] = [255, 255, 255]  # White patch
    
    return frame_out


# ============================================================================
# Data Processing Functions
# ============================================================================
def load_pose_data(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Load pose data from CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        DataFrame with pose data or None if failed
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading pose data from {csv_path}: {e}")
        return None


def get_keypoints_by_row_index(pose_df: pd.DataFrame, row_idx: int) -> Optional[np.ndarray]:
    """
    Get keypoints by row index (not frame number) from pose DataFrame.
    This matches video frame N to CSV row N.
    
    Args:
        pose_df: DataFrame with pose data
        row_idx: Row index to retrieve (0-based)
    
    Returns:
        Array of keypoints (17, 2) or None if not found
    """
    if pose_df is None:
        return None
    
    if row_idx < 0 or row_idx >= len(pose_df):
        return None
    
    row = pose_df.iloc[row_idx]
    
    # Extract keypoints
    keypoints = np.zeros((17, 2))
    for i in range(17):
        x_col = f'x{i+1}'
        y_col = f'y{i+1}'
        if x_col in row.index and y_col in row.index:
            keypoints[i] = [row[x_col], row[y_col]]
    
    return keypoints


def scale_keypoints(
    keypoints: np.ndarray,
    src_size: tuple,
    dst_size: tuple
) -> np.ndarray:
    """
    Scale keypoints from source size to destination size.
    
    Args:
        keypoints: Array of keypoints (N, 2) with (x, y) coordinates
        src_size: Source size (width, height) - the resolution keypoints were recorded at
        dst_size: Destination size (width, height) - the current frame resolution
    
    Returns:
        Scaled keypoints (N, 2)
    """
    if keypoints is None:
        return None
    
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    
    scaled = keypoints.copy()
    scaled[:, 0] = scaled[:, 0] * (dst_w / src_w)  # Scale x
    scaled[:, 1] = scaled[:, 1] * (dst_h / src_h)  # Scale y
    
    return scaled


def process_single_video(
    video_path: Path,
    pose_csv_path: Optional[Path],
    output_dir: Path,
    seg_model: YOLO,
    silhouette_size: tuple = (128, 64),
    masked_rgb_size: tuple = (128, 64),
    padding_ratio: float = 0.05,
    conf_threshold: float = 0.5,
    recorded_size: tuple = (1080, 1920),
    skip_if_exists: bool = True,
) -> dict:
    """
    Process a single video to extract silhouettes and masked RGB frames.
    
    Args:
        video_path: Path to input video
        pose_csv_path: Path to pose CSV file (can be None if not available)
        output_dir: Output directory for this video
        seg_model: YOLO segmentation model
        silhouette_size: Output silhouette size (H, W)
        masked_rgb_size: Output masked RGB size (H, W) - head covered with white patch
        padding_ratio: Padding ratio for normalization
        conf_threshold: Confidence threshold for detection
        recorded_size: Original video resolution (width, height) for keypoint scaling
        skip_if_exists: Skip processing if output already exists
    
    Returns:
        Dict with processing statistics
    """
    stats = {
        'total_frames': 0,
        'detected_frames': 0,
        'pose_frames': 0,
        'skipped': False,
        'error': None,
    }
    
    # Check if already processed
    silhouette_dir = output_dir / "silhouette"
    if skip_if_exists and silhouette_dir.exists():
        existing_frames = list(silhouette_dir.glob("*.png"))
        if len(existing_frames) > 0:
            logger.info(f"Skipping {video_path.name} - already processed ({len(existing_frames)} frames)")
            stats['skipped'] = True
            stats['total_frames'] = len(existing_frames)
            return stats
    
    # Create output subdirectories
    masked_rgb_dir = output_dir / "masked_rgb"  # RGB with white patch on head
    silhouette_dir.mkdir(parents=True, exist_ok=True)
    masked_rgb_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pose data (optional)
    pose_df = None
    if pose_csv_path and pose_csv_path.exists():
        pose_df = load_pose_data(pose_csv_path)
        if pose_df is None:
            logger.warning(f"Failed to load pose data from {pose_csv_path}")
    elif pose_csv_path:
        logger.warning(f"Pose CSV not found: {pose_csv_path}")
    else:
        logger.info(f"No pose CSV provided for {video_path.name}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        error_msg = f"Cannot open video: {video_path}"
        logger.error(error_msg)
        stats['error'] = error_msg
        return stats
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
                stats['detected_frames'] += 1
        
        # Normalize silhouette
        normalized_mask, transform_info = normalize_silhouette(
            silhouette_mask,
            target_size=silhouette_size,
            padding_ratio=padding_ratio
        )
        
        # Get pose keypoints for this frame (match by row index, not frame number)
        keypoints = None
        if pose_df is not None:
            keypoints = get_keypoints_by_row_index(pose_df, frame_idx)
            
            # Scale keypoints if video dimensions differ from original recording
            if keypoints is not None:
                rec_w, rec_h = recorded_size
                if width != rec_w or height != rec_h:
                    keypoints = scale_keypoints(keypoints, recorded_size, (width, height))
                stats['pose_frames'] += 1
        
        # Apply white patch to head in original frame (privacy protection)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if keypoints is not None:
            frame_masked = apply_head_white_patch(frame_rgb, keypoints)
        else:
            frame_masked = frame_rgb
        
        # Apply same transform to masked RGB frame
        masked_normalized = apply_transform_to_rgb(
            frame_masked,
            transform_info,
            target_size=masked_rgb_size
        )
        
        # Convert back to BGR for saving
        masked_normalized_bgr = cv2.cvtColor(masked_normalized, cv2.COLOR_RGB2BGR)
        
        # Save silhouette frame (white on black)
        silhouette_frame = np.zeros((silhouette_size[0], silhouette_size[1], 3), dtype=np.uint8)
        silhouette_frame[normalized_mask > 127] = [255, 255, 255]
        
        # Save frames
        cv2.imwrite(str(silhouette_dir / f"frame_{saved_count:06d}.png"), silhouette_frame)
        cv2.imwrite(str(masked_rgb_dir / f"frame_{saved_count:06d}.png"), masked_normalized_bgr)
        
        saved_count += 1
        frame_idx += 1
        stats['total_frames'] = frame_idx
    
    cap.release()
    
    # Copy pose data to output if available
    if pose_csv_path and pose_csv_path.exists():
        try:
            shutil.copy(pose_csv_path, output_dir / "pose_data.csv")
        except Exception as e:
            logger.warning(f"Failed to copy pose data: {e}")
    
    logger.info(f"Processed {video_path.name}: {stats['total_frames']} frames, "
                f"{stats['detected_frames']} detected, {stats['pose_frames']} with pose")
    
    return stats


def process_all_classes(
    data_root: Path,
    output_root: Path,
    class_definitions: List[Tuple[str, str, str]],
    model_name: str = "yolo11x-seg.pt",
    silhouette_size: tuple = (128, 64),
    masked_rgb_size: tuple = (128, 64),
    padding_ratio: float = 0.05,
    conf_threshold: float = 0.5,
    recorded_size: tuple = (1080, 1920),
    video_extensions: List[str] = None,
    video_to_csv_mapper: Optional[Callable] = None,
    skip_if_exists: bool = True,
):
    """
    Process all classes of data.
    
    Args:
        data_root: Root directory containing data
        output_root: Root directory for output
        class_definitions: List of (class_name, video_folder, table_folder) tuples
        model_name: YOLO segmentation model name
        silhouette_size: Output silhouette size (H, W)
        masked_rgb_size: Output masked RGB size (H, W)
        padding_ratio: Padding ratio for normalization
        conf_threshold: Confidence threshold for detection
        recorded_size: Original video resolution (width, height) for keypoint scaling
        video_extensions: List of video file extensions to process
        video_to_csv_mapper: Function to map video name to CSV name
        skip_if_exists: Skip processing if output already exists
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    logger.info("=" * 70)
    logger.info("Dataset Generator")
    logger.info("=" * 70)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Silhouette size: {silhouette_size}")
    logger.info(f"Masked RGB size: {masked_rgb_size}")
    logger.info(f"Classes: {[c[0] for c in class_definitions]}")
    logger.info("=" * 70)
    
    # Load segmentation model
    logger.info(f"\nLoading YOLO segmentation model: {model_name}")
    try:
        seg_model = YOLO(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return
    
    # Create output root
    output_root.mkdir(parents=True, exist_ok=True)
    
    total_stats = {
        'total_videos': 0,
        'total_frames': 0,
        'total_detected': 0,
        'total_pose_frames': 0,
        'skipped_videos': 0,
        'failed_videos': 0,
    }
    
    for class_name, video_folder, table_folder in class_definitions:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing class: {class_name.upper()}")
        logger.info(f"{'='*70}")
        
        video_dir = data_root / video_folder
        table_dir = data_root / table_folder
        output_class_dir = output_root / class_name
        
        if not video_dir.exists():
            logger.warning(f"Video directory not found: {video_dir}")
            continue
        
        # Get all video files with specified extensions
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(video_dir.glob(f"*{ext}")))
        video_files = sorted(video_files)
        
        logger.info(f"Found {len(video_files)} videos in {video_dir}")
        
        if len(video_files) == 0:
            logger.warning(f"No video files found in {video_dir}")
            continue
        
        for video_path in tqdm(video_files, desc=f"Processing {class_name}"):
            # Extract video ID
            video_id = extract_video_id(video_path)
            
            # Find corresponding CSV file
            csv_path = find_csv_for_video(
                video_path,
                table_dir,
                video_to_csv_mapper=video_to_csv_mapper,
                csv_extensions=['.csv']
            )
            
            if csv_path is None:
                logger.warning(f"No CSV found for {video_path.name}, continuing without pose data")
            
            # Create output directory for this video
            output_video_dir = output_class_dir / video_id
            
            # Process video
            try:
                stats = process_single_video(
                    video_path=video_path,
                    pose_csv_path=csv_path,
                    output_dir=output_video_dir,
                    seg_model=seg_model,
                    silhouette_size=silhouette_size,
                    masked_rgb_size=masked_rgb_size,
                    padding_ratio=padding_ratio,
                    conf_threshold=conf_threshold,
                    recorded_size=recorded_size,
                    skip_if_exists=skip_if_exists,
                )
                
                if stats['skipped']:
                    total_stats['skipped_videos'] += 1
                elif stats['error']:
                    total_stats['failed_videos'] += 1
                else:
                    total_stats['total_videos'] += 1
                    total_stats['total_frames'] += stats['total_frames']
                    total_stats['total_detected'] += stats['detected_frames']
                    total_stats['total_pose_frames'] += stats['pose_frames']
            except Exception as e:
                logger.error(f"Error processing {video_path.name}: {e}")
                total_stats['failed_videos'] += 1
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total videos processed: {total_stats['total_videos']}")
    logger.info(f"Skipped (already exists): {total_stats['skipped_videos']}")
    logger.info(f"Failed: {total_stats['failed_videos']}")
    logger.info(f"Total frames processed: {total_stats['total_frames']}")
    logger.info(f"Frames with detection: {total_stats['total_detected']}")
    logger.info(f"Frames with pose: {total_stats['total_pose_frames']}")
    if total_stats['total_frames'] > 0:
        logger.info(f"Detection rate: {total_stats['total_detected']/total_stats['total_frames']*100:.1f}%")
        if total_stats['total_pose_frames'] > 0:
            logger.info(f"Pose rate: {total_stats['total_pose_frames']/total_stats['total_frames']*100:.1f}%")
    logger.info(f"Output directory: {output_root}")


def process_single_class(
    class_name: str,
    data_root: Path,
    output_root: Path,
    class_definitions: List[Tuple[str, str, str]],
    model_name: str = "yolo11x-seg.pt",
    silhouette_size: tuple = (128, 64),
    masked_rgb_size: tuple = (128, 64),
    padding_ratio: float = 0.05,
    conf_threshold: float = 0.5,
    recorded_size: tuple = (1080, 1920),
    max_videos: Optional[int] = None,
    video_extensions: List[str] = None,
    video_to_csv_mapper: Optional[Callable] = None,
    skip_if_exists: bool = True,
):
    """
    Process a single class of data.
    
    Args:
        class_name: Class name to process
        data_root: Root directory containing data
        output_root: Root directory for output
        class_definitions: List of (class_name, video_folder, table_folder) tuples
        model_name: YOLO segmentation model name
        silhouette_size: Output silhouette size (H, W)
        masked_rgb_size: Output masked RGB size (H, W)
        padding_ratio: Padding ratio for normalization
        conf_threshold: Confidence threshold for detection
        recorded_size: Original video resolution (width, height) for keypoint scaling
        max_videos: Maximum number of videos to process (None for all)
        video_extensions: List of video file extensions to process
        video_to_csv_mapper: Function to map video name to CSV name
        skip_if_exists: Skip processing if output already exists
    """
    # Find class definition
    class_def = None
    for cname, vfolder, tfolder in class_definitions:
        if cname == class_name:
            class_def = (cname, vfolder, tfolder)
            break
    
    if class_def is None:
        logger.error(f"Unknown class '{class_name}'")
        logger.error(f"Valid classes: {[c[0] for c in class_definitions]}")
        return
    
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    logger.info("=" * 70)
    logger.info(f"Processing class: {class_name.upper()}")
    logger.info("=" * 70)
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    try:
        seg_model = YOLO(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return
    
    class_name, video_folder, table_folder = class_def
    video_dir = data_root / video_folder
    table_dir = data_root / table_folder
    output_class_dir = output_root / class_name
    
    # Get all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(video_dir.glob(f"*{ext}")))
    video_files = sorted(video_files)
    
    if max_videos:
        video_files = video_files[:max_videos]
    
    logger.info(f"Processing {len(video_files)} videos...")
    
    for video_path in tqdm(video_files, desc=f"Processing {class_name}"):
        video_id = extract_video_id(video_path)
        
        csv_path = find_csv_for_video(
            video_path,
            table_dir,
            video_to_csv_mapper=video_to_csv_mapper,
            csv_extensions=['.csv']
        )
        
        if csv_path is None:
            logger.warning(f"No CSV found for {video_path.name}")
        
        output_video_dir = output_class_dir / video_id
        
        try:
            stats = process_single_video(
                video_path=video_path,
                pose_csv_path=csv_path,
                output_dir=output_video_dir,
                seg_model=seg_model,
                silhouette_size=silhouette_size,
                masked_rgb_size=masked_rgb_size,
                padding_ratio=padding_ratio,
                conf_threshold=conf_threshold,
                recorded_size=recorded_size,
                skip_if_exists=skip_if_exists,
            )
            
            logger.info(f"  {video_id}: {stats['total_frames']} frames, "
                       f"{stats['detected_frames']} detected, "
                       f"{stats['pose_frames']} with pose")
        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {e}")


def test_single_frame(
    video_path: str,
    csv_path: str,
    frame_idx: int = 0,
    output_path: str = "test_output.png"
):
    """
    Test function to visualize keypoints and white patch on a single frame.
    Useful for debugging.
    
    Usage:
        python data_generator.py --test --video PATH --csv PATH --frame 0
    """
    print("=" * 70)
    print("Testing keypoint visualization")
    print("=" * 70)
    
    # Load video frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {total_frames} frames")
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Cannot read frame {frame_idx}")
        return
    
    # Load pose data
    pose_df = load_pose_data(Path(csv_path))
    if pose_df is None:
        print(f"Error: Cannot load CSV {csv_path}")
        return
    
    print(f"CSV: {len(pose_df)} rows")
    
    # Get keypoints
    keypoints = get_keypoints_by_row_index(pose_df, frame_idx)
    
    if keypoints is None:
        print(f"Error: No keypoints for row {frame_idx}")
        return
    
    print(f"\nKeypoints for frame {frame_idx}:")
    kpt_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    for i, name in enumerate(kpt_names):
        print(f"  {name}: ({keypoints[i, 0]:.1f}, {keypoints[i, 1]:.1f})")
    
    # Scale keypoints if needed (use default recorded size)
    recorded_size = DEFAULT_RECORDED_SIZE
    rec_w, rec_h = recorded_size
    if width != rec_w or height != rec_h:
        print(f"\nScaling keypoints from {recorded_size} to ({width}, {height})")
        keypoints = scale_keypoints(keypoints, recorded_size, (width, height))
    
    # Draw keypoints on frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_debug = frame_rgb.copy()
    
    # Draw all keypoints
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            color = (255, 0, 0) if i < 5 else (0, 255, 0)  # Red for head, green for body
            cv2.circle(frame_debug, (int(x), int(y)), 5, color, -1)
            cv2.putText(frame_debug, str(i), (int(x)+5, int(y)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Apply white patch
    frame_masked = apply_head_white_patch(frame_rgb, keypoints)
    
    # Create side-by-side comparison
    comparison = np.hstack([frame_debug, frame_masked])
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    
    # Save output
    cv2.imwrite(output_path, comparison_bgr)
    print(f"\nSaved test output to: {output_path}")
    print("Left: keypoints visualization, Right: with white patch")


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Dataset with Silhouettes and Masked RGB Frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all classes with defaults
  python data_generator.py
  
  # Process single class
  python data_generator.py --class_name normal
  
  # Use custom data root
  python data_generator.py --data_root /path/to/data --output_root /path/to/output
  
  # Use config file
  python data_generator.py --config config.json
  
  # Test mode
  python data_generator.py --test --video video.mp4 --csv pose.csv --frame 0
        """
    )
    
    # Configuration arguments
    parser.add_argument("--config", type=str, default=None,
                       help="Path to JSON configuration file")
    parser.add_argument("--data_root", type=str, default=None,
                       help="Root directory containing data (overrides config)")
    parser.add_argument("--output_root", type=str, default=None,
                       help="Root directory for output (overrides config)")
    
    # Processing arguments
    parser.add_argument("--class_name", type=str, default=None,
                       help="Process single class (e.g., normal, single_thoracic, double, single_lumbar)")
    parser.add_argument("--model", type=str, default="yolo11x-seg.pt",
                       help="YOLO segmentation model name")
    parser.add_argument("--silhouette_size", type=int, nargs=2, default=None,
                       help="Silhouette output size (height width)")
    parser.add_argument("--padding_ratio", type=float, default=None,
                       help="Padding ratio around silhouette (default: 0.05)")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                       help="Confidence threshold for detection (default: 0.5)")
    parser.add_argument("--recorded_size", type=int, nargs=2, default=None,
                       help="Original video resolution for keypoint scaling (width height)")
    parser.add_argument("--max_videos", type=int, default=None,
                       help="Maximum number of videos to process per class")
    parser.add_argument("--no_skip", action="store_true",
                       help="Don't skip already processed videos")
    
    # Test mode arguments
    parser.add_argument("--test", action="store_true",
                       help="Run test mode to visualize keypoints on single frame")
    parser.add_argument("--video", type=str, default=None,
                       help="Video path for test mode")
    parser.add_argument("--csv", type=str, default=None,
                       help="CSV path for test mode")
    parser.add_argument("--frame", type=int, default=0,
                       help="Frame index for test mode")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(Path(args.config) if args.config else None)
    
    # Override with command line arguments
    data_root = Path(args.data_root) if args.data_root else Path(config['data_root'])
    output_root = Path(args.output_root) if args.output_root else Path(config['output_root'])
    class_definitions = config.get('class_definitions', DEFAULT_CLASS_DEFINITIONS)
    
    silhouette_size = tuple(args.silhouette_size) if args.silhouette_size else tuple(config['silhouette_size'])
    masked_rgb_size = tuple(args.silhouette_size) if args.silhouette_size else tuple(config['masked_rgb_size'])
    padding_ratio = args.padding_ratio if args.padding_ratio is not None else config.get('padding_ratio', DEFAULT_PADDING_RATIO)
    recorded_size = tuple(args.recorded_size) if args.recorded_size else tuple(config.get('recorded_size', DEFAULT_RECORDED_SIZE))
    
    skip_if_exists = not args.no_skip
    
    # Validate paths
    if not data_root.exists():
        logger.error(f"Data root does not exist: {data_root}")
        exit(1)
    
    if args.test:
        if args.video and args.csv:
            test_single_frame(
                video_path=args.video,
                csv_path=args.csv,
                frame_idx=args.frame
            )
        else:
            logger.error("Test mode requires --video and --csv arguments")
            logger.error("Example: python data_generator.py --test --video path/to/video.mp4 --csv path/to/pose.csv --frame 0")
    elif args.class_name:
        process_single_class(
            class_name=args.class_name,
            data_root=data_root,
            output_root=output_root,
            class_definitions=class_definitions,
            model_name=args.model,
            silhouette_size=silhouette_size,
            masked_rgb_size=masked_rgb_size,
            padding_ratio=padding_ratio,
            conf_threshold=args.conf_threshold,
            recorded_size=recorded_size,
            max_videos=args.max_videos,
            skip_if_exists=skip_if_exists,
        )
    else:
        process_all_classes(
            data_root=data_root,
            output_root=output_root,
            class_definitions=class_definitions,
            model_name=args.model,
            silhouette_size=silhouette_size,
            masked_rgb_size=masked_rgb_size,
            padding_ratio=padding_ratio,
            conf_threshold=args.conf_threshold,
            recorded_size=recorded_size,
            skip_if_exists=skip_if_exists,
        )
