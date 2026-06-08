"""
Data sampler for SigLIP pretraining framework.
Handles loading and patching video and knowledge map pairs with non-overlapped 300-frame chunks.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict, Any
import cv2
import pickle
import hashlib
import json
import re
from pathlib import Path
import glob

# Add utils directory to path for knowledge_map import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from knowledge_map import GetAllFeatures
except ImportError:
    # Try relative import as fallback
    from .knowledge_map import GetAllFeatures


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Handle numpy 2.x pickle module path when running with numpy 1.x."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pickle_compat(path: str) -> Any:
    """Load pickle with numpy module-path compatibility fallback."""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if "numpy._core" not in str(e):
                raise
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


def _resolve_patch_pkl_path(
    raw_path: str,
    pkl_data_dir: str,
    patch_id: Optional[str] = None,
) -> str:
    """
    Resolve a patch PKL path from metadata.

    Metadata may store Windows-style relative paths (``.\\patches\\...``); on Linux
    those fail unless normalized. Falls back to ``<pkl_data_dir>/patches/patch_<id>.pkl``.
    """
    pkl_data_dir_abs = os.path.abspath(os.path.normpath(pkl_data_dir))
    raw = str(raw_path or "").replace("\\", "/")
    candidate = ""
    if raw:
        if os.path.isabs(raw):
            candidate = os.path.normpath(raw)
        else:
            candidate = os.path.normpath(os.path.abspath(raw))
        if os.path.isfile(candidate):
            return candidate

    pid = patch_id
    if pid is None and raw:
        pid = Path(raw).stem.replace("patch_", "")
    if pid is not None:
        fallback = os.path.join(pkl_data_dir_abs, "patches", f"patch_{pid}.pkl")
        if os.path.isfile(fallback):
            return fallback

    if candidate:
        return candidate
    if pid is not None:
        return os.path.join(pkl_data_dir_abs, "patches", f"patch_{pid}.pkl")
    return raw


def _to_float_tensor(arr: Any) -> torch.Tensor:
    """
    Convert array-like to float tensor.
    Falls back to list conversion when torch numpy bridge is unavailable.
    """
    try:
        return torch.from_numpy(arr).float()
    except RuntimeError as e:
        if "Numpy is not available" not in str(e):
            raise
        return torch.tensor(np.asarray(arr).tolist(), dtype=torch.float32)


def _to_long_tensor(arr: Any) -> torch.Tensor:
    """Convert array-like to long tensor with numpy-bridge fallback."""
    if isinstance(arr, np.ndarray):
        try:
            return torch.from_numpy(arr).long()
        except RuntimeError as e:
            if "Numpy is not available" not in str(e):
                raise
            return torch.tensor(arr.tolist(), dtype=torch.long)
    return torch.tensor(arr, dtype=torch.long)


def _count_rows_fast(csv_path: str) -> int:
    """
    Fast line count for CSV (skips header). Cheaper than pandas for large files.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with path.open("rb") as f:
        # Count newlines in chunks
        line_count = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1024 * 1024), b""))
    # Subtract header if present
    return max(line_count - 1, 0)


def pad_to_multiple(data: np.ndarray, multiple: int, mode: str = 'zero') -> np.ndarray:
    """
    Pad data to be a multiple of the specified value.
    
    Args:
        data: Input array of shape (length, ...) - can be 2D (length, features) or 4D (length, H, W, C)
        multiple: Target multiple (default 300)
        mode: Padding mode - 'repeat' (repeat last frame) or 'zero' (zero padding)
    
    Returns:
        Padded array with length that is a multiple of `multiple`
    """
    length = len(data)
    remainder = length % multiple
    
    if remainder == 0:
        return data
    
    pad_length = multiple - remainder
    
    if mode == 'repeat':
        # Repeat the last frame/row
        if len(data.shape) == 2:
            # 2D: (length, features)
            padding = np.tile(data[-1:], (pad_length, 1))
        elif len(data.shape) == 4:
            # 4D: (length, H, W, C)
            padding = np.tile(data[-1:], (pad_length, 1, 1, 1))
        else:
            # Generic case: repeat along first dimension
            padding = np.tile(data[-1:], (pad_length,) + (1,) * (len(data.shape) - 1))
    elif mode == 'zero':
        # Zero padding
        if len(data.shape) == 2:
            padding = np.zeros((pad_length, data.shape[1]), dtype=data.dtype)
        elif len(data.shape) == 4:
            padding = np.zeros((pad_length, data.shape[1], data.shape[2], data.shape[3]), dtype=data.dtype)
        else:
            padding_shape = (pad_length,) + data.shape[1:]
            padding = np.zeros(padding_shape, dtype=data.dtype)
    else:
        raise ValueError(f"Unknown padding mode: {mode}")
    
    padded_data = np.concatenate([data, padding], axis=0)
    return padded_data


def is_valid_patch(patch: np.ndarray, threshold: float = 0.8) -> bool:
    """
    Check if a patch is valid by filtering out patches with too many zeros or repeating values.
    
    Args:
        patch: Patch array of shape (patch_size, ...)
        threshold: Threshold for filtering (default 0.8 = 80%)
    
    Returns:
        True if patch is valid (should be kept), False if should be filtered out
    """
    # Flatten patch to check all values
    flat_patch = patch.flatten()
    total_values = len(flat_patch)
    
    if total_values == 0:
        return False
    
    # Check percentage of zeros
    zero_count = np.sum(flat_patch == 0)
    zero_ratio = zero_count / total_values
    if zero_ratio > threshold:
        return False
    
    # Check for repeating values: count consecutive identical values along temporal dimension
    if len(patch.shape) > 1:
        # For multi-dimensional patches, check along first dimension (temporal)
        # Compare each frame with the previous one
        if patch.shape[0] > 1:
            # Flatten spatial dimensions for comparison
            patch_flat = patch.reshape(patch.shape[0], -1)
            # Check if consecutive frames are identical
            diff = np.diff(patch_flat, axis=0)
            repeating_frames = np.sum(np.all(diff == 0, axis=1))
            repeating_ratio = repeating_frames / (patch.shape[0] - 1)
        else:
            repeating_ratio = 0
    else:
        # For 1D patches, check consecutive identical values
        if len(patch) > 1:
            diff = np.diff(patch)
            repeating_count = np.sum(diff == 0)
            repeating_ratio = repeating_count / (len(patch) - 1)
        else:
            repeating_ratio = 0
    
    if repeating_ratio > threshold:
        return False
    
    return True


def create_non_overlapping_patches(data: np.ndarray, patch_size: int, filter_patches: bool = True) -> List[np.ndarray]:
    """
    Create non-overlapping patches from data.
    
    Args:
        data: Input array of shape (length, features) where length is a multiple of patch_size
        patch_size: Size of each patch
        filter_patches: If True, filter out patches with >80% zeros or repeating values
    
    Returns:
        List of patches, each of shape (patch_size, features)
    """
    length = len(data)
    assert length % patch_size == 0, f"Data length ({length}) must be a multiple of patch_size ({patch_size})"
    
    num_patches = length // patch_size
    patches = []
    
    for i in range(num_patches):
        start_idx = i * patch_size
        end_idx = start_idx + patch_size
        patch = data[start_idx:end_idx]
        
        # Filter out invalid patches
        if filter_patches and not is_valid_patch(patch):
            continue
        
        patches.append(patch)
    
    return patches


def sample_frames_evenly(video_patch: np.ndarray, num_frames: int) -> np.ndarray:
    """
    Sample frames evenly from a video patch.
    
    Args:
        video_patch: Video patch of shape (patch_size, H, W, C) or (patch_size, ...)
        num_frames: Number of frames to sample
    
    Returns:
        Sampled video of shape (num_frames, H, W, C) or (num_frames, ...)
    """
    patch_size = len(video_patch)
    
    if patch_size == num_frames:
        return video_patch
    
    # Calculate indices for even sampling
    indices = np.linspace(0, patch_size - 1, num_frames, dtype=int)
    
    return video_patch[indices]


def augment_knowledge_map_gaussian_noise(
    km_data: np.ndarray,
    noise_std: float = 0.01,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian noise to knowledge map features.
    
    Args:
        km_data: Knowledge map array of shape (timesteps, features)
        noise_std: Standard deviation of Gaussian noise (default 0.01 = 1%)
        seed: Random seed for reproducibility
    
    Returns:
        Augmented knowledge map with same shape
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(0, noise_std, km_data.shape)
    return km_data + noise


def load_video_frames(video_path: str, max_frames: Optional[int] = None, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load video frames from a video file (optimized version).
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load (None for all)
        target_size: Optional (width, height) to resize frames
    
    Returns:
        Array of shape (num_frames, height, width, channels)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get total frame count for pre-allocation if possible
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    # Pre-allocate array if we know the size
    frames = []
    frame_count = 0
    
    # Optimize: read frames more efficiently
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames is not None and frame_count >= max_frames:
            break
        
        # Resize if target size is specified (do this before converting to RGB for efficiency)
        if target_size is not None:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames loaded from video: {video_path}")
    
    return np.array(frames, dtype=np.uint8)


def get_video_frame_count(video_path: str) -> int:
    """
    Get video frame count with caching to avoid repeated opens.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Total number of frames in video
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return frame_count


def load_video_frames_range(video_path: str, start_frame: int, end_frame: int, 
                            target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load a specific range of frames from a video file (optimized version).
    
    Optimizations:
    - Pre-allocates numpy array instead of list appending
    - Batch resize operations
    
    Args:
        video_path: Path to video file
        start_frame: Starting frame index
        end_frame: Ending frame index (exclusive)
        target_size: Optional (width, height) to resize frames
    
    Returns:
        Array of shape (num_frames, height, width, channels)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    num_frames = end_frame - start_frame
    
    # Load video (no cache)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Clamp end_frame to available frames
    end_frame = min(end_frame, total_frames)
    num_frames = end_frame - start_frame
    
    if num_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid frame range: start={start_frame}, end={end_frame}")
    
    # Determine final dimensions
    if target_size is not None:
        final_height, final_width = target_size[1], target_size[0]
    else:
        final_height, final_width = height, width
    
    # Pre-allocate array for better performance
    frames = np.zeros((num_frames, final_height, final_width, 3), dtype=np.uint8)
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_idx = 0
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize if needed
        if target_size is not None:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB and store directly in pre-allocated array
        frames[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_idx += 1
    
    cap.release()
    
    if frame_idx == 0:
        raise ValueError(f"No frames loaded from video: {video_path}")
    
    # Trim array if fewer frames were read
    if frame_idx < num_frames:
        frames = frames[:frame_idx]
    
    return frames


def load_knowledge_map(csv_path: str) -> np.ndarray:
    """
    Load and process knowledge map from CSV file (with optimized reading).
    
    Optimizations:
    - Uses pandas C engine for faster CSV reading
    
    Args:
        csv_path: Path to CSV file containing pose data
    
    Returns:
        Processed knowledge map array of shape (num_rows, features)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Use C engine for faster CSV reading (if available)
    try:
        df = pd.read_csv(csv_path, engine='c')
    except:
        # Fallback to Python engine if C engine fails
        df = pd.read_csv(csv_path, engine='python')
    
    if df.shape[0] == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")
    
    # Drop NaN values
    df = df.dropna()
    
    # Get values without frame index (assuming last column is frame index)
    values = df.values[:, :-1] if df.shape[1] > 1 else df.values
    
    # Process through GetAllFeatures to get knowledge map
    if len(values) == 0:
        raise ValueError(f"No valid data in CSV: {csv_path}")
    
    # Process through knowledge map extraction
    knowledge_map = GetAllFeatures(values)
    
    return knowledge_map


def extract_subject_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract subject ID from CSV or video filename.
    Assumes the filename contains a numeric subject ID.
    
    Args:
        filename: Name of the CSV or video file
    
    Returns:
        Subject ID as string, or None if not found
    """
    # Remove extension
    base_name = os.path.splitext(filename)[0]
    
    # Try to find numeric ID - common patterns:
    # - "subject_1.csv" -> "1"
    # - "1_forward.csv" -> "1"
    # - "subj001.csv" -> "1" (normalized)
    # - "1.csv" -> "1"
    
    # Pattern 1: Direct numeric at start or after underscore
    match = re.search(r'(?:^|_)(\d+)(?:_|$)', base_name)
    if match:
        return match.group(1)
    
    # Pattern 2: Any sequence of digits
    match = re.search(r'\d+', base_name)
    if match:
        return match.group(0)
    
    return None


def load_prompts_json(prompts_path: str) -> Dict:
    """
    Load prompts from JSON file.
    
    Args:
        prompts_path: Path to individual_gait_prompts.json
    
    Returns:
        Dictionary containing prompts
    """
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    
    with open(prompts_path, 'r', encoding='utf-8-sig') as f:
        prompts_data = json.load(f)
    
    return prompts_data


def get_prompts_for_sample(
    prompts_data: Dict,
    subject_id: str,
    direction: str,
    prompt_selection: str = 'top_feature_prompts'
) -> List[str]:
    """
    Get prompts for a specific subject and direction.
    
    Args:
        prompts_data: Loaded prompts JSON data
        subject_id: Subject ID as string (e.g., "1", "10")
        direction: Direction string ('going_forward' or 'going_backward')
        prompt_selection: Selection key ('forward', 'backward', 'both', 'top_feature_prompts', 'auto', 'concise_prompts')
                         If 'auto', automatically uses direction-specific prompts based on the file's direction
                         If 'concise_prompts', uses general concise prompts (not subject-specific)
    
    Returns:
        List of prompt strings
    """
    # Handle general prompts (concise_prompts) - not subject-specific
    if prompt_selection == 'concise_prompts':
        # Check if prompts_data has concise_prompts at top level
        if 'concise_prompts' in prompts_data:
            concise_prompts = prompts_data['concise_prompts']
            if isinstance(concise_prompts, list):
                return concise_prompts
            else:
                return []
        else:
            # Fallback: return empty list if concise_prompts not found
            return []
    
    # Map direction to prompt key
    direction_map = {
        'going_forward': 'forward',
        'going_backward': 'backward'
    }
    
    # Get subject prompts
    subject_prompts = prompts_data.get('subject_prompts', {})
    if subject_id not in subject_prompts:
        return []
    
    subject_data = subject_prompts[subject_id]
    
    # If prompt_selection is 'auto', use direction-specific prompts based on file direction
    if prompt_selection == 'auto' and direction is not None:
        direction_key = direction_map.get(direction)
        if direction_key:
            return subject_data.get(direction_key, [])
        else:
            # Fallback to top_feature_prompts if direction not recognized
            return subject_data.get('top_feature_prompts', [])
    # If prompt_selection is 'forward' or 'backward', use direction-specific prompts
    elif prompt_selection in ['forward', 'backward']:
        return subject_data.get(prompt_selection, [])
    elif prompt_selection == 'both':
        return subject_data.get('both', [])
    elif prompt_selection == 'top_feature_prompts':
        return subject_data.get('top_feature_prompts', [])
    else:
        # Default to top_feature_prompts
        return subject_data.get('top_feature_prompts', [])


def get_paired_samples(
    table_dir: str,
    video_dir: str,
    direction: str = 'going_backward'
) -> List[Tuple[str, str]]:
    """
    Get paired video and knowledge map file paths.
    
    Args:
        table_dir: Directory containing knowledge map CSV files
        video_dir: Directory containing video files
        direction: Subdirectory name ('going_backward' or 'going_forward')
    
    Returns:
        List of tuples (csv_path, video_path) for paired samples
    """
    table_path = os.path.join(table_dir, direction)
    video_path = os.path.join(video_dir, direction)
    
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"Table directory not found: {table_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video directory not found: {video_path}")
    
    # Get all CSV files in table directory
    csv_files = [f for f in os.listdir(table_path) if f.endswith('.csv')]
    
    paired_samples = []
    
    for csv_file in csv_files:
        csv_path = os.path.join(table_path, csv_file)
        
        # Try to find corresponding video file
        # Assuming video files have same base name but different extension
        base_name = os.path.splitext(csv_file)[0]
        
        # Try common video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_file = None
        
        for ext in video_extensions:
            potential_video = base_name + ext
            video_file_path = os.path.join(video_path, potential_video)
            if os.path.exists(video_file_path):
                video_file = video_file_path
                break
        
        if video_file is not None:
            paired_samples.append((csv_path, video_file))
        else:
            print(f"Warning: No corresponding video found for {csv_file}")
    
    return paired_samples


class SigLIPPretrainDataset(Dataset):
    """
    Dataset for SigLIP pretraining with paired video and knowledge map data.
    Handles inconsistent lengths by padding to multiples of 300 and creating non-overlapped patches.
    """
    
    def __init__(
        self,
        table_dir: str,
        video_dir: str,
        directions: List[str] = ['going_backward', 'going_forward'],
        patch_size: int = 300,
        video_frame_count: Optional[int] = None,
        pad_mode: str = 'zero',
        max_samples_per_file: Optional[int] = None,
        video_target_size: Optional[Tuple[int, int]] = None,
        prompts_path: Optional[str] = None,
        prompt_selection: str = 'top_feature_prompts'
    ):
        """
        Initialize SigLIP pretraining dataset.
        
        Args:
            table_dir: Base directory containing knowledge map CSV files
            video_dir: Base directory containing video files
            directions: List of direction subdirectories to include
            patch_size: Size of each patch for knowledge map (default 300)
            video_frame_count: Number of frames to sample from video patch (None = use patch_size)
            pad_mode: Padding mode - 'repeat' or 'zero'
            max_samples_per_file: Maximum number of patches per file (None for all)
            video_target_size: Optional (width, height) to resize video frames
            prompts_path: Path to individual_gait_prompts.json file (optional)
            prompt_selection: Selection key for prompts ('forward', 'backward', 'both', 'top_feature_prompts')
        """
        self.table_dir = table_dir
        self.video_dir = video_dir
        self.directions = directions
        self.patch_size = patch_size
        self.video_frame_count = video_frame_count if video_frame_count is not None else patch_size
        self.pad_mode = pad_mode
        self.max_samples_per_file = max_samples_per_file
        self.video_target_size = video_target_size
        self.prompts_path = prompts_path
        self.prompt_selection = prompt_selection
        
        # Load prompts if provided
        self.prompts_data = None
        if prompts_path is not None:
            try:
                self.prompts_data = load_prompts_json(prompts_path)
                print(f"Loaded prompts from: {prompts_path}")
                print(f"  Prompt selection: {prompt_selection}")
            except Exception as e:
                print(f"Warning: Could not load prompts: {e}")
                self.prompts_data = None
        
        # Load all paired samples
        self.samples = []
        for direction in directions:
            try:
                paired = get_paired_samples(table_dir, video_dir, direction)
                self.samples.extend(paired)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
        
        if len(self.samples) == 0:
            raise ValueError("No paired samples found!")
        
        print(f"Found {len(self.samples)} paired samples")
        
        # Store patch indices instead of pre-loading all patches (lazy loading)
        self.patch_indices = []
        self._create_patch_indices()
        
        print(f"Created {len(self.patch_indices)} patch indices")
        print(f"  Knowledge map patch size: {patch_size}")
        print(f"  Video frame count: {self.video_frame_count}")
    
    def _create_patch_indices(self):
        """Pre-process samples and create patch indices (lazy loading approach - no data loading)."""
        for csv_path, video_path in self.samples:
            try:
                # Fast row count (no full pandas load) to approximate km length
                km_length = _count_rows_fast(csv_path)
                
                # Get video frame count without loading all frames
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue
                video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Ensure they have the same length (take minimum)
                min_length = min(km_length, video_frame_count)
                
                if min_length < self.patch_size:
                    continue
                
                # Extract subject ID and direction from file paths
                csv_filename = os.path.basename(csv_path)
                subject_id = extract_subject_id_from_filename(csv_filename)
                
                # Determine direction from path
                direction = None
                for dir_name in self.directions:
                    if dir_name in csv_path:
                        direction = dir_name
                        break

                # Special case: sequences that are roughly twice the patch_size (e.g., 600 vs 300 frames).
                # These typically come from higher-fps recordings of the same physical duration.
                # For such files, we keep a SINGLE clip [0:min_length) and will downsample it
                # evenly to patch_size in __getitem__, so that the temporal coverage matches
                # 30 FPS data while reusing the existing 300-frame processing logic.
                if self.patch_size < min_length <= 2 * self.patch_size:
                    # Respect max_samples_per_file if set (at most 1 in this branch)
                    if self.max_samples_per_file is None or self.max_samples_per_file > 0:
                        self.patch_indices.append({
                            'csv_path': csv_path,
                            'video_path': video_path,
                            'start_idx': 0,
                            'end_idx': min_length,
                            'source_file': csv_filename,
                            'subject_id': subject_id,
                            'direction': direction
                        })
                    continue
                
                # Default: split into non-overlapping 300-frame chunks
                num_patches = min_length // self.patch_size
                
                # Store patch indices for this file
                for patch_idx in range(num_patches):
                    start_idx = patch_idx * self.patch_size
                    end_idx = start_idx + self.patch_size
                    
                    self.patch_indices.append({
                        'csv_path': csv_path,
                        'video_path': video_path,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'source_file': csv_filename,
                        'subject_id': subject_id,
                        'direction': direction
                    })
                    
                    # Limit patches per file if specified
                    if self.max_samples_per_file is not None:
                        file_patches = sum(1 for p in self.patch_indices if p['source_file'] == csv_filename)
                        if file_patches >= self.max_samples_per_file:
                            break
            
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
                continue
    
    def __len__(self):
        return len(self.patch_indices)
    
    def __getitem__(self, idx):
        """Lazy loading: load data on-demand."""
        patch_info = self.patch_indices[idx]
        csv_path = patch_info['csv_path']
        video_path = patch_info['video_path']
        start_idx = patch_info['start_idx']
        end_idx = patch_info['end_idx']
        
        # Load knowledge map
        km_data = load_knowledge_map(csv_path)
        
        # Extract knowledge map patch
        km_patch = km_data[start_idx:end_idx]
        
        # Validate knowledge map patch
        if not is_valid_patch(km_patch):
            # If invalid, try to get a valid patch by shifting
            # This is a fallback - ideally patches should be pre-validated
            pass
        
        # Load video frames for this patch range
        video_patch = load_video_frames_range(
            video_path, 
            start_idx, 
            end_idx, 
            target_size=self.video_target_size
        )
        
        # Validate video patch
        if not is_valid_patch(video_patch):
            # Fallback: try loading full video and extracting patch
            video_frames = load_video_frames(video_path, target_size=self.video_target_size)
            min_length = min(len(km_data), len(video_frames))
            video_frames = video_frames[:min_length]
            video_padded = pad_to_multiple(video_frames, self.patch_size, self.pad_mode)
            video_patches = create_non_overlapping_patches(video_padded, self.patch_size, filter_patches=False)
            patch_idx = start_idx // self.patch_size
            if patch_idx < len(video_patches):
                video_patch = video_patches[patch_idx]
            else:
                # Last resort: use the patch we already have
                pass
        
        # Ensure same length
        min_len = min(len(km_patch), len(video_patch))
        km_patch = km_patch[:min_len]
        video_patch = video_patch[:min_len]
        
        # Base indices for original timeline (before padding/sampling)
        km_indices = np.arange(start_idx, start_idx + len(km_patch))
        video_indices = np.arange(start_idx, start_idx + len(video_patch))

        # If this clip is longer than patch_size (e.g., ~600 frames for 60 Hz),
        # evenly sample it down to exactly patch_size steps so that higher‑FPS
        # recordings cover the same physical duration as 30 FPS recordings.
        original_length = len(km_patch)
        if original_length > self.patch_size:
            sample_idx = np.linspace(0, original_length - 1, self.patch_size, dtype=int)
            km_patch = km_patch[sample_idx]
            video_patch = video_patch[sample_idx]
            km_indices = km_indices[sample_idx]
            video_indices = video_indices[sample_idx]
        
        # Pad if needed (use -1 for padded indices to mark non-real timesteps)
        if len(km_patch) < self.patch_size:
            pad_len = self.patch_size - len(km_patch)
            km_patch = pad_to_multiple(km_patch, self.patch_size, self.pad_mode)
            km_indices = np.concatenate([km_indices, np.full(pad_len, -1, dtype=int)])
        if len(video_patch) < self.patch_size:
            pad_len = self.patch_size - len(video_patch)
            video_patch = pad_to_multiple(video_patch, self.patch_size, self.pad_mode)
            video_indices = np.concatenate([video_indices, np.full(pad_len, -1, dtype=int)])
        
        # Sample frames from video patch if needed, and record sampled indices
        if self.video_frame_count != self.patch_size and len(video_patch) == self.patch_size:
            sampled_idx = np.linspace(0, self.patch_size - 1, self.video_frame_count, dtype=int)
            video_patch = video_patch[sampled_idx]
            video_indices = video_indices[sampled_idx]
        
        # Convert to tensors
        knowledge_map = torch.from_numpy(km_patch).float()
        video = torch.from_numpy(video_patch).float()
        km_indices_tensor = torch.from_numpy(km_indices).long()
        video_indices_tensor = torch.from_numpy(video_indices).long()
        
        # Get prompts if available
        prompts = []
        if self.prompts_data is not None and patch_info.get('subject_id') is not None:
            subject_id = patch_info['subject_id']
            direction = patch_info.get('direction')
            prompts = get_prompts_for_sample(
                self.prompts_data,
                subject_id,
                direction,
                self.prompt_selection
            )
        
        return {
            'knowledge_map': knowledge_map,  # (patch_size, km_features)
            'video': video,  # (video_frame_count, H, W, C)
            'source_file': patch_info['source_file'],
            'prompts': prompts,  # List of prompt strings
            'km_indices': km_indices_tensor,  # (patch_size,) with -1 for padded steps
            'video_indices': video_indices_tensor  # (video_frame_count,) aligned to video
        }


class SigLIPPretrainDatasetPKL(Dataset):
    """
    Dataset for SigLIP pretraining that loads preprocessed data from pkl files.
    Much faster than on-the-fly processing.
    """
    
    def __init__(
        self,
        pkl_data_dir: str,
        metadata_path: Optional[str] = None,
        prompts_path: Optional[str] = None,
        prompt_selection: str = "top_feature_prompts",
        km_gaussian_noise_std: Optional[float] = None,
    ):
        """
        Initialize SigLIP pretraining dataset from pkl files.
        
        Args:
            pkl_data_dir: Directory containing preprocessed pkl files
            metadata_path: Path to patch_metadata.pkl file (if None, will look in pkl_data_dir)
            prompts_path: Path to prompts JSON file (optional, can also be read from config)
            prompt_selection: Prompt selection method (optional, can also be read from config)
            km_gaussian_noise_std: Standard deviation for Gaussian noise augmentation (None to disable)
        """
        self.pkl_data_dir = os.path.abspath(os.path.normpath(pkl_data_dir))
        self.km_gaussian_noise_std = km_gaussian_noise_std
        
        # Load metadata
        if metadata_path is None:
            metadata_path = os.path.join(self.pkl_data_dir, 'patch_metadata.pkl')
        
        if not os.path.exists(metadata_path):
            # Fallback: build metadata from existing patch pkl files
            patches_dir = os.path.join(self.pkl_data_dir, 'patches')
            patch_files = sorted(glob.glob(os.path.join(patches_dir, '*.pkl')))
            if len(patch_files) == 0:
                raise FileNotFoundError(f"Metadata file not found and no patch files in: {patches_dir}")
            print(f"Metadata not found. Building from {len(patch_files)} patch files...")
            patch_metadata = []
            for pkl_path in patch_files:
                try:
                    pkl_path = os.path.abspath(pkl_path)
                    with open(pkl_path, 'rb') as f:
                        patch_data = pickle.load(f)
                    patch_id = Path(pkl_path).stem.replace('patch_', '')
                    patch_metadata.append({
                        'patch_id': patch_id,
                        'pkl_path': pkl_path,
                        'subject_id': patch_data.get('subject_id', 'unknown'),
                        'direction': patch_data.get('direction', 'unknown'),
                        'source_file': patch_data.get('source_file', 'unknown'),
                        'start_idx': patch_data.get('start_idx', 0),
                        'end_idx': patch_data.get('end_idx', 0),
                        'knowledge_map_shape': patch_data['knowledge_map'].shape,
                        'video_shape': patch_data['video'].shape,
                        'num_prompts': len(patch_data.get('prompts', [])),
                    })
                except Exception as e:
                    print(f"Warning: failed to read {pkl_path}: {e}")
                    continue
            if len(patch_metadata) == 0:
                raise FileNotFoundError(f"Could not build metadata; all patch reads failed in: {patches_dir}")
            with open(metadata_path, 'wb') as f:
                pickle.dump({'patch_metadata': patch_metadata, 'config': {}}, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Metadata rebuilt and saved to: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.patch_metadata = metadata['patch_metadata']
        for entry in self.patch_metadata:
            entry['pkl_path'] = _resolve_patch_pkl_path(
                entry.get('pkl_path', ''),
                self.pkl_data_dir,
                entry.get('patch_id'),
            )
        self.config = metadata.get('config', {})
        
        # Get prompts_path and prompt_selection from args or config (mirror SigLIPFullGaitDatasetPKL)
        self.prompts_path = prompts_path or self.config.get('prompts_path')
        self.prompt_selection = prompt_selection or self.config.get('prompt_selection', 'top_feature_prompts')
        
        if prompts_path:
            print(f"📝 Using prompts_path from argument: {prompts_path}")
        elif self.config.get('prompts_path'):
            print(f"📝 Using prompts_path from config: {self.config.get('prompts_path')}")
        else:
            print(f"⚠️  No prompts_path provided (neither argument nor config)")
        
        if prompt_selection:
            print(f"📝 Using prompt_selection from argument: {prompt_selection}")
        elif self.config.get('prompt_selection'):
            print(f"📝 Using prompt_selection from config: {self.config.get('prompt_selection')}")
        else:
            print(f"📝 Using default prompt_selection: top_feature_prompts")
        
        # Load prompts if available
        self.prompts_data = None
        if self.prompts_path is not None:
            try:
                if not os.path.exists(self.prompts_path):
                    print(f"⚠️  Warning: Prompts file not found: {self.prompts_path}")
                    self.prompts_data = None
                else:
                    self.prompts_data = load_prompts_json(self.prompts_path)
                    print(f"✅ Loaded prompts from: {self.prompts_path}")
                    print(f"  Prompt selection: {self.prompt_selection}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load prompts: {e}")
                self.prompts_data = None
        else:
            print(f"⚠️  Warning: No prompts_path provided, prompts will be read from PKL only (if present)")
        
        print(f"Loaded {len(self.patch_metadata)} patches from pkl files")
        print(f"  Knowledge map patch size: {self.config.get('patch_size', 'unknown')}")
        print(f"  Video frame count: {self.config.get('video_frame_count', 'unknown')}")
    
    def __len__(self):
        return len(self.patch_metadata)
    
    def __getitem__(self, idx):
        """Load patch data from pkl file."""
        patch_info = self.patch_metadata[idx]
        pkl_path = patch_info['pkl_path']
        
        # Load patch data
        patch_data = _load_pickle_compat(pkl_path)
        
        # Convert numpy arrays to tensors
        knowledge_map = _to_float_tensor(patch_data['knowledge_map'])
        video = _to_float_tensor(patch_data['video'])
        source_file = patch_data.get('source_file', 'unknown')
        
        # Process prompts similar to SigLIPFullGaitDatasetPKL:
        # prefer dynamic prompts from JSON; fall back to prompts stored in PKL.
        prompts: List[str] = []
        if self.prompts_data is not None:
            subject_id = patch_data.get('subject_id')
            direction = patch_data.get('direction')
            if self.prompt_selection == 'concise_prompts':
                prompts = get_prompts_for_sample(
                    self.prompts_data,
                    "",
                    direction,
                    self.prompt_selection,
                )
            elif subject_id is not None:
                prompts = get_prompts_for_sample(
                    self.prompts_data,
                    subject_id,
                    direction,
                    self.prompt_selection,
                )
            if len(prompts) == 0 and idx < 3:
                print(f"⚠️  Warning: No prompts found for pretrain sample {idx}")
                print(f"   prompt_selection: {self.prompt_selection}")
                print(f"   subject_id: {subject_id}")
                print(f"   direction: {direction}")
        else:
            prompts = patch_data.get('prompts', [])

        # Optional KM Gaussian noise augmentation (like SigLIPFullGaitDatasetPKL)
        if self.km_gaussian_noise_std is not None and self.km_gaussian_noise_std > 0:
            noise = torch.randn_like(knowledge_map) * self.km_gaussian_noise_std
            knowledge_map = knowledge_map + noise

        # Optional indices (if stored in PKL) to mirror on-the-fly dataset outputs
        km_indices = None
        if 'km_indices' in patch_data and patch_data['km_indices'] is not None:
            km_indices = _to_long_tensor(patch_data['km_indices'])

        video_indices = None
        if 'video_indices' in patch_data and patch_data['video_indices'] is not None:
            video_indices = _to_long_tensor(patch_data['video_indices'])
        
        subject_id = patch_info.get('subject_id') or patch_data.get('subject_id')
        return {
            'knowledge_map': knowledge_map,  # (patch_size, km_features)
            'video': video,  # (video_frame_count, H, W, C)
            'source_file': source_file,
            'subject_id': str(subject_id) if subject_id is not None else source_file,
            'prompts': prompts,  # List of prompt strings
            'km_indices': km_indices,
            'video_indices': video_indices,
        }


def siglip_collate_fn(batch):
    """
    Collate function for SigLIP pretraining.
    Handles batching of patches.
    """
    knowledge_maps = []
    videos = []
    source_files = []
    subject_ids = []
    prompts_list = []
    km_indices_list = []
    video_indices_list = []
    
    for item in batch:
        knowledge_maps.append(item['knowledge_map'])
        videos.append(item['video'])
        source_files.append(item['source_file'])
        subject_ids.append(item.get('subject_id', item['source_file']))
        prompts_list.append(item.get('prompts', []))
        km_indices_list.append(item.get('km_indices'))
        video_indices_list.append(item.get('video_indices'))
    
    # Stack into batches
    knowledge_map_batch = torch.stack(knowledge_maps)  # (B, patch_size, km_features)
    video_batch = torch.stack(videos)  # (B, patch_size, C, H, W) or (B, patch_size, H, W, C)
    km_indices_batch = torch.stack(km_indices_list) if km_indices_list[0] is not None else None
    video_indices_batch = torch.stack(video_indices_list) if video_indices_list[0] is not None else None
    
    return {
        'knowledge_map': knowledge_map_batch,
        'video': video_batch,
        'source_files': source_files,
        'subject_ids': subject_ids,
        'prompts': prompts_list,  # List of lists of prompt strings
        'km_indices': km_indices_batch,
        'video_indices': video_indices_batch
    }


def get_paired_samples_fullgait(
    table_dir: str,
    video_dir: str,
) -> List[Tuple[str, str]]:
    """
    Get paired video and knowledge map file paths for fullgait dataset.
    Works with flat directory structure (no direction subdirectories).
    
    Handles naming differences:
    - CSV files: sz_{number}_step_1.csv (with underscore)
    - Video files: sz_{number}_step1.mp4 (without underscore)
    
    Args:
        table_dir: Directory containing knowledge map CSV files
        video_dir: Directory containing video files
    
    Returns:
        List of tuples (csv_path, video_path) for paired samples
    """
    if not os.path.exists(table_dir):
        raise FileNotFoundError(f"Table directory not found: {table_dir}")
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    
    # Get all CSV files in table directory
    csv_files = [f for f in os.listdir(table_dir) if f.endswith('.csv')]
    
    # Sort for reproducibility
    csv_files.sort()
    
    paired_samples = []
    
    for csv_file in csv_files:
        csv_path = os.path.join(table_dir, csv_file)
        
        # Convert CSV filename to video filename
        # sz_{number}_step_1.csv -> sz_{number}_step1.mp4
        base_name = os.path.splitext(csv_file)[0]
        # Replace step_1 with step1 (handle both _step_1 and step_1 patterns)
        video_base = base_name.replace('_step_1', '_step1').replace('step_1', 'step1')
        
        # Try common video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_file = None
        
        for ext in video_extensions:
            potential_video = video_base + ext
            video_file_path = os.path.join(video_dir, potential_video)
            if os.path.exists(video_file_path):
                video_file = video_file_path
                break
        
        # If exact match didn't work, try with original base name
        if video_file is None:
            for ext in video_extensions:
                potential_video = base_name + ext
                video_file_path = os.path.join(video_dir, potential_video)
                if os.path.exists(video_file_path):
                    video_file = video_file_path
                    break
        
        if video_file is not None:
            paired_samples.append((csv_path, video_file))
        else:
            print(f"Warning: No corresponding video found for {csv_file}")
    
    return paired_samples


def load_label_map(label_json_path: str, split: str) -> Dict[str, Any]:
    """
    Load label mapping from the train/test split JSON file.

    Args:
        label_json_path: Path to JSON file (train_indices.json or test_indices.json)
        split: Split key to load ('train' or 'test')

    Returns:
        Dict mapping subject/index (as string) to its label payload.
    """
    if not os.path.exists(label_json_path):
        raise FileNotFoundError(f"Label JSON not found: {label_json_path}")

    with open(label_json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if split not in data:
        raise KeyError(f"Split '{split}' not found in {label_json_path}")

    label_map: Dict[str, Any] = {}
    for entry in data[split]:
        idx = str(entry["index"])
        label_map[idx] = entry["label"]

    return label_map


class SigLIPFullGaitDataset_v2(Dataset):
    """
    SigLIP dataset for fullgait training with train/test mode support.
    
    Mode parameter controls sampling strategy:
    - Test mode: Uses full video length with even sampling (one sample per video) - consistent evaluation
    - Train mode: Splits videos into non-overlapping 96-frame chunks (multiple samples per video) - data augmentation
    - Label preservation: Uses sample_label_map to ensure all chunks inherit parent video's label
    - Augmentation: Only applies Gaussian noise augmentation in train mode (not in test mode)
    - Use case: Recommended for training/testing workflows where you need consistent test samples
               and want to maximize training data by chunking long videos
    
    Example: A 300-frame video will create:
             - Test mode: 1 sample (evenly sampled from all 300 frames to get 96 KM and 32 video timesteps)
             - Train mode: 3 samples (frames 0-95, 96-191, 192-287, skipping last 12 frames)
    """

    def __init__(
        self, 
        table_dir: str,
        video_dir: str,
        label_json_path: Optional[str] = None,
        split: Optional[str] = None,
        directions: Optional[List[str]] = None,  # Not used for fullgait, kept for compatibility
        patch_size: int = 96,
        video_frame_count: Optional[int] = 32,
        max_samples_per_file: Optional[int] = None,
        video_target_size: Optional[Tuple[int, int]] = None,
        prompts_path: Optional[str] = None,
        prompt_selection: str = "top_feature_prompts",
        binary_threshold: float = 11.0,
        km_gaussian_noise_std: Optional[float] = None,  # Gaussian noise std for knowledge map augmentation (None to disable)
        mode: str = "train",  # "train" or "test"
    ):
        self.table_dir = table_dir
        self.video_dir = video_dir
        # Directions not used for fullgait dataset (flat structure)
        self.directions = directions if directions is not None else []
        self.patch_size = patch_size  # 96 frames per chunk
        self.video_frame_count = video_frame_count if video_frame_count is not None else patch_size
        self.max_samples_per_file = max_samples_per_file
        self.video_target_size = video_target_size
        self.prompts_path = prompts_path
        self.prompt_selection = prompt_selection
        self.split = split
        self.binary_threshold = binary_threshold
        self.km_gaussian_noise_std = km_gaussian_noise_std
        self.mode = mode.lower()  # "train" or "test"
        
        if self.mode not in ["train", "test"]:
            raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")

        # Optional labels
        self.label_map: Optional[Dict[str, Any]] = None
        if label_json_path is not None and split is not None:
            try:
                self.label_map = load_label_map(label_json_path, split)
                print(f"Loaded {len(self.label_map)} labels for split '{split}' from {label_json_path}")
            except Exception as e:
                print(f"Warning: Could not load labels: {e}. Continuing without labels.")
                self.label_map = None
        else:
            print("No label JSON provided. Running in self-supervised mode.")

        # Prompts (optional)
        self.prompts_data = None
        if prompts_path is not None:
            try:
                self.prompts_data = load_prompts_json(prompts_path)
                print(f"Loaded prompts from: {prompts_path}")
                print(f"  Prompt selection: {prompt_selection}")
            except Exception as e:
                print(f"Warning: Could not load prompts: {e}")
                self.prompts_data = None

        # Pair files - use fullgait function for flat directory structure
        all_paired_samples: List[Tuple[str, str]] = []
        try:
            paired = get_paired_samples_fullgait(
                table_dir, video_dir
            )
            all_paired_samples.extend(paired)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

        if len(all_paired_samples) == 0:
            raise ValueError("No paired samples found!")

        print(f"Found {len(all_paired_samples)} total paired samples")

        # Filter samples to only include those with labels (if labels are provided)
        # Also create a mapping from csv_path to (subject_id, label, csv_filename) for use in _create_sample_indices
        # Optimized: Pre-compute all metadata in a single pass to avoid redundant operations
        self.samples: List[Tuple[str, str]] = []
        self.sample_label_map: Dict[str, Tuple[Optional[str], Any, str]] = {}  # csv_path -> (subject_id, label, csv_filename)
        if self.label_map is not None:
            print(f"Filtering samples to only include labeled subjects...")
            # Pre-compute subject_id extraction to avoid redundant regex operations
            for csv_path, video_path in all_paired_samples:
                csv_filename = os.path.basename(csv_path)
                subject_id = extract_subject_id_from_filename(csv_filename)
                if subject_id is not None:
                    subject_key = str(int(subject_id))
                    if subject_key in self.label_map:
                        label = self.label_map[subject_key]
                        self.samples.append((csv_path, video_path))
                        # Store mapping with filename to avoid re-extraction later
                        self.sample_label_map[csv_path] = (subject_id, label, csv_filename)
            print(f"Filtered to {len(self.samples)} samples with labels")
        else:
            # If no labels provided, use all samples and pre-compute filenames
            self.samples = all_paired_samples
            for csv_path, video_path in all_paired_samples:
                csv_filename = os.path.basename(csv_path)
                subject_id = extract_subject_id_from_filename(csv_filename)
                # Store mapping even without labels for consistency
                self.sample_label_map[csv_path] = (subject_id, None, csv_filename)
            print("No labels provided, using all paired samples")

        if len(self.samples) == 0:
            raise ValueError("No samples found after filtering by labels!")

        # Create sample indices based on mode
        self.sample_indices: List[Dict[str, Any]] = []
        self._create_sample_indices()

        if len(self.sample_indices) == 0:
            raise ValueError("No sample indices created; check data paths.")

        print(f"Created {len(self.sample_indices)} sample indices (mode: {self.mode})")
        print(f"  Knowledge map timesteps: {self.patch_size} (evenly sampled)")
        print(f"  Video timesteps: {self.video_frame_count} (evenly sampled)")

    def _create_sample_indices(self):
        """Pre-process samples and create sample indices based on mode.
        
        Optimizations:
        - Uses video metadata functions to get frame counts
        - Pre-computes sampling indices when possible
        - Uses pre-computed subject_id, label, and filename from filtering step
        - Uses counter dictionary for efficient chunk limiting
        
        - Test mode: One sample per video using evenly sampled 96 frames
        - Train mode: Multiple samples per video using non-overlapping 96-frame chunks
        """
        # Counter for chunk limiting (more efficient than linear search)
        file_chunk_counts: Dict[str, int] = {}
        
        for csv_path, video_path in self.samples:
            try:
                km_length = _count_rows_fast(csv_path)

                # Use video metadata function to get frame count
                try:
                    video_frame_count = get_video_frame_count(video_path)
                except Exception as e:
                    print(f"Warning: Could not get video frame count for {video_path}: {e}")
                    continue
                
                assert km_length == video_frame_count, f"km_length: {km_length}, video_frame_count: {video_frame_count}"
                min_length = min(km_length, video_frame_count)
                # Require minimum length (at least patch_size for train mode, or 32 for test mode)

                if min_length < 32:
                    continue
                # Get pre-computed metadata from filtering step (avoids redundant operations)
                if csv_path in self.sample_label_map:
                    subject_id, label_payload, csv_filename = self.sample_label_map[csv_path]
                else:
                    # Fallback (shouldn't happen, but safety check)
                    csv_filename = os.path.basename(csv_path)
                    subject_id = extract_subject_id_from_filename(csv_filename)
                    if self.label_map is not None and subject_id is not None:
                        subject_key = str(int(subject_id))
                        label_payload = self.label_map.get(subject_key, None)
                    else:
                        label_payload = None
                    if self.label_map is not None and label_payload is None:
                        continue  # Skip if labels required but not found

                # Fullgait dataset doesn't have directions
                direction = None

                if self.mode == "test":
                    # Test mode: Use full clip; precompute global output indices only.
                    # Relative indices for np.take() will be derived in __getitem__ (km_indices - start_idx).
                    start_idx = 0
                    
                    # Compute global output indices (km_indices/video_indices are always global)
                    if min_length >= self.patch_size:
                        km_indices = np.linspace(start_idx, start_idx + min_length - 1, self.patch_size, dtype=int)
                    else:
                        # Padding case: indices don't map to real data, use sequential from start_idx
                        km_indices = np.arange(start_idx, start_idx + self.patch_size, dtype=int)

                    if min_length >= self.video_frame_count:
                        video_indices = np.linspace(start_idx, start_idx + min_length - 1, self.video_frame_count, dtype=int)
                    else:
                        # Padding case: indices don't map to real data, use sequential from start_idx
                        video_indices = np.arange(start_idx, start_idx + self.video_frame_count, dtype=int)
                    
                    self.sample_indices.append(
                        {
                            "csv_path": csv_path,
                            "video_path": video_path,
                            "start_idx": start_idx,
                            "end_idx": min_length,  # Full clip for even sampling
                            "source_file": csv_filename,
                            "subject_id": subject_id,
                            "direction": direction,
                            "label": label_payload,
                            "km_indices": km_indices,  # Global output indices (len=patch_size)
                            "video_indices": video_indices,  # Global output indices (len=video_frame_count)
                            "original_length": min_length,  # Store for __getitem__
                        }
                    )
                else:  # train mode
                    # Train mode: Split into non-overlapping 96-frame chunks
                    # Skip unsampled rest frames
                    if min_length > self.patch_size:
                        num_chunks = min_length // self.patch_size
                    else:
                        continue
                    
                    # Initialize counter for this file if not exists
                    if csv_filename not in file_chunk_counts:
                        file_chunk_counts[csv_filename] = 0
                    
                    for chunk_idx in range(num_chunks):
                                                
                        start_idx = chunk_idx * self.patch_size
                        end_idx = start_idx + self.patch_size
                        
                        # Increment counter
                        file_chunk_counts[csv_filename] += 1
                        
                        # Check chunk limit using counter (O(1) lookup)
                        if self.max_samples_per_file is not None:
                            if file_chunk_counts[csv_filename] > self.max_samples_per_file:
                                break
                        
                        # For train mode chunks, we know exact length (patch_size)
                        # Pre-compute global output indices only.
                        # Relative indices for np.take() will be derived in __getitem__ (indices - start_idx).
                        chunk_length = self.patch_size
                        
                        # Final output indices (global timeline)
                        # For KM: chunk_length == patch_size, so no downsampling needed
                        if chunk_length >= self.patch_size:
                            km_indices = np.linspace(start_idx, start_idx + chunk_length - 1, self.patch_size, dtype=int)
                        else:
                            km_indices = np.arange(start_idx, start_idx + self.patch_size, dtype=int)
                        # For video: need to downsample from patch_size to video_frame_count
                        if chunk_length >= self.video_frame_count:
                            video_indices = np.linspace(start_idx, start_idx + chunk_length - 1, self.video_frame_count, dtype=int)
                        else:
                            video_indices = np.arange(start_idx, start_idx + self.video_frame_count, dtype=int)

                        self.sample_indices.append(
                            {
                                "csv_path": csv_path,
                                "video_path": video_path,
                                "start_idx": start_idx,
                                "end_idx": end_idx,
                                "source_file": csv_filename,
                                "subject_id": subject_id,
                                "direction": direction,
                                "label": label_payload,
                                "km_indices": km_indices,  # Global output indices (len=patch_size)
                                "video_indices": video_indices,  # Global output indices (len=video_frame_count)
                                "original_length": chunk_length,  # Store for __getitem__
                            }
                        )
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
                continue
    
    def __len__(self):
        return len(self.sample_indices)

    def _label_to_tensor(self, label_value: Any) -> Optional[torch.Tensor]:
        """Convert label payload to binary classification tensor based on threshold.
        
        Binary label is 1 if max(label) >= threshold, else 0.
        """
        if label_value is None:
            return None
        
        # Convert to binary classification based on threshold
        if isinstance(label_value, list):
            max_val = max(label_value)
            binary_label = 1 if max_val >= self.binary_threshold else 0
        elif isinstance(label_value, (int, float)):
            binary_label = 1 if label_value >= self.binary_threshold else 0
        else:
            # Try to convert to float and compare
            try:
                val = float(label_value)
                binary_label = 1 if val >= self.binary_threshold else 0
            except (ValueError, TypeError):
                # Fallback: treat as 0 if conversion fails
                binary_label = 0
        
        # Return as float for BCEWithLogitsLoss (binary classification)
        return torch.tensor(binary_label, dtype=torch.float32)

    def __getitem__(self, idx: int):
        sample_info = self.sample_indices[idx]
        csv_path = sample_info["csv_path"]
        video_path = sample_info["video_path"]
        start_idx = sample_info["start_idx"]
        end_idx = sample_info["end_idx"]

        # Load knowledge map and video for the specific range
        km_data = load_knowledge_map(csv_path)
        video_frames = load_video_frames_range(
            video_path,
            start_idx,
            end_idx,
            target_size=self.video_target_size,
        )

        # Get the clip for the specified range
        km_clip = km_data[start_idx:end_idx]
        video_clip = video_frames

        # Ensure same length - both should cover the same time period
        assert len(km_clip) == len(video_clip), f"km_clip: {len(km_clip)}, video_clip: {len(video_clip)}"
        min_len = min(len(km_clip), len(video_clip))
        original_length = min_len


        # Target timesteps: knowledge_map=96, video=32
        km_target_timesteps = self.patch_size  # 96
        video_target_timesteps = self.video_frame_count  # 32

        # Get pre-computed global indices and derive relative indices for np.take()
        km_indices_global = sample_info["km_indices"]
        video_indices_global = sample_info["video_indices"]
        
        # Sample knowledge map to exactly 96 timesteps using precomputed global indices.
        # Derive relative indices: global_indices - start_idx (for np.take on the sliced clip)
        if original_length < km_target_timesteps:
            # Pad if shorter - use np.pad instead of concatenate (more efficient)
            pad_len = km_target_timesteps - original_length
            km_clip = np.pad(km_clip[:original_length], ((0, pad_len), (0, 0)), mode='constant')
        elif original_length > km_target_timesteps:
            # Derive relative indices from global indices
            km_sampled_idx = km_indices_global - start_idx
            # Ensure indices are within bounds of km_clip
            km_sampled_idx = km_sampled_idx[(km_sampled_idx >= 0) & (km_sampled_idx < original_length)]
            if len(km_sampled_idx) == km_target_timesteps:
                km_clip = np.take(km_clip, km_sampled_idx, axis=0)
            else:
                # Fallback: recalculate if bounds check failed
                km_sampled_idx = np.linspace(0, original_length - 1, km_target_timesteps, dtype=int)
                km_clip = np.take(km_clip, km_sampled_idx, axis=0)
        else:
            km_clip = km_clip[:original_length]

        # Apply Gaussian noise augmentation if enabled (only for train mode)
        if self.mode == "train" and self.km_gaussian_noise_std is not None and self.km_gaussian_noise_std > 0:
            # Use idx as seed component for reproducibility (but still random per sample)
            seed = hash((idx, sample_info.get("source_file", ""))) % (2**31)
            km_clip = augment_knowledge_map_gaussian_noise(km_clip, self.km_gaussian_noise_std, seed=seed)

        # Sample video to exactly 32 timesteps using precomputed global indices.
        # Derive relative indices: global_indices - start_idx (for np.take on the sliced clip)
        if original_length < video_target_timesteps:
            # Pad if shorter - use np.pad instead of concatenate (more efficient)
            pad_len = video_target_timesteps - original_length
            video_clip = np.pad(
                video_clip[:original_length],
                ((0, pad_len), (0, 0), (0, 0), (0, 0)),
                mode='constant'
            )
        elif original_length > video_target_timesteps:
            # Derive relative indices from global indices
            video_sampled_idx = video_indices_global - start_idx
            # Ensure indices are within bounds of video_clip
            video_sampled_idx = video_sampled_idx[(video_sampled_idx >= 0) & (video_sampled_idx < original_length)]
            if len(video_sampled_idx) == video_target_timesteps:
                video_clip = np.take(video_clip, video_sampled_idx, axis=0)
            else:
                # Fallback: recalculate if bounds check failed
                video_sampled_idx = np.linspace(0, original_length - 1, video_target_timesteps, dtype=int)
                video_clip = np.take(video_clip, video_sampled_idx, axis=0)
        else:
            video_clip = video_clip[:original_length]

        # Convert to tensors
        # Use precomputed global indices directly (already computed once in _create_sample_indices)
        knowledge_map = torch.from_numpy(km_clip).float()
        video = torch.from_numpy(video_clip).float()
        km_indices_tensor = torch.from_numpy(km_indices_global).long()
        video_indices_tensor = torch.from_numpy(video_indices_global).long()

        prompts: List[str] = []
        if self.prompts_data is not None:
            # For concise_prompts, subject_id is not required (general prompts)
            # For other prompt types, subject_id is required
            if self.prompt_selection == 'concise_prompts':
                prompts = get_prompts_for_sample(
                    self.prompts_data,
                    "",  # subject_id not needed for concise_prompts
                    sample_info.get("direction"),
                    self.prompt_selection,
                )
            elif sample_info.get("subject_id") is not None:
                prompts = get_prompts_for_sample(
                    self.prompts_data,
                    sample_info["subject_id"],
                    sample_info.get("direction"),
                    self.prompt_selection,
                )

        label_tensor = self._label_to_tensor(sample_info["label"])

        result = {
            "knowledge_map": knowledge_map,
            "video": video,
            "source_file": sample_info["source_file"],
            "prompts": prompts,
            "km_indices": km_indices_tensor,
            "video_indices": video_indices_tensor,
        }
        
        if label_tensor is not None:
            result["label"] = label_tensor

        return result


def fullgait_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for SigLIP fullgait training and testing.
    Works with SigLIPFullGaitDataset_v2 which returns pre-processed tensors
    (96 KM timesteps, 32 video timesteps) sampled evenly from full samples or chunks.
    """
    knowledge_maps = []
    videos = []
    labels = []
    source_files = []
    prompts_list = []
    texts_list = []  # Processed text strings
    km_indices_list = []
    video_indices_list = []
    max_cobb_list = []

    has_max_cobb = False
    if len(batch) > 0 and "max_cobb" in batch[0] and batch[0]["max_cobb"] is not None:
        has_max_cobb = True

    for item in batch:
        knowledge_maps.append(item["knowledge_map"])
        videos.append(item["video"])
        lab = item.get("label")
        if lab is None:
            labels.append(torch.tensor(0.0, dtype=torch.float32))
        elif isinstance(lab, torch.Tensor):
            labels.append(lab.reshape(()).to(dtype=torch.float32))
        else:
            labels.append(torch.tensor(float(lab), dtype=torch.float32))
        source_files.append(item["source_file"])
        prompts_list.append(item.get("prompts", []))
        
        # Process prompts: convert list of lists to string
        prompts = item.get("prompts", [])
        if isinstance(prompts, list) and len(prompts) > 0:
            if isinstance(prompts[0], list):
                # List of lists: join inner lists with ". " and outer list with ". "
                text = ". ".join([". ".join(p) if isinstance(p, list) else str(p) for p in prompts])
            elif isinstance(prompts[0], str):
                # List of strings: join with ". "
                text = ". ".join(prompts)
            else:
                text = ""
        elif isinstance(prompts, str):
            text = prompts
        else:
            text = ""
        texts_list.append(text)
        
        km_indices_list.append(item.get("km_indices"))
        video_indices_list.append(item.get("video_indices"))

        if has_max_cobb:
            max_cobb_value = item.get("max_cobb")
            if isinstance(max_cobb_value, torch.Tensor):
                max_cobb_list.append(max_cobb_value)
            elif max_cobb_value is not None:
                max_cobb_list.append(torch.tensor(max_cobb_value, dtype=torch.float32))

    # Stack tensors (all should have same length: 96 for KM, 32 for video)
    knowledge_map_batch = torch.stack(knowledge_maps)
    video_batch = torch.stack(videos)
    km_indices_batch = (
        torch.stack(km_indices_list) if km_indices_list and km_indices_list[0] is not None else None
    )
    video_indices_batch = (
        torch.stack(video_indices_list) if video_indices_list and video_indices_list[0] is not None else None
    )
    
    result = {
        "knowledge_map": knowledge_map_batch,
        "video": video_batch,
        "source_files": source_files,
        "prompts": prompts_list,  # Keep original for reference
        "texts": texts_list,  # Processed text strings ready for text encoder
        "km_indices": km_indices_batch,
        "video_indices": video_indices_batch,
    }
    
    if len(labels) > 0:
        result["label"] = torch.stack(labels)
    if has_max_cobb and len(max_cobb_list) > 0:
        # Shape: (batch_size,) continuous Cobb angles
        result["max_cobb"] = torch.stack(max_cobb_list).view(-1)
    
    return result


class FullGaitTrainCollateWithKMNoise:
    """
    Picklable collate_fn for train batches: same as ``fullgait_collate_fn``, then
    adds i.i.d. Gaussian noise to the stacked ``knowledge_map``.

    Must be a **module-level** callable so Windows ``DataLoader(num_workers>0)``
    can pickle it for worker processes (nested/lambda collate fns fail with spawn).
    """

    __slots__ = ("_std", "_video_jitter", "_video_crop_min", "_video_crop_max")

    def __init__(
        self,
        km_gaussian_noise_std: Optional[float] = None,
        video_brightness_contrast_jitter: Optional[float] = None,
        video_random_crop_scale: Optional[Tuple[float, float]] = None,
    ):
        self._std = float(km_gaussian_noise_std) if km_gaussian_noise_std is not None else 0.0
        self._video_jitter = float(video_brightness_contrast_jitter) if video_brightness_contrast_jitter is not None else 0.0
        if video_random_crop_scale is not None:
            self._video_crop_min = float(video_random_crop_scale[0])
            self._video_crop_max = float(video_random_crop_scale[1])
        else:
            self._video_crop_min = 1.0
            self._video_crop_max = 1.0

    def _apply_video_augmentation(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply train-time video augmentation without horizontal flip.
        Input shape: (B, T, H, W, C)
        """
        if video.dim() != 5:
            return video

        out = video
        bsz, t, h, w, c = out.shape

        # Brightness / contrast jitter (per sample, applied consistently across all frames).
        if self._video_jitter > 0.0:
            jitter = self._video_jitter
            brightness = (torch.rand(bsz, 1, 1, 1, 1, device=out.device, dtype=out.dtype) * 2.0 - 1.0) * jitter + 1.0
            contrast = (torch.rand(bsz, 1, 1, 1, 1, device=out.device, dtype=out.dtype) * 2.0 - 1.0) * jitter + 1.0
            sample_mean = out.mean(dim=(1, 2, 3, 4), keepdim=True)
            out = (out - sample_mean) * contrast + sample_mean
            out = out * brightness
            # Keep original dynamic range to avoid out-of-distribution values.
            sample_min = video.amin(dim=(1, 2, 3, 4), keepdim=True)
            sample_max = video.amax(dim=(1, 2, 3, 4), keepdim=True)
            out = torch.max(torch.min(out, sample_max), sample_min)

        # Random crop + resize back (per sample, spatial only, consistent over time).
        if self._video_crop_min < 1.0 and self._video_crop_max >= self._video_crop_min:
            crop_scales = torch.empty(bsz, device=out.device, dtype=out.dtype).uniform_(
                self._video_crop_min, self._video_crop_max
            )
            aug_samples = []
            for i in range(bsz):
                scale = float(crop_scales[i].item())
                if scale >= 0.999:
                    aug_samples.append(out[i])
                    continue
                new_h = max(1, int(round(h * scale)))
                new_w = max(1, int(round(w * scale)))
                top = 0 if new_h >= h else int(torch.randint(0, h - new_h + 1, (1,), device=out.device).item())
                left = 0 if new_w >= w else int(torch.randint(0, w - new_w + 1, (1,), device=out.device).item())
                clip = out[i, :, top:top + new_h, left:left + new_w, :]  # (T, new_h, new_w, C)
                clip_chw = clip.permute(0, 3, 1, 2).contiguous()         # (T, C, new_h, new_w)
                clip_resized = F.interpolate(
                    clip_chw, size=(h, w), mode="bilinear", align_corners=False
                )
                aug_samples.append(clip_resized.permute(0, 2, 3, 1).contiguous())
            out = torch.stack(aug_samples, dim=0)

        return out

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = fullgait_collate_fn(batch)
        if self._std > 0.0 and "knowledge_map" in out:
            km = out["knowledge_map"]
            out["knowledge_map"] = km + torch.randn_like(km) * self._std
        if "video" in out and (self._video_jitter > 0.0 or self._video_crop_min < 1.0):
            out["video"] = self._apply_video_augmentation(out["video"])
        return out


def make_fullgait_train_collate_with_km_noise(
    km_gaussian_noise_std: Optional[float] = None,
    video_brightness_contrast_jitter: Optional[float] = None,
    video_random_crop_scale: Optional[Tuple[float, float]] = None,
) -> FullGaitTrainCollateWithKMNoise:
    """
    Build a collate_fn identical to ``fullgait_collate_fn``, then add i.i.d. Gaussian
    noise to the stacked ``knowledge_map`` batch (training augmentation only).

    Use this when the underlying dataset loads **clean** KM (no noise in ``__getitem__``)
    so validation / internal CV folds are evaluated without corruption, while training
    still sees stochastic augmentation each step.

    Args:
        km_gaussian_noise_std: If None or <= 0, behavior matches ``fullgait_collate_fn``.
        video_brightness_contrast_jitter: Brightness/contrast jitter strength. For example,
            0.1 means factors in [0.9, 1.1]. Applied in train collate only.
        video_random_crop_scale: Spatial crop scale range (min, max), e.g. (0.92, 1.0),
            then resized back to original (H, W). Applied in train collate only.
    """
    return FullGaitTrainCollateWithKMNoise(
        km_gaussian_noise_std=km_gaussian_noise_std,
        video_brightness_contrast_jitter=video_brightness_contrast_jitter,
        video_random_crop_scale=video_random_crop_scale,
    )


def fullgait_test_collate_fn_deprecated(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for fullgait test evaluation.
    Loads full clips from starting index instead of patches.
    
    This function expects batch items to have:
    - csv_path: Path to knowledge map CSV
    - video_path: Path to video file
    - start_idx: Starting index for the clip (default 0)
    - source_file: Source file name
    - subject_id: Subject ID for prompts
    - label: Label (optional)
    - direction: Direction (optional, for prompts)
    - prompts_data: Prompts data dict (optional)
    - prompt_selection: Prompt selection method (optional)
    - binary_threshold: Threshold for binary classification (optional)
    - video_target_size: Target size for video frames (optional)
    - video_frame_count: Number of video frames to sample (optional)
    """
    knowledge_maps = []
    videos = []
    labels = []
    source_files = []
    prompts_list = []
    texts_list = []
    km_indices_list = []
    video_indices_list = []
    
    has_labels = False
    
    for item in batch:
        csv_path = item.get("csv_path")
        video_path = item.get("video_path")
        start_idx = item.get("start_idx", 0)
        source_file = item.get("source_file", "unknown")
        subject_id = item.get("subject_id")
        direction = item.get("direction")
        
        # Load full knowledge map and video from starting index
        km_data = load_knowledge_map(csv_path)
        video_frames = load_video_frames_range(
            video_path,
            start_idx,
            len(km_data),  # Load until end of knowledge map
            target_size=item.get("video_target_size", (224, 224)),
        )
        
        # Get the clip from start_idx to end (or until min length)
        km_clip = km_data[start_idx:]
        video_clip = video_frames
        
        # Ensure same length - both should cover the same time period
        min_len = min(len(km_clip), len(video_clip))
        original_length = min_len
        
        # Target timesteps: knowledge_map=96, video=32
        km_target_timesteps = 96
        video_target_timesteps = item.get("video_frame_count", 32)
        
        # Sample knowledge map to exactly 96 timesteps (evenly spaced from original clip)
        if original_length > km_target_timesteps:
            km_sampled_idx = np.linspace(0, original_length - 1, km_target_timesteps, dtype=int)
            km_clip = km_clip[km_sampled_idx]
        elif original_length < km_target_timesteps:
            # Pad if shorter
            pad_len = km_target_timesteps - original_length
            padding = np.zeros((pad_len, km_clip.shape[1]), dtype=km_clip.dtype)
            km_clip = np.concatenate([km_clip[:original_length], padding], axis=0)
        else:
            km_clip = km_clip[:original_length]
        
        # Sample video to exactly 32 timesteps (evenly spaced from same original clip)
        # This ensures both cover the same time period
        if original_length > video_target_timesteps:
            video_sampled_idx = np.linspace(0, original_length - 1, video_target_timesteps, dtype=int)
            video_clip = video_clip[video_sampled_idx]
        elif original_length < video_target_timesteps:
            # Pad if shorter
            pad_len = video_target_timesteps - original_length
            padding = np.zeros((pad_len, video_clip.shape[1], video_clip.shape[2], video_clip.shape[3]), dtype=video_clip.dtype)
            video_clip = np.concatenate([video_clip[:original_length], padding], axis=0)
        else:
            video_clip = video_clip[:original_length]
        
        # Verify final lengths
        assert len(km_clip) == km_target_timesteps, f"KM length mismatch: {len(km_clip)} != {km_target_timesteps}"
        assert len(video_clip) == video_target_timesteps, f"Video length mismatch: {len(video_clip)} != {video_target_timesteps}"
        
        # Create indices - both sampled from the same original time period
        # KM indices: evenly sampled from [start_idx, start_idx + original_length - 1] to get 96 points
        # Video indices: evenly sampled from [start_idx, start_idx + original_length - 1] to get 32 points
        if original_length > km_target_timesteps:
            km_indices = np.linspace(start_idx, start_idx + original_length - 1, km_target_timesteps, dtype=int)
        else:
            km_indices = np.arange(start_idx, start_idx + len(km_clip))
        
        if original_length > video_target_timesteps:
            video_indices = np.linspace(start_idx, start_idx + original_length - 1, video_target_timesteps, dtype=int)
        else:
            video_indices = np.arange(start_idx, start_idx + len(video_clip))
        
        # Create indices
        km_indices = np.arange(start_idx, start_idx + len(km_clip))
        video_indices = np.arange(start_idx, start_idx + len(video_clip))
        
        # Convert to tensors
        knowledge_map = torch.from_numpy(km_clip).float()
        video = torch.from_numpy(video_clip).float()
        km_indices_tensor = torch.from_numpy(km_indices).long()
        video_indices_tensor = torch.from_numpy(video_indices).long()
        
        # Get prompts if available
        prompts = item.get("prompts", [])
        if not prompts and item.get("prompts_data") is not None and subject_id is not None:
            prompts = get_prompts_for_sample(
                item["prompts_data"],
                subject_id,
                direction,
                item.get("prompt_selection", "top_feature_prompts"),
            )
        
        # Process prompts to text string
        if isinstance(prompts, list) and len(prompts) > 0:
            if isinstance(prompts[0], list):
                text = ". ".join([". ".join(p) if isinstance(p, list) else str(p) for p in prompts])
            elif isinstance(prompts[0], str):
                text = ". ".join(prompts)
            else:
                text = ""
        elif isinstance(prompts, str):
            text = prompts
        else:
            text = ""
        
        # Handle label
        label = item.get("label")
        if label is not None:
            has_labels = True
            # Convert label to tensor if needed
            if isinstance(label, (list, tuple)):
                max_val = max(label) if label else 0.0
                binary_threshold = item.get("binary_threshold", 11.0)
                label_tensor = torch.tensor(1.0 if max_val >= binary_threshold else 0.0, dtype=torch.float32)
            elif isinstance(label, (int, float)):
                binary_threshold = item.get("binary_threshold", 11.0)
                label_tensor = torch.tensor(1.0 if label >= binary_threshold else 0.0, dtype=torch.float32)
            else:
                label_tensor = torch.tensor(float(label), dtype=torch.float32)
        else:
            label_tensor = None
        
        knowledge_maps.append(knowledge_map)
        videos.append(video)
        if label_tensor is not None:
            labels.append(label_tensor)
        source_files.append(source_file)
        prompts_list.append(prompts)
        texts_list.append(text)
        km_indices_list.append(km_indices_tensor)
        video_indices_list.append(video_indices_tensor)
    
    # Pad sequences to same length for batching (pad to max length in batch)
    if len(knowledge_maps) > 0:
        max_km_len = max(km.shape[0] for km in knowledge_maps)
        max_video_len = max(v.shape[0] for v in videos)
        
        # Pad knowledge maps
        padded_knowledge_maps = []
        for km in knowledge_maps:
            if km.shape[0] < max_km_len:
                pad_len = max_km_len - km.shape[0]
                padding = torch.zeros((pad_len, km.shape[1]), dtype=km.dtype)
                km = torch.cat([km, padding], dim=0)
            padded_knowledge_maps.append(km)
        
        # Pad videos
        padded_videos = []
        for v in videos:
            if v.shape[0] < max_video_len:
                pad_len = max_video_len - v.shape[0]
                padding = torch.zeros((pad_len, v.shape[1], v.shape[2], v.shape[3]), dtype=v.dtype)
                v = torch.cat([v, padding], dim=0)
            padded_videos.append(v)
        
        knowledge_map_batch = torch.stack(padded_knowledge_maps)
        video_batch = torch.stack(padded_videos)
    else:
        knowledge_map_batch = torch.empty(0)
        video_batch = torch.empty(0)
    
    km_indices_batch = (
        torch.stack(km_indices_list) if km_indices_list and km_indices_list[0] is not None else None
    )
    video_indices_batch = (
        torch.stack(video_indices_list) if video_indices_list and video_indices_list[0] is not None else None
    )
    
    result = {
        "knowledge_map": knowledge_map_batch,
        "video": video_batch,
        "source_files": source_files,
        "prompts": prompts_list,
        "texts": texts_list,
        "km_indices": km_indices_batch,
        "video_indices": video_indices_batch,
    }
    
    if has_labels and len(labels) > 0:
        result["label"] = torch.stack(labels)
    
    return result


class FullGaitTestDataset(Dataset):
    """
    Dataset for fullgait test evaluation that returns metadata for fullgait_test_collate_fn.
    Returns paths and indices instead of loaded patches, allowing the collate function
    to load full clips from starting index.
    """
    
    def __init__(
        self,
        table_dir: str,
        video_dir: str,
        label_json_path: Optional[str] = None,
        split: Optional[str] = None,
        video_target_size: Optional[Tuple[int, int]] = None,
        video_frame_count: Optional[int] = None,
        prompts_path: Optional[str] = None,
        prompt_selection: str = "top_feature_prompts",
        binary_threshold: float = 11.0,
        start_idx: int = 0,  # Starting index for clips
    ):
        self.table_dir = table_dir
        self.video_dir = video_dir
        self.video_target_size = video_target_size
        self.video_frame_count = video_frame_count
        self.start_idx = start_idx
        self.binary_threshold = binary_threshold
        
        # Load labels if provided
        self.label_map: Optional[Dict[str, Any]] = None
        if label_json_path is not None and split is not None:
            try:
                self.label_map = load_label_map(label_json_path, split)
                print(f"Loaded {len(self.label_map)} labels for split '{split}' from {label_json_path}")
            except Exception as e:
                print(f"Warning: Could not load labels: {e}. Continuing without labels.")
                self.label_map = None
        
        # Load prompts if provided
        self.prompts_data = None
        if prompts_path is not None:
            try:
                self.prompts_data = load_prompts_json(prompts_path)
                print(f"Loaded prompts from: {prompts_path}")
                print(f"  Prompt selection: {prompt_selection}")
            except Exception as e:
                print(f"Warning: Could not load prompts: {e}")
                self.prompts_data = None
        
        self.prompt_selection = prompt_selection
        
        # Get paired samples
        all_paired_samples: List[Tuple[str, str]] = []
        try:
            paired = get_paired_samples_fullgait(table_dir, video_dir)
            all_paired_samples.extend(paired)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
        
        if len(all_paired_samples) == 0:
            raise ValueError("No paired samples found!")
        
        print(f"Found {len(all_paired_samples)} total paired samples")
        
        # Filter samples to only include those with labels (if labels are provided)
        self.samples: List[Tuple[str, str]] = []
        if self.label_map is not None:
            print(f"Filtering samples to only include labeled subjects...")
            for csv_path, video_path in all_paired_samples:
                csv_filename = os.path.basename(csv_path)
                subject_id = extract_subject_id_from_filename(csv_filename)
                if subject_id is not None:
                    subject_key = str(int(subject_id))
                    if subject_key in self.label_map:
                        self.samples.append((csv_path, video_path))
            print(f"Filtered to {len(self.samples)} samples with labels")
        else:
            self.samples = all_paired_samples
            print("No labels provided, using all paired samples")
        
        if len(self.samples) == 0:
            raise ValueError("No samples found after filtering by labels!")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        csv_path, video_path = self.samples[idx]
        csv_filename = os.path.basename(csv_path)
        subject_id = extract_subject_id_from_filename(csv_filename)
        direction = None  # Fullgait doesn't have directions
        
        # Get label
        label = None
        if self.label_map is not None and subject_id is not None:
            subject_key = str(int(subject_id))
            label = self.label_map.get(subject_key)
        
        # Return metadata for collate function to load full clips
        return {
            "csv_path": csv_path,
            "video_path": video_path,
            "start_idx": self.start_idx,
            "source_file": csv_filename,
            "subject_id": subject_id,
            "direction": direction,
            "label": label,
            "prompts_data": self.prompts_data,
            "prompt_selection": self.prompt_selection,
            "binary_threshold": self.binary_threshold,
            "video_target_size": self.video_target_size,
            "video_frame_count": self.video_frame_count,
        }


class SigLIPFullGaitDatasetPKL(Dataset):
    """
    Dataset for SigLIP fullgait training that loads preprocessed data from pkl files.
    Much faster than on-the-fly processing.
    Compatible with SigLIPFullGaitDataset_v2 output format.
    
    Applies Gaussian noise augmentation during loading for true augmentation
    (different noise each epoch, only in train mode).
    """
    
    def __init__(
        self,
        pkl_data_dir: str,
        metadata_path: Optional[str] = None,
        km_gaussian_noise_std: Optional[float] = None,
        mode: str = "train",
        prompts_path: Optional[str] = None,
        prompt_selection: str = "top_feature_prompts",
        binary_threshold: Optional[float] = None,
    ):
        """
        Initialize SigLIP fullgait dataset from pkl files.
        
        Args:
            pkl_data_dir: Directory containing preprocessed pkl files
            metadata_path: Path to patch_metadata.pkl file (if None, will look in pkl_data_dir)
            km_gaussian_noise_std: Standard deviation for Gaussian noise augmentation (None to disable)
                                   Applied during __getitem__ for true augmentation
            mode: 'train' or 'test' mode (noise only applied in train mode)
            prompts_path: Path to prompts JSON file (optional, can also be read from config)
            prompt_selection: Prompt selection method (optional, can also be read from config)
            binary_threshold: Optional override for binary label threshold. If None, use stored label
                             from pkl (or config's binary_threshold when recomputing from label_value).
        """
        self.pkl_data_dir = pkl_data_dir
        self.km_gaussian_noise_std = km_gaussian_noise_std
        self.mode = mode.lower()
        self.binary_threshold = binary_threshold
        
        if self.mode not in ["train", "test"]:
            raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")
        
        # Load metadata
        if metadata_path is None:
            metadata_path = os.path.join(pkl_data_dir, 'patch_metadata.pkl')
        
        if not os.path.exists(metadata_path):
            # Fallback: build metadata from existing patch pkl files
            patches_dir = os.path.join(pkl_data_dir, 'patches')
            patch_files = sorted(glob.glob(os.path.join(patches_dir, '*.pkl')))
            if len(patch_files) == 0:
                raise FileNotFoundError(f"Metadata file not found and no patch files in: {patches_dir}")
            print(f"Metadata not found. Building from {len(patch_files)} patch files...")
            patch_metadata = []
            for pkl_path in patch_files:
                try:
                    with open(pkl_path, 'rb') as f:
                        patch_data = pickle.load(f)
                    patch_id = Path(pkl_path).stem.replace('patch_', '')
                    patch_metadata.append({
                        'patch_id': patch_id,
                        'pkl_path': pkl_path,
                        'subject_id': patch_data.get('subject_id'),
                        'direction': patch_data.get('direction'),
                        'source_file': patch_data.get('source_file', 'unknown'),
                        'start_idx': patch_data.get('start_idx', 0),
                        'end_idx': patch_data.get('end_idx', 0),
                        'knowledge_map_shape': patch_data['knowledge_map'].shape,
                        'video_shape': patch_data['video'].shape,
                        'num_prompts': len(patch_data.get('prompts', [])),
                        'has_label': patch_data.get('label') is not None,
                        'label': patch_data.get('label'),
                        'label_value': patch_data.get('label_value'),
                        'binary_threshold': patch_data.get('binary_threshold'),
                    })
                except Exception as e:
                    print(f"Warning: failed to read {pkl_path}: {e}")
                    continue
            if len(patch_metadata) == 0:
                raise FileNotFoundError(f"Could not build metadata; all patch reads failed in: {patches_dir}")
            with open(metadata_path, 'wb') as f:
                pickle.dump({'patch_metadata': patch_metadata, 'config': {}}, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Metadata rebuilt and saved to: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.patch_metadata = metadata['patch_metadata']
        pkl_data_dir_abs = os.path.abspath(os.path.normpath(self.pkl_data_dir))
        self.pkl_data_dir = pkl_data_dir_abs
        for entry in self.patch_metadata:
            entry['pkl_path'] = _resolve_patch_pkl_path(
                entry.get('pkl_path', ''),
                pkl_data_dir_abs,
                entry.get('patch_id'),
            )
        self.config = metadata.get('config', {})
        # binary_threshold from config (used when recomputing from label_value; override from __init__ takes precedence)
        self.binary_threshold_from_config = self.config.get('binary_threshold')

        # Get prompts_path and prompt_selection from args or config
        self.prompts_path = prompts_path or self.config.get('prompts_path')
        self.prompt_selection = prompt_selection or self.config.get('prompt_selection', 'top_feature_prompts')
        
        # Debug: print what we're using
        if prompts_path:
            print(f"📝 Using prompts_path from argument: {prompts_path}")
        elif self.config.get('prompts_path'):
            print(f"📝 Using prompts_path from config: {self.config.get('prompts_path')}")
        else:
            print(f"⚠️  No prompts_path provided (neither argument nor config)")
        
        if prompt_selection:
            print(f"📝 Using prompt_selection from argument: {prompt_selection}")
        elif self.config.get('prompt_selection'):
            print(f"📝 Using prompt_selection from config: {self.config.get('prompt_selection')}")
        else:
            print(f"📝 Using default prompt_selection: top_feature_prompts")
        
        # Load prompts if available
        self.prompts_data = None
        if self.prompts_path is not None:
            try:
                if not os.path.exists(self.prompts_path):
                    print(f"⚠️  Warning: Prompts file not found: {self.prompts_path}")
                    self.prompts_data = None
                else:
                    self.prompts_data = load_prompts_json(self.prompts_path)
                    print(f"✅ Loaded prompts from: {self.prompts_path}")
                    print(f"  Prompt selection: {self.prompt_selection}")
                    # Debug: print available keys in prompts_data
                    if self.prompts_data:
                        print(f"  Available keys in prompts_data: {list(self.prompts_data.keys())}")
                        if 'concise_prompts' in self.prompts_data:
                            concise_count = len(self.prompts_data['concise_prompts']) if isinstance(self.prompts_data['concise_prompts'], list) else 0
                            print(f"  concise_prompts count: {concise_count}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load prompts: {e}")
                import traceback
                traceback.print_exc()
                self.prompts_data = None
        else:
            print(f"⚠️  Warning: No prompts_path provided, prompts will be empty")
        
        print(f"Loaded {len(self.patch_metadata)} patches from pkl files")
        print(f"  Knowledge map timesteps: {self.config.get('patch_size', 'unknown')}")
        print(f"  Video timesteps: {self.config.get('video_frame_count', 'unknown')}")
        print(f"  Dataset mode: {self.mode}")
        if self.km_gaussian_noise_std is not None and self.km_gaussian_noise_std > 0 and self.mode == "train":
            print(f"  Gaussian noise augmentation: enabled (std={self.km_gaussian_noise_std})")
        else:
            print(f"  Gaussian noise augmentation: disabled")
    
    def __len__(self):
        return len(self.patch_metadata)
    
    def __getitem__(self, idx):
        """Load patch data from pkl file."""
        patch_info = self.patch_metadata[idx]
        pkl_path = patch_info['pkl_path']
        
        # Load patch data
        patch_data = _load_pickle_compat(pkl_path)
        
        # Convert numpy arrays to tensors
        knowledge_map = _to_float_tensor(patch_data['knowledge_map'])
        video = _to_float_tensor(patch_data['video'])
        source_file = patch_data.get('source_file', 'unknown')
        
        # Process prompts (load from prompts_data if available, otherwise use stored prompts)
        prompts: List[str] = []
        if self.prompts_data is not None:
            # Process prompts dynamically from prompts_data
            subject_id = patch_data.get('subject_id')
            direction = patch_data.get('direction')
            if self.prompt_selection == 'concise_prompts':
                prompts = get_prompts_for_sample(
                    self.prompts_data,
                    "",
                    direction,
                    self.prompt_selection,
                )
            elif subject_id is not None:
                prompts = get_prompts_for_sample(
                    self.prompts_data,
                    subject_id,
                    direction,
                    self.prompt_selection,
                )
            # Debug: log if prompts are empty (only for first few samples)
            if len(prompts) == 0 and idx < 3:  # Log for first 3 samples
                print(f"⚠️  Warning: No prompts found for sample {idx}")
                print(f"   prompt_selection: {self.prompt_selection}")
                print(f"   subject_id: {subject_id}")
                print(f"   direction: {direction}")
                print(f"   prompts_data keys: {list(self.prompts_data.keys()) if self.prompts_data else 'None'}")
                if self.prompts_data and 'concise_prompts' in self.prompts_data:
                    concise = self.prompts_data['concise_prompts']
                    print(f"   concise_prompts type: {type(concise)}, length: {len(concise) if isinstance(concise, list) else 'N/A'}")
        else:
            # Fallback: use stored prompts from pickle file (if available)
            prompts = patch_data.get('prompts', [])
            # Debug: log if prompts_data is None
            if idx == 0:  # Only log for first sample to avoid spam
                print(f"⚠️  Warning: prompts_data is None, using stored prompts (likely empty)")
                print(f"   prompts_path: {self.prompts_path}")
                print(f"   metadata stored prompts_path (from PKL): {self.config.get('prompts_path')}")
        
        # Apply Gaussian noise augmentation during loading (for true augmentation)
        # This gives different noise each epoch, unlike preprocessing-time augmentation
        if self.mode == "train" and self.km_gaussian_noise_std is not None and self.km_gaussian_noise_std > 0:
            # Generate random noise using PyTorch's random state (different each epoch)
            noise = torch.randn_like(knowledge_map) * self.km_gaussian_noise_std
            knowledge_map = knowledge_map + noise
        
        # Convert indices if present
        km_indices = None
        if 'km_indices' in patch_data and patch_data['km_indices'] is not None:
            km_indices = _to_long_tensor(patch_data['km_indices'])
        else:
            print(f"⚠️  Warning: km_indices is None")
        
        video_indices = None
        if 'video_indices' in patch_data and patch_data['video_indices'] is not None:
            video_indices = _to_long_tensor(patch_data['video_indices'])
        else:
            print(f"⚠️  Warning: video_indices is None")
        
        # Convert label: use stored binary label, or recompute from label_value if override threshold given.
        # Additionally, track the maximum continuous Cobb angle value as `max_cobb` for downstream use.
        label = None
        max_cobb = None
        label_value = patch_data.get('label_value')
        stored_label = patch_data.get('label')

        binary_threshold = self.binary_threshold
        if label_value is not None:
            # Compute max_cobb from the raw label_value (list or scalar).
            if isinstance(label_value, list):
                try:
                    max_cobb = float(max(label_value))
                except (TypeError, ValueError):
                    max_cobb = None
            else:
                try:
                    max_cobb = float(label_value)
                except (TypeError, ValueError):
                    max_cobb = None

        if label_value is not None and binary_threshold is not None:
            # Recompute binary label from raw label_value using (override or stored) threshold
            if isinstance(label_value, list):
                max_val = max(label_value)
                label = torch.tensor(1.0 if max_val >= binary_threshold else 0.0, dtype=torch.float32)
            elif isinstance(label_value, (int, float)):
                label = torch.tensor(1.0 if label_value >= binary_threshold else 0.0, dtype=torch.float32)
            else:
                try:
                    val = float(label_value)
                    label = torch.tensor(1.0 if val >= binary_threshold else 0.0, dtype=torch.float32)
                except (ValueError, TypeError):
                    label = torch.tensor(0.0, dtype=torch.float32)
        elif stored_label is not None:
            label = torch.tensor(stored_label, dtype=torch.float32)

        result = {
            'knowledge_map': knowledge_map,  # (patch_size, km_features) - typically (96, features)
            'video': video,  # (video_frame_count, H, W, C) - typically (32, H, W, C)
            'source_file': source_file,
            'prompts': prompts,  # List of prompt strings
            'km_indices': km_indices,  # (patch_size,) or None
            'video_indices': video_indices,  # (video_frame_count,) or None
        }
        
        if label is not None:
            result['label'] = label
        else:
            # PK / unlabeled external patches: treat as negative (0) for screening eval
            result['label'] = torch.tensor(0.0, dtype=torch.float32)
        if max_cobb is not None:
            # Continuous Cobb angle derived from label_value (e.g. max over timesteps)
            result['max_cobb'] = torch.tensor(max_cobb, dtype=torch.float32)
        
        return result

