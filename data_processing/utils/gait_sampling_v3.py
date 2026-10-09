"""
Peak-centered gait sampling (StraightTurningV3-style).

Shared by the integrated predict pipeline and batch PKL export.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

from utils.knowledge_map import GetAllFeatures


def load_pose_values_from_csv(
    csv_path: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> np.ndarray:
    """Load raw pose rows (no frame-index column) from a CSV segment."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, engine="c")
    except Exception:
        df = pd.read_csv(csv_path, engine="python")

    df = df.dropna()
    if df.shape[0] == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")

    values = df.values[:, :-1] if df.shape[1] > 1 else df.values
    if end_idx is None:
        end_idx = len(values)
    values = values[start_idx:end_idx]
    if len(values) == 0:
        raise ValueError(f"No pose rows in range [{start_idx}, {end_idx}) for {csv_path}")
    return values


def downsample_pose_high_fps(values: np.ndarray, threshold: int = 300) -> Tuple[np.ndarray, int]:
    """Halve temporal resolution while len > threshold (60 Hz → 30 Hz style). Returns (values, factor)."""
    out = values
    factor = 1
    while len(out) > threshold:
        out = out[::2]
        factor *= 2
    return out, factor


def peak_shift_mid(seq: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Find a single kinematic peak and circularly shift the sequence so the peak is at the centre.
    Returns (shifted_sequence, peak_index_before_shift).
    """
    whole_leng = len(seq)
    if whole_leng == 0:
        raise ValueError("Cannot peak-shift an empty sequence")

    seq_z = stats.zscore(np.sum(seq, axis=1))
    peaks, _ = find_peaks(seq_z, distance=whole_leng)
    if len(peaks) == 0:
        peak = int(np.argmax(seq_z))
    elif len(peaks) == 1:
        peak = int(peaks[0])
    else:
        peak = int(peaks[np.argmax(seq_z[peaks])])

    shift_len = whole_leng // 2 - peak
    if shift_len > 0:
        seq_filtered = np.concatenate((seq[-shift_len:], seq[:-shift_len]), axis=0)
    else:
        shift_len = abs(shift_len)
        seq_filtered = np.concatenate((seq[shift_len:], seq[:shift_len]), axis=0)

    return seq_filtered, peak


def subsample_frame_apart(
    seq: np.ndarray,
    max_seq: int,
    from_start: bool = True,
) -> np.ndarray:
    """Evenly subsample along time with FRAME_APART stepping (V3 _get_table_libs logic)."""
    if max_seq <= 0:
        raise ValueError("max_seq must be positive")

    n = len(seq)
    if n == 0:
        raise ValueError("Cannot subsample an empty sequence")

    frame_apart = max(1, n // max_seq)
    take_len = min(n, max_seq * frame_apart)

    if from_start:
        seq = seq[:take_len]
    else:
        seq = seq[-take_len:]

    seq = seq[::frame_apart]

    if len(seq) < max_seq:
        if seq.ndim == 1:
            pad_width = ((0, max_seq - len(seq)),)
        else:
            pad_width = ((0, max_seq - len(seq)), (0, 0))
        seq = np.pad(seq, pad_width, mode="edge")
    elif len(seq) > max_seq:
        seq = seq[:max_seq]

    return seq


def get_symmetric_indices(
    peak: int,
    max_seq: int,
    min_val: int = 0,
    max_val: Optional[int] = None,
) -> List[int]:
    """Indices expanding outward from peak (V3 _get_video_libs)."""
    if max_val is None:
        max_val = peak + max_seq

    original_indices = [int(peak)]
    left = int(peak) - 1
    right = int(peak) + 1
    count = 1

    while count < max_seq:
        if right <= max_val:
            original_indices.append(right)
            count += 1
            if count >= max_seq:
                break
        if left >= min_val:
            original_indices.append(left)
            count += 1
            if count >= max_seq:
                break
        if left < min_val and right > max_val:
            break
        left -= 1
        right += 1

    return sorted(original_indices)


def load_video_frames_at_indices(
    video_path: str,
    indices: List[int],
    target_size: Optional[Tuple[int, int]] = None,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """Load specific frame indices from a video (V3 _get_video_libs)."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    if target_size is not None:
        out_w, out_h = target_size[0], target_size[1]
    else:
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: List[np.ndarray] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ret, frame = cap.read()
        if not ret:
            break
        if target_size is not None:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=interpolation)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames loaded from {video_path} at indices {indices[:5]}...")

    return np.asarray(frames, dtype=np.uint8)


def get_video_frame_count_safe(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def process_clip_v3_style(
    csv_path: str,
    video_path: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    km_timesteps: int = 96,
    video_timesteps: int = 32,
    video_target_size: Optional[Tuple[int, int]] = None,
    high_fps_threshold: int = 300,
    from_start: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, int]:
    """
    Run the StraightTurningV3 sampling pipeline on one CSV/video pair.

    Returns:
        km_clip, video_clip, peak_idx, km_indices, video_indices, downsample_factor
    """
    if end_idx is None:
        try:
            df = pd.read_csv(csv_path, engine="c")
        except Exception:
            df = pd.read_csv(csv_path, engine="python")
        end_idx = len(df.dropna())
        if end_idx == 0:
            end_idx = get_video_frame_count_safe(video_path)

    pose_values = load_pose_values_from_csv(csv_path, start_idx, end_idx)
    pose_values, downsample_factor = downsample_pose_high_fps(pose_values, high_fps_threshold)

    km_full = np.squeeze(GetAllFeatures(pose_values))
    if km_full.ndim == 1:
        km_full = km_full.reshape(-1, 1)

    km_shifted, peak_idx = peak_shift_mid(km_full)
    km_clip = subsample_frame_apart(km_shifted, km_timesteps, from_start=from_start)

    timeline_len = len(pose_values)
    seg_start = start_idx
    video_peak = seg_start + peak_idx * downsample_factor

    video_indices_list = get_symmetric_indices(
        video_peak,
        video_timesteps,
        min_val=seg_start,
        max_val=seg_start + timeline_len * downsample_factor - 1,
    )

    total_frames = get_video_frame_count_safe(video_path)
    video_indices_list = [min(max(0, i), total_frames - 1) for i in video_indices_list]
    while len(video_indices_list) < video_timesteps:
        video_indices_list.append(video_indices_list[-1])

    video_clip = load_video_frames_at_indices(
        video_path,
        video_indices_list[:video_timesteps],
        target_size=video_target_size,
    )

    km_indices = np.arange(km_timesteps, dtype=int)
    video_indices = np.asarray(video_indices_list[:video_timesteps], dtype=int)

    return km_clip, video_clip, peak_idx, km_indices, video_indices, downsample_factor
