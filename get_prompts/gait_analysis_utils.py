# -*- coding: utf-8 -*-
"""
Gait Analysis Utilities for Scoliosis Research
Feature extraction and analysis functions for 2D pose time-series data.

@author: Olive
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import os

# YOLOv8 pose landmark names (17 keypoints)
LANDMARK_NAMES = [
    'nose',
    'left_eye', 'right_eye',
    'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]

def load_pose_csv(filepath):
    """
    Load a single pose CSV file.
    
    Returns:
        data: numpy array of shape (Time, 17, 2) - landmarks with x,y coords
        frames: frame indices
    """
    df = pd.read_csv(filepath)
    df = df.dropna()
    
    # Extract values (exclude 'frames' column)
    values = df.drop(columns=['frames']).values
    frames = df['frames'].values
    
    # Handle 60fps videos by downsampling to 30fps
    if len(values) > 300:
        values = values[::2]
        frames = frames[::2]
    
    # Reshape to (Time, 17, 2)
    n_frames = values.shape[0]
    data = values.reshape(n_frames, 17, 2)
    
    return data, frames


def get_distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_angle(p1, p2, p3):
    """
    Calculate angle at p2 formed by p1-p2-p3.
    Returns angle in degrees.
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def extract_gait_features_per_frame(frame_data):
    """
    Extract gait-relevant features from a single frame.
    
    Args:
        frame_data: array of shape (17, 2) - landmarks for one frame
        
    Returns:
        features: dict of feature values
    """
    # Helper to get landmark by name
    def lm(name):
        return frame_data[LANDMARK_NAMES.index(name)]
    
    features = {}
    
    # --- Distance features ---
    # Step width (lateral distance between ankles)
    features['step_width'] = abs(lm('left_ankle')[0] - lm('right_ankle')[0])
    
    # Step length proxy (vertical distance between ankles)
    features['ankle_vert_diff'] = abs(lm('left_ankle')[1] - lm('right_ankle')[1])
    
    # Ankle distance
    features['ankle_distance'] = get_distance(lm('left_ankle'), lm('right_ankle'))
    
    # Hip width
    features['hip_width'] = get_distance(lm('left_hip'), lm('right_hip'))
    
    # Shoulder width
    features['shoulder_width'] = get_distance(lm('left_shoulder'), lm('right_shoulder'))
    
    # --- Vertical position features (trunk posture) ---
    # Shoulder center height
    shoulder_center_y = (lm('left_shoulder')[1] + lm('right_shoulder')[1]) / 2
    hip_center_y = (lm('left_hip')[1] + lm('right_hip')[1]) / 2
    features['trunk_height'] = abs(shoulder_center_y - hip_center_y)
    
    # --- Asymmetry features ---
    # Shoulder tilt (height difference)
    features['shoulder_tilt'] = lm('left_shoulder')[1] - lm('right_shoulder')[1]
    
    # Hip tilt (height difference)
    features['hip_tilt'] = lm('left_hip')[1] - lm('right_hip')[1]
    
    # Trunk lateral shift (shoulder center vs hip center in x)
    shoulder_center_x = (lm('left_shoulder')[0] + lm('right_shoulder')[0]) / 2
    hip_center_x = (lm('left_hip')[0] + lm('right_hip')[0]) / 2
    features['trunk_lateral_shift'] = shoulder_center_x - hip_center_x
    
    # --- Angle features ---
    # Left knee angle
    features['left_knee_angle'] = get_angle(lm('left_hip'), lm('left_knee'), lm('left_ankle'))
    
    # Right knee angle
    features['right_knee_angle'] = get_angle(lm('right_hip'), lm('right_knee'), lm('right_ankle'))
    
    # Left hip angle (shoulder-hip-knee)
    features['left_hip_angle'] = get_angle(lm('left_shoulder'), lm('left_hip'), lm('left_knee'))
    
    # Right hip angle
    features['right_hip_angle'] = get_angle(lm('right_shoulder'), lm('right_hip'), lm('right_knee'))
    
    # Left elbow angle
    features['left_elbow_angle'] = get_angle(lm('left_shoulder'), lm('left_elbow'), lm('left_wrist'))
    
    # Right elbow angle
    features['right_elbow_angle'] = get_angle(lm('right_shoulder'), lm('right_elbow'), lm('right_wrist'))
    
    return features


def extract_temporal_features(data):
    """
    Extract aggregated temporal features from a full gait sample.
    
    Args:
        data: array of shape (Time, 17, 2)
        
    Returns:
        feature_vector: 1D numpy array of aggregated features
        feature_names: list of feature names
    """
    n_frames = data.shape[0]
    
    # Extract per-frame features for all frames
    frame_features_list = []
    for t in range(n_frames):
        ff = extract_gait_features_per_frame(data[t])
        frame_features_list.append(ff)
    
    # Convert to DataFrame for easy aggregation
    df_features = pd.DataFrame(frame_features_list)
    
    # Aggregation statistics
    aggregated = {}
    feature_names = []
    
    for col in df_features.columns:
        signal = df_features[col].values
        
        # Central tendency
        aggregated[f'{col}_mean'] = np.mean(signal)
        aggregated[f'{col}_std'] = np.std(signal)
        
        # Range/variability
        aggregated[f'{col}_range'] = np.max(signal) - np.min(signal)
        aggregated[f'{col}_iqr'] = np.percentile(signal, 75) - np.percentile(signal, 25)
        
        # Extremes
        aggregated[f'{col}_min'] = np.min(signal)
        aggregated[f'{col}_max'] = np.max(signal)
        
        # Skewness (asymmetry of distribution)
        aggregated[f'{col}_skew'] = stats.skew(signal)
    
    feature_names = list(aggregated.keys())
    feature_vector = np.array(list(aggregated.values()))
    
    return feature_vector, feature_names


def extract_bilateral_asymmetry_features(data):
    """
    Extract features specifically measuring left-right asymmetry over time.
    
    Args:
        data: array of shape (Time, 17, 2)
        
    Returns:
        asymmetry_features: dict of asymmetry metrics
    """
    n_frames = data.shape[0]
    
    left_knee_angles = []
    right_knee_angles = []
    left_hip_angles = []
    right_hip_angles = []
    
    for t in range(n_frames):
        ff = extract_gait_features_per_frame(data[t])
        left_knee_angles.append(ff['left_knee_angle'])
        right_knee_angles.append(ff['right_knee_angle'])
        left_hip_angles.append(ff['left_hip_angle'])
        right_hip_angles.append(ff['right_hip_angle'])
    
    asymmetry = {}
    
    # Knee angle asymmetry
    knee_diff = np.array(left_knee_angles) - np.array(right_knee_angles)
    asymmetry['knee_angle_diff_mean'] = np.mean(knee_diff)
    asymmetry['knee_angle_diff_std'] = np.std(knee_diff)
    asymmetry['knee_angle_diff_abs_mean'] = np.mean(np.abs(knee_diff))
    
    # Hip angle asymmetry
    hip_diff = np.array(left_hip_angles) - np.array(right_hip_angles)
    asymmetry['hip_angle_diff_mean'] = np.mean(hip_diff)
    asymmetry['hip_angle_diff_std'] = np.std(hip_diff)
    asymmetry['hip_angle_diff_abs_mean'] = np.mean(np.abs(hip_diff))
    
    return asymmetry


def load_dataset(folder_path):
    """
    Load all CSV files from a folder.
    
    Returns:
        samples: list of (Time, 17, 2) arrays
        filenames: list of filenames
    """
    samples = []
    filenames = []
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder_path, filename)
            try:
                data, _ = load_pose_csv(filepath)
                samples.append(data)
                filenames.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return samples, filenames


def build_feature_matrix(samples):
    """
    Build feature matrix from list of samples.
    
    Args:
        samples: list of (Time, 17, 2) arrays
        
    Returns:
        X: feature matrix of shape (n_samples, n_features)
        feature_names: list of feature names
    """
    feature_vectors = []
    feature_names = None
    
    for data in samples:
        # Extract temporal features
        fv, fn = extract_temporal_features(data)
        
        # Extract bilateral asymmetry features
        asymmetry = extract_bilateral_asymmetry_features(data)
        asymmetry_values = list(asymmetry.values())
        asymmetry_names = list(asymmetry.keys())
        
        # Combine all features
        combined_fv = np.concatenate([fv, asymmetry_values])
        
        feature_vectors.append(combined_fv)
        if feature_names is None:
            feature_names = fn + asymmetry_names
    
    X = np.array(feature_vectors)
    
    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    return X, feature_names

