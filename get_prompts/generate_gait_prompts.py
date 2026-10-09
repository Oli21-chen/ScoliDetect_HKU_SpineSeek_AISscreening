"""
Generate fine-grained, individual-specific gait prompts based on pose estimation data.

This version:
1. Analyzes individual subjects' forward and backward gait data
2. Uses HEALTHY POPULATION as normal reference (not all subjects including patients)
3. Identifies subject-specific discriminative features
4. Generates personalized prompts based on deviation from healthy norms
5. Creates prompts that capture individual gait characteristics

Key improvement: Uses healthy reference data to ensure clinically meaningful comparisons,
avoiding the bias of including patient data in the reference population.

Based on findings from Report_4class.md and individual pose estimation data.
"""

import json
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import sys
from gait_analysis_utils import load_pose_csv, extract_temporal_features, extract_bilateral_asymmetry_features

def load_subject_data(data_dir: str, subject_id: str = None) -> Dict[str, np.ndarray]:
    """
    Load forward and backward gait data for a subject or all subjects.
    
    Args:
        data_dir: Base directory containing going_forward and going_backward folders
        subject_id: Specific subject ID (if None, loads all)
    
    Returns:
        Dictionary with 'forward' and 'backward' keys, each containing list of (Time, 17, 2) arrays
    """
    forward_dir = os.path.join(data_dir, 'going_forward')
    backward_dir = os.path.join(data_dir, 'going_backward')
    
    data = {'forward': [], 'backward': []}
    
    # Get all CSV files
    if subject_id:
        # Match subject ID precisely (as parts[1] after splitting by underscore)
        # This prevents "10" from matching "101", "102", etc.
        def matches_subject_id(filename):
            parts = filename.replace('.csv', '').split('_')
            return len(parts) > 1 and parts[1] == subject_id
        
        forward_files = [f for f in os.listdir(forward_dir) if f.endswith('.csv') and matches_subject_id(f)]
        backward_files = [f for f in os.listdir(backward_dir) if f.endswith('.csv') and matches_subject_id(f)]
    else:
        forward_files = [f for f in os.listdir(forward_dir) if f.endswith('.csv')]
        backward_files = [f for f in os.listdir(backward_dir) if f.endswith('.csv')]
    
    # Load forward data
    for fname in sorted(forward_files):
        try:
            filepath = os.path.join(forward_dir, fname)
            data_array, _ = load_pose_csv(filepath)
            data['forward'].append((fname, data_array))
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    
    # Load backward data
    for fname in sorted(backward_files):
        try:
            filepath = os.path.join(backward_dir, fname)
            data_array, _ = load_pose_csv(filepath)
            data['backward'].append((fname, data_array))
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    
    return data


def extract_individual_features(data_array: np.ndarray) -> Dict[str, float]:
    """
    Extract all features for a single gait sample.
    
    Returns:
        Dictionary of feature_name: feature_value
    """
    # Extract temporal features
    fv, fn = extract_temporal_features(data_array)
    
    # Extract asymmetry features
    asymmetry = extract_bilateral_asymmetry_features(data_array)
    
    # Combine into dictionary
    features = dict(zip(fn, fv))
    features.update(asymmetry)
    
    return features


def compare_forward_backward(forward_features: Dict, backward_features: Dict) -> Dict[str, float]:
    """
    Compare forward vs backward gait features.
    
    Returns:
        Dictionary of feature_name: (forward_value, backward_value, difference, percent_change)
    """
    comparison = {}
    
    all_features = set(forward_features.keys()) | set(backward_features.keys())
    
    for feat_name in all_features:
        fwd_val = forward_features.get(feat_name, 0)
        bwd_val = backward_features.get(feat_name, 0)
        
        diff = fwd_val - bwd_val
        percent_change = (diff / (abs(fwd_val) + 1e-8)) * 100 if abs(fwd_val) > 1e-8 else 0
        
        comparison[feat_name] = {
            'forward': fwd_val,
            'backward': bwd_val,
            'difference': diff,
            'percent_change': percent_change,
            'abs_change': abs(diff)
        }
    
    return comparison


def identify_discriminative_features(
    all_subject_features: List[Dict],
    feature_names: List[str],
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Identify most discriminative features across subjects using variance/range.
    
    Args:
        all_subject_features: List of feature dictionaries for each subject
        feature_names: List of feature names to consider
        top_k: Number of top features to return
    
    Returns:
        List of (feature_name, discriminative_score) tuples, sorted by score
    """
    # Build feature matrix
    feature_matrix = []
    for subj_features in all_subject_features:
        row = [subj_features.get(fn, 0) for fn in feature_names]
        feature_matrix.append(row)
    
    feature_matrix = np.array(feature_matrix)
    
    # Calculate discriminative scores (coefficient of variation)
    scores = {}
    for i, feat_name in enumerate(feature_names):
        values = feature_matrix[:, i]
        if np.std(values) > 1e-8:
            # Coefficient of variation (std/mean) - higher = more discriminative
            cv = np.std(values) / (np.abs(np.mean(values)) + 1e-8)
            # Also consider range
            range_val = np.max(values) - np.min(values)
            # Combined score
            scores[feat_name] = cv * range_val
        else:
            scores[feat_name] = 0
    
    # Sort by score
    sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_features[:top_k]


def generate_feature_based_prompt(
    feature_name: str,
    feature_value: float,
    reference_value: Optional[float] = None,
    direction: str = "forward"
) -> str:
    """
    Generate a descriptive prompt based on a specific feature value.
    
    Args:
        feature_name: Name of the feature (e.g., 'right_hip_angle_mean')
        feature_value: Actual feature value
        reference_value: Reference/normal value for comparison
        direction: 'forward' or 'backward'
    
    Returns:
        Descriptive prompt string
    """
    # Parse feature name
    base_feature = feature_name.replace('_mean', '').replace('_std', '').replace('_range', '')
    
    # Feature category mapping
    feature_categories = {
        'hip_angle': 'hip extension',
        'knee_angle': 'knee flexion',
        'elbow_angle': 'elbow angle',
        'step_width': 'step width',
        'ankle_distance': 'ankle distance',
        'shoulder_tilt': 'shoulder alignment',
        'hip_tilt': 'hip alignment',
        'trunk_lateral_shift': 'trunk lateral shift',
    }
    
    # Get category
    category = None
    for key, val in feature_categories.items():
        if key in base_feature:
            category = val
            break
    
    if category is None:
        category = base_feature.replace('_', ' ')
    
    # Determine if it's left/right/bilateral
    side = None
    if 'left_' in base_feature:
        side = 'left'
    elif 'right_' in base_feature:
        side = 'right'
    
    # Determine statistic type
    stat_type = None
    if '_mean' in feature_name:
        stat_type = 'average'
    elif '_std' in feature_name:
        stat_type = 'variable'
    elif '_range' in feature_name:
        stat_type = 'wide range'
    elif '_min' in feature_name:
        stat_type = 'minimum'
    elif '_max' in feature_name:
        stat_type = 'maximum'
    
    # Build prompt based on value
    if reference_value is not None:
        diff = feature_value - reference_value
        if abs(diff) > 0.1 * abs(reference_value):  # 10% difference threshold
            if diff > 0:
                modifier = "increased" if stat_type != 'variable' else "high variability in"
            else:
                modifier = "reduced" if stat_type != 'variable' else "low variability in"
        else:
            modifier = "normal" if stat_type != 'variable' else "consistent"
    else:
        modifier = ""
    
    # Construct prompt
    parts = []
    if modifier:
        parts.append(modifier)
    if stat_type and stat_type != 'average':
        parts.append(stat_type)
    if side:
        parts.append(side)
    parts.append(category)
    if direction != "forward":
        parts.append(f"during {direction} gait")
    else:
        parts.append("during gait")
    
    prompt = " ".join(parts)
    
    # Clean up
    prompt = prompt.replace("  ", " ").strip()
    
    return prompt


def get_base_feature_name(feat_name: str) -> str:
    """
    Extract base feature name by removing statistic suffixes.
    
    E.g., 'shoulder_tilt_mean' -> 'shoulder_tilt'
          'left_knee_angle_std' -> 'left_knee_angle'
    """
    # Remove common statistic suffixes
    suffixes = ['_mean', '_std', '_range', '_min', '_max', '_median']
    base_name = feat_name
    for suffix in suffixes:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    return base_name


def deduplicate_prompts(prompts: List[str]) -> List[str]:
    """
    Remove duplicate or highly similar prompts.
    
    Args:
        prompts: List of prompt strings
    
    Returns:
        Deduplicated list of prompts
    """
    if not prompts:
        return prompts
    
    seen_base_concepts = set()
    deduplicated = []
    
    for prompt in prompts:
        # Extract key concepts from prompt (ignore modifiers like "increased", "reduced")
        # Remove common modifiers to get base concept
        base_prompt = prompt.lower()
        for modifier in ['increased ', 'reduced ', 'high variability in ', 'low variability in ', 
                         'variable ', 'wide range ', 'minimum ', 'maximum ', 'average ',
                         'normal ', 'consistent ']:
            base_prompt = base_prompt.replace(modifier, '')
        
        # Create a normalized key for comparison
        # This helps identify prompts about the same concept
        concept_key = base_prompt.strip()
        
        # Only add if we haven't seen this base concept yet
        if concept_key not in seen_base_concepts:
            seen_base_concepts.add(concept_key)
            deduplicated.append(prompt)
    
    return deduplicated


def generate_individual_prompts(
    subject_features: Dict[str, float],
    forward_backward_comparison: Optional[Dict] = None,
    top_features: Optional[List[str]] = None,
    direction: str = "forward",
    population_stats: Optional[Dict] = None
) -> List[str]:
    """
    Generate personalized prompts for an individual subject.
    
    Args:
        subject_features: Dictionary of feature_name: value for the subject
        forward_backward_comparison: Comparison dict from compare_forward_backward
        top_features: List of top discriminative features to focus on
        direction: 'forward', 'backward', or 'both'
    
    Returns:
        List of descriptive prompts (deduplicated)
    """
    prompts = []
    
    # Group features by base name to avoid duplicates
    # Priority: mean > std > range > max > min
    stat_priority = {'_mean': 5, '_std': 4, '_range': 3, '_max': 2, '_min': 1}
    
    base_feature_map = defaultdict(list)  # base_name -> [(priority, full_feature_name), ...]
    
    # Focus on top discriminative features if provided
    if top_features:
        features_to_use = top_features  # Already ordered by importance
    else:
        # Sort by variance first (from forward/backward comparison), then by abs_change
        def get_sort_key(feat_name):
            if forward_backward_comparison and feat_name in forward_backward_comparison:
                comp = forward_backward_comparison[feat_name]
                # Compute variance between forward and backward values
                fwd_val = comp['forward']
                bwd_val = comp['backward']
                mean_val = (fwd_val + bwd_val) / 2
                variance = ((fwd_val - mean_val) ** 2 + (bwd_val - mean_val) ** 2) / 2
                abs_change = comp['abs_change']
                # Return tuple: (variance, abs_change) for sorting - variance first, then abs_change
                return (variance, abs_change)
            else:
                # Fallback to absolute value if no comparison available
                return (0, abs(subject_features.get(feat_name, 0)))
        
        features_to_use = sorted(
            subject_features.keys(),
            key=get_sort_key,
            reverse=True
        )
    
    # Group features by base name
    for feat_name in features_to_use[:30]:  # Look at more features initially
        if feat_name in subject_features:
            base_name = get_base_feature_name(feat_name)
            
            # Determine priority based on statistic type
            priority = 0
            for suffix, prio in stat_priority.items():
                if feat_name.endswith(suffix):
                    priority = prio
                    break
            
            base_feature_map[base_name].append((priority, feat_name))
    
    # For each base feature, select the highest priority variant
    selected_features = []
    for base_name, variants in base_feature_map.items():
        # Sort by priority (descending) and take the first one
        variants.sort(key=lambda x: x[0], reverse=True)
        selected_features.append(variants[0][1])  # Take feature name with highest priority
    
    # Generate prompts for selected features (limit to top 10 unique base features)
    for feat_name in selected_features[:10]:
        if feat_name in subject_features:
            value = subject_features[feat_name]
            
            # Use population statistics as primary reference for individual-specific prompts
            ref_value = None
            is_notable = False
            if population_stats and feat_name in population_stats:
                pop_mean = population_stats[feat_name]['mean']
                pop_std = population_stats[feat_name]['std']
                # Only generate prompt if individual is notably different from population (> 1 std)
                if pop_std > 1e-8:
                    z_score = (value - pop_mean) / pop_std
                    if abs(z_score) > 1.0:  # More than 1 standard deviation away
                        is_notable = True
                        ref_value = pop_mean
                else:
                    # If no variance, compare to mean
                    if abs(value - pop_mean) > 0.1 * abs(pop_mean) if abs(pop_mean) > 1e-8 else abs(value) > 0.1:
                        is_notable = True
                        ref_value = pop_mean
            
            # Fallback to forward-backward comparison if no population stats
            if not is_notable and forward_backward_comparison and feat_name in forward_backward_comparison:
                if direction == "forward":
                    ref_value = forward_backward_comparison[feat_name]['backward']
                elif direction == "backward":
                    ref_value = forward_backward_comparison[feat_name]['forward']
                is_notable = True
            
            # Only generate prompt if notable difference found
            if is_notable:
                prompt = generate_feature_based_prompt(feat_name, value, ref_value, direction)
                if prompt and len(prompt) > 5:  # Filter out very short prompts
                    prompts.append(prompt)
    
    # Add forward-backward comparison prompts if available
    if forward_backward_comparison and direction == "both":
        comparison_prompts = []
        for feat_name, comp_data in list(forward_backward_comparison.items())[:10]:
            abs_change = comp_data['abs_change']
            if abs_change > np.percentile([v['abs_change'] for v in forward_backward_comparison.values()], 75):
                fwd_val = comp_data['forward']
                bwd_val = comp_data['backward']
                
                if abs_change > 0.1 * max(abs(fwd_val), abs(bwd_val), 1):
                    base_name = get_base_feature_name(feat_name)
                    if fwd_val > bwd_val:
                        comparison_prompts.append(f"greater {base_name.replace('_', ' ')} in forward than backward gait")
                    else:
                        comparison_prompts.append(f"greater {base_name.replace('_', ' ')} in backward than forward gait")
        
        # Deduplicate comparison prompts
        comparison_prompts = deduplicate_prompts(comparison_prompts)
        prompts.extend(comparison_prompts[:3])  # Limit to top 3 comparison prompts
    
    # Final deduplication pass
    prompts = deduplicate_prompts(prompts)
    
    return prompts


def compute_healthy_reference_stats(
    healthy_data_dir: str,
    direction: str = "forward"
) -> Dict[str, Dict[str, float]]:
    """
    Compute population statistics from healthy subjects only.
    
    Args:
        healthy_data_dir: Directory containing CSV files of healthy subjects
        direction: 'forward', 'backward', or 'both'
    
    Returns:
        Dictionary with feature_name: {'mean', 'std', 'median'} mapping
    """
    print(f"\nLoading healthy reference data from: {healthy_data_dir}")
    
    # Get all CSV files from healthy reference directory
    csv_files = [f for f in os.listdir(healthy_data_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} healthy reference samples")
    
    # Extract features from all healthy subjects
    healthy_features_list = []
    for fname in csv_files:
        try:
            filepath = os.path.join(healthy_data_dir, fname)
            data_array, _ = load_pose_csv(filepath)
            features = extract_individual_features(data_array)
            healthy_features_list.append(features)
        except Exception as e:
            print(f"  Error loading {fname}: {e}")
            continue
    
    if not healthy_features_list:
        print("WARNING: No healthy reference data loaded!")
        return {}
    
    # Compute statistics across all features
    all_feature_names = set()
    for features in healthy_features_list:
        all_feature_names.update(features.keys())
    
    population_stats = {}
    for feat_name in all_feature_names:
        values = [f.get(feat_name, 0) for f in healthy_features_list if feat_name in f]
        if values:
            population_stats[feat_name] = {
                'mean': np.mean(values),
                'std': np.std(values) if len(values) > 1 else 0,
                'median': np.median(values),
                'count': len(values)
            }
    
    print(f"Computed statistics for {len(population_stats)} features from healthy population")
    return population_stats


def analyze_all_subjects(
    data_dir: str,
    output_dir: str = "individual_prompts",
    max_subjects: Optional[int] = None,
    healthy_reference_dir: Optional[str] = None
) -> Dict:
    """
    Analyze all subjects and generate individual-specific prompts.
    
    Args:
        data_dir: Directory containing going_forward and going_backward folders
        output_dir: Output directory for results
        max_subjects: Maximum number of subjects to process (None = all)
        healthy_reference_dir: Directory containing healthy reference data (if None, uses all subjects)
    
    Returns:
        Dictionary with subject_id: prompts mapping
    """
    os.makedirs(output_dir, exist_ok=True)
    
    forward_dir = os.path.join(data_dir, 'going_forward')
    backward_dir = os.path.join(data_dir, 'going_backward')
    
    # Get all unique subject IDs from filenames
    forward_files = [f for f in os.listdir(forward_dir) if f.endswith('.csv')]
    
    # Extract subject IDs (assuming format like "subjectID_*.csv" or similar)
    subject_ids = set()
    for fname in forward_files:
        # Try to extract subject ID (adjust pattern as needed)
        parts = fname.replace('.csv', '').split('_')
        if len(parts) > 0:
            subject_ids.add(parts[1])
    
    subject_ids = sorted(list(subject_ids))
    if max_subjects:
        subject_ids = subject_ids[:max_subjects]
    
    print(f"Found {len(subject_ids)} subjects")
    
    all_subject_features = []
    all_feature_names = set()
    subject_prompts = {}
    subject_data_summary = {}
    
    # Process each subject
    for i, subject_id in enumerate(subject_ids):
        print(f"\nProcessing subject {i+1}/{len(subject_ids)}: {subject_id}")
        
        try:
            # Load subject data
            subject_data = load_subject_data(data_dir, subject_id)
            
            if not subject_data['forward'] and not subject_data['backward']:
                print(f"  No data found for {subject_id}")
                continue
            
            # Process forward gait
            forward_features_agg = {}
            if subject_data['forward']:
                # Aggregate features across all forward samples
                forward_features_list = []
                for fname, data_array in subject_data['forward']:
                    features = extract_individual_features(data_array)
                    forward_features_list.append(features)
                    all_feature_names.update(features.keys())
                
                # Average features across samples
                for feat_name in all_feature_names:
                    values = [f.get(feat_name, 0) for f in forward_features_list]
                    forward_features_agg[feat_name] = np.mean(values)
            
            # Process backward gait
            backward_features_agg = {}
            if subject_data['backward']:
                backward_features_list = []
                for fname, data_array in subject_data['backward']:
                    features = extract_individual_features(data_array)
                    backward_features_list.append(features)
                
                for feat_name in all_feature_names:
                    values = [f.get(feat_name, 0) for f in backward_features_list]
                    backward_features_agg[feat_name] = np.mean(values)
            
            # Compare forward vs backward
            comparison = None
            if forward_features_agg and backward_features_agg:
                comparison = compare_forward_backward(forward_features_agg, backward_features_agg)
            
            # Store features for discriminative analysis
            combined_features = {**forward_features_agg, **backward_features_agg}
            all_subject_features.append(combined_features)
            
            # Store subject data (prompts will be generated after we identify top features)
            subject_prompts[subject_id] = {
                'features': combined_features,
                'forward_features': forward_features_agg,
                'backward_features': backward_features_agg,
                'comparison': comparison
            }
            
            subject_data_summary[subject_id] = {
                'num_forward_samples': len(subject_data['forward']),
                'num_backward_samples': len(subject_data['backward']),
            }
            
        except Exception as e:
            print(f"  Error processing {subject_id}: {e}")
            continue
    
    # Identify top discriminative features across all subjects
    all_feature_names = sorted(list(all_feature_names))
    if len(all_subject_features) > 1:
        top_features = identify_discriminative_features(all_subject_features, all_feature_names, top_k=20)
        top_feature_names = [feat[0] for feat in top_features]
        print(f"\nTop discriminative features: {top_feature_names[:10]}")
    else:
        top_feature_names = all_feature_names[:20]
    
    # Compute population statistics for individual-specific prompts
    # Use HEALTHY REFERENCE data if provided, otherwise use all subjects
    population_stats_forward = {}
    population_stats_backward = {}
    population_stats_combined = {}
    
    if healthy_reference_dir and os.path.exists(healthy_reference_dir):
        print("\n" + "=" * 70)
        print("Using HEALTHY POPULATION as normal reference")
        print("=" * 70)
        
        # Compute statistics from healthy reference data only
        population_stats_combined = compute_healthy_reference_stats(healthy_reference_dir, direction="both")
        # For forward/backward, use the same healthy reference (assuming healthy data doesn't distinguish direction)
        population_stats_forward = population_stats_combined.copy()
        population_stats_backward = population_stats_combined.copy()
        
        print(f"Loaded {len(population_stats_combined)} feature statistics from healthy reference")
        
    else:
        print("\n" + "=" * 70)
        print("WARNING: Using ALL SUBJECTS (including patients) as reference")
        print("This is NOT recommended for clinical analysis!")
        print("=" * 70)
        
        # Fallback: Compute stats from all subjects (NOT RECOMMENDED)
        if len(subject_prompts) > 1:
            # Collect forward features
            forward_features_list = []
            backward_features_list = []
            combined_features_list = []
            
            for subj_data in subject_prompts.values():
                if subj_data.get('forward_features'):
                    forward_features_list.append(subj_data['forward_features'])
                if subj_data.get('backward_features'):
                    backward_features_list.append(subj_data['backward_features'])
                if subj_data.get('features'):
                    combined_features_list.append(subj_data['features'])
            
            # Compute stats for forward features
            if forward_features_list:
                for feat_name in all_feature_names:
                    values = [f.get(feat_name, 0) for f in forward_features_list if feat_name in f]
                    if values:
                        population_stats_forward[feat_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values) if len(values) > 1 else 0,
                            'median': np.median(values)
                        }
            
            # Compute stats for backward features
            if backward_features_list:
                for feat_name in all_feature_names:
                    values = [f.get(feat_name, 0) for f in backward_features_list if feat_name in f]
                    if values:
                        population_stats_backward[feat_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values) if len(values) > 1 else 0,
                            'median': np.median(values)
                        }
            
            # Compute stats for combined features
            if combined_features_list:
                for feat_name in all_feature_names:
                    values = [f.get(feat_name, 0) for f in combined_features_list if feat_name in f]
                    if values:
                        population_stats_combined[feat_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values) if len(values) > 1 else 0,
                            'median': np.median(values)
                        }
        else:
            # If only one subject, use their values as reference
            if subject_prompts:
                subj_data = list(subject_prompts.values())[0]
                for feat_name in all_feature_names:
                    if subj_data.get('forward_features') and feat_name in subj_data['forward_features']:
                        val = subj_data['forward_features'][feat_name]
                        population_stats_forward[feat_name] = {'mean': val, 'std': 0, 'median': val}
                    if subj_data.get('backward_features') and feat_name in subj_data['backward_features']:
                        val = subj_data['backward_features'][feat_name]
                        population_stats_backward[feat_name] = {'mean': val, 'std': 0, 'median': val}
                    if subj_data.get('features') and feat_name in subj_data['features']:
                        val = subj_data['features'][feat_name]
                        population_stats_combined[feat_name] = {'mean': val, 'std': 0, 'median': val}
    
    # Generate all prompts with top features (now that we know which features are most discriminative)
    for subject_id in subject_prompts.keys():
        subj_data = subject_prompts[subject_id]
        # Generate forward prompts with ordered features
        if subj_data.get('forward_features'):
            subj_data['forward'] = generate_individual_prompts(
                subj_data['forward_features'],
                subj_data['comparison'],
                top_features=top_feature_names,
                direction="forward",
                population_stats=population_stats_forward
            )
        else:
            subj_data['forward'] = []
        
        # Generate backward prompts with ordered features
        if subj_data.get('backward_features'):
            subj_data['backward'] = generate_individual_prompts(
                subj_data['backward_features'],
                subj_data['comparison'],
                top_features=top_feature_names,
                direction="backward",
                population_stats=population_stats_backward
            )
        else:
            subj_data['backward'] = []
        
        # Generate both prompts with ordered features
        subj_data['both'] = generate_individual_prompts(
            subj_data['features'],
            subj_data['comparison'],
            top_features=top_feature_names,
            direction="both",
            population_stats=population_stats_combined
        )
        
        # Generate top feature prompts
        subj_data['top_feature_prompts'] = generate_individual_prompts(
            subj_data['features'],
            subj_data['comparison'],
            top_features=top_feature_names,
            direction="both",
            population_stats=population_stats_combined
        )
        
        # Update summary with prompt counts
        subject_data_summary[subject_id]['num_prompts_forward'] = len(subj_data['forward'])
        subject_data_summary[subject_id]['num_prompts_backward'] = len(subj_data['backward'])
        subject_data_summary[subject_id]['num_prompts_both'] = len(subj_data['both'])
    
    # Save results (only prompts, no numerical values)
    output_file = os.path.join(output_dir, "individual_gait_prompts.json")
    # Create a clean version with only prompts
    prompts_only = {}
    for subject_id, subj_data in subject_prompts.items():
        prompts_only[subject_id] = {
            'forward': subj_data.get('forward', []),
            'backward': subj_data.get('backward', []),
            'both': subj_data.get('both', []),
            'top_feature_prompts': subj_data.get('top_feature_prompts', [])
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'subject_prompts': prompts_only,
            'summary': subject_data_summary,
            'top_discriminative_features': top_feature_names,
            'metadata': {
                'num_subjects': len(subject_prompts),
                'total_features': len(all_feature_names),
                'reference_type': 'healthy_population' if healthy_reference_dir else 'all_subjects',
                'healthy_reference_dir': healthy_reference_dir if healthy_reference_dir else None
            }
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ Saved results to {output_file}")
    
    return {
        'subject_prompts': subject_prompts,
        'summary': subject_data_summary,
        'top_features': top_feature_names
    }


def parse_report_for_key_findings(report_path: str) -> Dict:
    """
    Parse Report_4class.md to extract key findings for prompt generation.
    
    Args:
        report_path: Path to Report_4class.md
    
    Returns:
        Dictionary with extracted findings
    """
    findings = {
        'top_features': [],
        'group_differences': {},
        'key_insights': []
    }
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract top discriminative features (from section 3.1)
    # Pattern to match feature rows: | Rank | Feature | p-value | Eta-squared (η²) | Effect Size |
    feature_pattern = r'\|\s*(\d+)\s*\|\s*\*\*([^*]+)\*\*\s*\|\s*([^\|]+)\s*\|\s*\*\*([^\*]+)\*\*\s*\|\s*([^\|]+)\s*\|'
    feature_matches = re.findall(feature_pattern, content)
    
    for match in feature_matches[:20]:  # Top 20 features
        rank, feature, pval, eta_sq, effect_size = match
        findings['top_features'].append({
            'rank': int(rank),
            'feature': feature.strip(),
            'eta_squared': float(eta_sq.strip()),
            'effect_size': effect_size.strip()
        })
    
    # Extract key insights from executive summary
    if 'Trunk lateral shift' in content:
        findings['key_insights'].append('trunk_lateral_shift_most_discriminative')
    if 'Trunk height variability' in content:
        findings['key_insights'].append('trunk_height_variability_significant')
    if 'Postural and trunk-related features' in content:
        findings['key_insights'].append('postural_features_key_discriminators')
    
    return findings


def generate_general_prompts_from_report(
    report_path: str = r"C:\Users\Olive\scoliosis_predictive_factors_pose_estimation\Report_4class.md",
    output_path: str = "general_gait_prompts_from_report.json"
) -> Dict:
    """
    Generate general/abstract gait prompts based on Report_4class.md findings.
    
    This function extracts key findings from the report and generates moderately
    specific but general prompts (not individual-specific) for scoliotic gait patterns.
    
    Args:
        report_path: Path to Report_4class.md
        output_path: Output JSON file path
    
    Returns:
        Dictionary with categorized and concise prompts
    """
    print("=" * 70)
    print("GENERAL GAIT PROMPTS GENERATION FROM REPORT")
    print("=" * 70)
    print(f"Reading report: {report_path}")
    
    # Read report content
    with open(report_path, 'r', encoding='utf-8') as f:
        report_content = f.read()
    
    # Extract key features from report (top 20 discriminative features)
    # Pattern to extract feature table rows
    # Example: | 1 | **trunk_lateral_shift_mean** | < 0.0001 *** | **0.142** | Medium-Large |
    feature_pattern = r'\|\s*\d+\s*\|\s*\*\*([^*]+)\*\*\s*\|\s*[^\|]+\s*\|\s*\*\*([^\*]+)\*\*\s*\|\s*([^\|]+)\s*\|'
    feature_matches = re.findall(feature_pattern, report_content)
    
    top_features = []
    for match in feature_matches[:20]:
        feature_name, eta_sq, effect_size = match
        top_features.append({
            'name': feature_name.strip(),
            'eta_squared': float(eta_sq.strip()),
            'effect_size': effect_size.strip()
        })
    
    print(f"Extracted {len(top_features)} top discriminative features")
    
    # Feature category mapping based on feature names
    def categorize_feature(feature_name: str) -> str:
        """Categorize a feature into prompt categories."""
        feature_lower = feature_name.lower()
        
        if 'trunk_lateral_shift' in feature_lower:
            return 'postural_features'
        elif 'trunk_height' in feature_lower:
            return 'postural_features'
        elif 'hip_angle' in feature_lower or 'hip_width' in feature_lower:
            return 'hip_mechanics'
        elif 'knee_angle' in feature_lower:
            return 'knee_kinematics'
        elif 'ankle' in feature_lower and ('diff' in feature_lower or 'vert' in feature_lower):
            return 'bilateral_asymmetry'
        elif 'shoulder' in feature_lower and ('tilt' in feature_lower or 'width' in feature_lower):
            return 'postural_features'
        elif 'hip_tilt' in feature_lower:
            return 'postural_features'
        elif 'step_width' in feature_lower:
            return 'bilateral_asymmetry'
        elif 'diff' in feature_lower or 'asymmetry' in feature_lower:
            return 'bilateral_asymmetry'
        else:
            return 'gait_variability'
    
    # Generate prompts based on features
    categorized_prompts = {
        'hip_mechanics': [],
        'knee_kinematics': [],
        'bilateral_asymmetry': [],
        'normal_gait': [],
        'scoliotic_gait': [],
        'thoracic_scoliotic_gait': [],
        'gait_variability': [],
        'postural_features': []
    }
    
    # Generate prompts for each top feature
    for feat in top_features:
        feat_name = feat['name']
        eta_sq = feat['eta_squared']
        category = categorize_feature(feat_name)
        
        # Parse feature components
        base_feature = feat_name.replace('_mean', '').replace('_std', '').replace('_range', '').replace('_min', '').replace('_max', '').replace('_iqr', '')
        stat_type = None
        if '_mean' in feat_name:
            stat_type = 'mean'
        elif '_std' in feat_name:
            stat_type = 'variability'
        elif '_range' in feat_name:
            stat_type = 'range'
        elif '_min' in feat_name:
            stat_type = 'minimum'
        elif '_max' in feat_name:
            stat_type = 'maximum'
        elif '_iqr' in feat_name:
            stat_type = 'variability'
        
        # Generate prompts based on feature type and effect size
        if eta_sq > 0.10:  # Medium-large effect
            if 'trunk_lateral_shift' in feat_name:
                categorized_prompts['postural_features'].extend([
                    "lateral trunk shift during gait",
                    "trunk lateral displacement pattern",
                    "asymmetric trunk alignment during gait",
                    "lateral trunk deviation from normal"
                ])
                categorized_prompts['scoliotic_gait'].extend([
                    "abnormal trunk lateral shift pattern",
                    "increased lateral trunk displacement"
                ])
            elif 'trunk_height' in feat_name:
                if stat_type == 'variability':
                    categorized_prompts['postural_features'].extend([
                        "variable trunk height during gait",
                        "inconsistent trunk height pattern",
                        "trunk height variability during gait"
                    ])
                    categorized_prompts['gait_variability'].extend([
                        "high variability in trunk height",
                        "inconsistent trunk height control"
                    ])
                else:
                    categorized_prompts['postural_features'].extend([
                        "altered trunk height during gait",
                        "trunk height deviation from normal"
                    ])
            elif 'hip_width' in feat_name or 'hip_angle' in feat_name:
                if 'hip_angle' in feat_name:
                    categorized_prompts['hip_mechanics'].extend([
                        "altered hip angle during gait",
                        "hip angle deviation from normal",
                        "abnormal hip joint motion"
                    ])
                if stat_type == 'minimum':
                    categorized_prompts['hip_mechanics'].extend([
                        "reduced hip width during gait",
                        "decreased minimum hip width"
                    ])
            elif 'ankle' in feat_name and ('vert' in feat_name or 'diff' in feat_name):
                categorized_prompts['bilateral_asymmetry'].extend([
                    "asymmetric ankle vertical positioning",
                    "bilateral ankle height differences",
                    "vertical ankle asymmetry during gait"
                ])
            elif 'shoulder' in feat_name:
                if stat_type == 'minimum':
                    categorized_prompts['postural_features'].extend([
                        "reduced shoulder width during gait",
                        "decreased minimum shoulder width"
                    ])
                else:
                    categorized_prompts['postural_features'].extend([
                        "altered shoulder width pattern",
                        "shoulder width deviation during gait"
                    ])
            elif 'step_width' in feat_name:
                if stat_type == 'variability':
                    categorized_prompts['bilateral_asymmetry'].extend([
                        "variable step width during gait",
                        "inconsistent step width pattern"
                    ])
                    categorized_prompts['gait_variability'].extend([
                        "high variability in step width",
                        "irregular step width pattern"
                    ])
    
    # Add group-specific prompts based on report findings
    # Single Thoracic specific
    categorized_prompts['thoracic_scoliotic_gait'].extend([
        "increased lateral trunk shift in thoracic scoliosis",
        "compensatory postural adjustments for thoracic curve",
        "thoracic curve affecting trunk mechanics",
        "abnormal trunk lateral displacement in thoracic scoliosis"
    ])
    
    # General scoliotic gait patterns
    categorized_prompts['scoliotic_gait'].extend([
        "asymmetric gait pattern with postural compensation",
        "abnormal trunk mechanics during gait",
        "compensatory gait adjustments for spinal curvature",
        "altered postural control during gait",
        "asymmetric joint kinematics with postural deviation"
    ])
    
    # Normal gait (baseline)
    categorized_prompts['normal_gait'].extend([
        "symmetric trunk alignment during gait",
        "stable trunk height pattern",
        "balanced postural control",
        "normal trunk lateral positioning",
        "consistent trunk mechanics during gait"
    ])
    
    # Bilateral asymmetry (general)
    categorized_prompts['bilateral_asymmetry'].extend([
        "asymmetric lower limb positioning",
        "bilateral differences in joint mechanics",
        "left-right limb asymmetry during gait",
        "asymmetric postural alignment"
    ])
    
    # Gait variability
    categorized_prompts['gait_variability'].extend([
        "inconsistent gait pattern",
        "high variability in postural features",
        "irregular motion patterns during gait",
        "fluctuating gait mechanics"
    ])
    
    # Deduplicate prompts within each category
    for category in categorized_prompts:
        categorized_prompts[category] = list(dict.fromkeys(categorized_prompts[category]))  # Preserves order
    
    # Generate concise prompts (most representative from each category)
    concise_prompts = []
    
    # Select top prompts from each category
    concise_prompts.extend(categorized_prompts['postural_features'][:4])
    concise_prompts.extend(categorized_prompts['hip_mechanics'][:3])
    concise_prompts.extend(categorized_prompts['bilateral_asymmetry'][:3])
    concise_prompts.extend(categorized_prompts['scoliotic_gait'][:3])
    concise_prompts.extend(categorized_prompts['normal_gait'][:2])
    concise_prompts.extend(categorized_prompts['gait_variability'][:2])
    concise_prompts.extend(categorized_prompts['knee_kinematics'][:2])
    concise_prompts.extend(categorized_prompts['thoracic_scoliotic_gait'][:2])
    
    # Deduplicate concise prompts
    concise_prompts = list(dict.fromkeys(concise_prompts))
    
    # Count totals
    total_categorized = sum(len(prompts) for prompts in categorized_prompts.values())
    
    # Create output structure
    output_data = {
        "categorized_prompts": categorized_prompts,
        "concise_prompts": concise_prompts,
        "metadata": {
            "total_categorized": total_categorized,
            "total_concise": len(concise_prompts),
            "categories": list(categorized_prompts.keys()),
            "description": "General gait analysis prompts for distinguishing scoliotic gait patterns",
            "based_on": "Report_4class.md findings",
            "key_features": [
                f"trunk_lateral_shift_mean (η²={top_features[0]['eta_squared']:.3f})" if top_features else "N/A",
                f"trunk_height_std (η²={top_features[1]['eta_squared']:.3f})" if len(top_features) > 1 else "N/A",
                f"trunk_height_range (η²={top_features[2]['eta_squared']:.3f})" if len(top_features) > 2 else "N/A"
            ],
            "generation_method": "extracted_from_report",
            "report_path": report_path
        }
    }
    
    # Save to file
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generated general prompts:")
    print(f"   - Total categorized: {total_categorized}")
    print(f"   - Total concise: {len(concise_prompts)}")
    print(f"   - Categories: {len(categorized_prompts)}")
    print(f"   - Saved to: {output_path}")
    print("=" * 70)
    
    return output_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate gait prompts (individual or general)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["individual", "general"],
        default="general",
        help="Generation mode: 'individual' for subject-specific prompts, 'general' for abstract prompts from report"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\Users\Olive\.spyder-py3\sz621_table",
        help="Directory containing going_forward and going_backward folders (for individual mode)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pytorch/get_prompts",
        help="Output directory for results (individual mode) or output file path (general mode)"
    )
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process (individual mode only)"
    )
    parser.add_argument(
        "--healthy_reference_dir",
        type=str,
        default=r"C:\Users\Olive\scoliosis_predictive_factors_pose_estimation\scoligait_kinetfactors\new_XAI_normal_table50",
        help="Directory containing healthy reference data (individual mode only)"
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default=r"C:\Users\Olive\scoliosis_predictive_factors_pose_estimation\Report_4class.md",
        help="Path to Report_4class.md (general mode only)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "general":
        # Generate general prompts from report
        output_path = args.output_dir if args.output_dir.endswith('.json') else os.path.join(args.output_dir, "general_gait_prompts_from_report.json")
        generate_general_prompts_from_report(
            report_path=args.report_path,
            output_path=output_path
        )
    else:
        # Generate individual prompts (original functionality)
        print("=" * 70)
        print("FINE-GRAINED INDIVIDUAL GAIT PROMPT GENERATION")
        print("=" * 70)
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Healthy reference: {args.healthy_reference_dir}")
        print("=" * 70)
        
        # Analyze all subjects
        results = analyze_all_subjects(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_subjects=args.max_subjects,
            healthy_reference_dir=args.healthy_reference_dir
        )
        
        print("\n" + "=" * 70)
