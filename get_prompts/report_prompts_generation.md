# Technical Report: Gait Analysis Prompt Generation Methodology

## 1. Introduction

This report describes the methodology for generating text-based prompts from gait analysis data derived from 2D pose estimation. The system produces two types of prompts: (1) **general prompts** extracted from statistical analysis reports, and (2) **individual-specific prompts** generated from subject-level pose time-series data. These prompts serve as semantic descriptors for gait patterns, enabling downstream applications in vision-language models, retrieval systems, and clinical documentation.

## 2. System Architecture

The prompt generation pipeline consists of three main components:

1. **Feature Extraction Module** (`gait_analysis_utils.py`): Extracts kinematic and postural features from pose time-series data
2. **Prompt Generation Module** (`generate_gait_prompts.py`): Converts features into natural language prompts
3. **Embedding Generation Module** (`get_prompts_embeddings.py`): Optional step to generate vector embeddings for prompts

## 3. Feature Extraction Methodology

### 3.1 Data Format

Input data consists of CSV files containing 2D pose keypoint coordinates extracted from video frames. Each file contains:
- A `frames` column for temporal indexing
- 34 columns representing 17 body landmarks × 2 coordinates (x, y)

The landmarks follow the YOLOv8 pose estimation format:
```
[nose, left_eye, right_eye, left_ear, right_ear,
 left_shoulder, right_shoulder, left_elbow, right_elbow,
 left_wrist, right_wrist, left_hip, right_hip,
 left_knee, right_knee, left_ankle, right_ankle]
```

Data is reshaped into arrays of shape `(T, 17, 2)` where T is the number of frames.

### 3.2 Per-Frame Feature Extraction

For each frame, the following features are computed:

**Distance Features:**
- `step_width`: Lateral distance between ankles
- `ankle_vert_diff`: Vertical distance between ankles
- `ankle_distance`: Euclidean distance between ankles
- `hip_width`: Distance between hip landmarks
- `shoulder_width`: Distance between shoulder landmarks

**Postural Features:**
- `trunk_height`: Vertical distance between shoulder center and hip center
- `shoulder_tilt`: Height difference between left and right shoulders
- `hip_tilt`: Height difference between left and right hips
- `trunk_lateral_shift`: Horizontal offset between shoulder center and hip center

**Joint Angle Features:**
- `left_knee_angle`, `right_knee_angle`: Angle at knee joint (hip-knee-ankle)
- `left_hip_angle`, `right_hip_angle`: Angle at hip joint (shoulder-hip-knee)
- `left_elbow_angle`, `right_elbow_angle`: Angle at elbow joint

### 3.3 Temporal Aggregation

For each per-frame feature, temporal statistics are computed across the entire gait sequence:

- **Central Tendency**: `mean`, `median`
- **Variability**: `std`, `range`, `iqr` (interquartile range)
- **Extremes**: `min`, `max`
- **Distribution Shape**: `skewness`

This yields feature names in the format: `{base_feature}_{statistic}` (e.g., `trunk_lateral_shift_mean`, `trunk_height_std`).

### 3.4 Bilateral Asymmetry Features

Additional features quantify left-right asymmetry:

- `knee_angle_diff_mean`: Mean difference between left and right knee angles
- `knee_angle_diff_std`: Standard deviation of knee angle differences
- `knee_angle_diff_abs_mean`: Mean absolute difference
- `hip_angle_diff_mean`, `hip_angle_diff_std`, `hip_angle_diff_abs_mean`: Similar metrics for hip angles

## 4. Prompt Generation Methodologies

### 4.1 General Prompts from Statistical Reports

**Purpose**: Generate abstract, category-based prompts from population-level statistical findings.

**Input**: Markdown report (e.g., `Report_4class.md`) containing ANOVA or similar statistical analysis results with feature rankings, p-values, and effect sizes (η²).

**Algorithm**:

1. **Feature Extraction from Report**:
   - Parse markdown tables using regex pattern matching
   - Extract top 20 discriminative features with their effect sizes
   - Pattern: `| Rank | **feature_name** | p-value | **eta_squared** | Effect Size |`

2. **Feature Categorization**:
   Features are mapped to semantic categories:
   - `postural_features`: Trunk-related features (lateral shift, height, tilt)
   - `hip_mechanics`: Hip angle and width features
   - `knee_kinematics`: Knee angle features
   - `bilateral_asymmetry`: Left-right difference features
   - `gait_variability`: Standard deviation and range features
   - `scoliotic_gait`: General abnormal gait patterns
   - `thoracic_scoliotic_gait`: Thoracic-specific patterns
   - `normal_gait`: Baseline normal patterns

3. **Prompt Generation Rules**:
   For each top feature (η² > 0.10, medium-large effect):
   - Parse feature name to extract base feature and statistic type
   - Generate category-specific prompts based on feature semantics
   - Example: `trunk_lateral_shift_mean` → prompts: "lateral trunk shift during gait", "trunk lateral displacement pattern"

4. **Output Structure**:
   ```json
   {
     "categorized_prompts": {
       "category_name": ["prompt1", "prompt2", ...]
     },
     "concise_prompts": ["selected", "representative", "prompts"],
     "metadata": {...}
   }
   ```

### 4.2 Individual-Specific Prompts

**Purpose**: Generate personalized prompts for each subject based on their deviation from healthy population norms.

**Input**: 
- Subject pose CSV files (forward and backward gait)
- Healthy reference population data directory

**Algorithm**:

1. **Data Loading**:
   - Load all forward and backward gait samples for each subject
   - Extract subject ID from filename pattern: `*_{subject_id}_*.csv`

2. **Feature Extraction**:
   - For each sample, extract temporal and asymmetry features
   - Aggregate features across multiple samples (mean) to get subject-level features

3. **Healthy Reference Statistics**:
   - Load healthy population data
   - Compute population statistics for each feature:
     ```python
     population_stats[feature] = {
         'mean': μ,
         'std': σ,
         'median': median
     }
     ```

4. **Discriminative Feature Identification**:
   - Compute coefficient of variation (CV) for each feature across all subjects
   - Score = CV × range
   - Select top 20 most discriminative features

5. **Individual Prompt Generation**:
   
   For each subject:
   
   a. **Feature Selection**:
      - Prioritize features by statistic type: `mean > std > range > max > min`
      - Group by base feature name to avoid duplicates
      - Select top 10 unique base features
   
   b. **Notable Difference Detection**:
      - For each selected feature, compute z-score:
        ```python
        z_score = (subject_value - pop_mean) / pop_std
        ```
      - Generate prompt only if `|z_score| > 1.0` (more than 1 standard deviation from mean)
   
   c. **Prompt Construction**:
      - Parse feature name to extract:
        - Base feature (e.g., `trunk_lateral_shift`)
        - Statistic type (`mean`, `std`, `range`, etc.)
        - Side (`left_`, `right_`, or bilateral)
        - Category mapping (e.g., `trunk_lateral_shift` → "trunk lateral shift")
      
      - Determine modifier based on deviation:
        ```python
        if diff > 0:
            modifier = "increased" (or "high variability in" for std)
        else:
            modifier = "reduced" (or "low variability in" for std)
        ```
      
      - Construct prompt: `{modifier} {stat_type} {side} {category} during {direction} gait`
      - Example: "increased left hip extension during forward gait"
   
   d. **Forward-Backward Comparison**:
      - If both forward and backward data available:
        - Compare feature values between directions
        - Generate prompts for features with >75th percentile absolute change
        - Example: "greater trunk lateral shift in forward than backward gait"
   
   e. **Deduplication**:
      - Remove prompts with identical base concepts (ignoring modifiers)
      - Final output: up to 10-15 unique prompts per subject

6. **Output Structure**:
   ```json
   {
     "subject_prompts": {
       "subject_id": {
         "forward": ["prompt1", "prompt2", ...],
         "backward": ["prompt1", "prompt2", ...],
         "both": ["combined", "prompts"],
         "top_feature_prompts": ["most", "discriminative"]
       }
     },
     "summary": {...},
     "top_discriminative_features": [...]
   }
   ```

## 5. Implementation Details

### 5.1 Key Functions

**Feature Extraction**:
- `load_pose_csv(filepath)`: Loads and reshapes CSV data
- `extract_gait_features_per_frame(frame_data)`: Computes per-frame features
- `extract_temporal_features(data)`: Aggregates temporal statistics
- `extract_bilateral_asymmetry_features(data)`: Computes asymmetry metrics

**Prompt Generation**:
- `generate_feature_based_prompt(feature_name, value, reference, direction)`: Converts single feature to prompt
- `generate_individual_prompts(subject_features, ...)`: Generates full prompt set for subject
- `generate_general_prompts_from_report(report_path)`: Extracts prompts from statistical report
- `compute_healthy_reference_stats(healthy_data_dir)`: Computes population statistics

**Utility Functions**:
- `deduplicate_prompts(prompts)`: Removes semantically similar prompts
- `identify_discriminative_features(all_features, top_k)`: Ranks features by discriminative power

### 5.2 Thresholds and Parameters

- **Z-score threshold**: `|z_score| > 1.0` for notable differences
- **Percentage difference threshold**: `10%` for relative comparisons
- **Top features**: Top 20 discriminative features, top 10 for prompt generation
- **Statistic priority**: `mean (5) > std (4) > range (3) > max (2) > min (1)`

### 5.3 Clinical Considerations

- **Healthy Reference Population**: The system uses a separate healthy population dataset as the reference, avoiding bias from including patient data in the normal distribution
- **Direction-Specific Analysis**: Separate prompts for forward and backward gait capture direction-dependent patterns
- **Effect Size Filtering**: Only features with medium-large effect sizes (η² > 0.10) are used for general prompts

## 6. Output Formats

### 6.1 JSON Structure

Both prompt types output JSON files with:
- Prompt lists organized by category or subject
- Metadata including feature statistics, reference information, and generation parameters
- Summary statistics (number of prompts, subjects processed, etc.)

### 6.2 Embedding Generation (Optional)

The `get_prompts_embeddings.py` module can generate vector embeddings for prompts using:
- **Sentence Transformers** (default): `all-MiniLM-L6-v2` (384-dim) or `all-mpnet-base-v2` (768-dim)
- **CLIP text encoder**: For vision-language model compatibility
- **OpenAI embeddings**: API-based embeddings

Output formats: NumPy arrays (`.npy`), PyTorch tensors (`.pt`), HDF5 (`.h5`), or JSON.

## 7. Usage Workflow

### 7.1 General Prompts

```bash
python generate_gait_prompts.py \
    --mode general \
    --report_path "path/to/Report_4class.md" \
    --output_dir "output/"
```

**Example**:
```bash
python generate_gait_prompts.py \
    --mode general \
    --report_path "C:\Users\Olive\scoliosis_predictive_factors_pose_estimation\Report_4class.md" \
    --output_dir "."
```

This generates `general_gait_prompts_from_report.json` in the current directory.

### 7.2 Individual Prompts

```bash
python generate_gait_prompts.py \
    --mode individual \
    --data_dir "path/to/pose_data/" \
    --healthy_reference_dir "path/to/healthy_data/" \
    --output_dir "output/"
```

**Example**:
```bash
python generate_gait_prompts.py \
    --mode individual \
    --data_dir "C:\Users\Olive\.spyder-py3\sz621_table" \
    --healthy_reference_dir "C:\Users\Olive\scoliosis_predictive_factors_pose_estimation\scoligait_kinetfactors\new_XAI_normal_table50" \
    --output_dir "."
```

**Required Directory Structure**:
```
data_dir/
├── going_forward/
│   ├── subject_10_sample1.csv
│   ├── subject_10_sample2.csv
│   └── ...
└── going_backward/
    ├── subject_10_sample1.csv
    └── ...
```

This generates `individual_gait_prompts.json` containing prompts for each subject.

### 7.3 Embedding Generation

```bash
python get_prompts_embeddings.py \
    --json_path "output/general_gait_prompts_from_report.json" \
    --method sentence_transformers \
    --output_dir "embeddings/"
```

**For General Prompts**:
```bash
python get_prompts_embeddings.py \
    --json_path "general_gait_prompts_from_report.json" \
    --method sentence_transformers \
    --output_dir "embeddings/"
```

**For Individual Prompts**:
```bash
python get_prompts_embeddings.py \
    --use_individual \
    --json_path "individual_gait_prompts.json" \
    --method sentence_transformers \
    --output_dir "embeddings/"
```

**Available Methods**:
- `sentence_transformers` (default, recommended)
- `clip` (for vision-language models)
- `openai` (requires API key)

## 8. File Structure

```
pytorch/get_prompts/
├── generate_gait_prompts.py          # Main prompt generation script
├── gait_analysis_utils.py            # Feature extraction utilities
├── get_prompts_embeddings.py         # Embedding generation script
├── general_gait_prompts_from_report.json    # Output: General prompts
├── individual_gait_prompts.json            # Output: Individual prompts
└── PROMPT_GENERATION_METHODOLOGY.md  # This document
```

## 9. Key Design Decisions

### 9.1 Why Two Prompt Types?

- **General Prompts**: Capture population-level patterns from statistical analysis, useful for category-level classification and general pattern recognition
- **Individual Prompts**: Provide subject-specific characterization based on deviation from healthy norms, enabling personalized analysis

### 9.2 Healthy Reference Population

Using a separate healthy population as reference ensures:
- Clinically meaningful comparisons
- Avoids bias from including patient data in "normal" distribution
- Enables detection of deviations from true healthy patterns

### 9.3 Feature Prioritization

The statistic priority system (`mean > std > range > max > min`) ensures:
- Most clinically relevant features (means) are prioritized
- Variability features (std) are included when significant
- Redundant prompts are avoided by selecting highest-priority statistic per base feature

### 9.4 Z-Score Threshold

The `|z_score| > 1.0` threshold (1 standard deviation) ensures:
- Only clinically notable deviations generate prompts
- Reduces noise from normal variation
- Balances sensitivity and specificity

## 10. Limitations and Future Work

### Current Limitations

- **Fixed Feature Mappings**: Prompt generation relies on predefined feature-to-text mappings, limiting flexibility
- **Language**: Limited to English language prompts
- **Pose Format**: Assumes standard 17-keypoint YOLOv8 pose format
- **Threshold Selection**: Fixed thresholds may not be optimal for all populations

### Future Enhancements

- **LLM Integration**: Use large language models for more natural, context-aware prompt generation
- **Multi-language Support**: Extend to multiple languages for international clinical use
- **Adaptive Thresholds**: Implement population-specific threshold selection based on feature distributions
- **Clinical Terminology**: Integration with standardized clinical terminology (e.g., SNOMED CT)
- **Interactive Refinement**: Allow clinicians to refine or validate generated prompts
- **Temporal Patterns**: Generate prompts that capture temporal evolution of gait patterns

## 11. Validation and Quality Assurance

### Prompt Quality Metrics

- **Deduplication Rate**: Percentage of prompts removed due to similarity
- **Coverage**: Number of unique features represented in prompts
- **Clinical Relevance**: Manual review by domain experts (recommended)

### Testing Recommendations

1. **Unit Tests**: Test feature extraction on known pose data
2. **Integration Tests**: Verify end-to-end prompt generation pipeline
3. **Validation**: Compare generated prompts with expert annotations
4. **Reproducibility**: Ensure consistent outputs across runs

## 12. References

- YOLOv8 Pose Estimation: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- Sentence Transformers: [sbert.net](https://www.sbert.net/)
- CLIP: [OpenAI CLIP](https://github.com/openai/CLIP)

## 13. Conclusion

This methodology provides a systematic approach to converting quantitative gait analysis features into semantically meaningful text prompts. The two-tier system (general and individual) enables both population-level pattern recognition and subject-specific characterization, supporting applications in clinical documentation, vision-language model training, and gait analysis research.

The system's design emphasizes:
- **Clinical Validity**: Uses healthy reference populations and clinically meaningful thresholds
- **Flexibility**: Supports both abstract and personalized prompt generation
- **Extensibility**: Modular design allows for future enhancements and integration

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Maintained By**: Research Team
