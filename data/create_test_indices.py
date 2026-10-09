import os
import json
import re
import pandas as pd
from pathlib import Path

# Paths
excel_path = r"C:\Users\Olive\Desktop\video_retrival\video_retrival\Label_SZgait_4video.xlsx"
output_path = r"C:\Users\Olive\Desktop\Nature_Communication\code_video\pytorch\sft_dataset\test_indices.json"

# Define the folders
folders = {
    "normal": r"C:\Users\Olive\scoliosis_predictive_factors_pose_estimation\scoligait_kinetfactors\new_XAI_normal_table50",
    "double": r"C:\Users\Olive\scoliosis_predictive_factors_pose_estimation\scoligait_kinetfactors\new_XAI_double_table50",
    "single_lumbar": r"C:\Users\Olive\scoliosis_predictive_factors_pose_estimation\scoligait_kinetfactors\new_XAI_single_lumbar_table50",
    "single_thoracic": r"C:\Users\Olive\scoliosis_predictive_factors_pose_estimation\scoligait_kinetfactors\new_XAI_single_thoracic_table50"
}

# Pattern to match filenames: sz_{index}_step_1
pattern = re.compile(r'sz_(\d+)_step_1')

# Load Excel file to get M_cobb_l and M_cobb_r values
print("Loading Excel file...")
if not os.path.exists(excel_path):
    print(f"Warning: Excel file not found: {excel_path}")
    print("Will use default vector labels [0.0, 0.0] for normal and [1.0, 1.0] for others")
    df = None
    label_dict = {}
else:
    df = pd.read_excel(excel_path)
    print(f"Total rows in Excel: {len(df)}")
    
    # Create a dictionary mapping File No. to [M_cobb_l, M_cobb_r]
    label_dict = {}
    for _, row in df.iterrows():
        file_no = int(row['File No.'])
        m_cobb_l = row['M_cobb_l']
        m_cobb_r = row['M_cobb_r']
        
        # Check if both values are valid (not NaN)
        if pd.notna(m_cobb_l) and pd.notna(m_cobb_r):
            label_dict[file_no] = [float(m_cobb_l), float(m_cobb_r)]
        else:
            # If values are missing, use default based on folder type
            label_dict[file_no] = None
    
    print(f"Loaded {len(label_dict)} entries with valid M_cobb values from Excel")

# Dictionary to store indices with their labels
test_indices = []
seen_indices = set()  # Track indices already added to avoid duplicates
missing_labels = 0
default_labels = 0
missing_indices_in_excel = []  # Track indices not found in Excel

# Process each folder
for folder_type, folder_path in folders.items():
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        continue
    
    # Scan files in the folder
    files = os.listdir(folder_path)
    for filename in files:
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            
            # Skip if this index was already added (avoid duplicates)
            if index in seen_indices:
                print(f"Skipping duplicate: {filename} -> index: {index} (already added)")
                continue
            
            # Get label vector from Excel if available
            if df is not None and index in label_dict:
                label = label_dict[index]
                if label is not None:
                    # Use vector label from Excel
                    test_indices.append({
                        "index": index,
                        "label": label
                    })
                    seen_indices.add(index)
                    print(f"Found: {filename} -> index: {index}, label: {label}")
                else:
                    # Missing values in Excel, use default
                    default_label = [0.0, 0.0] if folder_type == "normal" else [11.0, 11.0]
                    test_indices.append({
                        "index": index,
                        "label": default_label
                    })
                    seen_indices.add(index)
                    default_labels += 1
                    missing_indices_in_excel.append(index)
                    print(f"Found: {filename} -> index: {index}, label: {default_label} (default - missing in Excel)")
            else:
                # Excel not available or index not found, use default based on folder type
                default_label = [0.0, 0.0] if folder_type == "normal" else [11.0, 11.0]
                test_indices.append({
                    "index": index,
                    "label": default_label
                })
                seen_indices.add(index)
                missing_labels += 1
                if df is not None:  # Only track if Excel was loaded
                    missing_indices_in_excel.append(index)
                print(f"Found: {filename} -> index: {index}, label: {default_label} (default - not in Excel)")

# Sort by index for consistency
test_indices.sort(key=lambda x: x["index"])

# Create the output structure
output = {
    "test": test_indices
}

# Save to JSON file
print(f"\nSaving test indices to: {output_path}")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSummary:")
print(f"  Total test indices found: {len(test_indices)}")
print(f"  Labels from Excel: {len(test_indices) - missing_labels - default_labels}")
print(f"  Default labels (missing in Excel): {default_labels}")
print(f"  Default labels (not in Excel): {missing_labels}")
print(f"  Output saved to: {output_path}")

# Print missing indices in Excel
if missing_indices_in_excel:
    missing_indices_in_excel.sort()
    print(f"\nWarning: Missing indices in Excel ({len(missing_indices_in_excel)} total):")
    print(f"  Indices: {missing_indices_in_excel}")
    # Print in a more readable format (grouped)
    if len(missing_indices_in_excel) > 20:
        print(f"\n  First 20: {missing_indices_in_excel[:20]}")
        print(f"  ... and {len(missing_indices_in_excel) - 20} more")
    else:
        print(f"  All missing indices: {missing_indices_in_excel}")
else:
    print(f"\nAll indices found in Excel!")

# Show sample entries
if test_indices:
    print(f"\nSample test entries (first 5):")
    for entry in test_indices[:5]:
        print(f"  Index: {entry['index']}, Label: {entry['label']}")

