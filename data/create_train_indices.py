import pandas as pd
import json
import numpy as np

# Paths
excel_path = r"C:\Users\Olive\Desktop\video_retrival\video_retrival\Label_DKgait_4video.xlsx"
test_indices_path = None
output_path = r"C:\Users\Olive\Desktop\Nature_Communication\code_video\pytorch\sft_dataset\dk_indices.json"

# Load Excel file
print("Loading Excel file...")
df = pd.read_excel(excel_path)
print(f"Total rows in Excel: {len(df)}")

# Load test indices (optional)
if test_indices_path is not None:
    print("Loading test indices...")
    with open(test_indices_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    test_indices_set = set(item['index'] for item in test_data['test'])
    print(f"Test indices count: {len(test_indices_set)}")
else:
    print("No test indices file provided - including all data")
    test_indices_set = set()

# Filter rows where both M_cobb_l and M_cobb_r are not empty
print("Filtering rows with valid M_cobb_l and M_cobb_r...")
df_valid = df[(df['M_cobb_l'].notna()) & (df['M_cobb_r'].notna())].copy()
print(f"Rows with valid M_cobb values: {len(df_valid)}")

# Create training indices
train_indices = []
skipped_empty = 0
skipped_test = 0

for _, row in df_valid.iterrows():
    index = int(row['File No.'])
    
    # Skip if index is in test set
    if index in test_indices_set:
        skipped_test += 1
        continue
    
    # Get label vector [M_cobb_l, M_cobb_r]
    m_cobb_l = float(row['M_cobb_l'])
    m_cobb_r = float(row['M_cobb_r'])
    label = [m_cobb_l, m_cobb_r]
    
    train_indices.append({
        "index": index,
        "label": label
    })

# Sort by index for consistency
train_indices.sort(key=lambda x: x["index"])

# Create output structure
output = {
    "train": train_indices
}

# Save to JSON file
print(f"\nSaving training indices to: {output_path}")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSummary:")
print(f"  Total valid rows (with M_cobb values): {len(df_valid)}")
print(f"  Skipped (in test set): {skipped_test}")
print(f"  Training indices created: {len(train_indices)}")
print(f"  Output saved to: {output_path}")

# Show sample entries
if train_indices:
    print(f"\nSample training entries (first 5):")
    for entry in train_indices[:5]:
        print(f"  Index: {entry['index']}, Label: {entry['label']}")

