import json
import os
import re
from pathlib import Path

train_path = r"C:\Users\Olive\Desktop\Nature_Communication\code_video\pytorch\sft_dataset\train_indices.json"
test_path = r"C:\Users\Olive\Desktop\Nature_Communication\code_video\pytorch\sft_dataset\test_indices.json"
normal_folder = r"C:\Users\Olive\Desktop\Nature_Communication\code_video\pytorch\get_silhouette\scoliosis1k-plus\normal"

# Load both files
with open(train_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)
with open(test_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Create sets of indices
train_set = set(x['index'] for x in train_data['train'])
test_set = set(x['index'] for x in test_data['test'])

# Check for overlap
overlap = train_set & test_set

print(f"Training indices: {len(train_set)}")
print(f"Test indices: {len(test_set)}")
print(f"Overlap: {len(overlap)}")
if overlap:
    print(f"Overlap indices: {sorted(overlap)}")
else:
    print("No overlap - perfect!")

# Verify positive/negative samples in test_indices with binary threshold of 10
BINARY_THRESHOLD = 12.0
positive_count = 0
negative_count = 0

for item in test_data['test']:
    label = item['label']
    # Check if any value in the label is >= threshold
    if max(label) >= BINARY_THRESHOLD:
        positive_count += 1
    else:
        negative_count += 1

print(f"\n=== Test Set Binary Classification (threshold = {BINARY_THRESHOLD}) ===")
print(f"Positive samples (max(label) >= {BINARY_THRESHOLD}): {positive_count}")
print(f"Negative samples (max(label) < {BINARY_THRESHOLD}): {negative_count}")
print(f"Total test samples: {len(test_set)}")
print(f"Positive ratio: {positive_count / len(test_set) * 100:.2f}%")
print(f"Negative ratio: {negative_count / len(test_set) * 100:.2f}%")

# Verify positive/negative samples in train_indices with binary threshold of 10
train_positive_count = 0
train_negative_count = 0

for item in train_data['train']:
    label = item['label']
    # Check if any value in the label is >= threshold
    if max(label) >= BINARY_THRESHOLD:
        train_positive_count += 1
    else:
        train_negative_count += 1

print(f"\n=== Train Set Binary Classification (threshold = {BINARY_THRESHOLD}) ===")
print(f"Positive samples (max(label) >= {BINARY_THRESHOLD}): {train_positive_count}")
print(f"Negative samples (max(label) < {BINARY_THRESHOLD}): {train_negative_count}")
print(f"Total train samples: {len(train_set)}")
print(f"Positive ratio: {train_positive_count / len(train_set) * 100:.2f}%")
print(f"Negative ratio: {train_negative_count / len(train_set) * 100:.2f}%")

# Check for contradictions: samples in normal folder with labels >= 10
print(f"\n=== Checking for Contradictions ===")
print(f"Normal folder: {normal_folder}")

if not os.path.exists(normal_folder):
    print(f"Warning: Normal folder not found: {normal_folder}")
else:
    # Get all directories in normal folder and extract indices
    normal_items = os.listdir(normal_folder)
    # Pattern to match: sz_{index} (directory names)
    pattern = re.compile(r'sz_(\d+)$')
    normal_indices = set()
    
    for item in normal_items:
        item_path = os.path.join(normal_folder, item)
        # Check if it's a directory
        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                index = int(match.group(1))
                normal_indices.add(index)
    
    print(f"Found {len(normal_indices)} unique indices in normal folder")
    
    # Find indices in normal folder that also exist in train_indices.json
    normal_in_train = normal_indices & train_set
    if normal_in_train:
        print(f"\n=== Indices in Normal Folder that are in Train Set ===")
        print(f"Found {len(normal_in_train)} indices in normal folder that also exist in train_indices.json")
        
        # Get full information including labels from train_indices
        normal_in_train_details = []
        for item in train_data['train']:
            if item['index'] in normal_in_train:
                normal_in_train_details.append({
                    'index': item['index'],
                    'label': item['label']
                })
        
        # Sort by index
        normal_in_train_details.sort(key=lambda x: x['index'])
        
        print(f"\n   Indices and their labels from train_indices.json:")
        for entry in normal_in_train_details:
            print(f"   Index: {entry['index']:4d}, Label: {entry['label']}")
        
        # Save to file
        output_file = r"C:\Users\Olive\Desktop\Nature_Communication\code_video\pytorch\sft_dataset\normal_in_train.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(normal_in_train_details, f, indent=2, ensure_ascii=False)
        print(f"\n   Results saved to: {output_file}")
    else:
        print(f"\n[OK] No indices from normal folder found in train_indices.json")
    
    # Find contradictions: indices in normal folder but with labels >= 10 in test_indices
    contradictions = []
    THRESHOLD = 12.0
    
    for item in test_data['test']:
        index = item['index']
        label = item['label']
        max_label = max(label)
        
        # Check if this index is in normal folder but has label >= 10
        if index in normal_indices and max_label >= THRESHOLD:
            contradictions.append({
                'index': index,
                'label': label,
                'max_label': max_label
            })
    
    if contradictions:
        print(f"\n[WARNING] Found {len(contradictions)} CONTRADICTIONS:")
        print(f"   These indices are in the 'normal' folder but have labels >= {THRESHOLD}")
        print(f"\n   Contradictory entries:")
        for cont in sorted(contradictions, key=lambda x: x['index']):
            print(f"   Index: {cont['index']:4d}, Label: {cont['label']}, Max: {cont['max_label']:.1f}")
        
        # Save to file
        output_file = r"C:\Users\Olive\Desktop\Nature_Communication\code_video\pytorch\sft_dataset\contradictions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(contradictions, f, indent=2, ensure_ascii=False)
        print(f"\n   Contradictions saved to: {output_file}")
    else:
        print(f"\n[OK] No contradictions found! All samples in normal folder have labels < {THRESHOLD}")

