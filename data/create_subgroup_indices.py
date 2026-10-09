import json
from collections import Counter
from pathlib import Path

import pandas as pd

# Paths
excel_path = Path(r"C:\Users\Olive\Desktop\video_retrival\video_retrival\Label_DKgait_4video.xlsx")
output_path = Path(r"C:\Users\Olive\Desktop\Nature_Style\code_video\pytorch\sft_dataset\subgroup_indices.json")

print("Loading Excel file...")
if not excel_path.exists():
    raise FileNotFoundError(f"Excel file not found: {excel_path}")

df = pd.read_excel(excel_path)
print(f"Total rows in Excel: {len(df)}")

# Identify subgroup columns, e.g. M_1, M_xxx, S_1, S_xxx
m_cols = [c for c in df.columns if str(c).startswith("M_")][:3]
s_cols = [c for c in df.columns if str(c).startswith("S_")][:3]

if not m_cols and not s_cols:
    raise ValueError('No columns found that start with "M_" or "S_".')

print(f"Found {len(m_cols)} M_* columns and {len(s_cols)} S_* columns")

# Convert subgroup columns to numeric so string values like "1" are handled.
subgroup_cols = m_cols + s_cols
subgroup_numeric = df[subgroup_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# Keep rows where at least one M_* OR S_* column is 1.
has_subgroup = subgroup_numeric.eq(1).any(axis=1)
df_selected = df[has_subgroup].copy()

print(f"Rows with at least one subgroup value == 1: {len(df_selected)}")

subgroup_indices = []
for _, row in df_selected.iterrows():
    row_vals = pd.to_numeric(row[subgroup_cols], errors="coerce").fillna(0)
    ones_count = int((row_vals == 1).sum())
    label1 = "single" if ones_count == 1 else "multi"

    # Merge M_/S_ location signals.
    has_thoracic = (
        pd.to_numeric(row.get("M_thoracic"), errors="coerce") == 1
        or pd.to_numeric(row.get("S_thoracic"), errors="coerce") == 1
    )
    has_lumbar = (
        pd.to_numeric(row.get("M_lumbar"), errors="coerce") == 1
        or pd.to_numeric(row.get("S_lumbar"), errors="coerce") == 1
    )
    has_thoracolumbar = (
        pd.to_numeric(row.get("M_thoralumbar"), errors="coerce") == 1
        or pd.to_numeric(row.get("S_thoralumbar"), errors="coerce") == 1
    )

    # Only assign location label for single-curve cases.
    if label1 == "single":
        if has_thoracic or has_thoracolumbar:
            label2 = "thoracic"
        elif has_lumbar:
            label2 = "lumbar"
        else:
            label2 = None
    else:
        label2 = None

    m_cobb_l = pd.to_numeric(row.get("M_cobb_l"), errors="coerce")
    m_cobb_r = pd.to_numeric(row.get("M_cobb_r"), errors="coerce")
    cobb_label = [
        float(m_cobb_l) if pd.notna(m_cobb_l) else 0.0,
        float(m_cobb_r) if pd.notna(m_cobb_r) else 0.0,
    ]

    subgroup_indices.append(
        {
            "index": int(row["File No."]),
            "label": cobb_label,
            "label1": label1,
            "label2": label2,
        }
    )

subgroup_indices.sort(key=lambda x: x["index"])

output = {"subgroup": subgroup_indices}

label1_counts = Counter(item["label1"] for item in subgroup_indices)
label2_counts = Counter(item["label2"] for item in subgroup_indices if item["label2"] is not None)

print(f"\nSaving subgroup indices to: {output_path}")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\nSummary:")
print(f"  Total rows in Excel: {len(df)}")
print(f"  Subgroup indices created: {len(subgroup_indices)}")
print(f"  single: {label1_counts.get('single', 0)}")
print(f"  multi: {label1_counts.get('multi', 0)}")
print(f"  thoracic: {label2_counts.get('thoracic', 0)}")
print(f"  lumbar: {label2_counts.get('lumbar', 0)}")
print(f"  Output saved to: {output_path}")

if subgroup_indices:
    print("\nSample subgroup entries (first 5):")
    for entry in subgroup_indices[:5]:
        print(
            f"  Index: {entry['index']}, "
            f"label: {entry['label']}, "
            f"label1: {entry['label1']}, label2: {entry['label2']}"
        )
