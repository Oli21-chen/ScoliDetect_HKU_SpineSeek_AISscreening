"""
Load PKL data, get knowledge map (batch, 96, 238),
sum over dim=-1, plot curves (x = 96 timestamps) and save in this script directory.
Default target is dataset/test_dk_pkl^1.
"""

import os
import sys
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List

# Add project root for imports (if needed later)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Compatibility for pickles produced with older NumPy internals.
if "numpy._core.numeric" not in sys.modules:
    sys.modules["numpy._core.numeric"] = np.core.numeric


def _resolve_pkl_path(entry: dict, pkl_data_dir_abs: str) -> str:
    """Resolve pkl_path: absolute, then fallback to patches/."""
    raw = entry.get("pkl_path", "")
    if not os.path.isabs(raw):
        path = os.path.normpath(os.path.abspath(raw))
    else:
        path = os.path.normpath(raw)
    if not os.path.exists(path):
        pid = entry.get("patch_id")
        if pid is None:
            pid = Path(entry.get("pkl_path", "")).stem.replace("patch_", "")
        fallback = os.path.join(pkl_data_dir_abs, "patches", f"patch_{pid}.pkl")
        if os.path.exists(fallback):
            path = fallback
    return path


def load_km_curves(pkl_data_dir_abs: str, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load all patches from pkl_data_dir, stack knowledge_map (N, 96, 238),
    sum over dim=-1 -> (N, 96). Return (x, curve_mean, curve_std, n_samples).
    """
    metadata_path = os.path.join(pkl_data_dir_abs, "patch_metadata.pkl")
    if not os.path.exists(metadata_path):
        patches_dir = os.path.join(pkl_data_dir_abs, "patches")
        patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.pkl")))
        if not patch_files:
            raise FileNotFoundError(f"No metadata and no patch files in {patches_dir}")
        patch_metadata = []
        for pkl_path in patch_files:
            try:
                with open(pkl_path, "rb") as f:
                    patch_data = pickle.load(f)
                patch_id = Path(pkl_path).stem.replace("patch_", "")
                patch_metadata.append({"patch_id": patch_id, "pkl_path": pkl_path})
            except Exception as e:
                print(f"Warning [{name}]: skip {pkl_path}: {e}")
                continue
    else:
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        patch_metadata = metadata["patch_metadata"]

    km_list = []
    for entry in patch_metadata:
        pkl_path = _resolve_pkl_path(entry, pkl_data_dir_abs)
        if not os.path.exists(pkl_path):
            continue
        try:
            with open(pkl_path, "rb") as f:
                patch_data = pickle.load(f)
            km = patch_data["knowledge_map"]
            if hasattr(km, "numpy"):
                km = km.numpy()
            km_list.append(np.asarray(km, dtype=np.float64))
        except Exception as e:
            print(f"Warning [{name}]: skip {pkl_path}: {e}")
            continue

    if not km_list:
        raise ValueError(f"No knowledge_map loaded for {name}")

    km_all = np.stack(km_list, axis=0)  # (N, 96, 238)
    km_sum = km_all.sum(axis=-1)  # (N, 96)
    curve_mean = np.mean(km_sum, axis=0)
    curve_std = np.std(km_sum, axis=0)
    x = np.arange(96)
    return x, curve_mean, curve_std, len(km_list)


def main():
    # Keep paths explicit and parser-free: this script focuses on test_dk_pkl^1.
    data_root = os.path.join(parent_dir, "dataset")
    dirs_to_compare = [
        ("test_dk_pkl^1", "test_dk_pkl^1"),
    ]

    results: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for dir_name, label in dirs_to_compare:
        pkl_data_dir_abs = os.path.abspath(os.path.normpath(os.path.join(data_root, dir_name)))
        if not os.path.isdir(pkl_data_dir_abs):
            print(f"Skip (not found): {pkl_data_dir_abs}")
            continue
        try:
            x, curve_mean, curve_std, n = load_km_curves(pkl_data_dir_abs, label)
            results.append((label, x, curve_mean, curve_std, n))
            print(f"{label}: shape (batch={n}, 96, 238), curve computed.")
        except Exception as e:
            print(f"Error loading {label}: {e}")
            continue

    if not results:
        print("Error: No datasets loaded.")
        return

    plt.figure(figsize=(9, 5))
    colors = ["steelblue", "coral"]
    for i, (label, x, curve_mean, curve_std, n) in enumerate(results):
        c = colors[i % len(colors)]
        plt.plot(x, curve_mean, color=c, linewidth=2, label=f"{label} (n={n})")
        plt.fill_between(x, curve_mean - curve_std, curve_mean + curve_std, alpha=0.25, color=c)
    plt.xlabel("Timestamp (0-95)")
    plt.ylabel("Sum over 238 features")
    plt.title("KM distribution: test_dk_pkl^1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(current_dir, "km_distribution_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
