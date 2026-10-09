# -*- coding: utf-8 -*-
"""
Step 1.5 — invariant pose CSV from step_1 2D keypoints.

Pipeline per clip:
  1. Load {para}_{id}_step_1.csv (YOLO COCO-17, camera 2D)
  2. Apply calibration mode (2D smooth/align or VideoPose3D rotation/frontal)
  3. Save calibrated CSV with mode-specific suffix

Downstream sampling (02_Sampling) and GetAllFeatures are unchanged; point
table_path at the directory containing calibrated CSVs manually.

@author: Olive
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

current_dir = os.path.dirname(os.path.abspath(__file__))
pytorch_dir = os.path.join(os.path.dirname(current_dir), "pytorch")
sys.path.insert(0, pytorch_dir)

from utils.canonical_pose import (  # noqa: E402
    bone_length_cv,
    calibrate_align2d,
    calibrate_align2d_scale,
    calibrate_smooth_fill,
    calibrate_smooth_only,
    lift_align_project_frontal,
    lift_rotate_calibrate_pixels,
)
from utils.invariant_pose_io import (  # noqa: E402
    flatten_joints_to_rows,
    load_step1_csv,
    reshape_flat_to_joints,
    save_step1_inv_csv,
    step1_calibrated_output_path,
    step1_input_path,
)
from utils.videopose3d_lift import (  # noqa: E402
    DEFAULT_IMAGE_SIZE,
    VideoPose3DLifter,
)

Mode = Literal[
    "rotation",
    "frontal",
    "smooth",
    "smooth_fill",
    "align2d",
    "align2d_scale",
    "rot_yaw_only",
    "rot_no_yflip",
    "rot_yaw_no_yflip",
    "rot_per_frame",
]

DEFAULT_SUFFIX_BY_MODE = {
    "rotation": "_step_1_rot.csv",
    "frontal": "_step_1_inv.csv",
    "smooth": "_step_1_smooth.csv",
    "smooth_fill": "_step_1_smooth_fill.csv",
    "align2d": "_step_1_align2d.csv",
    "align2d_scale": "_step_1_align2d_scale.csv",
    "rot_yaw_only": "_step_1_rot_yaw.csv",
    "rot_no_yflip": "_step_1_rot_no_yflip.csv",
    "rot_yaw_no_yflip": "_step_1_rot_yaw_no_yflip.csv",
    "rot_per_frame": "_step_1_rot_per_frame.csv",
}

ROTATION_MODES = {
    "rotation",
    "rot_yaw_only",
    "rot_no_yflip",
    "rot_yaw_no_yflip",
    "rot_per_frame",
}


def discover_indices(table_path: str, para_name: str) -> List[int]:
    """Find indices with step_1 CSV present."""
    suffix = "_step_1.csv"
    prefix = f"{para_name}_"
    extra_suffixes = tuple(DEFAULT_SUFFIX_BY_MODE.values())
    indices: List[int] = []
    if not os.path.isdir(table_path):
        return indices
    for fname in os.listdir(table_path):
        if not fname.startswith(prefix):
            continue
        if not fname.endswith(suffix) and not any(fname.endswith(es) for es in extra_suffixes):
            continue
        if fname.endswith(suffix):
            mid = fname[len(prefix) : -len(suffix)]
        else:
            matched = False
            for es in extra_suffixes:
                if fname.endswith(es):
                    mid = fname[len(prefix) : -len(es)]
                    matched = True
                    break
            if not matched:
                continue
        try:
            indices.append(int(mid))
        except ValueError:
            continue
    return sorted(set(indices))


def compute_clip_stats(pose_2d: np.ndarray) -> Dict[str, float]:
    """Bone-length CV on upper arm and thigh segments."""
    return {
        "l_arm_cv": bone_length_cv(pose_2d, 5, 7),
        "r_arm_cv": bone_length_cv(pose_2d, 6, 8),
        "l_thigh_cv": bone_length_cv(pose_2d, 11, 13),
        "r_thigh_cv": bone_length_cv(pose_2d, 12, 14),
    }


def count_low_confidence_frames(values: np.ndarray) -> float:
    """Fraction of frames with any zero-valued coordinate (YOLO miss)."""
    joints = reshape_flat_to_joints(values)
    bad = np.any(np.abs(joints) < 1e-6, axis=(1, 2))
    return float(np.mean(bad))


def apply_calibration(
    pose_2d: np.ndarray,
    lifter: Optional[VideoPose3DLifter],
    image_size: Tuple[int, int],
    *,
    mode: Mode,
) -> np.ndarray:
    if mode == "smooth":
        return calibrate_smooth_only(pose_2d)
    if mode == "smooth_fill":
        return calibrate_smooth_fill(pose_2d)
    if mode == "align2d":
        return calibrate_align2d(pose_2d)
    if mode == "align2d_scale":
        return calibrate_align2d_scale(pose_2d)
    if mode == "frontal":
        if lifter is None:
            raise ValueError("VideoPose3D lifter required for frontal mode")
        return lift_align_project_frontal(pose_2d, lifter, image_size=image_size)
    if mode in ROTATION_MODES:
        if lifter is None:
            raise ValueError(f"VideoPose3D lifter required for {mode}")
        return lift_rotate_calibrate_pixels(
            pose_2d,
            lifter,
            image_size=image_size,
            camera_y_down=mode not in ("rot_no_yflip", "rot_yaw_no_yflip"),
            yaw_only=mode in ("rot_yaw_only", "rot_yaw_no_yflip"),
            per_frame_rot=mode == "rot_per_frame",
        )
    raise ValueError(f"Unknown mode: {mode}")


def process_one_clip(
    input_csv: str,
    output_csv: str,
    lifter: Optional[VideoPose3DLifter],
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    *,
    mode: Mode = "rotation",
) -> Dict[str, float]:
    """Run invariant pose pipeline on one CSV."""
    values, frames = load_step1_csv(input_csv)
    pose_2d = reshape_flat_to_joints(values)
    stats_before = compute_clip_stats(pose_2d)
    pose_2d_out = apply_calibration(pose_2d, lifter, image_size, mode=mode)
    stats_after = compute_clip_stats(pose_2d_out)
    save_step1_inv_csv(output_csv, flatten_joints_to_rows(pose_2d_out), frames)

    return {
        "frames": len(frames),
        "low_conf_frac": count_low_confidence_frames(values),
        **{f"before_{k}": v for k, v in stats_before.items()},
        **{f"after_{k}": v for k, v in stats_after.items()},
    }


def validate_schema(output_csv: str) -> None:
    """Ensure output loads via gait_sampling_v3-style reader."""
    from utils.gait_sampling_v3 import load_pose_values_from_csv

    values = load_pose_values_from_csv(output_csv)
    joints = reshape_flat_to_joints(values)
    if joints.shape[1:] != (17, 2):
        raise ValueError(f"Schema check failed: {joints.shape}")


def run_diagnose(
    table_path: str,
    output_path: str,
    para_name: str,
    indices: List[int],
    lifter: Optional[VideoPose3DLifter],
    *,
    output_suffix: str,
    mode: Mode = "rotation",
    max_clips: int = 10,
) -> None:
    """Compare KM statistics between raw and calibrated CSVs on a subset."""
    try:
        import matplotlib.pyplot as plt
        from utils.knowledge_map import GetAllFeatures
    except ImportError as exc:
        print(f"Diagnose skipped (missing dependency): {exc}")
        return

    km_raw, km_cal = [], []
    used = 0
    image_size = DEFAULT_IMAGE_SIZE
    for idx in indices:
        if used >= max_clips:
            break
        raw_path = step1_input_path(table_path, para_name, idx)
        cal_path = step1_calibrated_output_path(output_path, para_name, idx, output_suffix)
        if not os.path.isfile(raw_path):
            continue
        if not os.path.isfile(cal_path):
            process_one_clip(raw_path, cal_path, lifter, image_size=image_size, mode=mode)

        raw_vals = load_step1_csv(raw_path)[0]
        cal_vals = load_step1_csv(cal_path)[0]
        km_raw.append(np.squeeze(GetAllFeatures(raw_vals)))
        km_cal.append(np.squeeze(GetAllFeatures(cal_vals)))
        used += 1

    if not km_raw:
        print("Diagnose: no clips processed.")
        return

    raw_stack = np.stack(km_raw, axis=0)
    cal_stack = np.stack(km_cal, axis=0)
    raw_mean = raw_stack.mean(axis=(0, 1))
    cal_mean = cal_stack.mean(axis=(0, 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(raw_mean, label="raw step_1 KM mean", alpha=0.8)
    ax.plot(cal_mean, label=f"calibrated KM mean ({mode})", alpha=0.8)
    ax.set_xlabel("KM feature index")
    ax.set_ylabel("Mean value")
    ax.legend()
    ax.set_title(f"KM comparison ({used} clips)")
    out_plot = os.path.join(output_path, f"{para_name}_km_diagnose.png")
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(out_plot, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnose plot saved: {out_plot}")


def run_plot(
    table_path: str,
    output_path: str,
    para_name: str,
    index: int,
    lifter: Optional[VideoPose3DLifter],
    *,
    output_suffix: str,
    mode: Mode = "rotation",
) -> None:
    """Save stick-figure overlay for raw vs calibrated 2D."""
    import matplotlib.pyplot as plt

    raw_path = step1_input_path(table_path, para_name, index)
    cal_path = step1_calibrated_output_path(output_path, para_name, index, output_suffix)
    if not os.path.isfile(cal_path):
        process_one_clip(raw_path, cal_path, lifter, mode=mode)

    raw = reshape_flat_to_joints(load_step1_csv(raw_path)[0])
    cal = reshape_flat_to_joints(load_step1_csv(cal_path)[0])

    skeleton = [
        (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for ax, pose, title in zip(axes, [raw, cal], ["raw step_1", f"calibrated ({mode})"]):
        mid = len(pose) // 2
        p = pose[mid]
        for a, b in skeleton:
            ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], "b-", lw=1.5)
        ax.scatter(p[:, 0], p[:, 1], s=12, c="r")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{para_name}_{index} mid-frame skeleton")
    out_plot = os.path.join(output_path, f"{para_name}_{index}_pose_compare.png")
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(out_plot, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_plot}")


def run_batch(
    para_name: str,
    table_path: str,
    output_path: str,
    indices: List[int],
    lifter: Optional[VideoPose3DLifter],
    *,
    overwrite: bool = False,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mode: Mode = "rotation",
    output_suffix: str = "_step_1_rot.csv",
) -> None:
    os.makedirs(output_path, exist_ok=True)
    for idx in indices:
        input_csv = step1_input_path(table_path, para_name, idx)
        output_csv = step1_calibrated_output_path(output_path, para_name, idx, output_suffix)
        if not os.path.isfile(input_csv):
            print(f"  skip {para_name}_{idx}: missing input")
            continue
        if os.path.isfile(output_csv) and not overwrite:
            print(f"  skip {para_name}_{idx}: output exists (use --overwrite)")
            continue
        try:
            stats = process_one_clip(
                input_csv,
                output_csv,
                lifter,
                image_size=image_size,
                mode=mode,
            )
            validate_schema(output_csv)
            print(
                f"  ok {para_name}_{idx}: frames={stats['frames']:.0f} "
                f"l_thigh_cv {stats['before_l_thigh_cv']:.4f}->{stats['after_l_thigh_cv']:.4f} "
                f"low_conf={stats['low_conf_frac']:.2%}"
            )
        except Exception as exc:
            print(f"  fail {para_name}_{idx}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1.5: pose calibration (2D smooth/align or VideoPose3D rotation/frontal)"
    )
    parser.add_argument("--para-name", default="sz", help="Dataset prefix (pk, sz, ...)")
    parser.add_argument(
        "--table-path",
        required=True,
        help="Directory containing *_step_1.csv files",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output directory for calibrated CSVs (default: table-path/step_1_inv)",
    )
    parser.add_argument(
        "--mode",
        choices=list(DEFAULT_SUFFIX_BY_MODE.keys()),
        default="rotation",
        help="Calibration mode",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Output filename suffix (default: mode-specific)",
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="Subject indices to process (default: --all)",
    )
    parser.add_argument("--all", action="store_true", help="Process all discovered indices")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="VideoPose3D checkpoint path (default: third_party/.../pretrained_h36m_detectron_coco.bin)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=DEFAULT_IMAGE_SIZE[0],
        help="Assumed video width for 2D normalization (step_1 default 1080)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=DEFAULT_IMAGE_SIZE[1],
        help="Assumed video height for 2D normalization (step_1 default 1920)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Plot mean KM profile raw vs calibrated on subset",
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=None,
        metavar="INDEX",
        help="Save skeleton comparison plot for one index",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (cuda/cpu, default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode: Mode = args.mode
    output_suffix = args.output_suffix or DEFAULT_SUFFIX_BY_MODE[mode]
    output_path = args.output_path or os.path.join(args.table_path, "step_1_inv")
    image_size = (args.image_width, args.image_height)

    if args.all or not args.indices:
        indices = discover_indices(args.table_path, args.para_name)
    else:
        indices = args.indices

    print(f"Step 1.5 pose calibration ({mode}): {args.para_name}, {len(indices)} clips")
    print(f"  input:  {args.table_path}")
    print(f"  output: {output_path}")
    print(f"  suffix: {output_suffix}")

    needs_lifter = mode in ROTATION_MODES or mode == "frontal"
    lifter = VideoPose3DLifter(checkpoint_path=args.checkpoint, device=args.device) if needs_lifter else None

    run_batch(
        args.para_name,
        args.table_path,
        output_path,
        indices,
        lifter,
        overwrite=args.overwrite,
        image_size=image_size,
        mode=mode,
        output_suffix=output_suffix,
    )

    if args.diagnose:
        run_diagnose(
            args.table_path,
            output_path,
            args.para_name,
            indices,
            lifter,
            output_suffix=output_suffix,
            mode=mode,
        )

    if args.plot is not None:
        run_plot(
            args.table_path,
            output_path,
            args.para_name,
            args.plot,
            lifter,
            output_suffix=output_suffix,
            mode=mode,
        )


if __name__ == "__main__":
    main()
