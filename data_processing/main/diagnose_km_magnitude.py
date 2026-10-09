# -*- coding: utf-8 -*-
"""
Diagnose KM magnitude gap between raw table CSVs and step_1_inv lift CSVs.

Outputs to infer_data_11_lift/diagnosis/:
  - q1_repro_diff.csv
  - domain_breakdown.json
  - domain_sum_profiles.png
  - pose_size_histogram.png
  - investigation_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

current_dir = os.path.dirname(os.path.abspath(__file__))
pytorch_dir = os.path.join(os.path.dirname(current_dir), "pytorch")
sys.path.insert(0, pytorch_dir)

from utils.canonical_pose import (  # noqa: E402
    lift_align_project_frontal,
    lift_rotate_calibrate_pixels,
)
from utils.invariant_pose_io import (  # noqa: E402
    load_step1_csv,
    reshape_flat_to_joints,
)
from utils.knowledge_map import FullBodyPoseEmbedder, GetAllFeatures  # noqa: E402
from utils.videopose3d_lift import DEFAULT_IMAGE_SIZE, VideoPose3DLifter  # noqa: E402

DOMAIN_SLICES = {
    "motion": (0, 34),
    "skeleton_dist": (34, 140),
    "skeleton_ang": (140, 172),
    "signal": (172, 238),
}

KM_GATE_THRESHOLDS = {
    "motion_corr": 0.85,
    "skeleton_dist_corr": 0.85,
    "skeleton_ang_corr": 0.85,
    "pose_size_ratio_min": 0.8,
    "pose_size_ratio_max": 1.2,
}


def evaluate_km_gate(cohort_agg: Dict[str, Any]) -> Dict[str, Any]:
    """Return pass/fail and per-criterion details for experiment pre-screen."""
    motion_corr = float(cohort_agg.get("motion_corr_cohort_mean", float("nan")))
    dist_corr = float(cohort_agg.get("skeleton_dist_corr_cohort_mean", float("nan")))
    ang_corr = float(cohort_agg.get("skeleton_ang_corr_cohort_mean", float("nan")))
    raw_ps = float(cohort_agg.get("pose_size_raw_median_cohort_median", 1.0))
    cal_ps = float(cohort_agg.get("pose_size_cal_median_cohort_median", 1.0))
    ps_ratio = raw_ps / max(cal_ps, 1e-8)

    checks = {
        "motion_corr": motion_corr > KM_GATE_THRESHOLDS["motion_corr"],
        "skeleton_dist_corr": dist_corr > KM_GATE_THRESHOLDS["skeleton_dist_corr"],
        "skeleton_ang_corr": ang_corr > KM_GATE_THRESHOLDS["skeleton_ang_corr"],
        "pose_size_ratio": (
            KM_GATE_THRESHOLDS["pose_size_ratio_min"]
            <= ps_ratio
            <= KM_GATE_THRESHOLDS["pose_size_ratio_max"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "motion_corr": motion_corr,
        "skeleton_dist_corr": dist_corr,
        "skeleton_ang_corr": ang_corr,
        "pose_size_ratio": ps_ratio,
        "checks": checks,
        "thresholds": KM_GATE_THRESHOLDS,
    }


def discover_paired_indices(
    table_dir: str,
    lift_dir: str,
    para_name: str = "sz",
    *,
    calibrated_suffix: str = "_step_1_rot.csv",
) -> List[int]:
    indices: List[int] = []
    raw_suffix = "_step_1.csv"
    prefix = f"{para_name}_"
    for fname in os.listdir(table_dir):
        if not fname.startswith(prefix) or not fname.endswith(raw_suffix):
            continue
        mid = fname[len(prefix) : -len(raw_suffix)]
        try:
            idx = int(mid)
        except ValueError:
            continue
        cal_path = os.path.join(lift_dir, f"{prefix}{idx}{calibrated_suffix}")
        if os.path.isfile(cal_path):
            indices.append(idx)
    return sorted(indices)


def compute_pose_size_series(pose_2d: np.ndarray, embedder: FullBodyPoseEmbedder) -> np.ndarray:
    """Per-frame pose_size from FullBodyPoseEmbedder logic."""
    sizes = []
    for t in range(len(pose_2d)):
        lm = np.asarray(pose_2d[t], dtype=np.float64)
        sizes.append(float(embedder._get_pose_size(lm, embedder._torso_size_multiplier)))
    return np.asarray(sizes)


def compute_embed_coor_series(pose_2d: np.ndarray, embedder: FullBodyPoseEmbedder) -> np.ndarray:
    """Per-frame embed_coors (34,) after embedder normalization."""
    rows = []
    for t in range(len(pose_2d)):
        _, _, ed = embedder(pose_2d[t])
        rows.append(ed.reshape(-1))
    return np.stack(rows, axis=0)


def split_km_domains(km: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    for name, (a, b) in DOMAIN_SLICES.items():
        out[name] = km[:, a:b]
    return out


def domain_sum_profile(km: np.ndarray) -> Dict[str, np.ndarray]:
    domains = split_km_domains(km)
    return {name: block.sum(axis=1) for name, block in domains.items()}


def analyze_clip(
    index: int,
    table_dir: str,
    lift_dir: str,
    para_name: str,
    embedder: FullBodyPoseEmbedder,
    lifter: Optional[VideoPose3DLifter] = None,
    *,
    calibrated_suffix: str = "_step_1_rot.csv",
    check_repro: bool = False,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Dict[str, Any]:
    raw_path = os.path.join(table_dir, f"{para_name}_{index}_step_1.csv")
    cal_path = os.path.join(lift_dir, f"{para_name}_{index}{calibrated_suffix}")

    raw_vals, _ = load_step1_csv(raw_path)
    cal_vals, _ = load_step1_csv(cal_path)
    raw_pose = reshape_flat_to_joints(raw_vals)
    cal_pose = reshape_flat_to_joints(cal_vals)

    result: Dict[str, Any] = {"index": index}

    # Q1 reproducibility
    if check_repro and lifter is not None:
        if calibrated_suffix == "_step_1_rot.csv":
            recomputed = lift_rotate_calibrate_pixels(raw_pose, lifter, image_size=image_size)
        else:
            recomputed = lift_align_project_frontal(raw_pose, lifter, image_size=image_size)
        max_diff = float(np.max(np.abs(recomputed - cal_pose)))
        mean_diff = float(np.mean(np.abs(recomputed - cal_pose)))
        result["q1_max_abs_diff"] = max_diff
        result["q1_mean_abs_diff"] = mean_diff

    # Input stats
    result["input_raw_xy_min"] = float(np.min(raw_pose))
    result["input_raw_xy_max"] = float(np.max(raw_pose))
    result["input_cal_xy_min"] = float(np.min(cal_pose))
    result["input_cal_xy_max"] = float(np.max(cal_pose))

    # pose_size
    raw_ps = compute_pose_size_series(raw_pose, embedder)
    cal_ps = compute_pose_size_series(cal_pose, embedder)
    result["pose_size_raw_mean"] = float(np.mean(raw_ps))
    result["pose_size_raw_median"] = float(np.median(raw_ps))
    result["pose_size_cal_mean"] = float(np.mean(cal_ps))
    result["pose_size_cal_median"] = float(np.median(cal_ps))

    # embed_coors
    raw_ec = compute_embed_coor_series(raw_pose, embedder)
    cal_ec = compute_embed_coor_series(cal_pose, embedder)
    result["embed_coor_raw_mean"] = float(np.mean(raw_ec))
    result["embed_coor_raw_std"] = float(np.std(raw_ec))
    result["embed_coor_cal_mean"] = float(np.mean(cal_ec))
    result["embed_coor_cal_std"] = float(np.std(cal_ec))

    # KM
    km_raw = np.squeeze(GetAllFeatures(raw_vals))
    km_cal = np.squeeze(GetAllFeatures(cal_vals))
    result["km_total_sum_mean_raw"] = float(np.mean(km_raw.sum(axis=1)))
    result["km_total_sum_mean_cal"] = float(np.mean(km_cal.sum(axis=1)))

    dom_raw = split_km_domains(km_raw)
    dom_cal = split_km_domains(km_cal)

    for name in DOMAIN_SLICES:
        r_block = dom_raw[name]
        c_block = dom_cal[name]
        result[f"{name}_mean_abs_raw"] = float(np.mean(np.abs(r_block)))
        result[f"{name}_mean_abs_cal"] = float(np.mean(np.abs(c_block)))
        result[f"{name}_sum_mean_raw"] = float(np.mean(r_block.sum(axis=1)))
        result[f"{name}_sum_mean_cal"] = float(np.mean(c_block.sum(axis=1)))
        if r_block.size > 1 and c_block.size > 1:
            corr = float(np.corrcoef(r_block.ravel(), c_block.ravel())[0, 1])
        else:
            corr = float("nan")
        result[f"{name}_corr"] = corr
        result[f"{name}_l2"] = float(np.linalg.norm(r_block - c_block))

    result["km_raw"] = km_raw
    result["km_cal"] = km_cal
    result["raw_pose_size_series"] = raw_ps
    result["cal_pose_size_series"] = cal_ps
    return result


def aggregate_cohort(clip_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize scalar metrics across clips."""
    keys = [
        k for k in clip_results[0]
        if k not in ("km_raw", "km_cal", "raw_pose_size_series", "cal_pose_size_series")
        and not (k.startswith("km_") and k not in ("km_total_sum_mean_raw", "km_total_sum_mean_cal"))
    ]
    agg: Dict[str, Any] = {"n_clips": len(clip_results)}
    for key in keys:
        if key == "index":
            continue
        vals = [r[key] for r in clip_results if key in r and not isinstance(r[key], str)]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals:
            agg[f"{key}_cohort_mean"] = float(np.mean(vals))
            agg[f"{key}_cohort_median"] = float(np.median(vals))
    return agg


def plot_domain_sum_profiles(
    clip_results: List[Dict[str, Any]],
    output_path: str,
    *,
    cal_label: str = "SZ rot",
) -> None:
    """4-panel mean domain sum profiles raw vs calibrated."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panel_keys = ["motion", "skeleton_dist", "skeleton_ang", "signal"]
    titles = ["Motion [0:34)", "Skeleton distances [34:140)", "Skeleton angles [140:172)", "Signal [172:238)"]

    for ax, key, title in zip(axes.ravel(), panel_keys, titles):
        raw_profiles, cal_profiles = [], []
        a, b = DOMAIN_SLICES[key]
        for r in clip_results:
            raw_profiles.append(r["km_raw"][:, a:b].sum(axis=1))
            cal_profiles.append(r["km_cal"][:, a:b].sum(axis=1))
        raw_mean = np.mean(np.stack(raw_profiles, axis=0), axis=0)
        cal_mean = np.mean(np.stack(cal_profiles, axis=0), axis=0)
        t = np.arange(len(raw_mean))
        ax.plot(t, raw_mean, color="#1f77b4", label="SZ raw", linewidth=1.5)
        ax.plot(t, cal_mean, color="#2ca02c", label=cal_label, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("KM timestep")
        ax.set_ylabel("Domain sum")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-domain KM sum profiles (cohort mean)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pose_size_histogram(
    clip_results: List[Dict[str, Any]],
    output_path: str,
    *,
    cal_label: str = "rot",
) -> None:
    raw_all = np.concatenate([r["raw_pose_size_series"] for r in clip_results])
    cal_all = np.concatenate([r["cal_pose_size_series"] for r in clip_results])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(raw_all, bins=50, alpha=0.6, label=f"raw (n={len(raw_all)})", color="#1f77b4")
    ax.hist(cal_all, bins=50, alpha=0.6, label=f"{cal_label} (n={len(cal_all)})", color="#2ca02c")
    ax.set_xlabel("pose_size per frame")
    ax.set_ylabel("Count")
    ax.set_title(f"pose_size distribution: raw table vs {cal_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_q1_csv(clip_results: List[Dict[str, Any]], output_path: str) -> None:
    rows = [r for r in clip_results if "q1_max_abs_diff" in r]
    if not rows:
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "q1_max_abs_diff", "q1_mean_abs_diff"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "index": r["index"],
                "q1_max_abs_diff": r["q1_max_abs_diff"],
                "q1_mean_abs_diff": r["q1_mean_abs_diff"],
            })


def write_report(
    output_path: str,
    cohort_agg: Dict[str, Any],
    q1_samples: List[Dict[str, Any]],
    calibration_recommendation: str,
    *,
    cal_label: str = "rotation-calibrated",
) -> None:
    lines = [
        "# KM Magnitude Investigation Report",
        "",
        f"## Comparison: raw table vs {cal_label}",
        "",
        "## Q1: Raw table → 1.5 pipeline reproduces saved CSVs?",
        "",
    ]
    if q1_samples:
        for r in q1_samples:
            lines.append(
                f"- sz_{r['index']}: max_abs_diff={r['q1_max_abs_diff']:.6f}, "
                f"mean_abs_diff={r['q1_mean_abs_diff']:.6f}"
            )
        max_d = max(r["q1_max_abs_diff"] for r in q1_samples)
        lines.append(f"\n**Conclusion Q1:** {'YES' if max_d < 1e-3 else 'CLOSE' if max_d < 0.01 else 'CHECK'} — "
                       f"max repro diff across samples = {max_d:.6f}")
    else:
        lines.append("Q1 repro not run (no lifter).")

    lines.extend([
        "",
        "## Q2: Domain magnitude breakdown (cohort means)",
        "",
        f"| Domain | mean_abs raw | mean_abs cal | sum_mean raw | sum_mean cal | corr |",
        "|--------|-------------|-------------|-------------|-------------|------|",
    ])
    for name in ["motion", "skeleton_dist", "skeleton_ang", "signal"]:
        corr = cohort_agg.get(f"{name}_corr_cohort_mean", float("nan"))
        lines.append(
            f"| {name} | "
            f"{cohort_agg.get(f'{name}_mean_abs_raw_cohort_mean', 0):.2f} | "
            f"{cohort_agg.get(f'{name}_mean_abs_cal_cohort_mean', 0):.2f} | "
            f"{cohort_agg.get(f'{name}_sum_mean_raw_cohort_mean', 0):.2f} | "
            f"{cohort_agg.get(f'{name}_sum_mean_cal_cohort_mean', 0):.2f} | "
            f"{corr:.4f} |"
        )

    motion_corr = cohort_agg.get("motion_corr_cohort_mean", float("nan"))
    dist_corr = cohort_agg.get("skeleton_dist_corr_cohort_mean", float("nan"))
    lines.extend([
        "",
        "## Success criteria (rotation-only experiment)",
        "",
        f"- motion corr: {motion_corr:.4f} (target > 0.8; old frontal lift was -0.91)",
        f"- skeleton_dist corr: {dist_corr:.4f} (target > 0.8; old lift was -0.89)",
        f"- pose_size raw median: {cohort_agg.get('pose_size_raw_median_cohort_median', 0):.0f}",
        f"- pose_size cal median: {cohort_agg.get('pose_size_cal_median_cohort_median', 0):.0f}",
        "",
        "## Notes",
        "",
        calibration_recommendation,
        "",
    ])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def recommend_calibration(cohort_agg: Dict[str, Any]) -> str:
    motion_corr = cohort_agg.get("motion_corr_cohort_mean", 0)
    dist_corr = cohort_agg.get("skeleton_dist_corr_cohort_mean", 0)
    ang_corr = cohort_agg.get("skeleton_ang_corr_cohort_mean", 0)
    ps_ratio = (
        cohort_agg.get("pose_size_raw_median_cohort_median", 1)
        / max(cohort_agg.get("pose_size_cal_median_cohort_median", 1), 1e-8)
    )
    passed = motion_corr > 0.8 and dist_corr > 0.8
    lines = [
        f"motion_corr={motion_corr:.3f}, skeleton_dist_corr={dist_corr:.3f}, "
        f"angle_corr={ang_corr:.3f}, pose_size ratio raw/cal ~ {ps_ratio:.2f}x.",
        f"Rotation calibration KM alignment: {'PASS' if passed else 'NEEDS REVIEW'}.",
    ]
    if not passed:
        lines.extend([
            "",
            "If correlations remain low, try toggling camera_y_down in rotate_pixels_hip_centered",
            "or applying yaw-only (in-plane) component of R.",
        ])
    return "\n".join(lines)


def run_diagnosis(
    table_dir: str,
    lift_dir: str,
    output_dir: str,
    *,
    para_name: str = "sz",
    calibrated_suffix: str = "_step_1_rot.csv",
    repro_indices: Optional[List[int]] = None,
    cohort_indices: Optional[List[int]] = None,
    skip_repro: bool = False,
    skip_plots: bool = False,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    embedder = FullBodyPoseEmbedder()
    cal_label = "SZ rot" if calibrated_suffix == "_step_1_rot.csv" else "SZ calibrated"

    all_indices = cohort_indices or discover_paired_indices(
        table_dir, lift_dir, para_name, calibrated_suffix=calibrated_suffix
    )
    repro_set = set(repro_indices or [884, 900, 980])

    lifter = None
    if not skip_repro and repro_set:
        print("Loading VideoPose3D for Q1 reproducibility check...")
        lifter = VideoPose3DLifter()

    clip_results: List[Dict[str, Any]] = []
    for i, idx in enumerate(all_indices):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  analyzing {para_name}_{idx} ({i+1}/{len(all_indices)})")
        r = analyze_clip(
            idx, table_dir, lift_dir, para_name, embedder,
            lifter=lifter,
            calibrated_suffix=calibrated_suffix,
            check_repro=(lifter is not None and idx in repro_set),
        )
        clip_results.append(r)

    if not skip_plots:
        plot_domain_sum_profiles(
            clip_results, os.path.join(output_dir, "domain_sum_profiles.png"), cal_label=cal_label
        )
        plot_pose_size_histogram(
            clip_results, os.path.join(output_dir, "pose_size_histogram.png"), cal_label=cal_label.replace("SZ ", "")
        )
        write_q1_csv(clip_results, os.path.join(output_dir, "q1_repro_diff.csv"))

    json_results = []
    for r in clip_results:
        jr = {k: v for k, v in r.items() if k not in ("km_raw", "km_cal", "raw_pose_size_series", "cal_pose_size_series")}
        json_results.append(jr)

    cohort_agg = aggregate_cohort(json_results)
    q1_samples = [r for r in json_results if "q1_max_abs_diff" in r]

    calibration_text = recommend_calibration(cohort_agg)
    breakdown = {
        "q1_samples": q1_samples,
        "cohort_aggregate": cohort_agg,
        "per_clip": json_results,
        "calibration_recommendation": calibration_text,
        "calibrated_suffix": calibrated_suffix,
        "km_gate": evaluate_km_gate(cohort_agg),
    }
    with open(os.path.join(output_dir, "domain_breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(breakdown, f, indent=2, default=str)

    write_report(
        os.path.join(output_dir, "investigation_report.md"),
        cohort_agg,
        q1_samples,
        calibration_text,
        cal_label=cal_label,
    )

    if not skip_plots:
        print(f"\nSaved diagnosis to {output_dir}")
    return breakdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KM magnitude gap diagnosis")
    parser.add_argument("--table-dir", default=r"C:\Users\Olive\Desktop\table")
    parser.add_argument("--lift-dir", default=r"C:\Users\Olive\Desktop\infer_data_11_rot")
    parser.add_argument("--output-dir", default=r"C:\Users\Olive\Desktop\infer_data_11_rot\diagnosis")
    parser.add_argument("--para-name", default="sz")
    parser.add_argument("--csv-suffix", default="_step_1_rot.csv", help="Calibrated CSV suffix")
    parser.add_argument("--repro-indices", nargs="*", type=int, default=[884, 900, 980])
    parser.add_argument("--skip-repro", action="store_true", help="Skip Q1 VideoPose3D repro (faster)")
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="Print KM gate pass/fail and exit (no plots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"KM magnitude diagnosis: {args.para_name}")
    print(f"  table: {args.table_dir}")
    print(f"  cal:   {args.lift_dir} ({args.csv_suffix})")
    breakdown = run_diagnosis(
        args.table_dir,
        args.lift_dir,
        args.output_dir,
        para_name=args.para_name,
        calibrated_suffix=args.csv_suffix,
        repro_indices=args.repro_indices,
        skip_repro=args.skip_repro or args.gate_only,
        skip_plots=args.gate_only,
    )
    agg = breakdown["cohort_aggregate"]
    gate = breakdown["km_gate"]
    if args.gate_only:
        print(json.dumps(gate, indent=2))
        raise SystemExit(0 if gate.get("passed") else 1)
    print("\n--- Cohort summary ---")
    print(f"  clips: {agg.get('n_clips')}")
    print(f"  km_total_sum_mean raw: {agg.get('km_total_sum_mean_raw_cohort_mean', 0):.2f}")
    print(f"  km_total_sum_mean cal: {agg.get('km_total_sum_mean_cal_cohort_mean', 0):.2f}")
    for name in DOMAIN_SLICES:
        print(f"  {name} corr: {agg.get(f'{name}_corr_cohort_mean', float('nan')):.4f}")


if __name__ == "__main__":
    main()
