# -*- coding: utf-8 -*-
"""
Step 4 — visualize inference results + gait symmetry scoring.

Charts (row 1):
  1. KM feature sum per sample (pos/neg colors)
  2. Model probability distribution
  3. Probability vs max(label_value)

Charts (row 2, gait_symmetry_analysis):
  4. Mean distribution symmetry score by label
  5. Probability vs mean symmetry score
  6. Cohort mean bilateral similarity scores (shoulder / arm / stride / trunk)

Additional figure:
  7. Max label value vs each similarity score (2x2 trends with linear fit)
  8. Negative sample value distributions (max Cobb, L/R Cobb, model probability)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _ensure_numpy2_pickle_compat() -> None:
    """Allow unpickling arrays saved with NumPy 2.x when running on NumPy 1.x."""
    if hasattr(np, "_core"):
        return
    core = np.core
    for name in ("_core", "_core.multiarray", "_core.numeric", "_core.umath"):
        alias = name.replace("_core", "core", 1)
        sys.modules.setdefault(f"numpy.{name}", sys.modules.get(f"numpy.{alias}", core))


_ensure_numpy2_pickle_compat()

DEFAULT_PKL_DIR = Path(r"C:\Users\Olive\Desktop\infer_data_11")
DEFAULT_RESULTS_JSON = DEFAULT_PKL_DIR / "inference" / "inference_results.json"
DEFAULT_SZ_NORM_PKL_DIR = Path(r"C:\Users\Olive\Desktop\infer_data_11_norm")
DEFAULT_SZ_NORM_RESULTS_JSON = DEFAULT_SZ_NORM_PKL_DIR / "inference" / "inference_results.json"
DEFAULT_SZ_LIFT_PKL_DIR = Path(r"C:\Users\Olive\Desktop\infer_data_11_lift_pkl")
DEFAULT_SZ_LIFT_RESULTS_JSON = DEFAULT_SZ_LIFT_PKL_DIR / "inference" / "inference_results.json"
DEFAULT_SZ_LIFT_TABLE_DIR = Path(r"C:\Users\Olive\Desktop\infer_data_11_lift")
DEFAULT_SZ_ROT_PKL_DIR = Path(r"C:\Users\Olive\Desktop\infer_data_11_rot_pkl")
DEFAULT_SZ_ROT_RESULTS_JSON = DEFAULT_SZ_ROT_PKL_DIR / "inference" / "inference_results.json"
DEFAULT_SZ_ROT_TABLE_DIR = Path(r"C:\Users\Olive\Desktop\infer_data_11_rot")
DEFAULT_TABLE_DIR = Path(r"C:\Users\Olive\Desktop\table")
DEFAULT_RAW_PKL_DIR = Path(r"C:\Users\Olive\Desktop\experiments\infer_data_raw_pkl")
DEFAULT_RAW_RESULTS_JSON = DEFAULT_RAW_PKL_DIR / "inference" / "inference_results.json"
DEFAULT_RAWNORM_PKL_DIR = Path(r"C:\Users\Olive\Desktop\experiments\infer_data_rawnorm_pkl")
DEFAULT_RAWNORM_RESULTS_JSON = DEFAULT_RAWNORM_PKL_DIR / "inference" / "inference_results.json"
DEFAULT_EXPERIMENTS_ROT_PKL_DIR = Path(r"C:\Users\Olive\Desktop\experiments\infer_data_rot_pkl")
DEFAULT_EXPERIMENTS_ROT_RESULTS_JSON = DEFAULT_EXPERIMENTS_ROT_PKL_DIR / "inference" / "inference_results.json"
DEFAULT_EXPERIMENTS_ROTNORM_PKL_DIR = Path(r"C:\Users\Olive\Desktop\experiments\infer_data_rotnorm_pkl")
DEFAULT_EXPERIMENTS_ROTNORM_RESULTS_JSON = DEFAULT_EXPERIMENTS_ROTNORM_PKL_DIR / "inference" / "inference_results.json"
DEFAULT_EXPERIMENTS_RESULTS_DIR = Path(r"C:\Users\Olive\Desktop\experiments\results")
TARGET_AUROC = 0.70
DEFAULT_EXPERIMENTS_TABLE_DIR = Path(r"C:\Users\Olive\Desktop\experiments\table")
DEFAULT_PK_PKL_DIR = Path(r"C:\Users\Olive\Desktop\infer_data_pk")
DEFAULT_PK_RESULTS_JSON = DEFAULT_PK_PKL_DIR / "inference" / "inference_results.json"
DEFAULT_PK_NORM_PKL_DIR = Path(r"C:\Users\Olive\Desktop\infer_data_pk_norm")
DEFAULT_PK_NORM_RESULTS_JSON = DEFAULT_PK_NORM_PKL_DIR / "inference" / "inference_results.json"
DEFAULT_PK_TABLE_DIR = Path(
    r"C:\Users\Olive\Desktop\video_retrival\video_retrival\pk_testpart_table"
)
DEFAULT_SCOLI_ROOT = Path(r"C:\Users\Olive\Desktop\ScoliDetect_deployversion")
SYMMETRY_THRESHOLD = 0.75

DEFAULT_BINARY_COBB_THRESHOLD = 15.0

COLOR_POS = "#d62728"
COLOR_NEG = "#1f77b4"
COLOR_PK = "#9467bd"
COLOR_PK_NORM = "#2ca02c"
COLOR_SZ_NORM = "#ff7f0e"
COLOR_SZ_LIFT = "#9467bd"
COLOR_SZ_ROT = "#2ca02c"
COLOR_SZ_NEG = "#1f77b4"
COLOR_BASELINE = "#4C72B0"
COLOR_POSTTRAIN = "#DD8452"
COLOR_PRED_POS = "#ff7f0e"
COLOR_PRED_NEG = "#2ca02c"

SIMILARITY_KEYS = [
    ("Shoulder", "双侧肩膀摆动相似度"),
    ("Arm", "双侧手臂摆动相似度"),
    ("Stride", "双侧步幅相似度"),
    ("Trunk", "双侧躯干旋转相似度"),
]


def _import_symmetry_module(scoli_root: str):
    root = str(Path(scoli_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    from utils.gait_symmetry_analysis import analyze_gait_symmetry

    return analyze_gait_symmetry


def _load_infer_helpers():
    infer_path = Path(__file__).resolve().parent / "03_Infer.py"
    if not infer_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("infer03", infer_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_inference_results(results_path: str) -> List[Dict[str, Any]]:
    with open(results_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"No records in {results_path}")
    return records


def max_label_value(label_value: Any) -> float:
    if label_value is None:
        return np.nan
    if isinstance(label_value, (list, tuple)):
        if len(label_value) == 0:
            return np.nan
        return float(max(label_value))
    try:
        return float(label_value)
    except (TypeError, ValueError):
        return np.nan


def cobb_components(label_value: Any) -> Tuple[float, float]:
    """Return (M_cobb_l, M_cobb_r) from label_value list."""
    if not isinstance(label_value, (list, tuple)) or len(label_value) < 2:
        return np.nan, np.nan
    try:
        return float(label_value[0]), float(label_value[1])
    except (TypeError, ValueError):
        return np.nan, np.nan


def negative_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if r.get("label") == 0.0]


def positive_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if r.get("label") == 1.0]


def ground_truth_negative_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """SZ-style negatives with real Cobb labels (excludes fixed_label placeholders)."""
    return [
        r for r in records
        if r.get("label") == 0.0 and r.get("label_value") is not None
    ]


def _pair_source_key(source_file: Any) -> Optional[str]:
    """Normalize PK source names for cross-cohort pairing (e.g. pk_100_step_1.csv vs pk_100_step1)."""
    if not source_file:
        return None
    match = re.search(r"pk_(\d+)", str(source_file), flags=re.IGNORECASE)
    if match:
        return f"pk_{match.group(1)}"
    return str(source_file)


def _records_by_pair_key(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    keyed: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        key = _pair_source_key(rec.get("source_file"))
        if key and key not in keyed:
            keyed[key] = rec
    return keyed


def _mann_whitney_p(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        from scipy.stats import mannwhitneyu

        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return None


def _cohort_probs(records: List[Dict[str, Any]]) -> List[float]:
    return [float(r["probability"]) for r in records]


def _cohort_symmetry_scores(
    records: List[Dict[str, Any]],
    symmetry_rows: List[Dict[str, Any]],
) -> List[float]:
    by_source = _symmetry_by_source(symmetry_rows)
    scores: List[float] = []
    for rec in records:
        sym = by_source.get(rec.get("source_file"))
        if not sym:
            continue
        score = _symmetry_score(sym)
        if not np.isnan(score):
            scores.append(score)
    return scores


def _cohort_similarity_means(
    records: List[Dict[str, Any]],
    symmetry_rows: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
    by_source = _symmetry_by_source(symmetry_rows)
    out: Dict[str, List[float]] = {eng: [] for eng, _ in SIMILARITY_KEYS}
    for rec in records:
        sym = by_source.get(rec.get("source_file"))
        if not sym:
            continue
        for eng, _ in SIMILARITY_KEYS:
            val = _similarity_value(sym, eng)
            if not np.isnan(val):
                out[eng].append(val)
    return out


def _mean_km_profile(
    records: List[Dict[str, Any]],
    pkl_dir: Optional[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    profiles, _, _ = load_km_sum_profiles(records, pkl_dir)
    if not profiles:
        return None, None
    stacked = np.stack(profiles, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def _cohort_summary_stats(
    name: str,
    records: List[Dict[str, Any]],
    symmetry_rows: List[Dict[str, Any]],
    *,
    threshold: float,
) -> Dict[str, Any]:
    probs = _cohort_probs(records)
    sym_scores = _cohort_symmetry_scores(records, symmetry_rows)
    sim = _cohort_similarity_means(records, symmetry_rows)
    n_pred_pos = sum(1 for p in probs if p >= threshold)
    summary: Dict[str, Any] = {
        "cohort": name,
        "n_inference": len(records),
        "n_symmetry": len(sym_scores),
        "threshold": threshold,
        "probability": {
            "mean": float(np.mean(probs)) if probs else None,
            "median": float(np.median(probs)) if probs else None,
            "std": float(np.std(probs)) if probs else None,
            "min": float(np.min(probs)) if probs else None,
            "max": float(np.max(probs)) if probs else None,
            "predicted_positive": n_pred_pos,
            "predicted_positive_rate": n_pred_pos / len(probs) if probs else None,
        },
        "symmetry_score": {
            "mean": float(np.mean(sym_scores)) if sym_scores else None,
            "median": float(np.median(sym_scores)) if sym_scores else None,
            "std": float(np.std(sym_scores)) if sym_scores else None,
        },
        "similarity_means": {
            eng: float(np.mean(vals)) if vals else None
            for eng, vals in sim.items()
        },
    }
    return summary


def build_pk_vs_sz_neg_comparison_figure(
    pk_records: List[Dict[str, Any]],
    sz_neg_records: List[Dict[str, Any]],
    pk_symmetry: List[Dict[str, Any]],
    sz_neg_symmetry: List[Dict[str, Any]],
    *,
    pk_pkl_dir: Optional[str],
    sz_pkl_dir: Optional[str],
    threshold: float,
    comparison_stats: Dict[str, Any],
    cohort_a_label: str = "PK",
    cohort_b_label: str = "SZ neg",
    cohort_a_color: str = COLOR_PK,
    cohort_b_color: str = COLOR_SZ_NEG,
    cohort_a_key: str = "pk",
    cohort_b_key: str = "sz_negative",
    figure_title: str = "PK cohort vs SZ negative samples",
) -> plt.Figure:
    """Side-by-side comparison of two inference cohorts."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    pk_probs = _cohort_probs(pk_records)
    sz_probs = _cohort_probs(sz_neg_records)
    bins = np.linspace(0.0, 1.0, 21)

    ax = axes[0, 0]
    if pk_probs:
        ax.hist(pk_probs, bins=bins, alpha=0.55, color=cohort_a_color, label=f"{cohort_a_label} (n={len(pk_probs)})", edgecolor="white")
    if sz_probs:
        ax.hist(sz_probs, bins=bins, alpha=0.55, color=cohort_b_color, label=f"{cohort_b_label} (n={len(sz_probs)})", edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Thr={threshold:.2f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Model probability distribution")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    pk_sym = _cohort_symmetry_scores(pk_records, pk_symmetry)
    sz_sym = _cohort_symmetry_scores(sz_neg_records, sz_neg_symmetry)
    sym_bins = np.linspace(0.0, 1.0, 21)
    if pk_sym:
        ax.hist(pk_sym, bins=sym_bins, alpha=0.55, color=cohort_a_color, label=f"{cohort_a_label} (n={len(pk_sym)})", edgecolor="white")
    if sz_sym:
        ax.hist(sz_sym, bins=sym_bins, alpha=0.55, color=cohort_b_color, label=f"{cohort_b_label} (n={len(sz_sym)})", edgecolor="white")
    ax.axvline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0, label=f"Sym thr={SYMMETRY_THRESHOLD}")
    ax.set_xlabel("Mean distribution symmetry score")
    ax.set_ylabel("Count")
    ax.set_title("Gait symmetry score distribution")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    labels_eng = [eng for eng, _ in SIMILARITY_KEYS]
    x = np.arange(len(labels_eng))
    width = 0.35
    pk_sim = _cohort_similarity_means(pk_records, pk_symmetry)
    sz_sim = _cohort_similarity_means(sz_neg_records, sz_neg_symmetry)
    pk_means = [float(np.mean(pk_sim[eng])) if pk_sim[eng] else 0.0 for eng in labels_eng]
    sz_means = [float(np.mean(sz_sim[eng])) if sz_sim[eng] else 0.0 for eng in labels_eng]
    ax.bar(x - width / 2, pk_means, width, label=cohort_a_label, color=cohort_a_color, alpha=0.8)
    ax.bar(x + width / 2, sz_means, width, label=cohort_b_label, color=cohort_b_color, alpha=0.8)
    ax.axhline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_eng)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean similarity score")
    ax.set_title("Bilateral similarity (cohort mean)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    pk_mean, pk_std = _mean_km_profile(pk_records, pk_pkl_dir)
    sz_mean, sz_std = _mean_km_profile(sz_neg_records, sz_pkl_dir)
    if pk_mean is not None:
        t = np.arange(len(pk_mean))
        ax.plot(t, pk_mean, color=cohort_a_color, linewidth=2, label=f"{cohort_a_label} mean")
        if pk_std is not None:
            ax.fill_between(t, pk_mean - pk_std, pk_mean + pk_std, color=cohort_a_color, alpha=0.2)
    if sz_mean is not None:
        t = np.arange(len(sz_mean))
        ax.plot(t, sz_mean, color=cohort_b_color, linewidth=2, label=f"{cohort_b_label} mean")
        if sz_std is not None:
            ax.fill_between(t, sz_mean - sz_std, sz_mean + sz_std, color=cohort_b_color, alpha=0.2)
    ax.set_xlabel("KM timestep")
    ax.set_ylabel("Sum of KM features")
    ax.set_title("KM feature sum (mean ± std)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for recs, sym_rows, color, marker, label in (
        (pk_records, pk_symmetry, cohort_a_color, "o", cohort_a_label),
        (sz_neg_records, sz_neg_symmetry, cohort_b_color, "s", cohort_b_label),
    ):
        by_source = _symmetry_by_source(sym_rows)
        xs, ys = [], []
        for rec in recs:
            sym = by_source.get(rec.get("source_file"))
            if not sym:
                continue
            score = _symmetry_score(sym)
            if np.isnan(score):
                continue
            xs.append(score)
            ys.append(float(rec["probability"]))
        if xs:
            ax.scatter(xs, ys, c=color, marker=marker, alpha=0.65, s=32, edgecolors="white", linewidths=0.4, label=label)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Mean distribution symmetry score")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Probability vs gait symmetry")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        figure_title,
        "",
        f"{cohort_a_label}: n={len(pk_records)}  |  {cohort_b_label}: n={len(sz_neg_records)}",
        f"Threshold: {threshold:.2f}",
        "",
        "Probability (mean ± std):",
    ]
    for key, label in ((cohort_a_key, cohort_a_label), (cohort_b_key, cohort_b_label)):
        s = comparison_stats.get(key, {})
        p = s.get("probability", {})
        if p.get("mean") is not None:
            lines.append(
                f"  {label}: {p['mean']:.3f} ± {p.get('std', 0):.3f}  "
                f"[{p.get('min', 0):.3f}, {p.get('max', 0):.3f}]"
            )
            rate = p.get("predicted_positive_rate")
            if rate is not None:
                lines.append(f"         pred+ rate: {100 * rate:.1f}%")
    lines.extend(["", "Symmetry score (mean):",])
    for key, label in ((cohort_a_key, cohort_a_label), (cohort_b_key, cohort_b_label)):
        sym = comparison_stats.get(key, {}).get("symmetry_score", {})
        if sym.get("mean") is not None:
            lines.append(f"  {label}: {sym['mean']:.3f}")
    tests = comparison_stats.get("statistical_tests", {})
    if tests:
        lines.extend(["", "Mann-Whitney p-values:"])
        for metric, pval in tests.items():
            if pval is not None:
                lines.append(f"  {metric}: p={pval:.4g}")
    ax.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f7f7f7", alpha=0.95),
    )

    fig.suptitle(figure_title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def run_pk_vs_sz_neg_comparison(
    *,
    pk_results_json: str,
    pk_pkl_dir: str,
    pk_table_dir: str,
    sz_results_json: str,
    sz_pkl_dir: str,
    sz_table_dir: str,
    scoli_root: str,
    output_dir: str,
    symmetry_plot_dir: Optional[str] = None,
    skip_symmetry: bool = False,
) -> Dict[str, Any]:
    """Compare infer_data_pk against ground-truth negatives from infer_data_11."""
    pk_records = load_inference_results(pk_results_json)
    sz_all = load_inference_results(sz_results_json)
    sz_neg_records = ground_truth_negative_records(sz_all)

    threshold = float(pk_records[0].get("threshold", sz_all[0].get("threshold", 0.5)))

    pk_symmetry: List[Dict[str, Any]] = []
    sz_neg_symmetry: List[Dict[str, Any]] = []
    if not skip_symmetry:
        sym_base = symmetry_plot_dir or os.path.join(output_dir, "comparison_symmetry_plots")
        pk_sym_dir = os.path.join(sym_base, "pk")
        sz_sym_dir = os.path.join(sym_base, "sz_negative")
        os.makedirs(pk_sym_dir, exist_ok=True)
        os.makedirs(sz_sym_dir, exist_ok=True)
        print(f"Symmetry: PK CSVs from {pk_table_dir}")
        pk_symmetry = run_symmetry_eval(pk_records, pk_table_dir, scoli_root, symmetry_plot_dir=pk_sym_dir)
        print(f"Symmetry: SZ neg CSVs from {sz_table_dir}")
        sz_neg_symmetry = run_symmetry_eval(sz_neg_records, sz_table_dir, scoli_root, symmetry_plot_dir=sz_sym_dir)

    pk_summary = _cohort_summary_stats("pk", pk_records, pk_symmetry, threshold=threshold)
    sz_summary = _cohort_summary_stats("sz_negative", sz_neg_records, sz_neg_symmetry, threshold=threshold)

    pk_probs = _cohort_probs(pk_records)
    sz_probs = _cohort_probs(sz_neg_records)
    pk_sym_scores = _cohort_symmetry_scores(pk_records, pk_symmetry)
    sz_sym_scores = _cohort_symmetry_scores(sz_neg_records, sz_neg_symmetry)

    comparison_stats: Dict[str, Any] = {
        "pk": pk_summary,
        "sz_negative": sz_summary,
        "statistical_tests": {
            "probability": _mann_whitney_p(pk_probs, sz_probs),
            "symmetry_score": _mann_whitney_p(pk_sym_scores, sz_sym_scores),
        },
    }
    for eng, _ in SIMILARITY_KEYS:
        pk_vals = _cohort_similarity_means(pk_records, pk_symmetry).get(eng, [])
        sz_vals = _cohort_similarity_means(sz_neg_records, sz_neg_symmetry).get(eng, [])
        comparison_stats["statistical_tests"][f"similarity_{eng.lower()}"] = _mann_whitney_p(pk_vals, sz_vals)

    os.makedirs(output_dir, exist_ok=True)
    fig = build_pk_vs_sz_neg_comparison_figure(
        pk_records,
        sz_neg_records,
        pk_symmetry,
        sz_neg_symmetry,
        pk_pkl_dir=pk_pkl_dir,
        sz_pkl_dir=sz_pkl_dir,
        threshold=threshold,
        comparison_stats=comparison_stats,
    )
    fig_path = os.path.join(output_dir, "pk_vs_sz_negative_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    json_path = os.path.join(output_dir, "pk_vs_sz_negative_comparison.json")
    save_json(
        {
            "comparison": comparison_stats,
            "pk_symmetry_per_sample": pk_symmetry,
            "sz_negative_symmetry_per_sample": sz_neg_symmetry,
        },
        json_path,
    )
    print(f"Saved: {json_path}")

    _print_pk_vs_sz_summary(comparison_stats, len(pk_records), len(sz_neg_records))
    return comparison_stats


def run_sz_pos_vs_neg_comparison(
    *,
    sz_results_json: str,
    sz_pkl_dir: str,
    table_dir: str,
    scoli_root: str,
    output_dir: str,
    symmetry_plot_dir: Optional[str] = None,
    skip_symmetry: bool = False,
) -> Dict[str, Any]:
    """Compare SZ positive vs negative samples from one inference results JSON."""
    all_records = load_inference_results(sz_results_json)
    sz_pos_records = positive_records(all_records)
    sz_neg_records = ground_truth_negative_records(all_records)
    if not sz_pos_records or not sz_neg_records:
        raise ValueError(
            f"Need both SZ positive and negative labeled records; "
            f"got pos={len(sz_pos_records)}, neg={len(sz_neg_records)}"
        )

    threshold = float(all_records[0].get("threshold", 0.5))

    pos_symmetry: List[Dict[str, Any]] = []
    neg_symmetry: List[Dict[str, Any]] = []
    if not skip_symmetry:
        sym_base = symmetry_plot_dir or os.path.join(output_dir, "comparison_symmetry_plots")
        pos_sym_dir = os.path.join(sym_base, "sz_positive")
        neg_sym_dir = os.path.join(sym_base, "sz_negative")
        os.makedirs(pos_sym_dir, exist_ok=True)
        os.makedirs(neg_sym_dir, exist_ok=True)
        print(f"Symmetry: SZ pos CSVs from {table_dir}")
        pos_symmetry = run_symmetry_eval(sz_pos_records, table_dir, scoli_root, symmetry_plot_dir=pos_sym_dir)
        print(f"Symmetry: SZ neg CSVs from {table_dir}")
        neg_symmetry = run_symmetry_eval(sz_neg_records, table_dir, scoli_root, symmetry_plot_dir=neg_sym_dir)

    pos_summary = _cohort_summary_stats("sz_positive", sz_pos_records, pos_symmetry, threshold=threshold)
    neg_summary = _cohort_summary_stats("sz_negative", sz_neg_records, neg_symmetry, threshold=threshold)

    pos_probs = _cohort_probs(sz_pos_records)
    neg_probs = _cohort_probs(sz_neg_records)
    pos_sym_scores = _cohort_symmetry_scores(sz_pos_records, pos_symmetry)
    neg_sym_scores = _cohort_symmetry_scores(sz_neg_records, neg_symmetry)

    comparison_stats: Dict[str, Any] = {
        "sz_positive": pos_summary,
        "sz_negative": neg_summary,
        "statistical_tests": {
            "probability": _mann_whitney_p(pos_probs, neg_probs),
            "symmetry_score": _mann_whitney_p(pos_sym_scores, neg_sym_scores),
        },
    }
    for eng, _ in SIMILARITY_KEYS:
        pos_vals = _cohort_similarity_means(sz_pos_records, pos_symmetry).get(eng, [])
        neg_vals = _cohort_similarity_means(sz_neg_records, neg_symmetry).get(eng, [])
        comparison_stats["statistical_tests"][f"similarity_{eng.lower()}"] = _mann_whitney_p(pos_vals, neg_vals)

    os.makedirs(output_dir, exist_ok=True)
    fig = build_pk_vs_sz_neg_comparison_figure(
        sz_pos_records,
        sz_neg_records,
        pos_symmetry,
        neg_symmetry,
        pk_pkl_dir=sz_pkl_dir,
        sz_pkl_dir=sz_pkl_dir,
        threshold=threshold,
        comparison_stats=comparison_stats,
        cohort_a_label="SZ pos",
        cohort_b_label="SZ neg",
        cohort_a_color=COLOR_POS,
        cohort_b_color=COLOR_NEG,
        cohort_a_key="sz_positive",
        cohort_b_key="sz_negative",
        figure_title="SZ positive vs negative samples",
    )
    fig_path = os.path.join(output_dir, "sz_pos_vs_neg_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    json_path = os.path.join(output_dir, "sz_pos_vs_neg_comparison.json")
    save_json(
        {
            "comparison": comparison_stats,
            "sz_positive_symmetry_per_sample": pos_symmetry,
            "sz_negative_symmetry_per_sample": neg_symmetry,
        },
        json_path,
    )
    print(f"Saved: {json_path}")

    print(f"\nSZ pos vs neg comparison (pos n={len(sz_pos_records)}, neg n={len(sz_neg_records)})")
    for key, label in (("sz_positive", "SZ pos"), ("sz_negative", "SZ neg")):
        s = comparison_stats.get(key, {})
        p = s.get("probability", {})
        sym = s.get("symmetry_score", {})
        print(f"  {label}:")
        if p.get("mean") is not None:
            print(
                f"    probability: mean={p['mean']:.4f}, median={p.get('median', 0):.4f}, "
                f"pred+={p.get('predicted_positive', 0)} ({100 * (p.get('predicted_positive_rate') or 0):.1f}%)"
            )
        if sym.get("mean") is not None:
            print(f"    symmetry: mean={sym['mean']:.4f}")
    pval = comparison_stats.get("statistical_tests", {}).get("probability")
    if pval is not None:
        print(f"  Mann-Whitney (probability): p={pval:.4g}")
    return comparison_stats


def _pair_records_by_source(
    baseline_records: List[Dict[str, Any]],
    posttrain_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    post_by_source = {r.get("source_file"): r for r in posttrain_records}
    paired: List[Dict[str, Any]] = []
    for base_rec in baseline_records:
        source = base_rec.get("source_file")
        post_rec = post_by_source.get(source)
        if post_rec is None:
            continue
        paired.append(
            {
                "source_file": source,
                "label": base_rec.get("label"),
                "label_value": base_rec.get("label_value"),
                "baseline_prob": float(base_rec["probability"]),
                "posttrain_prob": float(post_rec["probability"]),
                "baseline_pred": int(base_rec.get("prediction", 0)),
                "posttrain_pred": int(post_rec.get("prediction", 0)),
            }
        )
    return paired


def _load_inference_metrics(inference_dir: str) -> Dict[str, Any]:
    path = os.path.join(inference_dir, "metrics.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_baseline_vs_posttrain_delta_figure(
    paired: List[Dict[str, Any]],
    *,
    threshold: float,
    comparison_stats: Dict[str, Any],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    base_probs = [p["baseline_prob"] for p in paired]
    post_probs = [p["posttrain_prob"] for p in paired]
    deltas = [p["posttrain_prob"] - p["baseline_prob"] for p in paired]
    bins = np.linspace(0.0, 1.0, 21)

    ax = axes[0]
    if base_probs:
        ax.hist(base_probs, bins=bins, alpha=0.55, color=COLOR_BASELINE, label=f"Baseline (n={len(base_probs)})")
    if post_probs:
        ax.hist(post_probs, bins=bins, alpha=0.55, color=COLOR_POSTTRAIN, label=f"Post-trained (n={len(post_probs)})")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Holdout probability distributions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if base_probs and post_probs:
        ax.scatter(base_probs, post_probs, alpha=0.6, s=28, c=COLOR_POSTTRAIN, edgecolors="white", linewidths=0.4)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0)
    ax.axhline(threshold, color="black", linestyle=":", linewidth=0.8)
    ax.axvline(threshold, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Baseline probability")
    ax.set_ylabel("Post-trained probability")
    ax.set_title("Per-patch paired probabilities")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if deltas:
        delta_bins = np.linspace(-1.0, 1.0, 41)
        ax.hist(deltas, bins=delta_bins, color=COLOR_POSTTRAIN, alpha=0.75, edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Post-trained - baseline probability")
    ax.set_ylabel("Count")
    ax.set_title("Probability delta")
    ax.grid(True, alpha=0.3)

    pval = comparison_stats.get("statistical_tests", {}).get("baseline_vs_posttrain_probability")
    if pval is not None:
        fig.suptitle(f"Baseline vs post-trained (holdout n={len(paired)}, p={pval:.4g})", fontsize=12)
    else:
        fig.suptitle(f"Baseline vs post-trained (holdout n={len(paired)})", fontsize=12)
    fig.tight_layout()
    return fig


def run_baseline_vs_posttrain_comparison(
    *,
    baseline_inference_dir: str,
    posttrain_inference_dir: str,
    pkl_dir: str,
    table_dir: str,
    scoli_root: str,
    output_dir: str,
    symmetry_plot_dir: Optional[str] = None,
    skip_symmetry: bool = False,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare baseline vs post-trained inference on the same holdout patches."""
    baseline_results = os.path.join(baseline_inference_dir, "inference_results.json")
    posttrain_results = os.path.join(posttrain_inference_dir, "inference_results.json")
    if not os.path.isfile(baseline_results):
        raise FileNotFoundError(baseline_results)
    if not os.path.isfile(posttrain_results):
        raise FileNotFoundError(posttrain_results)

    baseline_records = load_inference_results(baseline_results)
    posttrain_records = load_inference_results(posttrain_results)
    if not baseline_records or not posttrain_records:
        raise ValueError("Need non-empty baseline and post-trained inference records")

    thr = float(threshold if threshold is not None else baseline_records[0].get("threshold", 0.5))

    baseline_symmetry: List[Dict[str, Any]] = []
    posttrain_symmetry: List[Dict[str, Any]] = []
    if not skip_symmetry:
        sym_base = symmetry_plot_dir or os.path.join(output_dir, "comparison_symmetry_plots")
        base_sym_dir = os.path.join(sym_base, "baseline")
        post_sym_dir = os.path.join(sym_base, "posttrain")
        os.makedirs(base_sym_dir, exist_ok=True)
        os.makedirs(post_sym_dir, exist_ok=True)
        baseline_symmetry = run_symmetry_eval(
            baseline_records, table_dir, scoli_root, symmetry_plot_dir=base_sym_dir
        )
        posttrain_symmetry = run_symmetry_eval(
            posttrain_records, table_dir, scoli_root, symmetry_plot_dir=post_sym_dir
        )

    baseline_summary = _cohort_summary_stats(
        "baseline", baseline_records, baseline_symmetry, threshold=thr
    )
    posttrain_summary = _cohort_summary_stats(
        "posttrain", posttrain_records, posttrain_symmetry, threshold=thr
    )

    paired = _pair_records_by_source(baseline_records, posttrain_records)
    base_probs = [p["baseline_prob"] for p in paired]
    post_probs = [p["posttrain_prob"] for p in paired]
    deltas = [p["posttrain_prob"] - p["baseline_prob"] for p in paired]

    comparison_stats: Dict[str, Any] = {
        "baseline": baseline_summary,
        "posttrain": posttrain_summary,
        "paired_n": len(paired),
        "baseline_metrics_file": _load_inference_metrics(baseline_inference_dir),
        "posttrain_metrics_file": _load_inference_metrics(posttrain_inference_dir),
        "statistical_tests": {
            "baseline_vs_posttrain_probability": _mann_whitney_p(base_probs, post_probs),
            "probability_delta_mean": float(np.mean(deltas)) if deltas else None,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    fig = build_pk_vs_sz_neg_comparison_figure(
        baseline_records,
        posttrain_records,
        baseline_symmetry,
        posttrain_symmetry,
        pk_pkl_dir=pkl_dir,
        sz_pkl_dir=pkl_dir,
        threshold=thr,
        comparison_stats=comparison_stats,
        cohort_a_label="Baseline",
        cohort_b_label="Post-trained",
        cohort_a_color=COLOR_BASELINE,
        cohort_b_color=COLOR_POSTTRAIN,
        cohort_a_key="baseline",
        cohort_b_key="posttrain",
        figure_title="Baseline vs post-trained (holdout)",
    )
    overview_path = os.path.join(output_dir, "baseline_vs_posttrain_overview.png")
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {overview_path}")

    delta_fig = build_baseline_vs_posttrain_delta_figure(paired, threshold=thr, comparison_stats=comparison_stats)
    delta_path = os.path.join(output_dir, "baseline_vs_posttrain_delta.png")
    delta_fig.savefig(delta_path, dpi=150, bbox_inches="tight")
    plt.close(delta_fig)
    print(f"Saved: {delta_path}")

    json_path = os.path.join(output_dir, "comparison_metrics.json")
    save_json(
        {"comparison": comparison_stats, "paired_samples": paired},
        json_path,
    )
    print(f"Saved: {json_path}")

    print(f"\nBaseline vs post-trained (holdout, paired n={len(paired)})")
    for key, label in (("baseline", "Baseline"), ("posttrain", "Post-trained")):
        s = comparison_stats.get(key, {})
        p = s.get("probability", {})
        m = comparison_stats.get(f"{key}_metrics_file", {})
        print(f"  {label}: n={s.get('n_inference', 0)}")
        if p.get("mean") is not None:
            print(f"    mean prob={p['mean']:.4f}")
        if m:
            print(
                f"    AUROC={m.get('auc_roc', m.get('auroc', 'N/A'))}, "
                f"acc={m.get('accuracy', 'N/A')}, "
                f"F1={m.get('f1', 'N/A')}"
            )
    pval = comparison_stats["statistical_tests"].get("baseline_vs_posttrain_probability")
    if pval is not None:
        print(f"  Mann-Whitney (probability): p={pval:.4g}")
    mean_delta = comparison_stats["statistical_tests"].get("probability_delta_mean")
    if mean_delta is not None:
        print(f"  Mean probability delta (post - base): {mean_delta:+.4f}")
    return comparison_stats


def _print_pk_vs_sz_summary(stats: Dict[str, Any], n_pk: int, n_sz_neg: int) -> None:
    print(f"\nPK vs SZ negative comparison (PK n={n_pk}, SZ neg n={n_sz_neg})")
    for key, label in (("pk", "PK"), ("sz_negative", "SZ neg")):
        s = stats.get(key, {})
        p = s.get("probability", {})
        sym = s.get("symmetry_score", {})
        print(f"  {label}:")
        if p.get("mean") is not None:
            print(
                f"    probability: mean={p['mean']:.4f}, median={p.get('median', 0):.4f}, "
                f"pred+={p.get('predicted_positive', 0)} ({100 * (p.get('predicted_positive_rate') or 0):.1f}%)"
            )
        if sym.get("mean") is not None:
            print(f"    symmetry:    mean={sym['mean']:.4f}")
    tests = stats.get("statistical_tests", {})
    if tests.get("probability") is not None:
        print(f"  Mann-Whitney (probability): p={tests['probability']:.4g}")
    if tests.get("symmetry_score") is not None:
        print(f"  Mann-Whitney (symmetry):    p={tests['symmetry_score']:.4g}")


def build_pk_vs_pk_norm_comparison_figure(
    pk_records: List[Dict[str, Any]],
    pk_norm_records: List[Dict[str, Any]],
    symmetry_rows: List[Dict[str, Any]],
    *,
    pk_pkl_dir: Optional[str],
    pk_norm_pkl_dir: Optional[str],
    threshold: float,
    comparison_stats: Dict[str, Any],
) -> plt.Figure:
    """Side-by-side comparison: infer_data_pk vs infer_data_pk_norm (KM z-score)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    pk_probs = _cohort_probs(pk_records)
    norm_probs = _cohort_probs(pk_norm_records)
    bins = np.linspace(0.0, 1.0, 21)

    ax = axes[0, 0]
    if pk_probs:
        ax.hist(pk_probs, bins=bins, alpha=0.55, color=COLOR_PK, label=f"PK (n={len(pk_probs)})", edgecolor="white")
    if norm_probs:
        ax.hist(norm_probs, bins=bins, alpha=0.55, color=COLOR_PK_NORM, label=f"PK norm (n={len(norm_probs)})", edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Thr={threshold:.2f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Model probability distribution")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    pk_sym = _cohort_symmetry_scores(pk_records, symmetry_rows)
    sym_bins = np.linspace(0.0, 1.0, 21)
    if pk_sym:
        ax.hist(pk_sym, bins=sym_bins, alpha=0.55, color=COLOR_PK, label=f"PK (n={len(pk_sym)})", edgecolor="white")
    ax.axvline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0, label=f"Sym thr={SYMMETRY_THRESHOLD}")
    ax.set_xlabel("Mean distribution symmetry score")
    ax.set_ylabel("Count")
    ax.set_title("Gait symmetry (same pose CSVs for both cohorts)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.98, 0.95, "Symmetry identical\n(same table CSVs)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax = axes[0, 2]
    labels_eng = [eng for eng, _ in SIMILARITY_KEYS]
    x = np.arange(len(labels_eng))
    width = 0.35
    pk_sim = _cohort_similarity_means(pk_records, symmetry_rows)
    pk_means = [float(np.mean(pk_sim[eng])) if pk_sim[eng] else 0.0 for eng in labels_eng]
    ax.bar(x, pk_means, width, label="PK / PK norm (shared)", color=COLOR_PK, alpha=0.8)
    ax.axhline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_eng)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean similarity score")
    ax.set_title("Bilateral similarity (shared pose)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    pk_mean, pk_std = _mean_km_profile(pk_records, pk_pkl_dir)
    norm_mean, norm_std = _mean_km_profile(pk_norm_records, pk_norm_pkl_dir)
    if pk_mean is not None:
        t = np.arange(len(pk_mean))
        ax.plot(t, pk_mean, color=COLOR_PK, linewidth=2, label="PK mean")
        if pk_std is not None:
            ax.fill_between(t, pk_mean - pk_std, pk_mean + pk_std, color=COLOR_PK, alpha=0.2)
    if norm_mean is not None:
        t = np.arange(len(norm_mean))
        ax.plot(t, norm_mean, color=COLOR_PK_NORM, linewidth=2, label="PK norm mean")
        if norm_std is not None:
            ax.fill_between(t, norm_mean - norm_std, norm_mean + norm_std, color=COLOR_PK_NORM, alpha=0.2)
    ax.set_xlabel("KM timestep")
    ax.set_ylabel("Sum of KM features")
    ax.set_title("KM feature sum (mean ± std)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    by_source = _symmetry_by_source(symmetry_rows)
    for recs, color, marker, label in (
        (pk_records, COLOR_PK, "o", "PK"),
        (pk_norm_records, COLOR_PK_NORM, "s", "PK norm"),
    ):
        xs, ys = [], []
        for rec in recs:
            sym = by_source.get(rec.get("source_file"))
            if not sym:
                continue
            score = _symmetry_score(sym)
            if np.isnan(score):
                continue
            xs.append(score)
            ys.append(float(rec["probability"]))
        if xs:
            ax.scatter(xs, ys, c=color, marker=marker, alpha=0.65, s=32, edgecolors="white", linewidths=0.4, label=label)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Mean distribution symmetry score")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Probability vs gait symmetry")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    paired_probs_a, paired_probs_b = [], []
    norm_by_source = _records_by_pair_key(pk_norm_records)
    for rec in pk_records:
        other = norm_by_source.get(_pair_source_key(rec.get("source_file")))
        if other:
            paired_probs_a.append(float(rec["probability"]))
            paired_probs_b.append(float(other["probability"]))
    if paired_probs_a:
        ax.scatter(paired_probs_a, paired_probs_b, c=COLOR_PK_NORM, alpha=0.6, s=28, edgecolors="white", linewidths=0.4)
        lo = min(min(paired_probs_a), min(paired_probs_b))
        hi = max(max(paired_probs_a), max(paired_probs_b))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1.0, label="y=x")
        if len(paired_probs_a) >= 2:
            r = float(np.corrcoef(paired_probs_a, paired_probs_b)[0, 1])
            ax.text(0.05, 0.95, f"paired n={len(paired_probs_a)}\nr={r:.3f}", transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("PK probability")
    ax.set_ylabel("PK norm probability")
    ax.set_title("Paired probability: PK vs PK norm")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    stats_lines = []
    for key, label in (("pk", "PK"), ("pk_norm", "PK norm")):
        p = comparison_stats.get(key, {}).get("probability", {})
        if p.get("mean") is not None:
            stats_lines.append(f"{label}: μ={p['mean']:.3f}, pred+={100 * (p.get('predicted_positive_rate') or 0):.0f}%")
    pval = comparison_stats.get("statistical_tests", {}).get("probability")
    if pval is not None:
        stats_lines.append(f"MW p={pval:.2e}")
    if stats_lines:
        ax.text(
            0.98, 0.05, "\n".join(stats_lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round", facecolor="#f7f7f7", alpha=0.9),
        )

    fig.suptitle("infer_data_pk vs infer_data_pk_norm", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def run_pk_vs_pk_norm_comparison(
    *,
    pk_results_json: str,
    pk_pkl_dir: str,
    pk_norm_results_json: str,
    pk_norm_pkl_dir: str,
    pk_table_dir: str,
    scoli_root: str,
    output_dir: str,
    symmetry_plot_dir: Optional[str] = None,
    skip_symmetry: bool = False,
) -> Dict[str, Any]:
    """Compare infer_data_pk vs infer_data_pk_norm (same clips, KM z-score on/off)."""
    pk_records = load_inference_results(pk_results_json)
    pk_norm_records = load_inference_results(pk_norm_results_json)
    threshold = float(pk_records[0].get("threshold", pk_norm_records[0].get("threshold", 0.5)))

    symmetry_rows: List[Dict[str, Any]] = []
    if not skip_symmetry:
        sym_dir = symmetry_plot_dir or os.path.join(output_dir, "comparison_symmetry_plots", "pk")
        os.makedirs(sym_dir, exist_ok=True)
        print(f"Symmetry: PK CSVs from {pk_table_dir} (shared by both cohorts)")
        symmetry_rows = run_symmetry_eval(pk_records, pk_table_dir, scoli_root, symmetry_plot_dir=sym_dir)

    pk_summary = _cohort_summary_stats("pk", pk_records, symmetry_rows, threshold=threshold)
    norm_summary = _cohort_summary_stats("pk_norm", pk_norm_records, symmetry_rows, threshold=threshold)

    pk_probs = _cohort_probs(pk_records)
    norm_probs = _cohort_probs(pk_norm_records)

    comparison_stats: Dict[str, Any] = {
        "pk": pk_summary,
        "pk_norm": norm_summary,
        "statistical_tests": {
            "probability": _mann_whitney_p(pk_probs, norm_probs),
        },
        "note": "Gait symmetry identical (same pose CSVs); KM differs via GetAllFeatures block z-score.",
    }

    os.makedirs(output_dir, exist_ok=True)
    fig = build_pk_vs_pk_norm_comparison_figure(
        pk_records,
        pk_norm_records,
        symmetry_rows,
        pk_pkl_dir=pk_pkl_dir,
        pk_norm_pkl_dir=pk_norm_pkl_dir,
        threshold=threshold,
        comparison_stats=comparison_stats,
    )
    fig_path = os.path.join(output_dir, "pk_vs_pk_norm_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    json_path = os.path.join(output_dir, "pk_vs_pk_norm_comparison.json")
    save_json(
        {"comparison": comparison_stats, "pk_symmetry_per_sample": symmetry_rows},
        json_path,
    )
    print(f"Saved: {json_path}")

    _print_pk_vs_pk_norm_summary(comparison_stats, len(pk_records), len(pk_norm_records))
    return comparison_stats


def _print_pk_vs_pk_norm_summary(stats: Dict[str, Any], n_pk: int, n_norm: int) -> None:
    print(f"\nPK vs PK norm comparison (PK n={n_pk}, PK norm n={n_norm})")
    for key, label in (("pk", "PK"), ("pk_norm", "PK norm")):
        s = stats.get(key, {})
        p = s.get("probability", {})
        print(f"  {label}:")
        if p.get("mean") is not None:
            print(
                f"    probability: mean={p['mean']:.4f}, median={p.get('median', 0):.4f}, "
                f"pred+={p.get('predicted_positive', 0)} ({100 * (p.get('predicted_positive_rate') or 0):.1f}%)"
            )
    tests = stats.get("statistical_tests", {})
    if tests.get("probability") is not None:
        print(f"  Mann-Whitney (probability): p={tests['probability']:.4g}")


def build_sz11_vs_sz11_norm_comparison_figure(
    sz_records: List[Dict[str, Any]],
    sz_norm_records: List[Dict[str, Any]],
    symmetry_rows: List[Dict[str, Any]],
    *,
    sz_pkl_dir: Optional[str],
    sz_norm_pkl_dir: Optional[str],
    threshold: float,
    comparison_stats: Dict[str, Any],
) -> plt.Figure:
    """Compare infer_data_11 vs infer_data_11_norm (labeled SZ cohort)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    sz_probs = _cohort_probs(sz_records)
    norm_probs = _cohort_probs(sz_norm_records)
    bins = np.linspace(0.0, 1.0, 21)

    ax = axes[0, 0]
    if sz_probs:
        ax.hist(sz_probs, bins=bins, alpha=0.55, color=COLOR_NEG, label=f"SZ (n={len(sz_probs)})", edgecolor="white")
    if norm_probs:
        ax.hist(norm_probs, bins=bins, alpha=0.55, color=COLOR_SZ_NORM, label=f"SZ norm (n={len(norm_probs)})", edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Thr={threshold:.2f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Model probability distribution")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for recs, color, label in ((sz_records, COLOR_NEG, "SZ"), (sz_norm_records, COLOR_SZ_NORM, "SZ norm")):
        xs = [max_label_value(r.get("label_value")) for r in recs]
        ys = [float(r["probability"]) for r in recs]
        valid = [(x, y) for x, y in zip(xs, ys) if not np.isnan(x)]
        if valid:
            ax.scatter([v[0] for v in valid], [v[1] for v in valid], c=color, alpha=0.7, s=32, edgecolors="white", linewidths=0.4, label=label)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Max Cobb angle (°)")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Probability vs max Cobb")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    labels_eng = [eng for eng, _ in SIMILARITY_KEYS]
    x = np.arange(len(labels_eng))
    width = 0.35
    sz_sim = _cohort_similarity_means(sz_records, symmetry_rows)
    sz_means = [float(np.mean(sz_sim[eng])) if sz_sim[eng] else 0.0 for eng in labels_eng]
    ax.bar(x, sz_means, width, label="SZ (table CSVs)", color=COLOR_NEG, alpha=0.8)
    ax.axhline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_eng)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean similarity score")
    ax.set_title("Bilateral similarity (SZ)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    sz_mean, sz_std = _mean_km_profile(sz_records, sz_pkl_dir)
    norm_mean, norm_std = _mean_km_profile(sz_norm_records, sz_norm_pkl_dir)
    if sz_mean is not None:
        t = np.arange(len(sz_mean))
        ax.plot(t, sz_mean, color=COLOR_NEG, linewidth=2, label="SZ mean")
        if sz_std is not None:
            ax.fill_between(t, sz_mean - sz_std, sz_mean + sz_std, color=COLOR_NEG, alpha=0.2)
    if norm_mean is not None:
        t = np.arange(len(norm_mean))
        ax.plot(t, norm_mean, color=COLOR_SZ_NORM, linewidth=2, label="SZ norm mean")
        if norm_std is not None:
            ax.fill_between(t, norm_mean - norm_std, norm_mean + norm_std, color=COLOR_SZ_NORM, alpha=0.2)
    ax.set_xlabel("KM timestep")
    ax.set_ylabel("Sum of KM features")
    ax.set_title("KM feature sum (mean ± std)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    by_source = _symmetry_by_source(symmetry_rows)
    for recs, color, marker, label in (
        (sz_records, COLOR_NEG, "o", "SZ"),
        (sz_norm_records, COLOR_SZ_NORM, "s", "SZ norm"),
    ):
        xs, ys = [], []
        for rec in recs:
            sym = by_source.get(rec.get("source_file"))
            if not sym:
                continue
            score = _symmetry_score(sym)
            if np.isnan(score):
                continue
            xs.append(score)
            ys.append(float(rec["probability"]))
        if xs:
            ax.scatter(xs, ys, c=color, marker=marker, alpha=0.65, s=32, edgecolors="white", linewidths=0.4, label=label)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Mean distribution symmetry score")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Probability vs gait symmetry")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    norm_by_source = {r.get("source_file"): r for r in sz_norm_records if r.get("source_file")}
    paired_a, paired_b = [], []
    for rec in sz_records:
        other = norm_by_source.get(rec.get("source_file"))
        if other:
            paired_a.append(float(rec["probability"]))
            paired_b.append(float(other["probability"]))
    if paired_a:
        ax.scatter(paired_a, paired_b, c=COLOR_SZ_NORM, alpha=0.65, s=36, edgecolors="white", linewidths=0.4)
        lo = min(min(paired_a), min(paired_b))
        hi = max(max(paired_a), max(paired_b))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1.0, label="y=x")
        if len(paired_a) >= 2:
            r = float(np.corrcoef(paired_a, paired_b)[0, 1])
            ax.text(0.05, 0.95, f"paired n={len(paired_a)}\nr={r:.3f}", transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("SZ probability")
    ax.set_ylabel("SZ norm probability")
    ax.set_title("Paired probability: SZ vs SZ norm")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    lines = []
    for key, label in (("sz", "SZ"), ("sz_norm", "SZ norm")):
        m = comparison_stats.get("classification_metrics", {}).get(key, {})
        if m.get("auc_roc") is not None:
            lines.append(f"{label}: AUROC={m['auc_roc']:.3f}, acc={m.get('accuracy', 0):.3f}")
        p = comparison_stats.get(key, {}).get("probability", {})
        if p.get("mean") is not None:
            lines.append(f"{label}: μ={p['mean']:.3f}, pred+={100*(p.get('predicted_positive_rate') or 0):.0f}%")
    if lines:
        ax.text(0.98, 0.05, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round", facecolor="#f7f7f7", alpha=0.9))

    fig.suptitle("infer_data_11 vs infer_data_11_norm", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def run_sz11_vs_sz11_norm_comparison(
    *,
    sz_results_json: str,
    sz_pkl_dir: str,
    sz_norm_results_json: str,
    sz_norm_pkl_dir: str,
    table_dir: str,
    scoli_root: str,
    output_dir: str,
    symmetry_plot_dir: Optional[str] = None,
    skip_symmetry: bool = False,
) -> Dict[str, Any]:
    """Compare infer_data_11 vs infer_data_11_norm (same subjects, normalized pose/KM)."""
    sz_records = load_inference_results(sz_results_json)
    sz_norm_records = load_inference_results(sz_norm_results_json)
    threshold = float(sz_records[0].get("threshold", sz_norm_records[0].get("threshold", 0.5)))

    infer_mod = _load_infer_helpers()
    compute_metrics = getattr(infer_mod, "compute_metrics", None) if infer_mod else None

    symmetry_rows: List[Dict[str, Any]] = []
    if not skip_symmetry:
        sym_dir = symmetry_plot_dir or os.path.join(output_dir, "comparison_symmetry_plots", "sz")
        os.makedirs(sym_dir, exist_ok=True)
        print(f"Symmetry: CSVs from {table_dir}")
        symmetry_rows = run_symmetry_eval(sz_records, table_dir, scoli_root, symmetry_plot_dir=sym_dir)

    sz_summary = _cohort_summary_stats("sz", sz_records, symmetry_rows, threshold=threshold)
    norm_summary = _cohort_summary_stats("sz_norm", sz_norm_records, symmetry_rows, threshold=threshold)

    sz_probs = _cohort_probs(sz_records)
    norm_probs = _cohort_probs(sz_norm_records)

    comparison_stats: Dict[str, Any] = {
        "sz": sz_summary,
        "sz_norm": norm_summary,
        "statistical_tests": {
            "probability": _mann_whitney_p(sz_probs, norm_probs),
        },
        "classification_metrics": {},
    }
    if compute_metrics is not None:
        comparison_stats["classification_metrics"]["sz"] = compute_metrics(sz_records, threshold)
        comparison_stats["classification_metrics"]["sz_norm"] = compute_metrics(sz_norm_records, threshold)

    os.makedirs(output_dir, exist_ok=True)
    fig = build_sz11_vs_sz11_norm_comparison_figure(
        sz_records,
        sz_norm_records,
        symmetry_rows,
        sz_pkl_dir=sz_pkl_dir,
        sz_norm_pkl_dir=sz_norm_pkl_dir,
        threshold=threshold,
        comparison_stats=comparison_stats,
    )
    fig_path = os.path.join(output_dir, "sz11_vs_sz11_norm_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    json_path = os.path.join(output_dir, "sz11_vs_sz11_norm_comparison.json")
    save_json({"comparison": comparison_stats, "sz_symmetry_per_sample": symmetry_rows}, json_path)
    print(f"Saved: {json_path}")

    _print_sz11_vs_sz11_norm_summary(comparison_stats, len(sz_records), len(sz_norm_records))
    return comparison_stats


def plot_probability_by_true_label(
    records: List[Dict[str, Any]],
    ax: plt.Axes,
    *,
    title: str,
    threshold: float,
) -> None:
    """Histogram of predicted probability split by ground-truth label (0/1)."""
    bins = np.linspace(0.0, 1.0, 21)
    for label_val, color, name in ((0.0, COLOR_NEG, "Label 0 (neg)"), (1.0, COLOR_POS, "Label 1 (pos)")):
        probs = [
            float(r["probability"])
            for r in records
            if r.get("label") == label_val
        ]
        if probs:
            ax.hist(
                probs,
                bins=bins,
                alpha=0.6,
                color=color,
                label=f"{name} (n={len(probs)})",
                edgecolor="white",
            )
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Thr={threshold:.2f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def build_raw_vs_rawnorm_comparison_figure(
    cohort_a_records: List[Dict[str, Any]],
    cohort_b_records: List[Dict[str, Any]],
    *,
    cohort_a_pkl_dir: Optional[str],
    cohort_b_pkl_dir: Optional[str],
    threshold: float,
    comparison_stats: Dict[str, Any],
    cohort_a_key: str = "raw",
    cohort_b_key: str = "rawnorm",
    cohort_a_label: str = "Raw",
    cohort_b_label: str = "RawNorm",
    figure_title: str = "Raw vs RawNorm (KM block z-score)",
) -> plt.Figure:
    """Compare two cohorts with label-split probability panels."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    plot_probability_by_true_label(
        cohort_a_records,
        axes[0, 0],
        title=f"{cohort_a_label} SZ — probability by true label",
        threshold=threshold,
    )
    plot_probability_by_true_label(
        cohort_b_records,
        axes[0, 1],
        title=f"{cohort_b_label} SZ — probability by true label",
        threshold=threshold,
    )

    ax = axes[0, 2]
    for recs, color, label in (
        (cohort_a_records, COLOR_NEG, cohort_a_label),
        (cohort_b_records, COLOR_SZ_NORM, cohort_b_label),
    ):
        xs = [max_label_value(r.get("label_value")) for r in recs]
        ys = [float(r["probability"]) for r in recs]
        valid = [(x, y) for x, y in zip(xs, ys) if not np.isnan(x)]
        if valid:
            ax.scatter(
                [v[0] for v in valid],
                [v[1] for v in valid],
                c=color,
                alpha=0.7,
                s=32,
                edgecolors="white",
                linewidths=0.4,
                label=label,
            )
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Max Cobb angle (°)")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Probability vs max Cobb")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    a_mean, a_std = _mean_km_profile(cohort_a_records, cohort_a_pkl_dir)
    b_mean, b_std = _mean_km_profile(cohort_b_records, cohort_b_pkl_dir)
    if a_mean is not None:
        t = np.arange(len(a_mean))
        ax.plot(t, a_mean, color=COLOR_NEG, linewidth=2, label=f"{cohort_a_label} mean")
        if a_std is not None:
            ax.fill_between(t, a_mean - a_std, a_mean + a_std, color=COLOR_NEG, alpha=0.2)
    if b_mean is not None:
        t = np.arange(len(b_mean))
        ax.plot(t, b_mean, color=COLOR_SZ_NORM, linewidth=2, label=f"{cohort_b_label} mean")
        if b_std is not None:
            ax.fill_between(t, b_mean - b_std, b_mean + b_std, color=COLOR_SZ_NORM, alpha=0.2)
    ax.set_xlabel("KM timestep")
    ax.set_ylabel("Sum of KM features")
    ax.set_title("KM feature sum (mean ± std)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    b_by_source = {r.get("source_file"): r for r in cohort_b_records if r.get("source_file")}
    paired_a, paired_b = [], []
    for rec in cohort_a_records:
        other = b_by_source.get(rec.get("source_file"))
        if other:
            paired_a.append(float(rec["probability"]))
            paired_b.append(float(other["probability"]))
    if paired_a:
        ax.scatter(paired_a, paired_b, c=COLOR_SZ_NORM, alpha=0.65, s=36, edgecolors="white", linewidths=0.4)
        lo = min(min(paired_a), min(paired_b))
        hi = max(max(paired_a), max(paired_b))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1.0, label="y=x")
        if len(paired_a) >= 2:
            r = float(np.corrcoef(paired_a, paired_b)[0, 1])
            ax.text(
                0.05, 0.95, f"paired n={len(paired_a)}\nr={r:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xlabel(f"{cohort_a_label} probability")
    ax.set_ylabel(f"{cohort_b_label} probability")
    ax.set_title(f"Paired probability: {cohort_a_label} vs {cohort_b_label}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")
    lines = ["Classification metrics", ""]
    record_map = {cohort_a_key: cohort_a_records, cohort_b_key: cohort_b_records}
    for key, label in ((cohort_a_key, cohort_a_label), (cohort_b_key, cohort_b_label)):
        m = comparison_stats.get("classification_metrics", {}).get(key, {})
        p = comparison_stats.get(key, {}).get("probability", {})
        lines.append(f"{label}:")
        if m.get("auc_roc") is not None:
            lines.append(f"  AUROC={m['auc_roc']:.4f}, acc={m.get('accuracy', 0):.4f}")
        if p.get("mean") is not None:
            lines.append(f"  prob mean={p['mean']:.4f}")
        recs = record_map[key]
        neg_probs = [float(r["probability"]) for r in recs if r.get("label") == 0.0]
        pos_probs = [float(r["probability"]) for r in recs if r.get("label") == 1.0]
        if neg_probs:
            lines.append(f"  label0 mean={np.mean(neg_probs):.4f} (n={len(neg_probs)})")
        if pos_probs:
            lines.append(f"  label1 mean={np.mean(pos_probs):.4f} (n={len(pos_probs)})")
        lines.append("")
    mw = comparison_stats.get("statistical_tests", {}).get("probability")
    if mw is not None:
        lines.append(f"Mann-Whitney p={mw:.4g}")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle(figure_title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def run_raw_vs_rawnorm_comparison(
    *,
    raw_results_json: str,
    raw_pkl_dir: str,
    rawnorm_results_json: str,
    rawnorm_pkl_dir: str,
    output_dir: str,
    cohort_a_key: str = "raw",
    cohort_b_key: str = "rawnorm",
    cohort_a_label: str = "Raw",
    cohort_b_label: str = "RawNorm",
    output_stem: str = "raw_vs_rawnorm",
    figure_title: str = "Raw vs RawNorm (KM block z-score)",
    comparison_note: str = "Cohort A: KM without block z-score; Cohort B: GetAllFeatures block z-score enabled.",
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare two PKL cohorts (default: infer_data_raw_pkl vs infer_data_rawnorm_pkl)."""
    cohort_a_records = load_inference_results(raw_results_json)
    cohort_b_records = load_inference_results(rawnorm_results_json)
    if threshold is None:
        threshold = float(cohort_a_records[0].get("threshold", cohort_b_records[0].get("threshold", 0.5)))
    else:
        threshold = float(threshold)

    infer_mod = _load_infer_helpers()
    compute_metrics = getattr(infer_mod, "compute_metrics", None) if infer_mod else None

    a_summary = _cohort_summary_stats(cohort_a_key, cohort_a_records, [], threshold=threshold)
    b_summary = _cohort_summary_stats(cohort_b_key, cohort_b_records, [], threshold=threshold)

    a_probs = _cohort_probs(cohort_a_records)
    b_probs = _cohort_probs(cohort_b_records)

    comparison_stats: Dict[str, Any] = {
        cohort_a_key: a_summary,
        cohort_b_key: b_summary,
        "statistical_tests": {
            "probability": _mann_whitney_p(a_probs, b_probs),
        },
        "classification_metrics": {},
        "note": comparison_note,
    }
    if compute_metrics is not None:
        comparison_stats["classification_metrics"][cohort_a_key] = compute_metrics(cohort_a_records, threshold)
        comparison_stats["classification_metrics"][cohort_b_key] = compute_metrics(cohort_b_records, threshold)

    os.makedirs(output_dir, exist_ok=True)
    fig = build_raw_vs_rawnorm_comparison_figure(
        cohort_a_records,
        cohort_b_records,
        cohort_a_pkl_dir=raw_pkl_dir,
        cohort_b_pkl_dir=rawnorm_pkl_dir,
        threshold=threshold,
        comparison_stats=comparison_stats,
        cohort_a_key=cohort_a_key,
        cohort_b_key=cohort_b_key,
        cohort_a_label=cohort_a_label,
        cohort_b_label=cohort_b_label,
        figure_title=figure_title,
    )
    fig_path = os.path.join(output_dir, f"{output_stem}_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    json_path = os.path.join(output_dir, f"{output_stem}_comparison.json")
    save_json({"comparison": comparison_stats}, json_path)
    print(f"Saved: {json_path}")

    _print_cohort_pair_summary(
        comparison_stats,
        len(cohort_a_records),
        len(cohort_b_records),
        cohort_a_key=cohort_a_key,
        cohort_b_key=cohort_b_key,
        cohort_a_label=cohort_a_label,
        cohort_b_label=cohort_b_label,
        title=f"{cohort_a_label} vs {cohort_b_label}",
    )
    return comparison_stats


def run_rot_vs_rotnorm_comparison(
    *,
    rot_results_json: str,
    rot_pkl_dir: str,
    rotnorm_results_json: str,
    rotnorm_pkl_dir: str,
    output_dir: str,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare experiments/infer_data_rot_pkl vs infer_data_rotnorm_pkl."""
    return run_raw_vs_rawnorm_comparison(
        raw_results_json=rot_results_json,
        raw_pkl_dir=rot_pkl_dir,
        rawnorm_results_json=rotnorm_results_json,
        rawnorm_pkl_dir=rotnorm_pkl_dir,
        output_dir=output_dir,
        cohort_a_key="rot",
        cohort_b_key="rotnorm",
        cohort_a_label="Rot",
        cohort_b_label="RotNorm",
        output_stem="rot_vs_rotnorm",
        figure_title="Rot vs RotNorm (KM block z-score)",
        comparison_note="Rot PKLs: rotation-calibrated pose, KM without block z-score; RotNorm: block z-score enabled.",
        threshold=threshold,
    )


def run_raw_vs_rot_comparison(
    *,
    raw_results_json: str,
    raw_pkl_dir: str,
    rot_results_json: str,
    rot_pkl_dir: str,
    output_dir: str,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare experiments/infer_data_raw_pkl vs infer_data_rot_pkl."""
    return run_raw_vs_rawnorm_comparison(
        raw_results_json=raw_results_json,
        raw_pkl_dir=raw_pkl_dir,
        rawnorm_results_json=rot_results_json,
        rawnorm_pkl_dir=rot_pkl_dir,
        output_dir=output_dir,
        cohort_a_key="raw",
        cohort_b_key="rot",
        cohort_a_label="Raw",
        cohort_b_label="Rot",
        output_stem="raw_vs_rot",
        figure_title="Raw vs Rot (rotation-calibrated pose)",
        comparison_note="Raw PKLs: step_1 pose; Rot PKLs: rotation-calibrated pose (KM without block z-score).",
        threshold=threshold,
    )


def _print_cohort_pair_summary(
    stats: Dict[str, Any],
    n_a: int,
    n_b: int,
    *,
    cohort_a_key: str,
    cohort_b_key: str,
    cohort_a_label: str,
    cohort_b_label: str,
    title: str,
) -> None:
    print(f"\n{title} comparison ({cohort_a_label} n={n_a}, {cohort_b_label} n={n_b})")
    for key, label in ((cohort_a_key, cohort_a_label), (cohort_b_key, cohort_b_label)):
        p = stats.get(key, {}).get("probability", {})
        m = stats.get("classification_metrics", {}).get(key, {})
        print(f"  {label}:")
        if p.get("mean") is not None:
            print(
                f"    probability: mean={p['mean']:.4f}, pred+={p.get('predicted_positive', 0)} "
                f"({100 * (p.get('predicted_positive_rate') or 0):.1f}%)"
            )
        if m.get("auc_roc") is not None:
            print(f"    AUROC={m['auc_roc']:.4f}, accuracy={m.get('accuracy', 0):.4f}, recall={m.get('recall', 0):.4f}")
    tests = stats.get("statistical_tests", {})
    if tests.get("probability") is not None:
        print(f"  Mann-Whitney (probability): p={tests['probability']:.4g}")


def _print_raw_vs_rawnorm_summary(stats: Dict[str, Any], n_raw: int, n_norm: int) -> None:
    _print_cohort_pair_summary(
        stats, n_raw, n_norm,
        cohort_a_key="raw", cohort_b_key="rawnorm",
        cohort_a_label="Raw", cohort_b_label="RawNorm",
        title="Raw vs RawNorm",
    )


def build_sz11_vs_sz_lift_comparison_figure(
    sz_records: List[Dict[str, Any]],
    sz_lift_records: List[Dict[str, Any]],
    symmetry_rows: List[Dict[str, Any]],
    *,
    sz_pkl_dir: Optional[str],
    sz_lift_pkl_dir: Optional[str],
    threshold: float,
    comparison_stats: Dict[str, Any],
) -> plt.Figure:
    """Compare infer_data_11 (raw step_1) vs infer_data_11_lift (VideoPose3D invariant CSV)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    sz_probs = _cohort_probs(sz_records)
    lift_probs = _cohort_probs(sz_lift_records)
    bins = np.linspace(0.0, 1.0, 21)

    ax = axes[0, 0]
    if sz_probs:
        ax.hist(sz_probs, bins=bins, alpha=0.55, color=COLOR_NEG, label=f"SZ (n={len(sz_probs)})", edgecolor="white")
    if lift_probs:
        ax.hist(lift_probs, bins=bins, alpha=0.55, color=COLOR_SZ_LIFT, label=f"SZ lift (n={len(lift_probs)})", edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Thr={threshold:.2f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Model probability distribution")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for recs, color, label in ((sz_records, COLOR_NEG, "SZ"), (sz_lift_records, COLOR_SZ_LIFT, "SZ lift")):
        xs = [max_label_value(r.get("label_value")) for r in recs]
        ys = [float(r["probability"]) for r in recs]
        valid = [(x, y) for x, y in zip(xs, ys) if not np.isnan(x)]
        if valid:
            ax.scatter([v[0] for v in valid], [v[1] for v in valid], c=color, alpha=0.7, s=32, edgecolors="white", linewidths=0.4, label=label)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Max Cobb angle (°)")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Probability vs max Cobb")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    labels_eng = [eng for eng, _ in SIMILARITY_KEYS]
    x = np.arange(len(labels_eng))
    width = 0.35
    sz_sim = _cohort_similarity_means(sz_records, symmetry_rows)
    sz_means = [float(np.mean(sz_sim[eng])) if sz_sim[eng] else 0.0 for eng in labels_eng]
    ax.bar(x, sz_means, width, label="SZ (raw table CSVs)", color=COLOR_NEG, alpha=0.8)
    ax.axhline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_eng)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean similarity score")
    ax.set_title("Bilateral similarity (raw pose)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    sz_mean, sz_std = _mean_km_profile(sz_records, sz_pkl_dir)
    lift_mean, lift_std = _mean_km_profile(sz_lift_records, sz_lift_pkl_dir)
    if sz_mean is not None:
        t = np.arange(len(sz_mean))
        ax.plot(t, sz_mean, color=COLOR_NEG, linewidth=2, label="SZ mean")
        if sz_std is not None:
            ax.fill_between(t, sz_mean - sz_std, sz_mean + sz_std, color=COLOR_NEG, alpha=0.2)
    if lift_mean is not None:
        t = np.arange(len(lift_mean))
        ax.plot(t, lift_mean, color=COLOR_SZ_LIFT, linewidth=2, label="SZ lift mean")
        if lift_std is not None:
            ax.fill_between(t, lift_mean - lift_std, lift_mean + lift_std, color=COLOR_SZ_LIFT, alpha=0.2)
    ax.set_xlabel("KM timestep")
    ax.set_ylabel("Sum of KM features")
    ax.set_title("KM feature sum (mean ± std)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    by_source = _symmetry_by_source(symmetry_rows)
    for recs, color, marker, label in (
        (sz_records, COLOR_NEG, "o", "SZ"),
        (sz_lift_records, COLOR_SZ_LIFT, "s", "SZ lift"),
    ):
        xs, ys = [], []
        for rec in recs:
            sym = by_source.get(rec.get("source_file"))
            if not sym:
                continue
            score = _symmetry_score(sym)
            if np.isnan(score):
                continue
            xs.append(score)
            ys.append(float(rec["probability"]))
        if xs:
            ax.scatter(xs, ys, c=color, marker=marker, alpha=0.65, s=32, edgecolors="white", linewidths=0.4, label=label)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Mean distribution symmetry score")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Probability vs gait symmetry")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    lift_by_source = {r.get("source_file"): r for r in sz_lift_records if r.get("source_file")}
    paired_a, paired_b = [], []
    for rec in sz_records:
        other = lift_by_source.get(rec.get("source_file"))
        if other:
            paired_a.append(float(rec["probability"]))
            paired_b.append(float(other["probability"]))
    if paired_a:
        ax.scatter(paired_a, paired_b, c=COLOR_SZ_LIFT, alpha=0.65, s=36, edgecolors="white", linewidths=0.4)
        lo = min(min(paired_a), min(paired_b))
        hi = max(max(paired_a), max(paired_b))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1.0, label="y=x")
        if len(paired_a) >= 2:
            r = float(np.corrcoef(paired_a, paired_b)[0, 1])
            ax.text(0.05, 0.95, f"paired n={len(paired_a)}\nr={r:.3f}", transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("SZ probability (step_1)")
    ax.set_ylabel("SZ lift probability")
    ax.set_title("Paired probability: SZ vs SZ lift")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("infer_data_11 vs infer_data_11_lift (VideoPose3D invariant pose)", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def run_sz11_vs_sz_lift_comparison(
    *,
    sz_results_json: str,
    sz_pkl_dir: str,
    sz_lift_results_json: str,
    sz_lift_pkl_dir: str,
    table_dir: str,
    scoli_root: str,
    output_dir: str,
    skip_symmetry: bool = False,
) -> Dict[str, Any]:
    """Compare infer_data_11 vs infer_data_11_lift (same videos, invariant pose CSV)."""
    sz_records = load_inference_results(sz_results_json)
    sz_lift_records = load_inference_results(sz_lift_results_json)
    threshold = float(sz_records[0].get("threshold", sz_lift_records[0].get("threshold", 0.5)))

    infer_mod = _load_infer_helpers()
    compute_metrics = getattr(infer_mod, "compute_metrics", None) if infer_mod else None

    symmetry_rows: List[Dict[str, Any]] = []
    if not skip_symmetry:
        sym_dir = os.path.join(output_dir, "comparison_symmetry_plots", "sz")
        os.makedirs(sym_dir, exist_ok=True)
        print(f"Symmetry: CSVs from {table_dir}")
        symmetry_rows = run_symmetry_eval(sz_records, table_dir, scoli_root, symmetry_plot_dir=sym_dir)

    sz_summary = _cohort_summary_stats("sz", sz_records, symmetry_rows, threshold=threshold)
    lift_summary = _cohort_summary_stats("sz_lift", sz_lift_records, symmetry_rows, threshold=threshold)

    sz_probs = _cohort_probs(sz_records)
    lift_probs = _cohort_probs(sz_lift_records)

    comparison_stats: Dict[str, Any] = {
        "sz": sz_summary,
        "sz_lift": lift_summary,
        "statistical_tests": {
            "probability": _mann_whitney_p(sz_probs, lift_probs),
        },
        "classification_metrics": {},
        "note": "Gait symmetry from raw step_1 CSVs; KM differs via step_1_inv invariant pose.",
    }
    if compute_metrics is not None:
        comparison_stats["classification_metrics"]["sz"] = compute_metrics(sz_records, threshold)
        comparison_stats["classification_metrics"]["sz_lift"] = compute_metrics(sz_lift_records, threshold)

    os.makedirs(output_dir, exist_ok=True)
    fig = build_sz11_vs_sz_lift_comparison_figure(
        sz_records,
        sz_lift_records,
        symmetry_rows,
        sz_pkl_dir=sz_pkl_dir,
        sz_lift_pkl_dir=sz_lift_pkl_dir,
        threshold=threshold,
        comparison_stats=comparison_stats,
    )
    fig_path = os.path.join(output_dir, "sz11_vs_sz_lift_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    json_path = os.path.join(output_dir, "sz11_vs_sz_lift_comparison.json")
    save_json({"comparison": comparison_stats, "sz_symmetry_per_sample": symmetry_rows}, json_path)
    print(f"Saved: {json_path}")

    _print_sz11_vs_sz_lift_summary(comparison_stats, len(sz_records), len(sz_lift_records))
    return comparison_stats


def build_sz11_vs_sz_rot_comparison_figure(
    sz_records: List[Dict[str, Any]],
    sz_rot_records: List[Dict[str, Any]],
    symmetry_rows: List[Dict[str, Any]],
    *,
    sz_pkl_dir: Optional[str],
    sz_rot_pkl_dir: Optional[str],
    threshold: float,
    comparison_stats: Dict[str, Any],
) -> plt.Figure:
    """Compare infer_data_11 (raw step_1) vs infer_data_11_rot (rotation-only CSV)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    sz_probs = _cohort_probs(sz_records)
    rot_probs = _cohort_probs(sz_rot_records)
    bins = np.linspace(0.0, 1.0, 21)

    ax = axes[0, 0]
    if sz_probs:
        ax.hist(sz_probs, bins=bins, alpha=0.55, color=COLOR_NEG, label=f"SZ (n={len(sz_probs)})", edgecolor="white")
    if rot_probs:
        ax.hist(rot_probs, bins=bins, alpha=0.55, color=COLOR_SZ_ROT, label=f"SZ rot (n={len(rot_probs)})", edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Thr={threshold:.2f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Model probability distribution")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for recs, color, label in ((sz_records, COLOR_NEG, "SZ"), (sz_rot_records, COLOR_SZ_ROT, "SZ rot")):
        xs = [max_label_value(r.get("label_value")) for r in recs]
        ys = [float(r["probability"]) for r in recs]
        valid = [(x, y) for x, y in zip(xs, ys) if not np.isnan(x)]
        if valid:
            ax.scatter([v[0] for v in valid], [v[1] for v in valid], c=color, alpha=0.7, s=32, edgecolors="white", linewidths=0.4, label=label)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Max Cobb angle (°)")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Probability vs max Cobb")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    labels_eng = [eng for eng, _ in SIMILARITY_KEYS]
    x = np.arange(len(labels_eng))
    width = 0.35
    sz_sim = _cohort_similarity_means(sz_records, symmetry_rows)
    sz_means = [float(np.mean(sz_sim[eng])) if sz_sim[eng] else 0.0 for eng in labels_eng]
    ax.bar(x, sz_means, width, label="SZ (raw table CSVs)", color=COLOR_NEG, alpha=0.8)
    ax.axhline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_eng)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean similarity score")
    ax.set_title("Bilateral similarity (raw pose)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    sz_mean, sz_std = _mean_km_profile(sz_records, sz_pkl_dir)
    rot_mean, rot_std = _mean_km_profile(sz_rot_records, sz_rot_pkl_dir)
    if sz_mean is not None:
        t = np.arange(len(sz_mean))
        ax.plot(t, sz_mean, color=COLOR_NEG, linewidth=2, label="SZ mean")
        if sz_std is not None:
            ax.fill_between(t, sz_mean - sz_std, sz_mean + sz_std, color=COLOR_NEG, alpha=0.2)
    if rot_mean is not None:
        t = np.arange(len(rot_mean))
        ax.plot(t, rot_mean, color=COLOR_SZ_ROT, linewidth=2, label="SZ rot mean")
        if rot_std is not None:
            ax.fill_between(t, rot_mean - rot_std, rot_mean + rot_std, color=COLOR_SZ_ROT, alpha=0.2)
    ax.set_xlabel("KM timestep")
    ax.set_ylabel("Sum of KM features")
    ax.set_title("KM feature sum (mean ± std)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    by_source = _symmetry_by_source(symmetry_rows)
    for recs, color, marker, label in (
        (sz_records, COLOR_NEG, "o", "SZ"),
        (sz_rot_records, COLOR_SZ_ROT, "s", "SZ rot"),
    ):
        xs, ys = [], []
        for rec in recs:
            sym = by_source.get(rec.get("source_file"))
            if not sym:
                continue
            score = _symmetry_score(sym)
            if np.isnan(score):
                continue
            xs.append(score)
            ys.append(float(rec["probability"]))
        if xs:
            ax.scatter(xs, ys, c=color, marker=marker, alpha=0.65, s=32, edgecolors="white", linewidths=0.4, label=label)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Mean distribution symmetry score")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Probability vs gait symmetry")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    rot_by_source = {r.get("source_file"): r for r in sz_rot_records if r.get("source_file")}
    paired_a, paired_b = [], []
    for rec in sz_records:
        other = rot_by_source.get(rec.get("source_file"))
        if other:
            paired_a.append(float(rec["probability"]))
            paired_b.append(float(other["probability"]))
    if paired_a:
        ax.scatter(paired_a, paired_b, c=COLOR_SZ_ROT, alpha=0.65, s=36, edgecolors="white", linewidths=0.4)
        lo = min(min(paired_a), min(paired_b))
        hi = max(max(paired_a), max(paired_b))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1.0, label="y=x")
        if len(paired_a) >= 2:
            r = float(np.corrcoef(paired_a, paired_b)[0, 1])
            ax.text(0.05, 0.95, f"paired n={len(paired_a)}\nr={r:.3f}", transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("SZ probability (step_1)")
    ax.set_ylabel("SZ rot probability")
    ax.set_title("Paired probability: SZ vs SZ rot")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("infer_data_11 vs infer_data_11_rot (rotation-only pose calibration)", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def run_sz11_vs_sz_rot_comparison(
    *,
    sz_results_json: str,
    sz_pkl_dir: str,
    sz_rot_results_json: str,
    sz_rot_pkl_dir: str,
    table_dir: str,
    scoli_root: str,
    output_dir: str,
    skip_symmetry: bool = False,
) -> Dict[str, Any]:
    """Compare infer_data_11 vs infer_data_11_rot (same videos, rotation-calibrated CSV)."""
    sz_records = load_inference_results(sz_results_json)
    sz_rot_records = load_inference_results(sz_rot_results_json)
    threshold = float(sz_records[0].get("threshold", sz_rot_records[0].get("threshold", 0.5)))

    infer_mod = _load_infer_helpers()
    compute_metrics = getattr(infer_mod, "compute_metrics", None) if infer_mod else None

    symmetry_rows: List[Dict[str, Any]] = []
    if not skip_symmetry:
        sym_dir = os.path.join(output_dir, "comparison_symmetry_plots", "sz")
        os.makedirs(sym_dir, exist_ok=True)
        print(f"Symmetry: CSVs from {table_dir}")
        symmetry_rows = run_symmetry_eval(sz_records, table_dir, scoli_root, symmetry_plot_dir=sym_dir)

    sz_summary = _cohort_summary_stats("sz", sz_records, symmetry_rows, threshold=threshold)
    rot_summary = _cohort_summary_stats("sz_rot", sz_rot_records, symmetry_rows, threshold=threshold)

    sz_probs = _cohort_probs(sz_records)
    rot_probs = _cohort_probs(sz_rot_records)

    comparison_stats: Dict[str, Any] = {
        "sz": sz_summary,
        "sz_rot": rot_summary,
        "statistical_tests": {
            "probability": _mann_whitney_p(sz_probs, rot_probs),
        },
        "classification_metrics": {},
        "note": "Gait symmetry from raw step_1 CSVs; KM differs via step_1_rot rotation calibration.",
    }
    if compute_metrics is not None:
        comparison_stats["classification_metrics"]["sz"] = compute_metrics(sz_records, threshold)
        comparison_stats["classification_metrics"]["sz_rot"] = compute_metrics(sz_rot_records, threshold)

    os.makedirs(output_dir, exist_ok=True)
    fig = build_sz11_vs_sz_rot_comparison_figure(
        sz_records,
        sz_rot_records,
        symmetry_rows,
        sz_pkl_dir=sz_pkl_dir,
        sz_rot_pkl_dir=sz_rot_pkl_dir,
        threshold=threshold,
        comparison_stats=comparison_stats,
    )
    fig_path = os.path.join(output_dir, "sz11_vs_sz_rot_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    json_path = os.path.join(output_dir, "sz11_vs_sz_rot_comparison.json")
    save_json({"comparison": comparison_stats, "sz_symmetry_per_sample": symmetry_rows}, json_path)
    print(f"Saved: {json_path}")

    _print_sz11_vs_sz_rot_summary(comparison_stats, len(sz_records), len(sz_rot_records))
    return comparison_stats


def _print_sz11_vs_sz_rot_summary(stats: Dict[str, Any], n_sz: int, n_rot: int) -> None:
    print(f"\nSZ vs SZ rot comparison (SZ n={n_sz}, SZ rot n={n_rot})")
    for key, label in (("sz", "SZ"), ("sz_rot", "SZ rot")):
        p = stats.get(key, {}).get("probability", {})
        m = stats.get("classification_metrics", {}).get(key, {})
        print(f"  {label}:")
        if p.get("mean") is not None:
            print(
                f"    probability: mean={p['mean']:.4f}, pred+={p.get('predicted_positive', 0)} "
                f"({100 * (p.get('predicted_positive_rate') or 0):.1f}%)"
            )
        if m.get("auc_roc") is not None:
            print(f"    AUROC={m['auc_roc']:.4f}, accuracy={m.get('accuracy', 0):.4f}, recall={m.get('recall', 0):.4f}")
    tests = stats.get("statistical_tests", {})
    if tests.get("probability") is not None:
        print(f"  Mann-Whitney (probability): p={tests['probability']:.4g}")


def _print_sz11_vs_sz_lift_summary(stats: Dict[str, Any], n_sz: int, n_lift: int) -> None:
    print(f"\nSZ vs SZ lift comparison (SZ n={n_sz}, SZ lift n={n_lift})")
    for key, label in (("sz", "SZ"), ("sz_lift", "SZ lift")):
        p = stats.get(key, {}).get("probability", {})
        m = stats.get("classification_metrics", {}).get(key, {})
        print(f"  {label}:")
        if p.get("mean") is not None:
            print(
                f"    probability: mean={p['mean']:.4f}, pred+={p.get('predicted_positive', 0)} "
                f"({100 * (p.get('predicted_positive_rate') or 0):.1f}%)"
            )
        if m.get("auc_roc") is not None:
            print(f"    AUROC={m['auc_roc']:.4f}, accuracy={m.get('accuracy', 0):.4f}, recall={m.get('recall', 0):.4f}")
    tests = stats.get("statistical_tests", {})
    if tests.get("probability") is not None:
        print(f"  Mann-Whitney (probability): p={tests['probability']:.4g}")


def _print_sz11_vs_sz11_norm_summary(stats: Dict[str, Any], n_sz: int, n_norm: int) -> None:
    print(f"\nSZ vs SZ norm comparison (SZ n={n_sz}, SZ norm n={n_norm})")
    for key, label in (("sz", "SZ"), ("sz_norm", "SZ norm")):
        p = stats.get(key, {}).get("probability", {})
        m = stats.get("classification_metrics", {}).get(key, {})
        print(f"  {label}:")
        if p.get("mean") is not None:
            print(
                f"    probability: mean={p['mean']:.4f}, pred+={p.get('predicted_positive', 0)} "
                f"({100 * (p.get('predicted_positive_rate') or 0):.1f}%)"
            )
        if m.get("auc_roc") is not None:
            print(f"    AUROC={m['auc_roc']:.4f}, accuracy={m.get('accuracy', 0):.4f}, recall={m.get('recall', 0):.4f}")
    tests = stats.get("statistical_tests", {})
    if tests.get("probability") is not None:
        print(f"  Mann-Whitney (probability): p={tests['probability']:.4g}")


def build_negative_sample_values_figure(
    records: List[Dict[str, Any]],
    *,
    binary_cobb_threshold: float = DEFAULT_BINARY_COBB_THRESHOLD,
) -> plt.Figure:
    """
    Visualize distributions for ground-truth negative samples (label=0).

    Panels: max Cobb histogram, M_cobb_l / M_cobb_r histograms, model probability.
    """
    neg = negative_records(records)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    max_vals = [max_label_value(r.get("label_value")) for r in neg]
    max_vals = [v for v in max_vals if not np.isnan(v)]

    ax0 = axes[0]
    if max_vals:
        ax0.hist(
            max_vals,
            bins=max(8, min(20, len(max_vals) // 2)),
            color=COLOR_NEG,
            alpha=0.75,
            edgecolor="white",
        )
        ax0.axvline(
            binary_cobb_threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Binary thr={binary_cobb_threshold:.0f}°",
        )
        mean_v = float(np.mean(max_vals))
        ax0.axvline(mean_v, color="#2ca02c", linestyle="-", linewidth=1.0, label=f"Mean={mean_v:.1f}°")
        ax0.text(
            0.98, 0.95,
            f"n={len(max_vals)}\nmin={min(max_vals):.1f}\nmax={max(max_vals):.1f}",
            transform=ax0.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    else:
        ax0.text(0.5, 0.5, "No negative samples", ha="center", va="center", transform=ax0.transAxes)
    ax0.set_xlabel("Max Cobb angle (°)")
    ax0.set_ylabel("Count")
    ax0.set_title("Negative samples: max Cobb distribution")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    cobb_l, cobb_r = [], []
    for rec in neg:
        left, right = cobb_components(rec.get("label_value"))
        if not np.isnan(left):
            cobb_l.append(left)
        if not np.isnan(right):
            cobb_r.append(right)
    if cobb_l or cobb_r:
        combined = cobb_l + cobb_r
        bins = max(8, min(20, len(combined) // 3))
        if cobb_l:
            ax1.hist(cobb_l, bins=bins, alpha=0.6, color="#6baed6", label=f"M_cobb_l (n={len(cobb_l)})", edgecolor="white")
        if cobb_r:
            ax1.hist(cobb_r, bins=bins, alpha=0.6, color="#fd8d3c", label=f"M_cobb_r (n={len(cobb_r)})", edgecolor="white")
    else:
        ax1.text(0.5, 0.5, "No Cobb components", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_xlabel("Cobb angle (°)")
    ax1.set_ylabel("Count")
    ax1.set_title("Negative samples: L / R Cobb distribution")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[2]
    probs = [float(r["probability"]) for r in neg]
    pred_threshold = float(records[0].get("threshold", 0.5)) if records else 0.5
    if probs:
        ax2.hist(
            probs,
            bins=np.linspace(0.0, 1.0, 21),
            color=COLOR_NEG,
            alpha=0.75,
            edgecolor="white",
        )
        ax2.axvline(
            pred_threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Model thr={pred_threshold:.2f}",
        )
        n_fp = sum(1 for p in probs if p >= pred_threshold)
        ax2.text(
            0.98, 0.95,
            f"n={len(probs)}\nFP rate={n_fp / len(probs):.1%}",
            transform=ax2.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    else:
        ax2.text(0.5, 0.5, "No probabilities", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Negative samples: model probability")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Ground-truth negative sample distributions", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def resolve_pkl_path(record: Dict[str, Any], pkl_dir: Optional[str], infer_mod) -> Optional[str]:
    pkl_path = record.get("pkl_path")
    if pkl_path and os.path.isfile(pkl_path):
        return pkl_path
    if pkl_dir and infer_mod is not None:
        meta = {"patch_id": record.get("patch_id"), "pkl_path": pkl_path}
        return infer_mod.resolve_patch_path(meta, pkl_dir)
    if pkl_dir and record.get("patch_id") is not None:
        alt = os.path.join(pkl_dir, "patches", f"patch_{record['patch_id']}.pkl")
        if os.path.isfile(alt):
            return alt
    return None


def resolve_csv_path(record: Dict[str, Any], table_dir: str) -> Optional[str]:
    """Map inference source_file (e.g. sz_884_step1) to step-1 pose CSV."""
    source = record.get("source_file") or ""
    if not source:
        return None
    if source.endswith("_step1"):
        csv_name = source.replace("_step1", "_step_1.csv")
    elif source.endswith("_step_1"):
        csv_name = f"{source}.csv"
    else:
        csv_name = f"{source}_step_1.csv"
    csv_path = os.path.join(table_dir, csv_name)
    return csv_path if os.path.isfile(csv_path) else None


def _compact_symmetry(symmetry: Dict[str, Any]) -> Dict[str, Any]:
    summary = symmetry.get("summary", {})
    similarity = summary.get("similarity_output", {})
    out: Dict[str, Any] = {
        "n_frames": symmetry.get("n_frames"),
        "mean_distribution_symmetry_score": summary.get("mean_distribution_symmetry_score"),
        "n_symmetric": summary.get("n_symmetric"),
        "n_asymmetric": summary.get("n_asymmetric"),
        "asymmetric_factors": summary.get("asymmetric_factors", []),
        "distribution_symmetry_threshold": summary.get("distribution_symmetry_threshold", SYMMETRY_THRESHOLD),
        "similarity_output": similarity,
        "similarity_strings": summary.get("similarity_strings", []),
    }
    for eng, cn_key in SIMILARITY_KEYS:
        raw = similarity.get(cn_key)
        if raw is not None:
            try:
                out[f"similarity_{eng.lower()}"] = float(raw)
            except (TypeError, ValueError):
                pass
    return out


def run_symmetry_eval(
    records: List[Dict[str, Any]],
    table_dir: str,
    scoli_root: str,
    *,
    symmetry_threshold: float = SYMMETRY_THRESHOLD,
    symmetry_plot_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run analyze_gait_symmetry on pose CSV for each inference record."""
    analyze_gait_symmetry = _import_symmetry_module(scoli_root)
    symmetry_rows: List[Dict[str, Any]] = []

    for rec in records:
        csv_path = resolve_csv_path(rec, table_dir)
        row: Dict[str, Any] = {
            "patch_id": rec.get("patch_id"),
            "source_file": rec.get("source_file"),
            "subject_id": rec.get("subject_id"),
            "label": rec.get("label"),
            "probability": rec.get("probability"),
            "csv_path": csv_path,
        }
        if not csv_path:
            row["error"] = "CSV not found"
            symmetry_rows.append(row)
            continue

        plot_path = None
        if symmetry_plot_dir:
            stem = rec.get("source_file") or rec.get("patch_id", "sample")
            plot_path = os.path.join(symmetry_plot_dir, f"{stem}_symmetry.png")

        try:
            symmetry = analyze_gait_symmetry(
                csv_path,
                distribution_symmetry_threshold=symmetry_threshold,
                plot_path=plot_path,
                source_video=rec.get("source_file"),
            )
            compact = _compact_symmetry(symmetry)
            row.update(compact)
            if plot_path and symmetry.get("symmetry_plot_path"):
                row["symmetry_plot_path"] = symmetry["symmetry_plot_path"]
        except Exception as exc:
            row["error"] = str(exc)

        symmetry_rows.append(row)

    return symmetry_rows


def _symmetry_score(row: Dict[str, Any]) -> float:
    val = row.get("mean_distribution_symmetry_score")
    return float(val) if val is not None else np.nan


def _similarity_value(sym_row: Dict[str, Any], eng: str) -> float:
    """Read one bilateral similarity score from a symmetry row."""
    cn_key = next(cn for e, cn in SIMILARITY_KEYS if e == eng)
    raw = sym_row.get(f"similarity_{eng.lower()}")
    if raw is None and sym_row.get("similarity_output"):
        raw = sym_row["similarity_output"].get(cn_key)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return np.nan


def _symmetry_by_source(symmetry_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        r.get("source_file"): r
        for r in symmetry_rows
        if r.get("error") is None and r.get("source_file")
    }


def plot_max_label_vs_similarity_trends(
    records: List[Dict[str, Any]],
    symmetry_rows: List[Dict[str, Any]],
    ax: plt.Axes,
    *,
    eng: str,
) -> None:
    """Scatter max(label_value) vs one similarity score with linear trend."""
    by_source = _symmetry_by_source(symmetry_rows)
    xs_all: List[float] = []
    ys_all: List[float] = []

    for rec in records:
        sym = by_source.get(rec.get("source_file"))
        if not sym:
            continue
        x = max_label_value(rec.get("label_value"))
        y = _similarity_value(sym, eng)
        if np.isnan(x) or np.isnan(y):
            continue
        xs_all.append(x)
        ys_all.append(y)

    if not xs_all:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(eng)
        return

    xs = np.array(xs_all)
    ys = np.array(ys_all)

    for label_val, color, name in ((1.0, COLOR_POS, "Pos"), (0.0, COLOR_NEG, "Neg")):
        mask = np.array([rec.get("label") == label_val for rec in records])
        # align mask with xs_all - need paired iteration
        x_pts, y_pts = [], []
        for rec in records:
            sym = by_source.get(rec.get("source_file"))
            if not sym or rec.get("label") != label_val:
                continue
            x = max_label_value(rec.get("label_value"))
            y = _similarity_value(sym, eng)
            if np.isnan(x) or np.isnan(y):
                continue
            x_pts.append(x)
            y_pts.append(y)
        if x_pts:
            ax.scatter(x_pts, y_pts, c=color, alpha=0.7, s=28, edgecolors="white", linewidths=0.4, label=name)

    if len(xs) >= 2:
        coef = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(x_line, np.polyval(coef, x_line), "k-", alpha=0.5, linewidth=1.5, label="Trend")
        r = float(np.corrcoef(xs, ys)[0, 1]) if np.std(xs) > 0 and np.std(ys) > 0 else np.nan
        if np.isfinite(r):
            ax.text(
                0.05, 0.95, f"r={r:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

    ax.axhline(SYMMETRY_THRESHOLD, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Max label value")
    ax.set_ylabel("Similarity score")
    ax.set_title(eng)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)


def build_similarity_vs_label_figure(
    records: List[Dict[str, Any]],
    symmetry_rows: List[Dict[str, Any]],
) -> plt.Figure:
    """2x2 panel: max Cobb (label value) vs shoulder/arm/stride/trunk similarity."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (eng, _) in zip(axes.ravel(), SIMILARITY_KEYS):
        plot_max_label_vs_similarity_trends(records, symmetry_rows, ax, eng=eng)
    fig.suptitle("Max label value vs bilateral similarity scores", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_symmetry_score_distribution(symmetry_rows: List[Dict[str, Any]], ax: plt.Axes) -> None:
    """Histogram of mean distribution symmetry score, split by ground-truth label."""
    valid = [r for r in symmetry_rows if r.get("error") is None and not np.isnan(_symmetry_score(r))]
    if not valid:
        ax.text(0.5, 0.5, "No symmetry scores", ha="center", va="center", transform=ax.transAxes)
        return

    bins = np.linspace(0.0, 1.0, 21)
    for label, color, name in ((1.0, COLOR_POS, "Positive"), (0.0, COLOR_NEG, "Negative")):
        scores = [_symmetry_score(r) for r in valid if r.get("label") == label]
        if scores:
            ax.hist(scores, bins=bins, alpha=0.6, color=color, label=f"{name} (n={len(scores)})", edgecolor="white")

    ax.axvline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.2, label=f"Threshold={SYMMETRY_THRESHOLD}")
    ax.set_xlabel("Mean distribution symmetry score")
    ax.set_ylabel("Count")
    ax.set_title("Gait symmetry score by label")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_probability_vs_symmetry_score(
    records: List[Dict[str, Any]],
    symmetry_rows: List[Dict[str, Any]],
    ax: plt.Axes,
) -> None:
    """Scatter: model probability vs mean symmetry score."""
    by_source = {r.get("source_file"): r for r in symmetry_rows if r.get("error") is None}
    xs, ys = [], []
    for rec in records:
        sym = by_source.get(rec.get("source_file"))
        if not sym:
            continue
        score = _symmetry_score(sym)
        if np.isnan(score):
            continue
        xs.append(score)
        ys.append(float(rec["probability"]))

    if not xs:
        ax.text(0.5, 0.5, "No symmetry / probability pairs", ha="center", va="center", transform=ax.transAxes)
        return

    for label_val, color, name in ((1.0, COLOR_POS, "Positive"), (0.0, COLOR_NEG, "Negative")):
        mask = [
            i
            for i, rec in enumerate(records)
            if rec.get("label") == label_val and rec.get("source_file") in by_source
            and not np.isnan(_symmetry_score(by_source[rec.get("source_file")]))
        ]
        if not mask:
            continue
        ax.scatter(
            [_symmetry_score(by_source[records[i]["source_file"]]) for i in mask],
            [float(records[i]["probability"]) for i in mask],
            c=color,
            alpha=0.75,
            s=40,
            edgecolors="white",
            linewidths=0.5,
            label=name,
        )

    threshold = float(records[0].get("threshold", 0.5)) if records else 0.5
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Mean distribution symmetry score")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Probability vs gait symmetry score")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_cohort_similarity_scores(symmetry_rows: List[Dict[str, Any]], ax: plt.Axes) -> None:
    """Bar chart of mean bilateral similarity scores (shoulder / arm / stride / trunk)."""
    valid = [r for r in symmetry_rows if r.get("error") is None]
    if not valid:
        ax.text(0.5, 0.5, "No symmetry data", ha="center", va="center", transform=ax.transAxes)
        return

    labels_eng = [eng for eng, _ in SIMILARITY_KEYS]
    x = np.arange(len(labels_eng))
    width = 0.35

    def _means(rows: List[Dict[str, Any]]) -> List[float]:
        out = []
        for eng, cn in SIMILARITY_KEYS:
            vals = []
            for r in rows:
                v = r.get(f"similarity_{eng.lower()}")
                if v is None and r.get("similarity_output"):
                    try:
                        v = float(r["similarity_output"].get(cn, np.nan))
                    except (TypeError, ValueError):
                        v = np.nan
                if v is not None and not np.isnan(v):
                    vals.append(float(v))
            out.append(float(np.mean(vals)) if vals else 0.0)
        return out

    pos_rows = [r for r in valid if r.get("label") == 1.0]
    neg_rows = [r for r in valid if r.get("label") == 0.0]
    means_pos = _means(pos_rows)
    means_neg = _means(neg_rows)

    if pos_rows:
        ax.bar(x - width / 2, means_pos, width, label=f"Positive (n={len(pos_rows)})", color=COLOR_POS, alpha=0.75)
    if neg_rows:
        ax.bar(x + width / 2, means_neg, width, label=f"Negative (n={len(neg_rows)})", color=COLOR_NEG, alpha=0.75)

    ax.axhline(SYMMETRY_THRESHOLD, color="black", linestyle="--", linewidth=1.0, label=f"Threshold={SYMMETRY_THRESHOLD}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_eng)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Similarity score")
    ax.set_title("Bilateral similarity (cohort mean)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def load_km_sum_profiles(
    records: List[Dict[str, Any]],
    pkl_dir: Optional[str],
) -> Tuple[List[np.ndarray], List[float], List[str]]:
    infer_mod = _load_infer_helpers()
    profiles: List[np.ndarray] = []
    labels: List[float] = []
    names: List[str] = []

    for rec in records:
        pkl_path = resolve_pkl_path(rec, pkl_dir, infer_mod)
        if not pkl_path:
            continue
        with open(pkl_path, "rb") as f:
            patch = pickle.load(f)
        km = np.asarray(patch["knowledge_map"], dtype=np.float64)
        profiles.append(km.sum(axis=1))
        labels.append(float(rec["label"]) if rec.get("label") is not None else np.nan)
        names.append(rec.get("source_file") or rec.get("patch_id", ""))

    return profiles, labels, names


def plot_km_sum_profiles(profiles: List[np.ndarray], labels: List[float], ax: plt.Axes) -> None:
    for profile, label in zip(profiles, labels):
        if label == 1.0:
            color, alpha = COLOR_POS, 0.45
        elif label == 0.0:
            color, alpha = COLOR_NEG, 0.45
        else:
            color, alpha = "gray", 0.25
        ax.plot(profile, color=color, alpha=alpha, linewidth=1.0)

    ax.plot([], [], color=COLOR_POS, linewidth=2, label=f"Positive (n={sum(l == 1.0 for l in labels)})")
    ax.plot([], [], color=COLOR_NEG, linewidth=2, label=f"Negative (n={sum(l == 0.0 for l in labels)})")
    ax.set_xlabel("KM timestep")
    ax.set_ylabel("Sum of KM features")
    ax.set_title("KM feature sum per sample")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def plot_prediction_distribution(records: List[Dict[str, Any]], ax: plt.Axes) -> None:
    probs = np.array([float(r["probability"]) for r in records])
    preds = np.array([int(r["prediction"]) for r in records])
    threshold = float(records[0].get("threshold", 0.5)) if records else 0.5
    bins = np.linspace(0.0, 1.0, 21)

    probs_neg = probs[preds == 0]
    probs_pos = probs[preds == 1]
    if len(probs_neg):
        ax.hist(probs_neg, bins=bins, alpha=0.6, color=COLOR_PRED_NEG, label=f"Pred neg (n={len(probs_neg)})", edgecolor="white")
    if len(probs_pos):
        ax.hist(probs_pos, bins=bins, alpha=0.6, color=COLOR_PRED_POS, label=f"Pred pos (n={len(probs_pos)})", edgecolor="white")

    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Thr={threshold:.2f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction probability distribution")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_probability_vs_label_value(records: List[Dict[str, Any]], ax: plt.Axes) -> None:
    xs_arr = np.array([max_label_value(r.get("label_value")) for r in records], dtype=float)
    ys_arr = np.array([float(r["probability"]) for r in records], dtype=float)
    valid = ~np.isnan(xs_arr)
    if not valid.any():
        ax.text(0.5, 0.5, "No label_value data", ha="center", va="center", transform=ax.transAxes)
        return

    for label_val, color, name in ((1.0, COLOR_POS, "Positive"), (0.0, COLOR_NEG, "Negative")):
        mask = valid & np.array([r.get("label") == label_val for r in records])
        if not mask.any():
            continue
        ax.scatter(xs_arr[mask], ys_arr[mask], c=color, alpha=0.75, s=40, edgecolors="white", linewidths=0.5, label=name)

    threshold = float(records[0].get("threshold", 0.5)) if records else 0.5
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    xs_valid = xs_arr[valid]
    if valid.sum() >= 2 and np.ptp(xs_valid) > 0:
        try:
            coef = np.polyfit(xs_valid, ys_arr[valid], 1)
            x_line = np.linspace(xs_valid.min(), xs_valid.max(), 100)
            ax.plot(x_line, np.polyval(coef, x_line), "k-", alpha=0.35, linewidth=1.5, label="Linear fit")
        except np.linalg.LinAlgError:
            pass

    ax.set_xlabel("Max label value (e.g. max Cobb)")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Probability vs max label value")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def build_analysis_figure(
    records: List[Dict[str, Any]],
    pkl_dir: Optional[str],
    symmetry_rows: Optional[List[Dict[str, Any]]] = None,
) -> plt.Figure:
    profiles, profile_labels, _ = load_km_sum_profiles(records, pkl_dir)
    symmetry_rows = symmetry_rows or []

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    if profiles:
        plot_km_sum_profiles(profiles, profile_labels, axes[0, 0])
    else:
        axes[0, 0].text(0.5, 0.5, "No KM PKL data", ha="center", va="center", transform=axes[0, 0].transAxes)
        axes[0, 0].set_title("KM feature sum per sample")

    plot_prediction_distribution(records, axes[0, 1])
    plot_probability_vs_label_value(records, axes[0, 2])

    plot_symmetry_score_distribution(symmetry_rows, axes[1, 0])
    plot_probability_vs_symmetry_score(records, symmetry_rows, axes[1, 1])
    plot_cohort_similarity_scores(symmetry_rows, axes[1, 2])

    fig.tight_layout()
    return fig


def _symmetry_cohort_summary(symmetry_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in symmetry_rows if r.get("error") is None]
    failed = [r for r in symmetry_rows if r.get("error")]

    def _label_mean(rows: List[Dict[str, Any]], label: float) -> Optional[float]:
        scores = [_symmetry_score(r) for r in rows if r.get("label") == label]
        scores = [s for s in scores if not np.isnan(s)]
        return float(np.mean(scores)) if scores else None

    return {
        "n_total": len(symmetry_rows),
        "n_analyzed": len(valid),
        "n_failed": len(failed),
        "symmetry_threshold": SYMMETRY_THRESHOLD,
        "mean_score_all": float(np.nanmean([_symmetry_score(r) for r in valid])) if valid else None,
        "mean_score_positive": _label_mean(valid, 1.0),
        "mean_score_negative": _label_mean(valid, 0.0),
    }


def _print_symmetry_summary(symmetry_rows: List[Dict[str, Any]]) -> None:
    summary = _symmetry_cohort_summary(symmetry_rows)
    print(f"\nGait symmetry analysis: {summary['n_analyzed']}/{summary['n_total']} samples")
    if summary["mean_score_all"] is not None:
        print(f"  Mean symmetry score (all):      {summary['mean_score_all']:.4f}")
    if summary["mean_score_positive"] is not None:
        print(f"  Mean symmetry score (positive): {summary['mean_score_positive']:.4f}")
    if summary["mean_score_negative"] is not None:
        print(f"  Mean symmetry score (negative): {summary['mean_score_negative']:.4f}")
    if summary["n_failed"]:
        print(f"  Failed: {summary['n_failed']} samples (see symmetry_results.json)")


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run_experiment_leaderboard(
    results_dir: str,
    *,
    baseline_auc: float = TARGET_AUROC,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Rank calibration experiment results by AUROC."""
    results_path = Path(results_dir)
    if not results_path.is_dir():
        raise FileNotFoundError(f"Experiment results directory not found: {results_dir}")

    rows: List[Dict[str, Any]] = []
    for path in sorted(results_path.glob("*.json")):
        if path.name == "leaderboard.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        auc = data.get("auc_roc")
        if data.get("pilot") and data.get("status") in ("pilot_fail", "pilot_pass"):
            phase = "pilot"
        elif data.get("status") == "rejected_km_gate":
            phase = "km_gate"
        else:
            phase = "full"
        rows.append({
            "variant": data.get("variant", path.stem),
            "track": data.get("track", "?"),
            "auc_roc": auc,
            "status": data.get("status"),
            "phase": phase,
            "km_profile": data.get("km_profile"),
            "km_gate_passed": (data.get("km_gate") or {}).get("passed"),
            "beats_baseline": auc is not None and float(auc) > baseline_auc,
            "path": str(path),
        })

    rows.sort(key=lambda r: (r["auc_roc"] is None, -(r["auc_roc"] or 0)))

    print(f"\nCalibration experiment leaderboard (baseline AUROC > {baseline_auc})")
    print(f"{'variant':28s} {'track':5s} {'phase':8s} {'AUROC':>8s} {'status':16s} {'beat':>5s}")
    print("-" * 80)
    for r in rows:
        auc_str = f"{r['auc_roc']:.4f}" if r["auc_roc"] is not None else "   n/a"
        beat = "YES" if r["beats_baseline"] else "no"
        print(
            f"{r['variant']:28s} {r['track']:5s} {r['phase']:8s} {auc_str:>8s} "
            f"{r['status'] or '':16s} {beat:>5s}"
        )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"baseline_auc": baseline_auc, "rankings": rows}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved leaderboard: {out}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize inference + gait symmetry results.")
    parser.add_argument("--results-json", default=str(DEFAULT_RESULTS_JSON))
    parser.add_argument("--pkl-dir", default=str(DEFAULT_PKL_DIR))
    parser.add_argument("--table-dir", default=str(DEFAULT_TABLE_DIR), help="Pose CSV directory (step_1)")
    parser.add_argument("--scoli-root", default=str(DEFAULT_SCOLI_ROOT))
    parser.add_argument("--output", default=None, help="Summary figure path")
    parser.add_argument(
        "--symmetry-plots-dir",
        default=None,
        help="Per-sample symmetry distribution PNGs (default: {results_dir}/symmetry_plots)",
    )
    parser.add_argument("--skip-symmetry", action="store_true", help="Skip gait symmetry scoring")
    parser.add_argument(
        "--binary-cobb-threshold",
        type=float,
        default=DEFAULT_BINARY_COBB_THRESHOLD,
        help="Cobb threshold used for binary labels (shown on negative value plots)",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--compare-pk-vs-sz-neg",
        action="store_true",
        help="Compare infer_data_pk vs ground-truth negatives from infer_data_11",
    )
    parser.add_argument(
        "--compare-sz-pos-vs-neg",
        action="store_true",
        help="Compare SZ positive vs negative samples from one inference JSON",
    )
    parser.add_argument(
        "--compare-baseline-vs-posttrain",
        action="store_true",
        help="Compare baseline vs post-trained inference on the same holdout patches",
    )
    parser.add_argument(
        "--baseline-inference-dir",
        default=None,
        help="Baseline inference output directory (contains inference_results.json)",
    )
    parser.add_argument(
        "--posttrain-inference-dir",
        default=None,
        help="Post-trained inference output directory (contains inference_results.json)",
    )
    parser.add_argument(
        "--compare-pk-vs-pk-norm",
        action="store_true",
        help="Compare infer_data_pk vs infer_data_pk_norm (KM block z-score)",
    )
    parser.add_argument(
        "--compare-sz11-vs-norm",
        action="store_true",
        help="Compare infer_data_11 vs infer_data_11_norm",
    )
    parser.add_argument(
        "--compare-sz11-vs-lift",
        action="store_true",
        help="Compare infer_data_11 vs infer_data_11_lift (VideoPose3D invariant pose)",
    )
    parser.add_argument(
        "--compare-sz11-vs-rot",
        action="store_true",
        help="Compare infer_data_11 vs infer_data_11_rot (rotation-only pose calibration)",
    )
    parser.add_argument(
        "--compare-raw-vs-rawnorm",
        action="store_true",
        help="Compare experiments/infer_data_raw_pkl vs infer_data_rawnorm_pkl (KM block z-score)",
    )
    parser.add_argument(
        "--compare-raw-vs-rot",
        action="store_true",
        help="Compare experiments/infer_data_raw_pkl vs infer_data_rot_pkl (rotation-calibrated pose)",
    )
    parser.add_argument(
        "--compare-rot-vs-rotnorm",
        action="store_true",
        help="Compare experiments/infer_data_rot_pkl vs infer_data_rotnorm_pkl (KM block z-score)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification probability cutoff for comparison metrics/plots (default: from inference JSON)",
    )
    parser.add_argument("--pk-results-json", default=str(DEFAULT_PK_RESULTS_JSON))
    parser.add_argument("--pk-pkl-dir", default=str(DEFAULT_PK_PKL_DIR))
    parser.add_argument("--pk-norm-results-json", default=str(DEFAULT_PK_NORM_RESULTS_JSON))
    parser.add_argument("--pk-norm-pkl-dir", default=str(DEFAULT_PK_NORM_PKL_DIR))
    parser.add_argument("--pk-table-dir", default=str(DEFAULT_PK_TABLE_DIR))
    parser.add_argument(
        "--sz-results-json",
        default=str(DEFAULT_RESULTS_JSON),
        help="infer_data_11 inference results (negatives extracted from label=0 + label_value)",
    )
    parser.add_argument("--sz-pkl-dir", default=str(DEFAULT_PKL_DIR))
    parser.add_argument("--sz-norm-results-json", default=str(DEFAULT_SZ_NORM_RESULTS_JSON))
    parser.add_argument("--sz-norm-pkl-dir", default=str(DEFAULT_SZ_NORM_PKL_DIR))
    parser.add_argument("--sz-lift-results-json", default=str(DEFAULT_SZ_LIFT_RESULTS_JSON))
    parser.add_argument("--sz-lift-pkl-dir", default=str(DEFAULT_SZ_LIFT_PKL_DIR))
    parser.add_argument("--sz-lift-table-dir", default=str(DEFAULT_SZ_LIFT_TABLE_DIR))
    parser.add_argument("--sz-rot-results-json", default=str(DEFAULT_SZ_ROT_RESULTS_JSON))
    parser.add_argument("--sz-rot-pkl-dir", default=str(DEFAULT_SZ_ROT_PKL_DIR))
    parser.add_argument("--sz-rot-table-dir", default=str(DEFAULT_SZ_ROT_TABLE_DIR))
    parser.add_argument("--raw-results-json", default=str(DEFAULT_RAW_RESULTS_JSON))
    parser.add_argument("--raw-pkl-dir", default=str(DEFAULT_RAW_PKL_DIR))
    parser.add_argument("--rawnorm-results-json", default=str(DEFAULT_RAWNORM_RESULTS_JSON))
    parser.add_argument("--rawnorm-pkl-dir", default=str(DEFAULT_RAWNORM_PKL_DIR))
    parser.add_argument("--rot-results-json", default=str(DEFAULT_EXPERIMENTS_ROT_RESULTS_JSON))
    parser.add_argument("--rot-pkl-dir", default=str(DEFAULT_EXPERIMENTS_ROT_PKL_DIR))
    parser.add_argument("--rotnorm-results-json", default=str(DEFAULT_EXPERIMENTS_ROTNORM_RESULTS_JSON))
    parser.add_argument("--rotnorm-pkl-dir", default=str(DEFAULT_EXPERIMENTS_ROTNORM_PKL_DIR))
    parser.add_argument(
        "--compare-output-dir",
        default=None,
        help="Output directory for comparison figures (default: parent of pk-results-json)",
    )
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Print AUROC leaderboard from experiments/results/*.json",
    )
    parser.add_argument(
        "--leaderboard-dir",
        default=str(DEFAULT_EXPERIMENTS_RESULTS_DIR),
        help="Directory with per-variant experiment result JSON files",
    )
    parser.add_argument(
        "--leaderboard-output",
        default=None,
        help="Optional path to save leaderboard JSON",
    )
    args = parser.parse_args()

    if args.leaderboard:
        run_experiment_leaderboard(
            args.leaderboard_dir,
            output_path=args.leaderboard_output,
        )
        return

    if args.compare_rot_vs_rotnorm:
        rot_results = os.path.abspath(args.rot_results_json)
        rotnorm_results = os.path.abspath(args.rotnorm_results_json)
        for path, name in ((rot_results, "Rot results"), (rotnorm_results, "RotNorm results")):
            if not os.path.isfile(path):
                raise SystemExit(f"{name} JSON not found: {path}")
        compare_out = args.compare_output_dir
        if compare_out is None:
            compare_out = str(Path(rotnorm_results).parent.parent / "comparison_rot_vs_rotnorm")
        run_rot_vs_rotnorm_comparison(
            rot_results_json=rot_results,
            rot_pkl_dir=os.path.abspath(args.rot_pkl_dir),
            rotnorm_results_json=rotnorm_results,
            rotnorm_pkl_dir=os.path.abspath(args.rotnorm_pkl_dir),
            output_dir=os.path.abspath(compare_out),
            threshold=args.threshold,
        )
        if args.show:
            plt.show()
        return

    if args.compare_raw_vs_rot:
        raw_results = os.path.abspath(args.raw_results_json)
        rot_results = os.path.abspath(args.rot_results_json)
        for path, name in ((raw_results, "Raw results"), (rot_results, "Rot results")):
            if not os.path.isfile(path):
                raise SystemExit(f"{name} JSON not found: {path}")
        compare_out = args.compare_output_dir
        if compare_out is None:
            compare_out = str(Path(rot_results).parent.parent / "comparison_raw_vs_rot")
        run_raw_vs_rot_comparison(
            raw_results_json=raw_results,
            raw_pkl_dir=os.path.abspath(args.raw_pkl_dir),
            rot_results_json=rot_results,
            rot_pkl_dir=os.path.abspath(args.rot_pkl_dir),
            output_dir=os.path.abspath(compare_out),
            threshold=args.threshold,
        )
        if args.show:
            plt.show()
        return

    if args.compare_raw_vs_rawnorm:
        raw_results = os.path.abspath(args.raw_results_json)
        rawnorm_results = os.path.abspath(args.rawnorm_results_json)
        for path, name in ((raw_results, "Raw results"), (rawnorm_results, "RawNorm results")):
            if not os.path.isfile(path):
                raise SystemExit(f"{name} JSON not found: {path}")
        compare_out = args.compare_output_dir
        if compare_out is None:
            compare_out = str(Path(rawnorm_results).parent.parent / "comparison_raw_vs_rawnorm")
        run_raw_vs_rawnorm_comparison(
            raw_results_json=raw_results,
            raw_pkl_dir=os.path.abspath(args.raw_pkl_dir),
            rawnorm_results_json=rawnorm_results,
            rawnorm_pkl_dir=os.path.abspath(args.rawnorm_pkl_dir),
            output_dir=os.path.abspath(compare_out),
            threshold=args.threshold,
        )
        if args.show:
            plt.show()
        return

    if args.compare_sz11_vs_rot:
        sz_results = os.path.abspath(args.sz_results_json)
        sz_rot_results = os.path.abspath(args.sz_rot_results_json)
        for path, name in ((sz_results, "SZ results"), (sz_rot_results, "SZ rot results")):
            if not os.path.isfile(path):
                raise SystemExit(f"{name} JSON not found: {path}")
        compare_out = args.compare_output_dir
        if compare_out is None:
            compare_out = os.path.join(os.path.dirname(sz_rot_results), "..", "comparison_sz11_vs_rot")
        run_sz11_vs_sz_rot_comparison(
            sz_results_json=sz_results,
            sz_pkl_dir=os.path.abspath(args.sz_pkl_dir),
            sz_rot_results_json=sz_rot_results,
            sz_rot_pkl_dir=os.path.abspath(args.sz_rot_pkl_dir),
            table_dir=os.path.abspath(args.table_dir),
            scoli_root=args.scoli_root,
            output_dir=os.path.abspath(compare_out),
            skip_symmetry=args.skip_symmetry,
        )
        if args.show:
            plt.show()
        return

    if args.compare_sz11_vs_lift:
        sz_results = os.path.abspath(args.sz_results_json)
        sz_lift_results = os.path.abspath(args.sz_lift_results_json)
        for path, name in ((sz_results, "SZ results"), (sz_lift_results, "SZ lift results")):
            if not os.path.isfile(path):
                raise SystemExit(f"{name} JSON not found: {path}")
        compare_out = args.compare_output_dir
        if compare_out is None:
            compare_out = os.path.join(os.path.dirname(sz_lift_results), "comparison_sz11_vs_lift")
        run_sz11_vs_sz_lift_comparison(
            sz_results_json=sz_results,
            sz_pkl_dir=os.path.abspath(args.sz_pkl_dir),
            sz_lift_results_json=sz_lift_results,
            sz_lift_pkl_dir=os.path.abspath(args.sz_lift_pkl_dir),
            table_dir=os.path.abspath(args.table_dir),
            scoli_root=args.scoli_root,
            output_dir=os.path.abspath(compare_out),
            skip_symmetry=args.skip_symmetry,
        )
        if args.show:
            plt.show()
        return

    if args.compare_sz11_vs_norm:
        sz_results = os.path.abspath(args.sz_results_json)
        sz_norm_results = os.path.abspath(args.sz_norm_results_json)
        for path, name in ((sz_results, "SZ results"), (sz_norm_results, "SZ norm results")):
            if not os.path.isfile(path):
                raise SystemExit(f"{name} JSON not found: {path}")
        compare_out = args.compare_output_dir
        if compare_out is None:
            compare_out = os.path.join(os.path.dirname(sz_norm_results), "comparison_sz11_vs_norm")
        run_sz11_vs_sz11_norm_comparison(
            sz_results_json=sz_results,
            sz_pkl_dir=os.path.abspath(args.sz_pkl_dir),
            sz_norm_results_json=sz_norm_results,
            sz_norm_pkl_dir=os.path.abspath(args.sz_norm_pkl_dir),
            table_dir=os.path.abspath(args.table_dir),
            scoli_root=args.scoli_root,
            output_dir=os.path.abspath(compare_out),
            skip_symmetry=args.skip_symmetry,
        )
        if args.show:
            plt.show()
        return

    if args.compare_pk_vs_pk_norm:
        pk_results = os.path.abspath(args.pk_results_json)
        pk_norm_results = os.path.abspath(args.pk_norm_results_json)
        for path, name in ((pk_results, "PK results"), (pk_norm_results, "PK norm results")):
            if not os.path.isfile(path):
                raise SystemExit(f"{name} JSON not found: {path}")
        compare_out = args.compare_output_dir
        if compare_out is None:
            compare_out = os.path.join(os.path.dirname(pk_norm_results), "comparison_pk_vs_norm")
        run_pk_vs_pk_norm_comparison(
            pk_results_json=pk_results,
            pk_pkl_dir=os.path.abspath(args.pk_pkl_dir),
            pk_norm_results_json=pk_norm_results,
            pk_norm_pkl_dir=os.path.abspath(args.pk_norm_pkl_dir),
            pk_table_dir=os.path.abspath(args.pk_table_dir),
            scoli_root=args.scoli_root,
            output_dir=os.path.abspath(compare_out),
            skip_symmetry=args.skip_symmetry,
        )
        if args.show:
            plt.show()
        return

    if args.compare_baseline_vs_posttrain:
        baseline_dir = args.baseline_inference_dir
        posttrain_dir = args.posttrain_inference_dir
        if not baseline_dir or not posttrain_dir:
            raise SystemExit(
                "--compare-baseline-vs-posttrain requires "
                "--baseline-inference-dir and --posttrain-inference-dir"
            )
        compare_out = args.compare_output_dir
        if compare_out is None:
            compare_out = os.path.join(os.path.dirname(os.path.abspath(baseline_dir)), "comparison_baseline_vs_posttrain")
        pkl_dir = os.path.abspath(args.pkl_dir) if args.pkl_dir else None
        if not pkl_dir:
            raise SystemExit("--compare-baseline-vs-posttrain requires --pkl-dir")
        run_baseline_vs_posttrain_comparison(
            baseline_inference_dir=os.path.abspath(baseline_dir),
            posttrain_inference_dir=os.path.abspath(posttrain_dir),
            pkl_dir=pkl_dir,
            table_dir=os.path.abspath(args.table_dir),
            scoli_root=args.scoli_root,
            output_dir=os.path.abspath(compare_out),
            skip_symmetry=args.skip_symmetry,
            threshold=args.threshold,
        )
        if args.show:
            plt.show()
        return

    if args.compare_sz_pos_vs_neg:
        sz_results = os.path.abspath(args.sz_results_json)
        if not os.path.isfile(sz_results):
            raise SystemExit(f"SZ results JSON not found: {sz_results}")
        compare_out = args.compare_output_dir
        if compare_out is None:
            compare_out = os.path.join(os.path.dirname(sz_results), "comparison_sz_pos_vs_neg")
        run_sz_pos_vs_neg_comparison(
            sz_results_json=sz_results,
            sz_pkl_dir=os.path.abspath(args.sz_pkl_dir),
            table_dir=os.path.abspath(args.table_dir),
            scoli_root=args.scoli_root,
            output_dir=os.path.abspath(compare_out),
            skip_symmetry=args.skip_symmetry,
        )
        if args.show:
            plt.show()
        return

    if args.compare_pk_vs_sz_neg:
        pk_results = os.path.abspath(args.pk_results_json)
        sz_results = os.path.abspath(args.sz_results_json)
        for path, name in ((pk_results, "PK results"), (sz_results, "SZ results")):
            if not os.path.isfile(path):
                raise SystemExit(f"{name} JSON not found: {path}")
        compare_out = args.compare_output_dir
        if compare_out is None:
            compare_out = os.path.join(os.path.dirname(pk_results), "comparison")
        run_pk_vs_sz_neg_comparison(
            pk_results_json=pk_results,
            pk_pkl_dir=os.path.abspath(args.pk_pkl_dir),
            pk_table_dir=os.path.abspath(args.pk_table_dir),
            sz_results_json=sz_results,
            sz_pkl_dir=os.path.abspath(args.sz_pkl_dir),
            sz_table_dir=os.path.abspath(args.table_dir),
            scoli_root=args.scoli_root,
            output_dir=os.path.abspath(compare_out),
            skip_symmetry=args.skip_symmetry,
        )
        if args.show:
            plt.show()
        return

    results_path = os.path.abspath(args.results_json)
    if not os.path.isfile(results_path):
        raise SystemExit(f"Results JSON not found: {results_path}")

    results_dir = os.path.dirname(results_path)
    pkl_dir = os.path.abspath(args.pkl_dir) if args.pkl_dir else None
    table_dir = os.path.abspath(args.table_dir)
    records = load_inference_results(results_path)

    symmetry_rows: List[Dict[str, Any]] = []
    if not args.skip_symmetry:
        sym_plot_dir = args.symmetry_plots_dir or os.path.join(results_dir, "symmetry_plots")
        os.makedirs(sym_plot_dir, exist_ok=True)
        print(f"Running gait symmetry analysis on CSVs in {table_dir}...")
        symmetry_rows = run_symmetry_eval(
            records,
            table_dir,
            args.scoli_root,
            symmetry_plot_dir=sym_plot_dir,
        )
        sym_json = os.path.join(results_dir, "symmetry_results.json")
        save_json(
            {"summary": _symmetry_cohort_summary(symmetry_rows), "per_sample": symmetry_rows},
            sym_json,
        )
        print(f"Saved: {sym_json}")
        _print_symmetry_summary(symmetry_rows)

    fig = build_analysis_figure(records, pkl_dir, symmetry_rows)
    out_path = args.output or os.path.join(results_dir, "results_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(f"  Inference samples: {len(records)}")

    fig_neg = build_negative_sample_values_figure(
        records,
        binary_cobb_threshold=args.binary_cobb_threshold,
    )
    neg_path = os.path.join(results_dir, "negative_sample_values.png")
    fig_neg.savefig(neg_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {neg_path}")
    n_neg = len(negative_records(records))
    print(f"  Negative samples: {n_neg}")
    if not args.show:
        plt.close(fig_neg)

    if symmetry_rows and not args.skip_symmetry:
        fig_sim = build_similarity_vs_label_figure(records, symmetry_rows)
        sim_path = os.path.join(results_dir, "similarity_vs_max_label.png")
        fig_sim.savefig(sim_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {sim_path}")
        if args.show:
            plt.figure(fig_sim.number)
        else:
            plt.close(fig_sim)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
