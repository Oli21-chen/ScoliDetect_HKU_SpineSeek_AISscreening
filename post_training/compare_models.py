# -*- coding: utf-8 -*-
"""
Compare baseline vs post-trained inference on the same holdout patches.

Can be used standalone or via 04_results_analyze --compare-baseline-vs-posttrain.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

CODE_VIDEO_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_VIDEO_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_VIDEO_ROOT))


def _load_analyze_module():
    import importlib.util

    analyze_path = CODE_VIDEO_ROOT / "main" / "04_results_analyze.py"
    spec = importlib.util.spec_from_file_location("results_analyze", analyze_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {analyze_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze = _load_analyze_module()


COLOR_BASELINE = "#4C72B0"
COLOR_POSTTRAIN = "#DD8452"


def load_metrics(inference_dir: str) -> Dict[str, Any]:
    path = os.path.join(inference_dir, "metrics.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pair_records_by_source(
    baseline_records: List[Dict[str, Any]],
    posttrain_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    post_by_source = {r.get("source_file"): r for r in posttrain_records}
    paired = []
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


def build_probability_delta_figure(
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
    ax.hist(base_probs, bins=bins, alpha=0.55, color=COLOR_BASELINE, label=f"Baseline (n={len(base_probs)})")
    ax.hist(post_probs, bins=bins, alpha=0.55, color=COLOR_POSTTRAIN, label=f"Post-trained (n={len(post_probs)})")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Holdout probability distributions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(base_probs, post_probs, alpha=0.6, s=28, c=COLOR_POSTTRAIN, edgecolors="white", linewidths=0.4)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0)
    ax.axhline(threshold, color="black", linestyle=":", linewidth=0.8)
    ax.axvline(threshold, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Baseline probability")
    ax.set_ylabel("Post-trained probability")
    ax.set_title("Per-patch paired probabilities")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    delta_bins = np.linspace(-1.0, 1.0, 41)
    ax.hist(deltas, bins=delta_bins, color=COLOR_POSTTRAIN, alpha=0.75, edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Post-trained - baseline probability")
    ax.set_ylabel("Count")
    ax.set_title("Probability delta")
    ax.grid(True, alpha=0.3)

    pval = comparison_stats.get("statistical_tests", {}).get("probability_delta")
    if pval is not None:
        fig.suptitle(f"Baseline vs post-trained (holdout n={len(paired)}, delta p={pval:.4g})", fontsize=12)
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
    skip_symmetry: bool = False,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Delegate to 04_results_analyze for a single implementation."""
    return analyze.run_baseline_vs_posttrain_comparison(
        baseline_inference_dir=baseline_inference_dir,
        posttrain_inference_dir=posttrain_inference_dir,
        pkl_dir=pkl_dir,
        table_dir=table_dir,
        scoli_root=scoli_root,
        output_dir=output_dir,
        skip_symmetry=skip_symmetry,
        threshold=threshold,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-inference-dir", required=True)
    parser.add_argument("--posttrain-inference-dir", required=True)
    parser.add_argument("--pkl-dir", required=True)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--scoli-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-symmetry", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    cli_args = parser.parse_args()
    run_baseline_vs_posttrain_comparison(
        baseline_inference_dir=cli_args.baseline_inference_dir,
        posttrain_inference_dir=cli_args.posttrain_inference_dir,
        pkl_dir=cli_args.pkl_dir,
        table_dir=cli_args.table_dir,
        scoli_root=cli_args.scoli_root,
        output_dir=cli_args.output_dir,
        skip_symmetry=cli_args.skip_symmetry,
        threshold=cli_args.threshold,
    )
