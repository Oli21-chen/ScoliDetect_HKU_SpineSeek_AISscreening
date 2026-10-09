# -*- coding: utf-8 -*-
"""
Step 3 — PKL batch inference + metrics.

Loads patches produced by step_02_Sampling and runs predict_from_arrays via
ScoliDetect inference_engine (same forward pass as predict_video).

conda run -n PytorchCuda11.8 python main\03_Infer.py --max-age 20 --threshold 0.5


"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _ensure_numpy2_pickle_compat() -> None:
    """Allow unpickling arrays saved with NumPy 2.x when running on NumPy 1.x."""
    if hasattr(np, "_core"):
        return
    core = np.core
    for name in ("_core", "_core.multiarray", "_core.numeric", "_core.umath"):
        alias = name.replace("_core", "core", 1)
        sys.modules.setdefault(f"numpy.{name}", sys.modules.get(f"numpy.{alias}", core))


_ensure_numpy2_pickle_compat()

DEFAULT_SCOLI_ROOT = Path(r"C:\Users\Olive\Desktop\ScoliDetect_deployversion")
DEFAULT_PKL_DIR = Path(r"C:\Users\Olive\Desktop\infer_data_11")
DEFAULT_LABEL_EXCEL = Path(
    r"C:\Users\Olive\Desktop\video_retrival\video_retrival\Label_SZpart2.xlsx"
)
# max_age >= this value is treated as no age filter (e.g. 100 = include all ages)
AGE_FILTER_OFF = 100.0


def _age_filter_active(max_age: Optional[float]) -> bool:
    """Return True when age < max_age filtering should be applied."""
    if max_age is None:
        return False
    return max_age < AGE_FILTER_OFF


def has_ground_truth(record: Dict[str, Any]) -> bool:
    """True when the patch has real Cobb/scalar labels (not fixed_label placeholder)."""
    return record.get("label_value") is not None

try:
    from sklearn.metrics import roc_auc_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _import_scoli_modules(scoli_root: str):
    """Import ScoliDetect inference helpers after sys.path is configured."""
    root = str(Path(scoli_root))
    if root not in sys.path:
        sys.path.insert(0, root)

    from utils.inference_engine import (
        default_checkpoint_path,
        load_model,
        predict_from_arrays,
    )
    from utils.prompts import default_prompts_path, load_gait_prompts_text

    return (
        default_checkpoint_path,
        load_model,
        predict_from_arrays,
        default_prompts_path,
        load_gait_prompts_text,
    )


def resolve_patch_path(meta_entry: Dict[str, Any], pkl_dir: str) -> Optional[str]:
    """Resolve patch PKL path; fall back to pkl_dir/patches/ if stored path is stale."""
    patch_path = meta_entry.get("pkl_path")
    if patch_path and os.path.isfile(patch_path):
        return patch_path

    patch_id = meta_entry.get("patch_id")
    if patch_id is not None:
        alt = os.path.join(pkl_dir, "patches", f"patch_{patch_id}.pkl")
        if os.path.isfile(alt):
            return alt

    if patch_path:
        alt = os.path.join(pkl_dir, "patches", os.path.basename(patch_path))
        if os.path.isfile(alt):
            return alt

    return None


def load_patch_metadata(pkl_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load patch_metadata.pkl from a step_02 output directory."""
    metadata_path = os.path.join(pkl_dir, "patch_metadata.pkl")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"patch_metadata.pkl not found: {metadata_path}")

    with open(metadata_path, "rb") as f:
        payload = pickle.load(f)

    patch_metadata = payload.get("patch_metadata", [])
    config = payload.get("config", {})
    if not patch_metadata:
        raise ValueError(f"No patches listed in {metadata_path}")
    return patch_metadata, config


def load_age_lookup(label_excel: str) -> Dict[int, float]:
    """Map File No. / subject_id -> age from label Excel."""
    import pandas as pd

    df = pd.read_excel(
        label_excel,
        usecols=["File No.", "age"],
    ).dropna(subset=["File No.", "age"])
    lookup: Dict[int, float] = {}
    for _, row in df.iterrows():
        lookup[int(row["File No."])] = float(row["age"])
    return lookup


def _subject_age(
    subject_id: Any,
    age_lookup: Optional[Dict[int, float]],
    meta_entry: Dict[str, Any],
    patch: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    if patch and patch.get("age") is not None:
        return float(patch["age"])
    if meta_entry.get("age") is not None:
        return float(meta_entry["age"])
    if age_lookup is None or subject_id is None:
        return None
    return age_lookup.get(int(subject_id))


def _validate_patch_shapes(patch: Dict[str, Any], meta_entry: Dict[str, Any]) -> None:
    km = patch.get("knowledge_map")
    video = patch.get("video")
    if km is None or video is None:
        return

    km_shape = getattr(km, "shape", None)
    video_shape = getattr(video, "shape", None)
    expected_km_t = meta_entry.get("knowledge_map_shape", (96, None))[0]
    expected_video = meta_entry.get("video_shape", (32, 224, 224, 3))

    if km_shape and km_shape[0] != expected_km_t:
        print(
            f"  Warning: {meta_entry.get('patch_id', '?')} km shape {km_shape} "
            f"(expected first dim {expected_km_t})"
        )
    if video_shape and tuple(video_shape) != tuple(expected_video):
        print(
            f"  Warning: {meta_entry.get('patch_id', '?')} video shape {video_shape} "
            f"(expected {expected_video})"
        )


def infer_one_patch(
    model,
    device,
    patch_path: str,
    meta_entry: Dict[str, Any],
    predict_from_arrays,
    *,
    prompts_text: str,
    threshold: float,
    age_lookup: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Run predict_from_arrays on one patch PKL."""
    with open(patch_path, "rb") as f:
        patch = pickle.load(f)

    _validate_patch_shapes(patch, meta_entry)

    probability, prediction = predict_from_arrays(
        model,
        patch["video"],
        patch["knowledge_map"],
        device,
        km_indices=patch.get("km_indices"),
        video_indices=patch.get("video_indices"),
        prompts_text=prompts_text,
        threshold=threshold,
    )

    label = patch.get("label")
    if label is None:
        label = meta_entry.get("label")

    label_value = patch.get("label_value")
    if label_value is None:
        label_value = meta_entry.get("label_value")

    subject_id = patch.get("subject_id") or meta_entry.get("subject_id")
    age = _subject_age(subject_id, age_lookup, meta_entry, patch)

    return {
        "patch_id": meta_entry.get("patch_id"),
        "pkl_path": patch_path,
        "source_file": patch.get("source_file") or meta_entry.get("source_file"),
        "subject_id": subject_id,
        "age": age,
        "probability": probability,
        "prediction": prediction,
        "label": label,
        "label_value": label_value,
        "has_ground_truth": has_ground_truth({"label_value": label_value}),
        "threshold": threshold,
        "peak_idx": patch.get("peak_idx", meta_entry.get("peak_idx")),
        "downsample_factor": patch.get("downsample_factor", meta_entry.get("downsample_factor")),
    }


def compute_screening_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summary stats for unlabeled screening runs (no ground-truth metrics)."""
    if not records:
        return {"n_screened": 0}
    probs = np.array([float(r["probability"]) for r in records], dtype=np.float64)
    preds = np.array([int(r["prediction"]) for r in records], dtype=np.int64)
    n_pos = int((preds == 1).sum())
    n_neg = int((preds == 0).sum())
    return {
        "n_screened": len(records),
        "predicted_positive": n_pos,
        "predicted_negative": n_neg,
        "positive_rate": n_pos / len(records),
        "mean_probability": float(probs.mean()),
        "median_probability": float(np.median(probs)),
        "min_probability": float(probs.min()),
        "max_probability": float(probs.max()),
    }


def compute_metrics(records: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """Compute aggregate binary classification metrics from inference records."""
    labeled = [r for r in records if has_ground_truth(r)]
    n_total = len(records)
    n_labeled = len(labeled)

    result: Dict[str, Any] = {
        "threshold": threshold,
        "n_total": n_total,
        "n_labeled": n_labeled,
        "n_unlabeled": n_total - n_labeled,
    }

    if n_labeled == 0:
        result["note"] = "No ground-truth labels (label_value); screening stats only."
        result["screening"] = compute_screening_stats(records)
        return result

    labels = np.array([float(r["label"]) for r in labeled], dtype=np.float64)
    probs = np.array([float(r["probability"]) for r in labeled], dtype=np.float64)
    preds = (probs >= threshold).astype(np.float64)

    tp = float(((preds == 1) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    total = tp + tn + fp + fn

    result.update(
        {
            "label_distribution": {
                "0.0": int((labels == 0).sum()),
                "1.0": int((labels == 1).sum()),
            },
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "confusion_matrix": {
                "labels": [0, 1],
                "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            },
            "accuracy": (tp + tn) / (total + 1e-8),
            "precision": tp / (tp + fp + 1e-8),
            "recall": tp / (tp + fn + 1e-8),
        }
    )
    prec = result["precision"]
    rec = result["recall"]
    result["f1"] = 2 * prec * rec / (prec + rec + 1e-8)

    unique_labels = np.unique(labels)
    if len(unique_labels) >= 2 and SKLEARN_AVAILABLE:
        try:
            result["auc_roc"] = float(roc_auc_score(labels, probs))
        except ValueError as exc:
            result["auc_roc_note"] = str(exc)
    elif len(unique_labels) < 2:
        result["auc_roc_note"] = "AUROC requires both classes in labeled set."
    else:
        result["auc_roc_note"] = "sklearn not installed; AUROC skipped."

    return result


def save_json(data: Any, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run_pkl_inference(
    pkl_dir: str,
    checkpoint_path: str,
    output_dir: str,
    *,
    scoli_root: str,
    prompts_path: Optional[str] = None,
    threshold: Optional[float] = None,
    device: Optional[str] = None,
    max_age: Optional[float] = None,
    label_excel: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str, str]:
    """Run inference on all patches under pkl_dir."""
    (
        _,
        load_model,
        predict_from_arrays,
        default_prompts_path,
        load_gait_prompts_text,
    ) = _import_scoli_modules(scoli_root)

    patch_metadata, pkl_config = load_patch_metadata(pkl_dir)

    age_lookup: Optional[Dict[int, float]] = None
    if _age_filter_active(max_age):
        if not label_excel or not os.path.isfile(label_excel):
            raise FileNotFoundError(
                f"--label-excel required for age filter (max_age={max_age}): {label_excel!r}"
            )
        age_lookup = load_age_lookup(label_excel)

    model, ckpt_config, dev = load_model(checkpoint_path, device=device)

    if threshold is None:
        threshold = float(ckpt_config.get("classification_prob_threshold", 0.5))

    resolved_prompts = prompts_path or ckpt_config.get("prompts_path") or default_prompts_path(scoli_root)
    prompts_text = load_gait_prompts_text(
        resolved_prompts,
        project_root=scoli_root,
        prompt_selection=ckpt_config.get("prompt_selection", "concise_prompts"),
    )

    records: List[Dict[str, Any]] = []
    n_age_skipped = 0
    print(f"Processing {len(patch_metadata)} patches from {pkl_dir}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Threshold:  {threshold}")
    print(f"  Device:     {dev}")
    if _age_filter_active(max_age):
        print(f"  Age filter: age < {max_age} (from {label_excel})")
    elif max_age is not None:
        print(f"  Age filter: off (max_age={max_age} >= {AGE_FILTER_OFF})")

    for meta_entry in patch_metadata:
        subject_id = meta_entry.get("subject_id")
        if _age_filter_active(max_age):
            age = _subject_age(subject_id, age_lookup, meta_entry)
            if age is None:
                n_age_skipped += 1
                continue
            if age >= max_age:
                n_age_skipped += 1
                continue

        patch_path = resolve_patch_path(meta_entry, pkl_dir)
        if not patch_path:
            stored = meta_entry.get("pkl_path")
            print(
                f"  skipped {meta_entry.get('patch_id', '?')}: PKL not found "
                f"({stored or 'no path'})"
            )
            continue

        try:
            record = infer_one_patch(
                model,
                dev,
                patch_path,
                meta_entry,
                predict_from_arrays,
                prompts_text=prompts_text,
                threshold=threshold,
                age_lookup=age_lookup,
            )
            records.append(record)
        except Exception as exc:
            print(f"  skipped {meta_entry.get('patch_id', '?')}: {exc}")
            continue

    metrics = compute_metrics(records, threshold)
    if _age_filter_active(max_age):
        metrics["age_filter"] = {"max_age_exclusive": max_age, "n_skipped_by_age": n_age_skipped}
        metrics["label_excel"] = label_excel

    results_path = os.path.join(output_dir, "inference_results.json")
    metrics_path = os.path.join(output_dir, "metrics.json")
    save_json(records, results_path)
    save_json(metrics, metrics_path)

    if _age_filter_active(max_age):
        print(f"  Age-skipped: {n_age_skipped} patches (age >= {max_age} or unknown)")

    return records, metrics, results_path, metrics_path


def print_confusion_matrix(metrics: Dict[str, Any]) -> None:
    """Print a 2x2 confusion matrix (rows=actual, cols=predicted)."""
    cm = metrics.get("confusion_matrix")
    if not cm:
        return

    matrix = cm["matrix"]
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    col_w = max(len(str(v)) for row in matrix for v in row)
    col_w = max(col_w, 6)

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"{'':14}{'Pred 0':>{col_w}}{'Pred 1':>{col_w + 2}}")
    print(f"{'Actual 0':14}{tn:>{col_w}}{fp:>{col_w + 2}}")
    print(f"{'Actual 1':14}{fn:>{col_w}}{tp:>{col_w + 2}}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")


def _print_summary(
    records: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    results_path: str,
    metrics_path: str,
) -> None:
    print(f"\nProcessed {len(records)} patches")
    print(f"  Results: {results_path}")
    print(f"  Metrics: {metrics_path}")
    if metrics.get("n_labeled", 0) > 0:
        print(f"  Labeled: {metrics['n_labeled']} / {metrics['n_total']}")
        print_confusion_matrix(metrics)
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  F1:        {metrics.get('f1', 0):.4f}")
        if "auc_roc" in metrics:
            print(f"  AUROC:     {metrics['auc_roc']:.4f}")
        elif metrics.get("auc_roc_note"):
            print(f"  AUROC:     ({metrics['auc_roc_note']})")
    else:
        print("  No ground-truth labels; screening-only summary:")
        screening = metrics.get("screening", {})
        if screening:
            n = screening.get("n_screened", 0)
            n_pos = screening.get("predicted_positive", 0)
            n_neg = screening.get("predicted_negative", 0)
            print(f"    Screened:           {n}")
            print(f"    Predicted positive: {n_pos} ({100 * screening.get('positive_rate', 0):.1f}%)")
            print(f"    Predicted negative: {n_neg}")
            print(f"    Mean probability:   {screening.get('mean_probability', 0):.4f}")
            print(f"    Median probability: {screening.get('median_probability', 0):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SFTRegressor inference on step_02 PKL patches and compute metrics."
    )
    parser.add_argument(
        "--pkl-dir",
        default=str(DEFAULT_PKL_DIR),
        help="Directory containing patch_metadata.pkl and patches/ (step_02 output)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="SFTRegressor checkpoint (.pth); default: ScoliDetect default_checkpoint_path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for JSON results (default: {pkl_dir}/inference)",
    )
    parser.add_argument(
        "--scoli-root",
        default=str(DEFAULT_SCOLI_ROOT),
        help="ScoliDetect_deployversion root for imports, checkpoints, prompts",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Gait prompts JSON (default: checkpoints/general_gait_prompts_from_report.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification probability cutoff (default: from checkpoint config)",
    )
    parser.add_argument("--device", default=None, help="Torch device: cuda, cuda:0, or cpu")
    parser.add_argument(
        "--max-age",
        type=float,
        default=None,
        help=(
            "Only include subjects with age < this value (requires --label-excel). "
            f"Values >= {AGE_FILTER_OFF:.0f} disable age filtering."
        ),
    )
    parser.add_argument(
        "--label-excel",
        default=None,
        help=(
            "Excel with File No. and age columns; required only when --max-age is active. "
            f"Default SZ path: {DEFAULT_LABEL_EXCEL}"
        ),
    )
    args = parser.parse_args()

    scoli_root = str(Path(args.scoli_root))
    (
        default_checkpoint_path,
        _,
        _,
        default_prompts_path_fn,
        _,
    ) = _import_scoli_modules(scoli_root)

    pkl_dir = os.path.abspath(args.pkl_dir)
    if not os.path.isdir(pkl_dir):
        raise SystemExit(f"PKL directory not found: {pkl_dir}")

    checkpoint = args.checkpoint or default_checkpoint_path(scoli_root)
    if not checkpoint or not os.path.isfile(checkpoint):
        raise SystemExit(f"Checkpoint not found: {checkpoint!r}")

    prompts = args.prompts or default_prompts_path_fn(scoli_root)
    if not os.path.isfile(prompts):
        raise SystemExit(f"Prompts JSON not found: {prompts}")

    output_dir = args.output_dir
    max_age = args.max_age
    if output_dir is None:
        if _age_filter_active(max_age):
            output_dir = os.path.join(pkl_dir, f"inference_age_under{int(max_age)}")
        else:
            output_dir = os.path.join(pkl_dir, "inference")

    label_excel = args.label_excel
    if label_excel is None and _age_filter_active(max_age):
        label_excel = str(DEFAULT_LABEL_EXCEL)

    records, metrics, results_path, metrics_path = run_pkl_inference(
        pkl_dir,
        checkpoint,
        output_dir,
        scoli_root=scoli_root,
        prompts_path=prompts,
        threshold=args.threshold,
        device=args.device,
        max_age=max_age,
        label_excel=label_excel,
    )
    _print_summary(records, metrics, results_path, metrics_path)


if __name__ == "__main__":
    main()
