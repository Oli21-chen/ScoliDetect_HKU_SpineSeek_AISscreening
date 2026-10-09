# -*- coding: utf-8 -*-
"""
Visualize per-modality inference gradients for 3 holdout examples.

Compares baseline deploy vs post-trained 3-shot and 5-shot on identical patches.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

PYTORCH_ROOT = Path(__file__).resolve().parents[1]
CODE_VIDEO_ROOT = PYTORCH_ROOT.parent
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))

from post_training.config import (  # noqa: E402
    BASE_CHECKPOINT,
    POST_TRAINING_ROOT,
    SHOT_RUNS_DIR,
    resolve_prompt_selection,
    resolve_prompts_path,
)
from post_training.model_io import load_deploy_checkpoint_model  # noqa: E402
from post_training.modality_gradients import (  # noqa: E402
    ModalityGradResult,
    checkpoint_fingerprint,
    compute_modality_gradients,
    example_filename,
    load_inference_records,
    load_single_patch_batch,
    plot_example_figure,
    select_example_records,
)
from utils.sft_utils import resolve_device  # noqa: E402

DEFAULT_HOLDOUT_PKL = SHOT_RUNS_DIR / "holdout_pkl"
DEFAULT_INFERENCE_JSON = SHOT_RUNS_DIR / "shot3" / "inference" / "inference_results.json"
DEFAULT_POSTTRAIN_3SHOT = SHOT_RUNS_DIR / "shot3" / "checkpoint_posttrain_head.pth"
DEFAULT_POSTTRAIN_5SHOT = SHOT_RUNS_DIR / "shot5" / "checkpoint_posttrain_head.pth"
DEFAULT_OUTPUT_DIR = POST_TRAINING_ROOT / "gradient_viz"
MANIFEST_JSON = "manifest.json"
REPORT_MD = "MODALITY_GRADIENT_REPORT.md"

MODEL_SPECS = (
    ("baseline", "Baseline"),
    ("posttrain_3shot", "Post-train 3-shot"),
    ("posttrain_5shot", "Post-train 5-shot"),
)


def _fmt_pct_map(pcts: Dict[str, float]) -> str:
    return ", ".join(f"{k}={pcts[k]:.1f}%" for k in ("video", "km", "text"))


def _shift_dict(
    reference: Dict[str, float],
    target: Dict[str, float],
) -> Dict[str, float]:
    return {k: target[k] - reference[k] for k in reference}


def write_report_md(manifest: Dict[str, Any], out_path: Path) -> None:
    lines: List[str] = [
        "# Modality Gradient Visualization Report",
        "",
        f"Generated: {manifest.get('created_at', '')}",
        "",
        "## Summary",
        "",
        "Per-modality gradient attribution during inference (backward from positive-class logit).",
        "Each figure compares **baseline deploy**, **post-trained 3-shot**, and **post-trained 5-shot** "
        "on the same holdout patch.",
        "",
        "**Checkpoint verification:** paths and MD5 hashes are embedded in each figure column and stored "
        "in `checkpoint_fingerprints` below. Encoders are identical across models; only fusion head "
        "weights differ. Input-level heatmaps therefore look nearly the same; use row 2 (logit/prob) "
        "and row 4 (ΔKM |grad| vs baseline) to see post-training effects.",
        "",
        f"- **Baseline checkpoint:** `{manifest.get('baseline_checkpoint')}`",
        f"- **Post-train 3-shot:** `{manifest.get('posttrain_3shot_checkpoint')}`",
        f"- **Post-train 5-shot:** `{manifest.get('posttrain_5shot_checkpoint')}`",
        f"- **Holdout PKL dir:** `{manifest.get('pkl_dir')}`",
        "",
        "## Examples",
        "",
    ]

    for ex in manifest.get("examples", []):
        lines.extend(
            [
                f"### {ex.get('title')}",
                "",
                f"- **Case:** {ex.get('case_tag')} | **Subject:** {ex.get('subject_id')} | "
                f"**Label:** {ex.get('label')} | **Figure:** `{ex.get('figure')}`",
                "",
                "| Model | Probability | Logit | Video % | KM % | Text % |",
                "|-------|-------------|-------|---------|------|--------|",
            ]
        )
        for model_key, label in MODEL_SPECS:
            m = ex.get(model_key, {})
            pcts = m.get("token_pcts", {})
            lines.append(
                f"| {label} | {m.get('probability', 0):.4f} | {m.get('logit', 0):.4f} | "
                f"{pcts.get('video', 0):.1f} | {pcts.get('km', 0):.1f} | {pcts.get('text', 0):.1f} |"
            )
        shift3 = ex.get("modality_shift_3shot_vs_baseline", {})
        shift5 = ex.get("modality_shift_5shot_vs_baseline", {})
        lines.extend(
            [
                "",
                f"**Modality mass shift (3-shot − baseline):** "
                f"video {shift3.get('video', 0):+.1f}pp, "
                f"km {shift3.get('km', 0):+.1f}pp, "
                f"text {shift3.get('text', 0):+.1f}pp",
                "",
                f"**Modality mass shift (5-shot − baseline):** "
                f"video {shift5.get('video', 0):+.1f}pp, "
                f"km {shift5.get('km', 0):+.1f}pp, "
                f"text {shift5.get('text', 0):+.1f}pp",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation guide",
            "",
            "- **Modality gradient mass (%):** L2 norm of ∂logit/∂(modality token), normalized to sum to 100%.",
            "- **Video temporal |grad|:** mean absolute input gradient over spatial dims per frame.",
            "- **KM heatmap:** |∂logit/∂knowledge_map| with motion (0–34), skeleton (34–172), signal (172–238) bands.",
            "- **Text top-k:** largest |grad| dimensions on the fused text embedding.",
            "",
            "## Artifacts",
            "",
            f"- Figures: `{manifest.get('output_dir')}/*.png`",
            f"- Manifest: `{manifest.get('output_dir')}/{MANIFEST_JSON}`",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_gradients_for_checkpoint(
    checkpoint: Path,
    batch: Dict[str, Any],
    device: torch.device,
) -> ModalityGradResult:
    model, _, _ = load_deploy_checkpoint_model(checkpoint, device)
    result = compute_modality_gradients(
        model, batch, device, checkpoint_path=checkpoint
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Modality gradient visualization (3 examples)")
    parser.add_argument("--pkl-dir", type=Path, default=DEFAULT_HOLDOUT_PKL)
    parser.add_argument("--inference-json", type=Path, default=DEFAULT_INFERENCE_JSON)
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--posttrain-3shot-checkpoint", type=Path, default=DEFAULT_POSTTRAIN_3SHOT)
    parser.add_argument("--posttrain-5shot-checkpoint", type=Path, default=DEFAULT_POSTTRAIN_5SHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Override auto TP/TN/error pick")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_inference_records(args.inference_json)
    examples = select_example_records(records, subject_ids=args.subjects)

    if args.dry_run:
        print("Selected examples:")
        for ex in examples:
            rec = ex.record
            print(
                f"  {ex.key} {ex.case_tag}: subject={ex.subject_id} "
                f"source={ex.source_file} label={rec.get('label')} "
                f"prob={float(rec.get('probability', 0)):.4f} pred={rec.get('prediction')}"
            )
        return

    device = resolve_device(None)
    print(f"Device: {device}")

    _, ckpt_config, _ = load_deploy_checkpoint_model(args.baseline_checkpoint, device)
    prompts_path = resolve_prompts_path(ckpt_config)
    prompt_selection = resolve_prompt_selection(ckpt_config)
    print(f"Prompts: {prompts_path} ({prompt_selection})")

    checkpoint_map = {
        "baseline": args.baseline_checkpoint,
        "posttrain_3shot": args.posttrain_3shot_checkpoint,
        "posttrain_5shot": args.posttrain_5shot_checkpoint,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_examples: List[Dict[str, Any]] = []

    for example in examples:
        pkl_path = Path(example.pkl_path)
        print(f"\nProcessing {example.case_tag} subject {example.subject_id} ({example.source_file})")
        batch = load_single_patch_batch(
            pkl_dir=args.pkl_dir,
            pkl_path=pkl_path,
            prompts_path=prompts_path,
            prompt_selection=prompt_selection,
        )

        results: Dict[str, ModalityGradResult] = {}
        for key, path in checkpoint_map.items():
            print(f"  Running {key}...")
            results[key] = run_gradients_for_checkpoint(path, batch, device)

        model_results: List[Tuple[str, ModalityGradResult]] = [
            (label, results[key]) for key, label in MODEL_SPECS
        ]
        fig_name = example_filename(example)
        fig_path = args.output_dir / fig_name
        plot_example_figure(example, model_results, fig_path)
        print(f"Saved figure: {fig_path}")

        baseline_pcts = results["baseline"].token_pcts
        manifest_examples.append(
            {
                "title": f"Example {example.key} ({example.case_tag})",
                "case_tag": example.case_tag,
                "subject_id": example.subject_id,
                "source_file": example.source_file,
                "label": float(example.record.get("label", 0)),
                "figure": fig_name,
                "pkl_path": str(pkl_path),
                "baseline": results["baseline"].to_dict(),
                "posttrain_3shot": results["posttrain_3shot"].to_dict(),
                "posttrain_5shot": results["posttrain_5shot"].to_dict(),
                "modality_shift_3shot_vs_baseline": _shift_dict(
                    baseline_pcts, results["posttrain_3shot"].token_pcts
                ),
                "modality_shift_5shot_vs_baseline": _shift_dict(
                    baseline_pcts, results["posttrain_5shot"].token_pcts
                ),
                "modality_shift_5shot_vs_3shot": _shift_dict(
                    results["posttrain_3shot"].token_pcts,
                    results["posttrain_5shot"].token_pcts,
                ),
            }
        )

    manifest: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pkl_dir": str(args.pkl_dir),
        "inference_json": str(args.inference_json),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "posttrain_3shot_checkpoint": str(args.posttrain_3shot_checkpoint),
        "posttrain_5shot_checkpoint": str(args.posttrain_5shot_checkpoint),
        "checkpoint_fingerprints": {
            key: checkpoint_fingerprint(path) for key, path in checkpoint_map.items()
        },
        "output_dir": str(args.output_dir),
        "examples": manifest_examples,
    }
    manifest_path = args.output_dir / MANIFEST_JSON
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nSaved manifest: {manifest_path}")

    report_path = args.output_dir / REPORT_MD
    write_report_md(manifest, report_path)
    print(f"Saved report: {report_path}")

    print("\nModality gradient mass:")
    for ex in manifest_examples:
        print(f"  {ex['case_tag']} sz_{ex['subject_id']}:")
        for key, label in MODEL_SPECS:
            pcts = ex[key]["token_pcts"]
            print(f"    {label}: ({_fmt_pct_map(pcts)})")


if __name__ == "__main__":
    main()
