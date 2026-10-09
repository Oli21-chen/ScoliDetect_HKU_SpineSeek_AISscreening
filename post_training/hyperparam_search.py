# -*- coding: utf-8 -*-
"""
Hyperparameter search for head-only post-training.

Trains each variant on support_pkl, evaluates on holdout_pkl, ranks by holdout AUROC.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PYTORCH_ROOT = Path(__file__).resolve().parents[1]
CODE_VIDEO_ROOT = PYTORCH_ROOT.parent
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))

from post_training.config import (  # noqa: E402
    BASE_CHECKPOINT,
    HOLDOUT_PKL_DIR,
    HP_RUNS_DIR,
    INFER_THRESHOLD,
    LABEL_EXCEL,
    MAX_AGE,
    SCOLI_ROOT,
    STEP_03,
    SUPPORT_PKL_DIR,
    python_cmd,
)
from post_training.head_finetune import train_head  # noqa: E402
from post_training.hyperparam_grid import HpVariant, iter_variants  # noqa: E402

BASELINE_METRICS_PATH = HOLDOUT_PKL_DIR / "inference_baseline" / "metrics.json"
LEADERBOARD_JSON = HP_RUNS_DIR / "leaderboard.json"
LEADERBOARD_CSV = HP_RUNS_DIR / "leaderboard.csv"
REPORT_MD = HP_RUNS_DIR / "HP_SEARCH_REPORT.md"
REJECT_AUROC_MARGIN = 0.01


def run_cmd(cmd: List[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(CODE_VIDEO_ROOT))


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_baseline_metrics() -> Dict[str, Any]:
    if not BASELINE_METRICS_PATH.is_file():
        raise FileNotFoundError(
            f"Baseline metrics not found: {BASELINE_METRICS_PATH}\n"
            "Run: python pytorch/post_training/run_eval_pipeline.py "
            "--sample-holdout --infer --skip-finetune --skip-symmetry"
        )
    return load_json(BASELINE_METRICS_PATH)


def run_holdout_inference(checkpoint: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = python_cmd(
        [
            str(STEP_03),
            "--pkl-dir",
            str(HOLDOUT_PKL_DIR),
            "--output-dir",
            str(output_dir),
            "--scoli-root",
            str(SCOLI_ROOT),
            "--threshold",
            str(INFER_THRESHOLD),
            "--max-age",
            str(MAX_AGE),
            "--label-excel",
            str(LABEL_EXCEL),
            "--checkpoint",
            str(checkpoint),
        ]
    )
    run_cmd(cmd)
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Inference metrics missing: {metrics_path}")
    return load_json(metrics_path)


def mean_prob_delta(
    baseline_results: Path,
    variant_results: Path,
) -> Optional[float]:
    base_path = baseline_results
    var_path = variant_results
    if not base_path.is_file() or not var_path.is_file():
        return None
    base_recs = load_json(base_path) if base_path.suffix == ".json" else None
    if base_recs is None:
        return None
    # inference_results.json is a list
    if isinstance(base_recs, list):
        base_by_src = {r.get("source_file"): float(r["probability"]) for r in base_recs}
        var_recs = load_json(var_path)
        if not isinstance(var_recs, list):
            return None
        deltas = []
        for rec in var_recs:
            src = rec.get("source_file")
            if src in base_by_src:
                deltas.append(float(rec["probability"]) - base_by_src[src])
        return float(sum(deltas) / len(deltas)) if deltas else None
    return None


def run_variant(
    variant: HpVariant,
    *,
    skip_train: bool = False,
    baseline_auc: float,
) -> Dict[str, Any]:
    run_dir = HP_RUNS_DIR / variant.name
    checkpoint = run_dir / "checkpoint_posttrain_head.pth"
    manifest_path = run_dir / "manifest.json"
    inference_dir = run_dir / "inference"

    result: Dict[str, Any] = {
        "name": variant.name,
        "hyperparameters": variant.as_dict(),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "status": "started",
    }

    try:
        if not skip_train:
            print(f"\n{'=' * 60}\nVariant: {variant.name}\n{'=' * 60}")
            train_result = train_head(
                support_pkl_dir=SUPPORT_PKL_DIR,
                base_checkpoint=BASE_CHECKPOINT,
                output_checkpoint=checkpoint,
                cfg=variant.to_finetune_config(),
            )
            manifest = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "variant": variant.name,
                "hyperparameters": variant.as_dict(),
                "training": train_result,
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            result["training"] = {
                "best_epoch": train_result.get("best_epoch"),
                "best_loss": train_result.get("best_loss"),
            }
        elif not checkpoint.is_file():
            result["status"] = "skipped"
            result["error"] = f"Checkpoint not found: {checkpoint}"
            return result

        holdout_metrics = run_holdout_inference(checkpoint, inference_dir)
        result["holdout_metrics"] = holdout_metrics
        result["holdout_auc_roc"] = holdout_metrics.get("auc_roc")
        result["holdout_accuracy"] = holdout_metrics.get("accuracy")
        result["holdout_f1"] = holdout_metrics.get("f1")
        result["baseline_auc_roc"] = baseline_auc
        result["delta_auc_roc"] = (
            float(holdout_metrics["auc_roc"]) - baseline_auc
            if holdout_metrics.get("auc_roc") is not None
            else None
        )
        result["rejected"] = (
            result["holdout_auc_roc"] is not None
            and result["holdout_auc_roc"] < baseline_auc - REJECT_AUROC_MARGIN
        )

        delta_prob = mean_prob_delta(
            HOLDOUT_PKL_DIR / "inference_baseline" / "inference_results.json",
            inference_dir / "inference_results.json",
        )
        result["mean_prob_delta"] = delta_prob
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        print(f"Variant {variant.name} failed: {exc}")

    return result


def load_previous_results(phase: str) -> List[Dict[str, Any]]:
    """Load prior sweep results when merging phase 2 into the combined leaderboard."""
    if phase != "phase2" or not LEADERBOARD_JSON.is_file():
        return []
    payload = load_json(LEADERBOARD_JSON)
    return list(payload.get("all_results", []))


def merge_results(
    previous: List[Dict[str, Any]],
    new_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_name = {r["name"]: r for r in previous}
    for row in new_results:
        by_name[row["name"]] = row
    return list(by_name.values())


def rank_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    completed = [r for r in results if r.get("status") == "completed" and not r.get("rejected")]

    def sort_key(r: Dict[str, Any]):
        auc = r.get("holdout_auc_roc") or -1.0
        acc = r.get("holdout_accuracy") or -1.0
        prob_delta = abs(r.get("mean_prob_delta") or 0.0)
        return (-auc, -acc, prob_delta)

    ranked = sorted(completed, key=sort_key)
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def write_leaderboard(
    results: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    phase: str,
    *,
    phases_run: Optional[List[str]] = None,
    n_new_variants: Optional[int] = None,
) -> None:
    HP_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "phases_run": phases_run or [phase],
        "baseline": baseline,
        "n_variants": len(results),
        "n_new_variants": n_new_variants if n_new_variants is not None else len(results),
        "n_completed": sum(1 for r in results if r.get("status") == "completed"),
        "n_failed": sum(1 for r in results if r.get("status") == "failed"),
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
        "all_results": results,
    }
    with open(LEADERBOARD_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved leaderboard: {LEADERBOARD_JSON}")

    if ranked:
        fieldnames = [
            "rank",
            "name",
            "lr",
            "epochs",
            "weight_decay",
            "loss_type",
            "km_gaussian_noise_std",
            "holdout_auc_roc",
            "holdout_accuracy",
            "holdout_f1",
            "delta_auc_roc",
            "mean_prob_delta",
            "best_loss",
            "best_epoch",
        ]
        with open(LEADERBOARD_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in ranked:
                hp = row.get("hyperparameters", {})
                training = row.get("training", {})
                writer.writerow(
                    {
                        "rank": row.get("rank"),
                        "name": row.get("name"),
                        "lr": hp.get("lr"),
                        "epochs": hp.get("epochs"),
                        "weight_decay": hp.get("weight_decay"),
                        "loss_type": hp.get("loss_type"),
                        "km_gaussian_noise_std": hp.get("km_gaussian_noise_std"),
                        "holdout_auc_roc": row.get("holdout_auc_roc"),
                        "holdout_accuracy": row.get("holdout_accuracy"),
                        "holdout_f1": row.get("holdout_f1"),
                        "delta_auc_roc": row.get("delta_auc_roc"),
                        "mean_prob_delta": row.get("mean_prob_delta"),
                        "best_loss": training.get("best_loss"),
                        "best_epoch": training.get("best_epoch"),
                    }
                )
        print(f"Saved leaderboard CSV: {LEADERBOARD_CSV}")


def _fmt_float(val: Any, digits: int = 4) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return str(val)


def _status_label(row: Dict[str, Any]) -> str:
    if row.get("status") == "failed":
        return "failed"
    if row.get("rejected"):
        return "rejected"
    if row.get("status") == "completed":
        return "ok"
    return str(row.get("status", "unknown"))


def write_report_md(payload: Dict[str, Any]) -> Path:
    """Write markdown summary from leaderboard payload."""
    HP_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    baseline = payload.get("baseline", {})
    baseline_auc = float(baseline.get("auc_roc", 0))
    ranked: List[Dict[str, Any]] = payload.get("ranked", [])
    all_results: List[Dict[str, Any]] = payload.get("all_results", [])
    best = payload.get("best")
    phase = payload.get("phase", "small")
    phases_run = payload.get("phases_run", [phase])
    created = payload.get("created_at", "")
    n_new = payload.get("n_new_variants", payload.get("n_variants", len(all_results)))

    lines: List[str] = [
        "# Head-Only Post-Training Hyperparameter Search Report",
        "",
        f"Generated: {created}",
        f"Phase: `{phase}`",
        f"Phases included: {', '.join(f'`{p}`' for p in phases_run)}",
        "",
        "## Executive Summary",
        "",
        f"- **Baseline holdout AUROC:** {_fmt_float(baseline_auc)}",
        f"- **Total variants in leaderboard:** {payload.get('n_variants', len(all_results))}",
    ]
    if phase == "combined" or len(phases_run) > 1:
        lines.append(f"- **New variants this run:** {n_new}")
    lines.extend(
        [
            f"- **Completed:** {payload.get('n_completed', 0)} | **Failed:** {payload.get('n_failed', 0)}",
        ]
    )

    if best:
        hp = best.get("hyperparameters", {})
        best_auc = best.get("holdout_auc_roc")
        delta = best.get("delta_auc_roc")
        lines.extend(
            [
                f"- **Best variant:** `{best.get('name')}` "
                f"(lr={hp.get('lr')}, epochs={hp.get('epochs')}, "
                f"wd={hp.get('weight_decay')}, loss={hp.get('loss_type')}, "
                f"noise={hp.get('km_gaussian_noise_std')})",
                f"- **Best holdout AUROC:** {_fmt_float(best_auc)} (delta {_fmt_float(delta, 4) if delta is not None else 'N/A'})",
                f"- **Best accuracy / F1:** {_fmt_float(best.get('holdout_accuracy'))} / {_fmt_float(best.get('holdout_f1'))}",
            ]
        )
        if delta is not None and float(delta) > 0:
            lines.append(
                "- **Recommendation:** Use post-trained checkpoint with winning hyperparameters "
                "(holdout AUROC improved vs deploy baseline)."
            )
        else:
            lines.append(
                "- **Recommendation:** Keep deploy baseline checkpoint; no variant clearly beat baseline on holdout AUROC."
            )
    else:
        lines.append("- **Recommendation:** No eligible variant; keep deploy baseline.")

    lines.extend(
        [
            "",
            "## Experimental Setup",
            "",
            "- **Support:** subjects 884 (pos) + 885 (neg), head-only fine-tune",
            "- **Holdout:** remaining SZ subjects (93 patches)",
            "- **Selection metric:** holdout AUROC (reject if < baseline - 0.01)",
            "- **Phase 1 (small):** lr in [1e-8, 1e-7, 1e-6, 1e-5]; epochs in [5, 10, 20]",
            "- **Phase 2:** refined LR, weight_decay, loss_type (bce/focal), km_gaussian_noise_std",
            "- **Default anchor:** lr=1e-7, epochs=10, weight_decay=1e-4, loss=bce, km_noise=0.01",
            "",
            "## Results (ranked)",
            "",
            "| Rank | Variant | LR | Ep | WD | Loss | Noise | AUROC | Δ AUROC | Acc | F1 | Train loss | Status |",
            "|------|---------|-----|-----|-----|------|-------|-------|---------|-----|-----|------------|--------|",
        ]
    )

    rank_by_name = {r.get("name"): r.get("rank") for r in ranked}
    display_rows = sorted(
        all_results,
        key=lambda r: (rank_by_name.get(r.get("name"), 999), r.get("name", "")),
    )
    for row in display_rows:
        hp = row.get("hyperparameters", {})
        training = row.get("training", {})
        rank = rank_by_name.get(row.get("name"), "-")
        lines.append(
            "| {rank} | `{name}` | {lr} | {ep} | {wd} | {loss} | {noise} | {auc} | {dauc} | {acc} | {f1} | {loss_t} | {status} |".format(
                rank=rank,
                name=row.get("name", ""),
                lr=hp.get("lr", ""),
                ep=hp.get("epochs", ""),
                wd=hp.get("weight_decay", ""),
                loss=hp.get("loss_type", ""),
                noise=hp.get("km_gaussian_noise_std", ""),
                auc=_fmt_float(row.get("holdout_auc_roc")),
                dauc=_fmt_float(row.get("delta_auc_roc"), 4),
                acc=_fmt_float(row.get("holdout_accuracy")),
                f1=_fmt_float(row.get("holdout_f1")),
                loss_t=_fmt_float(training.get("best_loss"), 6),
                status=_status_label(row),
            )
        )

    failed_or_rejected = [
        r for r in all_results if r.get("status") == "failed" or r.get("rejected")
    ]
    lines.extend(["", "## Failed / Rejected Runs", ""])
    if failed_or_rejected:
        for row in failed_or_rejected:
            err = row.get("error", "below rejection threshold" if row.get("rejected") else "")
            lines.append(f"- `{row.get('name')}`: {row.get('status')} — {err}")
    else:
        lines.append("None.")

    lines.extend(["", "## Best Configuration (copy to config.py)", ""])
    if best:
        hp = best.get("hyperparameters", {})
        lines.extend(
            [
                "```python",
                "@dataclass(frozen=True)",
                "class HeadFinetuneConfig:",
                f"    epochs: int = {int(hp.get('epochs', 10))}",
                f"    lr: float = {hp.get('lr')}",
                f"    batch_size: int = {hp.get('batch_size', 1)}",
                f"    weight_decay: float = {hp.get('weight_decay', 1e-4)}",
                f"    km_gaussian_noise_std: float = {hp.get('km_gaussian_noise_std', 0.01)}",
                f"    loss_type: str = \"{hp.get('loss_type', 'bce')}\"",
                "    # ...",
                "```",
                "",
                f"Checkpoint: `{best.get('checkpoint', '')}`",
            ]
        )
    else:
        lines.append("No winning configuration.")

    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "1. Copy winning hyperparameters into `pytorch/post_training/config.py`",
            "2. Re-run full evaluation:",
            "   ```powershell",
            "   python pytorch/post_training/run_eval_pipeline.py --finetune --infer --analyze --compare --skip-symmetry",
            "   ```",
            "3. Phase 2 merges into the combined leaderboard; re-run with `--write-report-only` after manual edits",
            "",
            "## Artifacts",
            "",
            f"- Leaderboard JSON: `{LEADERBOARD_JSON}`",
            f"- Leaderboard CSV: `{LEADERBOARD_CSV}`",
            f"- Per-variant dirs: `{HP_RUNS_DIR}/<variant_name>/`",
        ]
    )

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {REPORT_MD}")
    return REPORT_MD


def regenerate_report_from_leaderboard() -> Path:
    if not LEADERBOARD_JSON.is_file():
        raise FileNotFoundError(f"Leaderboard not found: {LEADERBOARD_JSON}")
    payload = load_json(LEADERBOARD_JSON)
    return write_report_md(payload)


def print_summary(ranked: List[Dict[str, Any]], baseline_auc: float) -> None:
    print(f"\n{'=' * 60}\nHyperparameter search summary\n{'=' * 60}")
    print(f"Baseline holdout AUROC: {baseline_auc:.4f}")
    if not ranked:
        print("No completed variants beat rejection threshold.")
        return
    best = ranked[0]
    hp = best.get("hyperparameters", {})
    print(f"Best variant: {best['name']} (rank 1)")
    print(f"  lr={hp.get('lr')}, epochs={hp.get('epochs')}")
    print(f"  holdout AUROC={best.get('holdout_auc_roc'):.4f} (delta {best.get('delta_auc_roc'):+.4f})")
    print(f"  accuracy={best.get('holdout_accuracy'):.4f}, F1={best.get('holdout_f1'):.4f}")
    print("\nTop 5:")
    for row in ranked[:5]:
        hp = row.get("hyperparameters", {})
        print(
            f"  #{row['rank']} {row['name']}: "
            f"AUROC={row.get('holdout_auc_roc'):.4f} "
            f"(lr={hp.get('lr')}, epochs={hp.get('epochs')})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training hyperparameter search")
    parser.add_argument("--phase", default="small", choices=["small", "medium", "phase2"])
    parser.add_argument("--skip-train", action="store_true", help="Re-infer existing checkpoints only")
    parser.add_argument("--dry-run", action="store_true", help="Print planned variants and exit")
    parser.add_argument(
        "--write-report-only",
        action="store_true",
        help="Regenerate HP_SEARCH_REPORT.md from existing leaderboard.json",
    )
    args = parser.parse_args()

    if args.write_report_only:
        regenerate_report_from_leaderboard()
        return

    variants = list(iter_variants(args.phase))
    print(f"Phase={args.phase}, variants={len(variants)}")

    if args.dry_run:
        for v in variants:
            print(
                f"  {v.name}: lr={v.lr}, epochs={v.epochs}, "
                f"wd={v.weight_decay}, loss={v.loss_type}, noise={v.km_gaussian_noise_std}"
            )
        return

    if not SUPPORT_PKL_DIR.joinpath("patch_metadata.pkl").is_file():
        raise SystemExit(
            f"Support PKL missing at {SUPPORT_PKL_DIR}. "
            "Run: python pytorch/post_training/head_finetune.py --skip-sample"
        )
    if not HOLDOUT_PKL_DIR.joinpath("patch_metadata.pkl").is_file():
        raise SystemExit(
            f"Holdout PKL missing at {HOLDOUT_PKL_DIR}. "
            "Run: python pytorch/post_training/run_eval_pipeline.py --sample-holdout"
        )

    baseline = load_baseline_metrics()
    baseline_auc = float(baseline["auc_roc"])
    print(f"Baseline holdout AUROC: {baseline_auc:.4f}")

    previous = load_previous_results(args.phase)
    if previous:
        print(f"Merging with {len(previous)} prior variant(s) from existing leaderboard")

    results: List[Dict[str, Any]] = []
    for variant in variants:
        results.append(
            run_variant(variant, skip_train=args.skip_train, baseline_auc=baseline_auc)
        )

    merged = merge_results(previous, results) if args.phase == "phase2" else results
    ranked = rank_results(merged)

    if args.phase == "phase2":
        prior_phases = []
        if LEADERBOARD_JSON.is_file() and previous:
            prior_payload = load_json(LEADERBOARD_JSON)
            prior_phases = list(prior_payload.get("phases_run", [prior_payload.get("phase", "small")]))
        phases_run = list(dict.fromkeys(prior_phases + ["phase2"]))
        write_leaderboard(
            merged,
            ranked,
            baseline,
            "combined",
            phases_run=phases_run,
            n_new_variants=len(results),
        )
    else:
        write_leaderboard(merged, ranked, baseline, args.phase, n_new_variants=len(results))

    print_summary(ranked, baseline_auc)
    write_report_md(load_json(LEADERBOARD_JSON))


if __name__ == "__main__":
    main()
