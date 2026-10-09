# How to run (by purpose)

All commands assume the repository root:

```bash
cd /root/private_data/Dong_project
```

Adjust checkpoint paths, fold numbers, and CSV names to match your machine and run.

---

## Train k-fold models (ViT / Video Swin / TimeSformer / 3D CNN)

Configuration lives **inside** each script (no CLI). Pick the backbone file, set paths and hyperparameters there, then:

```bash
python run_end2end_kfold_vivit.py
# or: run_end2end_kfold_videoswin.py, run_end2end_kfold_timesformer.py, run_end2end_kfold_3dcnn.py
```

Outputs typically go under `checkpoints/<run_name>/` with `config.json` and `fold_k/checkpoint_best.pth`.

---

## Evaluate one checkpoint on concatenated test PKLs (quick lab check)

`run_test.py` uses **hard-coded** `checkpoint_path`, `test_binary_threshold`, and `test_pkl_data_dir_override` near the top of the file. Edit those, then:

```bash
python run_test.py
```

Use this when you want attention maps, loss, and metrics for a **single** fold checkpoint without the full export pipeline.

---

## Export validation + external predictions (DK + PK, for papers / curves)

Writes a timestamped CSV under `plots/<model_name>/` with `label`, `probability`, `split`, `fold`, `subject_id`, `cohort_source`, etc. Requires GPU and valid PKL dirs (from `config.json` or defaults).

**One fold (example fold 1):**

```bash
python tools/export_kfold_predictions_external.py \
  --ckpt_base_dir checkpoints/kfold5_kvt_vivit_pretrained_cobb11 \
  --target_folds 1
```

**All folds:** omit `--target_folds` (or pass no values per script help).

**Custom checkpoint root / output tree:**

```bash
python tools/export_kfold_predictions_external.py \
  --ckpt_base_dir /path/to/your_kfold_run \
  --ckpt_filename checkpoint_best.pth \
  --plots_subdir plots
```

Note the printed output path, or locate the newest `*extexport*_predictions.csv` under `plots/<model_name>/`.

---

## External patch-level figures (ROC, PR, calibration, JSON metrics)

Point `--csv` at the CSV from the export step. `--out_prefix` controls where PNG + JSON are written (same stem, `_curves.png` and `_metrics.json`).

```bash
python tools/plot_external_curve_suite.py \
  --csv plots/kfold5_kvt_vivit_pretrained_cobb11/<your_stem>_extexport_<timestamp>_predictions.csv \
  --split external \
  --fold 1 \
  --out_prefix plots/outputs/external_fold1_suite \
  --n_bootstrap 2000
```

**Subject-level bootstrap** (needs `subject_id` in the CSV):

```bash
python tools/plot_external_curve_suite.py \
  --csv plots/kfold5_kvt_vivit_pretrained_cobb11/<your_stem>_extexport_<timestamp>_predictions.csv \
  --split external \
  --fold 1 \
  --out_prefix plots/outputs/external_fold1_suite_subject \
  --subject_level \
  --n_bootstrap 2000
```

---

## External AIS-type subgroups (ROC + confusion, shared DK− ∪ PK−)

`tools/subgroup_external_curves_curvetype.py` — subgroup ROC / confusion for DK AIS-type strata from `data/subgroup_indices.json` (single vs multi curve, thoracic vs lumbar, general Cobb>10), with **shared negatives** = all DK non-AIS patches ∪ all PK patches. Does **not** require a prior CSV export; it loads test PKLs and evaluates checkpoint(s) directly.

**Prerequisites:** run from repo root; `--checkpoint_run_dir` must contain `config.json` and `fold_k/<ckpt_filename>` (default `checkpoint_best.pth`). Defaults use `data/test_dk_pkl^1`, `data/test_pk_pkl^1`, and `data/subgroup_indices.json` unless overridden. Writes figures (`external_subgroup_chart*_*.png`, optional `external_subgroup_auc_forest.png`) and `external_subgroup_analysis_summary.json` under `--out_dir`, and prints **ROC / AUC diagnostics** to the terminal.

**Inspect all flags:**

```bash
python tools/subgroup_external_curves_curvetype.py --help
```

**Single fold (typical manuscript run):**

```bash
python tools/subgroup_external_curves_curvetype.py \
  --checkpoint_run_dir checkpoints/kfold5_kvt_vivit_pretrained_cobb11 \
  --fold 1 \
  --out_dir subgroup_plots/external_curvetype
```

**Ensemble mean probability across all folds listed in `config.json`:**

```bash
python tools/subgroup_external_curves_curvetype.py \
  --checkpoint_run_dir checkpoints/kfold5_kvt_vivit_pretrained_cobb11 \
  --use_all_folds \
  --out_dir subgroup_plots/external_curvetype_allfolds
```

**Common optional arguments** (defaults otherwise come from `config.json` or the script):

| Flag | Role |
|------|------|
| `--ckpt_filename` | Checkpoint file name under each `fold_k/` (default `checkpoint_best.pth`). |
| `--dk_pkl_dir`, `--pk_pkl_dir` | Override DK / PK test PKL directories. |
| `--subgroup_json` | Override path to subgroup index JSON. |
| `--prob_threshold` | Probability threshold for confusion / binary metrics (default `0.5`). |
| `--batch_size` | Override batch size (else `config.json` `batch_size`, else `8`). |
| `--n_bootstrap`, `--bootstrap_seed` | Subgroup uncertainty / forest plots (default `2000`, seed `42`). |
| `--target_sensitivity` | One or more screening sensitivities (default `0.95 0.98`). Use `--target_sensitivity` with no numbers after it to disable those operating points. |
| `--projected_prevalence` | For PPV/NPV-style projections (default `0.05`). |

---

## External Cobb severity subgroups (ROC + confusion, shared DK− ∪ PK−)

`tools/subgroup_external_curves_severity.py` — same evaluation setup as the curve-type script, but strata are **Cobb severity bins** derived from `label` in `data/subgroup_indices.json`:

| Stratum | Definition |
|---------|------------|
| mild | 10 ≤ Cobb < 25 |
| moderate | 25 ≤ Cobb < 40 |
| severe | Cobb > 40 |

Shared negatives and CLI flags match the curve-type script. Outputs under `--out_dir`:

- `external_severity_chart1_mild_moderate_roc.png` / `_confusion.png`
- `external_severity_chart2_moderate_severe_roc.png` / `_confusion.png`
- `external_severity_chart3_three_strata_roc.png` / `_confusion.png`
- `external_severity_auc_forest.png` (when bootstrap uncertainty runs)
- `external_subgroup_analysis_summary.json`

**Inspect all flags:**

```bash
python tools/subgroup_external_curves_severity.py --help
```

**Single fold:**

```bash
python tools/subgroup_external_curves_severity.py \
  --checkpoint_run_dir checkpoints/kfold5_kvt_vivit_pretrained_cobb11 \
  --fold 1 \
  --out_dir subgroup_plots/external_severity
```

**All folds (mean TPR ± s.d., ensemble confusion):**

```bash
python tools/subgroup_external_curves_severity.py \
  --checkpoint_run_dir checkpoints/kfold5_kvt_vivit_pretrained_cobb11 \
  --use_all_folds \
  --out_dir subgroup_plots/external_severity_allfolds
```

See the **Common optional arguments** table in the curve-type section above (`--ckpt_filename`, `--dk_pkl_dir`, `--pk_pkl_dir`, `--subgroup_json`, `--prob_threshold`, `--batch_size`, `--n_bootstrap`, `--target_sensitivity`, `--projected_prevalence`).

---

## Compare metrics between two CSVs (e.g. negative-set sensitivity)

Requires columns `label`, `probability`; optional `subject_id` / `subject` for clustered bootstrap.

**Two scenarios:**

```bash
python tools/compare_negative_set_metrics.py \
  --name_a "DK only" --csv_a plots/run_a_predictions.csv \
  --name_b "DK+PK" --csv_b plots/run_b_predictions.csv \
  --n_bootstrap 2000 \
  --out_json plots/outputs/negative_set_ablation.json
```

**Single scenario (e.g. external fold 1 only):**

```bash
python tools/compare_negative_set_metrics.py --single \
  --name_a "External" --csv_a plots/kfold5_kvt_vivit_pretrained_cobb11/<your>_predictions.csv \
  --filter_split external --filter_fold 1 \
  --n_bootstrap 2000
```

---

## Typical paper pipeline (short)

1. Train or locate `checkpoints/<run>/config.json` + per-fold checkpoints.  
2. `python tools/export_kfold_predictions_external.py --ckpt_base_dir ... --target_folds 1`  
3. `python tools/plot_external_curve_suite.py --csv ... --split external --fold 1 --out_prefix ...`  
4. Optional subgroups: `python tools/subgroup_external_curves_curvetype.py ...` and/or `python tools/subgroup_external_curves_severity.py ...`  
5. Optional ablation: `python tools/compare_negative_set_metrics.py ...`
