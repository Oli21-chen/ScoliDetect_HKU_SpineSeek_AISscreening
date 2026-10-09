# Head-Only Post-Training

Few-shot head-only fine-tuning for ScoliDetect: freeze `video_encoder`, `km_encoder`, and `text_encoder`, then train the fusion layers and regressor on two support subjects. Evaluate baseline vs post-trained models on a holdout set.

## Prerequisites

- Conda env: `PytorchCuda11.8` (or set `CONDA_PYTHON` in `config.py`)
- Run commands from the `code_video` root:

```
C:\Users\Olive\Desktop\Nature_Style\code_video
```

- Base deploy checkpoint:

```
C:\Users\Olive\Desktop\ScoliDetect_deployversion\checkpoints\kfold5_kvt_vivit_pretrained_cobb11\fold_1\checkpoint_best.pth
```

## Input Data (read-only)

| Purpose | Path |
|---------|------|
| Pose CSVs | `C:\Users\Olive\Desktop\experiments\table\` (`sz_*_step_1.csv`) |
| Trimmed videos | `C:\Users\Olive\Desktop\experiments\video\` (`sz_*_step1.mp4`) |
| Cobb labels | `C:\Users\Olive\Desktop\video_retrival\video_retrival\Label_SZpart2.xlsx` |
| Gait prompts | `pytorch/data/general_gait_prompts_from_report.json` (`concise_prompts`, same as deploy) |

**Support / holdout split** (subjects with index ≥ 884, Cobb threshold 11):

| Set | Subjects | Role |
|-----|----------|------|
| Support | **884** (first pos), **885** (first neg) | Head-only training |
| Holdout | All other paired SZ subjects | Inference + comparison |

KM profile for sampling: `SCOLI_KM_PROFILE=deploy_pose_size_no_zscore`

## Output Data (saved here)

All post-training artifacts live under:

```
C:\Users\Olive\Desktop\Nature_Style\code_video\pytorch\post_training\
```

| Path | Contents |
|------|----------|
| `support_pkl/` | PKL patches for subjects 884 + 885 (`patches/`, `patch_metadata.pkl`) |
| `holdout_pkl/` | PKL patches for all holdout subjects |
| `holdout_pkl/inference_baseline/` | Baseline deploy-model inference on holdout |
| `holdout_pkl/inference_posttrain/` | Post-trained model inference on holdout |
| `checkpoints/checkpoint_posttrain_head.pth` | Post-trained checkpoint (deploy-compatible) |
| `posttrain_manifest.json` | Support indices, hyperparameters, training history |
| `comparison/` | Baseline vs post-trained figures and metrics |
| `comparison/pipeline_summary.json` | End-to-end AUROC / accuracy summary |

### Inference outputs (per model)

Inside `inference_baseline/` or `inference_posttrain/`:

- `inference_results.json` — per-patch probabilities and labels
- `metrics.json` — AUROC, accuracy, precision, recall, F1
- `results_analysis.png` — probability / KM / label plots
- `negative_sample_values.png` — negative cohort distributions

### Comparison outputs

Inside `comparison/`:

- `baseline_vs_posttrain_overview.png` — side-by-side cohort comparison
- `baseline_vs_posttrain_delta.png` — per-patch probability shift
- `comparison_metrics.json` — paired metrics and Mann–Whitney test

## Quick Start (full pipeline)

```powershell
cd C:\Users\Olive\Desktop\Nature_Style\code_video
$env:SCOLI_KM_PROFILE = "deploy_pose_size_no_zscore"
$env:PYTHONIOENCODING = "utf-8"

python pytorch/post_training/run_eval_pipeline.py --all
```

This runs, in order:

1. Support sampling + head fine-tune (`head_finetune.py`)
2. Holdout sampling (`02_Sampling.py`)
3. Baseline inference (`03_Infer.py`)
4. Post-trained inference (`03_Infer.py` with post-trained checkpoint)
5. Per-model analysis (`04_results_analyze.py` × 2)
6. Baseline vs post-trained comparison

Use `--skip-symmetry` to skip gait symmetry scoring (faster):

```powershell
python pytorch/post_training/run_eval_pipeline.py --all --skip-symmetry
```

Skip re-training if a checkpoint already exists:

```powershell
python pytorch/post_training/run_eval_pipeline.py --all --skip-finetune
```

## Rerun Post-Training

All commands assume you are in the `code_video` root:

```powershell
cd C:\Users\Olive\Desktop\Nature_Style\code_video
$env:SCOLI_KM_PROFILE = "deploy_pose_size_no_zscore"
$env:PYTHONIOENCODING = "utf-8"
```

### Option A — Retrain only (fastest)

Use existing `support_pkl/`; overwrites `checkpoints/checkpoint_posttrain_head.pth` and `posttrain_manifest.json`.

```powershell
python pytorch/post_training/head_finetune.py --skip-sample
```

Change hyperparameters (defaults come from `config.py`):

```powershell
python pytorch/post_training/head_finetune.py --skip-sample --epochs 30 --lr 1e-4 --batch-size 4
```

### Option B — Resample support + retrain

Rebuild support PKLs from subjects 884 and 885, then train:

```powershell
python pytorch/post_training/head_finetune.py
```

### Option C — Retrain + re-evaluate holdout (recommended after config changes)

Retrain, then run inference and visualization on holdout (skips support resampling if `support_pkl/` exists):

```powershell
python pytorch/post_training/run_eval_pipeline.py --finetune --infer --analyze --compare --skip-symmetry
```

Full rerun from scratch (resample support, retrain, resample holdout, infer, visualize):

```powershell
python pytorch/post_training/run_eval_pipeline.py --all --skip-symmetry
```

## Visualize After Training

Run these **after** post-training (and holdout inference) finishes.

### Per-model holdout plots

Baseline model:

```powershell
python main/04_results_analyze.py `
  --results-json pytorch/post_training/holdout_pkl/inference_baseline/inference_results.json `
  --pkl-dir pytorch/post_training/holdout_pkl `
  --table-dir C:\Users\Olive\Desktop\experiments\table `
  --skip-symmetry
```

Post-trained model:

```powershell
python main/04_results_analyze.py `
  --results-json pytorch/post_training/holdout_pkl/inference_posttrain/inference_results.json `
  --pkl-dir pytorch/post_training/holdout_pkl `
  --table-dir C:\Users\Olive\Desktop\experiments\table `
  --skip-symmetry
```

Outputs are written next to each inference JSON:

- `holdout_pkl/inference_baseline/results_analysis.png`
- `holdout_pkl/inference_baseline/negative_sample_values.png`
- `holdout_pkl/inference_posttrain/results_analysis.png`
- `holdout_pkl/inference_posttrain/negative_sample_values.png`

### Baseline vs post-trained comparison

```powershell
python main/04_results_analyze.py `
  --compare-baseline-vs-posttrain `
  --baseline-inference-dir pytorch/post_training/holdout_pkl/inference_baseline `
  --posttrain-inference-dir pytorch/post_training/holdout_pkl/inference_posttrain `
  --pkl-dir pytorch/post_training/holdout_pkl `
  --table-dir C:\Users\Olive\Desktop\experiments\table `
  --compare-output-dir pytorch/post_training/comparison `
  --skip-symmetry
```

Or use the pipeline wrapper (same result):

```powershell
python pytorch/post_training/run_eval_pipeline.py --analyze --compare --skip-symmetry
```

Comparison figures saved to:

- `pytorch/post_training/comparison/baseline_vs_posttrain_overview.png`
- `pytorch/post_training/comparison/baseline_vs_posttrain_delta.png`
- `pytorch/post_training/comparison/comparison_metrics.json`
- `pytorch/post_training/comparison/pipeline_summary.json`

Add `--show` to any `04_results_analyze.py` command to open figures interactively:

```powershell
python main/04_results_analyze.py `
  --compare-baseline-vs-posttrain `
  --baseline-inference-dir pytorch/post_training/holdout_pkl/inference_baseline `
  --posttrain-inference-dir pytorch/post_training/holdout_pkl/inference_posttrain `
  --pkl-dir pytorch/post_training/holdout_pkl `
  --table-dir C:\Users\Olive\Desktop\experiments\table `
  --compare-output-dir pytorch/post_training/comparison `
  --skip-symmetry `
  --show
```

## Step-by-Step Usage

### 1. Head-only fine-tune

```powershell
python pytorch/post_training/head_finetune.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--support-pkl-dir` | `post_training/support_pkl` | Support PKL output / input |
| `--base-checkpoint` | deploy `checkpoint_best.pth` | Starting weights |
| `--output-checkpoint` | `post_training/checkpoints/checkpoint_posttrain_head.pth` | Saved model |
| `--epochs` | `1` | Training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--batch-size` | `1` | Batch size |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--km-gaussian-noise-std` | `0.01` | KM augmentation (train only) |
| `--loss-type` | `bce` | `bce` or `focal` |
| `--shots-per-class` | `1` | K positive + K negative support subjects |
| `--quiet` | off | Suppress per-batch logs |
| `--skip-sample` | off | Skip `02_Sampling` if support PKLs exist |

## Hyperparameter Search

Small sweep: **4 learning rates × 3 epoch counts = 12 runs**, ranked by holdout AUROC vs baseline.

### Prerequisites (one-time)

```powershell
cd C:\Users\Olive\Desktop\Nature_Style\code_video
$env:PYTHONIOENCODING = "utf-8"

python pytorch/post_training/head_finetune.py --skip-sample
python pytorch/post_training/run_eval_pipeline.py --sample-holdout --infer --skip-finetune --skip-symmetry
```

### Run 12-variant sweep (phase 1)

```powershell
python pytorch/post_training/hyperparam_search.py --phase small
```

### Run phase 2 sweep (12 variants, merges leaderboard)

Phase 2 explores refined LR, weight decay, focal loss, and KM noise around the phase 1 anchor (`lr=1e-7`, `epochs=10`). Results merge into the existing `leaderboard.json` (phase 1 runs are preserved).

```powershell
python pytorch/post_training/hyperparam_search.py --phase phase2
```

Preview planned runs without training:

```powershell
python pytorch/post_training/hyperparam_search.py --phase phase2 --dry-run
```

Preview phase 1 runs:

```powershell
python pytorch/post_training/hyperparam_search.py --phase small --dry-run
```

Re-score existing checkpoints (skip retrain):

```powershell
python pytorch/post_training/hyperparam_search.py --phase phase2 --skip-train
```

### Search grids

**Phase 1 (`small`)**

| Parameter | Values |
|-----------|--------|
| `lr` | `1e-8`, `1e-7`, `1e-6`, `1e-5` |
| `epochs` | `5`, `10`, `20` |

**Phase 2 (`phase2`)** — anchor: `lr=1e-7`, `epochs=10`, `weight_decay=1e-4`, `loss=bce`, `km_noise=0.01`

| Group | Swept |
|-------|-------|
| refined_lr | `lr`: 3e-8, 3e-7, 5e-7, 8e-7 |
| weight_decay | `weight_decay`: 0, 1e-5, 1e-3, 1e-2 |
| loss_noise | `(loss, noise)`: (bce,0), (focal,0.01), (focal,0), (bce,0.05) |

Fixed across both phases: `batch_size=1`

### Outputs

| Path | Contents |
|------|----------|
| `hp_runs/{name}/checkpoint_posttrain_head.pth` | Per-variant checkpoint |
| `hp_runs/{name}/manifest.json` | Training history |
| `hp_runs/{name}/inference/metrics.json` | Holdout metrics |
| `hp_runs/leaderboard.json` | Ranked results (best first) |
| `hp_runs/leaderboard.csv` | Spreadsheet-friendly summary |
| `hp_runs/HP_SEARCH_REPORT.md` | Full markdown results report (auto-generated after sweep) |

**Selection rule:** highest holdout AUROC; reject runs more than 0.01 below baseline.

Regenerate report without retraining:

```powershell
python pytorch/post_training/hyperparam_search.py --write-report-only
```

### Apply the winner

1. Open `hp_runs/leaderboard.json` and read `best.hyperparameters` (lr, epochs, weight_decay, loss_type, km_gaussian_noise_std)
2. Copy into [`config.py`](config.py) `HeadFinetuneConfig`
3. Re-run full eval:

```powershell
python pytorch/post_training/run_eval_pipeline.py --finetune --infer --analyze --compare --skip-symmetry
```

**Note:** With only 2 support patches, expect small AUROC deltas (±0.01). Holdout is used to pick HP — treat gains as directional calibration, not rigorous generalization.

## Few-Shot Experiments (1 / 3 / 5)

Compare head-only post-training at increasing shot counts using the **fixed best hyperparameters** from the HP sweep. All shot runs are evaluated on the **same fixed holdout** for fair AUROC comparison.

### Shot definition

- **K-shot** = K positive + K negative subjects (nested by label-Excel order among paired subjects)
- **1-shot:** 2 subjects (884 pos + 885 neg)
- **3-shot:** 6 subjects (3 pos + 3 neg)
- **5-shot:** 10 subjects (5 pos + 5 neg)

### Holdout protocol (fixed)

- **Support pool (max):** first 5 pos + first 5 neg paired subjects
- **Fixed holdout:** all other paired subjects (never used for training in any shot condition)
- Baseline deploy checkpoint inferred once on fixed holdout; all shot models evaluated on identical holdout PKLs

**Important:** 1-shot AUROC from `shot_runs/` uses a **smaller fixed holdout** than `hp_runs/` (which only excluded 884/885). Do not mix AUROC numbers across those directories.

### Best hyperparameters (fixed for all shot counts)

From `hp_runs/leaderboard.json` rank 1 (`lr1e-7_ep10_wd1e-3`):

| Parameter | Value |
|-----------|-------|
| `lr` | `1e-7` |
| `epochs` | `10` |
| `weight_decay` | `1e-3` |
| `loss_type` | `bce` |
| `km_gaussian_noise_std` | `0.01` |
| `batch_size` | `1` |

### Run few-shot experiment

```powershell
cd C:\Users\Olive\Desktop\Nature_Style\code_video
$env:SCOLI_KM_PROFILE = "deploy_pose_size_no_zscore"
$env:PYTHONIOENCODING = "utf-8"

# Preview support/holdout splits
python pytorch/post_training/shot_experiment.py --dry-run

# Run full 1/3/5-shot experiment
python pytorch/post_training/shot_experiment.py

# Subset of shot counts
python pytorch/post_training/shot_experiment.py --shots 3 5

# Re-infer existing checkpoints (skip retrain)
python pytorch/post_training/shot_experiment.py --skip-train

# Regenerate report from existing leaderboard
python pytorch/post_training/shot_experiment.py --write-report-only
```

Single-shot training via `head_finetune.py`:

```powershell
python pytorch/post_training/head_finetune.py --shots-per-class 3
```

### Few-shot outputs

| Path | Contents |
|------|----------|
| `shot_runs/shot{K}/support_pkl/` | Support PKL patches for K-shot |
| `shot_runs/shot{K}/checkpoint_posttrain_head.pth` | Post-trained checkpoint |
| `shot_runs/shot{K}/manifest.json` | Training history + support subjects |
| `shot_runs/shot{K}/inference/metrics.json` | Holdout metrics |
| `shot_runs/holdout_pkl/` | Shared fixed holdout PKLs |
| `shot_runs/holdout_pkl/inference_baseline/` | Baseline deploy inference |
| `shot_runs/leaderboard.json` | Comparison leaderboard (best AUROC per shot) |
| `shot_runs/leaderboard.csv` | Spreadsheet-friendly summary |
| `shot_runs/SHOT_EXPERIMENT_REPORT.md` | Markdown report with 1/3/5-shot AUROC table |

Primary result table (from `leaderboard.csv`):

| shots_per_class | n_support_subjects | holdout_auc_roc | delta_auc_roc |
|-----------------|-------------------|-----------------|---------------|
| 1 | 2 | ... | ... |
| 3 | 6 | ... | ... |
| 5 | 10 | ... | ... |

## Large-Support Experiments (10 / 15-shot)

Tests whether **more head-only support** (20–30 subjects) breaks the plateau seen at 1/3/5-shot. Uses a **separate runs directory** so prior `shot_runs/` artifacts are untouched.

### Shot definition

- **10-shot:** 20 subjects (10 pos + 10 neg), nested subset of 15-shot
- **15-shot:** 30 subjects (15 pos + 15 neg)
- **Fixed holdout:** all paired subjects outside the 15-shot support pool (~67 subjects)
- Same best HP as few-shot (`lr=1e-7`, `epochs=10`, `wd=1e-3`, BCE, `km_noise=0.01`)

**Important:** Holdout cohort differs from `shot_runs/` (which excluded only 5+5 support). Do **not** compare absolute AUROC across `shot_runs/` and `shot_runs_large/`.

### Run large-support experiment

```powershell
cd C:\Users\Olive\Desktop\Nature_Style\code_video
$env:SCOLI_KM_PROFILE = "deploy_pose_size_no_zscore"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Preview splits
python pytorch/post_training/shot_experiment.py --dry-run `
  --runs-dir pytorch/post_training/shot_runs_large `
  --max-shots-per-class 15 --shots 10 15

# Full train + infer + analysis
python pytorch/post_training/shot_experiment.py `
  --runs-dir pytorch/post_training/shot_runs_large `
  --max-shots-per-class 15 --shots 10 15 --analyze

# Re-infer existing checkpoints only
python pytorch/post_training/shot_experiment.py --skip-train `
  --runs-dir pytorch/post_training/shot_runs_large `
  --max-shots-per-class 15 --shots 10 15

# Re-run analysis only
python pytorch/post_training/shot_experiment.py --analyze-only `
  --runs-dir pytorch/post_training/shot_runs_large
```

### Large-support outputs

| Path | Contents |
|------|----------|
| `shot_runs_large/shot{10,15}/support_pkl/` | Support PKL patches |
| `shot_runs_large/shot{K}/checkpoint_posttrain_head.pth` | Post-trained checkpoint |
| `shot_runs_large/shot{K}/manifest.json` | Training history |
| `shot_runs_large/holdout_pkl/` | Shared fixed holdout PKLs |
| `shot_runs_large/leaderboard.json` | AUROC comparison |
| `shot_runs_large/SHOT_EXPERIMENT_REPORT.md` | Report + analysis section |
| `shot_runs_large/analysis/` | Scaling plot, prediction deltas, comparison figures |

Analysis module: [`shot_analysis.py`](shot_analysis.py) — AUROC vs support curve, prediction flip counts, baseline vs best-shot comparison, cross-reference to `shot_runs/` 1/3/5 results.

## Encoder Method Experiments (Partial Unfreeze vs Full Encoder FT)

Tests whether **representation-level** fine-tuning beats the head-only ceiling on the same 15-shot support (30 subjects) and `shot_runs_large` holdout. Primary metric: **holdout AUROC** vs deploy baseline **0.7476**.

### Methods

| Method | Policy | Encoder LR | Head LR | Epochs |
|--------|--------|------------|---------|--------|
| `partial_unfreeze` | Last 2 blocks video+km + fusion head | `1e-7` | `1e-5` | 8 |
| `full_encoder_lowlr` | Full video+km + fusion; text frozen | `3e-8` | `3e-8` | 5 |

Shared: BCE loss, `weight_decay=1e-3`, `km_noise=0.01`, `batch_size=1`, support PKL from `shot_runs_large/shot15/support_pkl`.

### Run method experiment

```powershell
cd C:\Users\Olive\Desktop\Nature_Style\code_video
$env:SCOLI_KM_PROFILE = "deploy_pose_size_no_zscore"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Train both methods + infer + analysis
python pytorch/post_training/method_experiment.py --methods partial_unfreeze full_encoder_lowlr --analyze

# Re-infer existing checkpoints only
python pytorch/post_training/method_experiment.py --skip-train --analyze

# Analysis only
python pytorch/post_training/method_experiment.py --analyze-only
```

### Method outputs

| Path | Contents |
|------|----------|
| `method_runs/partial_unfreeze/checkpoint_posttrain.pth` | Partial-unfreeze checkpoint |
| `method_runs/full_encoder_lowlr/checkpoint_posttrain.pth` | Full-encoder checkpoint |
| `method_runs/{method}/manifest.json` | Training history + freeze stats |
| `method_runs/{method}/inference/metrics.json` | Holdout metrics |
| `method_runs/leaderboard.json` | AUROC comparison vs baseline |
| `method_runs/METHOD_EXPERIMENT_REPORT.md` | Report + analysis section |
| `method_runs/analysis/` | AUROC bar chart, per-subject CSV, confusion counts |

Analysis module: [`method_analysis.py`](method_analysis.py).

**Success criteria:** Δ AUROC > +0.01 is meaningful; both ≤ baseline suggests encoder FT also insufficient at 30 support subjects.

### Partial unfreeze at 1-shot and 5-shot (`shot_runs` cohort)

Trains partial unfreeze on the **same support and holdout** as existing 1/5-shot head-only runs (`shot_runs/`, 85 labeled patches). **Not comparable** to the 15-shot `MODEL_COMPARISON_REPORT.md` (different holdout).

```powershell
cd C:\Users\Olive\Desktop\Nature_Style\code_video
$env:SCOLI_KM_PROFILE = "deploy_pose_size_no_zscore"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Train partial_unfreeze at 1-shot + 5-shot, infer, 3-way comparison report
python pytorch/post_training/method_experiment.py --shot-runs --shots 1 5 --methods partial_unfreeze --analyze

# Re-infer only + regenerate comparison
python pytorch/post_training/method_experiment.py --shot-runs --shots 1 5 --methods partial_unfreeze --skip-train --analyze

# Comparison report only (baseline vs head-only vs partial)
python pytorch/post_training/method_experiment.py --shot-compare-only
```

| Path | Contents |
|------|----------|
| `method_runs/shot1/partial_unfreeze/` | 1-shot partial-unfreeze checkpoint + inference |
| `method_runs/shot5/partial_unfreeze/` | 5-shot partial-unfreeze checkpoint + inference |
| `method_runs/shot_method_leaderboard.json` | AUROC summary for 1/5-shot partial unfreeze |
| `method_runs/SHOT_METHOD_COMPARISON_REPORT.md` | 3-way comparison vs deploy baseline + head-only |
| `method_runs/analysis/shot_auroc_comparison.png` | Grouped AUROC bar chart |

### Unified shot scaling (1 / 5 / 10 / 15-shot, same holdout)

Fair comparison of **deploy baseline**, **head-only**, and **partial unfreeze** at K ∈ {1, 5, 10, 15} on the **same** `shot_runs_large/holdout_pkl` (65 patches, baseline AUROC **0.7476**). Support sets are nested subsets of the 15-shot max pool (2 / 10 / 20 / 30 subjects).

**Holdout comparability rules:**

| Cohort | Holdout | Patches | Comparable? |
|--------|---------|---------|-------------|
| `shot_runs/` | Small-support fixed holdout | 85 | No — different from large cohort |
| `shot_runs_large/` | 15-max-pool fixed holdout | 65 | **Yes** — use for unified scaling |

Head-only checkpoints for 1/5 live in `shot_runs/`; 10/15 in `shot_runs_large/`. Only **re-inference** on the large holdout is needed for head-only (no retraining). Partial unfreeze is trained per shot on large-cohort support PKLs; 15-shot reuses `method_runs/partial_unfreeze/checkpoint_posttrain.pth`.

```powershell
cd C:\Users\Olive\Desktop\Nature_Style\code_video
$env:SCOLI_KM_PROFILE = "deploy_pose_size_no_zscore"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Full pipeline: sample support (if missing), train partial_unfreeze 1/5/10,
# skip-train + infer 15, re-infer head-only, unified report
python pytorch/post_training/method_experiment.py --large-shot-runs --shots 1 5 10 15 --methods partial_unfreeze --analyze

# Re-infer existing partial-unfreeze checkpoints + head-only only
python pytorch/post_training/method_experiment.py --large-shot-runs --shots 1 5 10 15 --methods partial_unfreeze --skip-train --analyze

# Unified comparison report only
python pytorch/post_training/method_experiment.py --unified-compare-only
# or
python pytorch/post_training/method_analysis.py --unified-compare
```

| Path | Contents |
|------|----------|
| `method_runs/shot{K}/partial_unfreeze/` | Partial-unfreeze checkpoint + inference (large holdout) |
| `method_runs/shot{K}/head_only/inference/` | Head-only re-inferred on large holdout |
| `method_runs/unified_shot_scaling_leaderboard.json` | AUROC summary across K |
| `method_runs/UNIFIED_SHOT_SCALING_REPORT.md` | Scaling table + confusion matrices |
| `method_runs/analysis/unified_shot_scaling_auroc.png` | AUROC vs support-subject line chart |
| `method_runs/analysis/unified_shot_holdout_per_subject_results.csv` | Per-subject baseline vs head-only vs partial |

> Prior 1/5-shot partial-unfreeze numbers in [`SHOT_METHOD_COMPARISON_REPORT.md`](method_runs/SHOT_METHOD_COMPARISON_REPORT.md) used the `shot_runs` holdout and are **not comparable** to this report.

## Modality Gradient Visualization

Explain **which modality (video / KM / text) drives each inference prediction** via gradient attribution. Produces 3 holdout examples (TP, TN, FN/borderline) with **baseline deploy vs post-trained 3-shot vs post-trained 5-shot** side-by-side on identical patches.

### What is shown

| Panel | Content |
|-------|---------|
| Modality bars | % of gradient L2 mass on video / KM / text tokens |
| Video temporal | Mean \|∂logit/∂video\| per frame |
| KM heatmap | \|∂logit/∂knowledge_map\| with motion / skeleton / signal bands |
| Text top-k | Largest \|grad\| dimensions on text embedding |

### Run

```powershell
cd C:\Users\Olive\Desktop\Nature_Style\code_video
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Preview auto-selected examples (TP, TN, error)
python pytorch/post_training/visualize_modality_gradients.py --dry-run

# Generate 3 figures + report
python pytorch/post_training/visualize_modality_gradients.py

# Custom subjects
python pytorch/post_training/visualize_modality_gradients.py --subjects 903 958 941
```

Defaults: holdout `shot_runs/holdout_pkl`, post-train `shot_runs/shot3/` and `shot_runs/shot5/checkpoint_posttrain_head.pth`, baseline deploy `BASE_CHECKPOINT`.

### Outputs

| Path | Contents |
|------|----------|
| `gradient_viz/example_*.png` | 4×3 panel figures (baseline \| 3-shot \| 5-shot) |
| `gradient_viz/manifest.json` | Numeric gradient breakdown per example |
| `gradient_viz/MODALITY_GRADIENT_REPORT.md` | Summary tables and interpretation |

### 2. Holdout evaluation only

```powershell
# Sample holdout PKLs
python pytorch/post_training/run_eval_pipeline.py --sample-holdout

# Run both inferences
python pytorch/post_training/run_eval_pipeline.py --infer

# Generate plots + comparison
python pytorch/post_training/run_eval_pipeline.py --analyze --compare --skip-symmetry
```

### 3. Compare via `04_results_analyze.py` directly

```powershell
python main/04_results_analyze.py `
  --compare-baseline-vs-posttrain `
  --baseline-inference-dir pytorch/post_training/holdout_pkl/inference_baseline `
  --posttrain-inference-dir pytorch/post_training/holdout_pkl/inference_posttrain `
  --pkl-dir pytorch/post_training/holdout_pkl `
  --table-dir C:\Users\Olive\Desktop\experiments\table `
  --compare-output-dir pytorch/post_training/comparison `
  --skip-symmetry
```

## Configuration

Edit [`config.py`](config.py) to change paths and defaults:

```python
@dataclass(frozen=True)
class HeadFinetuneConfig:
    epochs: int = 10
    lr: float = 1e-7
    batch_size: int = 1
    weight_decay: float = 1e-3
    km_gaussian_noise_std: float = 0.01
    loss_type: str = "bce"
    num_workers: int = 0
    verbose: bool = True   # per-batch / per-epoch training logs
```

## Module Overview

| File | Role |
|------|------|
| `config.py` | Paths, hyperparameters, Python executable |
| `support_split.py` | Resolve support/holdout indices; K-shot support pools |
| `freeze.py` | Freeze policies (head-only, partial unfreeze, full encoder) + param groups |
| `model_io.py` | Load deploy-compatible `SFTRegressor` checkpoint |
| `head_finetune.py` | Sample support PKLs and run head-only training |
| `run_eval_pipeline.py` | Orchestrate 02 → 03 → 04 → comparison |
| `hyperparam_grid.py` | LR/epochs search grids |
| `hyperparam_search.py` | Train + infer sweep, write leaderboard |
| `shot_experiment.py` | Few-shot / large-support experiment orchestrator + leaderboard |
| `shot_analysis.py` | Post-run analysis: scaling curve, prediction flips, comparisons |
| `method_experiment.py` | Encoder method comparison (partial unfreeze vs full encoder FT) |
| `method_analysis.py` | Method experiment analysis: AUROC chart, per-subject deltas |
| `modality_gradients.py` | Per-modality gradient attribution + plotting |
| `visualize_modality_gradients.py` | CLI: 3-example baseline vs post-train gradient figures |

## Notes

- The post-trained checkpoint uses the same format as the deploy model, so `03_Infer.py --checkpoint` loads it without code changes.
- With only two support subjects, expect support-set overfitting; holdout AUROC is the primary metric.
- Training logs appear when `verbose=True` in `HeadFinetuneConfig` (batch loss, accuracy, AUC per step).
