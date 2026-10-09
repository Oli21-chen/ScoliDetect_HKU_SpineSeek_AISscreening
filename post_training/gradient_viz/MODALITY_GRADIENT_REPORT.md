# Modality Gradient Visualization Report

Generated: 2026-07-08T09:04:18.151539+00:00

## Summary

Per-modality gradient attribution during inference (backward from positive-class logit).
Each figure compares **baseline deploy**, **post-trained 3-shot**, and **post-trained 5-shot** on the same holdout patch.

**Checkpoint verification:** paths and MD5 hashes are embedded in each figure column and stored in `checkpoint_fingerprints` below. Encoders are identical across models; only fusion head weights differ. Input-level heatmaps therefore look nearly the same; use row 2 (logit/prob) and row 4 (ΔKM |grad| vs baseline) to see post-training effects.

- **Baseline checkpoint:** `C:\Users\Olive\Desktop\ScoliDetect_deployversion\checkpoints\kfold5_kvt_vivit_pretrained_cobb11\fold_1\checkpoint_best.pth`
- **Post-train 3-shot:** `C:\Users\Olive\Desktop\Nature_Style\code_video\pytorch\post_training\shot_runs\shot3\checkpoint_posttrain_head.pth`
- **Post-train 5-shot:** `C:\Users\Olive\Desktop\Nature_Style\code_video\pytorch\post_training\shot_runs\shot5\checkpoint_posttrain_head.pth`
- **Holdout PKL dir:** `C:\Users\Olive\Desktop\Nature_Style\code_video\pytorch\post_training\shot_runs\holdout_pkl`

## Examples

### Example A (TP)

- **Case:** TP | **Subject:** 903 | **Label:** 1.0 | **Figure:** `example_A_TP_sz_903.png`

| Model | Probability | Logit | Video % | KM % | Text % |
|-------|-------------|-------|---------|------|--------|
| Baseline | 0.7438 | 1.0660 | 33.6 | 35.1 | 31.3 |
| Post-train 3-shot | 0.7427 | 1.0600 | 33.6 | 35.1 | 31.3 |
| Post-train 5-shot | 0.7425 | 1.0591 | 33.6 | 35.1 | 31.3 |

**Modality mass shift (3-shot − baseline):** video -0.0pp, km +0.0pp, text -0.0pp

**Modality mass shift (5-shot − baseline):** video -0.0pp, km +0.0pp, text -0.0pp

### Example B (TN)

- **Case:** TN | **Subject:** 958 | **Label:** 0.0 | **Figure:** `example_B_TN_sz_958.png`

| Model | Probability | Logit | Video % | KM % | Text % |
|-------|-------------|-------|---------|------|--------|
| Baseline | 0.3184 | -0.7612 | 33.4 | 35.8 | 30.8 |
| Post-train 3-shot | 0.3173 | -0.7663 | 33.4 | 35.8 | 30.8 |
| Post-train 5-shot | 0.3168 | -0.7685 | 33.5 | 35.9 | 30.7 |

**Modality mass shift (3-shot − baseline):** video -0.0pp, km -0.0pp, text +0.0pp

**Modality mass shift (5-shot − baseline):** video +0.0pp, km +0.1pp, text -0.1pp

### Example C (FN)

- **Case:** FN | **Subject:** 941 | **Label:** 1.0 | **Figure:** `example_C_FN_sz_941.png`

| Model | Probability | Logit | Video % | KM % | Text % |
|-------|-------------|-------|---------|------|--------|
| Baseline | 0.4978 | -0.0088 | 33.8 | 36.7 | 29.6 |
| Post-train 3-shot | 0.4963 | -0.0148 | 33.8 | 36.7 | 29.6 |
| Post-train 5-shot | 0.4958 | -0.0170 | 33.8 | 36.7 | 29.6 |

**Modality mass shift (3-shot − baseline):** video -0.0pp, km +0.0pp, text +0.0pp

**Modality mass shift (5-shot − baseline):** video -0.0pp, km +0.0pp, text +0.0pp

## Interpretation guide

- **Modality gradient mass (%):** L2 norm of ∂logit/∂(modality token), normalized to sum to 100%.
- **Video temporal |grad|:** mean absolute input gradient over spatial dims per frame.
- **KM heatmap:** |∂logit/∂knowledge_map| with motion (0–34), skeleton (34–172), signal (172–238) bands.
- **Text top-k:** largest |grad| dimensions on the fused text embedding.

## Artifacts

- Figures: `C:\Users\Olive\Desktop\Nature_Style\code_video\pytorch\post_training\gradient_viz/*.png`
- Manifest: `C:\Users\Olive\Desktop\Nature_Style\code_video\pytorch\post_training\gradient_viz/manifest.json`
