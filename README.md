# ScoliDetect

**Explainable multimodal learning from gait video for scalable adolescent idiopathic scoliosis (AIS) screening**

<div align="center">

<img src="figs/image2.png" alt="ScoliDetect framework: trimodal contrastive alignment, supervised multimodal fusion, and KKM-based structural interpretability" width="100%">

*<strong>a</strong>, Trimodal contrastive pretraining aligns video, KKM, and text encoders. <strong>b</strong>, Supervised screening head with bidirectional video–KKM cross-attention, gated pooling, bottleneck fusion, and optional auxiliary alignment loss. <strong>c</strong>, Fixed-index KKM domains (motion, self-skeleton, signal cross-correlation) enabling top-<em>k</em> factor readouts.*

</div>

[![GitHub](https://img.shields.io/badge/GitHub-ScoliDetect-blue?logo=github)](https://github.com/Oli21-chen/ScoliDetect_HKU_SpineSeek_AISscreening)
[![License](https://img.shields.io/badge/License-Research%20Only-red)](LICENSE)

> **Release status:** This repository is under active development. Pretrained weights, full training scripts, sample data, capture protocol checklist, and supplementary materials **will be released soon**. Check back for updates.

> **Clinical disclaimer:** ScoliDetect is an assistive **referral-triage** tool for use under standardized capture protocols. It is **not** a substitute for clinical examination or radiographic Cobb-angle measurement. Intended users include school nurses, primary-care staff, and trained screening personnel.

---

## Overview

ScoliDetect screens for adolescent idiopathic scoliosis (AIS) from **monocular gait video** using a multimodal deep learning framework built around the **Kinematic Knowledge Map (KKM)**—a fixed-index structured latent that encodes gait across three scientific domains:


| Domain                       | KKM indices | Description                                      |
| ---------------------------- | ----------- | ------------------------------------------------ |
| **Motion**                   | 0–63        | Absolute joint motion trajectories               |
| **Self-skeleton**            | 64–171      | Intra-skeleton configuration (angles, distances) |
| **Signal cross-correlation** | 172–237     | Joint–joint temporal correlations                |


Each aligned gait cycle is serialized as a **96 × 238** array. Video, KKM, and weak rule-based kinematic text are fused through **bidirectional cross-attention** with **Perceiver-style latent-bottleneck aggregation**, enabling both strong discrimination and **factor-level interpretability** at predefined indices rather than opaque pixel saliency.

Evaluated in a multicenter cohort (**n = 1,858** after exclusions) across **Hospital A**, **Hospital B**, and **School A**, the full KKM–video–text (KVT) model achieves **external ROC-AUC = 0.972** with stable performance across Cobb severity and curve-type subgroups.

---

## Key results


| Setting                    | Model                           | External ROC-AUC |
| -------------------------- | ------------------------------- | ---------------- |
| Unimodal baseline          | Video only                      | 0.784            |
| Unimodal baseline          | KKM only                        | 0.927            |
| Multimodal fusion          | KKM + Video (KV)                | 0.947            |
| Full model (from scratch)  | KKM + Video + Text (KVT)        | 0.961            |
| Full model (+ pretraining) | KVT + trimodal contrastive init | **0.972**        |


Subgroup external AUC remains stable (~0.969–0.973) across mild / moderate / severe Cobb strata and single-thoracic / single-lumbar / multi-curve phenotypes. At sensitivity ≥ 0.95, projected NPV ≥ 0.997 at 5% assumed prevalence.

---

## Architecture

See the [framework figure at the top of this page](#scolidetect) or open [`figs/image2.png`](figs/image2.png) directly.

**a — Trimodal contrastive alignment.** Video, KKM, and weak kinematic text are encoded into a shared embedding space with pairwise contrastive losses (video–KKM, video–text, KKM–text).

**b — Supervised multimodal fusion.** Bidirectional video–KKM cross-attention with temporal masking, gated token pooling, Perceiver-style bottleneck fusion, and text concatenation feed a binary screening classifier. Optional auxiliary video–KKM InfoNCE preserves cross-modal consistency during fine-tuning.

**c — Structural interpretability.** The 96 × 238 KKM decomposes into motion, self-skeleton, and signal cross-correlation domains, enabling top-*k* factor readouts tied to model inputs rather than post-hoc pixel saliency.

**Training protocol (staged):**

1. **Architecture selection** — prespecified supervised from-scratch ablations on held-out external screening data define modality and fusion design.
2. **Trimodal contrastive pretraining** — pairwise InfoNCE losses align video, KKM, and text encoders in a shared embedding space (encoder initialization only).
3. **Supervised fine-tuning** — focal loss with optional auxiliary video–KKM InfoNCE; peak-anchored temporal registration shared across stages.

---

## Repository structure

```
ScoliDetect_HKU_HKUSZH/
├── figs/
│   └── image2.png            # Algorithm overview (contrastive alignment, fusion, KKM domains)
├── LICENSE                   # Research & evaluation license (patent pending)
├── models/
│   ├── sft_regressor.py      # KVT classifier: cross-attn + bottleneck fusion + screening head
│   ├── video_encoder.py      # ViViT and alternative video backbones
│   ├── knowledge_encoder.py  # Transformer encoder for KKM sequences
│   └── text_encoder.py       # Sentence-Transformer wrapper
├── utils/
│   ├── data_sampler.py       # PKL / on-the-fly gait dataset loaders
│   ├── knowledge_map.py      # KKM construction utilities
│   ├── sft_utils.py          # SFT training, evaluation, checkpoint helpers
│   ├── training.py           # Trimodal contrastive pretraining loops
│   ├── km_interpretability_core6.py  # Fig. 5-style factor attribution plots
│   └── plot_style_nature.py  # Publication-style figure styling
├── checkpoints/              # Model configs and k-fold summaries
├── logs/                     # Evaluation outputs (metrics, predictions, interpretability)
└── run_test.py               # External-test evaluation entry point
```

---

## Installation

**Requirements:** Python 3.9+, CUDA-capable GPU recommended.

```bash
git clone https://github.com/Oli21-chen/ScoliDetect_HKU_SpineSeek_AISscreening.git
cd ScoliDetect_HKU_SpineSeek_AISscreening

# Install PyTorch for your platform first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

See `[requirements.txt](requirements.txt)` for the full dependency list.

> **Note:** `models/video_encoder.py` and `models/knowledge_encoder.py` import companion packages (`video_encoders`, `knowledge_encoder`) that must be available on your `PYTHONPATH`. These will be included in the full public release; until then, copy them from the internal training environment if needed.

---

## Quick start — evaluation

> **Coming soon:** End-to-end reproduction (pretrained checkpoints, preprocessed PKL samples, and step-by-step evaluation instructions) will be published in a future release. The steps below describe the intended workflow.

This release focuses on **reproducing external-test metrics and interpretability** from a trained checkpoint.

### 1. Prepare data

Place preprocessed PKL gait patches and metadata under `data/`:

```
data/
├── test_hospital_a_pkl/    # External Hospital A cohort patches
├── test_school_a_pkl/      # External School A control patches
├── general_gait_prompts_from_report.json
├── test_indices_hospital_a.json
└── subgroup_indices.json
```

Each PKL sample should contain aligned video tensors, KKM arrays (96×238), kinematic text prompts, and Cobb-angle labels.

### 2. Configure checkpoint path

Edit `run_test.py` and set:

```python
checkpoint_path = "checkpoints/your_model/checkpoint_best.pth"
test_pkl_data_dir_override = ["./data/test_hospital_a_pkl", "./data/test_school_a_pkl"]
```

The checkpoint must embed a `config` dict (see `checkpoints/config.json` for reference fields).

### 3. Run evaluation

```bash
python run_test.py
```

**Outputs** (written to `logs/test_<checkpoint>_<timestamp>/`):


| File                        | Description                                               |
| --------------------------- | --------------------------------------------------------- |
| `test_results.txt`          | Accuracy, sensitivity, specificity, AUC, confusion matrix |
| `*_predictions.csv`         | Per-sample probabilities for ROC analysis                 |
| `*_metrics.csv`             | Summary metrics row                                       |
| `*_threshold_sweep.csv`     | Operating-point sweep                                     |
| KM attention / x-grad plots | Factor-level interpretability (Fig. 5-style)              |


Example external-test result (KVT + ViViT, Cobb threshold 11°):

```
Macro-averaged AUC (OVR): 97.25%
Best F1 threshold: t = 0.30 (sensitivity 99.7%, specificity 94.5%)
```

---

## Data capture protocol (summary)

Standardized monocular capture for deployment-aligned evaluation:

- **Camera:** mobile phone, 1080p @ 30 fps, tripod height ~1.5 m
- **Subject:** walks ~4 m toward camera; 3 trials per session
- **Clothing:** torso unobstructed; no backpacks or heavy outerwear
- **Processing:** hybrid pose estimation → adaptive normalization → KKM construction → onset + peak-anchored gait registration

Full protocol details will be included in the upcoming public release.

---

## Cohort summary

| Site       | Role                         | Participants |
| ---------- | ---------------------------- | ------------ |
| Hospital A | External test                | 299          |
| Hospital B | Train / validation           | 820          |
| School A   | Pretrain + external controls | 739          |

Total enrolled: **1,974**; after radiograph exclusions: **1,858**. External evaluation combines Hospital A (n = 299) and radiographically verified School A controls (n = 155). Patient-level splits ensure no subject overlap across partitions.

---

## Interpretability

Explanations operate on **fixed KKM factor indices**, not raw pixels:

- **Structural:** each input dimension maps to a named kinematic factor (motion / skeleton / correlation).
- **Attributional:** temporal self-attention × input gradients rank factors and gait phases for a fixed classifier—these are correlates, not causal claims.

Top-attributed factors are consistent across curve subgroups (e.g., horizontal ear position, upper-extremity angle), supporting auditable biomechanical readouts.

---

## License & intellectual property

Copyright © 2026 The University of Hong Kong and The University of Hong Kong–Shenzhen Hospital. **All rights reserved.**

**Patent pending.** ScoliDetect and related methods are protected by pending patent application(s). This repository is released under the [ScoliDetect Research and Evaluation License](LICENSE) — **not** an open-source license such as MIT or Apache.


| Use case                                           | Allowed?                           |
| -------------------------------------------------- | ---------------------------------- |
| Non-commercial academic research & reproducibility | Yes, under [LICENSE](LICENSE)      |
| Clinical screening or patient care                 | **No**                             |
| Commercial products or paid services               | **No** — separate license required |
| Patent implementation without permission           | **No**                             |


For **commercialization**, **hospital deployment**, or **patent licensing**, contact [olichen@connect.hku.hk](mailto:olichen@connect.hku.hk) or your institution's technology transfer office.

---

## Data availability

The following will be released **soon**:

- Full training and evaluation code (including `video_encoders` and `knowledge_encoder` modules)
- Pretrained model weights and example checkpoints
- De-identified kinematic summary statistics sufficient to reproduce external-test metrics
- Mobile capture protocol checklist and README figures

**Currently available:** core model definitions, evaluation utilities (`run_test.py`), and reference configs in this repository.

**Raw video** remains restricted by institutional IRB approvals. Access may be granted under data-transfer agreements—contact [olichen@connect.hku.hk](mailto:olichen@connect.hku.hk).

---

## Authors & affiliations

Dong Chen, Zonglin He, Kenneth MC Cheung

Orthopaedic Centre, The University of Hong Kong–Shenzhen Hospital, Shenzhen, China  
Department of Orthopaedics & Traumatology, Li Ka Shing Faculty of Medicine, The University of Hong Kong, Hong Kong, China

---
## Contact

**Email:** [olichen@connect.hku.hk](mailto:olichen@connect.hku.hk)

For collaboration, data access, deployment, or licensing questions, please email the address above or open a [GitHub issue](https://github.com/Oli21-chen/ScoliDetect_HKU_SpineSeek_AISscreening/issues).