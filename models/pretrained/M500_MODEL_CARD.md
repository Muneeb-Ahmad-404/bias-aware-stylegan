# 🧬 M500 — Pretrained Melanoma Generation Model

> **Role:** Baseline generative model for skin-tone representation and bias mitigation research.

---

## 📌 Model Overview

M500 is a pretrained **StyleGAN2-ADA** checkpoint provided as part of the CAN5600 & CAN2000 research resources.

In this project, it serves as the starting point for investigating melanoma image generation and the representation of underrepresented skin tones in synthetic dermatological datasets.

| Property | Details |
|:--|:--|
| **Model** | M500 |
| **Architecture** | StyleGAN2-ADA |
| **Model Type** | Generative Adversarial Network |
| **Task** | Melanoma Image Generation |
| **Checkpoint** | `M500.pkl` |
| **Format** | PyTorch Pickle |
| **Original Author** | Seung Seog Han |
| **Release** | 21 January 2023 |
| **Source** | Figshare |
| **Project Role** | Pretrained Baseline |
| **Storage** | DagsHub + DVC |

---

## 🔗 Original Source

**CAN5600 & CAN2000 — GAN Model Weights & Synthetic Images**

- **Author:** Seung Seog Han
- **Platform:** Figshare
- **DOI:** [10.6084/m9.figshare.21507189](https://doi.org/10.6084/m9.figshare.21507189)

The original resource provides pretrained GAN checkpoints and synthetic skin-lesion image datasets.

---

## 🎯 Purpose in This Research

The primary objective is to investigate whether a pretrained melanoma generation model can be adapted to improve the representation of underrepresented skin tones.

M500 provides the initial baseline against which subsequent fine-tuning experiments will be compared.

### Research Workflow

1. **Baseline Inference** — Load M500 and generate initial melanoma samples.
2. **Baseline Evaluation** — Assess image quality, diversity and visual characteristics.
3. **Dataset Preparation** — Curate melanoma images with reliable skin-tone annotations.
4. **Fine-tuning** — Investigate adapting the pretrained model using the curated dataset.
5. **Comparative Evaluation** — Compare baseline and fine-tuned models.
6. **Bias Analysis** — Examine skin-tone representation and potential improvements.

---

## 🏗️ Architecture

M500 is based on **StyleGAN2-ADA**, a generative adversarial network architecture incorporating Adaptive Discriminator Augmentation.

StyleGAN2-ADA is designed to improve GAN training stability when working with limited training data through adaptive augmentation.

The pretrained checkpoint is distributed as a serialized PyTorch pickle file.

---

## 📂 Checkpoint Management

The original checkpoint is tracked using **DVC (Data Version Control)**, with DagsHub configured as the remote storage provider.

```text
models/
└── pretrained/
    ├── M500.pkl.dvc
    ├── M500_MODEL_CARD.md
    └── .gitignore
```

| Component | Purpose |
|:--|:--|
| `M500.pkl` | Original pretrained checkpoint |
| `M500.pkl.dvc` | DVC tracking pointer |
| `.gitignore` | Excludes the locally stored model binary |
| `M500_MODEL_CARD.md` | Model documentation |

**Retrieve the checkpoint:**

```bash
dvc pull models/pretrained/M500.pkl.dvc
```

> The actual checkpoint is stored in DVC remote storage, while Git tracks its corresponding metadata pointer.

---

## 📊 Evaluation Strategy

The pretrained checkpoint will be evaluated before any modifications to establish a baseline.

| Evaluation | Purpose |
|:--|:--|
| **FID** | Measure similarity between real and generated image distributions |
| **Visual Quality** | Inspect realism and generation artifacts |
| **Image Diversity** | Examine variation across generated samples |
| **Skin-tone Representation** | Assess representation across skin-tone groups |
| **Lesion Characteristics** | Inspect preservation of plausible melanoma-related features |

*Evaluation results will be added after baseline inference and experimentation.*

---

## ⚠️ Limitations & Considerations

- The original checkpoint was not specifically developed for this project's skin-tone fairness objective.
- Its ability to generate representative images across different skin tones has not yet been established.
- Fine-tuning may affect image quality, diversity and model stability.
- Synthetic images are not clinically validated melanoma images.
- Generated samples require appropriate quality assessment before being considered for dataset augmentation.

---

## 🧪 Experiment Tracking

All experiments derived from M500 will be documented separately to preserve reproducibility and model lineage.

Planned records include:

- Training configurations and hyperparameters
- Dataset versions and preprocessing details
- Checkpoint versions
- FID and other evaluation metrics
- Baseline versus fine-tuned comparisons

**Experiment tracking:** MLflow  
**Model and dataset versioning:** DVC + DagsHub

---

## 📚 References

1. Han, S. S. (2023). *CAN5600 & CAN2000 — GAN Model Weights & Synthetic Images.* Figshare. [https://doi.org/10.6084/m9.figshare.21507189](https://doi.org/10.6084/m9.figshare.21507189)

2. Karras, T., et al. (2020). *Training Generative Adversarial Networks with Limited Data.* NeurIPS 2020. [Paper](https://arxiv.org/abs/2006.06676)

---

<sub>Maintained as part of the Bias-Aware StyleGAN research project. This documentation describes the original pretrained checkpoint and its intended role in the research workflow.</sub>