# GNR638 — Coding Assignment 2
## Pre-trained CNN Representation Transfer and Robustness Analysis

> **Course:** GNR638 — Machine Learning for Remote Sensing
> **Dataset:** Aerial Images Dataset (AID) — 30-class land-use classification
> **Hardware:** Lenovo LOQ · RTX 4060 8 GB · 16 GB RAM · Intel i7-13650HX · CUDA

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Experiments at a Glance](#experiments-at-a-glance)
- [Setup & Requirements](#setup--requirements)
- [Dataset Preparation](#dataset-preparation)
- [Running the Notebook](#running-the-notebook)
- [Runtime Summary](#runtime-summary)
- [Results Summary](#results-summary)
  - [Scenario 4.1 — Linear Probe](#scenario-41--linear-probe)
  - [Scenario 4.2 — Fine-Tuning Strategies](#scenario-42--fine-tuning-strategies)
  - [Scenario 4.3 — Few-Shot Learning](#scenario-43--few-shot-learning)
  - [Scenario 4.4 — Corruption Robustness](#scenario-44--corruption-robustness)
  - [Scenario 4.5 — Layer-Wise Feature Probing](#scenario-45--layer-wise-feature-probing)
- [Output Figures](#output-figures)
- [Report](#report)
- [Key Findings](#key-findings)
- [Reproducibility](#reproducibility)

---

## Overview

This repository contains the full experimental pipeline for GNR638 Assignment 2, which benchmarks **three ImageNet pre-trained CNN backbones** across five transfer-learning and robustness scenarios on the 30-class **AID (Aerial Images Dataset)**:

| Backbone | Params | MACs | Architecture style |
|---|---|---|---|
| **ResNet-50** | 23.6 M | 4.11 G | Classic residual blocks |
| **EfficientNet-B0** | 4.0 M | 0.40 G | Compound-scaled mobile inverted bottleneck |
| **ConvNeXt-Tiny** | 27.8 M | 4.47 G | Modernised pure-CNN (large-kernel, LayerNorm) |

All models are loaded from [`timm`](https://github.com/huggingface/pytorch-image-models) with ImageNet-1k pre-trained weights.

---

## Repository Structure

```
GNR638-A2/
│
├── A2_notebook.ipynb          # Main experiment notebook (all 5 scenarios)
│
├── output/                    # Generated plots and visualisations
│   │
│   ├── # ── Scenario 4.1 / 4.2 Training Curves ──────────────────────
│   ├── resnet50_linear_probe_curves.png
│   ├── resnet50_full_ft_curves.png
│   ├── resnet50_last_block_curves.png
│   ├── resnet50_selective_20_curves.png
│   ├── efficientnet_b0_linear_probe_curves.png
│   ├── efficientnet_b0_full_ft_curves.png
│   ├── efficientnet_b0_last_block_curves.png
│   ├── efficientnet_b0_selective_20_curves.png
│   ├── convnext_tiny_linear_probe_curves.png
│   ├── convnext_tiny_full_ft_curves.png
│   ├── convnext_tiny_last_block_curves.png
│   ├── convnext_tiny_selective_20_curves.png
│   │
│   ├── # ── Scenario 4.1 Confusion Matrices & PCA ────────────────────
│   ├── resnet50_linear_probe_confusion.png
│   ├── resnet50_linear_probe_pca.png
│   ├── efficientnet_b0_linear_probe_confusion.png
│   ├── efficientnet_b0_linear_probe_pca.png
│   ├── convnext_tiny_linear_probe_confusion.png
│   ├── convnext_tiny_linear_probe_pca.png
│   │
│   ├── # ── Scenario 4.2 Gradient Norm Heatmaps & Lines ──────────────
│   ├── resnet50_{strategy}_grad_norm_heatmap.png   # strategy ∈ {linear_probe, last_block, selective_20, full_ft}
│   ├── resnet50_{strategy}_grad_norm_lines.png
│   ├── efficientnet_b0_{strategy}_grad_norm_heatmap.png
│   ├── efficientnet_b0_{strategy}_grad_norm_lines.png
│   ├── convnext_tiny_{strategy}_grad_norm_heatmap.png
│   ├── convnext_tiny_{strategy}_grad_norm_lines.png
│   │
│   ├── # ── Scenario 4.2 Convergence & Summary ───────────────────────
│   ├── resnet50_4_2_convergence.png
│   ├── efficientnet_b0_4_2_convergence.png
│   ├── convnext_tiny_4_2_convergence.png
│   ├── 4_2_accuracy_vs_unfrozen.png
│   │
│   └── # ── Scenario 4.5 Layer-Wise Probing ──────────────────────────
│       ├── scenario_4_5_probe_accuracy.png
│       ├── scenario_4_5_feature_norms.png
│       ├── scenario_4_5_pca_{model}_{depth}.png   # model ∈ {resnet50, efficientnet_b0, convnext_tiny}
│       │                                           # depth ∈ {early, middle, final}  → 9 PCA plots
|  ├── resnet50_linear_probe_best.pth
|  ├── resnet50_last_block_best.pth
|  ├── resnet50_selective_20_best.pth
|  ├── resnet50_full_ft_best.pth
|  ├── resnet50_fewshot_100pct_best.pth
|  ├── resnet50_fewshot_20pct_best.pth
|  ├── resnet50_fewshot_5pct_best.pth
|  ├── efficientnet_b0_linear_probe_best.pth
|  ├── efficientnet_b0_last_block_best.pth
|  ├── efficientnet_b0_selective_20_best.pth
|  ├── efficientnet_b0_full_ft_best.pth
|  ├── efficientnet_b0_fewshot_100pct_best.pth
|  ├── efficientnet_b0_fewshot_20pct_best.pth
|  ├── efficientnet_b0_fewshot_5pct_best.pth
|  ├── convnext_tiny_linear_probe_best.pth
|  ├── convnext_tiny_last_block_best.pth
|  ├── convnext_tiny_selective_20_best.pth
|  ├── convnext_tiny_full_ft_best.pth
|  ├── convnext_tiny_fewshot_100pct_best.pth
|  ├── convnext_tiny_fewshot_20pct_best.pth
|  └── convnext_tiny_fewshot_5pct_best.pth
│
├── report/
│   └── GNR638_A2_Report.pdf   # report
│
└── README.md                  # This file
```

> **Note:** The `output/` directory holds **57 PNG files** in total. All paths in the LaTeX report use `figures/` as an alias for `output/` — symlink or copy the folder accordingly before compiling.

---

## Experiments at a Glance

| # | Scenario | Description |
|---|---|---|
| 4.1 | **Linear Probe** | Frozen backbone + trained head only |
| 4.2 | **Fine-Tuning Strategies** | LP vs Last-Block vs Selective-20% vs Full Fine-Tune |
| 4.3 | **Few-Shot Learning** | Full fine-tune at 100% / 20% / 5% training data |
| 4.4 | **Corruption Robustness** | Gaussian noise (σ=0.05/0.10/0.20), motion blur, brightness shift |
| 4.5 | **Layer-Wise Probing** | Linear probes on early / middle / final intermediate features |

---

## Setup & Requirements

### Python Environment

```bash
# Recommended: Python 3.10+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm
pip install scikit-learn matplotlib seaborn
pip install fvcore          # for MACs/FLOPs computation
pip install numpy pandas
pip install jupyter
```

### Verified Versions

| Package | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.x (CUDA 12.x) |
| timm | 0.9.x |
| scikit-learn | 1.3.x |
| matplotlib | 3.7.x |

---

## Dataset Preparation

1. Download the **AID (Aerial Images Dataset)** from the [official source](https://captain-whu.github.io/AID/) or Kaggle mirrors.
2. Extract and structure the directory as a standard `ImageFolder` layout:

```
train_data/
├── airport/
│   ├── airport_001.jpg
│   └── ...
├── beach/
│   └── ...
├── ...
└── viaduct/      # 30 classes total
```

3. Update the `data_root` key in `CONFIG` (Section 2 of the notebook):

```python
CONFIG = {
    "data_root": "./train_data",   # <-- update to your local path
    ...
}
```

### Dataset Statistics

| Split | Images | Classes |
|---|---|---|
| Train (80%) | 5,594 | 30 |
| Validation (20%) | 1,399 | 30 |
| **Total** | **6,993** | **30** |

---

## Running the Notebook

```bash
# Clone the repo
git clone https://github.com/<your-username>/gnr638-a2.git
cd gnr638-a2

# Launch Jupyter
jupyter notebook A2_notebook.ipynb
```

### Execution Order

The notebook is structured into **numbered sections** and should be run **top to bottom**:

| Section | Content | Cell |
|---|---|---|
| 0 | Imports & dependency checks | Cell 0 |
| 1 | Dataset & augmentation utilities | Cell 1 |
| 2 | Global config & seed | Cell 2 |
| 3–11 | Model loaders, trainers, plotters, helpers | Cells 3–11 |
| 12 | **Main experiment runner** (Scenarios 4.1, 4.2-baseline, 4.3, 4.4) | Cell 12 |
| 13 | Entry point (`if __name__ == "__main__"`) | Cell 13 |
| 14 | **Scenario 4.2 comprehensive** (all 4 strategies × 3 models) | Cell 14 |
| 15 | **Scenario 4.5** layer-wise feature probing | Cell 15 |

> Cells 12–15 are **independent runs** that each reproduce all figures. Run them sequentially; re-running a cell will overwrite its output PNGs.

---

## Runtime Summary

| Block | Scenarios | Approx. Time |
|---|---|---|
| Cell 12 (main) | 4.1, 4.2-baseline, 4.3, 4.4 | ~250 minutes |
| Cell 14 (comprehensive) | 4.2 all strategies | ~150 minutes |
| Cell 15 | 4.5 layer probing | ~5 minutes |
| **Total** | | **~405 minutes** |

> Times measured on RTX 4060 8 GB with batch size 16.

---

## Results Summary

### Shared Hyperparameters

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Train / val split | 80 / 20 |
| Loss | CrossEntropyLoss |
| Optimiser | Adam (lr=1e-3, wd=1e-4) |
| LR schedule | CosineAnnealingLR |
| Batch size | 16 |
| Max epochs (full data) | 30 |
| Max epochs (few-shot) | 20 |
| Input size | 224 × 224 |

---

### Scenario 4.1 — Linear Probe

Only the final classification head (30 units) is trained; backbone is fully frozen.

| Model | Trainable Params | Best Val Acc |
|---|---|---|
| ResNet-50 | 61,470 (0.26%) | 87.85% |
| EfficientNet-B0 | 38,430 (0.95%) | 85.56% |
| **ConvNeXt-Tiny** | **23,070 (0.08%)** | **94.21%** |

---

### Scenario 4.2 — Fine-Tuning Strategies

Four strategies compared per model. LR for full fine-tune reduced to `1e-4`.

| Model | Linear Probe | Last Block | Selective 20% | Full Fine-Tune |
|---|---|---|---|---|
| ResNet-50 | 88.13% | **96.64%** | 95.64% | 95.64% |
| EfficientNet-B0 | 85.56% | 95.64% | 95.43% | 96.43% |
| ConvNeXt-Tiny | 94.21% | 96.28% | 95.85% | **96.93%** |

> **Key insight:** ResNet-50 Last-Block (96.64%) *outperforms* ResNet-50 Full Fine-Tune (95.64%), showing that aggressive full unfreezing can hurt generalisation.

---

### Scenario 4.3 — Few-Shot Learning

Full fine-tuning applied to 100% / 20% / 5% of training data.

| Model | 100% | 20% | 5% | Δ (100%→5%) |
|---|---|---|---|---|
| ResNet-50 | 96.57% | 91.42% | 78.98% | 0.182 |
| EfficientNet-B0 | 96.71% | 91.71% | 75.20% | 0.222 |
| ConvNeXt-Tiny | 75.41% | 38.17% | **4.29%** | **0.943** |

> **Critical failure:** ConvNeXt-Tiny collapses to near-random (4.29%) at 5% data, while its *frozen* linear probe achieves 94.21%. Full fine-tuning a large-capacity model under extreme data scarcity causes catastrophic divergence.

---

### Scenario 4.4 — Corruption Robustness

Evaluated on fully fine-tuned models. Clean accuracies: ResNet-50 96.14%, EfficientNet-B0 96.14%, ConvNeXt-Tiny 96.78%.

| Corruption | ResNet-50 | EfficientNet-B0 | ConvNeXt-Tiny |
|---|---|---|---|
| Gaussian σ=0.05 | 87.40% | 69.98% | **93.81%** |
| Gaussian σ=0.10 | 54.94% | 5.09% | **67.24%** |
| Gaussian σ=0.20 | 5.32% | 3.23% | **17.72%** |
| Motion Blur | **71.29%** | 60.62% | 70.44% |
| Brightness +0.2 | 98.70% | **99.03%** | 99.23% |

> **ConvNeXt-Tiny is most noise-robust** (large kernels + LayerNorm). EfficientNet-B0 is most noise-fragile (channel-independent depthwise ops). All models *improve* under brightness shift — a sign that AID images are slightly underexposed relative to ImageNet.

---

### Scenario 4.5 — Layer-Wise Feature Probing

Logistic regression probes on frozen intermediate features (900-image fixed subset for PCA).

| Model | Early Layer | Early Acc | Middle Layer | Middle Acc | Final Layer | Final Acc |
|---|---|---|---|---|---|---|
| ResNet-50 | layer1 (256-D) | 83.20% | layer3 (1024-D) | **94.28%** | layer4 (2048-D) | 91.14% |
| EfficientNet-B0 | blocks.1 (24-D) | 66.62% | blocks.4 (112-D) | 89.42% | blocks.6 (320-D) | **90.64%** |
| ConvNeXt-Tiny | stages.0 (96-D) | 80.20% | stages.2 (384-D) | **95.00%** | stages.3 (768-D) | 93.42% |

> **Middle layers are the most transferable.** Both ResNet-50 and ConvNeXt-Tiny show a *decrease* from middle to final layer — the final layers are over-specialised to ImageNet categories, while middle layers capture richer domain-agnostic semantics.

---

## Output Figures

All 57 figures are saved to `output/`. A quick reference:

| File Pattern | Description |
|---|---|
| `{model}_linear_probe_curves.png` | Train/val loss & accuracy — linear probe (×3 models) |
| `{model}_linear_probe_confusion.png` | 30×30 confusion matrix — linear probe (×3 models) |
| `{model}_linear_probe_pca.png` | 2-D PCA of frozen features — linear probe (×3 models) |
| `{model}_{strategy}_curves.png` | Train/val curves per strategy (×3 models × 4 strategies) |
| `{model}_{strategy}_grad_norm_heatmap.png` | Gradient norm heatmap layer×epoch (×3 × 4) |
| `{model}_{strategy}_grad_norm_lines.png` | Gradient norm line plots (×3 × 4) |
| `{model}_4_2_convergence.png` | Strategy convergence overlay (×3 models) |
| `4_2_accuracy_vs_unfrozen.png` | Accuracy vs % unfrozen — all models on one plot |
| `scenario_4_5_probe_accuracy.png` | Probe accuracy vs depth — all 3 models |
| `scenario_4_5_feature_norms.png` | Mean L2 norm per layer & model |
| `scenario_4_5_pca_{model}_{depth}.png` | PCA scatter per model×depth (9 plots) |

---

## Report

The `report/` directory contains:

| File | Description |
|---|---|
| `GNR638_A2_Report.pdf` | Pre-compiled 17-page PDF |

### Report Structure

| Section | Content |
|---|---|
| 1 | Introduction & Experimental Setup |
| 2 | Computational Budget & Efficiency |
| 3 | Scenario 4.1 — Linear Probe |
| 4 | Scenario 4.2 — Fine-Tuning Strategies |
| 5 | Scenario 4.3 — Few-Shot Learning |
| 6 | Scenario 4.4 — Corruption Robustness |
| 7 | Scenario 4.5 — Layer-Wise Probing |
| 8 | Key Intellectual Findings (5 grading questions + unexpected finding + failure case) |
| Appendix | Additional training curves & gradient norm lines |

---

## Key Findings

1. **Best transfer (full data):** ConvNeXt-Tiny — highest LP accuracy (94.21%) and full fine-tune (96.93%), driven by its stronger pre-training and large-kernel design.
2. **Best efficiency:** EfficientNet-B0 — 96%+ accuracy with only 4M parameters (~24% acc/M params).
3. **Most transferable layers:** Middle layers of all models outperform or match final layers for domain transfer probing.
4. **Most robust to noise:** ConvNeXt-Tiny (large kernels + LayerNorm act as spatial smoothers).
5. **Most fragile to noise:** EfficientNet-B0 collapses at σ=0.10 (5.09% accuracy).
6. **Few-shot winner:** ResNet-50 retains 78.98% at 5% data; ConvNeXt-Tiny crashes to 4.29%.
7. **Fine-tuning can hurt:** ResNet-50 Last-Block > ResNet-50 Full Fine-Tune — aggressive unfreezing of early layers degrades generalisation.
8. **Unexpected:** Brightness +0.2 *improves* accuracy for all models — AID images are slightly underexposed relative to ImageNet's distribution.

---

## Reproducibility

All experiments use a fixed global seed of **42**, applied to:

```python
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
os.environ['PYTHONHASHSEED']       = '42'
```

Given identical hardware and software versions, all results should reproduce exactly. Minor floating-point non-determinism may occur across different GPU architectures due to CUDA kernel implementations.
We used the help of AI tools to generate some parts of Readme and neat print statements in the .ipynb file.
---

*GNR638 Assignment 2 — Aerial Image Transfer Learning Study*
