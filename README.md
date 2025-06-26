# Davit-dual-attention-transformer
A PyTorch implementation of DaViT (Dual Attention Vision Transformer) with multiscale patch embedding, relative positional encoding, and dual attention blocks, supporting complete training, evaluation, and reporting.

# 🔬 DaViT: Dual Attention Vision Transformer in PyTorch

This repository implements a powerful and flexible **DaViT (Dual Attention Vision Transformer)** model using PyTorch. It combines the strengths of CNNs and transformers with dual attention, multi-scale patch embedding, and relative position encoding (RoPE), providing state-of-the-art performance on classification tasks.

## 🧠 Key Features

- 🔁 **Dual Attention**: Spatial + Channel-wise attention
- 🧩 **Multiscale Patch Embedding** with Conv Stem
- 🧠 **Transformer Blocks** with optional Conv-MLPs
- 📐 **RoPE**: Relative Positional Encoding
- 📊 Full training-validation-testing pipeline
- 📉 Auto-generated:
  - Loss curves
  - Accuracy plots (Top-1 / Top-3)
  - Confusion matrices
  - Classification reports
  - FLOPs, parameter count, inference time
- 🧪 Checkpointing and resume support

## 🛠 Requirements
pip install torch torchvision matplotlib seaborn scikit-learn tqdm ptflops timm

## 🚀 How to Use

1. Prepare dataset (ImageFolder format):
SoyMCData/
├── train/
├── val/
└── test/

2. Set hyperparameters in the script:

lr = 0.005
epochs = 5
save_dir = "./results"

3. Run training:
python Davit.py

## 📁 Output Files

| File                        | Description                       |
| --------------------------- | --------------------------------- |
| `results.json`              | All metrics and plots in one file |
| `trained_model.pth`         | Final trained model               |
| `top1_top3_comparison.png`  | Accuracy visualization            |
| `loss_curves.png`           | Training vs. Validation loss      |
| `test_confusion_matrix.png` | Confusion matrix for test set     |
| `classification_report.txt` | Precision, recall, F1 report      |
| `model_statistics.json`     | Params, FLOPs, inference time     |
| `checkpoints/*.pth`         | Checkpoints for recovery          |
