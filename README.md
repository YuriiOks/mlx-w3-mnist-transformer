# MNIST Digit Classifier using Vision Transformer (ViT) 🖼️➡️🔢🧠

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/YuriiOks/mlx-w3-mnist-transformer)
[![Contributors](https://img.shields.io/github/contributors/YuriiOks/mlx-w3-mnist-transformer?style=for-the-badge)](https://github.com/YuriiOks/mlx-w3-mnist-transformer/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/YuriiOks/mlx-w3-mnist-transformer?style=for-the-badge)](https://github.com/YuriiOks/mlx-w3-mnist-transformer/issues)
[![License](https://img.shields.io/github/license/YuriiOks/mlx-w3-mnist-transformer?style=for-the-badge)](https://github.com/YuriiOks/mlx-w3-mnist-transformer/blob/main/LICENSE)

[![MLX Institute Logo](https://ml.institute/logo.png)](http://ml.institute)

A project by **Team Backprop Bunch** 🎉 (Yurii, Amy, Guillaume, Aygun) for Week 3 of the MLX Institute Intensive Program.

## Project Overview 📋

This project implements a **Vision Transformer (ViT)** from scratch to classify MNIST digits. We will tackle this progressively in three phases:

1.  **Phase 1:** Recognize a **single** MNIST digit (28x28 image).
2.  **Phase 2:** Recognize **four** MNIST digits arranged in a **2x2 grid** (56x56 image).
3.  **Phase 3:** Recognize a **dynamic number** of digits placed randomly within a larger image, potentially including empty spaces.

The core idea is to adapt the Transformer architecture, originally designed for text, to process images by splitting them into patches and using self-attention to learn spatial relationships.

## Key Features & Modules 🛠️

*   **Configuration:** ⚙️ Centralized parameters via `config.yaml` (image size, patch size, ViT architecture dimensions, training settings per phase).
*   **Data Handling (`src/mnist_transformer/dataset.py`):** 🔄 Loads standard MNIST (`torchvision`), generates synthetic data for Phase 2 (2x2 grid) and potentially Phase 3 (dynamic layout). Includes necessary data transformations.
*   **ViT Modules (`src/mnist_transformer/modules.py`):** 🧩 Defines core ViT components: `PatchEmbedding`, `Attention` (Multi-Head Self-Attention), `MLPBlock`, `TransformerEncoderBlock`.
*   **Model Architecture (`src/mnist_transformer/model.py`):** 🏗️ Defines the main `VisionTransformer` class, assembling the modules and adapting for different phases.
*   **Training (`src/mnist_transformer/trainer.py`):** 🏋️‍♀️ Implements the training loop using standard Cross-Entropy Loss (adapting for multi-output in later phases), AdamW optimizer, and potentially LR scheduling/warmup.
*   **Utilities (`utils/`):** 🔧 Shared functions for logging, device (CPU/MPS/CUDA) setup, config loading, artifact saving (plots, losses).
*   **Experiment Tracking (`wandb`):** 📊 Integrated for logging hyperparameters, metrics (loss, accuracy), and saving model artifacts.
*   **Main Script (`scripts/train_mnist_vit.py`):** 🚀 Orchestrates loading, setup, training (selecting the phase), and saving process.

## Directory Structure 📁

A detailed breakdown is available in `docs/STRUCTURE.MD`.

## Setup 💻

1.  **Clone the Repository:** 📥
    ```bash
    # TODO: Update with correct Week 3 Repo URL
    git clone https://github.com/YOUR_ORG/YOUR_W3_REPO_NAME.git
    cd YOUR_W3_REPO_NAME
    ```
2.  **Create & Activate Virtual Environment:** 🐍
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `\.venv\Scripts\activate`
    ```
3.  **Install Dependencies:** 📦 (Mainly `torch`, `torchvision`, `wandb`, `pyyaml`, `tqdm`. `timm` might be useful later).
    ```bash
    pip install -r requirements.txt
    ```
4.  **Weights & Biases Login:** 🔑
    ```bash
    wandb login
    ```
5.  **MNIST Data:** 🗃️ The MNIST dataset will be downloaded automatically by `torchvision` into the `data/` directory the first time you run the training script for Phase 1.

## Usage 🚦

1.  **Configuration:** ⚙️ Review and adjust parameters in `config.yaml` for the target phase (e.g., `model` dimensions, `training.phaseX_...` settings like epochs, batch size, learning rate).
2.  **Run Training:** 🏃‍♂️ Execute the main training script from the project root directory, potentially specifying the phase:
    ```bash
    # Example: Train Phase 1 using config defaults
    python scripts/train_mnist_vit.py --phase 1

    # Example: Train Phase 1 overriding epochs and LR
    python scripts/train_mnist_vit.py --phase 1 --epochs 10 --lr 5e-4
    ```
    *   Training progress (`tqdm` bars) shown in console.
    *   Metrics logged to Weights & Biases (link provided).
    *   Model artifacts saved locally under `models/mnist_vit/<W&B_RUN_NAME>/`.

Use `python scripts/train_mnist_vit.py --help` for all command-line options (once arguments are added).

## Phased Plan & Next Steps 🔮

*   ✅ **Phase 1: Single Digit Recognition:** Implement basic ViT, train on standard MNIST, achieve high accuracy.
*   ➡️ **Phase 2: 2x2 Grid Recognition:** Implement synthetic data generation, adapt model output and loss function for 4 digits.
*   ➡️ **Phase 3: Dynamic Layout Recognition:** Implement dynamic data generation, adapt model/loss for variable number of digits + empty detection.
*   ✨ **Apply Training Tricks:** Incorporate techniques like strong data augmentation, learning rate warmup/cosine decay, AdamW, potentially Stochastic Depth or Label Smoothing to improve robustness and performance, especially for later phases.
*   📊 **Evaluation:** Implement robust evaluation in `scripts/evaluate.py` (overall accuracy, per-digit accuracy).
*   **(Stretch)** Explore different ViT configurations or attention mechanisms.
