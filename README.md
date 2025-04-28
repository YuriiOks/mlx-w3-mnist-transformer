# MNIST Digit Classifier using Vision Transformer (ViT) 🖼️➡️🔢🧠

[![MLX Institute Logo](https://ml.institute/logo.png)](http://ml.institute)

A project by **Team Backprop Bunch** 🎉 (Yurii, Amy, Guillaume, Aygun) for Week 3 of the MLX Institute Intensive Program.

## Project Overview 📋

This project implements a **Vision Transformer (ViT)** from scratch to classify MNIST digits. We are tackling this progressively in three phases:

1.  ✅ **Phase 1:** Recognize a **single** MNIST digit (28x28 image).
2.  ✅ **Phase 2:** Recognize **four** MNIST digits arranged in a **2x2 grid** (56x56 image).
3.  ➡️ **Phase 3:** Recognize a **dynamic number** of digits (0-N) placed randomly within a larger image (e.g., 64x64), including identifying empty spaces.

The core idea is to adapt the Transformer architecture (using self-attention) to process images by splitting them into patches, removing the need for traditional convolutions.

## Key Features & Modules 🛠️

*   **Configuration:** ⚙️ Centralized parameters via `config.yaml` (image size, patch size, ViT architecture dimensions, training settings per phase).
*   **Data Handling (`src/mnist_transformer/dataset.py`):** 🔄 Loads standard MNIST (`torchvision`), generates synthetic 2x2 grid data for Phase 2. Includes data transformations. (Phase 3 generation pending).
*   **ViT Modules (`src/mnist_transformer/modules.py`):** 🧩 Defines core ViT components from scratch: `PatchEmbedding`, `Attention` (Multi-Head Self-Attention using `AttentionHead`), `MLPBlock`, `TransformerEncoderBlock`.
*   **Model Architecture (`src/mnist_transformer/model.py`):** 🏗️ Defines the main `VisionTransformer` class, assembling the modules and adapting its input/output structure for Phase 1 and Phase 2.
*   **Training (`src/mnist_transformer/trainer.py`):** 🏋️‍♀️ Implements the training and evaluation loops using Cross-Entropy Loss (adapted for multi-output in Phase 2), AdamW optimizer, and basic LR scheduling.
*   **Utilities (`utils/`):** 🔧 Shared functions for logging (with color!), device (CPU/MPS/CUDA) setup, config loading, artifact saving.
*   **Experiment Tracking (`wandb`):** 📊 Integrated for logging hyperparameters, metrics (loss, accuracy - batch and epoch), model gradients, and saving final artifacts.
*   **Main Script (`scripts/train_mnist_vit.py`):** 🚀 Orchestrates loading, setup, training (selecting the phase via `--phase`), evaluation, and saving process.

## Current Status & Results 📈📉

*   ✅ **Phase 1 Complete:** Successfully trained a small ViT (Depth=4, Embed=64, Heads=4) on single MNIST digits, achieving **~98.3%** validation accuracy after 15 epochs.
*   ✅ **Phase 2 Complete:** Successfully adapted the model and training logic for the 2x2 grid task. Trained for 20 epochs, achieving **~95.7%** validation accuracy (accuracy calculated per digit).
*   ✅ **Core Modules Built:** Implemented ViT components including Multi-Head Self-Attention from scratch.
*   ✅ **Pipeline Functional:** End-to-end training and evaluation pipeline is working, including W&B integration and artifact saving.

## Directory Structure 📁

A detailed breakdown is available in `docs/STRUCTURE.MD`.

## Setup 💻

1.  **Clone the Repository:** 📥
    ```bash
    # TODO: Update with correct Week 3 Repo URL
    git clone https://github.com/YOUR_ORG/mlx-w3-mnist-transformer.git
    cd mlx-w3-mnist-transformer
    ```
2.  **Create & Activate Virtual Environment:** 🐍
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `\.venv\Scripts\activate`
    ```
3.  **Install Dependencies:** 📦
    ```bash
    pip install -r requirements.txt
    # You might need: pip install torchinfo matplotlib seaborn pandas # If not already included
    ```
4.  **Weights & Biases Login:** 🔑
    ```bash
    wandb login
    ```
5.  **MNIST Data:** 🗃️ The MNIST dataset will be downloaded automatically by `torchvision` into the `data/` directory the first time you run the training script.

## Usage 🚦

1.  **Configuration:** ⚙️ Review `config.yaml`. Phase 1 and 2 parameters are set. Phase 3 parameters are placeholders.
2.  **Run Training:** 🏃‍♂️ Execute the main training script from the project root directory:
    ```bash
    # Train Phase 1 (Example: 15 epochs, default LR from config)
    python scripts/train_mnist_vit.py --phase 1 --epochs 15

    # Train Phase 2 (Example: 20 epochs, default LR from config)
    python scripts/train_mnist_vit.py --phase 2 --epochs 20

    # Override LR for a phase
    # python scripts/train_mnist_vit.py --phase 1 --epochs 10 --lr 5e-4
    ```
    *   Training progress (`tqdm` bars) shown in console.
    *   Metrics logged to Weights & Biases (link provided).
    *   Model artifacts saved locally under `models/mnist_vit/<W&B_RUN_NAME>/`.

Use `python scripts/train_mnist_vit.py --help` for command-line options.

## Phased Plan & Next Steps 🔮

*   ✅ **Phase 1: Single Digit Recognition:** DONE.
*   ✅ **Phase 2: 2x2 Grid Recognition:** DONE.
*   ➡️ **Phase 3: Dynamic Layout Recognition:** Implement dynamic data generation (`dataset.py`), adapt model output (`model.py`), adapt loss/accuracy calculation (`trainer.py`), update main script & config. Use Grid Classification approach with "Empty" class (see `notebooks/03_phase3_design_ideas.ipynb`).
*   ✨ **Apply Training Tricks:** Consider adding LR Warmup, more sophisticated LR scheduling, stronger augmentation, or regularization like Stochastic Depth if needed for Phase 3 or further refinement.
*   📊 **Evaluation Script:** Fully implement `scripts/evaluate.py` for standalone evaluation of saved models.