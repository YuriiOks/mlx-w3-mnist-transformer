# MNIST Digit Classifier (Transformer) - Project Structure

This document outlines the detailed directory structure and the purpose of key files for the Week 3 MNIST Vision Transformer project by **Team Backprop Bunch**. The project aims to build and train a Vision Transformer model progressively, starting with single-digit recognition and scaling up to more complex scenarios.

## Directory Tree 🌳

```
📁 mlx-w3-mnist-transformer/
├── 📁 app/
│   └── 📄 __init__.py
├── 📄 config.yaml           # ⚙️ NEW Configuration for ViT (phases, model, training)
├── 📁 data/                 # 📊 MNIST dataset (auto-downloaded)
│   └── MNIST/
│       ├── processed/
│       └── raw/
├── 🐳 Dockerfile
├── 📁 docs/                 # 📄 Project documentation
│   ├── 📄 05_QKV.md         # (Example Transformer Note)
│   ├── (...)             # (Other relevant general notes: 06-12)
│   └── 📄 STRUCTURE.MD      # 👈 This file
├── 📁 logs/                 # 📝 Runtime logs (e.g., mnist_vit_train.log)
├── 📁 models/               # 🧠 Saved model artifacts
│   └── 📁 mnist_vit/        #    Subdir for this project's models
│       └── 📁 phase1_run_XYZ/ #    Example run output
│           ├── 📄 model_final.pth
│           ├── 📄 training_losses.json
│           └── 📄 loss_plot.png
├── 📁 notebooks/            # 📓 Jupyter notebooks for exploration
│   └── 📄 01_mnist_vit_exploration.ipynb
├── 📄 .gitignore
├── 📄 LICENSE
├── 📄 README.md             # 👋 Updated project overview
├── 📄 requirements.txt      # 📦 Python dependencies (torch, torchvision, timm?, wandb)
├── 📁 scripts/              # ▶️ Runnable Python scripts
│   ├── 📄 train_mnist_vit.py # 👈 Main training script (handles phases)
│   ├── 📄 evaluate.py       # 👈 Evaluation script
│   └── 📄 generate_project_doc.py # (Optional utility)
├── 🐍 src/                  # 🐍 Core Python source code
│   ├── 📄 __init__.py
│   └── 📁 mnist_transformer/ # 👈 Module for ViT logic
│       ├── 📄 __init__.py
│       ├── 📄 dataset.py    # 👈 Handles MNIST (P1), 2x2 grid (P2), dynamic (P3)
│       ├── 📄 modules.py    # 👈 ViT building blocks (PatchEmbed, Attention, MLP, EncoderBlock)
│       ├── 📄 model.py      # 👈 Defines the VisionTransformer class (adapts for P1/P2/P3)
│       ├── 📄 trainer.py    # 👈 Training loop logic (adapts loss/steps for P1/P2/P3)
│       └── 📄 evaluation.py # 👈 (Optional) Metrics calculation
│
└── 🛠️ utils/                # 🛠️ Shared utility modules
    ├── 📄 __init__.py
    ├── 📄 device_setup.py
    ├── 📄 logging.py
    └── 📄 run_utils.py
```

## Detailed Directory & File Descriptions 📜

*   **Root Directory (`mlx-w3-mnist-transformer/`)**: Contains configuration, main scripts, documentation, and subdirectories for source code, data, models, etc. for the MNIST Vision Transformer project.
    *   📄 **`.gitignore`**: Specifies untracked files/directories (e.g., `.venv`, `logs/`, `models/`, `wandb/`, `__pycache__/`, `data/MNIST/`).
    *   📄 **`config.yaml`** ⚙️: Central configuration file for the ViT project. Defines paths, dataset parameters (image size, patch size, classes per phase), model hyperparameters (embedding dim, depth, heads, MLP ratio), training settings (epochs, batch size, learning rate, optimizer, warmup per phase), and logging settings.
    *   📄 **`Dockerfile`** 🐳: (Placeholder/Optional) Instructions for building a Docker container image, enabling deployment or reproducible environments.
    *   📄 **`LICENSE`**: Contains the project's license information.
    *   📄 **`README.md`** 👋: Main project overview. Describes the MNIST ViT goal (progressive phases), setup instructions, usage for training/evaluation, and team members. *(Needs updating for Week 3)*.
    *   📄 **`requirements.txt`** 📦: Lists Python dependencies (`torch`, `torchvision`, `wandb`, `PyYAML`, `tqdm`, potentially `timm`, `matplotlib`, `numpy`, `pandas`, `scikit-learn`).

*   📁 **`app/`**: (Placeholder) Contains code for a potential future user-facing application or API service.
    *   📄 **`__init__.py`**: Marks `app` as a Python package.

*   📁 **`data/`** 📊: Default location for datasets.
    *   📁 **`MNIST/`**: Directory where `torchvision.datasets.MNIST` will automatically download and store the MNIST dataset files. (Should be gitignored).

*   📁 **`docs/`** 📄: Contains project documentation, including conceptual notes on Transformers and ViT.
    *   📄 `0X_*.md`: Markdown files explaining core concepts (Attention, ViT architecture, Training Tricks, etc.).
    *   📄 **`STRUCTURE.MD`**: This file.

*   📁 **`logs/`** 📝: Stores runtime log files generated by the training/evaluation scripts (configured in `utils/logging.py`). (Gitignored).

*   📁 **`models/`** 🧠: Root directory for storing saved model artifacts. (Gitignored).
    *   📁 **`mnist_vit/`**: Contains subdirectories specific to trained MNIST Vision Transformer runs.
        *   📁 **`<RUN_NAME>/`** (e.g., `phase1_run_XYZ/`): A subdirectory created for each training run, named based on phase, hyperparameters or W&B run name.
            *   📄 **`model_final.pth`**: Saved state dictionary of the trained ViT model.
            *   📄 **`training_losses.json`**, 📄 **`loss_plot.png`**: Saved training metrics and visualizations.

*   📁 **`notebooks/`** 📓: Location for Jupyter notebooks.
    *   📄 **`01_mnist_vit_exploration.ipynb`**: Notebook for initial MNIST data loading, visualization, implementing/testing patching logic, and exploring basic ViT concepts.
    *   📄 `(Optional) 02_phase2_data_generation.ipynb`, `03_phase3_design_ideas.ipynb`: Potential future notebooks for prototyping later phases.

*   📁 **`scripts/`** ▶️: Contains the main executable Python scripts.
    *   📄 **`train_mnist_vit.py`**: The primary script to orchestrate ViT model training. Handles different phases based on configuration or arguments, loads data, initializes model, calls the trainer, saves artifacts, logs to W&B.
    *   📄 **`evaluate.py`**: Script to load a trained model checkpoint and evaluate its performance (e.g., accuracy) on the MNIST test set or phase-specific test data.
    *   📄 `(Optional) generate_project_doc.py`: Utility to auto-generate documentation.

*   📁 **`src/`** 🐍: Contains the core source code organized into Python modules.
    *   📄 **`__init__.py`**: Makes `src` a Python package.
    *   📁 **`mnist_transformer/`** ✨: Core module implementing the Vision Transformer logic for MNIST.
        *   📄 **`__init__.py`**: Marks `mnist_transformer` as a sub-package.
        *   📄 **`dataset.py`**: Contains functions and `Dataset` classes for loading standard MNIST (Phase 1), generating synthetic 2x2 grid data (Phase 2), and potentially handling dynamic layouts (Phase 3). Includes data transforms.
        *   📄 **`modules.py`**: Defines the building blocks of the ViT: `PatchEmbedding`, `Attention` (Multi-Head Self-Attention), `MLPBlock`, `TransformerEncoderBlock`.
        *   📄 **`model.py`**: Defines the main `VisionTransformer` nn.Module class, assembling the components from `modules.py` and adapting its input/output based on the project phase.
        *   📄 **`trainer.py`**: Implements the training loop logic (`train_epoch`, `train_model`). Adapts the loss calculation and training steps for different phases (single output classification, multi-output classification, etc.).
        *   📄 `(Optional) evaluation.py`: Functions for calculating specific metrics like overall accuracy, per-digit accuracy (Phase 2), etc.

*   📁 **`utils/`** 🛠️: Contains shared utility modules.
    *   📄 **`__init__.py`**: Marks `utils` as a package.
    *   📄 **`device_setup.py`**: Function (`get_device`) for CPU/MPS/CUDA selection.
    *   📄 **`logging.py`**: Configures the project logger (`Backprop Bunch`).
    *   📄 **`run_utils.py`**: Helper functions (e.g., `load_config`, `save_losses`, `plot_losses`).

*   📁 **`wandb/`** ☁️: Local cache and log directory for Weights & Biases. (Gitignored).

```

