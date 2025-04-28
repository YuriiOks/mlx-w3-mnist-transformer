# MNIST Digit Classifier (Transformer) - Project Structure

This document outlines the detailed directory structure and the purpose of key files for the Week 3 MNIST Vision Transformer project by **Team Backprop Bunch**. The project aims to build and train a Vision Transformer model progressively across three phases.

## Directory Tree 🌳

```
📁 mlx-w3-mnist-transformer/
├── 📁 app/
│ └── 📄 init.py
├── 📄 config.yaml # ⚙️ ViT configuration (phases, model, training)
├── 📁 data/ # 📊 MNIST dataset (auto-downloaded)
│ └── MNIST/
│ ├── processed/
│ └── raw/
├── 🐳 Dockerfile
├── 📁 docs/ # 📄 Project documentation
│ ├── 📄 05_QKV.md # (Example Transformer Note)
│ ├── (...) # (Other relevant general notes: 06-12)
│ └── 📄 STRUCTURE.MD # 👈 This file
├── 📁 logs/ # 📝 Runtime logs (e.g., mnist_vit_train.log)
├── 📁 models/ # 🧠 Saved model artifacts
│ └── 📁 mnist_vit/ # Subdir for this project's models
│ ├── 📁 Phase1_E15_.../ # Example Phase 1 run output
│ │ ├── 📄 model_final.pth
│ │ ├── ...
│ └── 📁 Phase2_E20_.../ # Example Phase 2 run output
│ ├── 📄 model_final.pth
│ └── ...
├── 📁 notebooks/ # 📓 Jupyter notebooks
│ ├── 📄 01_mnist_vit_exploration.ipynb # EDA, Patching tests
│ ├── 📄 02_phase2_results_viz.ipynb # Visualization for Phase 2
│ └── 📄 03_phase3_design_ideas.ipynb # Brainstorming for Phase 3
├── 📄 .gitignore
├── 📄 LICENSE
├── 📄 README.md # 👋 Project overview
├── 📄 requirements.txt # 📦 Python dependencies
├── 📁 scripts/ # ▶️ Runnable Python scripts
│ ├── 📄 train_mnist_vit.py # 👈 Main training script (handles phases)
│ ├── 📄 evaluate.py # 👈 Evaluation script
│ └── 📄 generate_project_doc.py # (Optional utility)
├── 🐍 src/ # 🐍 Core Python source code
│ ├── 📄 init.py
│ └── 📁 mnist_transformer/ # 👈 Module for ViT logic
│ ├── 📄 init.py
│ ├── 📄 dataset.py # 👈 Handles MNIST (P1), 2x2 grid (P2), dynamic (P3)
│ ├── 📄 modules.py # 👈 ViT building blocks (PatchEmbed, Attention, MLP, EncoderBlock)
│ ├── 📄 model.py # 👈 Defines VisionTransformer class (adapts P1/P2/P3)
│ ├── 📄 trainer.py # 👈 Training/Eval loops (adapts loss/acc for P1/P2/P3)
│ └── 📄 evaluation.py # 👈 (Optional) Metrics calculation functions
│
└── 🛠️ utils/ # 🛠️ Shared utility modules
├── 📄 init.py
├── 📄 device_setup.py
├── 📄 logging.py
└── 📄 run_utils.py
```


## Detailed Directory & File Descriptions 📜

*   **Root Directory (`mlx-w3-mnist-transformer/`)**: Main project directory for the MNIST Vision Transformer.
    *   📄 **`.gitignore`**: Lists files/dirs Git should ignore (e.g., `.venv`, `logs/`, `models/`, `wandb/`, `data/MNIST/`).
    *   📄 **`config.yaml`** ⚙️: Central config for ViT project. Defines paths, dataset params (per phase), model hyperparameters (shared/phase-specific), training settings (per phase), evaluation, and logging.
    *   📄 **`Dockerfile`** 🐳: (Optional) For building a Docker container.
    *   📄 **`LICENSE`**: Project license.
    *   📄 **`README.md`** 👋: High-level overview, setup, usage, status, and plan.
    *   📄 **`requirements.txt`** 📦: Python dependencies (`torch`, `torchvision`, `wandb`, `PyYAML`, `tqdm`, `coloredlogs`, etc.).

*   📁 **`app/`**: (Optional) For future API/UI deployment.
    *   📄 **`__init__.py`**: Makes `app` a Python package.

*   📁 **`data/`** 📊: Default location for datasets.
    *   📁 **`MNIST/`**: Auto-created by `torchvision` to store downloaded MNIST files. (Gitignored).

*   📁 **`docs/`** 📄: Project documentation and notes.
    *   📄 `0X_*.md`: Conceptual notes on Transformers, ViT, etc.
    *   📄 **`STRUCTURE.MD`**: This file, describing the project structure.
    *   📄 **`DEV_PLAN_W3.md`**: Detailed development plan and task breakdown for Week 3.

*   📁 **`logs/`** 📝: Stores runtime log files (e.g., `mnist_vit_train.log`). (Gitignored).

*   📁 **`models/`** 🧠: Root directory for saved model artifacts. (Gitignored).
    *   📁 **`mnist_vit/`**: Contains subdirectories for trained MNIST ViT runs.
        *   📁 **`<RUN_NAME>/`**: Subdirectory for each training run (e.g., `Phase1_...`, `Phase2_...`). Contains saved model weights (`.pth`), loss data (`.json`), and plots (`.png`).

*   📁 **`notebooks/`** 📓: Jupyter notebooks for exploration, visualization, and design.
    *   📄 **`01_mnist_vit_exploration.ipynb`**: Initial EDA, patching tests.
    *   📄 **`02_phase2_results_viz.ipynb`**: Visualization of Phase 2 model predictions.
    *   📄 **`03_phase3_design_ideas.ipynb`**: Brainstorming and planning for Phase 3.

*   📁 **`scripts/`** ▶️: Main executable Python scripts.
    *   📄 **`train_mnist_vit.py`**: Primary script for training the ViT model. Handles different phases via `--phase` argument, loads config, data, model, runs training loop, saves results, logs to W&B.
    *   📄 **`evaluate.py`**: Script to load a saved model checkpoint and evaluate its performance on the appropriate test set (Phase 1 or Phase 2/3).
    *   📄 `(Optional) generate_project_doc.py`: Utility for auto-documentation.

*   📁 **`src/`** 🐍: Core source code modules.
    *   📄 **`__init__.py`**: Makes `src` a Python package.
    *   📁 **`mnist_transformer/`** ✨: Core module for the ViT implementation.
        *   📄 **`__init__.py`**: Marks `mnist_transformer` as a sub-package.
        *   📄 **`dataset.py`**: Handles data loading (`get_mnist_dataset`), transformations (`get_mnist_transforms`), Phase 2 grid generation (`generate_2x2_grid_image`, `MNISTGridDataset`), and potentially Phase 3 dynamic data generation.
        *   📄 **`modules.py`**: Defines ViT building blocks: `PatchEmbedding`, `Attention` (custom MHA using `AttentionHead`), `MLPBlock`, `TransformerEncoderBlock`.
        *   📄 **`model.py`**: Defines the main `VisionTransformer` class, assembling modules and adapting for different phases (input size, output heads).
        *   📄 **`trainer.py`**: Implements the training (`train_epoch`) and evaluation (`evaluate_model`) loops. Adapts loss and accuracy calculation based on the training phase. Contains the main `train_model` orchestrator.
        *   📄 `(Optional) evaluation.py`: Could hold specific metric functions if needed.

*   📁 **`utils/`** 🛠️: Shared utility modules.
    *   📄 **`__init__.py`**: Marks `utils` as a package.
    *   📄 **`device_setup.py`**: `get_device` function.
    *   📄 **`logging.py`**: Configures the `Backprop Bunch` logger with colored console output.
    *   📄 **`run_utils.py`**: Helper functions (`load_config`, `save_losses`, `plot_losses`).

*   📁 **`wandb/`** ☁️: Local W&B cache and logs. (Gitignored).