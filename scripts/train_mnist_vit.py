# MNIST Digit Classifier (Transformer)
# File: scripts/train_mnist_vit.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Main script to train the Vision Transformer model on MNIST.
# Created: 2025-04-28
# Updated: 2025-04-28

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau # Example Schedulers
from pathlib import Path
import functools # For partial collate function if needed

# --- Add project root to sys.path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    print(f"🚀 [train_script] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)

# --- Project-specific imports ---
from utils import (
    logger, get_device, load_config,
    save_losses, plot_losses # Assuming these utils exist and are suitable
)
from src.mnist_transformer.dataset import get_mnist_dataset, get_mnist_dataloader
from src.mnist_transformer.model import VisionTransformer
from src.mnist_transformer.trainer import train_model

# W&B Import
try:
    import wandb
except ImportError:
    logger.warning("⚠️ wandb not installed. Experiment tracking disabled.")
    wandb = None

# --- Argument Parsing ---
def parse_args(config: dict):
    """Parses command-line arguments, using config for defaults."""
    parser = argparse.ArgumentParser(description="Train MNIST Vision Transformer.")

    # --- Key Arguments ---
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
                        help='Training phase (1: single digit, 2: 2x2 grid, 3: dynamic)')
    parser.add_argument('--config-path', type=str, default='config.yaml',
                        help='Path to the configuration file.') # Allow overriding config path

    # --- Get phase-specific defaults from config ---
    phase = parser.parse_known_args()[0].phase # Sneak peek at phase arg
    train_cfg_key = f"phase{phase}_training" if f"phase{phase}_training" in config.get("training", {}) else "training"
    train_cfg = config.get("training", {}).get(f"phase{phase}_training", config.get("training", {})) # Fallback slightly complex

    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    paths_cfg = config.get('paths', {})

    # --- Training Hyperparameters (with phase-specific defaults) ---
    parser.add_argument('--epochs', type=int, default=train_cfg.get(f'phase{phase}_epochs', train_cfg.get('epochs', 10)),
                        help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=train_cfg.get(f'phase{phase}_batch_size', train_cfg.get('batch_size', 128)),
                        help='Training batch size.')
    parser.add_argument('--lr', type=float, default=train_cfg.get(f'phase{phase}_base_lr', train_cfg.get('learning_rate', 1e-3)),
                        help='Base learning rate.')
    parser.add_argument('--wd', type=float, default=train_cfg.get(f'phase{phase}_weight_decay', train_cfg.get('weight_decay', 0.03)),
                        help='Weight decay for optimizer (AdamW).')
    parser.add_argument('--warmup-epochs', type=int, default=train_cfg.get(f'phase{phase}_warmup_epochs', train_cfg.get('warmup_epochs', 3)),
                        help='Number of learning rate warmup epochs.')

    # --- Model Configuration (usually defaults from config are fine) ---
    # You could add overrides here if needed, e.g., --embed-dim, --depth, --heads
    parser.add_argument('--model-save-dir', type=str, default=paths_cfg.get('model_save_dir', 'models/mnist_vit'),
                        help='Base directory to save models.')

    # --- W&B ---
    parser.add_argument('--wandb-project', type=str, default='mnist-vit-transformer',
                        help='W&B project name.')
    parser.add_argument('--wandb-entity', type=str, default=None, # Often your username or team name
                        help='W&B entity (username or team).')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Custom W&B run name (defaults to auto-generated).')
    parser.add_argument('--no-wandb', action='store_true', default=False,
                        help='Disable W&B logging.')

    # --- Other ---
    parser.add_argument('--num-workers', type=int, default=os.cpu_count() // 2 if os.cpu_count() else 0,
                        help='Number of DataLoader workers.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')


    args = parser.parse_args()

    # Log effective configuration
    logger.info("--- Effective Configuration ---")
    for arg, value in vars(args).items():
        logger.info(f"  --{arg.replace('_', '-'):<20}: {value}")
    logger.info("-----------------------------")

    return args

# --- Main Function ---
def main():
    # --- Initial Setup ---
    config = load_config() # Load default config first
    if config is None:
        logger.error("❌ Failed to load configuration. Exiting.")
        return
    args = parse_args(config) # Parse args, using config for defaults

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = get_device()

    # --- ✨ Initialize W&B ✨ ---
    run = None # Initialize run object to None
    if wandb is not None and not args.no_wandb:
        try:
            # Construct a meaningful run name if not provided
            if args.wandb_run_name is None:
                run_name = f"Phase{args.phase}_E{args.epochs}_LR{args.lr}_B{args.batch_size}_ViT"
                # Add more details if needed (e.g., embed_dim, depth)
            else:
                run_name = args.wandb_run_name

            # Initialize W&B
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,  # Can be None, W&B uses default
                name=run_name,
                config=vars(args)  # Log all command-line args and resolved defaults
            )
            logger.info(f"📊 Initialized W&B run: {run.name} ({run.get_url()})")

        except Exception as e:
            logger.error(f"❌ Failed W&B initialization: {e}", exc_info=True)
            run = None # Ensure run is None if init fails
    else:
        logger.info("📊 W&B logging disabled.")
        run_name = "local_run" # Default run name for saving if W&B is off
    # --- End W&B Init ---

    # --- Load Data (Adapt based on phase) ---
    logger.info(f"--- Loading Data for Phase {args.phase} ---")
    # For Phase 1, use standard MNIST loading
    if args.phase == 1:
        # Use augmentation only for training set
        train_dataset = get_mnist_dataset(train=True, use_augmentation=True)
        test_dataset = get_mnist_dataset(train=False, use_augmentation=False)
    # Add elif args.phase == 2: ... etc. for later phases, using different dataset functions
    else:
         logger.error(f"❌ Phase {args.phase} data loading not implemented yet!")
         if run: run.finish(exit_code=1)
         return

    if train_dataset is None or test_dataset is None:
        logger.error("❌ Failed to load datasets. Exiting.")
        if run: run.finish(exit_code=1)
        return

    train_dataloader = get_mnist_dataloader(
        train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_dataloader = get_mnist_dataloader( # Used for evaluation later
        test_dataset, args.batch_size * 2, shuffle=False, num_workers=args.num_workers # Often use larger batch for eval
    )

    # --- Initialize Model (Read params from config['model']) ---
    logger.info("--- Initializing Vision Transformer Model ---")
    model_params = config.get('model', {})
    dataset_params = config.get('dataset', {}) # Needed for img_size, patch_size etc.

    # Ensure necessary params are present
    required_model_params = ['embed_dim', 'depth', 'num_heads', 'mlp_ratio']
    required_dataset_params = ['image_size', 'patch_size', 'in_channels', 'num_classes']
    if not all(k in model_params for k in required_model_params) or \
       not all(k in dataset_params for k in required_dataset_params):
        logger.error("❌ Missing required model or dataset parameters in config.yaml!")
        if run: run.finish(exit_code=1)
        return

    model = VisionTransformer(
        img_size=dataset_params['image_size'],
        patch_size=dataset_params['patch_size'],
        in_channels=dataset_params['in_channels'],
        num_classes=dataset_params['num_classes'],
        embed_dim=model_params['embed_dim'],
        depth=model_params['depth'],
        num_heads=model_params['num_heads'],
        mlp_ratio=model_params['mlp_ratio'],
        attention_dropout=model_params.get('attention_dropout', 0.1), # Use .get for optional
        mlp_dropout=model_params.get('mlp_dropout', 0.1)
    )

    # --- Loss Function & Optimizer ---
    criterion = nn.CrossEntropyLoss() # Standard for classification
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )
    logger.info(f"Optimizer: AdamW (LR={args.lr}, WD={args.wd})")

    # --- LR Scheduler (Example: Cosine Annealing with Warmup) ---
    # Note: Implementing warmup often requires a custom scheduler or a library like timm
    # Simple Cosine Annealing example (without warmup):
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01) # Example: decay to 1% of base LR
    lr_scheduler = None # Set to None initially, implement properly later if needed
    logger.info(f"LR Scheduler: {'CosineAnnealingLR (Example)' if lr_scheduler else 'None'}") # Update if scheduler used

    # --- Train ---
    logger.info("--- Starting Training ---")
    epoch_losses = train_model(
        model=model,
        train_dataloader=train_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs, # Use parsed epochs
        model_save_dir=args.model_save_dir,
        run_name=run.name if run else run_name, # Use W&B run name for saving folder
        wandb_run=run, # Pass the W&B run object to the trainer
        lr_scheduler=lr_scheduler
        # Pass val_dataloader and evaluation_fn here later for validation
    )

    # --- Finalize ---
    logger.info("--- Finalizing Run ---")
    run_save_path = Path(args.model_save_dir) / (run.name if run else run_name)
    loss_file = save_losses(epoch_losses, run_save_path)
    plot_file = plot_losses(epoch_losses, run_save_path)
    model_file = run_save_path / "model_final.pth" # Path where trainer saved

    # Log final artifacts to W&B
    if run:
        logger.info("☁️ Logging final artifacts to W&B...")
        try:
            final_artifact = wandb.Artifact(f"mnist_vit_final_{run.id}", type="model")
            if model_file.exists():
                 final_artifact.add_file(str(model_file))
            else:
                 logger.warning(f"Model file not found at {model_file}, cannot add to W&B artifact.")
            if loss_file and Path(loss_file).exists(): final_artifact.add_file(loss_file)
            if plot_file and Path(plot_file).exists(): final_artifact.add_file(plot_file)
            final_artifact.add_file(args.config_path) # Log the config used
            run.log_artifact(final_artifact)
            logger.info("  Logged final model, results, and config artifact.")
        except Exception as e:
            logger.error(f"❌ Failed final W&B artifact logging: {e}", exc_info=True)

        run.finish() # End the W&B run
        logger.info("☁️ W&B run finished.")

    logger.info("✅ Training script completed.")

# --- Entry Point ---
if __name__ == "__main__":
    # Import utility for random seeding - needs numpy
    import numpy as np
    import random

    main()