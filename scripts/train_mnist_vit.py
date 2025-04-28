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
from torch.optim.lr_scheduler import CosineAnnealingLR # Example Scheduler
from pathlib import Path
import functools # For partial collate function if needed
import numpy as np # For seeding
import random # For seeding

# --- Add project root to sys.path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    print(f"🚀 [train_script] Adding project root to sys.path: "
          f"{project_root}")
    sys.path.insert(0, project_root)

# --- Project-specific imports ---
from utils import (
    logger, get_device, load_config,
    save_losses, plot_losses # Use these functions at the end
)
from src.mnist_transformer.dataset import (
    MNISTGridDataset, get_mnist_dataset, get_mnist_dataloader
)
from src.mnist_transformer.model import VisionTransformer
from src.mnist_transformer.trainer import (
    train_model, evaluate_model # Import evaluation fn
)

# W&B Import
try:
    import wandb
except ImportError:
    logger.warning("⚠️ wandb not installed. Experiment tracking disabled.")
    wandb = None

# --- Argument Parsing ---
def parse_args(config: dict):
    """Parses command-line arguments, using config for defaults."""
    parser = argparse.ArgumentParser(
        description="Train MNIST Vision Transformer."
    )

    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
                        help='Training phase (1: single digit, '
                             '2: 2x2 grid, 3: dynamic)')
    parser.add_argument('--config-path', type=str, default='config.yaml',
                        help='Path to the configuration file.')

    # --- Get phase-specific defaults ---
    # Use parse_known_args to peek at --phase without fully parsing yet
    temp_args, _ = parser.parse_known_args()
    phase = temp_args.phase
    train_cfg_key = f"phase{phase}_training" # Base key for phase block
    train_cfg = config.get("training", {}).get(
        f"phase{phase}", config.get("training", {})
    ) # Get phase sub-dict

    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    paths_cfg = config.get('paths', {})
    eval_cfg = config.get('evaluation', {})

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int,
                        default=train_cfg.get('epochs', 10),
                        help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int,
                        default=train_cfg.get('batch_size', 128),
                        help='Training batch size.')
    parser.add_argument('--lr', type=float,
                        default=train_cfg.get('base_lr', 1e-3),
                        help='Base learning rate.')
    parser.add_argument('--wd', type=float,
                        default=train_cfg.get('weight_decay', 0.03),
                        help='Weight decay for optimizer (AdamW).')
    parser.add_argument('--warmup-epochs', type=int,
                        default=train_cfg.get('warmup_epochs', 0),
                        help='Number of learning rate warmup epochs '
                             '(Scheduler needed).')

    # --- Model ---
    parser.add_argument('--model-save-dir', type=str,
                        default=paths_cfg.get('model_save_dir',
                                              'models/mnist_vit'),
                        help='Base directory to save models.')

    # --- W&B ---
    parser.add_argument('--wandb-project', type=str,
                        default='mnist-vit-transformer',
                        help='W&B project name.')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='W&B entity (username or team).')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Custom W&B run name.')
    parser.add_argument('--no-wandb', action='store_true', default=False,
                        help='Disable W&B logging.')

    # --- Other ---
    default_workers = os.cpu_count() // 2 if os.cpu_count() else 0
    parser.add_argument('--num-workers', type=int, default=default_workers,
                        help='Number of DataLoader workers.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    args = parser.parse_args()

    logger.info("--- Effective Configuration ---")
    for arg, value in vars(args).items():
        logger.info(f"  --{arg.replace('_', '-'):<20}: {value}")
    logger.info("-----------------------------")
    return args

# --- Main Function ---
def main():
    config = load_config()
    if config is None: return
    args = parse_args(config)

    # --- Setup ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = get_device()

    # --- W&B Init ---
    run = None
    run_name = "local_run" # Default if W&B off
    if wandb is not None and not args.no_wandb:
        try:
            if args.wandb_run_name is None:
                run_name = (f"Phase{args.phase}_E{args.epochs}_"
                            f"LR{args.lr}_B{args.batch_size}_ViT")
            else:
                run_name = args.wandb_run_name
            run = wandb.init(project=args.wandb_project,
                             entity=args.wandb_entity,
                             name=run_name,
                             config=vars(args))
            logger.info(f"📊 Initialized W&B run: {run.name} ({run.url})")
        except Exception as e:
            logger.error(f"❌ Failed W&B initialization: {e}",
                         exc_info=True)
            run = None
    else: logger.info("📊 W&B logging disabled.")

    # --- Load Data (Adapt based on phase) ---
    logger.info(f"--- Loading Data for Phase {args.phase} ---")
    if args.phase == 1:
        train_dataset = get_mnist_dataset(train=True,
                                          use_augmentation=True)
        val_dataset = get_mnist_dataset(train=False,
                                        use_augmentation=False)
    elif args.phase == 2:
        logger.info("Loading Phase 2 (2x2 Grid) MNIST dataset...")
        # Load base MNIST datasets for MNISTGridDataset to sample from
        base_train_dataset = get_mnist_dataset(train=True,
                                               use_augmentation=False)
        base_val_dataset = get_mnist_dataset(train=False,
                                             use_augmentation=False)

        if base_train_dataset is None or base_val_dataset is None:
             logger.error("❌ Failed to load base MNIST datasets for Phase 2.")
             if run: run.finish(exit_code=1); return
        else:
            # Define number of synthetic samples (could come from config)
            train_set_length = len(base_train_dataset)
            val_set_length = len(base_val_dataset)
            logger.info(f"Generating {train_set_length} train and "
                        f"{val_set_length} validation grid samples.")

            # Get grid size from config (or default)
            dataset_cfg = config.get('dataset', {})
            grid_img_size = dataset_cfg.get('image_size_phase2', 56)

            # Create the grid datasets
            train_dataset = MNISTGridDataset(
                mnist_dataset=base_train_dataset,
                length=train_set_length,
                grid_size=grid_img_size
            )
            val_dataset = MNISTGridDataset(
                mnist_dataset=base_val_dataset,
                length=val_set_length,
                grid_size=grid_img_size
            )
    else:
         logger.error(f"❌ Phase {args.phase} data loading not implemented!")
         if run: run.finish(exit_code=1); return


    if train_dataset is None or val_dataset is None:
        logger.error("❌ Failed to load datasets. Exiting.");
        if run: run.finish(exit_code=1); return

    train_dataloader = get_mnist_dataloader(
        train_dataset, args.batch_size, shuffle=True,
        num_workers=args.num_workers
    )
    val_dataloader = get_mnist_dataloader(
        val_dataset, config['evaluation']['batch_size'], shuffle=False,
        num_workers=args.num_workers
    )

    # --- Initialize Model (Read params based on phase if needed) ---
    logger.info("--- Initializing Vision Transformer Model ---")
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})

    # Adapt params for phase if needed (Example for Phase 2 size)
    default_img_size = dataset_cfg['image_size']
    img_size = dataset_cfg.get('image_size_phase2', default_img_size) \
        if args.phase == 2 else default_img_size
    num_outputs = dataset_cfg.get('num_outputs_phase2', 1) \
        if args.phase == 2 else 1
    patch_size = dataset_cfg.get('patch_size_phase2',
                                 dataset_cfg['patch_size']) \
        if args.phase == 2 else dataset_cfg['patch_size']

    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size, # Pass patch size
        in_channels=dataset_cfg['in_channels'],
        num_classes=dataset_cfg['num_classes'],
        embed_dim=model_cfg['embed_dim'],
        depth=model_cfg['depth'],
        num_heads=model_cfg['num_heads'],
        mlp_ratio=model_cfg['mlp_ratio'],
        attention_dropout=model_cfg.get('attention_dropout', 0.1),
        mlp_dropout=model_cfg.get('mlp_dropout', 0.1),
        num_outputs=num_outputs
    )

    # --- Loss Function & Optimizer ---
    # Adapt loss for Phase 2 later if needed
    criterion = nn.CrossEntropyLoss()
    optimizer_name = config.get("training", {}).get("optimizer", "AdamW")
    if optimizer_name.lower() == "adamw":
         optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                 weight_decay=args.wd)
    else: # Default or other optimizers like Adam
         optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.wd) # AdamW handles WD
    logger.info(f"Optimizer: {optimizer_name} (LR={args.lr}, WD={args.wd})")

    # --- LR Scheduler (Implement basic Cosine Annealing) ---
    eta_min_val = args.lr * 0.01
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                     eta_min=eta_min_val)
    logger.info(f"LR Scheduler: CosineAnnealingLR (T_max={args.epochs}, "
                f"eta_min={eta_min_val})")

    # --- Train ---
    logger.info("--- Starting Training ---")
    epoch_losses, last_val_metrics = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        model_save_dir=args.model_save_dir,
        run_name=run.name if run else run_name,
        wandb_run=run,
        lr_scheduler=lr_scheduler,
        phase=args.phase
    )

    # --- Finalize ---
    logger.info("--- Finalizing Run ---")
    logger.info(f"Final Validation Metrics: {last_val_metrics}")
    run_save_path = Path(args.model_save_dir) / (run.name if run else run_name)
    loss_file = save_losses(epoch_losses, run_save_path)
    plot_file = plot_losses(epoch_losses, run_save_path) # Plot train losses
    model_file = run_save_path / "model_final.pth"

    # Log final artifacts to W&B
    if run:
        logger.info("☁️ Logging final artifacts to W&B...")
        try:
            artifact_name = f"mnist_vit_final_{run.id}"
            final_artifact = wandb.Artifact(artifact_name, type="model")
            if model_file.exists():
                final_artifact.add_file(str(model_file))
            else:
                logger.warning(f"Model file {model_file} not found "
                               f"for artifact.")
            if loss_file and Path(loss_file).exists():
                final_artifact.add_file(loss_file)
            if plot_file and Path(plot_file).exists():
                final_artifact.add_file(plot_file)
            # Also log the config file used for this run
            config_log_path = Path(args.config_path)
            if config_log_path.exists():
                 final_artifact.add_file(str(config_log_path))
            else:
                 logger.warning(f"Config file {config_log_path} not found "
                                f"for artifact.")
            run.log_artifact(final_artifact)
            logger.info("  Logged final model, results, and config artifact.")
        except Exception as e:
            logger.error(f"❌ Failed W&B artifact logging: {e}",
                         exc_info=True)
        run.finish()
        logger.info("☁️ W&B run finished.")

    logger.info("✅ Training script completed.")

# --- Entry Point ---
if __name__ == "__main__":
    main()