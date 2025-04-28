# MNIST Digit Classifier (Transformer) - MLX Version
# File: scripts/train_mnist_vit_mlx.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Main script to train the ViT model on MNIST using MLX.
# Created: 2025-04-28
# Updated: 2025-04-28

import os
import sys
import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from pathlib import Path
import numpy as np
import random
import time

# --- Add project root to sys.path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    print(f"🚀 [train_script_mlx] Adding project root to sys.path: "
          f"{project_root}")
    sys.path.insert(0, project_root)

# --- Project-specific imports ---
from utils import (
    logger, load_config, save_losses, plot_losses
)
from src.mnist_transformer_mlx.dataset_mlx import get_mnist_data_arrays
from src.mnist_transformer_mlx.model_mlx import VisionTransformerMLX
from src.mnist_transformer_mlx.trainer_mlx import train_model_mlx

# W&B Import
try:
    import wandb
except ImportError:
    logger.warning("⚠️ wandb not installed. Experiment tracking disabled.")
    wandb = None

# --- Argument Parsing ---
def parse_args(config: dict):
    """ Parses command-line arguments, using config for defaults (MLX). """
    parser = argparse.ArgumentParser(
        description="Train MNIST Vision Transformer (MLX)."
    )

    parser.add_argument('--phase', type=int, default=1, choices=[1],
                        help='Training phase (Only 1 supported for MLX).')
    parser.add_argument('--config-path', type=str, default='config.yaml',
                        help='Path to config file.')

    temp_args, _ = parser.parse_known_args()
    phase = temp_args.phase
    train_cfg_key = f"phase{phase}"
    train_cfg = config.get("training", {}).get(
        train_cfg_key, config.get("training", {})
    )

    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    paths_cfg = config.get('paths', {})
    eval_cfg = config.get('evaluation', {})

    parser.add_argument('--epochs', type=int,
                        default=train_cfg.get('epochs', 5),
                        help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int,
                        default=train_cfg.get('batch_size', 128),
                        help='Training batch size.')
    parser.add_argument('--lr', type=float,
                        default=train_cfg.get('base_lr', 1e-3),
                        help='Base learning rate.')
    parser.add_argument('--wd', type=float,
                        default=train_cfg.get('weight_decay', 0.03),
                        help='Weight decay for optimizer.')

    parser.add_argument('--model-save-dir', type=str,
                        default=paths_cfg.get('model_save_dir',
                                              'models/mnist_vit'),
                        help='Base directory to save models.')

    parser.add_argument('--wandb-project', type=str,
                        default='mnist-vit-transformer-mlx',
                        help='W&B project name.')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='W&B entity.')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Custom W&B run name.')
    parser.add_argument('--no-wandb', action='store_true', default=False,
                        help='Disable W&B logging.')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')

    args = parser.parse_args()

    logger.info("--- Effective Configuration (MLX) ---")
    for arg, value in vars(args).items():
        logger.info(f"  --{arg.replace('_', '-'):<20}: {value}")
    logger.info("------------------------------------")
    return args

# --- Main Function ---
def main():
    # --- Setup ---
    config = load_config()
    if config is None:
        return
    args = parse_args(config)

    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)
    logger.info(f"🌱 Seed set to {args.seed}")

    # --- W&B Init ---
    run_name_base = (
        f"Phase{args.phase}_E{args.epochs}_LR{args.lr}_B{args.batch_size}"
    )
    run_name = f"MLX_{run_name_base}_ViT"  # Prepend MLX_
    if args.wandb_run_name:
        run_name = args.wandb_run_name

    if wandb is not None and not args.no_wandb:
        try:
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args)
            )
            logger.info(f"📊 Initialized W&B run: {run.name} ({run.url})")
        except Exception as e:
            logger.error(f"❌ Failed W&B init: {e}", exc_info=True)
            run = None
    else:
        logger.info("📊 W&B logging disabled.")

    logger.info(f"--- Loading Data for Phase {args.phase} (MLX) ---")
    if args.phase == 1:
        train_images_np, train_labels_np = get_mnist_data_arrays(train=True)
        val_images_np, val_labels_np = get_mnist_data_arrays(train=False)
    else:
        logger.error(f"❌ MLX data loading for Phase {args.phase} not "
                     "fully implemented in script yet!")
        if run:
            run.finish(exit_code=1)
        return

    if train_images_np is None or val_images_np is None:
        logger.error("❌ Failed to load numpy datasets. Exiting.")
        if run:
            run.finish(exit_code=1)
        return

    train_images = mx.array(train_images_np)
    train_labels = mx.array(train_labels_np)
    val_images = mx.array(val_images_np)
    val_labels = mx.array(val_labels_np)
    logger.info("✅ Converted NumPy data to MLX arrays.")

    del train_images_np, train_labels_np, val_images_np, val_labels_np

    logger.info("--- Initializing MLX Vision Transformer Model ---")
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})

    img_size = dataset_cfg.get('image_size', 28)
    patch_size = dataset_cfg.get('patch_size', 7)
    num_outputs = 1

    model = VisionTransformerMLX(
        img_size=img_size,
        patch_size=patch_size,
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
    mx.eval(model.parameters())
    
    # --- 👇 CORRECTED PARAMETER COUNT ---
    # Use tree_flatten to get all leaf arrays (parameters)
    # leaves = mx.tree_flatten(model.parameters())
    leaves = tree_flatten(model.parameters())
    # leaves is a list of (key_path, mx.array) tuples
    nparams = sum(arr.size for _, arr in leaves)
    # --- End Correction ---

    logger.info(f"Model parameter count: {nparams / 1e6:.3f} M")

    optimizer_name = config.get("training", {}).get("optimizer", "AdamW")
    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(learning_rate=args.lr)
    logger.info(f"Optimizer: {optimizer_name} (LR={args.lr}, "
                f"WD={args.wd if optimizer_name.lower() == 'adamw' else 'N/A'})")

    lr_scheduler = None
    logger.info(f"LR Scheduler: {'None'}")

    logger.info("--- Starting MLX Training ---")
    start_time = time.time()

    epoch_losses, last_val_metrics = train_model_mlx(
        model=model,
        optimizer=optimizer,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_dir=args.model_save_dir,
        run_name=run.name if run else run_name,
        wandb_run=run,
        phase=args.phase
    )

    training_time = time.time() - start_time
    logger.info(f"Total Training Time: {training_time:.2f} seconds")

    logger.info("--- Finalizing Run ---")
    logger.info(f"Final Validation Metrics: {last_val_metrics}")
    run_save_path = Path(args.model_save_dir) / (run.name if run else run_name)

    loss_file = save_losses(epoch_losses, run_save_path)
    plot_file = plot_losses(epoch_losses, run_save_path)
    model_file = run_save_path / "model_weights.safetensors"

    if run:
        logger.info("☁️ Logging final artifacts to W&B...")
        try:
            final_artifact = wandb.Artifact(
                f"mnist_vit_mlx_final_{run.id}", type="model"
            )
            if model_file.exists():
                final_artifact.add_file(str(model_file))
            else:
                logger.warning(f"MLX Model weights file {model_file} not found.")
            if loss_file and Path(loss_file).exists():
                final_artifact.add_file(loss_file)
            if plot_file and Path(plot_file).exists():
                final_artifact.add_file(plot_file)
            config_log_path = Path(args.config_path)
            if config_log_path.exists():
                final_artifact.add_file(str(config_log_path))
            run.log_artifact(final_artifact)
            logger.info("  Logged final model weights, results, and config.")
        except Exception as e:
            logger.error(f"❌ Failed W&B artifact logging: {e}", exc_info=True)
        run.finish()
        logger.info("☁️ W&B run finished.")

    logger.info("✅ MLX Training script completed.")

# --- Entry Point ---
if __name__ == "__main__":
    main()