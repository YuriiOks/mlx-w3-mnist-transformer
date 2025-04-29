# MNIST Digit Classifier (Transformer) - PyTorch Version
# File: scripts/train_mnist_vit.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Main script to train the ViT model on MNIST (All Phases).
# Created: 2025-04-28
# Updated: 2025-04-29

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import functools
import numpy as np
import random
import time
from tqdm import tqdm

# --- Add project root to sys.path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    print(f"🚀 [train_script] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)

# --- Project-specific imports ---
from utils import (
    logger, get_device, load_config,
    save_metrics, plot_metrics
)
from src.mnist_transformer.dataset import (
    get_mnist_dataset, get_dataloader, # Use generic dataloader name
    MNISTGridDataset, MNISTDynamicDataset, # Import Phase 2/3 datasets
    DEFAULT_DATA_DIR,
    get_mnist_transforms
)
from src.mnist_transformer.model import VisionTransformer, EncoderDecoderViT
from src.mnist_transformer.trainer import train_model

from torchvision import datasets as datasets # Use alias to avoid confusion


try:
    from utils.tokenizer_utils import DECODER_VOCAB_SIZE, PAD_TOKEN_ID
except ImportError:
    logger.warning("⚠️ Could not import tokenizer utils. Phase 3 might fail.")
    DECODER_VOCAB_SIZE = 13
    PAD_TOKEN_ID = 0

try:
    import wandb
except ImportError:
    logger.warning("⚠️ wandb not installed. Experiment tracking disabled.")
    wandb = None

def parse_args(config: dict):
    """ Parses command-line arguments, using config for defaults. """
    parser = argparse.ArgumentParser(
        description="Train MNIST Vision Transformer (PyTorch)."
    )

    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
        help='Training phase (1: single, 2: 2x2 grid, 3: dynamic seq).')
    parser.add_argument('--config-path', type=str, default='config.yaml',
        help='Path to config file.')

    temp_args, _ = parser.parse_known_args()
    phase = temp_args.phase
    train_cfg_key = f"phase{phase}"
    train_cfg = config.get("training", {}).get(train_cfg_key, {})
    general_train_cfg = config.get("training", {})

    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    paths_cfg = config.get('paths', {})
    eval_cfg = config.get('evaluation', {})

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
        help='Weight decay for optimizer.')

    parser.add_argument('--model-save-dir', type=str,
        default=paths_cfg.get('model_save_dir', 'models/mnist_vit'),
        help='Base directory to save models.')

    parser.add_argument('--wandb-project', type=str,
        default='mnist-vit-transformer', help='W&B project name.')
    parser.add_argument('--wandb-entity', type=str, default=None,
        help='W&B entity.')
    parser.add_argument('--wandb-run-name', type=str, default=None,
        help='Custom W&B run name.')
    parser.add_argument('--no-wandb', action='store_true', default=False,
        help='Disable W&B logging.')

    parser.add_argument('--num-workers', type=int,
        default=os.cpu_count() // 2 if os.cpu_count() else 0,
        help='Number of DataLoader workers.')
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed.')

    args = parser.parse_args()

    logger.info("--- Effective Configuration (PyTorch) ---")
    for arg, value in vars(args).items():
        logger.info(f"  --{arg.replace('_', '-'):<20}: {value}")
    logger.info("------------------------------------")
    return args

def main():
    config = load_config()
    if config is None:
        return
    args = parse_args(config)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = get_device()

    run = None
    run_name_base = (
        f"Phase{args.phase}_E{args.epochs}_LR{args.lr}_B{args.batch_size}"
    )
    run_name = f"PT_{run_name_base}_ViT"
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
        run_name = f"PT_{run_name_base}_local"

    logger.info(f"--- Loading Data for Phase {args.phase} (PyTorch) ---")
    train_dataset = None
    val_dataset = None
    dataset_cfg = config.get('dataset', {})
    paths_cfg = config.get('paths', {})
    data_dir = paths_cfg.get('data_dir', DEFAULT_DATA_DIR)

    if args.phase == 1:
        p1_transform_train = get_mnist_transforms(
            image_size=dataset_cfg['image_size'], augment=True
        )
        p1_transform_val = get_mnist_transforms(
            image_size=dataset_cfg['image_size'], augment=False
        )
        train_dataset = get_mnist_dataset(
            train=True, data_dir=data_dir, transform=p1_transform_train
        )
        val_dataset = get_mnist_dataset(
            train=False, data_dir=data_dir, transform=p1_transform_val
        )
    elif args.phase == 2:
        base_transform = get_mnist_transforms(image_size=28, augment=False)
        base_train_dataset = get_mnist_dataset(
            train=True, data_dir=data_dir, transform=base_transform
        )
        base_val_dataset = get_mnist_dataset(
            train=False, data_dir=data_dir, transform=base_transform
        )
        if base_train_dataset and base_val_dataset:
            train_set_length = len(base_train_dataset)
            val_set_length = len(base_val_dataset)
            p2_grid_size = dataset_cfg.get('image_size_phase2', 56)
            train_dataset = MNISTGridDataset(
                base_train_dataset, train_set_length, p2_grid_size
            )
            val_dataset = MNISTGridDataset(
                base_val_dataset, val_set_length, p2_grid_size
            )
    elif args.phase == 3:
        base_train_pil = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=None
        )
        base_val_pil = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=None
        )
        if base_train_pil and base_val_pil:
            train_set_length = len(base_train_pil)
            val_set_length = len(base_val_pil)
            train_dataset = MNISTDynamicDataset(
                base_train_pil, train_set_length, config
            )
            val_dataset = MNISTDynamicDataset(
                base_val_pil, val_set_length, config
            )
    else:
        logger.error(f"❌ Phase {args.phase} data loading not implemented!")
        if run:
            run.finish(exit_code=1)
        return

    if train_dataset is None or val_dataset is None:
        logger.error("❌ Failed to prepare datasets. Exiting.")
        if run:
            run.finish(exit_code=1)
        return

    eval_batch_size = config.get('evaluation', {}).get(
        'batch_size', args.batch_size * 2
    )
    train_dataloader = get_dataloader(
        train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_dataloader = get_dataloader(
        val_dataset, eval_batch_size, shuffle=False, num_workers=args.num_workers
    )

    logger.info(f"--- Initializing PyTorch Model for Phase {args.phase} ---")
    model_cfg = config.get('model', {})
    tokenizer_cfg = config.get('tokenizer', {})

    if args.phase == 1 or args.phase == 2:
        img_size = dataset_cfg.get(
            f'image_size_phase{args.phase}', dataset_cfg['image_size']
        )
        patch_size = dataset_cfg.get('patch_size')
        num_outputs = dataset_cfg.get(
            f'num_outputs_phase{args.phase}', 1
        )
        num_classes = dataset_cfg.get('num_classes')
        model = VisionTransformer(
            img_size=img_size, patch_size=patch_size,
            in_channels=dataset_cfg['in_channels'],
            num_classes=num_classes, embed_dim=model_cfg['embed_dim'],
            depth=model_cfg['depth'], num_heads=model_cfg['num_heads'],
            mlp_ratio=model_cfg['mlp_ratio'],
            dropout=model_cfg.get('dropout', 0.1),
            attention_dropout=model_cfg.get('attention_dropout', 0.1),
            num_outputs=num_outputs
        )
    elif args.phase == 3:
        img_size = dataset_cfg.get('image_size_phase3')
        patch_size = dataset_cfg.get('patch_size_phase3')
        decoder_vocab_size = tokenizer_cfg.get('vocab_size')
        model = EncoderDecoderViT(
            img_size=img_size, patch_size=patch_size,
            in_channels=dataset_cfg['in_channels'],
            encoder_embed_dim=model_cfg['embed_dim'],
            encoder_depth=model_cfg['depth'],
            encoder_num_heads=model_cfg['num_heads'],
            decoder_vocab_size=decoder_vocab_size,
            decoder_embed_dim=model_cfg['decoder_embed_dim'],
            decoder_depth=model_cfg['decoder_depth'],
            decoder_num_heads=model_cfg['decoder_num_heads'],
            mlp_ratio=model_cfg['mlp_ratio'],
            dropout=model_cfg.get('dropout', 0.1),
            attention_dropout=model_cfg.get('attention_dropout', 0.1)
        )
    else:
        logger.error(f"❌ Model instantiation for Phase {args.phase} not defined!")
        if run:
            run.finish(exit_code=1)
        return

    pad_token_id = tokenizer_cfg.get('pad_token_id', PAD_TOKEN_ID)
    if args.phase == 3:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        logger.info(
            f"Using CrossEntropyLoss with ignore_index={pad_token_id} for Phase 3."
        )
    else:
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    optimizer_name = config.get("training", {}).get("optimizer", "AdamW")
    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )
    logger.info(f"Optimizer: {optimizer_name} (LR={args.lr}, WD={args.wd})")

    phase_epochs = config.get("training", {}).get(
        f"phase{args.phase}", {}
    ).get("epochs", args.epochs)
    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=phase_epochs, eta_min=args.lr * 0.01
    )
    logger.info(
        f"LR Scheduler: CosineAnnealingLR (T_max={phase_epochs}, "
        f"eta_min={args.lr * 0.01})"
    )

    logger.info("--- Starting PyTorch Training ---")
    start_time = time.time()

    # --- Train and collect full metrics history ---
    metrics_history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        model_save_dir=args.model_save_dir,
        config=config,
        phase=args.phase,
        run_name=run.name if run else run_name,
        wandb_run=run,
        lr_scheduler=lr_scheduler
    )

    training_time = time.time() - start_time
    logger.info(f"Total Training Time: {training_time:.2f} seconds")

    logger.info("--- Finalizing Run ---")
    run_save_path = Path(args.model_save_dir) / (run.name if run else run_name)

    # --- Save and plot all metrics ---
    metrics_file = save_metrics(metrics_history, run_save_path)
    plot_file = plot_metrics(metrics_history, run_save_path)

    model_file = run_save_path / "model_final.pth"

    if run:
        logger.info("☁️ Logging final artifacts to W&B...")
        try:
            artifact_name = f"mnist_vit_pt_final_{run.id}"
            final_artifact = wandb.Artifact(artifact_name, type="model")
            if model_file.exists():
                final_artifact.add_file(str(model_file))
            else:
                logger.warning(f"Model file {model_file} not found.")
            if metrics_file and Path(metrics_file).exists():
                final_artifact.add_file(metrics_file)
            if plot_file and Path(plot_file).exists():
                final_artifact.add_file(plot_file)
            config_log_path = Path(args.config_path)
            if config_log_path.exists():
                final_artifact.add_file(str(config_log_path))
            run.log_artifact(final_artifact)
            logger.info("  Logged final model, results, and config artifact.")
        except Exception as e:
            logger.error(f"❌ Failed W&B artifact logging: {e}", exc_info=True)
        run.finish()
        logger.info("☁️ W&B run finished.")

    logger.info("✅ PyTorch Training script completed.")

if __name__ == "__main__":
    main()