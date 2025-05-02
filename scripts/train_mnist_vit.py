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
import pickle

# --- Add project root to sys.path ---
scriPyTorch_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(scriPyTorch_dir)
if project_root not in sys.path:
    print(f"🚀 [train_script] Adding project root to sys.path: "
          f"{project_root}")
    sys.path.insert(0, project_root)

# --- Project-specific imports ---
from utils import (
    logger, get_device, load_config,
    save_metrics, plot_metrics
)
from src.mnist_transformer.dataset import (
    get_mnist_dataset, get_dataloader,
    MNISTGridDataset, MNISTDynamicDataset,
    DEFAULT_DATA_DIR,
    get_mnist_transforms
)
from src.mnist_transformer.model import VisionTransformer, EncoderDecoderViT
from src.mnist_transformer.trainer import (
    train_model,
    evaluate_model,
    save_checkpoint_pt,
    load_checkpoint_pt
)

from torchvision import datasets as datasets

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
    """
    Parses command-line arguments, using config for defaults.

    Args:
        config (dict): Configuration dictionary loaded from YAML.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train MNIST Vision Transformer (PyTorch)."
    )

    parser.add_argument(
        '--phase',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='Training phase (1: single, 2: 2x2 grid, 3: dynamic seq).'
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default='config.yaml',
        help='Path to config file.'
    )

    temp_args, _ = parser.parse_known_args()
    phase = temp_args.phase
    train_cfg_key = f"phase{phase}"
    train_cfg = config.get("training", {}).get(train_cfg_key, {})
    general_train_cfg = config.get("training", {})

    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    paths_cfg = config.get('paths', {})
    eval_cfg = config.get('evaluation', {})

    parser.add_argument(
        '--epochs',
        type=int,
        default=train_cfg.get('epochs', 10),
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=train_cfg.get('batch_size', 128),
        help='Training batch size.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=train_cfg.get('base_lr', 1e-3),
        help='Base learning rate.'
    )
    parser.add_argument(
        '--wd',
        type=float,
        default=train_cfg.get('weight_decay', 0.03),
        help='Weight decay for optimizer.'
    )
    parser.add_argument(
        '--model-save-dir',
        type=str,
        default=paths_cfg.get('model_save_dir', 'models/mnist_vit'),
        help='Base directory to save models.'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='mnist-vit-transformer',
        help='W&B project name.'
    )
    parser.add_argument(
        '--wandb-entity',
        type=str,
        default=None,
        help='W&B entity.'
    )
    parser.add_argument(
        '--wandb-run-name',
        type=str,
        default=None,
        help='Custom W&B run name.'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        default=False,
        help='Disable W&B logging.'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=os.cpu_count() // 2 if os.cpu_count() else 0,
        help='Number of DataLoader workers.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Resume training from the latest checkpoint.'
    )
    parser.add_argument(
        '--resume-dir', type=str, default=None,
        help='Explicit path to checkpoint directory to resume from (overrides run name logic).'
    )

    args = parser.parse_args()

    logger.info("--- Effective Configuration (PyTorch) ---")
    for arg, value in vars(args).items():
        logger.info(f"  --{arg.replace('_', '-'):<20}: {value}")
    logger.info("------------------------------------")
    return args

def main():
    """
    Main entry point for training the MNIST Vision Transformer model.

    Loads configuration, prepares data, initializes model, optimizer,
    scheduler, and handles training, evaluation, checkpointing, and
    experiment tracking.

    Steps:
        1. Load config and parse arguments.
        2. Set random seeds and device.
        3. Initialize W&B run if enabled.
        4. Prepare datasets and dataloaders for the selected phase.
        5. Instantiate model, loss, optimizer, and scheduler.
        6. Resume from checkpoint if requested.
        7. Train the model and collect metrics.
        8. Save metrics, plots, and model.
        9. Log artifacts to W&B if enabled.
    """
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
    run_name = f"PyTorch_{run_name_base}_ViT"
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
        run_name = f"PyTorch_{run_name_base}_local"

    logger.info(f"--- Loading Data for Phase {args.phase} (PyTorch) ---")
    train_dataset = None
    val_dataset = None
    dataset_cfg = config.get('dataset', {})
    paths_cfg = config.get('paths', {})
    data_dir = paths_cfg.get('data_dir', DEFAULT_DATA_DIR)

    if args.phase == 1:
        p1_transform_train = get_mnist_transforms(
            image_size=dataset_cfg['image_size'],
            augment=True
        )
        p1_transform_val = get_mnist_transforms(
            image_size=dataset_cfg['image_size'],
            augment=False
        )
        train_dataset = get_mnist_dataset(
            train=True,
            data_dir=data_dir,
            transform=p1_transform_train
        )
        val_dataset = get_mnist_dataset(
            train=False,
            data_dir=data_dir,
            transform=p1_transform_val
        )
    elif args.phase == 2:
        base_transform = get_mnist_transforms(
            image_size=28,
            augment=False
        )
        base_train_dataset = get_mnist_dataset(
            train=True,
            data_dir=data_dir,
            transform=base_transform
        )
        base_val_dataset = get_mnist_dataset(
            train=False,
            data_dir=data_dir,
            transform=base_transform
        )
        if base_train_dataset and base_val_dataset:
            train_set_length = len(base_train_dataset)
            val_set_length = len(base_val_dataset)
            p2_grid_size = dataset_cfg.get('image_size_phase2', 56)
            train_dataset = MNISTGridDataset(
                base_train_dataset,
                train_set_length,
                p2_grid_size
            )
            val_dataset = MNISTGridDataset(
                base_val_dataset,
                val_set_length,
                p2_grid_size
            )
    elif args.phase == 3:
        base_train_pil = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=None
        )
        base_val_pil = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=None
        )
        if base_train_pil and base_val_pil:
            train_set_length = len(base_train_pil)
            val_set_length = len(base_val_pil)
            train_dataset = MNISTDynamicDataset(
                base_train_pil,
                train_set_length,
                config
            )
            val_dataset = MNISTDynamicDataset(
                base_val_pil,
                val_set_length,
                config
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
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataloader = get_dataloader(
        val_dataset,
        eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    logger.info(f"--- Initializing PyTorch Model for Phase {args.phase} ---")
    model_cfg = config.get('model', {})
    tokenizer_cfg = config.get('tokenizer', {})

    if args.phase == 1 or args.phase == 2:
        img_size = dataset_cfg.get(
            f'image_size_phase{args.phase}',
            dataset_cfg['image_size']
        )
        patch_size = dataset_cfg.get('patch_size')
        num_outputs = dataset_cfg.get(
            f'num_outputs_phase{args.phase}', 1
        )
        num_classes = dataset_cfg.get('num_classes')
        model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=dataset_cfg['in_channels'],
            num_classes=num_classes,
            embed_dim=model_cfg['embed_dim'],
            depth=model_cfg['depth'],
            num_heads=model_cfg['num_heads'],
            mlp_ratio=model_cfg['mlp_ratio'],
            dropout=model_cfg.get('dropout', 0.1),
            num_outputs=num_outputs
        )
    elif args.phase == 3:
        img_size = dataset_cfg.get('image_size_phase3')
        patch_size = dataset_cfg.get('patch_size_phase3')
        decoder_vocab_size = tokenizer_cfg.get('vocab_size')
        model = EncoderDecoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=dataset_cfg['in_channels'],
            encoder_embed_dim=model_cfg['embed_dim'],
            encoder_depth=model_cfg['depth'],
            encoder_num_heads=model_cfg['num_heads'],
            decoder_vocab_size=decoder_vocab_size,
            decoder_embed_dim=model_cfg['decoder_embed_dim'],
            decoder_depth=model_cfg['decoder_depth'],
            decoder_num_heads=model_cfg['decoder_num_heads'],
            mlp_ratio=model_cfg['mlp_ratio'],
            dropout=model_cfg.get('dropout', 0.1)
        )
    else:
        logger.error(f"❌ Model instantiation for Phase {args.phase} "
                     f"not defined!")
        if run:
            run.finish(exit_code=1)
        return

    pad_token_id = tokenizer_cfg.get('pad_token_id', PAD_TOKEN_ID)
    if args.phase == 3:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        logger.info(
            f"Using CrossEntropyLoss with ignore_index={pad_token_id} "
            f"for Phase 3."
        )
    else:
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    optimizer_name = config.get("training", {}).get("optimizer", "AdamW")
    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd
        )
    logger.info(f"Optimizer: {optimizer_name} (LR={args.lr}, WD={args.wd})")

    phase_epochs = config.get("training", {}).get(
        f"phase{args.phase}", {}
    ).get("epochs", args.epochs)
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=phase_epochs,
        eta_min=args.lr * 0.01
    )
    logger.info(
        f"LR Scheduler: CosineAnnealingLR (T_max={phase_epochs}, "
        f"eta_min={args.lr * 0.01})"
    )

    logger.info("--- Starting PyTorch Training ---")
    start_time = time.time()

    # --- Train and collect full metrics history ---
    start_epoch = 0
    metrics_history = {
        "avg_train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": []
    }
    run_save_path = Path(args.model_save_dir) / (run.name if run else run_name)
    model_config = config.get('model', {})
    dataset_config = config.get('dataset', {})
    tokenizer_cfg = config.get('tokenizer', {})
    
    # --- Determine Checkpoint Load Path if Resuming ---
    checkpoint_load_path = None
    if args.resume_dir:
        checkpoint_load_path = Path(args.resume_dir)
        logger.info(f"Resume directory explicitly set: {checkpoint_load_path}")
    elif args.resume:
        checkpoint_load_path = run_save_path
        logger.info(f"Resume flag set. Attempting to load checkpoint from: {checkpoint_load_path}")
        if not checkpoint_load_path.exists():
            logger.warning(f"Checkpoint directory not found at {checkpoint_load_path}. Cannot resume.")
            checkpoint_load_path = None

    if args.resume and checkpoint_load_path:
        (
            model_state_dict,
            optimizer_state_dict,
            scheduler_state_dict,
            start_epoch,
            metrics_history,
            loaded_model_cfg,
            loaded_dataset_cfg,
            loaded_phase
        ) = load_checkpoint_pt(checkpoint_load_path, device)
        # Use loaded configs if available
        if loaded_model_cfg is not None:
            model_config = loaded_model_cfg
        if loaded_dataset_cfg is not None:
            dataset_config = loaded_dataset_cfg
        if loaded_phase is not None:
            args.phase = loaded_phase
        # Re-instantiate model with loaded config
        if args.phase == 1 or args.phase == 2:
            img_size = dataset_config.get(
                f'image_size_phase{args.phase}',
                dataset_config['image_size']
            )
            patch_size = dataset_config.get('patch_size')
            num_outputs = dataset_config.get(
                f'num_outputs_phase{args.phase}', 1
            )
            num_classes = dataset_config.get('num_classes')
            model = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=dataset_config['in_channels'],
                num_classes=num_classes,
                embed_dim=model_config['embed_dim'],
                depth=model_config['depth'],
                num_heads=model_config['num_heads'],
                mlp_ratio=model_config['mlp_ratio'],
                dropout=model_config.get('dropout', 0.1),
                num_outputs=num_outputs
            )
        elif args.phase == 3:
            img_size = dataset_config.get('image_size_phase3')
            patch_size = dataset_config.get('patch_size_phase3')
            decoder_vocab_size = tokenizer_cfg.get('vocab_size')
            model = EncoderDecoderViT(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=dataset_config['in_channels'],
                encoder_embed_dim=model_config['embed_dim'],
                encoder_depth=model_config['depth'],
                encoder_num_heads=model_config['num_heads'],
                decoder_vocab_size=decoder_vocab_size,
                decoder_embed_dim=model_config['decoder_embed_dim'],
                decoder_depth=model_config['decoder_depth'],
                decoder_num_heads=model_config['decoder_num_heads'],
                mlp_ratio=model_config['mlp_ratio'],
                dropout=model_config.get('dropout', 0.1)
            )
        model.to(device)
        if model_state_dict is not None:
            model.load_state_dict(model_state_dict)
        # Recreate optimizer and scheduler
        optimizer_name = config.get("training", {}).get("optimizer", "AdamW")
        if optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.wd
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.wd
            )
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
            for state_val in optimizer.state.values():
                for k, v in state_val.items():
                    if isinstance(v, torch.Tensor):
                        state_val[k] = v.to(device)
        phase_epochs = config.get("training", {}).get(
            f"phase{args.phase}", {}
        ).get("epochs", args.epochs)
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=phase_epochs,
            eta_min=args.lr * 0.01
        )
        if scheduler_state_dict is not None:
            lr_scheduler.load_state_dict(scheduler_state_dict)
    epochs_to_run = args.epochs - start_epoch
    if epochs_to_run <= 0:
        logger.info("Training already completed based on checkpoint.")
    else:
        logger.info(f"Training from epoch {start_epoch} for "
                    f"{epochs_to_run} more epochs.")
        metrics_history_update = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            start_epoch=start_epoch,
            initial_metrics_history=metrics_history,
            model_save_dir=args.model_save_dir,
            config=config,
            phase=args.phase,
            run_name=run.name if run else run_name,
            wandb_run=run,
            lr_scheduler=lr_scheduler,
            resume_from_checkpoint=False
        )
        metrics_history = metrics_history_update

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
            artifact_name = f"mnist_vit_PyTorch_final_{run.id}"
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