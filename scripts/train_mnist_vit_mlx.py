# MNIST Digit Classifier (Transformer) - MLX Version
# File: scripts/train_mnist_vit_mlx.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Main script to train the ViT model on MNIST using MLX.
# Created: 2025-04-28
# Updated: 2025-04-30

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
from tqdm import tqdm

# --- Add project root to sys.path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    print(f"🚀 [train_script_mlx] Adding project root to sys.path: "
          f"{project_root}")
    sys.path.insert(0, project_root)

# --- Project-specific imports ---
from utils import (
    logger, load_config, save_metrics, plot_metrics
)
from src.mnist_transformer_mlx.dataset_mlx import (
    get_mnist_data_arrays,
    MNISTGridDatasetMLX,
    MNISTDynamicDatasetMLX,
    numpy_normalize
)
from src.mnist_transformer_mlx.model_mlx import (
    VisionTransformerMLX, EncoderDecoderViTMLX
)
from src.mnist_transformer_mlx.trainer_mlx import train_model_mlx

# W&B Import
try:
    import wandb
except ImportError:
    logger.warning("⚠️ wandb not installed. Experiment tracking disabled.")
    wandb = None

def parse_args(
    config: dict
) -> argparse.Namespace:
    """
    Parses command-line arguments, using config for defaults (MLX version).

    Args:
        config (dict): Configuration dictionary loaded from YAML.

    Returns:
        argparse.Namespace: Parsed arguments with defaults from config.
    """
    parser = argparse.ArgumentParser(
        description="Train MNIST Vision Transformer (MLX)."
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
    train_cfg = config.get("training", {}).get(
        train_cfg_key, config.get("training", {})
    )

    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    paths_cfg = config.get('paths', {})
    eval_cfg = config.get('evaluation', {})
    tokenizer_cfg = config.get('tokenizer', {})

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
        default='mnist-vit-transformer-mlx',
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
        '--seed',
        type=int,
        default=42,
        help='Random seed.'
    )

    args = parser.parse_args()

    logger.info("--- Effective Configuration (MLX) ---")
    for arg, value in vars(args).items():
        logger.info(f"  --{arg.replace('_', '-'):<20}: {value}")
    logger.info("------------------------------------")
    return args

def pre_generate_synthetic_data(
    DatasetClass,
    base_dataset_np_tuple,
    length,
    config,
    phase
):
    """
    Pre-generates the full synthetic dataset using the Dataset class.

    Args:
        DatasetClass (type): Dataset class to use for generation.
        base_dataset_np_tuple (tuple): Tuple of (images, labels) numpy arrays.
        length (int): Number of samples to generate.
        config (dict): Configuration dictionary.
        phase (int): Training phase (2 or 3).

    Returns:
        tuple: (images_mlx, labels_mlx) as MLX arrays, or (None, None) on error.
    """
    logger.info(
        f"Pre-generating {length} Phase {phase} MLX samples..."
    )
    base_images_np, base_labels_np = base_dataset_np_tuple

    if DatasetClass == MNISTGridDatasetMLX:
        grid_size = config.get('dataset', {}).get('image_size_phase2', 56)
        temp_dataset = DatasetClass(
            base_images_np,
            base_labels_np,
            length,
            grid_size=grid_size
        )
    elif DatasetClass == MNISTDynamicDatasetMLX:
        try:
            from PIL import Image
            base_images_pil = [
                Image.fromarray(np.squeeze(img)) for img in base_images_np
            ]
            temp_dataset = DatasetClass(
                base_images_pil,
                base_labels_np,
                length,
                config=config
            )
        except Exception as e:
            logger.error(
                f"Failed to prepare PIL images for P3 dataset: {e}"
            )
            return None, None
    else:
        logger.error(
            f"❌ Unknown DatasetClass for pre-generation: {DatasetClass}"
        )
        return None, None

    all_images_mlx = []
    all_labels_mlx = []
    for i in tqdm(
        range(length),
        desc=f"Generating Phase {phase} Data"
    ):
        try:
            img, lbl = temp_dataset[i]
            if isinstance(img, mx.array) and isinstance(lbl, mx.array):
                all_images_mlx.append(img)
                all_labels_mlx.append(lbl)
            else:
                logger.warning(
                    f"Item {i} from generator not MLX. Skipping."
                )
        except Exception as e:
            logger.error(
                f"Error during generation item {i}: {e}", exc_info=True
            )

    if (not all_images_mlx or not all_labels_mlx or
            len(all_images_mlx) != length):
        logger.error(
            f"❌ Failed to generate sufficient valid samples for "
            f"Phase {phase}."
        )
        return None, None

    try:
        images_mlx = mx.stack(all_images_mlx, axis=0)
        labels_mlx = mx.stack(all_labels_mlx, axis=0)
        logger.info(
            f"✅ Pre-generated Phase {phase} Data - Images: "
            f"{images_mlx.shape}, Labels: {labels_mlx.shape}"
        )
        return images_mlx, labels_mlx
    except Exception as e:
        logger.error(
            f"❌ Failed stacking generated samples for Phase {phase}: {e}",
            exc_info=True
        )
        return None, None

def main():
    """
    Main entry point for training the MNIST Vision Transformer (MLX).

    Loads configuration, parses arguments, prepares data, initializes model,
    and runs training and logging.

    Steps:
        1. Load configuration and parse arguments.
        2. Set random seeds for reproducibility.
        3. Initialize W&B run if enabled.
        4. Load or generate training and validation data.
        5. Instantiate the model based on phase.
        6. Set up optimizer and (optional) scheduler.
        7. Train the model and log metrics.
        8. Save results, plot metrics, and log artifacts.
    """
    config = load_config()
    if config is None:
        return
    args = parse_args(config)

    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)
    logger.info(f"🌱 Seed set to {args.seed}")

    run = None
    run_name_base = (
        f"Phase{args.phase}_E{args.epochs}_LR{args.lr}_B{args.batch_size}"
    )
    run_name = f"MLX_{run_name_base}_ViT"
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
            logger.info(
                f"📊 Initialized W&B run: {run.name} ({run.url})"
            )
        except Exception as e:
            logger.error(
                f"❌ Failed W&B init: {e}", exc_info=True
            )
            run = None
    else:
        logger.info("📊 W&B logging disabled.")
        run_name = (
            f"MLX_{run_name_base}_local"
            if args.wandb_run_name is None else args.wandb_run_name
        )

    # --- Load or Generate Data ---
    logger.info(
        f"--- Loading/Generating Data for Phase {args.phase} (MLX) ---"
    )
    train_images = val_images = train_labels = val_labels = None
    dataset_cfg = config.get('dataset', {})
    paths_cfg = config.get('paths', {})
    train_cfg = config.get("training", {}).get(f"phase{args.phase}", {})
    data_dir = paths_cfg.get('data_dir', None)

    base_train_np = get_mnist_data_arrays(
        train=True,
        data_dir=data_dir
    )
    base_val_np = get_mnist_data_arrays(
        train=False,
        data_dir=data_dir
    )
    if base_train_np is None or base_val_np is None:
        logger.error(
            "❌ Failed to load base MNIST NumPy data. Exiting."
        )
        sys.exit(1)

    if args.phase == 1:
        logger.info("Normalizing and converting base data for Phase 1...")
        train_images = mx.array(numpy_normalize(base_train_np[0]))
        train_labels = mx.array(base_train_np[1])
        val_images = mx.array(numpy_normalize(base_val_np[0]))
        val_labels = mx.array(base_val_np[1])
    elif args.phase == 2 or args.phase == 3:
        train_set_length = train_cfg.get(
            'num_train_samples',
            len(base_train_np[0])
        )
        val_set_length = train_cfg.get(
            'num_val_samples',
            len(base_val_np[0])
        )
        logger.info(
            f"Target synthetic dataset size: "
            f"Train={train_set_length}, Val={val_set_length}"
        )
        DatasetClass = (
            MNISTGridDatasetMLX if args.phase == 2
            else MNISTDynamicDatasetMLX
        )
        train_images, train_labels = pre_generate_synthetic_data(
            DatasetClass=DatasetClass,
            base_dataset_np_tuple=base_train_np,
            length=train_set_length,
            config=config,
            phase=args.phase
        )
        val_images, val_labels = pre_generate_synthetic_data(
            DatasetClass=DatasetClass,
            base_dataset_np_tuple=base_val_np,
            length=val_set_length,
            config=config,
            phase=args.phase
        )
    else:
        logger.error(f"❌ Invalid phase specified: {args.phase}")
        if run:
            run.finish(exit_code=1)
        sys.exit(1)

    if train_images is None or val_images is None:
        logger.error(
            "❌ Failed to prepare MLX datasets for training. Exiting."
        )
        if run:
            run.finish(exit_code=1)
        sys.exit(1)

    logger.info(
        f"Prepared Train Data - Images: {train_images.shape}, "
        f"Labels: {train_labels.shape}"
    )
    logger.info(
        f"Prepared Val Data   - Images: {val_images.shape}, "
        f"Labels: {val_labels.shape}"
    )
    del base_train_np, base_val_np

    logger.info(
        f"--- Initializing MLX Model for Phase {args.phase} ---"
    )
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    tokenizer_cfg = config.get('tokenizer', {})
    model = None
    if args.phase == 1 or args.phase == 2:
        img_size = dataset_cfg.get(
            f'image_size_phase{args.phase}',
            dataset_cfg.get('image_size')
        )
        patch_size = dataset_cfg.get('patch_size')
        num_outputs = dataset_cfg.get(
            f'num_outputs_phase{args.phase}',
            1
        )
        num_classes = dataset_cfg.get('num_classes')
        model = VisionTransformerMLX(
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
        max_seq_len = dataset_cfg.get('max_seq_len')
        model = EncoderDecoderViTMLX(
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
            max_seq_len=max_seq_len,
            mlp_ratio=model_cfg['mlp_ratio'],
            dropout=model_cfg.get('dropout', 0.1)
        )
    else:
        logger.error(
            f"❌ Cannot instantiate model for invalid phase: {args.phase}"
        )
        if run:
            run.finish(exit_code=1)
        sys.exit(1)

    mx.eval(model.parameters())
    leaves = tree_flatten(model.parameters())
    nparams = sum(arr.size for _, arr in leaves)
    logger.info(f"Model parameter count: {nparams / 1e6:.3f} M")

    optimizer_name = config.get("training", {}).get("optimizer", "AdamW")
    lr = args.lr
    wd = args.wd
    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(
            learning_rate=lr,
            weight_decay=wd
        )
    else:
        optimizer = optim.Adam(
            learning_rate=lr
        )
    logger.info(
        f"Optimizer: {optimizer_name} (LR={lr}, "
        f"WD={wd if optimizer_name.lower()=='adamw' else 'N/A'})"
    )

    lr_scheduler = None
    logger.info(f"LR Scheduler: {'None'}")

    logger.info("--- Starting MLX Training ---")
    start_time = time.time()

    metrics_history, last_val_metrics = train_model_mlx(
        model=model,
        optimizer=optimizer,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_dir=args.model_save_dir,
        config=config,
        phase=args.phase,
        run_name=run.name if run else run_name,
        wandb_run=run,
    )

    training_time = time.time() - start_time
    logger.info(f"Total Training Time: {training_time:.2f} seconds")

    logger.info("--- Finalizing Run ---")
    logger.info(f"Final Validation Metrics: {last_val_metrics}")
    run_save_path = Path(args.model_save_dir) / (
        run.name if run else run_name
    )
    metrics_file = save_metrics(metrics_history, run_save_path)
    plot_file = plot_metrics(metrics_history, run_save_path)
    model_file = run_save_path / "model_weights.safetensors"

    if run:
        logger.info("☁️ Logging final artifacts to W&B...")
        try:
            artifact_name = f"mnist_vit_mlx_final_{run.id}"
            final_artifact = wandb.Artifact(
                artifact_name,
                type="model"
            )
            if model_file.exists():
                final_artifact.add_file(str(model_file))
            else:
                logger.warning(
                    f"MLX Model weights file {model_file} not found."
                )
            if metrics_file and Path(metrics_file).exists():
                final_artifact.add_file(metrics_file)
            if plot_file and Path(plot_file).exists():
                final_artifact.add_file(plot_file)
            config_log_path = Path(args.config_path)
            if config_log_path.exists():
                final_artifact.add_file(str(config_log_path))
            run.log_artifact(final_artifact)
            logger.info(
                "  Logged final model weights, results, and config artifact."
            )
        except Exception as e:
            logger.error(
                f"❌ Failed W&B artifact logging: {e}", exc_info=True
            )
        run.finish()
        logger.info("☁️ W&B run finished.")

    logger.info("✅ MLX Training script completed.")

if __name__ == "__main__":
    main()