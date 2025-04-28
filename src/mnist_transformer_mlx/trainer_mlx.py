# MNIST Digit Classifier (Transformer) - MLX Version
# File: src/mnist_transformer_mlx/trainer_mlx.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Implements the training & evaluation logic using MLX.
# Created: 2025-04-28
# Updated: 2025-04-28

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np  # For batching
import time
import math  # For math.ceil
import os
import sys
from pathlib import Path
from tqdm import tqdm 
from typing import List, Dict, Optional, Any, Tuple

# --- Add project root for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent  # Go up two levels
if str(project_root) not in sys.path:
    print(f"🚂 [trainer_mlx.py] Adding project root to sys.path: "
          f"{project_root}")
    sys.path.insert(0, str(project_root))

from utils import logger

# W&B Import Handling
try:
    import wandb
except ImportError:
    logger.warning("⚠️ wandb not installed. W&B logging disabled.")
    wandb = None

# --- Loss and Accuracy Calculation ---

def calculate_loss_acc_mlx(
    model: nn.Module,
    images: mx.array,  # Shape (B, H, W, C)
    labels: mx.array,  # Shape (B,) for P1, (B, 4) for P2, (B, N_patches) for P3
    phase: int = 1
) -> Tuple[mx.array, mx.array]:
    """
    Performs forward pass, calculates loss and accuracy for MLX model.

    Args:
        model: The MLX model.
        images: Input image batch.
        labels: Target labels batch.
        phase: Current training phase (1, 2, or 3).

    Returns:
        Tuple containing:
        - Scalar loss value for the batch.
        - Scalar accuracy value for the batch (0.0 to 1.0).
    """
    logits = model(images)  # Shape (B, C) or (B, O, C)
    num_classes = logits.shape[-1]  # Last dimension is class count

    # Adapt loss calculation based on phase
    if phase == 1:
        # Standard Cross Entropy
        # logits: (B, C=10), labels: (B,)
        loss = mx.mean(nn.losses.cross_entropy(logits, labels))
        predicted = mx.argmax(logits, axis=-1)  # (B,)
        correct = mx.sum(predicted == labels)
        total = labels.size
    elif phase == 2:
        # logits: (B, O=4, C=10), labels: (B, O=4)
        # Reshape for cross_entropy: (B*O, C) and (B*O,)
        batch_size, num_outputs, _ = logits.shape
        loss = mx.mean(
            nn.losses.cross_entropy(
                logits.reshape(-1, num_classes), labels.reshape(-1)
            )
        )
        predicted = mx.argmax(logits, axis=-1)  # (B, O)
        correct = mx.sum(predicted == labels)
        total = labels.size  # B * O
    elif phase == 3:
        # logits: (B, N_patches, C=11), labels: (B, N_patches)
        batch_size, num_outputs, _ = logits.shape
        loss = mx.mean(
            nn.losses.cross_entropy(
                logits.reshape(-1, num_classes), labels.reshape(-1)
            )
        )
        predicted = mx.argmax(logits, axis=-1)  # (B, N_patches)
        correct = mx.sum(predicted == labels)
        total = labels.size  # B * N_patches
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    accuracy = correct / total if total > 0 else mx.array(0.0)
    return loss, accuracy

# --- MLX Training Epoch ---

def train_epoch_mlx(
    model: nn.Module,
    optimizer: optim.Optimizer,
    images_full: mx.array,  # Full training image set
    labels_full: mx.array,  # Full training label set
    batch_size: int,
    epoch_num: int,
    total_epochs: int,
    wandb_run: Optional[Any] = None,
    log_frequency: int = 100,
    phase: int = 1
) -> Tuple[nn.Module, float]:
    """ Trains the MLX model for one epoch using manual batching. """
    model.train()
    total_loss = 0.0
    num_samples = images_full.shape[0]
    num_batches = math.ceil(num_samples / batch_size)

    # --- Create loss and grad function ---
    loss_and_grad_fn = nn.value_and_grad(model, calculate_loss_acc_mlx)

    # Manual shuffling of indices for batches
    perm = np.random.permutation(num_samples)

    # Use tqdm for progress bar
    data_iterator = tqdm(
        range(int(num_batches)),
        desc=f"Epoch {epoch_num+1}/{total_epochs}",
        leave=False,
        unit="batch"
    )

    for i in data_iterator:
        # Manual batching
        np_batch_indices = perm[i * batch_size : (i + 1) * batch_size]

        # --- 👇 Convert indices to MLX array ---
        mlx_batch_indices = mx.array(np_batch_indices)

        # Index using the MLX array
        batch_images = images_full[mlx_batch_indices]
        batch_labels = labels_full[mlx_batch_indices]

        # Get loss and gradients for the batch
        (loss, acc), grads = loss_and_grad_fn(
            model, batch_images, batch_labels, phase=phase
        )

        # Update model parameters and optimizer state
        optimizer.update(model, grads)

        # Explicit evaluation is needed for gradients and updates in MLX
        mx.eval(loss, model.parameters(), optimizer.state)

        # --- Logging ---
        batch_loss = loss.item()
        batch_acc = acc.item() * 100.0
        total_loss += batch_loss
        data_iterator.set_postfix(
            loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.2f}%"
        )

        if (
            wandb is not None and wandb_run is not None
            and i % log_frequency == 0
        ):
            global_step = epoch_num * num_batches + i
            try:
                wandb_run.log({
                    "batch_loss": batch_loss,
                    "batch_accuracy": batch_acc,
                    "learning_rate": optimizer.learning_rate.item(),
                    "epoch": (epoch_num + (i / num_batches)),
                    "global_step": global_step
                })
            except Exception as e:
                logger.warning(f"⚠️ Failed W&B batch log: {e}")

    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return model, average_loss

# --- MLX Evaluation Function ---

def evaluate_model_mlx(
    model: nn.Module,
    images_full: mx.array,  # Full validation/test image set
    labels_full: mx.array,  # Full validation/test label set
    batch_size: int,
    phase: int = 1
) -> Dict[str, float]:
    """ Evaluates the MLX model. """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_samples = images_full.shape[0]
    num_batches = math.ceil(num_samples / batch_size)

    if num_batches == 0:
        logger.warning("⚠️ Evaluation data is empty.")
        return {"val_loss": 0.0, "val_accuracy": 0.0}

    logger.info("🧪 Starting evaluation...")
    data_iterator = tqdm(
        range(int(num_batches)),
        desc="Evaluating",
        leave=False,
        unit="batch"
    )

    for i in data_iterator:
        # Generate batch indices with NumPy
        np_batch_indices = np.arange(
            i * batch_size, min(num_samples, (i + 1) * batch_size)
        )

        # --- 👇 Convert indices to MLX array ---
        mlx_batch_indices = mx.array(np_batch_indices)

        # Index using the MLX array
        batch_images = images_full[mlx_batch_indices]
        batch_labels = labels_full[mlx_batch_indices]

        # Get loss and accuracy for the batch
        loss, acc = calculate_loss_acc_mlx(
            model, batch_images, batch_labels, phase=phase
        )

        # Evaluate the computation
        mx.eval(loss, acc)

        # Accumulate results
        total_loss += loss.item() * len(np_batch_indices)
        total_correct += acc.item() * batch_labels.size
        total_samples += batch_labels.size

        data_iterator.set_postfix(
            loss=f"{loss.item():.4f}", acc=f"{acc.item()*100:.2f}%"
        )

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = (
        (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    )

    logger.info(
        f"🧪 Evaluation finished. Avg Loss: {avg_loss:.4f}, "
        f"Accuracy: {accuracy:.2f}% ({int(total_correct)}/{total_samples})"
    )
    return {"val_loss": avg_loss, "val_accuracy": accuracy}

# --- Main Training Orchestrator Function (MLX) ---

def train_model_mlx(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_images: mx.array,
    train_labels: mx.array,
    val_images: mx.array,
    val_labels: mx.array,
    epochs: int,
    batch_size: int,
    model_save_dir: str | Path,
    run_name: str = "mnist_vit_mlx_run",
    wandb_run: Optional[Any] = None,
    phase: int = 1
) -> Tuple[List[float], Dict]:
    """ Orchestrates the overall MLX model training process. """
    logger.info(
        f"🚀 Starting MLX Model Training: Run='{run_name}', Phase={phase}"
    )
    logger.info(f"   Epochs: {epochs}")

    epoch_train_losses = []
    last_val_metrics = {}

    if wandb is not None and wandb_run is not None:
        logger.info("📊 W&B logging enabled (manual metric logging).")

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # --- Training Epoch ---
        model, avg_train_loss = train_epoch_mlx(
            model=model,
            optimizer=optimizer,
            images_full=train_images,
            labels_full=train_labels,
            batch_size=batch_size,
            epoch_num=epoch,
            total_epochs=epochs,
            wandb_run=wandb_run,
            phase=phase
        )
        epoch_train_losses.append(avg_train_loss)
        epoch_time = time.time() - epoch_start_time

        # --- Validation Step ---
        val_metrics = evaluate_model_mlx(
            model=model,
            images_full=val_images,
            labels_full=val_labels,
            batch_size=batch_size * 2,
            phase=phase
        )
        last_val_metrics = val_metrics

        # Log epoch summary
        log_str = (
            f"✅ Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['val_accuracy']:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )
        logger.info(log_str)

        # --- Log Epoch Metrics to W&B ---
        if wandb is not None and wandb_run is not None:
            try:
                wandb_log = {
                    "epoch": epoch + 1,
                    "avg_train_loss": avg_train_loss,
                    "val_loss": val_metrics['val_loss'],
                    "val_accuracy": val_metrics['val_accuracy'],
                    "epoch_time_sec": epoch_time,
                    "learning_rate": optimizer.learning_rate.item()
                }
                wandb_run.log(wandb_log)
            except Exception as e:
                logger.error(f"❌ W&B epoch log failed: {e}")

    logger.info("🏁 MLX Training finished.")

    # --- Save Final Model Weights ---
    final_save_dir = Path(model_save_dir) / run_name
    try:
        final_save_dir.mkdir(parents=True, exist_ok=True)
        model_path = final_save_dir / "model_weights.safetensors"
        model.save_weights(str(model_path))
        logger.info(f"💾 Final model weights saved to: {model_path}")
    except Exception as e:
        logger.error(
            f"❌ Failed to save final MLX model weights: {e}", exc_info=True
        )

    return epoch_train_losses, last_val_metrics

# --- Test Block ---
# (Skipping complex test block for MLX trainer; test via main script) ---
# if __name__ == "__main__":
#    logger.info("🧪 Running trainer_mlx.py directly is complex. "
#                "Test via train script.")