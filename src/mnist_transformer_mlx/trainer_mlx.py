# MNIST Digit Classifier (Transformer) - MLX Version
# File: src/mnist_transformer_mlx/trainer_mlx.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Implements the training & evaluation logic using MLX (All Phases).
# Created: 2025-04-28
# Updated: 2025-04-30

import pickle # For saving python objects like optimizer state, epoch num
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
import math
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from tqdm import tqdm  # Ensure this is imported correctly

# --- Add project root ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent
if str(project_root) not in sys.path:
    print(f"🚂 [trainer_mlx.py] Adding project root: {project_root}")
    sys.path.insert(0, str(project_root))

from utils import logger
# Import tokenizer info for PAD ID
try:
    from utils.tokenizer_utils import PAD_TOKEN_ID
except ImportError:
    logger.warning("⚠️ Tokenizer utils or PAD_TOKEN_ID not found.")
    PAD_TOKEN_ID = 0  # Default assumption

# W&B Import Handling
try:
    import wandb
except ImportError:
    logger.warning("⚠️ wandb not installed.")
    wandb = None

# --- Loss and Accuracy Calculation (Handles Phases) ---

def calculate_loss_acc_mlx(
    model: nn.Module,
    images: mx.array,
    labels: mx.array,
    phase: int = 1,
    pad_token_id: int = PAD_TOKEN_ID,
) -> Tuple[mx.array, mx.array]:
    """
    Calculates loss and accuracy for MLX model, adapted for phases.

    Handles different model inputs and output shapes for each phase.
    For Phase 3, calculates loss and accuracy ignoring padding tokens.

    Args:
        model (nn.Module): The MLX model (either VisionTransformerMLX or 
            EncoderDecoderViTMLX).
        images (mx.array): Input image batch (B, H, W, C). Only used in 
            Phase 3 if model is EncoderDecoder.
        labels (mx.array): Target labels batch. Shape depends on phase:
            - Phase 1: (B,) - Single digit labels.
            - Phase 2: (B, 4) - Four digit labels for the grid.
            - Phase 3: (B, SeqLen) - Full target sequence including 
              START/END/PAD tokens.
        phase (int): Current training phase (1, 2, or 3).
        pad_token_id (int): The ID used for padding in sequences 
            (relevant for Phase 3).

    Returns:
        Tuple[mx.array, mx.array]: 
            - Scalar loss value for the batch (averaged over non-pad tokens 
              for P3).
            - Scalar accuracy value for the batch (averaged over non-pad 
              tokens for P3).
    """
    # --- Adapt forward pass based on phase ---
    if phase == 1 or phase == 2:
        logits = model(images)
    elif phase == 3:
        decoder_input = labels[:, :-1]
        logits = model(images, decoder_input)
    else:
        raise ValueError(f"Unsupported phase for loss calculation: {phase}")
    # --- End Adapt forward pass ---

    num_classes = logits.shape[-1]

    # --- Adapt Loss & Accuracy Calculation ---
    if phase == 1:
        per_item_loss = nn.losses.cross_entropy(
            logits, labels, reduction='none')
        loss = mx.mean(per_item_loss)
        predicted = mx.argmax(logits, axis=-1)
        correct = mx.sum(predicted == labels)
        total = labels.size
    elif phase == 2:
        logits_flat = logits.reshape(-1, num_classes)
        labels_flat = labels.reshape(-1)
        per_item_loss = nn.losses.cross_entropy(
            logits_flat, labels_flat, reduction='none')
        loss = mx.mean(per_item_loss)
        predicted = mx.argmax(logits, axis=-1)
        correct = mx.sum(predicted == labels)
        total = labels.size
    elif phase == 3:
        decoder_target = labels[:, 1:]
        logits_flat = logits.reshape(-1, num_classes)
        target_flat = decoder_target.reshape(-1)
        per_token_loss = nn.losses.cross_entropy(
            logits_flat, target_flat, reduction='none')
        mask = (target_flat != pad_token_id)
        predicted_flat = mx.argmax(logits_flat, axis=-1)
        correct = mx.sum((predicted_flat == target_flat) * mask)
        total = mx.sum(mask).item()
        loss = mx.sum(per_token_loss * mask) / total if total > 0 \
            else mx.array(0.0)
    else:
        loss = mx.array(0.0)
        correct = mx.array(0)
        total = 0

    accuracy = correct / total if total > 0 else mx.array(0.0)

    if loss.ndim != 0:
        loss = mx.mean(loss)

    return loss, accuracy

# --- MLX Training Epoch ---
def train_epoch_mlx(
    model: nn.Module,
    optimizer: optim.Optimizer,
    images_full: mx.array,
    labels_full: mx.array,
    batch_size: int,
    epoch_num: int,
    total_epochs: int,
    wandb_run: Optional[Any] = None,
    log_frequency: int = 100,
    phase: int = 1,
    pad_token_id: int = PAD_TOKEN_ID
) -> float:
    """
    Trains the MLX model for one epoch using manual batching and shuffling.

    Handles gradient calculation and optimizer updates. Logs batch loss.

    Args:
        model (nn.Module): The MLX model to train.
        optimizer (optim.Optimizer): The MLX optimizer.
        images_full (mx.array): Full training image dataset as an mx.array.
        labels_full (mx.array): Full training label dataset as an mx.array.
        batch_size (int): The number of samples per batch.
        epoch_num (int): The current epoch number (0-indexed).
        total_epochs (int): The total number of epochs for training.
        wandb_run (Optional[Any]): Optional W&B run object for logging.
        log_frequency (int): How often (in batches) to log to W&B.
        phase (int): Current training phase (1, 2, or 3).
        pad_token_id (int): Token ID for padding (used in loss wrapper for 
            Phase 3).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_samples = images_full.shape[0]
    num_batches = math.ceil(num_samples / batch_size)
    if num_batches == 0:
        logger.warning("⚠️ Training data is empty, skipping epoch.")
        return 0.0

    def loss_fn_wrapper(
        model,
        img,
        lbl
    ):
        loss, _ = calculate_loss_acc_mlx(
            model, img, lbl, phase, pad_token_id
        )
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn_wrapper)

    perm = np.random.permutation(num_samples)
    data_iterator = tqdm(
        range(int(num_batches)),
        desc=f"Epoch {epoch_num+1}/{total_epochs} Train",
        leave=False,
        unit="batch"
    )

    for i in data_iterator:
        np_indices = perm[i * batch_size: (i + 1) * batch_size]
        mlx_indices = mx.array(np_indices)
        batch_images = images_full[mlx_indices]
        batch_labels = labels_full[mlx_indices]

        (loss), grads = loss_and_grad_fn(
            model, batch_images, batch_labels
        )

        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        batch_loss = loss.item()
        total_loss += batch_loss
        data_iterator.set_postfix(loss=f"{batch_loss:.4f}")

        if (
            wandb is not None and
            wandb_run is not None and
            i % log_frequency == 0
        ):
            global_step = epoch_num * num_batches + i
            log_dict = {
                "batch_loss": batch_loss,
                "learning_rate": optimizer.learning_rate.item(),
                "epoch_fractional": (epoch_num + (i / num_batches)),
                "global_step": global_step
            }
            try:
                wandb_run.log(log_dict)
            except Exception as e:
                logger.warning(f"⚠️ Failed W&B batch log: {e}")

    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss

# --- MLX Evaluation Function ---
def evaluate_model_mlx(
    model: nn.Module,
    images_full: mx.array,
    labels_full: mx.array,
    batch_size: int,
    phase: int = 1,
    pad_token_id: int = PAD_TOKEN_ID
) -> Dict[str, float]:
    """
    Evaluates the MLX model on a given dataset (validation or test).

    Adapts metrics calculation based on the training phase, correctly handling
    padding for Phase 3 accuracy and loss averaging.

    Args:
        model (nn.Module): The MLX model to evaluate.
        images_full (mx.array): Full evaluation image dataset as an mx.array.
        labels_full (mx.array): Full evaluation label dataset as an mx.array.
        batch_size (int): Batch size for evaluation.
        phase (int): Current training phase (1, 2, or 3).
        pad_token_id (int): Token ID for padding (used for Phase 3 metrics).

    Returns:
        Dict[str, float]: Dictionary containing average validation loss 
            ('val_loss') and average validation accuracy ('val_accuracy').
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_evaluated = 0
    num_samples = images_full.shape[0]
    num_batches = math.ceil(num_samples / batch_size)
    if num_batches == 0:
        logger.warning("⚠️ Evaluation data is empty.")
        return {"val_loss": 0.0, "val_accuracy": 0.0}

    logger.info(f"🧪 Starting evaluation (Phase {phase})...")
    data_iterator = tqdm(
        range(int(num_batches)),
        desc="Evaluating",
        leave=False,
        unit="batch"
    )

    for i in data_iterator:
        np_indices = np.arange(
            i * batch_size, min(num_samples, (i + 1) * batch_size)
        )
        mlx_indices = mx.array(np_indices)
        batch_images = images_full[mlx_indices]
        batch_labels = labels_full[mlx_indices]

        loss, acc = calculate_loss_acc_mlx(
            model, batch_images, batch_labels, phase, pad_token_id
        )
        mx.eval(loss, acc)

        current_batch_size = len(mlx_indices)
        if phase == 3:
            target_flat = batch_labels[:, 1:].reshape(-1)
            mask = (target_flat != pad_token_id)
            num_valid_in_batch = mx.sum(mask).item()
        else:
            num_valid_in_batch = batch_labels.size

        if num_valid_in_batch > 0:
            total_loss += loss.item() * num_valid_in_batch
            total_correct += acc.item() * num_valid_in_batch
            total_evaluated += num_valid_in_batch

        data_iterator.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{acc.item() * 100:.2f}%"
        )

    avg_loss = (
        total_loss / total_evaluated if total_evaluated > 0 else 0.0
    )
    avg_accuracy = (
        (total_correct / total_evaluated) * 100.0
        if total_evaluated > 0 else 0.0
    )

    logger.info(
        f"🧪 Eval finished. Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.2f}% "
        f"({int(total_correct)}/{int(total_evaluated)})"
    )
    return {"val_loss": avg_loss, "val_accuracy": avg_accuracy}

# --- Checkpointing Helpers ---

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics_history: Dict,
    save_dir: Path,
    is_best: bool = False
):
    """
    Save model weights, optimizer state, and epoch number to disk.

    Args:
        model (nn.Module): The MLX model whose weights to save.
        optimizer (optim.Optimizer): The optimizer whose state to save.
        epoch (int): The current epoch number.
        metrics_history (Dict): Dictionary of training/validation metrics.
        save_dir (Path): Directory to save checkpoint files.
        is_best (bool): If True, also save as best model weights.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_path = save_dir / "model_weights.safetensors"
    model.save_weights(str(weights_path))
    state_path = save_dir / "training_state.pkl"
    state = {
        'epoch': epoch + 1,
        'optimizer_state': optimizer.state,
        'metrics_history': metrics_history,
    }
    try:
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"💾 Checkpoint saved to {save_dir}")
    except Exception as e:
        logger.error(f"❌ Failed to save training state: {e}")
    if is_best:
        best_weights_path = save_dir.parent / "best_model_weights.safetensors"
        model.save_weights(str(best_weights_path))
        logger.info(f"🏆 Best model weights saved to {best_weights_path}")

def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    save_dir: Path
) -> Tuple[nn.Module, optim.Optimizer, int, Dict]:
    """
    Load model weights, optimizer state, and epoch from checkpoint.

    Args:
        model (nn.Module): The MLX model to load weights into.
        optimizer (optim.Optimizer): The optimizer to load state into.
        save_dir (Path): Directory containing checkpoint files.

    Returns:
        Tuple[nn.Module, optim.Optimizer, int, Dict]: 
            Updated model, optimizer, start epoch, and metrics history.
    """
    weights_path = save_dir / "model_weights.safetensors"
    state_path = save_dir / "training_state.pkl"
    start_epoch = 0
    metrics_history = {
        "avg_train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": []
    }
    if weights_path.exists() and state_path.exists():
        logger.info(f"♻️ Resuming training from checkpoint in {save_dir}")
        try:
            model.load_weights(str(weights_path))
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            optimizer.state = state['optimizer_state']
            start_epoch = state['epoch']
            metrics_history = state.get('metrics_history', metrics_history)
            logger.info(
                f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}."
            )
            mx.eval(model.parameters(), optimizer.state)
        except Exception as e:
            logger.error(
                f"❌ Failed to load checkpoint: {e}. Starting fresh.",
                exc_info=True
            )
            start_epoch = 0
            metrics_history = {
                "avg_train_loss": [],
                "val_loss": [],
                "val_accuracy": [],
                "learning_rate": []
            }
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
    return model, optimizer, start_epoch, metrics_history

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
    config: dict,
    phase: int = 1,
    run_name: str = "mnist_vit_mlx_run",
    wandb_run: Optional[Any] = None,
    resume_from_checkpoint: bool = True,
    save_every: int = 5
) -> Tuple[Dict[str, List[float]], Dict]:
    """
    Orchestrate the MLX model training process with checkpointing.

    Handles training, validation, checkpointing, and logging.

    Args:
        model (nn.Module): The MLX model to train.
        optimizer (optim.Optimizer): The optimizer for training.
        train_images (mx.array): Training images.
        train_labels (mx.array): Training labels.
        val_images (mx.array): Validation images.
        val_labels (mx.array): Validation labels.
        epochs (int): Total number of epochs to train.
        batch_size (int): Batch size for training.
        model_save_dir (str | Path): Directory to save checkpoints.
        config (dict): Configuration dictionary.
        phase (int): Training phase (1, 2, or 3).
        run_name (str): Name for the training run.
        wandb_run (Optional[Any]): Weights & Biases run object.
        resume_from_checkpoint (bool): Resume from checkpoint if available.
        save_every (int): Save checkpoint every N epochs.

    Returns:
        Tuple[Dict[str, List[float]], Dict]: 
            Metrics history and last validation metrics.
    """
    logger.info(
        f"🚀 Starting MLX Model Training: Run='{run_name}', Phase={phase}"
    )
    logger.info(f"   Target Epochs: {epochs}")
    run_save_path = Path(model_save_dir) / run_name
    start_epoch = 0
    metrics_history = {
        "avg_train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": []
    }
    if resume_from_checkpoint:
        model, optimizer, start_epoch, metrics_history = load_checkpoint(
            model, optimizer, run_save_path
        )
    logger.info(
        f"   Starting from Epoch: {start_epoch}, "
        f"Training until Epoch: {epochs}"
    )
    pad_token_id = config.get(
        'tokenizer', {}
    ).get('pad_token_id', PAD_TOKEN_ID)
    best_val_accuracy = 0.0
    if metrics_history["val_accuracy"]:
        best_val_accuracy = max(metrics_history["val_accuracy"])
    if start_epoch >= epochs:
        logger.warning(
            f"Start epoch ({start_epoch}) is >= target epochs ({epochs}). "
            "Training finished."
        )
        return metrics_history, metrics_history
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        avg_train_loss = train_epoch_mlx(
            model=model,
            optimizer=optimizer,
            images_full=train_images,
            labels_full=train_labels,
            batch_size=batch_size,
            epoch_num=epoch,
            total_epochs=epochs,
            wandb_run=wandb_run,
            phase=phase,
            pad_token_id=pad_token_id
        )
        metrics_history["avg_train_loss"].append(avg_train_loss)
        epoch_time = time.time() - epoch_start_time
        val_metrics = {}
        if val_images is not None and val_labels is not None:
            val_metrics = evaluate_model_mlx(
                model=model,
                images_full=val_images,
                labels_full=val_labels,
                batch_size=batch_size * 2,
                phase=phase,
                pad_token_id=pad_token_id
            )
            metrics_history["val_loss"].append(
                val_metrics.get('val_loss', float('nan'))
            )
            metrics_history["val_accuracy"].append(
                val_metrics.get('val_accuracy', float('nan'))
            )
        else:
            metrics_history["val_loss"].append(float('nan'))
            metrics_history["val_accuracy"].append(float('nan'))
        current_lr = optimizer.learning_rate.item()
        metrics_history["learning_rate"].append(current_lr)
        log_str = (
            f"✅ Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
        )
        for k, v in val_metrics.items():
            log_str += f"{k}: {v:.4f} | "
        log_str += f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s"
        logger.info(log_str)
        if wandb is not None and wandb_run is not None:
            try:
                log_dict_epoch = {
                    "epoch": epoch + 1,
                    "avg_train_loss": avg_train_loss,
                    **val_metrics
                }
                log_dict_epoch["learning_rate"] = current_lr
                log_dict_epoch["epoch_time_sec"] = epoch_time
                wandb_run.log(log_dict_epoch)
            except Exception as e:
                logger.error(f"❌ W&B epoch log failed: {e}")
        is_best = val_metrics.get('val_accuracy', -1) > best_val_accuracy
        if is_best:
            best_val_accuracy = val_metrics['val_accuracy']
            logger.info(
                f"🏆 New best validation accuracy: {best_val_accuracy:.2f}%"
            )
        if (
            (epoch + 1) % save_every == 0 or
            epoch == epochs - 1 or
            is_best
        ):
            save_checkpoint(
                model,
                optimizer,
                epoch,
                metrics_history,
                run_save_path,
                is_best=is_best
            )
    logger.info("🏁 MLX Training finished.")
    save_checkpoint(
        model, optimizer, epochs - 1, metrics_history, run_save_path
    )
    return metrics_history, val_metrics

# --- Test Block ---
if __name__ == "__main__":
    logger.info("🧪 Running trainer_mlx.py basic checks...")

    logger.info("\n--- Testing P1/P2 loss/acc calc ---")
    try:
        dummy_logits_p1 = mx.random.normal(shape=(4, 10))
        dummy_labels_p1 = mx.array([1, 5, 9, 3])
        dummy_model_p1 = lambda img: dummy_logits_p1
        loss_p1, acc_p1 = calculate_loss_acc_mlx(
            dummy_model_p1, None, dummy_labels_p1, phase=1
        )
        mx.eval(loss_p1, acc_p1)
        logger.info(
            f"P1 Dummy Loss: {loss_p1.item():.4f}, Acc: {acc_p1.item()*100:.2f}%"
        )

        dummy_logits_p2 = mx.random.normal(shape=(2, 4, 10))
        dummy_labels_p2 = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        dummy_model_p2 = lambda img: dummy_logits_p2
        loss_p2, acc_p2 = calculate_loss_acc_mlx(
            dummy_model_p2, None, dummy_labels_p2, phase=2
        )
        mx.eval(loss_p2, acc_p2)
        logger.info(
            f"P2 Dummy Loss: {loss_p2.item():.4f}, Acc: {acc_p2.item()*100:.2f}%"
        )
        logger.info("✅ P1/P2 calc OK.")
    except Exception as e:
        logger.error(f"❌ P1/P2 Calc Error: {e}", exc_info=True)

    # --- Add a simple check for phase 3 in calculate_loss_acc_mlx ---
    logger.info("\n--- Testing P3 loss/acc calc ---")
    try:
        _B, _S, _V = 2, 10, 13
        dummy_logits_p3 = mx.random.normal(shape=(_B, _S - 1, _V))  # B, Seq-1, V
        dummy_labels_p3 = mx.random.randint(1, _V - 2, (_B, _S)).astype(mx.uint32)
        dummy_labels_p3[:, -2:] = PAD_TOKEN_ID
        dummy_labels_p3[:, 0] = 1  # Assume 1 is START token
        mx.eval(dummy_labels_p3)
        dummy_model_p3 = lambda img, tgt_seq: dummy_logits_p3

        loss_p3, acc_p3 = calculate_loss_acc_mlx(
            dummy_model_p3,
            mx.zeros((_B, 32, 32, 1)),  # Dummy image input
            dummy_labels_p3,
            phase=3,
            pad_token_id=PAD_TOKEN_ID
        )
        mx.eval(loss_p3, acc_p3)
        logger.info(
            f"P3 Dummy Loss: {loss_p3.item():.4f}, Acc: {acc_p3.item()*100:.2f}%"
        )
        logger.info("✅ P3 calc OK.")
    except Exception as e:
        logger.error(f"❌ P3 Calc Error: {e}", exc_info=True)

    logger.info("\n✅ trainer_mlx.py basic checks finished.")
