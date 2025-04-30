# MNIST Digit Classifier (Transformer) - MLX Version
# File: src/mnist_transformer_mlx/trainer_mlx.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Implements the training & evaluation logic using MLX (All Phases).
# Created: 2025-04-28
# Updated: 2025-04-30

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
from tqdm import tqdm # Ensure this is imported correctly

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
    PAD_TOKEN_ID = 0 # Default assumption

# W&B Import Handling
try: import wandb
except ImportError: logger.warning("⚠️ wandb not installed."); wandb = None

# --- Loss and Accuracy Calculation (Handles Phases) ---

def calculate_loss_acc_mlx(
    model: nn.Module,
    images: mx.array, # (B, H, W, C)
    labels: mx.array, # (B,) P1; (B, 4) P2; (B, SeqLen) P3 (full seq w/ START/END)
    phase: int = 1,
    pad_token_id: int = PAD_TOKEN_ID,
) -> Tuple[mx.array, mx.array]:
    """
    Calculates loss and accuracy for MLX model, adapted for phases.

    Handles different model inputs and output shapes for each phase.
    For Phase 3, calculates loss and accuracy ignoring padding tokens.

    Args:
        model: The MLX model (either VisionTransformerMLX or EncoderDecoderViTMLX).
        images: Input image batch (B, H, W, C). Only used in Phase 3 if model is EncoderDecoder.
        labels: Target labels batch. Shape depends on phase:
                - Phase 1: (B,) - Single digit labels.
                - Phase 2: (B, 4) - Four digit labels for the grid.
                - Phase 3: (B, SeqLen) - Full target sequence including START/END/PAD tokens.
        phase: Current training phase (1, 2, or 3).
        pad_token_id: The ID used for padding in sequences (relevant for Phase 3).

    Returns:
        Tuple containing:
        - Scalar loss value for the batch (averaged over non-pad tokens for P3).
        - Scalar accuracy value for the batch (averaged over non-pad tokens for P3).
    """

    # --- Adapt forward pass based on phase ---
    if phase == 1 or phase == 2:
        # Encoder-only model forward pass
        logits = model(images) # (B, C) or (B, 4, C)
    elif phase == 3:
        # Encoder-Decoder: needs image and shifted target sequence
        # labels shape: (B, SeqLen) -> e.g., [START, d1, d2, END, PAD..]
        decoder_input = labels[:, :-1] # Input: [START, d1, d2, END, PAD..] (Remove last token)
        # Assuming the model instance passed is EncoderDecoderViTMLX for phase 3
        logits = model(images, decoder_input) # Output: (B, SeqLen-1, VocabSize)
    else:
        raise ValueError(f"Unsupported phase for loss calculation: {phase}")
    # --- End Adapt forward pass ---

    num_classes = logits.shape[-1] # Last dimension size

    # --- Adapt Loss & Accuracy Calculation ---
    if phase == 1:
        # logits: (B, C=10), labels: (B,)
        # Calculate loss per item, then mean
        per_item_loss = nn.losses.cross_entropy(logits, labels, reduction='none')
        loss = mx.mean(per_item_loss)
        predicted = mx.argmax(logits, axis=-1) # (B,)
        correct = mx.sum(predicted == labels)
        total = labels.size
    elif phase == 2:
        # logits: (B, O=4, C=10), labels: (B, O=4)
        logits_flat = logits.reshape(-1, num_classes) # (B*O, C)
        labels_flat = labels.reshape(-1) # (B*O,)
        # Calculate loss per item, then mean
        per_item_loss = nn.losses.cross_entropy(logits_flat, labels_flat, reduction='none')
        loss = mx.mean(per_item_loss)
        predicted = mx.argmax(logits, axis=-1) # (B, O)
        correct = mx.sum(predicted == labels)
        total = labels.size # B * O
    elif phase == 3:
        # logits: (B, SeqLen-1, V_dec), labels: (B, SeqLen)
        # Target should be shifted left: [d1, d2, END, PAD...]
        decoder_target = labels[:, 1:] # Target: (B, SeqLen-1)
        logits_flat = logits.reshape(-1, num_classes) # (B*(SeqLen-1), V_dec)
        target_flat = decoder_target.reshape(-1) # (B*(SeqLen-1),)

        # Calculate loss per token, ignoring padding index later
        per_token_loss = nn.losses.cross_entropy(logits_flat, target_flat, reduction='none')

        # --- Masking for Loss and Accuracy (Important!) ---
        mask = (target_flat != pad_token_id) # Bool array: True where not padding
        predicted_flat = mx.argmax(logits_flat, axis=-1) # (B*(SeqLen-1),)

        # Calculate correct only where mask is True
        correct = mx.sum((predicted_flat == target_flat) * mask)
        total = mx.sum(mask).item() # Count only non-pad tokens

        # Average loss only over non-padded tokens
        loss = mx.sum(per_token_loss * mask) / total if total > 0 else mx.array(0.0)
        # --- End Masking ---
    else: # Should not happen
        loss = mx.array(0.0); correct = mx.array(0); total = 0

    # Calculate average accuracy across valid samples/tokens
    accuracy = correct / total if total > 0 else mx.array(0.0)

    # Ensure loss is scalar before returning
    if loss.ndim != 0:
        loss = mx.mean(loss) # Should already be scalar due to logic above, but as safeguard

    return loss, accuracy


# --- MLX Training Epoch ---
def train_epoch_mlx(
    model: nn.Module, optimizer: optim.Optimizer,
    images_full: mx.array, labels_full: mx.array,
    batch_size: int, epoch_num: int, total_epochs: int,
    wandb_run: Optional[Any] = None, log_frequency: int = 100,
    phase: int = 1, pad_token_id: int = PAD_TOKEN_ID
) -> float: # Return average loss
    """
    Trains the MLX model for one epoch using manual batching and shuffling.

    Handles gradient calculation and optimizer updates. Logs batch loss.

    Args:
        model: The MLX model to train.
        optimizer: The MLX optimizer.
        images_full: Full training image dataset as an mx.array.
        labels_full: Full training label dataset as an mx.array.
        batch_size: The number of samples per batch.
        epoch_num: The current epoch number (0-indexed).
        total_epochs: The total number of epochs for training.
        wandb_run: Optional W&B run object for logging.
        log_frequency: How often (in batches) to log to W&B.
        phase: Current training phase (1, 2, or 3).
        pad_token_id: Token ID for padding (used in loss wrapper for Phase 3).

    Returns:
        The average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_samples = images_full.shape[0]
    num_batches = math.ceil(num_samples / batch_size)
    if num_batches == 0:
        logger.warning("⚠️ Training data is empty, skipping epoch.")
        return 0.0

    # --- Pass phase and pad_id to loss function ---
    def loss_fn_wrapper(model, img, lbl):
        # Only return loss for gradient calculation
        loss, _ = calculate_loss_acc_mlx(model, img, lbl, phase, pad_token_id)
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn_wrapper)
    # --- End loss fn wrapper ---

    perm = np.random.permutation(num_samples)
    data_iterator = tqdm(
        range(int(num_batches)),
        desc=f"Epoch {epoch_num+1}/{total_epochs} Train",
        leave=False,
        unit="batch"
    )

    for i in data_iterator:
        np_indices = perm[i * batch_size : (i + 1) * batch_size]
        mlx_indices = mx.array(np_indices)
        batch_images = images_full[mlx_indices]
        batch_labels = labels_full[mlx_indices]

        # Get loss and gradients
        (loss), grads = loss_and_grad_fn(model, batch_images, batch_labels)

        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state) # Evaluate loss and updates

        # Log batch loss
        batch_loss = loss.item()
        total_loss += batch_loss # Accumulate raw batch loss
        data_iterator.set_postfix(loss=f"{batch_loss:.4f}")

        # --- W&B Logging ---
        if wandb is not None and wandb_run is not None and i % log_frequency == 0:
             global_step = epoch_num * num_batches + i
             log_dict = {
                 "batch_loss": batch_loss,
                 "learning_rate": optimizer.learning_rate.item(),
                 "epoch_fractional": (epoch_num + (i / num_batches)),
                 "global_step": global_step
                 # Optional: Calculate and log batch accuracy here if needed frequently
                 # _, batch_acc_val = calculate_loss_acc_mlx(model, batch_images, batch_labels, phase, pad_token_id)
                 # mx.eval(batch_acc_val)
                 # log_dict["batch_accuracy"] = batch_acc_val.item() * 100.0
             }
             try:
                 wandb_run.log(log_dict)
             except Exception as e:
                 logger.warning(f"⚠️ Failed W&B batch log: {e}")

    # Return average loss across batches
    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss


# --- MLX Evaluation Function ---
def evaluate_model_mlx(
    model: nn.Module, images_full: mx.array, labels_full: mx.array,
    batch_size: int, phase: int = 1, pad_token_id: int = PAD_TOKEN_ID
) -> Dict[str, float]:
    """
    Evaluates the MLX model on a given dataset (validation or test).

    Adapts metrics calculation based on the training phase, correctly handling
    padding for Phase 3 accuracy and loss averaging.

    Args:
        model: The MLX model to evaluate.
        images_full: Full evaluation image dataset as an mx.array.
        labels_full: Full evaluation label dataset as an mx.array.
        batch_size: Batch size for evaluation.
        phase: Current training phase (1, 2, or 3).
        pad_token_id: Token ID for padding (used for Phase 3 metrics).

    Returns:
        A dictionary containing average validation loss ('val_loss')
        and average validation accuracy ('val_accuracy').
    """
    model.eval()
    total_loss = 0.0; total_correct = 0; total_evaluated = 0
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

    # No gradients needed for evaluation
    for i in data_iterator:
        np_indices = np.arange(i*batch_size, min(num_samples, (i+1)*batch_size))
        mlx_indices = mx.array(np_indices)
        batch_images = images_full[mlx_indices]
        batch_labels = labels_full[mlx_indices]

        # Use calculate_loss_acc_mlx to get both metrics
        loss, acc = calculate_loss_acc_mlx(
            model, batch_images, batch_labels, phase, pad_token_id
        )
        mx.eval(loss, acc) # Force computation

        # Accumulate results based on valid items/tokens
        current_batch_size = len(mlx_indices)
        if phase == 3:
            # For P3, accuracy is avg over non-pad tokens in the batch.
            # We need the count of non-pad tokens to scale correctly.
            target_flat = batch_labels[:, 1:].reshape(-1)
            mask = (target_flat != pad_token_id)
            num_valid_in_batch = mx.sum(mask).item()
        else:
            # For P1/P2, all items are valid.
            num_valid_in_batch = batch_labels.size

        if num_valid_in_batch > 0:
            # Accumulate weighted loss sum and correct count sum
            total_loss += loss.item() * num_valid_in_batch
            total_correct += acc.item() * num_valid_in_batch
            total_evaluated += num_valid_in_batch

        data_iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item()*100:.2f}%")

    # Final average metrics
    avg_loss = total_loss / total_evaluated if total_evaluated > 0 else 0.0
    avg_accuracy = (total_correct / total_evaluated) * 100.0 if total_evaluated > 0 else 0.0

    logger.info(f"🧪 Eval finished. Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.2f}% ({int(total_correct)}/{int(total_evaluated)})")
    return {"val_loss": avg_loss, "val_accuracy": avg_accuracy}


# --- Main Training Orchestrator Function (MLX) ---
def train_model_mlx(
    model: nn.Module, optimizer: optim.Optimizer,
    train_images: mx.array, train_labels: mx.array,
    val_images: mx.array, val_labels: mx.array,
    epochs: int, batch_size: int,
    model_save_dir: str | Path,
    config: dict, # Pass config for params
    phase: int = 1,
    run_name: str = "mnist_vit_mlx_run",
    wandb_run: Optional[Any] = None,
    # lr_scheduler_fn: Optional[Callable] = None # Placeholder if implementing scheduler
) -> Tuple[Dict[str, List[float]], Dict]: # Return history dict and last val metrics
    """
    Orchestrates the overall MLX model training process over multiple epochs.

    Handles epoch looping, calling training and evaluation functions,
    logging metrics, and saving the final model weights.

    Args:
        model: The MLX model instance.
        optimizer: The MLX optimizer instance.
        train_images: Full training image dataset (mx.array).
        train_labels: Full training label dataset (mx.array).
        val_images: Full validation image dataset (mx.array).
        val_labels: Full validation label dataset (mx.array).
        epochs: Total number of epochs to train.
        batch_size: Batch size for training and evaluation.
        model_save_dir: Directory to save the final model weights.
        config: Dictionary containing configuration parameters (e.g., tokenizer pad_id).
        phase: Current training phase (1, 2, or 3).
        run_name: Name for the training run (used for saving).
        wandb_run: Optional W&B run object for logging.

    Returns:
        A tuple containing:
        - metrics_history: A dictionary storing lists of metrics per epoch
          ('avg_train_loss', 'val_loss', 'val_accuracy', 'learning_rate').
        - last_val_metrics: A dictionary with the validation metrics from the final epoch.
    """
    logger.info(f"🚀 Starting MLX Model Training: Run='{run_name}', Phase={phase}")
    logger.info(f"   Epochs: {epochs}")

    # --- Get PAD token ID from config ---
    pad_token_id = config.get('tokenizer', {}).get('pad_token_id', PAD_TOKEN_ID)

    metrics_history = { "avg_train_loss": [], "val_loss": [], "val_accuracy": [], "learning_rate": [] }
    last_val_metrics = {}

    # W&B Watch (Not directly available for MLX, log gradients manually if needed)
    if wandb is not None and wandb_run is not None:
        logger.info("📊 W&B logging enabled (manual metric logging).")
        # wandb.watch(model) # Not directly supported in MLX

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # --- Training Epoch ---
        avg_train_loss = train_epoch_mlx( # Model state updated inside
            model=model, optimizer=optimizer, images_full=train_images,
            labels_full=train_labels, batch_size=batch_size, epoch_num=epoch,
            total_epochs=epochs, wandb_run=wandb_run, phase=phase,
            pad_token_id=pad_token_id
        )
        metrics_history["avg_train_loss"].append(avg_train_loss)
        epoch_time = time.time() - epoch_start_time

        # --- Validation Step ---
        val_metrics = evaluate_model_mlx(
            model=model, images_full=val_images, labels_full=val_labels,
            batch_size=batch_size * 2, phase=phase, pad_token_id=pad_token_id
        )
        metrics_history["val_loss"].append(val_metrics.get('val_loss', float('nan')))
        metrics_history["val_accuracy"].append(val_metrics.get('val_accuracy', float('nan')))
        last_val_metrics = val_metrics

        # --- Log epoch summary ---
        current_lr = optimizer.learning_rate.item()
        metrics_history["learning_rate"].append(current_lr) # Store LR per epoch
        log_str = (f"✅ Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | ")
        for k, v in val_metrics.items():
            # Format accuracy with %
            format_str = "{:.2f}%" if "accuracy" in k else "{:.4f}"
            log_str += f"{k}: {format_str.format(v)} | "
        log_str += f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s"
        logger.info(log_str)

        # --- LR Scheduler Step (Manual for MLX) ---
        # Example: Placeholder for manual scheduler update
        # if lr_scheduler_fn is not None:
        #     new_lr = lr_scheduler_fn(epoch) # Get new LR based on epoch
        #     optimizer.learning_rate = mx.array(new_lr) # Update optimizer's LR

        # --- Log Epoch Metrics to W&B ---
        if wandb is not None and wandb_run is not None:
             try:
                 # Log the latest value for each metric at the current epoch step
                 wandb_log_step = {k: v[-1] for k, v in metrics_history.items() if v} # Get last item if list not empty
                 wandb_log_step["epoch"] = epoch + 1 # Ensure epoch is scalar
                 wandb_log_step["epoch_time_sec"] = epoch_time # Add epoch time
                 wandb_run.log(wandb_log_step)
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
        logger.error(f"❌ Failed to save final MLX weights: {e}", exc_info=True)

    # Return the full history and last validation metrics
    return metrics_history, last_val_metrics

# --- Test Block ---
if __name__ == "__main__":
     logger.info("🧪 Running trainer_mlx.py basic checks...")

     # --- Basic check for Phase 1/2 loss/acc ---
     logger.info("
--- Testing P1/P2 loss/acc calc ---")
     try:
         dummy_logits_p1 = mx.random.normal(shape=(4, 10)) # B, C
         dummy_labels_p1 = mx.array([1, 5, 9, 3])
         dummy_model_p1 = lambda img: dummy_logits_p1
         loss_p1, acc_p1 = calculate_loss_acc_mlx(dummy_model_p1, None, dummy_labels_p1, phase=1)
         mx.eval(loss_p1, acc_p1)
         logger.info(f"P1 Dummy Loss: {loss_p1.item():.4f}, Acc: {acc_p1.item()*100:.2f}%")

         dummy_logits_p2 = mx.random.normal(shape=(2, 4, 10)) # B, O, C
         dummy_labels_p2 = mx.array([[1,2,3,4], [5,6,7,8]])
         dummy_model_p2 = lambda img: dummy_logits_p2
         loss_p2, acc_p2 = calculate_loss_acc_mlx(dummy_model_p2, None, dummy_labels_p2, phase=2)
         mx.eval(loss_p2, acc_p2)
         logger.info(f"P2 Dummy Loss: {loss_p2.item():.4f}, Acc: {acc_p2.item()*100:.2f}%")
         logger.info("✅ P1/P2 calc OK.")
     except Exception as e: logger.error(f"❌ P1/P2 Calc Error: {e}", exc_info=True)


     # --- Add a simple check for phase 3 in calculate_loss_acc_mlx ---
     logger.info("
--- Testing P3 loss/acc calc ---")
     try:
          _B, _S, _V = 2, 10, 13
          # Dummy model output (logits for sequence shifted right)
          dummy_logits_p3 = mx.random.normal(shape=(_B, _S - 1, _V)) # B, Seq-1, V
          # Dummy target labels (full sequence with START/END/PAD)
          dummy_labels_p3 = mx.random.randint(1, _V-2, (_B, _S)).astype(mx.uint32) # B, Seq (avoid 0=PAD for now)
          dummy_labels_p3 = dummy_labels_p3.at[:, -2:].set(PAD_TOKEN_ID) # Add some padding at the end
          dummy_labels_p3 = dummy_labels_p3.at[:, 0].set(1) # Assume 1 is START token

          # Dummy model just returns pre-defined logits
          # Needs to accept image and target sequence args for phase 3
          dummy_model_p3 = lambda img, tgt_seq: dummy_logits_p3

          loss_p3, acc_p3 = calculate_loss_acc_mlx(
              dummy_model_p3,
              mx.zeros((_B, 32, 32, 1)), # Dummy image input
              dummy_labels_p3,
              phase=3,
              pad_token_id=PAD_TOKEN_ID
          )
          mx.eval(loss_p3, acc_p3)
          logger.info(f"P3 Dummy Loss: {loss_p3.item():.4f}, Acc: {acc_p3.item()*100:.2f}%")
          logger.info("✅ P3 calc OK.")
     except Exception as e:
         logger.error(f"❌ P3 Calc Error: {e}", exc_info=True)

     logger.info("
✅ trainer_mlx.py basic checks finished.")