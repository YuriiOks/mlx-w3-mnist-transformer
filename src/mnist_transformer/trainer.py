# MNIST Digit Classifier (Transformer)
# File: src/mnist_transformer/trainer.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Implements the training loop logic for the ViT model.
# Created: 2025-04-28
# Updated: 2025-04-28

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Any

# --- Add project root to sys.path for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent # Go up two levels
if str(project_root) not in sys.path:
    print(f"🚂 [trainer.py] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

from utils import logger

# W&B Import Handling
try:
    import wandb
except ImportError:
    logger.warning("⚠️ wandb not installed. W&B logging disabled.")
    wandb = None

# --- Epoch Training Function ---

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module, # Loss function (e.g., CrossEntropyLoss)
    optimizer: optim.Optimizer,
    device: torch.device, # Device to train on
    epoch_num: int,
    total_epochs: int,
    wandb_run: Optional[Any] = None, # Optional W&B run object
    log_frequency: int = 100, # How often to log batch loss
    gradient_clipping: Optional[float] = 1.0 # Max grad norm, None to disable
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader for the training data.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to move data/model to.
        epoch_num: Current epoch number (0-indexed).
        total_epochs: Total number of epochs planned.
        wandb_run: Optional W&B run object for logging.
        log_frequency (int): Log batch loss every N batches.
        gradient_clipping (Optional[float]): Max norm for gradient clipping.

    Returns:
        float: The average loss for the epoch.
    """
    model.train() # Set model to training mode
    total_loss = 0.0
    num_batches = len(dataloader)
    if num_batches == 0:
        logger.warning("⚠️ Empty dataloader, skipping epoch.")
        return 0.0

    data_iterator = tqdm(
        dataloader,
        desc=f"Epoch {epoch_num+1}/{total_epochs}",
        leave=False,
        unit="batch"
    )

    for batch_idx, (images, labels) in enumerate(data_iterator):
        # Move data to the specified device
        images = images.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images) # Get logits

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient Clipping (optional but recommended)
        if gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        # Optimizer step
        optimizer.step()

        # Accumulate loss
        batch_loss = loss.item()
        total_loss += batch_loss

        # Update progress bar
        data_iterator.set_postfix(loss=f"{batch_loss:.4f}")

        # Log batch loss to W&B (if enabled and interval met)
        if wandb is not None and wandb_run is not None and batch_idx % log_frequency == 0:
            global_step = epoch_num * num_batches + batch_idx
            try:
                wandb_run.log({
                    "batch_loss": batch_loss,
                    "learning_rate": optimizer.param_groups[0]['lr'], # Log current LR
                    "epoch": (epoch_num + (batch_idx / num_batches)), # Continuous epoch
                    "global_step": global_step
                })
            except Exception as e:
                logger.warning(f"⚠️ Failed to log batch metrics to W&B: {e}")

    # Calculate average loss for the epoch
    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss

# --- Main Training Orchestrator Function ---

def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    model_save_dir: str | Path,
    run_name: str = "mnist_vit_run", # For saving model folder
    wandb_run: Optional[Any] = None, # Pass W&B run object
    lr_scheduler: Optional[Any] = None, # Optional LR scheduler
    val_dataloader: Optional[DataLoader] = None, # For validation
    evaluation_fn: Optional[Any] = None, # For validation metrics
) -> List[float]:
    """
    Orchestrates the overall model training process.

    Args:
        model: The PyTorch model instance.
        train_dataloader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Compute device.
        epochs: Number of epochs to train.
        model_save_dir (str | Path): Base directory to save model checkpoints/final model.
        run_name (str): Specific name for this run (used for save subdirectory).
        wandb_run: Optional W&B run object.
        lr_scheduler: Optional learning rate scheduler.

    Returns:
        List[float]: List of average training losses per epoch.
    """
    logger.info(f"🚀 Starting Model Training: Run='{run_name}'")
    logger.info(f"   Epochs: {epochs}, Device: {device.type.upper()}")
    model.to(device) # Move model to the specified device

    epoch_losses = []

    # W&B Watch (optional - can log gradients, but adds overhead)
    if wandb is not None and wandb_run is not None:
        try:
            wandb.watch(model, log="gradients", log_freq=500) # Example: Log gradients every 500 steps
            logger.info("📊 W&B watching model parameters and gradients.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initiate wandb.watch: {e}")

    for epoch in range(epochs):
        avg_train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_num=epoch,
            total_epochs=epochs,
            wandb_run=wandb_run,
            # Add other args like log_frequency, gradient_clipping from config if needed
        )
        logger.info(f"✅ Epoch {epoch+1}/{epochs} | Avg Train Loss: {avg_train_loss:.4f}")
        epoch_losses.append(avg_train_loss)

        # --- Optional: Validation Step ---
        if val_dataloader and evaluation_fn:
            val_metrics = evaluation_fn(model, val_dataloader, device)
            logger.info(f"  Validation Metrics: {val_metrics}")
            if wandb is not None and wandb_run is not None:
                wandb_run.log({"epoch": epoch + 1, **val_metrics}) # Log validation metrics

        # Learning Rate Scheduler Step (if provided)
        if lr_scheduler is not None:
            # Handle different scheduler types (e.g., ReduceLROnPlateau needs metrics)
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
               lr_scheduler.step(val_metrics['val_loss']) # Needs validation loss
            else:
                lr_scheduler.step() # For schedulers like CosineAnnealingLR, StepLR etc.


        # --- Log Epoch Metrics to W&B ---
        if wandb is not None and wandb_run is not None:
             try:
                 wandb_log = {"epoch": epoch + 1, "avg_train_loss": avg_train_loss}
                 # Could add validation metrics here if validation is implemented
                 wandb_run.log(wandb_log)
             except Exception as e:
                 logger.error(f"❌ W&B epoch log failed: {e}")


    logger.info("🏁 Training finished.")

    # --- Save Final Model ---
    final_save_dir = Path(model_save_dir) / run_name
    try:
        final_save_dir.mkdir(parents=True, exist_ok=True)
        model_path = final_save_dir / "model_final.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"💾 Final model state saved to: {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save final model: {e}", exc_info=True)

    return epoch_losses


# --- Test Block (Optional - More involved to set up full training here) ---
if __name__ == "__main__":
    logger.info("🧪 Running trainer.py script directly for testing (basic checks)...")
    # Note: Running this directly requires setting up dummy model, data, optimizer etc.
    # It's generally better to test this via the main training script.

    # Example basic check: Can the functions be called?
    logger.info("Checking function signatures...")
    try:
        # Example dummy components
        _device = torch.device("cpu")
        _model = nn.Linear(10, 2) # Dummy model
        _data = [(torch.randn(10), torch.randint(0, 2, (1,)).squeeze()) for _ in range(5)]
        _loader = DataLoader(_data, batch_size=2)
        _crit = nn.CrossEntropyLoss()
        _optim = optim.Adam(_model.parameters(), lr=0.01)

        logger.info("Testing train_epoch...")
        avg_loss = train_epoch(_model, _loader, _crit, _optim, _device, 0, 1)
        logger.info(f"Dummy train_epoch avg loss: {avg_loss:.4f}")

        logger.info("Testing train_model...")
        losses = train_model(_model, _loader, _crit, _optim, _device, 2, "./temp_test_model", "test_run")
        logger.info(f"Dummy train_model epoch losses: {losses}")
        # Clean up dummy save directory
        if os.path.exists("./temp_test_model"):
             import shutil
             shutil.rmtree("./temp_test_model")

        logger.info("✅ Basic trainer function calls successful.")

    except Exception as e:
        logger.error(f"❌ Error during basic trainer tests: {e}", exc_info=True)