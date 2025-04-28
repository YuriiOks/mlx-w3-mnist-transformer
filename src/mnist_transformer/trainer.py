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
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_num: int,
    total_epochs: int,
    wandb_run: Optional[Any] = None,
    log_frequency: int = 100,
    gradient_clipping: Optional[float] = 1.0,
    phase: int = 1
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
        gradient_clipping (Optional[float]): Max norm for gradient
            clipping. None to disable.
        phase (int): Training phase (1 or 2).

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
        labels = labels.to(device) # Shape: (B) for P1, (B, 4) for P2

        optimizer.zero_grad()
        outputs = model(images) # Shape: (B, 10) for P1, (B, 4, 10) for P2

        # Calculate loss
        # --- 👇 Adapted Loss Calculation for Phase ---
        if phase == 1:
            loss = criterion(outputs, labels)
        elif phase == 2:
            # Reshape for CrossEntropyLoss which expects (N, C) and (N)
            # Outputs: (B, 4, 10) -> (B*4, 10)
            # Labels: (B, 4) -> (B*4)
            batch_size, num_outputs, num_classes = outputs.shape
            loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
            # Check if labels are valid (e.g. not -1 if using dummy data)
            if (labels == -1).any():
                 logger.warning(
                     f"Detected dummy label (-1) in batch {batch_idx}, "
                     f"loss might be inaccurate."
                 )
        else: # Handle Phase 3 later
            logger.error(
                f"❌ Loss calculation for Phase {phase} not implemented!"
            )
            # Placeholder loss
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        # --- End Adapt Loss Calculation ---

        # Backward pass
        loss.backward()

        # Gradient Clipping (optional but recommended)
        if gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clipping
            )

        # Optimizer step
        optimizer.step()

        # Accumulate loss
        batch_loss = loss.item()
        total_loss += batch_loss

        # Update progress bar
        data_iterator.set_postfix(loss=f"{batch_loss:.4f}")

        # Log batch loss to W&B (if enabled and interval met)
        if (wandb is not None and
                wandb_run is not None and
                batch_idx % log_frequency == 0):
            global_step = epoch_num * num_batches + batch_idx
            try:
                current_lr = optimizer.param_groups[0]['lr']
                # Continuous epoch (fractional)
                cont_epoch = epoch_num + (batch_idx / num_batches)
                wandb_run.log({
                    "batch_loss": batch_loss,
                    "learning_rate": current_lr,
                    "epoch": cont_epoch,
                    "global_step": global_step
                })
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to log batch metrics to W&B: {e}"
                )

    # Calculate average loss for the epoch
    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss

# --- Evaluation Function ---

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phase: int = 1
) -> Dict[str, float]:
    """
    Evaluates the model on a given dataset.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: DataLoader for the validation or test data.
        criterion: The loss function (e.g., CrossEntropyLoss).
        device: The device to move data/model to.
        phase (int): Evaluation phase (1 or 2).

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
                          (e.g., 'val_loss', 'val_accuracy').
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    num_batches = len(dataloader)

    if num_batches == 0:
        logger.warning("⚠️ Evaluation dataloader is empty.")
        return {"val_loss": 0.0, "val_accuracy": 0.0}

    logger.info("🧪 Starting evaluation...")
    with torch.no_grad(): # Disable gradient calculations for efficiency
        data_iterator = tqdm(
            dataloader, desc="Evaluating", leave=False, unit="batch"
        )
        for images, labels in data_iterator:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images) # Get logits

            # Calculate loss
            # --- 👇 Adapted Loss & Accuracy Calculation for Phase ---
            if phase == 1:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels).sum().item()
            elif phase == 2:
                batch_size, num_outputs, num_classes = outputs.shape
                # Reshape for loss calculation
                loss = criterion(
                    outputs.view(-1, num_classes), labels.view(-1)
                )
                # Reshape for accuracy calculation
                # Get predictions per output head -> (B, 4)
                predicted = torch.max(outputs.data, 2)[1]
                # Total digits in batch = B * 4
                batch_total = labels.numel()
                # Compare element-wise
                batch_correct = (predicted == labels).sum().item()
            else:
                logger.error(
                    f"❌ Evaluation for Phase {phase} not implemented!"
                )
                loss = torch.tensor(0.0, device=device)
                # Assume phase 1 shape for safety
                batch_total = labels.size(0)
                batch_correct = 0
            # --- End Adapt Loss & Accuracy ---

            total_loss += loss.item()

            # Calculate accuracy
            # Get the index of the max logit
            total_loss += loss.item()
            total_samples += batch_total
            correct_predictions += batch_correct

            data_iterator.set_postfix(loss=f"{loss.item():.4f}")


    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = (
        (correct_predictions / total_samples) * 100.0
        if total_samples > 0 else 0.0
    )
    logger.info(
        f"🧪 Evaluation finished. Avg Loss: {avg_loss:.4f}, "
        f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})"
    )

    # Use consistent naming for W&B logging
    return {"val_loss": avg_loss, "val_accuracy": accuracy}

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
    phase: int = 1 # Phase 1 or 2
) -> tuple[List[float], Dict[str, float]]:
    """
    Orchestrates the overall model training process.

    Args:
        model: The PyTorch model instance.
        train_dataloader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Compute device.
        epochs: Number of epochs to train.
        model_save_dir (str | Path): Base directory to save model
            checkpoints/final model.
        run_name (str): Specific name for this run (used for save
            subdirectory).
        wandb_run: Optional W&B run object.
        lr_scheduler: Optional learning rate scheduler.
        val_dataloader: Optional DataLoader for validation data.
        phase (int): Training phase (1 or 2).

    Returns:
        tuple[List[float], Dict[str, float]]: List of average training
            losses per epoch and final validation metrics.
    """
    logger.info(f"🚀 Starting Model Training: Run='{run_name}'")
    logger.info(f"   Epochs: {epochs}, Device: {device.type.upper()}")
    model.to(device) # Move model to the specified device

    epoch_losses = []
    final_val_metrics = {} # Store last validation metrics

    # W&B Watch (optional - can log gradients, but adds overhead)
    if wandb is not None and wandb_run is not None:
        try:
            # Example: Log gradients every 500 steps
            wandb.watch(model, log="gradients", log_freq=500)
            logger.info("📊 W&B watching model parameters and gradients.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initiate wandb.watch: {e}")

    for epoch in range(epochs):
        # --- Training Epoch ---
        avg_train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_num=epoch,
            total_epochs=epochs,
            wandb_run=wandb_run,
            phase=phase
        )
        logger.info(
            f"✅ Epoch {epoch+1}/{epochs} | "
            f"Avg Train Loss: {avg_train_loss:.4f}"
        )
        epoch_losses.append(avg_train_loss)

        # --- Validation Step ---
        val_metrics = {} # Dictionary to store validation results
        if val_dataloader:
            val_metrics = evaluate_model( # Call the evaluation function
                model=model,
                dataloader=val_dataloader,
                criterion=criterion, # Use the same loss function
                device=device,
                phase=phase
            )
            # Log validation metrics to console
            log_str = f"  🧪 Validation | "
            for key, value in val_metrics.items():
                 log_str += f"{key}: {value:.4f} | "
            logger.info(log_str.strip())
            final_val_metrics = val_metrics # Update final metrics
        # --- End Validation Step ---

        # Learning Rate Scheduler Step
        if lr_scheduler is not None:
            if isinstance(
                lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                # Need validation loss if using ReduceLROnPlateau
                if "val_loss" in val_metrics:
                     lr_scheduler.step(val_metrics['val_loss'])
                else:
                     logger.warning(
                         "⚠️ ReduceLROnPlateau scheduler needs 'val_loss' "
                         "but validation was not run."
                     )
            else:
                lr_scheduler.step()

        # --- Log Epoch Metrics to W&B ---
        if wandb is not None and wandb_run is not None:
            try:
                # Combine train and validation metrics for logging
                wandb_log = {
                    "epoch": epoch + 1,
                    "avg_train_loss": avg_train_loss,
                    # Add all validation metrics (e.g., val_loss, ...)
                    **val_metrics
                    }
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

    return epoch_losses, final_val_metrics


# --- Test Block (Optional - More involved to set up full training here) ---
if __name__ == "__main__":
    logger.info(
        "🧪 Running trainer.py script directly for testing (basic checks)..."
    )
    # Note: Running this directly requires setting up dummy model, data,
    # optimizer etc. It's generally better to test this via the main
    # training script.

    # Example basic check: Can the functions be called?
    logger.info("Checking function signatures...")
    try:
        # Example dummy components
        _device = torch.device("cpu")
        # Dummy for Phase 1 (10 classes)
        _model_p1 = nn.Linear(10, 10)
        # Dummy for Phase 2 (4*10 classes)
        _model_p2 = nn.Linear(10, 40)
        _data_p1 = [
            (torch.randn(10), torch.randint(0, 10, (1,)).squeeze())
            for _ in range(5)
        ]
        # Phase 2 labels shape (B=1, 4)
        _data_p2 = [
            (torch.randn(10), torch.randint(0, 10, (4,)))
            for _ in range(5)
        ]
        _loader_p1 = DataLoader(_data_p1, batch_size=2)
        _loader_p2 = DataLoader(_data_p2, batch_size=2)
        _crit = nn.CrossEntropyLoss()
        _optim_p1 = optim.Adam(_model_p1.parameters(), lr=0.01)
        _optim_p2 = optim.Adam(_model_p2.parameters(), lr=0.01)

        logger.info("Testing train_epoch (Phase 1)...")
        avg_loss_p1 = train_epoch(
            _model_p1, _loader_p1, _crit, _optim_p1, _device, 0, 1, phase=1
        )
        logger.info(f"Dummy P1 train_epoch avg loss: {avg_loss_p1:.4f}")

        logger.info("Testing train_epoch (Phase 2)...")
        # Need to adapt the dummy model forward for Phase 2 shape
        class DummyP2Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 40)
            def forward(self, x):
                # Simulate reshape
                return self.fc(x).view(x.shape[0], 4, 10)
        _model_p2_wrap = DummyP2Model()
        _optim_p2 = optim.Adam(_model_p2_wrap.parameters(), lr=0.01)
        avg_loss_p2 = train_epoch(
            _model_p2_wrap, _loader_p2, _crit, _optim_p2, _device, 0, 1,
            phase=2
        )
        logger.info(f"Dummy P2 train_epoch avg loss: {avg_loss_p2:.4f}")


        logger.info("Testing evaluate_model (Phase 2)...")
        val_metrics = evaluate_model(
            _model_p2_wrap, _loader_p2, _crit, _device, phase=2
        )
        logger.info(f"Dummy P2 evaluate_model metrics: {val_metrics}")

        logger.info("✅ Basic trainer function calls successful.")

    except Exception as e:
        logger.error(
            f"❌ Error during basic trainer tests: {e}", exc_info=True
        )