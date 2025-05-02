# MNIST Digit Classifier (Transformer) - PyTorch Version
# File: src/mnist_transformer/trainer.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Implements the training & evaluation logic for ViT models.
# Created: 2025-04-28
# Updated: 2025-04-30

import pickle
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Tuple, Callable
import time
import numpy as np

# --- Add project root to sys.path for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent
if str(project_root) not in sys.path:
    print(f"🚂 [trainer.py] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

from utils import logger
try:
    from utils.tokenizer_utils import PAD_TOKEN_ID
except ImportError:
    logger.error("❌ Failed to import PAD_TOKEN_ID from tokenizer_utils.")
    PAD_TOKEN_ID = 0

try:
    import wandb
except ImportError:
    logger.warning("⚠️ wandb not installed. W&B logging disabled.")
    wandb = None

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
    phase: int = 1,
    pad_token_id: int = PAD_TOKEN_ID
) -> float:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to use for computation.
        epoch_num (int): Current epoch number (0-based).
        total_epochs (int): Total number of epochs.
        wandb_run (Optional[Any]): Weights & Biases run object.
        log_frequency (int): Frequency of logging to wandb.
        gradient_clipping (Optional[float]): Max norm for gradients.
        phase (int): Training phase (1, 2, or 3).
        pad_token_id (int): Padding token id for phase 3.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    if num_batches == 0:
        logger.warning("⚠️ Empty dataloader, skipping training epoch.")
        return 0.0

    data_iterator = tqdm(
        dataloader,
        desc=f"Epoch {epoch_num+1}/{total_epochs} Train",
        leave=False,
        unit="batch"
    )

    for batch_idx, batch_data in enumerate(data_iterator):
        images = batch_data[0].to(device)
        labels = batch_data[1].to(device)
        optimizer.zero_grad()

        if phase == 1 or phase == 2:
            outputs = model(images)
        elif phase == 3:
            decoder_input = labels[:, :-1]
            outputs = model(images, decoder_input)
        else:
            logger.error(
                f"❌ Forward pass for Phase {phase} not implemented!")
            continue

        if phase == 1:
            loss = criterion(outputs, labels)
        elif phase == 2:
            num_classes = outputs.shape[-1]
            loss = criterion(
                outputs.reshape(-1, num_classes),
                labels.reshape(-1)
            )
        elif phase == 3:
            decoder_target = labels[:, 1:]
            vocab_size = outputs.shape[-1]
            loss = criterion(
                outputs.reshape(-1, vocab_size),
                decoder_target.reshape(-1)
            )
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        loss.backward()
        if gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clipping
            )
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        data_iterator.set_postfix(loss=f"{batch_loss:.4f}")

        if (wandb is not None and wandb_run is not None and
                batch_idx % log_frequency == 0):
            global_step = epoch_num * num_batches + batch_idx
            log_dict = {
                "batch_loss": batch_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_fractional": (epoch_num + (batch_idx / num_batches)),
                "global_step": global_step
            }
            try:
                wandb_run.log(log_dict)
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to log batch metrics to W&B: {e}"
                )

    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phase: int = 1,
    pad_token_id: int = PAD_TOKEN_ID
) -> Dict[str, float]:
    """
    Evaluate the model on a dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for evaluation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use for computation.
        phase (int): Evaluation phase (1, 2, or 3).
        pad_token_id (int): Padding token id for phase 3.

    Returns:
        Dict[str, float]: Dictionary with 'val_loss' and 'val_accuracy'.
    """
    model.eval()
    total_loss = 0.0
    total_correct_predictions = 0
    total_samples_evaluated = 0
    num_batches = len(dataloader)

    if num_batches == 0:
        logger.warning("⚠️ Evaluation dataloader is empty.")
        return {"val_loss": 0.0, "val_accuracy": 0.0}

    logger.info(f"🧪 Starting evaluation (Phase {phase})...")
    with torch.no_grad():
        data_iterator = tqdm(
            dataloader, desc="Evaluating", leave=False, unit="batch"
        )
        for batch_data in data_iterator:
            images = batch_data[0].to(device)
            labels = batch_data[1].to(device)

            if phase == 1:
                outputs = model(images)
                loss = criterion(outputs, labels)
                predicted = torch.argmax(outputs.data, 1)
                mask = torch.ones_like(labels, dtype=torch.bool)
            elif phase == 2:
                outputs = model(images)
                num_classes = outputs.shape[-1]
                loss = criterion(
                    outputs.view(-1, num_classes),
                    labels.view(-1)
                )
                predicted = torch.argmax(outputs.data, 2)
                mask = torch.ones_like(labels, dtype=torch.bool)
            elif phase == 3:
                decoder_input = labels[:, :-1]
                decoder_target = labels[:, 1:]
                outputs = model(images, decoder_input)
                vocab_size = outputs.shape[-1]
                loss = criterion(
                    outputs.reshape(-1, vocab_size),
                    decoder_target.reshape(-1)
                )
                predicted = torch.argmax(outputs.data, -1)
                mask = (decoder_target != pad_token_id)
            else:
                logger.error(
                    f"❌ Evaluation for Phase {phase} not implemented!"
                )
                loss = torch.tensor(0.0, device=device)
                mask = torch.zeros_like(labels, dtype=torch.bool)

            if phase == 1 or phase == 2:
                labels_masked = labels[mask]
                predicted_masked = predicted[mask]
                batch_correct = (predicted_masked == labels_masked).sum().item()
                batch_total = mask.sum().item()
                total_loss += loss.item() * batch_total
            elif phase == 3:
                target_masked = decoder_target[mask]
                predicted_masked = predicted[mask]
                batch_correct = (predicted_masked == target_masked).sum().item()
                batch_total = mask.sum().item()
                total_loss += loss.item() * batch_total
            else:
                batch_correct = 0
                batch_total = labels.numel()
                total_loss += loss.item() * labels.size(0)

            total_correct_predictions += batch_correct
            total_samples_evaluated += batch_total
            current_acc = (batch_correct / batch_total) * 100.0 \
                if batch_total > 0 else 0.0
            data_iterator.set_postfix(
                loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%"
            )

    avg_loss = (total_loss / total_samples_evaluated
                if total_samples_evaluated > 0 else 0.0)
    avg_accuracy = (total_correct_predictions / total_samples_evaluated) * 100.0 \
        if total_samples_evaluated > 0 else 0.0

    logger.info(
        f"🧪 Evaluation finished. Avg Loss: {avg_loss:.4f}, "
        f"Accuracy: {avg_accuracy:.2f}% "
        f"({int(total_correct_predictions)}/{total_samples_evaluated})"
    )
    return {"val_loss": avg_loss, "val_accuracy": avg_accuracy}

def save_checkpoint_pt(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    metrics_history: Dict,
    save_dir: Path,
    is_best: bool = False,
    model_config: dict = None,
    dataset_config: dict = None,
    phase: int = None
):
    """
    Save PyTorch model checkpoint.

    Args:
        model (nn.Module): Model to save.
        optimizer (optim.Optimizer): Optimizer.
        scheduler (Optional[_LRScheduler]): LR scheduler.
        epoch (int): Current epoch.
        metrics_history (Dict): Training/validation metrics.
        save_dir (Path): Directory to save checkpoint.
        is_best (bool): If True, also save as best model.
        model_config (dict): Model config for reproducibility.
        dataset_config (dict): Dataset config for reproducibility.
        phase (int): Training phase.

    Saves:
        - model_final.pth: Model weights.
        - training_state_pt.pkl: Optimizer, scheduler, metrics, configs.
        - best_model_pt.pth: Best model weights (if is_best).
    """
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "model_final.pth"
        state_path = save_dir / "training_state_pt.pkl"
        torch.save(model.state_dict(), model_path)
        state = {
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics_history': metrics_history,
            'model_config': model_config,
            'dataset_config': dataset_config,
            'phase': phase
        }
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(
            f"\U0001F4BE PyTorch Checkpoint saved to {save_dir} (Epoch {epoch+1})"
        )
        if is_best:
            best_model_path = save_dir.parent / "best_model_pt.pth"
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"\U0001F3C6 Best PT model weights saved to {best_model_path}"
            )
    except Exception as e:
        logger.error(
            f"❌ Failed to save PyTorch checkpoint: {e}", exc_info=True
        )

def load_checkpoint_pt(
    save_dir: Path,
    device: torch.device
) -> tuple:
    """
    Load PyTorch checkpoint and configs for reproducibility.

    Args:
        save_dir (Path): Directory containing checkpoint files.
        device (torch.device): Device to map tensors.

    Returns:
        tuple: (
            model_state_dict,
            optimizer_state_dict,
            scheduler_state_dict,
            start_epoch,
            metrics_history,
            model_config,
            dataset_config,
            phase
        )
    """
    model_path = save_dir / "model_final.pth"
    state_path = save_dir / "training_state_pt.pkl"
    start_epoch = 0
    metrics_history = {
        "avg_train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": []
    }
    model_state_dict = None
    optimizer_state_dict = None
    scheduler_state_dict = None
    model_config = None
    dataset_config = None
    phase = None
    if model_path.exists() and state_path.exists():
        logger.info(f"♻️ Resuming PyTorch training from checkpoint in {save_dir}")
        try:
            model_state_dict = torch.load(model_path, map_location=device)
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            optimizer_state_dict = state.get('optimizer_state_dict')
            scheduler_state_dict = state.get('scheduler_state_dict')
            start_epoch = state.get('epoch', 0)
            loaded_history = state.get('metrics_history', {})
            for key in metrics_history.keys():
                metrics_history[key] = loaded_history.get(key, [])
            model_config = state.get('model_config')
            dataset_config = state.get('dataset_config')
            phase = state.get('phase')
            logger.info(
                f"✅ PT Checkpoint loaded. Resuming from epoch {start_epoch}."
            )
        except Exception as e:
            logger.error(
                f"❌ Failed loading PT checkpoint: {e}. Starting fresh.",
                exc_info=True
            )
            start_epoch = 0
            metrics_history = {k: [] for k in metrics_history}
    else:
        logger.info("No PyTorch checkpoint found. Starting training from scratch.")
    return (
        model_state_dict,
        optimizer_state_dict,
        scheduler_state_dict,
        start_epoch,
        metrics_history,
        model_config,
        dataset_config,
        phase
    )

def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    model_save_dir: str | Path,
    config: dict,
    phase: int = 1,
    val_dataloader: Optional[DataLoader] = None,
    run_name: str = "mnist_vit_run",
    wandb_run: Optional[Any] = None,
    lr_scheduler: Optional[Any] = None,
    resume_from_checkpoint: bool = False,
    save_every: int = 5,
    start_epoch: int = 0,
    initial_metrics_history: Optional[Dict[str, list]] = None
) -> Dict[str, List[float]]:
    """
    Train the model for multiple epochs, with optional validation,
    checkpointing, and learning rate scheduling.

    Args:
        model (nn.Module): Model to train.
        train_dataloader (DataLoader): Training data loader.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to use for computation.
        epochs (int): Number of epochs to train.
        model_save_dir (str | Path): Directory to save checkpoints.
        config (dict): Training/configuration dictionary.
        phase (int): Training phase (1, 2, or 3).
        val_dataloader (Optional[DataLoader]): Validation data loader.
        run_name (str): Name for this training run.
        wandb_run (Optional[Any]): Weights & Biases run object.
        lr_scheduler (Optional[Any]): Learning rate scheduler.
        resume_from_checkpoint (bool): Resume from checkpoint if True.
        save_every (int): Save checkpoint every N epochs.
        start_epoch (int): Epoch to start training from.
        initial_metrics_history (Optional[Dict[str, list]]): Initial metrics history.

    Returns:
        Dict[str, List[float]]: Training/validation metrics history.
    """
    logger.info(
        f"🚀 Starting PyTorch Model Training: Run='{run_name}', Phase={phase}"
    )
    run_save_path = Path(model_save_dir) / run_name
    if initial_metrics_history is not None:
        metrics_history = initial_metrics_history
    else:
        metrics_history = {
            "avg_train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": []
        }
    model_config = config.get('model', {})
    dataset_config = config.get('dataset', {})
    if resume_from_checkpoint:
        (
            model_state_dict,
            optimizer_state_dict,
            scheduler_state_dict,
            start_epoch,
            metrics_history,
            loaded_model_cfg,
            loaded_dataset_cfg,
            loaded_phase
        ) = load_checkpoint_pt(run_save_path, device)
        # Re-instantiate model/optimizer/scheduler if needed using loaded configs
        # (This logic should be handled in the main script for full reproducibility)
        if model_state_dict is not None:
            model.load_state_dict(model_state_dict)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
            for state_val in optimizer.state.values():
                for k, v in state_val.items():
                    if isinstance(v, torch.Tensor):
                        state_val[k] = v.to(device)
        if lr_scheduler and scheduler_state_dict is not None:
            lr_scheduler.load_state_dict(scheduler_state_dict)
        if loaded_model_cfg is not None:
            model_config = loaded_model_cfg
        if loaded_dataset_cfg is not None:
            dataset_config = loaded_dataset_cfg
        if loaded_phase is not None:
            phase = loaded_phase
    model.to(device)
    logger.info(f"   Target Epochs: {epochs}")
    logger.info(
        f"   Starting from Epoch: {start_epoch}, "
        f"Training until Epoch: {epochs}"
    )
    pad_token_id = config.get('tokenizer', {}).get('pad_token_id', PAD_TOKEN_ID)
    if phase == 3:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    best_val_accuracy = 0.0
    if metrics_history.get("val_accuracy"):
        try:
            best_val_accuracy = max(
                filter(lambda x: not np.isnan(x),
                       metrics_history["val_accuracy"])
            )
        except ValueError:
            best_val_accuracy = 0.0
    if start_epoch >= epochs:
        logger.warning("Training already completed based on checkpoint.")
        return metrics_history
    if wandb_run is not None and metrics_history is not None and start_epoch > 0:
        # Log all previous metrics to W&B before resuming
        for epoch_idx in range(start_epoch):
            log_dict = {}
            if len(metrics_history.get("avg_train_loss", [])) > epoch_idx:
                log_dict["avg_train_loss"] = metrics_history["avg_train_loss"][epoch_idx]
            if len(metrics_history.get("val_loss", [])) > epoch_idx:
                log_dict["val_loss"] = metrics_history["val_loss"][epoch_idx]
            if len(metrics_history.get("val_accuracy", [])) > epoch_idx:
                log_dict["val_accuracy"] = metrics_history["val_accuracy"][epoch_idx]
            if len(metrics_history.get("learning_rate", [])) > epoch_idx:
                log_dict["learning_rate"] = metrics_history["learning_rate"][epoch_idx]
            log_dict["epoch"] = epoch_idx + 1
            try:
                wandb_run.log(log_dict, step=epoch_idx + 1)
            except Exception as e:
                logger.warning(f"⚠️ Failed to log previous metrics to W&B: {e}")
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        avg_train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_num=epoch,
            total_epochs=epochs,
            wandb_run=wandb_run,
            log_frequency=100,
            gradient_clipping=config.get('training', {}).get(
                'gradient_clipping'),
            phase=phase,
            pad_token_id=pad_token_id
        )
        metrics_history["avg_train_loss"].append(avg_train_loss)
        val_metrics = {}
        if val_dataloader:
            val_metrics = evaluate_model(
                model=model,
                dataloader=val_dataloader,
                criterion=criterion,
                device=device,
                phase=phase,
                pad_token_id=pad_token_id
            )
            metrics_history["val_loss"].append(
                val_metrics.get('val_loss', float('nan')))
            metrics_history["val_accuracy"].append(
                val_metrics.get('val_accuracy', float('nan')))
        else:
            metrics_history["val_loss"].append(float('nan'))
            metrics_history["val_accuracy"].append(float('nan'))
        current_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler:
            pass
        metrics_history["learning_rate"].append(current_lr)
        current_val_acc = val_metrics.get('val_accuracy', -1)
        is_best = current_val_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = current_val_acc
            logger.info(
                f"🏆 New best validation accuracy: "
                f"{best_val_accuracy:.2f}% at Epoch {epoch+1}"
            )
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1 or is_best:
            save_checkpoint_pt(
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                epoch=epoch,
                metrics_history=metrics_history,
                save_dir=run_save_path,
                is_best=is_best,
                model_config=model_config,
                dataset_config=dataset_config,
                phase=phase
            )
    logger.info("🏁 PyTorch Training finished.")
    return metrics_history

# (The __main__ test block is omitted for brevity and clarity.)
