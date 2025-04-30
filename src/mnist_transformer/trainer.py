# MNIST Digit Classifier (Transformer) - PyTorch Version
# File: src/mnist_transformer/trainer.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Implements the training & evaluation logic for ViT models.
# Created: 2025-04-28
# Updated: 2025-04-30

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Tuple, Callable

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
    Trains the model for one epoch. Adapts loss/forward pass for phases.
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
            logger.error(f"❌ Forward pass for Phase {phase} not implemented!")
            continue

        if phase == 1:
            loss = criterion(outputs, labels)
        elif phase == 2:
            num_classes = outputs.shape[-1]
            loss = criterion(outputs.reshape(-1, num_classes),
                             labels.reshape(-1))
        elif phase == 3:
            decoder_target = labels[:, 1:]
            vocab_size = outputs.shape[-1]
            loss = criterion(outputs.reshape(-1, vocab_size),
                             decoder_target.reshape(-1))
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
    Evaluates the model on a given dataset (validation or test).
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
                loss = criterion(outputs.view(-1, num_classes),
                                 labels.view(-1))
                predicted = torch.argmax(outputs.data, 2)
                mask = torch.ones_like(labels, dtype=torch.bool)
            elif phase == 3:
                decoder_input = labels[:, :-1]
                decoder_target = labels[:, 1:]
                outputs = model(images, decoder_input)
                vocab_size = outputs.shape[-1]
                loss = criterion(outputs.reshape(-1, vocab_size),
                                 decoder_target.reshape(-1))
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
) -> Dict[str, List[float]]:
    """
    Orchestrates the overall model training process for PyTorch models.
    """
    logger.info(
        f"🚀 Starting PyTorch Model Training: Run='{run_name}', Phase={phase}"
    )
    logger.info(f"   Epochs: {epochs}, Device: {device.type.upper()}")
    model.to(device)

    pad_token_id = config.get('tokenizer', {}).get('pad_token_id', PAD_TOKEN_ID)
    if phase == 3:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        logger.info(
            f"Using CrossEntropyLoss with ignore_index={pad_token_id} "
            "for Phase 3."
        )
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info(f"Using standard CrossEntropyLoss for Phase {phase}.")
    criterion.to(device)

    # --- 👇 Initialize metrics history dictionary ---
    metrics_history = {
        "avg_train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": []
    }

    if wandb is not None and wandb_run is not None:
        try:
            wandb.watch(model, log="gradients", log_freq=500)
            logger.info("📊 W&B watching model parameters and gradients.")
        except Exception as e:
            logger.warning(f"⚠️ Failed W&B watch: {e}")

    for epoch in range(epochs):
        import time
        epoch_start_time = time.time()

        avg_train_loss = train_epoch(
            model=model, dataloader=train_dataloader, criterion=criterion,
            optimizer=optimizer, device=device, epoch_num=epoch,
            total_epochs=epochs, wandb_run=wandb_run, phase=phase,
            log_frequency=100,
            gradient_clipping=config.get('training', {}).get(
                'gradient_clipping'),
            pad_token_id=pad_token_id
        )
        metrics_history["avg_train_loss"].append(avg_train_loss)
        epoch_time = time.time() - epoch_start_time

        val_metrics = {}
        if val_dataloader:
            val_metrics = evaluate_model(
                model=model, dataloader=val_dataloader, criterion=criterion,
                device=device, phase=phase, pad_token_id=pad_token_id
            )
            metrics_history["val_loss"].append(val_metrics.get("val_loss", float('nan')))
            metrics_history["val_accuracy"].append(val_metrics.get("val_accuracy", float('nan')))
            log_str = (f"✅ Epoch {epoch+1}/{epochs} | Train Loss: "
                       f"{avg_train_loss:.4f} | ")
            for key, value in val_metrics.items():
                log_str += f"{key}: {value:.4f} | "
            log_str += f"Time: {epoch_time:.2f}s"
            logger.info(log_str)
        else:
            metrics_history["val_loss"].append(float('nan'))
            metrics_history["val_accuracy"].append(float('nan'))
            logger.info(
                f"✅ Epoch {epoch+1}/{epochs} | Avg Train Loss: "
                f"{avg_train_loss:.4f} | Time: {epoch_time:.2f}s"
            )

        if lr_scheduler is not None:
            if isinstance(
                lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                if "val_loss" in val_metrics:
                    lr_scheduler.step(val_metrics['val_loss'])
                else:
                    logger.warning("⚠️ ReduceLROnPlateau needs val_loss.")
            else:
                lr_scheduler.step()
            metrics_history["learning_rate"].append(optimizer.param_groups[0]['lr'])
        else:
            metrics_history["learning_rate"].append(optimizer.param_groups[0]['lr'])

        if wandb is not None and wandb_run is not None:
            try:
                wandb_log = {
                    "epoch": epoch + 1,
                    "avg_train_loss": avg_train_loss,
                    **val_metrics
                }
                wandb_log["learning_rate"] = optimizer.param_groups[0]['lr']
                wandb_log["epoch_time_sec"] = epoch_time
                wandb_run.log(wandb_log)
            except Exception as e:
                logger.error(f"❌ W&B epoch log failed: {e}")

    logger.info("🏁 Training finished.")

    final_save_dir = Path(model_save_dir) / run_name
    try:
        final_save_dir.mkdir(parents=True, exist_ok=True)
        model_path = final_save_dir / "model_final.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"💾 Final model state saved to: {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save final model: {e}", exc_info=True)

    # --- 👇 Return the full history ---
    return metrics_history

if __name__ == "__main__":
    logger.info("🧪 Running trainer.py script directly for testing...")
    _device = torch.device("cpu")
    _crit_p12 = nn.CrossEntropyLoss()
    _crit_p3 = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    logger.info("\n--- Testing Phase 1 Logic ---")
    _model_p1 = nn.Linear(10, 10)
    _data_p1 = [
        (
            torch.randn(10,),
            torch.randint(0, 10, (1,), dtype=torch.long).squeeze()
        )
        for _ in range(5)
    ]
    _loader_p1 = DataLoader(_data_p1, batch_size=2)
    _optim_p1 = optim.Adam(_model_p1.parameters(), lr=0.01)
    try:
        train_epoch(
            _model_p1, _loader_p1, _crit_p12, _optim_p1, _device, 0, 1, phase=1
        )
        evaluate_model(
            _model_p1, _loader_p1, _crit_p12, _device, phase=1
        )
        logger.info("✅ Phase 1 trainer calls OK.")
    except Exception as e:
        logger.error(f"❌ Phase 1 Test Error: {e}", exc_info=True)

    logger.info("\n--- Testing Phase 2 Logic ---")
    class DummyP2Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 40)
        def forward(self, x):
            return self.fc(x).view(x.shape[0], 4, 10)
    _model_p2 = DummyP2Model()
    _data_p2 = [
        (torch.randn(1, 10), torch.randint(0, 10, (4,)))
        for _ in range(5)
    ]
    _loader_p2 = DataLoader(_data_p2, batch_size=2)
    _optim_p2 = optim.Adam(_model_p2.parameters(), lr=0.01)
    try:
        train_epoch(
            _model_p2, _loader_p2, _crit_p12, _optim_p2, _device, 0, 1, phase=2
        )
        evaluate_model(
            _model_p2, _loader_p2, _crit_p12, _device, phase=2
        )
        logger.info("✅ Phase 2 trainer calls OK.")
    except Exception as e:
        logger.error(f"❌ Phase 2 Test Error: {e}", exc_info=True)

    logger.info("\n--- Testing Phase 3 Logic ---")
    class DummyP3Model(nn.Module):
        def __init__(self, vocab_size=13, embed_dim=10):
            super().__init__()
            self.embed_dim = embed_dim
            self.dummy_embed = nn.Embedding(vocab_size, embed_dim)
            self.fc = nn.Linear(embed_dim, vocab_size)
        def forward(self, img, tgt_seq):
            embedded_tgt = self.dummy_embed(tgt_seq)
            output = self.fc(embedded_tgt)
            return output

    _seq_len = 10
    _vocab_size = 13
    _model_p3 = DummyP3Model(vocab_size=_vocab_size, embed_dim=10)
    _data_p3 = [
        (
            torch.randn(1, 1, 10, 10),
            torch.randint(0, _vocab_size, (_seq_len,))
        )
        for _ in range(5)
    ]
    _loader_p3 = DataLoader(_data_p3, batch_size=2)
    _optim_p3 = optim.Adam(_model_p3.parameters(), lr=0.01)
    _crit_p3 = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    try:
        train_epoch(
            _model_p3, _loader_p3, _crit_p3, _optim_p3, _device, 0, 1,
            phase=3, pad_token_id=PAD_TOKEN_ID
        )
        evaluate_model(
            _model_p3, _loader_p3, _crit_p3, _device, phase=3,
            pad_token_id=PAD_TOKEN_ID
        )
        logger.info("✅ Phase 3 trainer calls OK.")
    except Exception as e:
        logger.error(f"❌ Phase 3 Test Error: {e}", exc_info=True)

    logger.info("\n✅ trainer.py test execution finished.")
