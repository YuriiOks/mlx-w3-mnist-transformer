# utils/run_utils.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Utility functions for running experiments, including
# saving/plotting metrics.
# Created: 2025-04-28
# Updated: 2025-04-30

import os
import json
import yaml
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from .logging import logger

# --- Config Loading (Keep as is) ---
def load_config(
    config_path: str = "config.yaml"
) -> dict | None:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict | None: Loaded configuration as a dictionary, or None if
        loading fails.
    """
    logger.info(f"🔍 Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Configuration loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        return None

def format_num_words(
    num_words: int
) -> str:
    """
    Formats large numbers for filenames.

    Args:
        num_words (int): Number to format.

    Returns:
        str: Formatted string (e.g., 1000 -> '1k', 1000000 -> '1M').
    """
    if num_words == -1:
        return "All"
    if num_words >= 1_000_000:
        return f"{num_words // 1_000_000}M"
    if num_words >= 1_000:
        return f"{num_words // 1_000}k"
    return str(num_words)

# --- Saving Metrics ---
def save_metrics(
    metrics_history: Dict[str, List[float]],
    save_dir: str | Path,
    filename: str = "training_metrics.json"
) -> str | None:
    """
    Saves epoch metrics history to a JSON file.

    Args:
        metrics_history (Dict[str, List[float]]): Dictionary containing
            lists of metric values per epoch.
        save_dir (str | Path): Directory to save the JSON file.
        filename (str): Name of the JSON file.

    Returns:
        str | None: Path to the saved file, or None if saving fails.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / filename
    try:
        with open(metrics_file, 'w', encoding='utf-8') as f:
            serializable_metrics = {
                k: np.array(v).tolist()
                if isinstance(v, (list, np.ndarray)) else v
                for k, v in metrics_history.items()
            }
            json.dump(serializable_metrics, f, indent=2)
        logger.info(f"📊 Metrics history saved to: {metrics_file}")
        return str(metrics_file)
    except Exception as e:
        logger.error(f"❌ Failed to save metrics history: {e}")
        return None

# --- Plotting Metrics ---
def plot_metrics(
    metrics_history: Dict[str, List[float]],
    save_dir: str | Path,
    filename: str = "training_metrics.png"
) -> str | None:
    """
    Plots training and validation metrics (loss, accuracy) and saves
    the plot.

    Assumes keys like 'avg_train_loss', 'val_loss', 'val_accuracy'.

    Args:
        metrics_history (Dict[str, List[float]]): Dictionary containing
            lists of metric values per epoch.
        save_dir (str | Path): Directory to save the plot.
        filename (str): Name of the plot file.

    Returns:
        str | None: Path to the saved plot, or None if plotting fails.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_file = save_dir / filename

    required_keys = ['avg_train_loss']
    if not all(key in metrics_history for key in required_keys):
        logger.error(
            f"❌ Metrics history dictionary missing required keys "
            f"({required_keys}). Cannot plot."
        )
        return None

    train_loss = metrics_history.get('avg_train_loss', [])
    val_loss = metrics_history.get('val_loss', [])
    val_acc = metrics_history.get('val_accuracy', [])

    num_epochs = len(train_loss)
    if num_epochs == 0:
        logger.warning("⚠️ No epoch data found in metrics history to plot.")
        return None

    epochs = range(1, num_epochs + 1)

    num_plots = 1
    if val_loss:
        num_plots += 1
    if val_acc:
        num_plots += 1

    fig, axes = plt.subplots(
        num_plots, 1, figsize=(10, 5 * num_plots), sharex=True
    )
    if num_plots == 1:
        axes = [axes]
    plot_idx = 0

    # Plot Training & Validation Loss
    axes[plot_idx].plot(
        epochs, train_loss, marker='o', linestyle='-', label='Avg Train Loss'
    )
    if val_loss:
        if len(val_loss) == num_epochs:
            axes[plot_idx].plot(
                epochs, val_loss, marker='x', linestyle='--',
                label='Validation Loss'
            )
        else:
            logger.warning("Validation loss length mismatch, skipping plot.")
    axes[plot_idx].set_ylabel("Loss")
    axes[plot_idx].set_title("Training & Validation Loss per Epoch")
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, ls='--')
    plot_idx += 1

    # Plot Validation Accuracy
    if val_acc:
        if len(val_acc) == num_epochs:
            axes[plot_idx].plot(
                epochs, val_acc, marker='o', linestyle='-',
                color='green', label='Validation Accuracy'
            )
            axes[plot_idx].set_ylabel("Accuracy (%)")
            axes[plot_idx].set_title("Validation Accuracy per Epoch")
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, ls='--')
            plot_idx += 1
        else:
            logger.warning("Validation accuracy length mismatch, skipping plot.")

    axes[-1].set_xlabel("Epoch")
    if num_epochs < 20:
        axes[-1].set_xticks(epochs)

    plt.tight_layout()

    try:
        plt.savefig(plot_file)
        logger.info(f"📈 Metrics plot saved to: {plot_file}")
        plt.close(fig)
        return str(plot_file)
    except Exception as e:
        logger.error(f"❌ Failed to plot metrics: {e}")
        plt.close(fig)
        return None

def save_losses(
    losses: List[float],
    save_dir: str,
    filename: str = "training_losses.json"
) -> str | None:
    """
    Saves epoch losses to a JSON file.

    Args:
        losses (List[float]): List of loss values per epoch.
        save_dir (str): Directory to save the JSON file.
        filename (str): Name of the JSON file.

    Returns:
        str | None: Path to the saved file, or None if saving fails.
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    loss_file = os.path.join(save_dir, filename)
    try:
        with open(loss_file, 'w', encoding='utf-8') as f:
            json.dump({'epoch_losses': losses}, f, indent=2)
        logger.info(f"📉 Training losses saved to: {loss_file}")
        return loss_file
    except Exception as e:
        logger.error(f"❌ Failed to save losses: {e}")
        return None

def plot_losses(
    losses: List[float],
    save_dir: str,
    filename: str = "training_loss.png"
) -> str | None:
    """
    Plots epoch losses and saves the plot.

    Args:
        losses (List[float]): List of loss values per epoch.
        save_dir (str): Directory to save the plot.
        filename (str): Name of the plot file.

    Returns:
        str | None: Path to the saved plot, or None if plotting fails.
    """
    if not losses:
        return None
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plot_file = os.path.join(save_dir, filename)
    try:
        epochs = range(1, len(losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, marker='o')
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.xticks(epochs)
        plt.grid(True, ls='--')
        plt.savefig(plot_file)
        logger.info(f"📈 Training loss plot saved to: {plot_file}")
        plt.close()
        return plot_file
    except Exception as e:
        logger.error(f"❌ Failed to plot losses: {e}")
        return None