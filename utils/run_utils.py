# utils/run_utils.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Utility functions for running experiments.
# Created: 2025-04-28

import os
import json
import yaml
import matplotlib.pyplot as plt
from typing import List
from .logging import logger

def format_num_words(num_words: int) -> str:
    '''Formats large numbers for filenames.'''
    if num_words == -1: return "All"
    if num_words >= 1_000_000: return f"{num_words // 1_000_000}M"
    if num_words >= 1_000: return f"{num_words // 1_000}k"
    return str(num_words)

def load_config(config_path: str = "config.yaml") -> dict | None:
    '''Loads configuration from a YAML file.'''
    logger.info(f"🔍 Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Configuration loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        return None

def save_losses(losses: List[float], save_dir: str, 
                filename: str = "training_losses.json") -> str | None:
    '''Saves epoch losses to a JSON file.'''
    if not os.path.isdir(save_dir):  # This might fail if dir creation raced
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

def plot_losses(losses: List[float], save_dir: str,
               filename: str = "training_loss.png") -> str | None:
    '''Plots epoch losses and saves the plot.'''
    if not losses: return None  # Maybe epoch_losses is empty?
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