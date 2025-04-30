# MNIST Digit Classifier (Transformer)
# File: utils/device_setup.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Detects and sets the appropriate PyTorch device.
# Created: 2025-04-28
# Updated: 2025-04-30

import torch
import os
# Import the logger instance directly from the logging module within the same package
from .logging import logger

# --- Recommended Environment Variables (Informational) ---
# Consider setting these externally for stability:
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def get_device(
) -> torch.device:
    """
    Detects and returns the most suitable PyTorch device.

    Checks for Apple Silicon MPS (Metal Performance Shaders) support,
    then falls back to CPU if unavailable. Logs the selected device
    and relevant environment variable status using the project's logger.

    Returns:
        torch.device: The selected PyTorch device object, either 'mps'
            if available and built, or 'cpu' otherwise.
    """
    selected_device = None
    mps_built = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
    )

    logger.debug(
        "⚙️  Checking for available hardware accelerators..."
    )

    if torch.backends.mps.is_available():
        selected_device = torch.device("mps")
        logger.info(
            f"✅ MPS device found and available (Built: {mps_built}). "
            "Selecting MPS."
        )
        fallback_enabled = (
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"
        )
        logger.debug(
            f"  -> MPS fallback to CPU is "
            f"{'ENABLED' if fallback_enabled else 'DISABLED'}."
        )
    else:
        selected_device = torch.device("cpu")
        logger.warning(
            f"⚠️ MPS not available (Available: "
            f"{torch.backends.mps.is_available()}, Built: {mps_built}). "
            "Falling back to CPU."
        )

    logger.info(
        f"✨ Selected compute device: {selected_device.type.upper()}"
    )
    return selected_device

# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    logger.info("🚀 Running device setup check directly...")
    device = get_device()
    logger.info(f"🔍 Device object returned: {device}")

    try:
        logger.debug(
            f"Attempting to create test tensor on device '{device}'..."
        )
        x = torch.randn(3, 3, device=device)
        logger.info(
            f"✅ Successfully created test tensor on device '{device}'."
        )
        logger.debug(f"Test tensor value:\n{x}")
        if device.type == 'mps':
            logger.debug(
                f"  Tensor backend device check: {x.device}"
            )
    except Exception as e:
        logger.error(
            f"❌ Failed to create test tensor on device '{device}': {e}",
            exc_info=True
        )