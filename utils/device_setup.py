# MNIST Digit Classifier (Transformer)
# File: utils/device_setup.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Detects and sets the appropriate PyTorch device.
# Created: 2025-04-28
# Updated: 2025-04-28

import torch
import os
# Import the logger instance directly from the logging module within the same package
# This assumes logging.py and __init__.py are set up correctly in the utils folder
# The logger instance gets configured automatically on first import via logging.py
from .logging import logger

# --- Recommended Environment Variables (Informational) ---
# Consider setting these externally for stability:
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Allows PyTorch to fall back to CPU for unsupported MPS ops
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" # Can sometimes help allocate more memory

def get_device() -> torch.device:
    """
    Checks for MPS (Apple Silicon GPU), CUDA (NVIDIA GPU), or falls back to CPU.

    Logs the selected device using the project's logger ('DropoutDisco').

    Returns:
        torch.device: The selected PyTorch device object.
    """
    selected_device = None
    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()

    # Use the imported logger instance directly
    logger.debug("⚙️  Checking for available hardware accelerators...")

    if torch.backends.mps.is_available():
        # Check if MPS is available and built
        selected_device = torch.device("mps")
        logger.info(f"✅ MPS device found and available (Built: {mps_built}). Selecting MPS.")
        fallback_enabled = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"
        logger.debug(f"  -> MPS fallback to CPU is {'ENABLED' if fallback_enabled else 'DISABLED'}.")

    else:
        # Fallback to CPU
        selected_device = torch.device("cpu")
        logger.warning(f"⚠️ MPS not available (Available: {torch.backends.mps.is_available()}, Built: {mps_built}). Falling back to CPU.")
        # Add if CUDA check exists:
        # logger.warning(f"⚠️ MPS/CUDA not available. Falling back to CPU.")

    # Log the final selected device clearly
    logger.info(f"✨ Selected compute device: {selected_device.type.upper()}")
    return selected_device

# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    # When running this script directly (python utils/device_setup.py),
    # the logger setup in logging.py (triggered by the import) should run first.
    logger.info("🚀 Running device setup check directly...")
    device = get_device()
    logger.info(f"🔍 Device object returned: {device}")

    # Try creating a small tensor on the selected device
    try:
        logger.debug(f"Attempting to create test tensor on device '{device}'...")
        x = torch.randn(3, 3, device=device)
        logger.info(f"✅ Successfully created test tensor on device '{device}'.")
        logger.debug(f"Test tensor value:\n{x}")
        # Optional: For MPS, check if it's actually on the backend
        if device.type == 'mps':
             logger.debug(f"  Tensor backend device check: {x.device}")
    except Exception as e:
        logger.error(f"❌ Failed to create test tensor on device '{device}': {e}", exc_info=True) # Log traceback info