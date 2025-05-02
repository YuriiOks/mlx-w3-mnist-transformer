# MNIST Digit Classifier (Transformer) - Model Loader
# File: app/model_loader.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Model loading logic for Streamlit app.
# Created: 2025-05-02

from PIL import Image
import numpy as np

def load_selected_model(
    framework: str,
    config: dict,
    model_path: str,
    run_name: str
) -> tuple:
    """
    Loads the selected model from disk based on the specified framework,
    configuration, model path, and run name.

    Args:
        framework (str): The ML framework to use (e.g., 'pytorch', 'mlx').
        config (dict): Configuration dictionary for the model.
        model_path (str): Path to the saved model file.
        run_name (str): Name of the training run or experiment.

    Returns:
        tuple: (model_object, phase_loaded)
            model_object: Loaded model instance or a placeholder object.
            phase_loaded (int): The phase or epoch loaded, or 0 on failure.

    Note:
        This is a placeholder implementation. Actual model loading logic
        for PyTorch or MLX should be implemented here.
    """
    # TODO: Implement actual model loading logic for PyTorch/MLX
    # For now, return a dummy object and phase
    return (
        f"LoadedModel({framework}, {run_name})",
        config.get('phase', 0)
    )

def preprocess_image(
    pil_img: Image.Image,
    input_size: int,
    framework: str
) -> np.ndarray:
    """
    Resize and normalize a PIL image for model input.

    Args:
        pil_img (PIL.Image.Image): Input image to preprocess.
        input_size (int): Target size (width and height) for resizing.
        framework (str): ML framework ('PyTorch' or 'MLX') to determine
            output format.

    Returns:
        np.ndarray: Preprocessed image as a float32 numpy array.
            - For MLX: shape (H, W, C), values in [0, 1].
            - For PyTorch: shape (C, H, W), values in [0, 1].

    Notes:
        - Converts grayscale images to (H, W, 1) if needed.
        - Uses LANCZOS resampling for resizing.
    """
    img = pil_img.resize((input_size, input_size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if framework == "PyTorch":
        # Return CHW for PyTorch
        arr = arr.transpose(2, 0, 1)
    return arr
