# MNIST Digit Classifier (Transformer) - Model Loader
# File: app/model_loader.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Model loading logic for Streamlit app.
# Created: 2025-05-02

from PIL import Image
import numpy as np
import torch
from pathlib import Path
from src.mnist_transformer.model import VisionTransformer, EncoderDecoderViT
import pickle

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
    if framework.lower() == "pytorch":
        # Find checkpoint directory
        model_path_obj = Path(model_path)
        checkpoint_dir = model_path_obj.parent
        state_path = checkpoint_dir / "training_state_pt.pkl"
        if not model_path_obj.exists() or not state_path.exists():
            return (None, 0)
        # Load state dict and config
        device = torch.device("cpu")
        model_state_dict = torch.load(model_path_obj, map_location=device)
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        model_cfg = state.get('model_config', config.get('model', {}))
        dataset_cfg = state.get('dataset_config', config.get('dataset', {}))
        phase = state.get('phase', config.get('phase', 1))
        # Instantiate model
        if phase == 1 or phase == 2:
            img_size = dataset_cfg.get(f'image_size_phase{phase}', dataset_cfg.get('image_size', 28))
            patch_size = dataset_cfg.get('patch_size', 7)
            num_outputs = dataset_cfg.get(f'num_outputs_phase{phase}', 1)
            num_classes = dataset_cfg.get('num_classes', 10)
            model = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=dataset_cfg.get('in_channels', 1),
                num_classes=num_classes,
                embed_dim=model_cfg.get('embed_dim', 64),
                depth=model_cfg.get('depth', 4),
                num_heads=model_cfg.get('num_heads', 4),
                mlp_ratio=model_cfg.get('mlp_ratio', 2.0),
                dropout=model_cfg.get('dropout', 0.1),
                num_outputs=num_outputs
            )
        elif phase == 3:
            img_size = dataset_cfg.get('image_size_phase3', 64)
            patch_size = dataset_cfg.get('patch_size_phase3', 8)
            decoder_vocab_size = config.get('tokenizer', {}).get('vocab_size', 13)
            model = EncoderDecoderViT(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=dataset_cfg.get('in_channels', 1),
                encoder_embed_dim=model_cfg.get('embed_dim', 64),
                encoder_depth=model_cfg.get('depth', 4),
                encoder_num_heads=model_cfg.get('num_heads', 4),
                decoder_vocab_size=decoder_vocab_size,
                decoder_embed_dim=model_cfg.get('decoder_embed_dim', 64),
                decoder_depth=model_cfg.get('decoder_depth', 4),
                decoder_num_heads=model_cfg.get('decoder_num_heads', 4),
                mlp_ratio=model_cfg.get('mlp_ratio', 2.0),
                dropout=model_cfg.get('dropout', 0.1)
            )
        else:
            return (None, 0)
        model.load_state_dict(model_state_dict)
        model.eval()
        return (model, phase)
    # ...existing code for MLX or fallback...
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
    pil_img = pil_img.resize((input_size, input_size)).convert("L")
    arr = np.array(pil_img).astype(np.float32) / 255.0
    # Invert if background is white (assume drawn images are white on black)
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    if framework == "PyTorch":
        # Normalize with MNIST mean/std
        arr = (arr - 0.1307) / 0.3081
        arr = arr[None, :, :]  # CHW
    else:
        arr = arr[:, :, None]  # HWC
    return arr
