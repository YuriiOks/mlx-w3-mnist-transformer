# MNIST Digit Classifier (Transformer) - Model Loader
# File: app/model_loader.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Model loading logic for Streamlit app.
# Created: 2025-05-02

def load_selected_model(
    framework,
    config,
    model_path,
    run_name
):
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
