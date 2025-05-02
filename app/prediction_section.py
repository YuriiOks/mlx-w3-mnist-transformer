# MNIST Digit Classifier (Transformer) - Prediction Section
# File: app/prediction_section.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, 
# Guillaume, Aygun)
# Description: Handles model inference and output display.
# Created: 2025-05-02
# Updated: 2025-05-02

import streamlit as st

def prediction_controls(
    config,
    session_state,
    input_state
):
    """
    Handles model inference and output display in the Streamlit app.

    Args:
        config (dict): Configuration parameters for the model and UI.
        session_state (dict): State dictionary for managing session 
            variables and model state.
        input_state (dict): State dictionary containing user input 
            (e.g., drawn digit).

    Returns:
        None. Displays prediction results and related UI elements in 
        the Streamlit app.
    """
    st.header("2. Chef's Analysis (Prediction) 🧠")
    result_area = st.empty()
    model = session_state.model
    phase_loaded = session_state.phase_loaded
    input_img = input_state.get('processed_image')
    display_img = input_state.get('display_image')

    if model is None:
        st.info("No model loaded. Please load a model from the sidebar.")
        return
    if input_img is None:
        st.info("Please provide an input image using the controls on the left.")
        return

    import numpy as np
    from app.utils_streamlit import plot_probabilities
    import torch
    import torch.nn.functional as F
    with st.spinner("Running model inference..."):
        if hasattr(model, 'forward') and hasattr(model, 'eval'):
            # Assume PyTorch model
            if phase_loaded == 1:
                # Preprocess input for PyTorch: (C, H, W), float32, batch dim
                img = input_img
                if img is not None:
                    # Ensure input is [1, 1, 28, 28] for PyTorch
                    img = np.asarray(img)
                    st.write(f"[DEBUG] Input tensor stats before model: shape={img.shape}, min={img.min()}, max={img.max()}, mean={img.mean():.4f}")
                    if img.shape == (28, 28):
                        img = img[None, None, :, :]
                    elif img.shape == (1, 28, 28):
                        img = img[None, :, :, :]
                    elif img.shape == (28, 28, 1):
                        img = img.transpose(2, 0, 1)[None, :, :, :]
                    elif img.shape == (1, 28, 28, 1):
                        img = img.transpose(0, 3, 1, 2)
                    elif img.shape == (28, 28, 3):
                        img = img.transpose(2, 0, 1)[None, :1, :, :]
                    elif img.shape == (1, 28, 28, 3):
                        img = img.transpose(0, 3, 1, 2)[:, :1, :, :]
                    # else: assume already correct
                    img_tensor = torch.from_numpy(img).float()
                    st.write(f"[DEBUG] Input tensor after reshape: shape={img_tensor.shape}, min={img_tensor.min()}, max={img_tensor.max()}, mean={img_tensor.mean():.4f}")
                    with torch.no_grad():
                        logits = model(img_tensor)
                        probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
                        pred_digit = int(np.argmax(probs))
                    st.success(f"Predicted Digit: {pred_digit}")
                    plot_probabilities(probs, phase_loaded, result_area)
            elif phase_loaded == 2:
                img = input_img
                if img is not None:
                    img_tensor = torch.from_numpy(img).unsqueeze(0).float()
                    with torch.no_grad():
                        logits = model(img_tensor)
                        # logits shape: (1, 4, 10)
                        probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
                        pred_digits = np.argmax(probs, axis=-1)
                    st.success(f"Predicted Digits (TL, TR, BL, BR): {pred_digits.tolist()}")
                    plot_probabilities(probs, phase_loaded, result_area)
            elif phase_loaded == 3:
                st.info("Sequence prediction visualization not implemented.")
            else:
                st.warning("Phase not supported for prediction or probability plotting.")
        else:
            # Dummy fallback
            if phase_loaded == 1:
                pred_digit = np.random.randint(0, 10)
                probs = np.random.dirichlet(np.ones(10), size=1)[0]
                st.success(f"Predicted Digit: {pred_digit}")
                plot_probabilities(probs, phase_loaded, result_area)
            elif phase_loaded == 2:
                pred_digits = np.random.randint(0, 10, size=4)
                probs = np.random.dirichlet(np.ones(10), size=4)
                st.success(f"Predicted Digits (TL, TR, BL, BR): {pred_digits.tolist()}")
                plot_probabilities(probs, phase_loaded, result_area)
            elif phase_loaded == 3:
                seq_len = np.random.randint(1, 8)
                pred_seq = np.random.randint(0, 10, size=seq_len)
                st.success(f"Predicted Sequence: {pred_seq.tolist()}")
                st.info("(Sequence prediction visualization not implemented)")
            else:
                st.warning("Phase not supported for prediction or probability plotting.")
    return
