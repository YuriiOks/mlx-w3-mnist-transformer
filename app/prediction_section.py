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

    # --- Placeholder Inference Logic ---
    with st.spinner("Running model inference..."):
        # TODO: Replace with actual MLX/PyTorch inference
        # Example: pred = model(input_img)
        # For now, use dummy output
        import numpy as np
        if phase_loaded == 1:
            pred_digit = np.random.randint(0, 10)
            probs = np.random.dirichlet(np.ones(10), size=1)[0]
            st.success(f"Predicted Digit: {pred_digit}")
            from app.utils_streamlit import plot_probabilities
            plot_probabilities(probs, phase_loaded, result_area)
        elif phase_loaded == 2:
            st.info("Phase 2 prediction logic not implemented yet.")
        elif phase_loaded == 3:
            st.info("Phase 3 prediction logic not implemented yet.")
        else:
            st.warning("Unknown phase loaded.")
    return
