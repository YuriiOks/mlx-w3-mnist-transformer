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
    with st.spinner("Running model inference..."):
        if phase_loaded == 1:
            # Dummy: single digit prediction
            pred_digit = np.random.randint(0, 10)
            probs = np.random.dirichlet(np.ones(10), size=1)[0]
            st.success(f"Predicted Digit: {pred_digit}")
            plot_probabilities(probs, phase_loaded, result_area)
        elif phase_loaded == 2:
            # Dummy: 4 digits (2x2 grid)
            pred_digits = np.random.randint(0, 10, size=4)
            probs = np.random.dirichlet(np.ones(10), size=4) # 4x10
            st.success(f"Predicted Digits (TL, TR, BL, BR): {pred_digits.tolist()}")
            plot_probabilities(probs, phase_loaded, result_area)
        elif phase_loaded == 3:
            # Dummy: sequence prediction (up to 7 digits)
            seq_len = np.random.randint(1, 8)
            pred_seq = np.random.randint(0, 10, size=seq_len)
            st.success(f"Predicted Sequence: {pred_seq.tolist()}")
            st.info("(Sequence prediction visualization not implemented)")
        else:
            st.warning("Phase not supported for prediction or probability plotting.")
    return
