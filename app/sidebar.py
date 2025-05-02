# MNIST Digit Classifier (Transformer) - Sidebar Controls
# File: app/sidebar.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Handles sidebar controls for framework, phase, and model selection.
# Created: 2025-05-02
# Updated: 2025-05-02

import streamlit as st

def sidebar_controls(config):
    """Handles sidebar controls: framework, phase, model selection, load button."""
    st.sidebar.header("✨ Chef's Station ✨")
    # Framework selection
    framework_options = []
    if config.get('pytorch_available', True):
        framework_options.append("PyTorch")
    if config.get('mlx_available', True):
        framework_options.append("MLX")
    selected_framework = st.sidebar.radio("Choose ML Framework:", framework_options, key="framework_choice")
    # Phase selection
    st.sidebar.subheader("2. Select Cooking Phase")
    selected_phase = st.sidebar.radio("Choose Phase:", (1, 2, 3), format_func=lambda x: f"Phase {x}", key="phase_choice")
    # Model selection
    st.sidebar.subheader("3. Select Trained Model")
    model_base_dir = config.get('model_base_dir', 'models/mnist_vit')
    prefix = "PT_" if selected_framework == "PyTorch" else "MLX_"
    phase_prefix = f"{prefix}Phase{selected_phase}_"
    import os
    from pathlib import Path
    try:
        available_runs = sorted([
            d.name for d in Path(model_base_dir).iterdir()
            if d.is_dir() and d.name.startswith(phase_prefix)
        ], reverse=True)
    except Exception:
        available_runs = []
    model_run_name = st.sidebar.selectbox(
        f"Select {selected_framework} Phase {selected_phase} Run:", available_runs) if available_runs else None
    # Load model button
    st.sidebar.subheader("4. Load Model")
    load_model_clicked = st.sidebar.button("Load Model")
    return {
        'framework': selected_framework,
        'phase': selected_phase,
        'model_run_name': model_run_name,
        'load_model_clicked': load_model_clicked,
        'model_base_dir': model_base_dir
    }
