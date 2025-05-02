import streamlit as st
# MNIST Digit Classifier (Transformer) - Unified App
# File: app/app.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Main Streamlit app file.
# Created: 2025-05-02
# Updated: 2025-05-02

import sys
from pathlib import Path

st.set_page_config(
    page_title="MNIST ViT Kitchen",
    layout="wide"
)

# --- Add project root to sys.path ---
try:
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    if str(project_root) not in sys.path:
        print(
            f"🎨 [App] Adding project root to sys.path: "
            f"{project_root}"
        )
        sys.path.insert(0, str(project_root))
    # --- Now imports should work ---
    from utils import load_config
    from app.model_loader import load_selected_model
    from app.sidebar import sidebar_controls
    from app.input_section import input_controls
    from app.prediction_section import prediction_controls
except ImportError as e:
    st.error(
        f"ERROR: Failed to set up project path or import modules: {e}"
    )
    st.error(
        "Please ensure you run 'streamlit run app/app.py' from the "
        "project root directory."
    )
    st.stop()


def handle_model_loading(
    selected_framework: str,
    config: dict,
    model_base_dir: str,
    selected_run_name: str,
    load_model_clicked: bool
):
    """
    Handles loading of the selected model when the user clicks the button.

    Args:
        selected_framework (str): The ML framework selected by the user,
            e.g., "🤖 PyTorch" or "🦙 MLX".
        config (dict): The loaded configuration dictionary.
        model_base_dir (str): Base directory where models are stored.
        selected_run_name (str): The selected model run name.
        load_model_clicked (bool): Whether the load button was clicked.

    Returns:
        None. Updates Streamlit session state directly with the loaded
        model, phase, and run name. Displays sidebar messages for
        success or failure.
    """
    if load_model_clicked:
        if selected_run_name:
            ext = ".pth" if selected_framework == "🤖 PyTorch" \
                else ".safetensors"
            fname = (
                "model_final"
                if selected_framework == "🤖 PyTorch"
                else "model_weights"
            )
            model_path = (
                Path(model_base_dir)
                / selected_run_name
                / f"{fname}{ext}"
            )
            model_path_str = str(model_path)
            st.sidebar.info(f"DEBUG: Checking model path: {model_path_str}")
            st.sidebar.info(f"DEBUG: Exists? {model_path.exists()}")
            if model_path and model_path.exists():
                with st.spinner(
                    f"Loading {selected_framework} model: "
                    f"{selected_run_name}..."
                ):
                    loaded_model, loaded_phase = load_selected_model(
                        selected_framework.split(" ")[-1],
                        config,
                        model_path_str,
                        selected_run_name
                    )
                    st.session_state.model = loaded_model
                    st.session_state.phase_loaded = loaded_phase
                    st.session_state.model_run_name_loaded = (
                        selected_run_name
                    )
                    if loaded_model:
                        st.sidebar.success(
                            f"Loaded: {selected_run_name}"
                        )
                    else:
                        st.sidebar.error("Model loading failed.")
            else:
                st.sidebar.error(
                    f"Model file not found: {model_path}"
                )
                st.session_state.model = None
                st.session_state.phase_loaded = 0
                st.session_state.model_run_name_loaded = None
        else:
            st.sidebar.warning(
                "Please select a model run first."
            )


# --- Main App Logic ---
st.title("🍳 The MNIST Transformer Kitchen")

# Load config (handle potential failure)
config = load_config()
if config is None:
    st.error(
        "❌ Could not load config.yaml. Please ensure it exists in the "
        "project root."
    )
    st.stop()

# --- Initialize Session State ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'phase_loaded' not in st.session_state:
    st.session_state.phase_loaded = 0
if 'model_run_name_loaded' not in st.session_state:
    st.session_state.model_run_name_loaded = None
# --- End Session State ---

# Render Sidebar and get selections
sidebar_state = sidebar_controls(
    config
)
selected_framework = sidebar_state['framework']
selected_phase = sidebar_state['phase']
selected_run_name = sidebar_state['model_run_name']
load_model_clicked = sidebar_state['load_model_clicked']
model_base_dir = sidebar_state['model_base_dir']

# --- Handle Model Loading on Button Click ---
handle_model_loading(
    selected_framework=selected_framework,
    config=config,
    model_base_dir=model_base_dir,
    selected_run_name=selected_run_name,
    load_model_clicked=load_model_clicked
)
# --- After loading, set phase_loaded to sidebar selection ---
if load_model_clicked and st.session_state.model is not None:
    st.session_state.phase_loaded = selected_phase
# --- End Handle Model Loading ---

if st.session_state.model is not None:
    st.sidebar.info(
        f"**Current Model:**\n"
        f"`{st.session_state.model_run_name_loaded}`\n"
        f"(Phase {st.session_state.phase_loaded})"
    )
else:
    st.sidebar.info("No model loaded.")

# Main area layout
col_input, col_output = st.columns([1, 1])

with col_input:
    input_state = input_controls(
        config,
        st.session_state
    )

with col_output:
    result_area = st.empty()
    prediction_result = st.empty()
    if input_state.get('predict_clicked', False):
        prediction_controls(
            config,
            st.session_state,
            input_state
        )
    else:
        with result_area:
            st.write("")  # Placeholder for graph
        with prediction_result:
            st.info("Prediction result will appear here after you click 'Predict'.")

# Add footer or other elements if desired
st.markdown("---")
st.write("Built by Backprop Bunch 🎉")