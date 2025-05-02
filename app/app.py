# MNIST Digit Classifier (Transformer) - Unified App
# File: app/app.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Main Streamlit app file.
# Created: 2025-05-02
# Updated: 2025-05-02

import streamlit as st
import sys
from pathlib import Path

# --- Add project root to sys.path ---
try:
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    if str(project_root) not in sys.path:
        print(f"🎨 [App] Adding project root to sys.path: {project_root}")
        sys.path.insert(0, str(project_root))
    # --- Now imports should work ---
    from utils import load_config
    from app.sidebar import sidebar_controls
    from app.input_section import input_controls
    from app.prediction_section import prediction_controls
except ImportError as e:
     st.error(f"ERROR: Failed to set up project path or import modules: {e}")
     st.error("Please ensure you run 'streamlit run app/app.py' from the project root directory.")
     st.stop()

# --- Main App Logic ---
st.set_page_config(page_title="MNIST ViT Kitchen", layout="wide")
st.title("🍳 The MNIST Transformer Kitchen")

# Load config (handle potential failure)
config = load_config()
if config is None:
     st.error("❌ Could not load config.yaml. Please ensure it exists in the project root.")
     st.stop()

# Initialize state for shared variables if needed (example)
if 'model' not in st.session_state:
    st.session_state.model = None
if 'phase_loaded' not in st.session_state:
    st.session_state.phase_loaded = 0
if 'model_run_name' not in st.session_state:
     st.session_state.model_run_name = None

# Render Sidebar and get selections
sidebar_state = sidebar_controls(config)

# Main area layout
col_input, col_output = st.columns([1, 1])

with col_input:
    input_state = input_controls(config, sidebar_state)

with col_output:
    prediction_controls(config, sidebar_state, input_state)

# Add footer or other elements if desired
st.markdown("---")
st.write("Built by Backprop Bunch 🎉")