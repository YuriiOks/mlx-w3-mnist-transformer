import streamlit as st
from pathlib import Path

# Assuming load_selected_model is defined elsewhere and imported
# from some_module import load_selected_model

model_base_dir = Path("/path/to/models")
selected_framework = "PyTorch"  # or "TensorFlow"
config = {}  # Assuming config is defined elsewhere

model = None
phase_loaded = None

# Sidebar for model selection and loading
model_run_name = st.sidebar.text_input("Model Run Name")

if model_run_name:
    ext = ".pth" if selected_framework == "PyTorch" else ".safetensors"
    fname = "model_final" if selected_framework == "PyTorch" else "model_weights"
    model_path = model_base_dir / model_run_name / f"{fname}{ext}"
    model_path_str = str(model_path)
    if st.sidebar.button("Load Model"):
        if model_path.exists():
            # Use the correct loader function and update global model/phase_loaded
            model_loaded, phase_loaded_loaded = load_selected_model(selected_framework, config, model_path_str, model_run_name)
            if model_loaded:
                model = model_loaded
                phase_loaded = phase_loaded_loaded
                st.sidebar.success(f"Loaded: {model_run_name}")
            else:
                model = None
                st.sidebar.error("Load failed, check logs.")
        else:
            st.sidebar.error(f"File not found: {model_path}")
            model = None
    # Display currently loaded model info
    if model is not None:
         st.sidebar.info(f"**Current Model:**\n{model_run_name}\n(Phase {phase_loaded})")
    else:
         st.sidebar.info("No model loaded.")