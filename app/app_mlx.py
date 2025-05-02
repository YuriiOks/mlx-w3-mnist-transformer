# MNIST Digit Classifier (Transformer) - MLX Version
# File: app/app_mlx.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Streamlit app for visualizing ViT inference with MLX.
# Created: 2025-04-30
# Updated: 2025-04-30

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image, ImageOps # Need ImageOps for centering potentially
import os
import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt # For probability plots
import math
from typing import Optional
# --- Constants ---
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mnist") if os.name == 'posix' else os.path.join(os.path.expanduser("~"), "MNIST")

# --- Add project root to sys.path ---
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    print(f"🎨 [App] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

# --- Imports from Project ---
try:
    from utils import logger, load_config
    from src.mnist_transformer_mlx.model_mlx import VisionTransformerMLX, EncoderDecoderViTMLX
    from src.mnist_transformer_mlx.dataset_mlx import numpy_normalize, MNIST_MEAN, MNIST_STD
    # Import tokenizer utils for Phase 3 decoding
    from utils.tokenizer_utils import (
        sequence_to_labels, DECODER_VOCAB_SIZE,
        PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID, ID_TO_DECODER_TOKEN
    )
    # Try importing torchvision for base dataset loading (for P3 generator)
    try:
         from torchvision import datasets as tv_datasets
         TORCHVISION_AVAILABLE = True
    except ImportError:
         TORCHVISION_AVAILABLE = False
         st.warning("torchvision not found, Phase 3 'Generate Sample' disabled.")

except ImportError as e:
    st.error(f"Failed to import project modules: {e}. Run from project root.")
    sys.exit(1)

# --- Constants ---
DEFAULT_MODEL_RUN = "MLX_Phase3_E50_LR0.0001_B64_ViT" # Default to latest P3 run

# --- Helper Functions ---

@st.cache_resource # Cache config loading
def cached_load_config(config_path):
    logger.info("Loading config for app...")
    return load_config(config_path)

@st.cache_resource # Cache model loading
def cached_load_model(config, model_path_str, run_name):
    """Loads the appropriate MLX model based on run name."""
    logger.info(f"Loading MLX model for run '{run_name}'...")
    cfg_model = config.get('model', {})
    cfg_dataset = config.get('dataset', {})
    tokenizer_cfg = config.get('tokenizer', {})

    phase = 1 # Default
    model_class = VisionTransformerMLX
    if "Phase2" in run_name: phase = 2; model_class = VisionTransformerMLX
    elif "Phase3" in run_name: phase = 3; model_class = EncoderDecoderViTMLX

    try:
        logger.info(f"Instantiating {model_class.__name__} for Phase {phase}...")
        if phase == 1 or phase == 2:
            img_size = cfg_dataset.get(f'image_size_phase{phase}', cfg_dataset['image_size'])
            patch_size = cfg_dataset.get('patch_size')
            num_outputs = cfg_dataset.get(f'num_outputs_phase{phase}', 1)
            model = model_class(
                img_size=img_size, patch_size=patch_size, in_channels=cfg_dataset['in_channels'],
                num_classes=cfg_dataset['num_classes'], embed_dim=cfg_model['embed_dim'],
                depth=cfg_model['depth'], num_heads=cfg_model['num_heads'],
                mlp_ratio=cfg_model['mlp_ratio'], dropout=0.0, attention_dropout=0.0, # Use dropout=0 for inference
                num_outputs=num_outputs
            )
        elif phase == 3:
            img_size = cfg_dataset.get('image_size_phase3')
            patch_size = cfg_dataset.get('patch_size_phase3')
            decoder_vocab_size = tokenizer_cfg.get('vocab_size')
            max_seq_len = cfg_dataset.get('max_seq_len')
            model = model_class(
                 img_size=img_size, patch_size=patch_size, in_channels=cfg_dataset['in_channels'],
                 encoder_embed_dim=cfg_model['embed_dim'], encoder_depth=cfg_model['depth'], encoder_num_heads=cfg_model['num_heads'],
                 decoder_vocab_size=decoder_vocab_size, decoder_embed_dim=cfg_model['decoder_embed_dim'],
                 decoder_depth=cfg_model['decoder_depth'], decoder_num_heads=cfg_model['decoder_num_heads'],
                 max_seq_len=max_seq_len, mlp_ratio=cfg_model['mlp_ratio'],
                 dropout=cfg_model.get('dropout', 0.1)
            )
        else: raise ValueError(f"Unknown phase {phase}")

        if Path(model_path_str).exists():
             model.load_weights(model_path_str)
             mx.eval(model.parameters()) # Realize parameters
             model.eval() # Set model layers (like dropout) to evaluation mode
             logger.info(f"✅ Weights loaded successfully into {model_class.__name__}.")
             return model, phase
        else: raise FileNotFoundError(f"Weights file not found: {model_path_str}")
    except Exception as e:
        st.error(f"Failed to instantiate/load model from {model_path_str}: {e}")
        logger.error(f"Model load failed: {e}", exc_info=True)
        return None, phase


def preprocess_image(pil_image: Image.Image, target_size: int) -> Optional[np.ndarray]:
    """Converts PIL Image -> Grayscale -> Resize -> Normalize -> NumPy (H, W, C)."""
    try:
        img = pil_image.convert('L') # Ensure grayscale
        # Optional: Add centering logic here if needed (using PIL/Numpy)
        # img = center_digit(img) # If you implement center_digit for PIL/NP
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        img_np = np.array(img, dtype=np.uint8).reshape(target_size, target_size, 1)
        normalized_np = numpy_normalize(img_np)
        return normalized_np
    except Exception as e:
        st.error(f"Failed to preprocess image: {e}")
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return None

def plot_probabilities(probs: np.ndarray, phase: int, result_area):
    """Displays probability bar chart."""
    with result_area:
        if phase == 1:
            labels = [str(i) for i in range(10)]
            fig, ax = plt.subplots(figsize=(6, 2))
            bars = ax.bar(labels, probs.flatten(), color='skyblue')
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.set_title("Output Probabilities")
            # Add probability values on top of bars
            for bar in bars:
                 height = bar.get_height()
                 ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                         f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            st.pyplot(fig)
        elif phase == 2:
            labels = [str(i) for i in range(10)]
            fig, axes = plt.subplots(2, 2, figsize=(7, 4))
            fig.suptitle("Output Probabilities per Position")
            pos = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
            probs_flat = probs.reshape(4, 10)
            for i, ax in enumerate(axes.flat):
                 bars = ax.bar(labels, probs_flat[i], color='skyblue')
                 ax.set_title(pos[i])
                 ax.set_ylim(0, 1)
                 ax.set_xticks(range(10))
                 ax.tick_params(axis='x', labelsize=8)
                 for bar in bars: # Add values
                      height = bar.get_height()
                      ax.text(bar.get_x()+bar.get_width()/2., height+0.02, f'{height:.1f}', ha='center', va='bottom', fontsize=6)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            st.pyplot(fig)
        # Phase 3 plot might be too complex (probabilities per sequence step) - skip for now


# --- App Layout & Logic ---
st.set_page_config(page_title="MNIST ViT Kitchen (MLX)", layout="wide")
st.image("https://ml-institute.com/images/logo.png", width=150) # Placeholder logo
st.title("🍳 The MNIST Transformer Kitchen (MLX Version)")
st.write("**Chef:** Team Backprop Bunch (Yurii, Amy, Guillaume, Aygun)")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("✨ Chef's Station ✨")

# Load Config
config = cached_load_config(project_root / "config.yaml")
model = None
phase_loaded = 0

if not config:
    st.error("Failed to load config.yaml!")
else:
    # Model Selection
    st.sidebar.subheader("1. Select Model 🤖")
    model_base_dir = project_root / config['paths']['model_save_dir']
    try:
        available_runs = sorted([d.name for d in model_base_dir.iterdir() if d.is_dir() and d.name.startswith("MLX_")])
    except FileNotFoundError:
        available_runs = []
        st.sidebar.error(f"Model directory not found: {model_base_dir}")

    if not available_runs:
        st.sidebar.warning("No trained MLX models found in models/mnist_vit/")
    else:
        # Try to find a reasonable default
        selected_index = 0
        for i, run_name in enumerate(available_runs):
            if DEFAULT_MODEL_RUN in run_name: # Check if default name exists
                 selected_index = i; break
        model_run_name = st.sidebar.selectbox(
            "Select Trained MLX Model Run:", available_runs, index=selected_index
        )
        if model_run_name:
            model_path = model_base_dir / model_run_name / "model_weights.safetensors"
            model, phase_loaded = cached_load_model(config, str(model_path), model_run_name)
            if model:
                 st.sidebar.success(f"Loaded Phase {phase_loaded} Model: \n{model_run_name}")
            else:
                 st.sidebar.error(f"Failed to load: {model_run_name}")

# --- Main Area ---
col_input, col_output = st.columns([1, 1]) # Input on left, output on right

# Input Selection
with col_input:
    st.header("1. Prepare Your Ingredient (Image) 🖼️")
    if model and phase_loaded > 0:
        dataset_cfg = config.get('dataset', {})
        img_array_normalized = None
        input_pil_display = None # Image to display before normalization

        # Set target size based on loaded model's phase
        if phase_loaded == 1: input_size = dataset_cfg.get('image_size', 28)
        elif phase_loaded == 2: input_size = dataset_cfg.get('image_size_phase2', 56)
        elif phase_loaded == 3: input_size = dataset_cfg.get('image_size_phase3', 64)
        else: input_size = 28 # Fallback

        input_method = st.radio(
            f"Input method for Phase {phase_loaded} (Target Size: {input_size}x{input_size})",
            ("Draw Digit", "Upload Image", "Generate Sample" if phase_loaded==3 and TORCHVISION_AVAILABLE else None),
            key=f"input_phase_{phase_loaded}", # Key changes with phase
            horizontal=True
        )
        # Filter out None option
        input_method = input_method if input_method else "Draw Digit"


        if input_method == "Draw Digit":
            st.write("Draw a single digit (0-9) in the center:")
            stroke_width = 20 # Fixed stroke width for drawing
            canvas_size = 280 # Larger canvas
            canvas_result = st_canvas(
                fill_color="black", stroke_width=stroke_width, stroke_color="white",
                background_color="black", height=canvas_size, width=canvas_size,
                drawing_mode="freedraw", key=f"canvas_p{phase_loaded}",
            )
            if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0: # Check if something was drawn
                img_drawn = canvas_result.image_data.astype(np.uint8)
                input_pil_display = Image.fromarray(img_drawn).convert("L")
                img_array_normalized = preprocess_image(input_pil_display, input_size)

        elif input_method == "Upload Image":
            uploaded_file = st.file_uploader("Upload an MNIST-like image...", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                input_pil_display = Image.open(uploaded_file).convert("L")
                img_array_normalized = preprocess_image(input_pil_display, input_size)

        elif input_method == "Generate Sample" and phase_loaded == 3:
            st.write("Generate a random dynamic layout image:")
            if st.button("Generate"):
                 # Load base data for generator
                 data_dir = config['paths'].get('data_dir', DEFAULT_DATA_DIR)
                 base_test_pil = None; base_test_lbl = None
                 try:
                      ds = tv_datasets.MNIST(root=data_dir, train=False, download=True, transform=None)
                      base_test_pil = [img for img, _ in ds]
                      base_test_lbl = np.array([lbl for _, lbl in ds], dtype=np.uint32)
                 except Exception as e: st.error(f"Failed MNIST load: {e}")

                 if base_test_pil and base_test_lbl is not None:
                     from src.mnist_transformer_mlx.dataset_mlx import generate_dynamic_digit_image_seq_np # Import generator
                     cfg_ds = config.get('dataset', {}); cfg_tk = config.get('tokenizer', {})
                     canvas_np, target_seq_np = generate_dynamic_digit_image_seq_np(
                         base_test_pil, base_test_lbl,
                         canvas_size=input_size, # Use P3 size
                         max_digits=cfg_ds.get('max_digits_phase3', 7),
                         augment_digits=True,
                         max_seq_len=cfg_ds.get('max_seq_len', 10),
                         start_token_id=cfg_tk.get('start_token_id', 1),
                         end_token_id=cfg_tk.get('end_token_id', 2),
                         pad_token_id=cfg_tk.get('pad_token_id', 0),
                     )
                     if canvas_np is not None:
                         img_array_normalized = numpy_normalize(canvas_np)
                         input_pil_display = Image.fromarray(canvas_np.squeeze().astype(np.uint8))
                         # Display ground truth for generated sample
                         decoded_labels = sequence_to_labels(target_seq_np.tolist())
                         st.info(f"Generated Sample Ground Truth: `{decoded_labels}`")

        # Display the input image before normalization
        if input_pil_display:
             st.image(input_pil_display, caption='Input Image (Before Preprocessing)', width=canvas_size if input_method == "Draw Digit" else 224)


# Prediction Area
with col_output:
    st.header("2. Chef's Analysis (Prediction) 🧠")
    # --- Fix: Ensure img_array_normalized is defined before use ---
    if 'img_array_normalized' in locals() and img_array_normalized is not None and model is not None:
        # Prepare input for MLX model (add batch dim)
        input_mlx = mx.array(img_array_normalized[np.newaxis, ...]) # Shape (1, H, W, C)

        st.write(f"**Running Phase {phase_loaded} Inference...**")
        st.write(f"Input shape to model: `{input_mlx.shape}`")

        start_time = time.time()

        # --- Inference Logic ---
        if phase_loaded == 1 or phase_loaded == 2:
            logits = model(input_mlx)
            mx.eval(logits) # Force computation
            probs = mx.softmax(logits, axis=-1)
            if phase_loaded == 1:
                 prediction = mx.argmax(logits, axis=-1).item()
                 confidence = mx.max(probs).item() * 100
                 st.success(f"**Predicted Digit:** `{prediction}`")
                 st.info(f"**Confidence:** `{confidence:.2f}%`")
                 plot_probabilities(np.array(probs), phase=1, result_area=st)
            else: # Phase 2
                 predictions = mx.argmax(logits, axis=-1).squeeze().tolist()
                 confidences = mx.max(probs, axis=-1).squeeze().tolist()
                 st.success(f"**Predicted Digits (TL, TR, BL, BR):** `{predictions}`")
                 st.info(f"**Confidences:** `{[f'{c*100:.1f}%' for c in confidences]}`")
                 plot_probabilities(np.array(probs), phase=2, result_area=st)

        elif phase_loaded == 3:
             # --- Autoregressive Decoding ---
             max_len = config['dataset']['max_seq_len']
             start_id = config['tokenizer']['start_token_id']
             end_id = config['tokenizer']['end_token_id']
             pad_id = config['tokenizer']['pad_token_id']

             decoder_input_ids = mx.array([[start_id]]) # Start token (B=1, Seq=1)
             generated_tokens = []
             st.write("**Generated Sequence:**")
             result_placeholder = st.empty()
             current_output_str = ""

             with st.spinner("Generating sequence step-by-step..."):
                  for step in range(max_len - 1):
                       logits = model(input_mlx, decoder_input_ids) # Pass image + current sequence
                       mx.eval(logits)
                       next_token_logits = logits[:, -1, :] # Logits for the last token: (1, VocabSize)
                       next_token_id = mx.argmax(next_token_logits, axis=-1) # Greedy prediction: (1,)
                       token_id_item = next_token_id.item()

                       generated_tokens.append(token_id_item)

                       # Display progress
                       token_str = ID_TO_DECODER_TOKEN.get(token_id_item, "?")
                       current_output_str += f" {token_str}"
                       result_placeholder.text(f"Step {step+1}: {current_output_str}")
                       time.sleep(0.1) # Small delay for visual effect

                       if token_id_item == end_id: break # Stop if END token

                       # Prepare input for next step
                       decoder_input_ids = mx.concatenate(
                            [decoder_input_ids, next_token_id.reshape(1, 1)], axis=1
                       )
             st.success("Sequence generation complete.")
             decoded_labels = sequence_to_labels(generated_tokens)
             st.markdown("---")
             st.markdown(f"**Final Decoded Labels:** `{decoded_labels}`")
             st.markdown(f"**Raw Token IDs:** `{generated_tokens}`")
             st.markdown("---")

        # --- End Inference Logic ---

        end_time = time.time()
        st.sidebar.metric("Inference Time (sec)", f"{end_time - start_time:.4f}")

    elif model is None and model_run_name:
        st.warning("Model could not be loaded. Please check logs and path.")
    else:
        st.info("Provide input using the sidebar to see the prediction.")