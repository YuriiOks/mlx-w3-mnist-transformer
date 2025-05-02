# MNIST Digit Classifier (Transformer) - Input Section
# File: app/input_section.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Handles input image drawing, upload, and generation logic.
# Created: 2025-05-02
# Updated: 2025-05-02

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import time # For potential delay/spinner

# Assume needed functions and classes are imported in app.py and passed via session_state or config
# e.g., numpy_normalize, generate_dynamic_digit_image_seq_np, base datasets

# --- Imports for Phase 3 Generation ---
import os
import sys
from pathlib import Path
try: # Need to load base MNIST data for generation
    from torchvision import datasets as tv_datasets
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
try: # Need generator function
    from src.mnist_transformer_mlx.dataset_mlx import generate_dynamic_digit_image_seq_np, DEFAULT_DATA_DIR
    # Note: Using the NumPy generator function from the MLX dataset file
    GENERATOR_AVAILABLE = True
except ImportError:
    GENERATOR_AVAILABLE = False
try: # Need tokenizer utils for display
    from utils.tokenizer_utils import sequence_to_labels
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
# --- End Imports for Phase 3 ---


def input_controls(config: dict, session_state: st.session_state) -> dict:
    """
    Handles input image drawing/upload/generation logic based on loaded phase.

    Args:
        config (dict): The application configuration.
        session_state (st.session_state): Streamlit's session state,
                                           must contain 'phase_loaded'.

    Returns:
        dict: Dictionary containing input state, primarily:
              {'processed_image': numpy_array | None, 'display_image': pil_image | None}
              processed_image is normalized HWC numpy array for model input.
              display_image is the PIL image before normalization for display.
    """
    st.header("1. Prepare Your Ingredient (Image) 🖼️")

    phase_loaded = session_state.phase_loaded
    framework = session_state.get('framework', 'MLX') # Get selected framework

    # Return default state if no model/phase loaded
    if phase_loaded == 0:
        st.info("Load a model using the sidebar first.")
        return {'processed_image': None, 'display_image': None}

    dataset_cfg = config.get('dataset', {})
    processed_image_np = None # This will hold the HWC, normalized numpy array
    display_image_pil = None # This will hold the PIL image for display

    # --- Determine Input Size Based on Phase ---
    if phase_loaded == 1: input_size = dataset_cfg.get('image_size', 28)
    elif phase_loaded == 2: input_size = dataset_cfg.get('image_size_phase2', 56)
    elif phase_loaded == 3: input_size = dataset_cfg.get('image_size_phase3', 64)
    else: input_size = 28 # Fallback

    # --- Input Method Selection ---
    input_options = ["Draw Digit", "Upload Image"]
    if phase_loaded == 3 and TORCHVISION_AVAILABLE and GENERATOR_AVAILABLE:
        input_options.append("Generate Sample")

    input_method = st.radio(
        f"Input method for Phase {phase_loaded} (Target Size: {input_size}x{input_size}):",
        input_options, key=f"input_{phase_loaded}_{framework}", # Key includes framework
        horizontal=True
    )

    # --- Handle Different Input Methods ---
    if input_method == "Draw Digit":
        st.write("Draw a single digit (best for Phase 1):")
        canvas_size = 280
        stroke_width = st.slider( # Allow changing stroke width
            "Stroke width: ", 10, 35, 20, key=f"stroke_{phase_loaded}"
        )
        canvas_result = st_canvas(
            fill_color="black", stroke_width=stroke_width, stroke_color="white",
            background_color="black", height=canvas_size, width=canvas_size,
            drawing_mode="freedraw", key=f"canvas_{phase_loaded}",
        )
        if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0:
            img_drawn_rgba = canvas_result.image_data.astype(np.uint8)
            # Convert RGBA from canvas to Grayscale PIL
            display_image_pil = Image.fromarray(img_drawn_rgba).convert("L")
            # Preprocess for the required size
            from app.model_loader import preprocess_image # Use common preprocessor
            processed_image_np = preprocess_image(display_image_pil, input_size, framework)


    elif input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            f"Upload {input_size}x{input_size} image...", type=["png", "jpg", "jpeg"],
            key=f"upload_{phase_loaded}"
        )
        if uploaded_file is not None:
            display_image_pil = Image.open(uploaded_file).convert("L")
            from app.model_loader import preprocess_image # Use common preprocessor
            processed_image_np = preprocess_image(display_image_pil, input_size, framework)


    elif input_method == "Generate Sample" and phase_loaded == 3:
        st.write("Generate a random dynamic layout image (64x64):")
        if st.button("Generate", key=f"generate_{phase_loaded}"):
            # Load base data required by the generator
            data_dir = config['paths'].get('data_dir', DEFAULT_DATA_DIR)
            base_pil, base_lbl = None, None
            try:
                 ds = tv_datasets.MNIST(root=data_dir, train=False, download=True, transform=None)
                 base_pil = [img for img, _ in ds]
                 base_lbl = np.array([lbl for _, lbl in ds], dtype=np.uint32)
            except Exception as e: st.error(f"Failed MNIST load for generator: {e}")

            if base_pil and base_lbl is not None:
                 cfg_ds = config.get('dataset', {}); cfg_tk = config.get('tokenizer', {})
                 # Call the generator function directly (returns uint8 numpy, uint32 numpy sequence)
                 canvas_np, target_seq_np = generate_dynamic_digit_image_seq_np(
                     base_pil, base_lbl,
                     canvas_size=input_size, # Should be P3 size (64)
                     max_digits=cfg_ds.get('max_digits_phase3', 7),
                     augment_digits=True, # Use augmentation
                     max_seq_len=cfg_ds.get('max_seq_len', 10),
                     start_token_id=cfg_tk.get('start_token_id', START_TOKEN_ID),
                     end_token_id=cfg_tk.get('end_token_id', END_TOKEN_ID),
                     pad_token_id=cfg_tk.get('pad_token_id', PAD_TOKEN_ID),
                 )
                 if canvas_np is not None:
                      # Store the unnormalized PIL for display
                      display_image_pil = Image.fromarray(canvas_np.squeeze().astype(np.uint8))
                      # Store the normalized numpy array for processing
                      processed_image_np = numpy_normalize(canvas_np) # Use numpy normalize

                      # Display ground truth for generated sample
                      if TOKENIZER_AVAILABLE:
                           decoded_labels = sequence_to_labels(target_seq_np.tolist())
                           st.info(f"Generated Sample Ground Truth: `{decoded_labels}`")
                           # Store target sequence in session state? Or just display here?
                           session_state.generated_target_seq = target_seq_np.tolist()
                 else:
                      st.error("Failed to generate Phase 3 sample.")

    # Display the input image (before normalization)
    if display_image_pil:
         st.image(
             display_image_pil,
             caption='Input Image (Before Preprocessing)',
             use_column_width=True
         )

    # Return the state needed by the prediction section
    return {
        'processed_image': processed_image_np, # Normalized HWC NumPy array
        'display_image': display_image_pil     # PIL Image for reference
        # Add generated_target_seq if needed for comparison display
    }
