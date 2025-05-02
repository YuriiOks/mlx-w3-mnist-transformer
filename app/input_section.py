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
import time  # For potential delay/spinner
from typing import Dict, Any

# --- Add project root to sys.path for imports ---
import sys
from pathlib import Path
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    print(
        f"🎨 [Input Section] Adding project root to sys.path: {project_root}"
    )
    sys.path.insert(0, str(project_root))

# --- Imports from Project ---
try:
    from utils import logger
    from src.mnist_transformer_mlx.dataset_mlx import (
        numpy_normalize, DEFAULT_DATA_DIR
    )
    from utils.tokenizer_utils import (
        sequence_to_labels, PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID
    )
    from app.model_loader import preprocess_image
    try:
        from torchvision import datasets as tv_datasets
        from src.mnist_transformer_mlx.dataset_mlx import (
            generate_dynamic_digit_image_seq_np
        )
        TORCHVISION_AVAILABLE = True
        GENERATOR_AVAILABLE = True
    except ImportError:
        TORCHVISION_AVAILABLE = False
        GENERATOR_AVAILABLE = False
    TOKENIZER_AVAILABLE = True
except ImportError as e:
    st.error(f"Input Section Error: Failed imports - {e}")
    DEFAULT_DATA_DIR = "./data"
    numpy_normalize = lambda x: x
    sequence_to_labels = lambda x: ["Error"]
    PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID = 0, 1, 2
    TOKENIZER_AVAILABLE = False
    GENERATOR_AVAILABLE = False
    TORCHVISION_AVAILABLE = False
    def preprocess_image(*args): return None


def input_controls(
    config: dict,
    session_state: Dict[str, Any]
) -> dict:
    """
    Handles input image drawing, upload, and generation logic based on
    the loaded phase. Returns processed and display images.

    Args:
        config (dict): The application configuration dictionary.
        session_state (Dict[str, Any]): Streamlit's session state.
            Must contain 'phase_loaded' and may contain 'framework'.

    Returns:
        dict: Dictionary containing input state:
            {
                'processed_image': np.ndarray or None,
                'display_image': PIL.Image.Image or None
            }
            - processed_image: normalized HWC numpy array for model input.
            - display_image: PIL image before normalization for display.
    """
    st.header("1. Prepare Your Ingredient (Image) 🖼️")

    phase_loaded = session_state.phase_loaded
    framework = session_state.get('framework', 'MLX')

    input_state = {
        'processed_image': None,
        'display_image': None
    }

    dataset_cfg = config.get('dataset', {})
    processed_image_np = None
    display_image_pil = None

    # --- Determine Input Size Based on Phase ---
    default_img_size = dataset_cfg.get('image_size', 28)
    if phase_loaded == 1:
        input_size = dataset_cfg.get('image_size', 28)
    elif phase_loaded == 2:
        input_size = dataset_cfg.get('image_size_phase2', 56)
    elif phase_loaded == 3:
        input_size = dataset_cfg.get('image_size_phase3', 64)
    else:
        input_size = default_img_size

    # --- Input Method Selection ---
    input_options = ["Draw Digit", "Upload Image"]
    if TORCHVISION_AVAILABLE and GENERATOR_AVAILABLE and TOKENIZER_AVAILABLE:
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
        stroke_width = st.slider(
            "Stroke width: ",
            10,
            35,
            20,
            key=f"stroke_{phase_loaded}"
        )
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=stroke_width,
            stroke_color="white",
            background_color="black",
            height=canvas_size,
            width=canvas_size,
            drawing_mode="freedraw",
            key=f"canvas_{phase_loaded}",
        )
        if (
            canvas_result.image_data is not None
            and canvas_result.image_data.sum() > 0
        ):
            img_drawn_rgba = canvas_result.image_data.astype(np.uint8)
            display_image_pil = Image.fromarray(img_drawn_rgba).convert("L")
            processed_image_np = preprocess_image(
                display_image_pil,
                input_size,
                framework
            )

    elif input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            f"Upload {input_size}x{input_size} image...",
            type=["png", "jpg", "jpeg"],
            key=f"upload_{phase_loaded}"
        )
        if uploaded_file is not None:
            display_image_pil = Image.open(uploaded_file).convert("L")
            processed_image_np = preprocess_image(
                display_image_pil,
                input_size,
                framework
            )

    elif input_method == "Generate Sample":
        st.write(f"Generate a random sample for Phase {phase_loaded} ({input_size}x{input_size}):")
        if st.button("Generate", key=f"generate_{phase_loaded}"):
            data_dir = config['paths'].get('data_dir', DEFAULT_DATA_DIR)
            base_pil, base_lbl = None, None
            try:
                ds = tv_datasets.MNIST(
                    root=data_dir,
                    train=False,
                    download=True,
                    transform=None
                )
                base_pil = [img for img, _ in ds]
                base_lbl = np.array([lbl for _, lbl in ds], dtype=np.uint32)
            except Exception as e:
                st.error(f"Failed MNIST load for generator: {e}")

            if base_pil and base_lbl is not None:
                cfg_ds = config.get('dataset', {})
                cfg_tk = config.get('tokenizer', {})
                # Use the generator for all phases, but adjust parameters
                canvas_np, target_seq_np = generate_dynamic_digit_image_seq_np(
                    base_pil,
                    base_lbl,
                    canvas_size=input_size,
                    max_digits=cfg_ds.get('max_digits_phase3', 7) if phase_loaded == 3 else 1,
                    augment_digits=True,
                    max_seq_len=cfg_ds.get('max_seq_len', 10),
                    start_token_id=cfg_tk.get('start_token_id', START_TOKEN_ID),
                    end_token_id=cfg_tk.get('end_token_id', END_TOKEN_ID),
                    pad_token_id=cfg_tk.get('pad_token_id', PAD_TOKEN_ID),
                )
                if canvas_np is not None:
                    display_image_pil = Image.fromarray(
                        canvas_np.squeeze().astype(np.uint8)
                    )
                    processed_image_np = numpy_normalize(canvas_np)
                    if TOKENIZER_AVAILABLE and target_seq_np is not None:
                        decoded_labels = sequence_to_labels(
                            target_seq_np.tolist()
                        )
                        st.info(
                            f"Generated Sample Ground Truth: `{decoded_labels}`"
                        )
                        session_state.generated_target_seq = (
                            target_seq_np.tolist()
                        )
                else:
                    st.error("Failed to generate sample.")

    # Display the input image (before normalization)
    if display_image_pil:
        st.image(
            display_image_pil,
            caption='Input Image (Before Preprocessing)',
            use_container_width=True
        )

    input_state['processed_image'] = processed_image_np
    input_state['display_image'] = display_image_pil
    return input_state
