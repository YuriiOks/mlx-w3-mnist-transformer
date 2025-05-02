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
        f"🎨 [Input Section] Adding project root to sys.path: "
        f"{project_root}"
    )
    sys.path.insert(0, str(project_root))

# --- Imports from Project ---
try:
    from utils import logger
    from src.mnist_transformer_mlx.dataset_mlx import (
        numpy_normalize,
        DEFAULT_DATA_DIR
    )
    from utils.tokenizer_utils import (
        sequence_to_labels,
        PAD_TOKEN_ID,
        START_TOKEN_ID,
        END_TOKEN_ID
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
    def preprocess_image(*args):
        return None


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
    # If a real PyTorch model is loaded, force framework to 'PyTorch'
    if hasattr(session_state.get('model', None), 'forward'):
        framework = 'PyTorch'

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
    if (
        TORCHVISION_AVAILABLE and
        GENERATOR_AVAILABLE and
        TOKENIZER_AVAILABLE
    ):
        input_options.append("Generate Sample")

    input_method = st.radio(
        f"Input method for Phase {phase_loaded} "
        f"(Target Size: {input_size}x{input_size}):",
        input_options,
        key=f"input_{phase_loaded}_{framework}",
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
        # Move Predict button right under the canvas
        predict_clicked = st.button("Predict", key=f"predict_button_{phase_loaded}")
        if (
            canvas_result.image_data is not None and
            canvas_result.image_data.sum() > 0
        ):
            img_drawn_rgba = canvas_result.image_data.astype(np.uint8)
            display_image_pil = Image.fromarray(img_drawn_rgba).convert("L")
            processed_image_np = preprocess_image(
                display_image_pil,
                input_size,
                framework
            )
    else:
        predict_clicked = st.button("Predict", key=f"predict_button_{phase_loaded}")

    if input_method == "Upload Image":
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
        st.caption("The 'Generated Sample Ground Truth' below is the correct label for the generated image. This is what the model should predict.")
        generate_clicked = st.button("Generate", key=f"generate_{phase_loaded}")
        if generate_clicked or (
            'generated_sample' in session_state and session_state.get('generated_sample_phase') == phase_loaded
        ):
            if generate_clicked:
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
                    canvas_np, target_seq_np = generate_dynamic_digit_image_seq_np(
                        base_pil, base_lbl, input_size,
                        cfg_ds.get('max_digits_phase3', 7) if phase_loaded == 3 else 1,
                        True,
                        cfg_ds.get('max_seq_len', 10),
                        cfg_tk.get('start_token_id', START_TOKEN_ID),
                        cfg_tk.get('end_token_id', END_TOKEN_ID),
                        cfg_tk.get('pad_token_id', PAD_TOKEN_ID),
                    )
                    if canvas_np is not None:
                        session_state['generated_sample'] = canvas_np
                        session_state['generated_sample_label'] = target_seq_np
                        session_state['generated_sample_phase'] = phase_loaded
                    else:
                        st.error("Failed to generate sample.")
                        session_state['generated_sample'] = None
                        session_state['generated_sample_label'] = None
                        session_state['generated_sample_phase'] = None
            # Use the last generated sample if it exists and is for this phase
            canvas_np = session_state.get('generated_sample', None)
            target_seq_np = session_state.get('generated_sample_label', None)
            if canvas_np is not None:
                display_image_pil = Image.fromarray(
                    canvas_np.squeeze().astype(np.uint8)
                )
                processed_image_np = numpy_normalize(canvas_np)
                if target_seq_np is not None:
                    if phase_loaded == 1:
                        label_val = int(target_seq_np) if np.isscalar(target_seq_np) or getattr(target_seq_np, 'size', 0) == 1 else int(target_seq_np[0])
                        st.info(f"Generated Sample Ground Truth: `{label_val}`")
                        session_state.generated_target_seq = label_val
                    elif TOKENIZER_AVAILABLE:
                        decoded_labels = sequence_to_labels(target_seq_np.tolist())
                        st.info(f"Generated Sample Ground Truth: `{decoded_labels}`")
                        session_state.generated_target_seq = target_seq_np.tolist()

    # Display the input image (before normalization)
    if display_image_pil:
        st.image(
            display_image_pil,
            caption='Input Image (Before Preprocessing)',
            use_container_width=True
        )
        # Add download button for the drawn/uploaded image
        import io
        buf = io.BytesIO()
        display_image_pil.save(buf, format='PNG')
        st.download_button(
            label="Download Input Image as PNG",
            data=buf.getvalue(),
            file_name="input_image.png",
            mime="image/png"
        )
        # Debug: Show pixel values and invert option
        with st.expander("🔬 Inspect Raw Pixel Values & Invert Image"):
            arr = np.array(display_image_pil)
            st.write(f"Shape: {arr.shape}, dtype: {arr.dtype}")
            st.write(
                f"Min: {arr.min()}, Max: {arr.max()}, "
                f"Mean: {arr.mean():.2f}"
            )
            st.write(arr)
            if st.checkbox("Show Inverted Image", key="invert_img"):
                inv_img = Image.fromarray(255 - arr)
                st.image(
                    inv_img,
                    caption="Inverted Image",
                    use_container_width=True
                )

    input_state['processed_image'] = processed_image_np
    input_state['display_image'] = display_image_pil
    input_state['predict_clicked'] = predict_clicked
    return input_state
