# MNIST Digit Classifier (Transformer) - MLX Version
# File: src/mnist_transformer_mlx/dataset_mlx.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: MNIST dataset loading and preprocessing for MLX (All Phases).
# Created: 2025-04-28
# Updated: 2025-04-28 # <-- Update date

import mlx.core as mx
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import random

# --- Add project root to sys.path for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent
if str(project_root) not in sys.path:
    print(f"🏗️ [dataset_mlx.py] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

try:
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    print("⚠️ torchvision not found. Cannot download/load MNIST automatically.")
    TORCHVISION_AVAILABLE = False

from utils import logger, load_config # Import config loader

# --- Constants ---
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
DEFAULT_DATA_DIR = project_root / "data"
EMPTY_CLASS_LABEL = 10 # Label for empty grid cells in Phase 3

# --- Phase 1: Base MNIST Data Loading ---

def numpy_normalize(np_array: np.ndarray) -> np.ndarray:
    """Normalizes a NumPy array using MNIST stats."""
    normalized = np_array / 255.0
    normalized = (normalized - MNIST_MEAN) / MNIST_STD
    return normalized

def get_mnist_data_arrays(
    train: bool = True,
    data_dir: str | Path = DEFAULT_DATA_DIR
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Loads MNIST using torchvision and returns raw NumPy arrays (N, H, W, C).
    Normalization will be applied later after potential transformations.
    """
    if not TORCHVISION_AVAILABLE:
        return None
    split_name = "Train" if train else "Test"
    logger.debug(f"Loading raw MNIST {split_name} for base dataset...")
    try:
        mnist_dataset = datasets.MNIST(
            root=data_dir, train=train, download=True, transform=None
        )
        images_np = [
            np.array(img, dtype=np.float32).reshape(28, 28, 1)
            for img, _ in mnist_dataset
        ]
        labels_np = [np.array(label, dtype=np.uint32) for _, label in mnist_dataset]
        images_np = np.stack(images_np)
        labels_np = np.stack(labels_np)
        logger.debug(
            f"Loaded raw NumPy {split_name} data. Images: {images_np.shape}, "
            f"Labels: {labels_np.shape}"
        )
        return images_np, labels_np
    except Exception as e:
        logger.error(
            f"❌ Failed loading/converting raw MNIST {split_name}: {e}",
            exc_info=True
        )
        return None

# --- Phase 2: 2x2 Grid Data Generation ---

def generate_2x2_grid_image_np(
    base_images: np.ndarray, # Shape (N, 28, 28, 1)
    base_labels: np.ndarray, # Shape (N,)
    output_size: int = 56
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Generates a single 2x2 grid image (NumPy array)."""
    num_base_images = base_images.shape[0]
    if num_base_images < 4:
        return None
    indices = random.sample(range(num_base_images), 4)
    try:
        images = [base_images[i] for i in indices]
        labels = [base_labels[i] for i in indices]
        grid_image = np.zeros((output_size, output_size, 1), dtype=np.float32)
        grid_image[0:28, 0:28, :] = images[0]
        grid_image[0:28, 28:56, :] = images[1]
        grid_image[28:56, 0:28, :] = images[2]
        grid_image[28:56, 28:56, :] = images[3]
        labels_np = np.array(labels, dtype=np.uint32)
        return grid_image, labels_np
    except Exception as e:
        logger.error(
            f"❌ Error generating 2x2 grid: {e}", exc_info=True
        )
        return None

class MNISTGridDatasetMLX:
    """Generates 2x2 MNIST grid images (MLX arrays) on the fly."""
    def __init__(
        self,
        base_images_mlx: mx.array,
        base_labels_mlx: mx.array,
        length: int,
        grid_size: int = 56
    ):
        self.base_images_mlx = base_images_mlx
        self.base_labels_mlx = base_labels_mlx
        self.length = length
        self.grid_size = grid_size
        self.num_base = base_images_mlx.shape[0]
        if self.num_base < 4:
            raise ValueError("Base dataset too small!")
        logger.info(
            f"🧠 MNISTGridDatasetMLX initialized. Generating {length} synthetic "
            f"{grid_size}x{grid_size} images."
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
            np_indices = np.random.choice(self.num_base, 4, replace=False)
            mlx_indices = mx.array(np_indices)

            images = self.base_images_mlx[mlx_indices] # MLX array, shape (4, 28, 28, 1)
            labels = self.base_labels_mlx[mlx_indices] # MLX array, shape (4,)

            # --- Grid Creation using NumPy ---
            # Convert sampled MLX arrays BACK to NumPy for tiling
            # logger.info(f"DEBUG: images dtype: {images.dtype}") 
            images_np = [np.array(img) for img in images] # <--- Converts each (28, 28, 1) to NumPy

            grid_image_np = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.float32) # NumPy is fine here
            grid_image_np[0:28, 0:28, :] = images_np[0]
            grid_image_np[0:28, 28:56, :] = images_np[1]
            grid_image_np[28:56, 0:28, :] = images_np[2]
            grid_image_np[28:56, 28:56, :] = images_np[3]

            # Normalize NumPy array and convert final grid BACK to MLX
            grid_image_mlx = mx.array(numpy_normalize(grid_image_np))

            return grid_image_mlx, labels # labels is already an MLX array

# --- Phase 3: Dynamic Layout Data Generation ---

def generate_dynamic_digit_image_np(
    base_images: np.ndarray, # Shape (N, 28, 28, 1)
    base_labels: np.ndarray, # Shape (N,)
    canvas_size: int = 64,
    patch_size: int = 8,
    max_digits: int = 5,
    empty_label: int = EMPTY_CLASS_LABEL
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates image with 0-max_digits placed randomly and target grid label.
    """
    num_base_images = base_images.shape[0]
    canvas = np.zeros((canvas_size, canvas_size, 1), dtype=np.float32)
    grid_w = grid_h = canvas_size // patch_size
    num_cells = grid_w * grid_h
    target_grid = np.full(
        (grid_h, grid_w), fill_value=empty_label, dtype=np.uint32
    )
    num_digits = random.randint(0, max_digits)
    if num_digits == 0:
        return canvas, target_grid.flatten()
    indices = random.sample(range(num_base_images), num_digits)
    placed_boxes = []
    for i in indices:
        digit_img = base_images[i]
        digit_label = base_labels[i]
        img_h, img_w, _ = digit_img.shape
        placed = False
        for _ in range(10):
            max_y = canvas_size - img_h
            max_x = canvas_size - img_w
            if max_y < 0 or max_x < 0:
                continue
            start_y = random.randint(0, max_y)
            start_x = random.randint(0, max_x)
            end_y = start_y + img_h
            end_x = start_x + img_w
            overlap = False
            for box in placed_boxes:
                if not (
                    end_x <= box[0] or start_x >= box[2] or
                    end_y <= box[1] or start_y >= box[3]
                ):
                    overlap = True
                    break
            if not overlap:
                canvas[start_y:end_y, start_x:end_x, :] = digit_img
                placed_boxes.append((start_x, start_y, end_x, end_y))
                center_y = start_y + img_h // 2
                center_x = start_x + img_w // 2
                cell_y = center_y // patch_size
                cell_x = center_x // patch_size
                if 0 <= cell_y < grid_h and 0 <= cell_x < grid_w:
                    target_grid[cell_y, cell_x] = digit_label
                placed = True
                break
    return canvas, target_grid.flatten()

class MNISTDynamicDatasetMLX:
    """Generates dynamic MNIST images (MLX arrays) on the fly."""
    def __init__(
        self,
        base_images_np: np.ndarray,
        base_labels_np: np.ndarray,
        length: int,
        config: dict
    ):
        self.base_images_np = base_images_np
        self.base_labels_np = base_labels_np
        self.length = length
        ds_cfg = config.get('dataset', {})
        self.canvas_size = ds_cfg.get('image_size_phase3', 64)
        self.patch_size = ds_cfg.get('patch_size_phase3', 8)
        self.max_digits = ds_cfg.get('max_digits_phase3', 5)
        self.empty_label = ds_cfg.get('num_classes', 10)
        if self.canvas_size % self.patch_size != 0:
            raise ValueError(
                "Phase 3 canvas size must be divisible by patch size"
            )
        logger.info(
            f"🧠 MNISTDynamicDatasetMLX initialized. Generating {length} "
            f"synthetic {self.canvas_size}x{self.canvas_size} images."
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        canvas_np, target_np = generate_dynamic_digit_image_np(
            self.base_images_np, self.base_labels_np,
            self.canvas_size, self.patch_size,
            self.max_digits, self.empty_label
        )
        if canvas_np is None:
            canvas_np = np.zeros(
                (self.canvas_size, self.canvas_size, 1), dtype=np.float32
            )
            num_cells = (self.canvas_size // self.patch_size) ** 2
            target_np = np.full(
                (num_cells,), fill_value=self.empty_label, dtype=np.uint32
            )
        canvas_mlx = mx.array(numpy_normalize(canvas_np))
        target_mlx = mx.array(target_np)
        return canvas_mlx, target_mlx

# --- Test Block ---
if __name__ == "__main__":
    logger.info("🧪 Running dataset_mlx.py script directly for testing...")
    config = load_config() or {}
    base_train_images_np, base_train_labels_np = get_mnist_data_arrays(
        train=True
    )
    base_test_images_np, base_test_labels_np = get_mnist_data_arrays(
        train=False
    )
    if base_train_images_np is None or base_test_images_np is None:
        logger.error(
            "❌ Failed to load base NumPy data. Cannot proceed with tests."
        )
        sys.exit(1)
    logger.info("\n--- Testing Phase 1 (MLX Conversion & Norm) ---")
    try:
        p1_images = mx.array(numpy_normalize(base_train_images_np))
        p1_labels = mx.array(base_train_labels_np)
        logger.info(
            f"Phase 1 MLX Shapes - Images: {p1_images.shape}, "
            f"Labels: {p1_labels.shape}"
        )
        assert p1_images.shape[0] == 60000 and p1_labels.shape[0] == 60000
        logger.info("✅ Phase 1 MLX conversion seems OK.")
    except Exception as e:
        logger.error(
            f"❌ Error during Phase 1 MLX conversion test: {e}", exc_info=True
        )
    logger.info("\n--- Testing Phase 2 (2x2 Grid Generation) ---")
    try:
        p2_dataset = MNISTGridDatasetMLX(
            mx.array(base_train_images_np),
            mx.array(base_train_labels_np),
            length=100
        )
        logger.info(f"Created Phase 2 Dataset, length: {len(p2_dataset)}")
        p2_img, p2_labels = p2_dataset[0]
        logger.info(
            f"Phase 2 Sample - Image: {p2_img.shape}, Labels: {p2_labels.shape}"
        )
        assert p2_img.shape[-1] == 1 and p2_labels.shape == (4,)
        logger.info("✅ Phase 2 Dataset seems OK.")
    except Exception as e:
        logger.error(
            f"❌ Error during Phase 2 Dataset test: {e}", exc_info=True
        )
    logger.info("\n--- Testing Phase 3 (Dynamic Layout Generation) ---")
    try:
        p3_dataset = MNISTDynamicDatasetMLX(
            base_train_images_np, base_train_labels_np, length=100, config=config
        )
        logger.info(f"Created Phase 3 Dataset, length: {len(p3_dataset)}")
        p3_img, p3_labels = p3_dataset[0]
        expected_cells = (
            config['dataset']['image_size_phase3'] //
            config['dataset']['patch_size_phase3']
        ) ** 2
        logger.info(
            f"Phase 3 Sample - Image: {p3_img.shape}, Labels: {p3_labels.shape} "
            f"(Expected cells: {expected_cells})"
        )
        assert p3_img.shape[-1] == 1 and p3_labels.shape[0] == expected_cells
        logger.info("✅ Phase 3 Dataset seems OK.")
    except Exception as e:
        logger.error(
            f"❌ Error during Phase 3 Dataset test: {e}", exc_info=True
        )
    logger.info("\n✅ dataset_mlx.py test execution finished.")