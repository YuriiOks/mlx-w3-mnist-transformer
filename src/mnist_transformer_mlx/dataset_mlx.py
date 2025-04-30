# MNIST Digit Classifier (Transformer) - MLX Version
# File: src/mnist_transformer_mlx/dataset_mlx.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: MNIST dataset loading and preprocessing for MLX (All Phases).
# Created: 2025-04-28
# Updated: 2025-04-30

import mlx.core as mx
import numpy as np
from PIL import Image, ImageFilter
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import random
import matplotlib.pyplot as plt  # For test block visualization

# --- Add project root to sys.path for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent  # Go up two levels
if str(project_root) not in sys.path:
    print(f"🏗️ [dataset_mlx.py] Adding project root: {project_root}")
    sys.path.insert(0, str(project_root))

# --- Project Imports ---
try:
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    print("⚠️ torchvision not found. Cannot load MNIST automatically.")
    TORCHVISION_AVAILABLE = False

from utils import logger, load_config
try:
    from utils.tokenizer_utils import (
        labels_to_sequence, sequence_to_labels, PAD_TOKEN_ID,
        START_TOKEN_ID, END_TOKEN_ID
    )
    TOKENIZER_AVAILABLE = True
except ImportError:
    logger.error("❌ Tokenizer utils not found. Phase 3 will fail.")
    TOKENIZER_AVAILABLE = False
    PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID = 0, 1, 2
    def labels_to_sequence(*args, **kwargs): return []
    def sequence_to_labels(*args, **kwargs): return []
# --- End Project Imports ---

# --- Constants ---
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
DEFAULT_DATA_DIR = project_root / "data"
EMPTY_CLASS_LABEL = 10  # For P3 grid classification approach (if used)

# --- Augmentation Pipeline (for Phase 3 individual digits) ---
digit_augmentation_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=20, translate=(0.15, 0.15), scale=(0.8, 1.2),
        shear=15, fill=0
    ),
    transforms.RandomApply(
        [transforms.ElasticTransform(alpha=35.0, sigma=4.5)], p=0.5
    ),
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.4
    )
])

# --- Normalization Helper ---
def numpy_normalize(np_array: np.ndarray) -> np.ndarray:
    """Normalizes NumPy array (H, W, C) using MNIST stats."""
    normalized = np_array.astype(np.float32) / 255.0
    normalized = (normalized - MNIST_MEAN) / MNIST_STD
    return normalized

# --- Phase 1: Base MNIST Data Loading (to NumPy) ---
def get_mnist_data_arrays(
    train: bool = True,
    data_dir: str | Path = DEFAULT_DATA_DIR
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Loads raw MNIST data as NumPy arrays (N, H, W, C)."""
    if not TORCHVISION_AVAILABLE:
        return None
    split = "Train" if train else "Test"
    logger.debug(f"Loading raw MNIST {split} for base dataset...")
    try:
        mnist_pil_dataset = datasets.MNIST(
            root=data_dir, train=train, download=True, transform=None
        )
        images_np = [
            np.array(img, dtype=np.uint8).reshape(28, 28, 1)
            for img, _ in mnist_pil_dataset
        ]
        labels_np = [
            np.array(label, dtype=np.uint32)
            for _, label in mnist_pil_dataset
        ]
        images_np = np.stack(images_np)
        labels_np = np.stack(labels_np)
        logger.debug(
            f"Loaded raw NumPy {split}. Imgs: {images_np.shape}, "
            f"Lbls: {labels_np.shape}"
        )
        return images_np, labels_np
    except Exception as e:
        logger.error(f"❌ Failed raw MNIST load {split}: {e}", exc_info=True)
        return None

# --- Phase 2: 2x2 Grid Data Generation (from NumPy) ---
def generate_2x2_grid_image_np(
    base_images_np: np.ndarray,
    base_labels_np: np.ndarray,
    output_size: int = 56
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Generates a single 2x2 grid image (uint8 NumPy array)."""
    num_base = base_images_np.shape[0]
    if num_base < 4:
        return None
    indices = random.sample(range(num_base), 4)
    try:
        images = [base_images_np[i] for i in indices]
        labels = [base_labels_np[i] for i in indices]
        grid_image = np.zeros((output_size, output_size, 1), dtype=np.uint8)
        grid_image[0:28, 0:28, :] = images[0]
        grid_image[0:28, 28:56, :] = images[1]
        grid_image[28:56, 0:28, :] = images[2]
        grid_image[28:56, 28:56, :] = images[3]
        return grid_image, np.array(labels, dtype=np.uint32)
    except Exception as e:
        logger.error(f"❌ Error generating 2x2 grid: {e}", exc_info=True)
        return None

class MNISTGridDatasetMLX:
    """Generates 2x2 MNIST grid images (MLX arrays) on the fly."""
    def __init__(
        self,
        base_images_np: np.ndarray,
        base_labels_np: np.ndarray,
        length: int,
        grid_size: int = 56
    ):
        self.base_images_np = base_images_np
        self.base_labels_np = base_labels_np
        self.length = length
        self.grid_size = grid_size
        if len(base_images_np) < 4:
            raise ValueError("Base dataset too small!")
        logger.info(
            f"🧠 MNISTGridDatasetMLX init: {length} synth "
            f"{grid_size}x{grid_size} images."
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        grid_image_np, labels_np = generate_2x2_grid_image_np(
            self.base_images_np, self.base_labels_np, self.grid_size
        )
        if grid_image_np is None:
            grid_image_np = np.zeros(
                (self.grid_size, self.grid_size, 1), dtype=np.uint8
            )
            labels_np = np.array([-1]*4, dtype=np.uint32)

        grid_image_mlx = mx.array(numpy_normalize(grid_image_np))
        labels_mlx = mx.array(labels_np)
        return grid_image_mlx, labels_mlx

# --- Phase 3: Dynamic Layout Data Generation (Sequence Output) ---
def generate_dynamic_digit_image_seq_np(
    base_images_pil: List[Image.Image],
    base_labels_np: np.ndarray,
    canvas_size: int = 64,
    max_digits: int = 5,
    augment_digits: bool = True,
    max_seq_len: int = 10,
    start_token_id: int = START_TOKEN_ID,
    end_token_id: int = END_TOKEN_ID,
    pad_token_id: int = PAD_TOKEN_ID
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates image (uint8 NumPy) with random digits & target sequence.
    """
    num_base_images = len(base_images_pil)
    canvas_np = np.zeros((canvas_size, canvas_size, 1), dtype=np.uint8)
    placed_boxes = []
    placed_positions = []
    num_digits = random.randint(0, max_digits)

    if num_digits > 0:
        indices = random.sample(range(num_base_images), num_digits)
        for i in indices:
            digit_pil = base_images_pil[i]
            digit_label = base_labels_np[i]
            if augment_digits:
                try:
                    digit_pil_aug = digit_augmentation_transform(digit_pil)
                except Exception as e:
                    logger.warning(f"Aug failed sample {i}: {e}")
                    digit_pil_aug = digit_pil
            else:
                digit_pil_aug = digit_pil
            digit_np_uint8 = np.array(digit_pil_aug, dtype=np.uint8)
            digit_np_uint8 = digit_np_uint8.reshape(28, 28, 1)
            img_h, img_w, _ = digit_np_uint8.shape

            placed = False
            for _ in range(20):
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
                    if not (end_x <= box[0] or start_x >= box[2] or
                            end_y <= box[1] or start_y >= box[3]):
                        overlap = True
                        break
                if not overlap:
                    canvas_np[start_y:end_y, start_x:end_x] = np.maximum(
                        canvas_np[start_y:end_y, start_x:end_x],
                        digit_np_uint8
                    )
                    placed_boxes.append((start_x, start_y, end_x, end_y))
                    placed_positions.append({
                        "y": start_y, "x": start_x, "label": int(digit_label)
                    })
                    placed = True
                    break

    placed_positions.sort(key=lambda p: (p["y"], p["x"]))
    ordered_labels = [p["label"] for p in placed_positions]

    if not TOKENIZER_AVAILABLE:
        return None, None
    target_sequence = labels_to_sequence(
        ordered_labels, max_seq_len, start_token_id, end_token_id, pad_token_id
    )
    target_sequence_np = np.array(target_sequence, dtype=np.uint32)

    return canvas_np, target_sequence_np

class MNISTDynamicDatasetMLX:
    """Generates dynamic MNIST images and target sequences (MLX arrays)."""
    def __init__(
        self,
        base_images_pil: List[Image.Image],
        base_labels_np: np.ndarray,
        length: int,
        config: dict,
        use_augmentation: bool = True
    ):
        self.base_images_pil = base_images_pil
        self.base_labels_np = base_labels_np
        self.length = length
        self.use_augmentation = use_augmentation
        cfg_ds = config.get('dataset', {})
        cfg_tk = config.get('tokenizer', {})
        self.canvas_size = cfg_ds.get('image_size_phase3', 64)
        self.patch_size = cfg_ds.get('patch_size_phase3', )
        self.max_digits = cfg_ds.get('max_digits_phase3', 5)
        self.max_seq_len = cfg_ds.get('max_seq_len', 10)
        self.pad_token_id = cfg_tk.get('pad_token_id', PAD_TOKEN_ID)
        self.start_token_id = cfg_tk.get('start_token_id', START_TOKEN_ID)
        self.end_token_id = cfg_tk.get('end_token_id', END_TOKEN_ID)
        logger.info(
            f"🧠 MNISTDynamicDatasetMLX init: Aug={self.use_augmentation}. "
            f"{length} synth {self.canvas_size}x{self.canvas_size} images."
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        canvas_np, target_sequence_np = generate_dynamic_digit_image_seq_np(
            self.base_images_pil, self.base_labels_np, self.canvas_size,
            self.max_digits, self.use_augmentation,
            self.max_seq_len, self.start_token_id,
            self.end_token_id, self.pad_token_id
        )
        if canvas_np is None:
            canvas_np = np.zeros(
                (self.canvas_size, self.canvas_size, 1), dtype=np.uint8
            )
            target_sequence_np = np.array(
                [self.start_token_id, self.end_token_id] +
                [self.pad_token_id] * (self.max_seq_len - 2), dtype=np.uint32
            )
        canvas_mlx = mx.array(numpy_normalize(canvas_np))
        target_mlx = mx.array(target_sequence_np)
        return canvas_mlx, target_mlx

# --- Test Block ---
if __name__ == "__main__":
    logger.info("🧪 Running dataset_mlx.py tests for All Phases...")
    config = load_config(project_root / "config.yaml") or {}

    def imshow(img_tensor, title=''):
        npimg = np.array(img_tensor)
        if npimg.min() < -0.5:
            npimg = (npimg * MNIST_STD) + MNIST_MEAN
        npimg = np.clip(npimg * 255.0, 0, 255).astype(np.uint8)
        plt.imshow(np.squeeze(npimg), cmap='gray')
        plt.title(title)
        plt.axis('off')

    base_train_images_np, base_train_labels_np = get_mnist_data_arrays(True)
    base_test_images_np, base_test_labels_np = get_mnist_data_arrays(False)
    base_train_pil = [
        Image.fromarray(np.squeeze(img)) for img in base_train_images_np
    ] if base_train_images_np is not None else None

    if base_train_images_np is None or base_train_pil is None:
        logger.error("❌ Base data loading failed. Exiting tests.")
        sys.exit(1)

    logger.info("\n--- Testing Phase 1 ---")
    try:
        p1_img = mx.array(numpy_normalize(base_train_images_np[0]))
        p1_lbl = mx.array(base_train_labels_np[0])
        logger.info(f"P1 Sample - Img: {p1_img.shape}, Lbl: {p1_lbl.shape}")
        logger.info("✅ P1 OK.")
    except Exception as e:
        logger.error(f"❌ P1 Error: {e}", exc_info=True)

    logger.info("\n--- Testing Phase 2 ---")
    try:
        p2_dataset = MNISTGridDatasetMLX(
            base_train_images_np, base_train_labels_np, length=1
        )
        p2_img, p2_labels = p2_dataset[0]
        logger.info(f"P2 Sample - Img: {p2_img.shape}, Lbl: {p2_labels.shape}")
        logger.info("✅ P2 OK.")
    except Exception as e:
        logger.error(f"❌ P2 Error: {e}", exc_info=True)

    logger.info("\n--- Testing Phase 3 ---")
    if not TOKENIZER_AVAILABLE:
        logger.error("❌ Cannot test P3: tokenizer utils missing.")
    else:
        try:
            p3_dataset = MNISTDynamicDatasetMLX(
                base_train_pil, base_train_labels_np, length=8,
                config=config, use_augmentation=True
            )
            logger.info(f"Created P3 Dataset, length: {len(p3_dataset)}")
            num_samples_to_show = 4
            plt.figure(figsize=(10, 5))
            plt.suptitle("Phase 3 Generated Samples (MLX Dataset)", y=1.0)
            for i in range(num_samples_to_show):
                p3_img, p3_seq = p3_dataset[i]
                labels_decoded = sequence_to_labels(p3_seq.tolist())
                plt.subplot(1, num_samples_to_show, i + 1)
                title = f"Labels: {labels_decoded}\nSeq: {p3_seq.tolist()}"
                imshow(p3_img, title=title)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            logger.info("Displaying P3 plot. Close window to continue.")
            logger.info("✅ P3 Dataset seems OK.")
        except Exception as e:
            logger.error(f"❌ P3 Error: {e}", exc_info=True)

    logger.info("\n✅ dataset_mlx.py test execution finished.")