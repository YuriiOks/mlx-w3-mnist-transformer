# MNIST Digit Classifier (Transformer) - PyTorch Version
# File: src/mnist_transformer/dataset.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: MNIST dataset loading, preprocessing, and synthetic data
# generation (All Phases).
# Created: 2025-04-28
# Updated: 2025-04-29

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import os
import sys
from pathlib import Path
import random
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt

# --- Add project root to sys.path for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent
if str(project_root) not in sys.path:
    print(f"🏗️ [dataset.py] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

# --- Imports from Project ---
try:
    from utils import logger, load_config
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.warning("Could not import project logger/config loader from utils.")
    def load_config(*args, **kwargs): return {}

try:
    from utils.tokenizer_utils import (
        labels_to_sequence,
        PAD_TOKEN_ID,
        START_TOKEN_ID,
        END_TOKEN_ID,
    )
    TOKENIZER_AVAILABLE = True
except ImportError:
    logger.error("❌ Could not import tokenizer_utils from utils/. "
                 "Phase 3 sequence generation will fail.")
    TOKENIZER_AVAILABLE = False
    PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID = 0, 1, 2
    def labels_to_sequence(*args, **kwargs): return []
# --- End Project Imports ---

# --- Constants ---
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
DEFAULT_DATA_DIR = project_root / "data"

# --- Transformations ---
def get_mnist_transforms(image_size: int = 28, augment: bool = False):
    """
    Returns MNIST transforms (Resize, ToTensor, Normalize).
    Optionally includes augmentation.
    """
    transform_list = []
    if augment:
        transform_list.append(
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
        )
        logger.debug("Applying basic augmentation (RandomAffine) to transforms.")
    if image_size != 28:
        transform_list.append(
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        )
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))
    return transforms.Compose(transform_list)

# --- Phase 1: Standard MNIST Dataset Loading ---
def get_mnist_dataset(
    train: bool = True,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    transform: Optional[transforms.Compose] = None,
    use_augmentation: bool = False
) -> Optional[Dataset]:
    """ Loads the standard MNIST dataset using torchvision. """
    split_name = "Train" if train else "Test"
    if transform is None:
        transform = get_mnist_transforms(
            image_size=28, augment=(train and use_augmentation)
        )
    logger.info(f"💾 Loading standard MNIST {split_name} dataset...")
    logger.info(f"   Data directory: {data_dir}")
    logger.info(f"   Using provided transforms: {'Yes' if transform else 'No'}")
    try:
        dataset = datasets.MNIST(
            root=data_dir, train=train, download=True, transform=transform
        )
        logger.info(
            f"✅ Standard MNIST {split_name} dataset loaded "
            f"({len(dataset)} samples)."
        )
        return dataset
    except Exception as e:
        logger.error(
            f"❌ Failed loading standard MNIST {split_name}: {e}",
            exc_info=True
        )
        return None

# --- Phase 2: 2x2 Grid Data Generation ---
def generate_2x2_grid_image_pt(
    base_mnist_dataset: Dataset,
    output_size: int = 56
) -> Optional[Tuple[torch.Tensor, List[int]]]:
    """ Generates a single 2x2 grid image (PyTorch Tensor) and labels. """
    if len(base_mnist_dataset) < 4:
        return None
    indices = random.sample(range(len(base_mnist_dataset)), 4)
    try:
        images = [base_mnist_dataset[i][0] for i in indices]
        labels = [base_mnist_dataset[i][1] for i in indices]
        if not all(
            isinstance(img, torch.Tensor) and img.shape == (1, 28, 28)
            for img in images
        ):
            logger.error("❌ Sampled Phase 2 base images have incorrect shape/type.")
            return None, None
        grid_image = torch.zeros(
            (1, output_size, output_size), dtype=images[0].dtype
        )
        grid_image[:, 0:28, 0:28] = images[0]
        grid_image[:, 0:28, 28:56] = images[1]
        grid_image[:, 28:56, 0:28] = images[2]
        grid_image[:, 28:56, 28:56] = images[3]
        return grid_image, labels
    except Exception as e:
        logger.error(
            f"❌ Error generating 2x2 grid tensor: {e}", exc_info=True
        )
        return None, None

class MNISTGridDataset(Dataset):
    """ PyTorch Dataset generating 2x2 MNIST grid images on the fly. """
    def __init__(
        self, base_mnist_dataset: Dataset, length: int, grid_size: int = 56
    ):
        self.base_dataset = base_mnist_dataset
        self.length = length
        self.grid_size = grid_size
        if len(base_mnist_dataset) < 4:
            raise ValueError("Base MNIST dataset too small!")
        logger.info(
            f"🧠 MNISTGridDataset initialized. Generating {length} synthetic "
            f"{grid_size}x{grid_size} images."
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        grid_image, labels = generate_2x2_grid_image_pt(
            self.base_dataset, self.grid_size
        )
        if grid_image is None:
            logger.warning("Retrying grid image generation in __getitem__...")
            grid_image, labels = generate_2x2_grid_image_pt(
                self.base_dataset, self.grid_size
            )
            if grid_image is None:
                logger.error("Failed grid generation after retry!")
                return (
                    torch.zeros((1, self.grid_size, self.grid_size)),
                    torch.tensor([-1]*4, dtype=torch.long)
                )
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return grid_image, labels_tensor

# --- Phase 3: Dynamic Layout Data Generation ---
def generate_dynamic_digit_image_pt(
    base_mnist_tensor_dataset: Dataset,
    canvas_size: int = 64,
    max_digits: int = 5,
) -> Optional[Tuple[torch.Tensor, List[int]]]:
    """
    Generates image tensor with 0-max_digits placed randomly.
    Returns canvas tensor (unnormalized 0-1 range) and list of ordered digit
    labels (0-9).
    """
    num_base_images = len(base_mnist_tensor_dataset)
    canvas_np = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    placed_boxes = []
    placed_positions = []
    num_digits = random.randint(0, max_digits)
    if num_digits == 0:
        return torch.from_numpy(canvas_np).unsqueeze(0), []
    indices = (
        random.sample(range(num_base_images), num_digits)
        if num_digits > 0 else []
    )
    for i in indices:
        try:
            digit_tensor, digit_label = base_mnist_tensor_dataset[i]
            digit_np = digit_tensor.squeeze().numpy()
            img_h, img_w = digit_np.shape
        except Exception as e:
            logger.warning(
                f"Could not retrieve sample {i} from base tensor dataset: {e}"
            )
            continue
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
                if not (
                    end_x <= box[0] or start_x >= box[2] or
                    end_y <= box[1] or start_y >= box[3]
                ):
                    overlap = True
                    break
            if not overlap:
                canvas_np[start_y:end_y, start_x:end_x] = np.maximum(
                    canvas_np[start_y:end_y, start_x:end_x], digit_np
                )
                placed_boxes.append((start_x, start_y, end_x, end_y))
                placed_positions.append(
                    {"y": start_y, "x": start_x, "label": digit_label}
                )
                placed = True
                break
    placed_positions.sort(key=lambda p: (p["y"], p["x"]))
    ordered_labels = [p["label"] for p in placed_positions]
    canvas_tensor = torch.from_numpy(canvas_np).unsqueeze(0)
    return canvas_tensor, ordered_labels

class MNISTDynamicDataset(Dataset):
    """ PyTorch Dataset generating dynamic MNIST images and target sequences. """
    def __init__(
        self, base_mnist_pil_dataset: Dataset, length: int, config: Dict
    ):
        self.base_dataset_pil = base_mnist_pil_dataset
        self.length = length
        cfg = config.get('dataset', {})
        tokenizer_cfg = config.get('tokenizer', {})
        self.canvas_size = cfg.get('image_size_phase3', 64)
        self.max_digits = cfg.get('max_digits_phase3', 5)
        self.max_seq_len = cfg.get('max_seq_len', 10)
        self.pad_token_id = tokenizer_cfg.get('pad_token_id', PAD_TOKEN_ID)
        self.start_token_id = tokenizer_cfg.get(
            'start_token_id', START_TOKEN_ID
        )
        self.end_token_id = tokenizer_cfg.get('end_token_id', END_TOKEN_ID)
        self.final_transform = transforms.Compose([
            transforms.Normalize(MNIST_MEAN, MNIST_STD)
        ])
        self.base_dataset_tensor = datasets.MNIST(
            root=DEFAULT_DATA_DIR,
            train=base_mnist_pil_dataset.train,
            download=False,
            transform=transforms.ToTensor()
        )
        if not TOKENIZER_AVAILABLE:
            raise ImportError(
                "Tokenizer utilities are required for MNISTDynamicDataset."
            )
        logger.info(
            f"🧠 MNISTDynamicDataset initialized. Generating {length} synthetic "
            f"{self.canvas_size}x{self.canvas_size} images."
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        canvas_tensor, ordered_labels = generate_dynamic_digit_image_pt(
            self.base_dataset_tensor, self.canvas_size, self.max_digits
        )
        if canvas_tensor is None:
            logger.error("Failed dynamic image generation in getitem!")
            canvas_tensor = torch.zeros((1, self.canvas_size, self.canvas_size))
            ordered_labels = []
        target_sequence = labels_to_sequence(
            ordered_labels, self.max_seq_len,
            self.start_token_id, self.end_token_id, self.pad_token_id
        )
        target_sequence_tensor = torch.tensor(target_sequence, dtype=torch.long)
        final_image_tensor = self.final_transform(canvas_tensor)
        return final_image_tensor, target_sequence_tensor

# --- DataLoader Function ---
def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """ Creates a DataLoader for the given dataset. """
    pin_memory = torch.cuda.is_available() and num_workers > 0
    logger.info(
        f"📦 Creating DataLoader: batch_size={batch_size}, "
        f"shuffle={shuffle}, num_workers={num_workers}, "
        f"pin_memory={pin_memory}"
    )
    persistent_workers = (num_workers > 0)
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

# --- Test Block ---
if __name__ == "__main__":
    logger.info("🧪 Running dataset.py script directly for testing All Phases...")
    config = load_config() or {}

    # --- 👇 Add imports needed for visualization IN test block ---
    import matplotlib.pyplot as plt
    import numpy as np
    # --- End Add Imports ---

    # --- 👇 Define imshow locally for testing ---
    def imshow(img_tensor, title=''):
        """ Helper function to display a single image tensor (expects C, H, W). """
        # Assume input tensor is 0-1 range if coming directly from generator
        # Or unnormalize if it came from dataset __getitem__
        if img_tensor.min() < 0: # Likely normalized, unnormalize
             mean = torch.tensor(MNIST_MEAN); std = torch.tensor(MNIST_STD)
             img_tensor = img_tensor.cpu() * std[:, None, None] + mean[:, None, None]
        img_tensor = torch.clamp(img_tensor, 0, 1)
        npimg = img_tensor.cpu().numpy()
        plt.imshow(np.squeeze(npimg), cmap='gray')
        plt.title(title); plt.axis('off')
    # --- End imshow definition ---


    # Load base PIL dataset needed for Phase 3 generator
    base_train_pil = datasets.MNIST(root=DEFAULT_DATA_DIR, train=True, download=True, transform=None)
    # Load base Tensor dataset needed for Phase 3 generator's internal use
    base_train_tensor = datasets.MNIST(root=DEFAULT_DATA_DIR, train=True, download=False, transform=transforms.ToTensor())

    if base_train_pil is None or base_train_tensor is None:
         logger.error("❌ Failed to load base MNIST datasets for testing.")
         sys.exit(1)


    # --- Test Phase 1 (Keep as is) ---
    logger.info("\n--- Testing Phase 1 (Standard MNIST) ---")
    # ... (Phase 1 test code remains the same) ...
    p1_transform = get_mnist_transforms(image_size=28, augment=False)
    p1_train_data = get_mnist_dataset(train=True, transform=p1_transform)
    if p1_train_data: logger.info("✅ Phase 1 Dataset Loaded.")


    # --- Test Phase 2 (Keep as is) ---
    logger.info("\n--- Testing Phase 2 (2x2 Grid Generation) ---")
    # ... (Phase 2 test code remains the same) ...
    if p1_train_data: # Need transformed base data for grid generation
        p2_dataset = MNISTGridDataset(base_mnist_dataset=p1_train_data, length=100)
        p2_img, p2_labels = p2_dataset[0]
        logger.info(f"Phase 2 Sample - Img: {p2_img.shape}, Labels: {p2_labels.shape}")
        assert p2_img.shape == (1, 56, 56) and p2_labels.shape == (4,), "P2 Shape mismatch"
        logger.info("✅ Phase 2 Dataset seems OK.")
    else: logger.warning("Skipping P2 test")


    # --- Test Phase 3 ---
    logger.info("\n--- Testing Phase 3 (Dynamic Layout Generation) ---")
    if base_train_pil and base_train_tensor:
        if not TOKENIZER_AVAILABLE:
             logger.error("❌ Cannot test Phase 3: tokenizer_utils failed to import.")
        else:
            cfg_dataset = config.get('dataset', {})
            cfg_tokenizer = config.get('tokenizer', {})
            p3_canvas_size = cfg_dataset.get('image_size_phase3', 64)
            p3_max_digits = cfg_dataset.get('max_digits_phase3', 5)
            p3_max_seq_len = cfg_dataset.get('max_seq_len', 10)
            p3_pad_token_id = cfg_tokenizer.get('pad_token_id', PAD_TOKEN_ID)
            p3_start_token_id = cfg_tokenizer.get('start_token_id', START_TOKEN_ID)
            p3_end_token_id = cfg_tokenizer.get('end_token_id', END_TOKEN_ID)

            # --- 👇 Generate and Visualize Multiple Samples ---
            num_samples_to_show = 8
            logger.info(f"🎨 Generating and Visualizing {num_samples_to_show} Phase 3 samples...")
            plt.figure(figsize=(12, 6)) # Adjust as needed
            plt.suptitle("Phase 3 Sample Generated Images & Target Sequences", y=1.02)

            for i in range(num_samples_to_show):
                 # Call generator directly to get unnormalized canvas and labels list
                 canvas_tensor, ordered_labels = generate_dynamic_digit_image_pt(
                      base_train_tensor, p3_canvas_size, p3_max_digits
                 )
                 # Convert labels to sequence
                 target_sequence = labels_to_sequence(
                     ordered_labels, p3_max_seq_len,
                     p3_start_token_id, p3_end_token_id, p3_pad_token_id
                 )

                 plt.subplot(2, num_samples_to_show // 2, i + 1)
                 title = f"Labels: {ordered_labels}\nSeq: {target_sequence}"
                 imshow(canvas_tensor, title=title) # Display unnormalized canvas

            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            # plt.savefig("temp_p3_samples_generated.png") # Save plot
            # logger.info("💾 Saved Phase 3 sample plot to temp_p3_samples_generated.png")
            # plt.close()
            plt.show() # Show plot
            logger.info("✅ Phase 3 samples generated and visualized.")
            # --- End Generate and Visualize ---

            # Optional: Test the Dataset class itself
            logger.info("Testing MNISTDynamicDataset class...")
            p3_dataset = MNISTDynamicDataset(base_mnist_pil_dataset=base_train_pil, length=10, config=config)
            p3_img, p3_seq = p3_dataset[0] # Get one sample from the dataset class
            logger.info(f"Sample from Dataset - Img: {p3_img.shape}, Seq: {p3_seq.shape}")
            assert p3_img.shape == (1, p3_canvas_size, p3_canvas_size), "P3 Dataset Img Shape mismatch"
            assert p3_seq.shape == (p3_max_seq_len,), "P3 Dataset Seq Shape mismatch"
            logger.info("✅ Phase 3 Dataset class seems OK.")

    else: logger.warning("Skipping P3 test")


    logger.info("\n✅ dataset.py test execution finished.")

    # Optional: Clean up the saved plot file
    if os.path.exists("temp_p3_samples_generated.png"):
        os.remove("temp_p3_samples_generated.png")