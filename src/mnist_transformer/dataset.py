# MNIST Digit Classifier (Transformer)
# File: src/mnist_transformer/dataset.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: MNIST dataset loading and preprocessing.
# Created: 2025-04-28
# Updated: 2025-04-28

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import os
import sys
from pathlib import Path
import random
import numpy as np
from typing import Tuple, List

import matplotlib.pyplot as plt

# --- Add project root to sys.path for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent
if str(project_root) not in sys.path:
    print(f"🏗️ [dataset.py] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

from utils import logger  # Assuming logger is setup in utils

# --- Constants ---
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
DEFAULT_DATA_DIR = project_root / "data"

# --- Transformations ---
def get_mnist_transforms(augment: bool = False):
    """
    Returns the standard MNIST transforms (ToTensor, Normalize).
    Optionally includes basic augmentation.
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ]
    if augment:
        transform_list.insert(
            0,
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
        )
        logger.info("Appling basic augmentation (RandomAffine) to transforms.")
    return transforms.Compose(transform_list)

# --- Dataset Loading Function ---
def get_mnist_dataset(
    train: bool = True,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    use_augmentation: bool = False
) -> Dataset | None:
    """
    Loads the MNIST dataset using torchvision.
    """
    split_name = "Train" if train else "Test"
    transform = get_mnist_transforms(
        augment=use_augmentation if train else False
    )
    logger.info(f"💾 Loading MNIST {split_name} dataset...")
    logger.info(f"   Data directory: {data_dir}")
    aug_status = use_augmentation if train else 'Disabled (Test Set)'
    logger.info(f"   Augmentation: {aug_status}")
    try:
        dataset = datasets.MNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )
        logger.info(
            f"✅ MNIST {split_name} dataset loaded successfully "
            f"({len(dataset)} samples)."
        )
        return dataset
    except Exception as e:
        logger.error(
            f"❌ Failed to load MNIST {split_name} dataset: {e}",
            exc_info=True
        )
        return None

# --- (Optional) DataLoader Function ---
def get_mnist_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Creates a DataLoader for the given MNIST dataset.
    """
    if num_workers == 0:
        pin_memory = False
    else:
        pin_memory = torch.cuda.is_available()
    logger.info(
        f"📦 Creating DataLoader: batch_size={batch_size}, "
        f"shuffle={shuffle}, num_workers={num_workers}, "
        f"pin_memory={pin_memory}"
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

# --- Phase 2: 2x2 Grid Data Generation ---
def generate_2x2_grid_image(
    mnist_dataset: Dataset,
    output_size: int = 56
) -> Tuple[torch.Tensor | None, List[int] | None]:
    """
    Generates a single 2x2 grid image by randomly sampling 4 MNIST digits.
    """
    if len(mnist_dataset) < 4:
        logger.error("❌ MNIST dataset too small to sample 4 images.")
        return None, None
    indices = random.sample(range(len(mnist_dataset)), 4)
    try:
        images = [mnist_dataset[i][0] for i in indices]
        labels = [mnist_dataset[i][1] for i in indices]
    except Exception as e:
        logger.error(
            f"❌ Error sampling images/labels from dataset: {e}",
            exc_info=True
        )
        return None, None
    if not all(img.shape == (1, 28, 28) for img in images):
        logger.error("❌ Sampled images have incorrect shape.")
        return None, None
    grid_image = torch.zeros(
        (1, output_size, output_size), dtype=images[0].dtype
    )
    grid_image[:, 0:28, 0:28] = images[0]
    grid_image[:, 0:28, 28:56] = images[1]
    grid_image[:, 28:56, 0:28] = images[2]
    grid_image[:, 28:56, 28:56] = images[3]
    return grid_image, labels

class MNISTGridDataset(Dataset):
    """
    PyTorch Dataset that generates 2x2 MNIST grid images on the fly.
    """
    def __init__(
        self,
        mnist_dataset: Dataset,
        length: int,
        grid_size: int = 56
    ):
        self.mnist_dataset = mnist_dataset
        self.length = length
        self.grid_size = grid_size
        if len(mnist_dataset) < 4:
            raise ValueError("Base MNIST dataset is too small!")
        logger.info(
            f"🧠 MNISTGridDataset initialized. Will generate {length} "
            f"synthetic {grid_size}x{grid_size} images."
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        grid_image, labels = generate_2x2_grid_image(
            self.mnist_dataset, self.grid_size
        )
        if grid_image is None:
            logger.warning("Retrying grid image generation...")
            grid_image, labels = generate_2x2_grid_image(
                self.mnist_dataset, self.grid_size
            )
            if grid_image is None:
                logger.error("Failed to generate grid image after retry!")
                return (
                    torch.zeros((1, self.grid_size, self.grid_size)),
                    torch.tensor([-1, -1, -1, -1], dtype=torch.long)
                )
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return grid_image, labels_tensor

# --- Test Block ---
if __name__ == "__main__":

    def imshow(img_tensor, title=''):
        """Helper function to display a single image tensor."""
        mean = torch.tensor(MNIST_MEAN)
        std = torch.tensor(MNIST_STD)
        img_tensor = img_tensor * std[:, None, None] + mean[:, None, None]
        img_tensor = torch.clamp(img_tensor, 0, 1)
        npimg = img_tensor.numpy()
        plt.imshow(np.squeeze(npimg), cmap='gray')
        plt.title(title)
        plt.axis('off')

    logger.info("🧪 Running dataset.py script directly for testing...")

    logger.info("\n--- Testing Phase 1 (Standard MNIST) ---")
    train_data_p1 = get_mnist_dataset(train=True, use_augmentation=False)
    if train_data_p1:
        img, label = train_data_p1[0]
        logger.info(
            f"Phase 1 Sample - Image shape: {img.shape}, Label: {label}"
        )
        loader_p1 = get_mnist_dataloader(train_data_p1, batch_size=4)
        try:
            img_batch, label_batch = next(iter(loader_p1))
            logger.info(
                f"Phase 1 Batch - Images shape: {img_batch.shape}, "
                f"Labels shape: {label_batch.shape}"
            )
            logger.info("✅ Phase 1 loading seems OK.")
        except Exception as e:
            logger.error(
                f"❌ Phase 1 DataLoader iteration failed: {e}",
                exc_info=True
            )
    else:
        logger.error("❌ Failed to load standard MNIST for Phase 1 test.")

    logger.info("\n--- Testing Phase 2 (2x2 Grid Generation) ---")
    if train_data_p1:
        grid_dataset_test = MNISTGridDataset(
            mnist_dataset=train_data_p1, length=100
        )
        logger.info(
            f"Created MNISTGridDataset with length {len(grid_dataset_test)}"
        )
        grid_img, grid_labels = grid_dataset_test[0]
        logger.info(
            f"Phase 2 Sample - Image shape: {grid_img.shape}, "
            f"Labels: {grid_labels.tolist()}, Labels shape: {grid_labels.shape}"
        )
        assert grid_img.shape == (1, 56, 56), "Phase 2 image shape mismatch!"
        assert grid_labels.shape == (4,), "Phase 2 labels shape mismatch!"
        logger.info("🎨 Visualizing sample grid image (will close automatically)...")
        plt.figure(figsize=(4, 4))
        imshow(grid_img, title=f"Labels: {grid_labels.tolist()}")
        plt.show()
        grid_loader_test = get_mnist_dataloader(
            grid_dataset_test, batch_size=4, shuffle=False
        )
        logger.info("Iterating through one batch of Phase 2 DataLoader...")
        try:
            grid_img_batch, grid_label_batch = next(iter(grid_loader_test))
            logger.info(
                f"Phase 2 Batch - Images shape: {grid_img_batch.shape}, "
                f"Labels shape: {grid_label_batch.shape}"
            )
            assert grid_img_batch.shape == (4, 1, 56, 56), \
                "Phase 2 batch image shape mismatch!"
            assert grid_label_batch.shape == (4, 4), \
                "Phase 2 batch labels shape mismatch!"
            logger.info("✅ Phase 2 DataLoader test successful.")
        except Exception as e:
            logger.error(
                f"❌ Phase 2 DataLoader iteration failed: {e}",
                exc_info=True
            )
    else:
        logger.warning(
            "🤷 Skipping Phase 2 tests because Phase 1 dataset failed to load."
        )

    logger.info("\n✅ dataset.py test execution finished.")
