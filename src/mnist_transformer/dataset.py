# MNIST Digit Classifier (Transformer)
# File: src/mnist_transformer/dataset.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: MNIST dataset loading and preprocessing.
# Created: 2025-04-28
# Updated: 2025-04-28

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset  # For splitting later
import os
import sys
from pathlib import Path

# --- Add project root to sys.path for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels (mnist_transformer -> src -> root)
project_root = Path(script_dir).parent.parent
if str(project_root) not in sys.path:
    print(f"🏗️ [dataset.py] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

from utils import logger  # Assuming logger is setup in utils

# --- Constants ---
# Mean and Std Dev for MNIST normalization
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
DEFAULT_DATA_DIR = project_root / "data"


# --- Transformations ---
def get_mnist_transforms(augment: bool = False):
    """
    Returns the standard MNIST transforms (ToTensor, Normalize).
    Optionally includes basic augmentation.

    Args:
        augment (bool): Whether to include basic augmentation (RandomAffine).

    Returns:
        transforms.Compose: Composed torchvision transforms.
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ]
    # Basic augmentation (optional, can be expanded later)
    if augment:
        # Example: Slight rotation and translation
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

    Args:
        train (bool): If True, load the training dataset, otherwise test dataset.
        data_dir (str | Path): The directory to download/load the data from.
        use_augmentation (bool): If True and train=True, apply augmentation.

    Returns:
        Dataset | None: The loaded PyTorch Dataset object, or None if fails.
    """
    split_name = "Train" if train else "Test"
    # Augment only train set
    transform = get_mnist_transforms(augment=use_augmentation if train else False)

    logger.info(f"💾 Loading MNIST {split_name} dataset...")
    logger.info(f"   Data directory: {data_dir}")
    aug_status = use_augmentation if train else 'Disabled (Test Set)'
    logger.info(f"   Augmentation: {aug_status}")

    try:
        dataset = datasets.MNIST(
            root=data_dir,
            train=train,
            download=True,  # Download if not present
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
    num_workers: int = 0  # Default to 0 for simplicity, increase later
) -> DataLoader:
    """
    Creates a DataLoader for the given MNIST dataset.
    """
    if num_workers == 0:
        # Pin memory only works well with num_workers > 0 and CUDA
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
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive
    )


# --- Test Block ---
if __name__ == "__main__":
    logger.info("🧪 Running dataset.py script directly for testing...")

    # Test loading train dataset
    train_data = get_mnist_dataset(train=True, use_augmentation=True)

    if train_data:
        logger.info(f"Train dataset type: {type(train_data)}")
        # Get a sample
        img, label = train_data[0]
        logger.info(
            f"Sample - Image shape: {img.shape}, Label: {label}, "
            f"Image dtype: {img.dtype}, Min: {img.min():.2f}, "
            f"Max: {img.max():.2f}"
        )

        # Test dataloader
        train_loader_test = get_mnist_dataloader(
            train_data, 
            batch_size=4, 
            shuffle=True, 
            num_workers=0
        )
        logger.info("Iterating through one batch of DataLoader...")
        try:
            img_batch, label_batch = next(iter(train_loader_test))
            logger.info(
                f"Batch - Images shape: {img_batch.shape}, "
                f"Labels shape: {label_batch.shape}"
            )
            logger.info(f"✅ DataLoader test successful.")
        except Exception as e:
            logger.error(f"❌ DataLoader iteration failed: {e}", exc_info=True)

    # Test loading test dataset
    test_data = get_mnist_dataset(train=False)
    if test_data:
        img, label = test_data[0]
        logger.info(f"Test Sample - Image shape: {img.shape}, Label: {label}")

    logger.info("✅ dataset.py test execution finished.")