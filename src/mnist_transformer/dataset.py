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
import random
import numpy as np
from typing import Tuple, List

import matplotlib.pyplot as plt

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

# --- Phase 2: 2x2 Grid Data Generation ---

def generate_2x2_grid_image(
    mnist_dataset: Dataset, # Use the loaded MNIST dataset
    output_size: int = 56   # Target size (2 * 28)
) -> Tuple[torch.Tensor | None, List[int] | None]:
    """
    Generates a single 2x2 grid image by randomly sampling 4 MNIST digits.

    Args:
        mnist_dataset (Dataset): The loaded torchvision MNIST dataset.
        output_size (int): The height and width of the output grid image.

    Returns:
        Tuple[torch.Tensor | None, List[int] | None]:
            - A tensor representing the 56x56 grid image (1, 56, 56).
            - A list containing the 4 labels [top_left, top_right, bottom_left, bottom_right].
            Returns (None, None) if dataset is too small or errors occur.
    """
    if len(mnist_dataset) < 4:
        logger.error("❌ MNIST dataset too small to sample 4 images.")
        return None, None

    # Sample 4 random indices
    indices = random.sample(range(len(mnist_dataset)), 4)

    # Get the images and labels
    try:
        images = [mnist_dataset[i][0] for i in indices] # List of 4 tensors [1, 28, 28]
        labels = [mnist_dataset[i][1] for i in indices] # List of 4 integer labels
    except Exception as e:
        logger.error(f"❌ Error sampling images/labels from dataset: {e}", exc_info=True)
        return None, None

    # Ensure images are correct shape (C, H, W)
    if not all(img.shape == (1, 28, 28) for img in images):
        logger.error("❌ Sampled images have incorrect shape.")
        return None, None

    # Create the 56x56 grid (assuming output_size is 56)
    grid_image = torch.zeros((1, output_size, output_size), dtype=images[0].dtype)

    # Tile the images (top_left, top_right, bottom_left, bottom_right)
    grid_image[:, 0:28, 0:28] = images[0]    # Top-left
    grid_image[:, 0:28, 28:56] = images[1]    # Top-right
    grid_image[:, 28:56, 0:28] = images[2]    # Bottom-left
    grid_image[:, 28:56, 28:56] = images[3]    # Bottom-right

    return grid_image, labels


class MNISTGridDataset(Dataset):
    """
    PyTorch Dataset that generates 2x2 MNIST grid images on the fly.

    Args:
        mnist_dataset (Dataset): The underlying standard MNIST dataset (train or test).
        length (int): The desired length of this synthetic dataset.
        grid_size (int): The size of the output grid image (e.g., 56 for 2x2).
    """
    def __init__(self, mnist_dataset: Dataset, length: int, grid_size: int = 56):
        self.mnist_dataset = mnist_dataset
        self.length = length # How many synthetic images to generate
        self.grid_size = grid_size
        if len(mnist_dataset) < 4:
            raise ValueError("Base MNIST dataset is too small!")
        logger.info(f"🧠 MNISTGridDataset initialized. Will generate {length} synthetic {grid_size}x{grid_size} images.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Generate a new grid image every time an item is requested
        grid_image, labels = generate_2x2_grid_image(self.mnist_dataset, self.grid_size)
        if grid_image is None:
             # Handle error: maybe try again or raise error? For simplicity, let's try again once.
             logger.warning("Retrying grid image generation...")
             grid_image, labels = generate_2x2_grid_image(self.mnist_dataset, self.grid_size)
             if grid_image is None:
                  # If it fails again, return dummy data or raise an error
                  logger.error("Failed to generate grid image after retry!")
                  # Return dummy data matching expected types
                  return torch.zeros((1, self.grid_size, self.grid_size)), torch.tensor([-1, -1, -1, -1], dtype=torch.long)


        # Labels need to be a tensor for the DataLoader
        labels_tensor = torch.tensor(labels, dtype=torch.long) # Shape [4]
        return grid_image, labels_tensor


# --- Test Block ---
if __name__ == "__main__":

    # --- imshow HELPER FUNCTION HERE ---
    def imshow(img_tensor, title=''):
        """Helper function to display a single image tensor."""
        # Reverse the normalization for display purposes
        mean = torch.tensor(MNIST_MEAN) # Use defined constants
        std = torch.tensor(MNIST_STD)
        # Ensure mean/std are correct shape for broadcasting (C, 1, 1)
        img_tensor = img_tensor * std[:, None, None] + mean[:, None, None]

        # Clamp values to [0, 1] range
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # Convert to numpy for plotting
        npimg = img_tensor.numpy()

        # Plot
        plt.imshow(np.squeeze(npimg), cmap='gray') # Use np.squeeze for grayscale
        plt.title(title)
        plt.axis('off') # Hide axes for cleaner look

    # --- End imshow Definition ---
    logger.info("🧪 Running dataset.py script directly for testing...")

    # --- Test Phase 1 Loading ---
    logger.info("\n--- Testing Phase 1 (Standard MNIST) ---")
    train_data_p1 = get_mnist_dataset(train=True, use_augmentation=False) # No aug for this test
    if train_data_p1:
        img, label = train_data_p1[0]
        logger.info(f"Phase 1 Sample - Image shape: {img.shape}, Label: {label}")
        loader_p1 = get_mnist_dataloader(train_data_p1, batch_size=4)
        try:
            img_batch, label_batch = next(iter(loader_p1))
            logger.info(f"Phase 1 Batch - Images shape: {img_batch.shape}, Labels shape: {label_batch.shape}")
            logger.info("✅ Phase 1 loading seems OK.")
        except Exception as e:
            logger.error(f"❌ Phase 1 DataLoader iteration failed: {e}", exc_info=True)
    else:
        logger.error("❌ Failed to load standard MNIST for Phase 1 test.")


    # --- Test Phase 2 Data Generation ---
    logger.info("\n--- Testing Phase 2 (2x2 Grid Generation) ---")
    if train_data_p1: # Reuse loaded train_data_p1
        # Create a small instance of the grid dataset for testing
        grid_dataset_test = MNISTGridDataset(mnist_dataset=train_data_p1, length=100) # Generate 100 samples
        logger.info(f"Created MNISTGridDataset with length {len(grid_dataset_test)}")

        # Get a sample grid image
        grid_img, grid_labels = grid_dataset_test[0]
        logger.info(f"Phase 2 Sample - Image shape: {grid_img.shape}, Labels: {grid_labels.tolist()}, Labels shape: {grid_labels.shape}")
        assert grid_img.shape == (1, 56, 56), "Phase 2 image shape mismatch!"
        assert grid_labels.shape == (4,), "Phase 2 labels shape mismatch!"

        # Visualize a sample grid image
        logger.info("🎨 Visualizing sample grid image (will close automatically)...")
        plt.figure(figsize=(4, 4))
        imshow(grid_img, title=f"Labels: {grid_labels.tolist()}")
        plt.show()

        # plt.savefig("temp_grid_sample.png")
        # logger.info("💾 Saved sample grid plot to temp_grid_sample.png")
        # plt.close() # Close the plot window immediately


        # Test DataLoader for Phase 2 dataset
        grid_loader_test = get_mnist_dataloader(grid_dataset_test, batch_size=4, shuffle=False)
        logger.info("Iterating through one batch of Phase 2 DataLoader...")
        try:
            grid_img_batch, grid_label_batch = next(iter(grid_loader_test))
            logger.info(f"Phase 2 Batch - Images shape: {grid_img_batch.shape}, Labels shape: {grid_label_batch.shape}")
            assert grid_img_batch.shape == (4, 1, 56, 56), "Phase 2 batch image shape mismatch!"
            assert grid_label_batch.shape == (4, 4), "Phase 2 batch labels shape mismatch!"
            logger.info("✅ Phase 2 DataLoader test successful.")
        except Exception as e:
            logger.error(f"❌ Phase 2 DataLoader iteration failed: {e}", exc_info=True)
    else:
        logger.warning("🤷 Skipping Phase 2 tests because Phase 1 dataset failed to load.")

    logger.info("\n✅ dataset.py test execution finished.")