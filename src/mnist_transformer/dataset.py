# MNIST Digit Classifier (Transformer) - PyTorch Version
# File: src/mnist_transformer/dataset.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: MNIST dataset loading, preprocessing, and synthetic data
# generation (All Phases).
# Created: 2025-04-28
# Updated: 2025-04-30

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import os
import sys
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, List, Optional, Dict

# --- Add project root to sys.path for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent
if str(project_root) not in sys.path:
    print(f"🏗️ [dataset.py] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

# --- Imports from Project ---
from utils import logger, load_config
try:
    from utils.tokenizer_utils import (
        labels_to_sequence,
        sequence_to_labels,
        PAD_TOKEN_ID,
        START_TOKEN_ID,
        END_TOKEN_ID
    )
    TOKENIZER_AVAILABLE = True
except ImportError:
    logger.error(
        "❌ Could not import tokenizer_utils. Phase 3 sequence generation "
        "will fail."
    )
    TOKENIZER_AVAILABLE = False
    PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID = 0, 1, 2
    def labels_to_sequence(*args, **kwargs): return []
# --- End Project Imports ---

# --- Constants ---
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
DEFAULT_DATA_DIR = project_root / "data"
EMPTY_CLASS_LABEL = 10

# --- Transformations ---
digit_augmentation_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=20,
        translate=(0.15, 0.15),
        scale=(0.8, 1.2),
        shear=15,
        fill=0
    ),
    transforms.RandomApply([
        transforms.ElasticTransform(alpha=35.0, sigma=4.5)
    ], p=0.5),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
    ], p=0.4)
])

def get_mnist_transforms(
    image_size: int = 28,
    augment: bool = False
) -> transforms.Compose:
    """
    Returns a composed torchvision transform for MNIST images.

    Args:
        image_size (int): Desired output image size (height and width).
        augment (bool): If True, applies basic augmentation for training.

    Returns:
        transforms.Compose: Composed transform for MNIST images.
    """
    transform_list = []
    if augment and image_size == 28:
        transform_list.append(
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1)
            )
        )
        logger.debug("Applying basic augmentation for standard MNIST.")
    if image_size != 28:
        transform_list.append(
            transforms.Resize((image_size, image_size))
        )
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))
    return transforms.Compose(transform_list)

def get_mnist_dataset(
    train: bool = True,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    transform: Optional[transforms.Compose] = None
) -> Optional[Dataset]:
    """
    Loads the standard MNIST dataset using torchvision.

    Args:
        train (bool): If True, loads training set. If False, loads test set.
        data_dir (str | Path): Directory to save/load the dataset.
        transform (Optional[transforms.Compose]): Transformations to apply.

    Returns:
        Optional[Dataset]: Loaded MNIST dataset or None if loading fails.
    """
    split_name = "Train" if train else "Test"
    if transform is None:
        transform = get_mnist_transforms(
            image_size=28,
            augment=train
        )
    logger.info(f"💾 Loading standard MNIST {split_name} dataset...")
    logger.info(f"   Data directory: {data_dir}")
    try:
        dataset = datasets.MNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
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

def generate_2x2_grid_image_pt(
    mnist_dataset: Dataset,
    output_size: int = 56
) -> Optional[Tuple[torch.Tensor, List[int]]]:
    """
    Generates a 2x2 grid image tensor from the MNIST dataset.

    Returns a tensor of shape (1, output_size, output_size) and a list of
    labels.

    Args:
        mnist_dataset (Dataset): Base MNIST dataset.
        output_size (int): Size of the output grid image.

    Returns:
        Optional[Tuple[torch.Tensor, List[int]]]: Grid image tensor and labels,
        or (None, None) if failed.
    """
    if len(mnist_dataset) < 4:
        return None
    indices = random.sample(range(len(mnist_dataset)), 4)
    try:
        images = [mnist_dataset[i][0] for i in indices]
        labels = [mnist_dataset[i][1] for i in indices]
        if not all(
            isinstance(img, torch.Tensor) and img.shape == (1, 28, 28)
            for img in images
        ):
            logger.error(
                "❌ Sampled Phase 2 base images have incorrect shape/type."
            )
            return None, None
        grid_image = torch.zeros(
            (1, output_size, output_size),
            dtype=images[0].dtype
        )
        grid_image[:, 0:28, 0:28] = images[0]
        grid_image[:, 0:28, 28:56] = images[1]
        grid_image[:, 28:56, 0:28] = images[2]
        grid_image[:, 28:56, 28:56] = images[3]
        return grid_image, labels
    except Exception as e:
        logger.error(
            f"❌ Error generating 2x2 grid tensor: {e}",
            exc_info=True
        )
        return None, None

class MNISTGridDataset(Dataset):
    """
    PyTorch Dataset generating 2x2 grid MNIST images and labels.

    Args:
        base_mnist_dataset (Dataset): The base MNIST dataset.
        length (int): Number of synthetic samples to generate.
        grid_size (int): Size of the output grid image (default: 56).
    """
    def __init__(
        self,
        base_mnist_dataset: Dataset,
        length: int,
        grid_size: int = 56
    ):
        """
        Initialize MNISTGridDataset.

        Args:
            base_mnist_dataset (Dataset): The base MNIST dataset.
            length (int): Number of synthetic samples to generate.
            grid_size (int): Size of the output grid image.
        """
        self.base_dataset = base_mnist_dataset
        self.length = length
        self.grid_size = grid_size
        if len(base_mnist_dataset) < 4:
            raise ValueError("Base MNIST dataset too small!")
        logger.info(
            f"🧠 MNISTGridDataset initialized. Generating {length} synthetic "
            f"{grid_size}x{grid_size} images."
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.length

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a 2x2 grid image and returns it along with the labels.

        If grid generation fails, returns a zero tensor and empty labels.

        Args:
            idx (int): Index (ignored, as samples are generated randomly).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Grid image and labels tensor.
        """
        grid_image, labels = generate_2x2_grid_image_pt(
            self.base_dataset,
            self.grid_size
        )
        if grid_image is None:
            logger.warning("Retrying grid image generation in __getitem__...")
            grid_image, labels = generate_2x2_grid_image_pt(
                self.base_dataset,
                self.grid_size
            )
            if grid_image is None:
                logger.error("Failed grid generation after retry!")
                return (
                    torch.zeros((1, self.grid_size, self.grid_size)),
                    torch.tensor([-1]*4, dtype=torch.long)
                )
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return grid_image, labels_tensor

def generate_dynamic_digit_image_pt(
    base_mnist_pil_dataset: Dataset,
    canvas_size: int = 64,
    max_digits: int = 5,
    augment_digits: bool = True
) -> Optional[Tuple[torch.Tensor, List[int]]]:
    """
    Generates an image tensor with 0 to max_digits placed randomly.

    Applies augmentation to individual digits before placement.
    Returns canvas tensor (unnormalized, 0-1 range) and ordered labels.

    Args:
        base_mnist_pil_dataset (Dataset): Base MNIST dataset (PIL).
        canvas_size (int): Size of the output canvas.
        max_digits (int): Maximum number of digits to place.
        augment_digits (bool): Whether to apply augmentation.

    Returns:
        Tuple[torch.Tensor, List[int]]: Canvas tensor and ordered labels.
    """
    num_base_images = len(base_mnist_pil_dataset)
    canvas_np = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    placed_boxes = []
    placed_positions = []
    num_digits = random.randint(0, max_digits)
    if num_digits == 0:
        return torch.from_numpy(canvas_np).unsqueeze(0), []
    indices = random.sample(range(num_base_images), num_digits)
    for i in indices:
        digit_pil, digit_label = base_mnist_pil_dataset[i]
        if augment_digits:
            try:
                digit_pil_aug = digit_augmentation_transform(digit_pil)
            except Exception as e:
                logger.warning(
                    f"Augmentation failed for sample {i}, using original. "
                    f"Error: {e}"
                )
                digit_pil_aug = digit_pil
        else:
            digit_pil_aug = digit_pil
        digit_np_uint8 = np.array(digit_pil_aug, dtype=np.uint8)
        digit_np = digit_np_uint8.astype(np.float32) / 255.0
        img_h, img_w = digit_np.shape
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
    """
    PyTorch Dataset generating dynamic MNIST images and target sequences.

    Args:
        base_mnist_pil_dataset (Dataset): Base MNIST dataset (PIL).
        length (int): Number of synthetic samples to generate.
        config (Dict): Configuration dictionary.
        use_augmentation (bool): Whether to use digit augmentation.
    """
    def __init__(
        self,
        base_mnist_pil_dataset: Dataset,
        length: int,
        config: Dict,
        use_augmentation: bool = True
    ):
        """
        Initialize MNISTDynamicDataset.

        Args:
            base_mnist_pil_dataset (Dataset): Base MNIST dataset (PIL).
            length (int): Number of synthetic samples to generate.
            config (Dict): Configuration dictionary.
            use_augmentation (bool): Whether to use digit augmentation.
        """
        self.base_dataset_pil = base_mnist_pil_dataset
        self.length = length
        self.use_augmentation = use_augmentation
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
        logger.info(
            f"🧠 MNISTDynamicDataset initialized. Augmentation: "
            f"{self.use_augmentation}. Generating {length} synthetic "
            f"{self.canvas_size}x{self.canvas_size} images."
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.length

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a dynamic image and returns it along with the labels.

        If generation fails, returns a zero tensor and empty labels.

        Args:
            idx (int): Index (ignored, as samples are generated randomly).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image and target sequence tensor.
        """
        canvas_tensor, ordered_labels = generate_dynamic_digit_image_pt(
            self.base_dataset_pil,
            self.canvas_size,
            self.max_digits,
            self.use_augmentation
        )
        if canvas_tensor is None:
            logger.error("Failed dynamic image generation in getitem!")
            canvas_tensor = torch.zeros((1, self.canvas_size, self.canvas_size))
            ordered_labels = []
        if not TOKENIZER_AVAILABLE:
            logger.error(
                "Tokenizer utils not available, cannot create target sequence!"
            )
            target_sequence = []
        else:
            target_sequence = labels_to_sequence(
                ordered_labels,
                self.max_seq_len,
                self.start_token_id,
                self.end_token_id,
                self.pad_token_id
            )
        target_sequence_tensor = torch.tensor(target_sequence, dtype=torch.long)
        final_image_tensor = self.final_transform(canvas_tensor)
        return final_image_tensor, target_sequence_tensor

def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker threads for loading data.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    pin_memory = torch.cuda.is_available() and num_workers > 0
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
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False
    )

if __name__ == "__main__":
    logger.info("🧪 Running dataset.py script directly for testing All Phases...")
    config = load_config(Path(project_root) / "config.yaml") or {}
    import matplotlib.pyplot as plt
    def imshow(
        img_tensor,
        title: str = ''
    ):
        """
        Utility to display a tensor image using matplotlib.

        Args:
            img_tensor (torch.Tensor): Image tensor.
            title (str): Title for the plot.
        """
        mean = torch.tensor(MNIST_MEAN)
        std = torch.tensor(MNIST_STD)
        img_tensor = img_tensor.cpu()
        if img_tensor.min() < -0.5:
            img_tensor = img_tensor * std[:, None, None] + mean[:, None, None]
        img_tensor = torch.clamp(img_tensor, 0, 1)
        npimg = img_tensor.numpy()
        plt.imshow(np.squeeze(npimg), cmap='gray')
        plt.title(title)
        plt.axis('off')
    base_train_pil = None
    if 'datasets' in globals():
        try:
            base_train_pil = datasets.MNIST(
                root=DEFAULT_DATA_DIR,
                train=True,
                download=True,
                transform=None
            )
        except Exception as e:
            logger.error(f"Failed to load base PIL MNIST: {e}")
    logger.info("\n--- Testing Phase 1 (Standard MNIST) ---")
    p1_transform = get_mnist_transforms(
        image_size=28,
        augment=False
    )
    p1_train_data = get_mnist_dataset(
        train=True,
        transform=p1_transform
    )
    if p1_train_data:
        logger.info("✅ Phase 1 Dataset Loaded.")
    else:
        logger.error("❌ Failed Phase 1 Load")
    logger.info("\n--- Testing Phase 2 (2x2 Grid Generation) ---")
    if p1_train_data:
        try:
            p2_dataset = MNISTGridDataset(
                base_mnist_dataset=p1_train_data,
                length=10
            )
            p2_img, p2_labels = p2_dataset[0]
            logger.info(
                f"Phase 2 Sample - Img: {p2_img.shape}, Labels: {p2_labels.shape}"
            )
            assert p2_img.shape == (1, 56, 56) and p2_labels.shape == (4,)
            logger.info("✅ Phase 2 Dataset seems OK.")
        except Exception as e:
            logger.error(f"❌ Phase 2 Test Error: {e}", exc_info=True)
    else:
        logger.warning("Skipping P2 test")
    logger.info("\n--- Testing Phase 3 (Dynamic Layout Generation) ---")
    if base_train_pil:
        if not TOKENIZER_AVAILABLE:
            logger.error("❌ Cannot test Phase 3: tokenizer_utils failed.")
        else:
            logger.info("🧪 Testing Phase 3 WITHOUT augmentation...")
            try:
                p3_dataset_noaug = MNISTDynamicDataset(
                    base_mnist_pil_dataset=base_train_pil,
                    length=4,
                    config=config,
                    use_augmentation=False
                )
                img_noaug, seq_noaug = p3_dataset_noaug[0]
                logger.info(
                    f"P3 NoAug Sample - Img: {img_noaug.shape}, "
                    f"Seq: {seq_noaug.shape} {seq_noaug.tolist()}"
                )
            except Exception as e:
                logger.error(
                    f"❌ Error getting P3 NoAug sample: {e}", exc_info=True
                )
            logger.info("🧪 Testing Phase 3 WITH augmentation...")
            try:
                p3_dataset_aug = MNISTDynamicDataset(
                    base_mnist_pil_dataset=base_train_pil,
                    length=4,
                    config=config,
                    use_augmentation=True
                )
                img_aug, seq_aug = p3_dataset_aug[0]
                logger.info(
                    f"P3 Aug Sample - Img: {img_aug.shape}, "
                    f"Seq: {seq_aug.shape} {seq_aug.tolist()}"
                )
            except Exception as e:
                logger.error(
                    f"❌ Error getting P3 Aug sample: {e}", exc_info=True
                )
            logger.info("🎨 Visualizing Phase 3 samples (No Aug vs Aug):")
            try:
                plt.figure(figsize=(8, 5))
                plt.suptitle("Phase 3 Sample Generation", y=1.0)
                plt.subplot(1, 2, 1)
                labels_noaug_decoded = sequence_to_labels(seq_noaug.tolist())
                title_noaug = (
                    f"No Aug - Labels: {labels_noaug_decoded}\n"
                    f"Seq: {seq_noaug.tolist()}"
                )
                imshow(img_noaug, title=title_noaug)
                plt.subplot(1, 2, 2)
                labels_aug_decoded = sequence_to_labels(seq_aug.tolist())
                title_aug = (
                    f"With Aug - Labels: {labels_aug_decoded}\n"
                    f"Seq: {seq_aug.tolist()}"
                )
                imshow(img_aug, title=title_aug)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()
                logger.info(
                    "Displaying Phase 3 sample plot. Close the plot window "
                    "to continue..."
                )
                logger.info("✅ Phase 3 Dataset testing seems OK (check plot).")
            except NameError:
                logger.error("imshow not defined for plotting.")
            except Exception as viz_e:
                logger.error(f"Viz Error: {viz_e}", exc_info=True)
    else:
        logger.warning("Skipping P3 test - base PIL dataset failed to load.")
    logger.info("\n✅ dataset.py test execution finished.")
