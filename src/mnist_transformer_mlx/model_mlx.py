# MNIST Digit Classifier (Transformer) - MLX Version
# File: src/mnist_transformer_mlx/model_mlx.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Defines the Vision Transformer (ViT) model architecture using MLX.
# Created: 2025-04-28
# Updated: 2025-04-28

import mlx.core as mx
import mlx.nn as nn
import os
import sys
from pathlib import Path
from typing import List, Optional

# --- Add project root for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent  # Go up two levels
if str(project_root) not in sys.path:
    print(f"🧠 [model_mlx.py] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

# Import building blocks from modules_mlx.py
try:
    from src.mnist_transformer_mlx.modules_mlx import (
        PatchEmbeddingMLX,
        TransformerEncoderBlockMLX,
    )
    from utils import logger
except ImportError as e:
    print(f"❌ Error importing modules in model_mlx.py: {e}")
    # Define dummy classes if import fails
    PatchEmbeddingMLX = nn.Module
    TransformerEncoderBlockMLX = nn.Module
    import logging
    logger = logging.getLogger("model_mlx")

# --- Vision Transformer Model (MLX) ---

class VisionTransformerMLX(nn.Module):
    """
    Vision Transformer (ViT) model using MLX.

    Args:
        img_size (int): Size of the input image (square).
        patch_size (int): Size of each square patch.
        in_channels (int): Number of input image channels.
        num_classes (int): Number of classes per output head.
        embed_dim (int): Dimensionality of the token/patch embeddings.
        depth (int): Number of transformer encoder layers.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        attention_dropout (float): Dropout rate for attention weights.
        mlp_dropout (float): Dropout rate for MLP layers.
        num_outputs (int): Number of separate classification outputs (1 for P1,
            4 for P2). Default 1.
    """
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        attention_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        num_outputs: int = 1,  # For Phase 1 and Phase 2 adaptation
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_outputs
        self.embed_dim = embed_dim

        # 1. Patch Embedding (+ CLS Token, + Pos Embedding)
        self.patch_embed = PatchEmbeddingMLX(
            image_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        # 2. Transformer Encoder Blocks
        # Use a standard Python list to hold the layers
        self.encoder_blocks = [
            TransformerEncoderBlockMLX(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(depth)
        ]

        # 3. Final Layer Normalization (applied to CLS token)
        self.norm = nn.LayerNorm(embed_dim)

        # 4. MLP Classifier Head(s)
        # Output size depends on the phase/number of outputs
        self.head = nn.Linear(embed_dim, num_classes * num_outputs)

        # Note: MLX generally handles weight initialization internally,
        # but explicit initialization can be done if needed after instantiation.
        if logger:
            logger.info(
                f"🧠 VisionTransformerMLX initialized: depth={depth}, "
                f"heads={num_heads}, embed_dim={embed_dim}, "
                f"num_outputs={num_outputs}"
            )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of the Vision Transformer MLX model.

        Args:
            x (mx.array): Input image batch (Batch, H, W, C - Channels Last!).

        Returns:
            mx.array: Logits output. Shape depends on num_outputs:
                - If num_outputs=1: (Batch, NumClasses)
                - If num_outputs>1: (Batch, NumOutputs, NumClasses)
        """
        # 1. Patch Embeddings
        # (B, H, W, C) -> (B, N+1, D)
        x = self.patch_embed(x)

        # 2. Transformer Encoder Blocks
        for block in self.encoder_blocks:
            x = block(x)

        # 3. Extract CLS token and normalize
        # Standard ViT uses the CLS token's output representation for classification
        cls_token_output = x[:, 0]  # Select CLS token -> (B, D)
        cls_token_output = self.norm(cls_token_output)  # Apply final normalization

        # 4. MLP Head
        logits = self.head(cls_token_output)  # (B, NumOutputs * NumClasses)

        # 5. Reshape for multi-output phases (like Phase 2)
        if self.num_outputs > 1:
            B = x.shape[0]  # Get batch size dynamically
            logits = logits.reshape(
                B, self.num_outputs, self.num_classes
            )  # (B, NumOutputs, NumClasses)

        return logits

# --- Test Block ---
if __name__ == '__main__':
    if logger:
        logger.info("🧪 Testing VisionTransformerMLX Model...")

    # --- Test Phase 1 Config ---
    logger.info("\n--- Testing Phase 1 Configuration ---")
    p1_batch_size = 4
    p1_img_size = 28
    p1_patch_size = 7
    p1_in_channels = 1
    p1_num_classes = 10
    p1_embed_dim = 64
    p1_depth = 4
    p1_num_heads = 4
    p1_num_outputs = 1
    p1_mlp_ratio = 2.0

    # Input shape (B, H, W, C) for MLX
    p1_dummy_images = mx.random.normal(
        shape=(p1_batch_size, p1_img_size, p1_img_size, p1_in_channels)
    )
    if logger:
        logger.info(f"Phase 1 Input shape: {p1_dummy_images.shape}")

    # Instantiate the MLX model
    p1_vit_model = VisionTransformerMLX(
        img_size=p1_img_size,
        patch_size=p1_patch_size,
        in_channels=p1_in_channels,
        num_classes=p1_num_classes,
        embed_dim=p1_embed_dim,
        depth=p1_depth,
        num_heads=p1_num_heads,
        mlp_ratio=p1_mlp_ratio,
        num_outputs=p1_num_outputs,
    )
    mx.eval(p1_vit_model.parameters())  # Evaluate parameters after initialization

    try:
        p1_output_logits = p1_vit_model(p1_dummy_images)
        mx.eval(p1_output_logits)  # Evaluate the output tensor
        if logger:
            logger.info(f"✅ Phase 1 MLX Model Forward Pass Successful!")
            logger.info(f"Phase 1 Output logits shape: {p1_output_logits.shape}")
        assert p1_output_logits.shape == (
            p1_batch_size,
            p1_num_classes,
        ), "Phase 1 Output shape mismatch!"
    except Exception as e:
        if logger:
            logger.error(
                f"❌ Error during Phase 1 MLX model test: {e}", exc_info=True
            )

    # --- Test Phase 2 Config ---
    logger.info("\n--- Testing Phase 2 Configuration ---")
    p2_batch_size = 2
    p2_img_size = 56
    p2_patch_size = 7
    p2_in_channels = 1
    p2_num_classes = 10
    p2_embed_dim = 64
    p2_depth = 4
    p2_num_heads = 4
    p2_num_outputs = 4
    p2_mlp_ratio = 2.0

    p2_dummy_images = mx.random.normal(
        shape=(p2_batch_size, p2_img_size, p2_img_size, p2_in_channels)
    )
    if logger:
        logger.info(f"Phase 2 Input shape: {p2_dummy_images.shape}")

    p2_vit_model = VisionTransformerMLX(
        img_size=p2_img_size,
        patch_size=p2_patch_size,
        in_channels=p2_in_channels,
        num_classes=p2_num_classes,
        embed_dim=p2_embed_dim,
        depth=p2_depth,
        num_heads=p2_num_heads,
        mlp_ratio=p2_mlp_ratio,
        num_outputs=p2_num_outputs,
    )
    mx.eval(p2_vit_model.parameters())

    try:
        p2_output_logits = p2_vit_model(p2_dummy_images)
        mx.eval(p2_output_logits)
        if logger:
            logger.info(f"✅ Phase 2 MLX Model Forward Pass Successful!")
            logger.info(f"Phase 2 Output logits shape: {p2_output_logits.shape}")
        assert p2_output_logits.shape == (
            p2_batch_size,
            p2_num_outputs,
            p2_num_classes,
        ), "Phase 2 Output shape mismatch!"
    except Exception as e:
        if logger:
            logger.error(
                f"❌ Error during Phase 2 MLX model test: {e}", exc_info=True
            )