# MNIST Digit Classifier (Transformer)
# File: src/mnist_transformer/model.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Defines the Vision Transformer (ViT) model architecture.
# Created: 2025-04-28
# Updated: 2025-04-28

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from typing import Dict, Any

# --- Add project root to sys.path for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent # Go up two levels
if str(project_root) not in sys.path:
    print(f"🏗️ [model.py] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

# Import building blocks from modules.py
try:
    from src.mnist_transformer.modules import (
        PatchEmbedding,
        TransformerEncoderBlock,
        # Attention and MLPBlock are used *inside* TransformerEncoderBlock
    )
    from utils import logger # Optional: if logging needed within the model
except ImportError as e:
     print(f"❌ Error importing modules in model.py: {e}")
     print("   Ensure you are running from project root or necessary paths are set.")
     # Define dummy classes if import fails, useful for initial checks
     PatchEmbedding = nn.Module
     TransformerEncoderBlock = nn.Module
     logger = None

# --- Vision Transformer Model ---

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model for image classification.

    Args:
        img_size (int): Size of the input image (assumed square). Default 28.
        patch_size (int): Size of each square patch. Default 7.
        in_channels (int): Number of input image channels. Default 1.
        num_classes (int): Number of output classes. Default 10.
        embed_dim (int): Dimensionality of the token/patch embeddings. Default 64.
        depth (int): Number of transformer encoder layers. Default 4.
        num_heads (int): Number of attention heads. Default 4.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Default 2.0.
        attention_dropout (float): Dropout rate for attention weights. Default 0.1.
        mlp_dropout (float): Dropout rate for MLP layers. Default 0.1.
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
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # 1. Patch + Position Embedding (+ CLS Token)
        self.patch_embed = PatchEmbedding(
            image_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        # Calculate number of patches for convenience
        self.num_patches = self.patch_embed.num_patches

        # Optional: Dropout after patch+pos embedding
        # self.pos_drop = nn.Dropout(p=dropout) # Example if adding dropout here

        # 2. Transformer Encoder Blocks
        # Create a stack of TransformerEncoderBlocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(depth)])

        # 3. Final Layer Normalization (applied before MLP head)
        self.norm = nn.LayerNorm(embed_dim)

        # 4. MLP Classifier Head
        # Takes the output of the CLS token
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights (optional but recommended for stability)
        self._initialize_weights()

        if logger: logger.info(f"🧠 VisionTransformer initialized: depth={depth}, heads={num_heads}, embed_dim={embed_dim}")

    def _initialize_weights(self):
        # Simple initialization: LayerNorm biases zero, weights one. Linear layers Xavier init.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        # Special init for pos_embed and cls_token (often near zero)
        nn.init.trunc_normal_(self.patch_embed.position_embedding, std=.02)
        nn.init.trunc_normal_(self.patch_embed.cls_token, std=.02)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input image batch (Batch, C, H, W).

        Returns:
            torch.Tensor: Logits output (Batch, NumClasses).
        """
        # 1. Get Patch Embeddings + CLS token + Positional Embeddings
        # (B, N+1, D)
        x = self.patch_embed(x)
        # Optional: self.pos_drop(x)

        # 2. Pass through Transformer Encoder Blocks
        for block in self.encoder_blocks:
            x = block(x)

        # 3. Apply final Layer Normalization to the CLS token output
        # We only need the representation of the first token ([CLS]) for classification
        cls_token_output = x[:, 0] # Select CLS token (B, 1, D) -> (B, D)
        cls_token_output = self.norm(cls_token_output)

        # 4. Pass CLS token output through MLP Head
        # (B, D) -> (B, NumClasses)
        logits = self.head(cls_token_output)

        return logits

# --- Test Block ---
if __name__ == '__main__':
    if logger: logger.info("🧪 Testing VisionTransformer Model...")

    # Test Parameters (Phase 1 MNIST)
    _batch_size = 4
    _image_size = 28
    _patch_size = 7
    _in_channels = 1
    _num_classes = 10
    _embed_dim = 64
    _depth = 4
    _num_heads = 4
    _mlp_ratio = 2.0

    # Create dummy input image batch
    dummy_images = torch.randn(_batch_size, _in_channels, _image_size, _image_size)
    if logger: logger.info(f"Input image batch shape: {dummy_images.shape}")

    # Instantiate the model
    vit_model = VisionTransformer(
        img_size=_image_size,
        patch_size=_patch_size,
        in_channels=_in_channels,
        num_classes=_num_classes,
        embed_dim=_embed_dim,
        depth=_depth,
        num_heads=_num_heads,
        mlp_ratio=_mlp_ratio
    )

    # Perform forward pass
    try:
        output_logits = vit_model(dummy_images)
        if logger: logger.info(f"✅ Model Forward Pass Successful!")
        if logger: logger.info(f"Output logits shape: {output_logits.shape}") # Expected: (BatchSize, NumClasses)
        assert output_logits.shape == (_batch_size, _num_classes), "Output shape mismatch!"

        # Print model summary (optional - requires torchinfo)
        try:
            from torchinfo import summary
            print("\n--- Model Summary ---")
            summary(vit_model, input_size=dummy_images.shape)
        except ImportError:
            if logger: logger.warning("torchinfo not installed, skipping model summary.")
            print(vit_model) # Print basic structure

    except Exception as e:
        if logger: logger.error(f"❌ Error during model forward pass test: {e}", exc_info=True)