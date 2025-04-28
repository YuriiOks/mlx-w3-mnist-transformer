# MNIST Digit Classifier (Transformer)
# File: src/mnist_transformer/modules.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Building blocks for the Vision Transformer model.
# Created: 2025-04-28
# Updated: 2025-04-28

import torch
import torch.nn as nn
import math
import os
import sys
from pathlib import Path

# --- Add project root to sys.path for imports ---
# Get the current script directory
project_root = Path(os.getcwd())
if str(project_root) not in sys.path:
    print(f"📂 Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))


from utils import logger

# --- 1. Patch Embedding ---
# Takes an image, splits into patches, flattens, and projects to embed_dim.
# Also adds CLS token and positional embeddings.

class PatchEmbedding(nn.Module):
    """
    Splits image into patches, flattens, and linearly projects them.
    Prepends a CLS token and adds positional embeddings.

    Args:
        image_size (int): Size of the input image (assumed square). 
            Default 28 for MNIST.
        patch_size (int): Size of each square patch. Default 7 for MNIST.
        in_channels (int): Number of input image channels. Default 1 for MNIST.
        embed_dim (int): The dimensionality of the patch embeddings. Default 64.
    """
    def __init__(self, image_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        if image_size % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch_size.")

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # --- Layers ---
        # CLS token (learnable parameter)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # (1, 1, D)

        # Linear projection layer for patches
        # Can be implemented efficiently with a Conv2d layer:
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        # Alternatively, use unfold + Linear (as in notebook)

        # Positional embedding (learnable parameter)
        # +1 for the CLS token
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )  # (1, N+1, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding.

        Args:
            x (torch.Tensor): Input image tensor (Batch, C, H, W).

        Returns:
            torch.Tensor: Embedded patches with CLS token and pos embedding 
            (Batch, N+1, D).
        """
        batch_size = x.shape[0]

        # Project patches using Conv2d: (B, C, H, W) -> (B, D, H/P, W/P)
        x = self.projection(x)

        # Flatten the spatial dimensions: (B, D, H/P, W/P) -> (B, D, N)
        x = x.flatten(2)

        # Transpose to match transformer input format: (B, D, N) -> (B, N, D)
        x = x.transpose(1, 2)

        # Expand CLS token to batch size: (1, 1, D) -> (B, 1, D)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Prepend CLS token to patch sequence: (B, 1, D) + (B, N, D) -> (B, N+1, D)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding: (B, N+1, D) + (1, N+1, D) -> (B, N+1, D)
        # Broadcasting handles the batch dimension
        x = x + self.position_embedding

        return x

# --- 2. Multi-Head Self-Attention ---
# Using PyTorch's built-in MultiheadAttention for convenience

class Attention(nn.Module):
    """
    Wrapper around nn.MultiheadAttention.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability. Default 0.0.
    """
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
             raise ValueError("embed_dim must be divisible by num_heads")
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # IMPORTANT: Assumes input shape (B, Seq, Dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head self-attention.

        Args:
            x (torch.Tensor): Input tensor (Batch, SeqLen, Dim).

        Returns:
            torch.Tensor: Output tensor after attention (Batch, SeqLen, Dim).
        """
        # nn.MultiheadAttention expects query, key, value
        # For self-attention, they are the same
        # It returns attn_output, attn_output_weights (we only need the output)
        attn_output, _ = self.attention(x, x, x)
        return attn_output

# --- 3. MLP Block ---
# Standard feedforward block used in Transformers.

class MLPBlock(nn.Module):
    """
    Standard Transformer MLP block 
    (Linear -> Activation -> Dropout -> Linear -> Dropout).

    Args:
        embed_dim (int): Input and output dimension.
        mlp_ratio (float): Ratio to expand hidden dimension. Default 2.0.
        dropout (float): Dropout probability. Default 0.1.
    """
    def __init__(self, embed_dim=64, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()  # GELU is common in ViTs
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

# --- 4. Transformer Encoder Block ---
# Combines Multi-Head Attention and MLP with LayerNorm and Residual Connections.

class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer Encoder block.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP expansion ratio. Default 2.0.
        attention_dropout (float): Dropout for attention module. Default 0.1.
        mlp_dropout (float): Dropout for MLP block. Default 0.1.
    """
    def __init__(
        self, 
        embed_dim=64, 
        num_heads=4, 
        mlp_ratio=2.0, 
        attention_dropout=0.1, 
        mlp_dropout=0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=mlp_dropout
        )
        # Optional: Stochastic Depth (DropPath) could be added here later

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-block
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = x + residual  # Residual connection 1

        # MLP sub-block
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual  # Residual connection 2

        return x

# --- Test Block (Optional) ---
if __name__ == '__main__':
    logger.info("🧪 Testing Vision Transformer Modules...")

    # Test Parameters (Phase 1 MNIST)
    _batch_size = 4
    _image_size = 28
    _patch_size = 7
    _in_channels = 1
    _embed_dim = 64
    _num_heads = 4
    _depth = 4  # Number of encoder blocks
    _mlp_ratio = 2.0

    # Create dummy input image batch
    dummy_images = torch.randn(
        _batch_size, _in_channels, _image_size, _image_size
    )
    logger.info(f"Input image batch shape: {dummy_images.shape}")

    # 1. Test PatchEmbedding
    patch_embed = PatchEmbedding(
        image_size=_image_size,
        patch_size=_patch_size,
        in_channels=_in_channels,
        embed_dim=_embed_dim
    )
    patch_output = patch_embed(dummy_images)
    logger.info(f"PatchEmbedding output shape: {patch_output.shape}")
    _expected_seq_len = (28//7)**2 + 1  # Num patches + CLS token
    assert patch_output.shape == (
        _batch_size, _expected_seq_len, _embed_dim
    ), "PatchEmbedding shape mismatch!"

    # 2. Test TransformerEncoderBlock
    encoder_block = TransformerEncoderBlock(
        embed_dim=_embed_dim,
        num_heads=_num_heads,
        mlp_ratio=_mlp_ratio
    )
    encoder_output = encoder_block(patch_output)
    logger.info(f"TransformerEncoderBlock output shape: {encoder_output.shape}")
    assert encoder_output.shape == patch_output.shape, "EncoderBlock shape mismatch!"

    # 3. Test Attention module directly (optional)
    attn_module = Attention(embed_dim=_embed_dim, num_heads=_num_heads)
    attn_output = attn_module(patch_output)
    logger.info(f"Attention module output shape: {attn_output.shape}")
    assert attn_output.shape == patch_output.shape, "Attention module shape mismatch!"

    # 4. Test MLP Block directly (optional)
    mlp_module = MLPBlock(embed_dim=_embed_dim, mlp_ratio=_mlp_ratio)
    mlp_output = mlp_module(patch_output)
    logger.info(f"MLPBlock output shape: {mlp_output.shape}")
    assert mlp_output.shape == patch_output.shape, "MLPBlock shape mismatch!"

    logger.info("✅ All module tests passed (based on shape checks).")