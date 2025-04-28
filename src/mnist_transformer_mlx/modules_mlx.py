# MNIST Digit Classifier (Transformer) - MLX Version
# File: src/mnist_transformer_mlx/modules_mlx.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Building blocks for the Vision Transformer model using MLX.
# Created: 2025-04-28
# Updated: 2025-04-28

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional

# --- Add project root for logger ---
import sys
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent
if str(project_root) not in sys.path:
    print(f"🧠 [modules_mlx.py] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))
from utils import logger

# --- 1. Patch Embedding (MLX Version) ---

class PatchEmbeddingMLX(nn.Module):
    """
    Splits image into patches, flattens, linearly projects, adds CLS & Pos embeds.
    Args:
        image_size (int): Size of the input image (assumed square).
        patch_size (int): Size of each square patch.
        in_channels (int): Number of input image channels.
        embed_dim (int): The dimensionality of the patch embeddings.
    """
    def __init__(
        self, image_size: int = 28, patch_size: int = 7,
        in_channels: int = 1, embed_dim: int = 64
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch_size.")

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.cls_token = mx.random.normal(shape=(1, 1, embed_dim)) * 0.02
        self.position_embedding = mx.random.normal(
            shape=(1, self.num_patches + 1, embed_dim)
        ) * 0.02

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass for patch embedding.
        Args:
            x (mx.array): Input image tensor (Batch, H, W, C).
        Returns:
            mx.array: Embedded patches with CLS token and pos embedding.
        """
        B, H, W, C = x.shape
        x = self.projection(x)
        x = x.reshape(B, -1, self.embed_dim)
        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = mx.concatenate((cls_tokens, x), axis=1)
        x = x + self.position_embedding
        return x

# --- 2. Attention Head (MLX Version) ---
class AttentionHeadMLX(nn.Module):
    """ Single head of self-attention (MLX). """
    def __init__(self, embed_dim: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.head_dim = head_dim
        self.scale = float(head_dim) ** -0.5
        self.q_proj = nn.Linear(embed_dim, head_dim)
        self.k_proj = nn.Linear(embed_dim, head_dim)
        self.v_proj = nn.Linear(embed_dim, head_dim)
        self.attn_dropout = nn.Dropout(p=dropout)

    def __call__(
        self, x_q: mx.array, x_kv: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass for single head. Allows for self-attention (x_kv is None)
        or cross-attention (x_kv is provided).
        Args:
            x_q (mx.array): Input for Query projection (B, N_q, D).
            x_kv (mx.array, optional): Input for Key/Value projection (B, N_kv, D).
        Returns:
            mx.array: Output tensor for this head (B, N_q, head_dim).
        """
        if x_kv is None:
            x_kv = x_q

        B_q, N_q, C_q = x_q.shape
        B_kv, N_kv, C_kv = x_kv.shape

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        attn_scores = (q @ k.transpose(0, 2, 1)) * self.scale
        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = attn_weights @ v
        return output

# --- 3. Multi-Head Attention (MLX Version) ---
class AttentionMLX(nn.Module):
    """ Multi-Head Attention using AttentionHeadMLX (MLX). """
    def __init__(
        self, embed_dim: int = 64, num_heads: int = 4,
        attention_dropout: float = 0.0, projection_dropout: float = 0.0
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.heads = [
            AttentionHeadMLX(embed_dim, self.head_dim, dropout=attention_dropout)
            for _ in range(num_heads)
        ]
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(p=projection_dropout)

    def __call__(
        self, x_q: mx.array, x_kv: Optional[mx.array] = None
    ) -> mx.array:
        head_outputs = [head(x_q, x_kv) for head in self.heads]
        x = mx.concatenate(head_outputs, axis=-1)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x

# --- 4. MLP Block (MLX Version) ---
class MLPBlockMLX(nn.Module):
    """ Standard Transformer MLP block (MLX). """
    def __init__(
        self, embed_dim: int = 64, mlp_ratio: float = 2.0, dropout: float = 0.1
    ):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(p=dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

# --- 5. Transformer Encoder Block (MLX Version) ---
class TransformerEncoderBlockMLX(nn.Module):
    """ Standard Transformer Encoder block (MLX). """
    def __init__(
        self, embed_dim: int = 64, num_heads: int = 4, mlp_ratio: float = 2.0,
        attention_dropout: float = 0.1, mlp_dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = AttentionMLX(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlockMLX(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=mlp_dropout
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm)
        x = residual + attn_out
        residual = x
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out
        return x

# --- Test Block ---
if __name__ == '__main__':
    if logger:
        logger.info("🧪 Testing Vision Transformer MLX Modules...")

    _batch_size = 4
    _image_size = 28
    _patch_size = 7
    _in_channels = 1
    _embed_dim = 64
    _num_heads = 4
    _depth = 4
    _mlp_ratio = 2.0

    dummy_images = mx.random.normal(
        shape=(_batch_size, _image_size, _image_size, _in_channels)
    )
    if logger:
        logger.info(f"Input image batch shape: {dummy_images.shape}")

    patch_embed = PatchEmbeddingMLX(
        image_size=_image_size, patch_size=_patch_size,
        in_channels=_in_channels, embed_dim=_embed_dim
    )
    patch_output = patch_embed(dummy_images)
    if logger:
        logger.info(f"PatchEmbeddingMLX output shape: {patch_output.shape}")
    _expected_seq_len = (_image_size // _patch_size) ** 2 + 1
    assert patch_output.shape == (
        _batch_size, _expected_seq_len, _embed_dim
    ), "PatchEmbedding shape mismatch!"

    encoder_block = TransformerEncoderBlockMLX(
        embed_dim=_embed_dim, num_heads=_num_heads, mlp_ratio=_mlp_ratio
    )
    mx.eval(encoder_block.parameters())
    encoder_output = encoder_block(patch_output)
    mx.eval(encoder_output)
    if logger:
        logger.info(
            f"TransformerEncoderBlockMLX output shape: {encoder_output.shape}"
        )
    assert encoder_output.shape == patch_output.shape, \
        "EncoderBlock shape mismatch!"

    attn_module = AttentionMLX(embed_dim=_embed_dim, num_heads=_num_heads)
    mx.eval(attn_module.parameters())
    attn_output = attn_module(patch_output)
    mx.eval(attn_output)
    if logger:
        logger.info(f"AttentionMLX module output shape: {attn_output.shape}")
    assert attn_output.shape == patch_output.shape, \
        "Attention module shape mismatch!"

    mlp_module = MLPBlockMLX(embed_dim=_embed_dim, mlp_ratio=_mlp_ratio)
    mx.eval(mlp_module.parameters())
    mlp_output = mlp_module(patch_output)
    mx.eval(mlp_output)
    if logger:
        logger.info(f"MLPBlockMLX output shape: {mlp_output.shape}")
    assert mlp_output.shape == patch_output.shape, \
        "MLPBlock shape mismatch!"

    if logger:
        logger.info("✅ All MLX module tests passed (based on shape checks).")
