# MNIST Digit Classifier (Transformer) - PyTorch Version
# File: src/mnist_transformer/modules.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Building blocks for ViT model (Encoder & Decoder components).
# Created: 2025-04-28
# Updated: 2025-04-29

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
from pathlib import Path
from typing import Optional

# --- Add project root to sys.path for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent  # Go up two levels
if str(project_root) not in sys.path:
    print(f"🧩 [modules.py] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

# Use logger from utils if available
try:
    from utils import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.info("Using basic fallback logger for modules.py")


# --- Patch Embedding (PyTorch Version) ---
class PatchEmbedding(nn.Module):
    """
    Splits image into patches, flattens, linearly projects them (PyTorch).
    Prepends a CLS token and adds positional embeddings.

    Args:
        image_size (int): Size of the input image (assumed square). Default 28.
        patch_size (int): Size of each square patch. Default 7.
        in_channels (int): Number of input image channels. Default 1.
        embed_dim (int): The dimensionality of the patch embeddings. Default 64.
    """
    def __init__(
        self, image_size=28, patch_size=7, in_channels=1, embed_dim=64
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        if image_size % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch_size.")

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.position_embedding, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding
        return x


# --- Attention Head (PyTorch Version) ---
class AttentionHead(nn.Module):
    """
    A single head of self-attention / cross-attention (PyTorch).

    Args:
        embed_dim (int): Total embedding dimension of the input.
        head_dim (int): Dimension of the subspace for this head.
        dropout (float): Dropout probability for attention weights.
    """
    def __init__(
        self, embed_dim: int, head_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, head_dim)
        self.k_proj = nn.Linear(embed_dim, head_dim)
        self.v_proj = nn.Linear(embed_dim, head_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = attn_weights @ v
        return output


# --- Multi-Head Attention (PyTorch Version) ---
class Attention(nn.Module):
    """
    Multi-Head Attention module using multiple AttentionHead instances.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability for attention weights.
        projection_dropout (float): Dropout probability after projection.
    """
    def __init__(
        self, embed_dim=64, num_heads=4, dropout=0.0, projection_dropout=0.0
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([
            AttentionHead(embed_dim, self.head_dim, dropout=dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(projection_dropout)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        head_outputs = [
            head(query, key, value, attn_mask) for head in self.heads
        ]
        x = torch.cat(head_outputs, dim=-1)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


# --- MLP Block (PyTorch Version) ---
class MLPBlock(nn.Module):
    """
    Standard Transformer MLP block (Linear -> Activation -> Dropout -> ...).
    """
    def __init__(self, embed_dim=64, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
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


# --- Transformer Encoder Block (PyTorch Version) ---
class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer Encoder block (PyTorch).
    """
    def __init__(
        self, embed_dim=64, num_heads=4, mlp_ratio=2.0, dropout=0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_out = self.attention(query=x_norm, key=x_norm, value=x_norm)
        x = residual + attn_out
        residual = x
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out
        return x


# --- Transformer Decoder Block (PyTorch Version) ---
class TransformerDecoderBlock(nn.Module):
    """
    Standard Transformer Decoder block (PyTorch).

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP expansion ratio.
        dropout (float): General dropout rate.
    """
    def __init__(
        self, embed_dim=64, num_heads=4, mlp_ratio=2.0, dropout=0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.masked_self_attn = Attention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = Attention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(
            embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = tgt
        tgt_norm = self.norm1(tgt)
        self_attn_out = self.masked_self_attn(
            query=tgt_norm, key=tgt_norm, value=tgt_norm, attn_mask=tgt_mask
        )
        tgt = residual + self.dropout(self_attn_out)
        residual = tgt
        tgt_norm = self.norm2(tgt)
        cross_attn_out = self.cross_attn(
            query=tgt_norm, key=memory, value=memory, attn_mask=memory_mask
        )
        tgt = residual + self.dropout(cross_attn_out)
        residual = tgt
        tgt_norm = self.norm3(tgt)
        mlp_out = self.mlp(tgt_norm)
        tgt = residual + self.dropout(mlp_out)
        return tgt


# --- Test Block ---
if __name__ == '__main__':
    logger.info("🧪 Testing PyTorch Transformer Modules (Encoder & Decoder)...")
    _B = 4
    _S_src = 65
    _S_tgt = 10
    _D = 64
    _H = 4
    encoder_input = torch.randn(_B, _S_src, _D)
    decoder_input = torch.randn(_B, _S_tgt, _D)
    causal_mask_bool = torch.triu(
        torch.ones(_S_tgt, _S_tgt, dtype=torch.bool), diagonal=1
    )
    causal_mask = torch.tril(torch.ones(_S_tgt, _S_tgt))
    logger.info("\n--- Testing Encoder Block ---")
    encoder_block = TransformerEncoderBlock(
        embed_dim=_D, num_heads=_H, dropout=0.1
    )
    encoder_output = encoder_block(encoder_input)
    logger.info(f"Encoder Block Output Shape: {encoder_output.shape}")
    assert encoder_output.shape == (_B, _S_src, _D)
    logger.info("\n--- Testing Decoder Block ---")
    decoder_block = TransformerDecoderBlock(
        embed_dim=_D, num_heads=_H, dropout=0.1
    )
    decoder_output = decoder_block(
        tgt=decoder_input,
        memory=encoder_output,
        tgt_mask=causal_mask
    )
    logger.info(f"Decoder Block Output Shape: {decoder_output.shape}")
    assert decoder_output.shape == (_B, _S_tgt, _D)
    logger.info("✅ All module tests passed (based on shape checks).")
