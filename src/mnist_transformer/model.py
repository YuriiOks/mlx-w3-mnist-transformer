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
    )
    from utils import logger # Optional: if logging needed within the model
except ImportError as e:
     print(f"❌ Error importing modules in model.py: {e}")
     print("   Ensure you are running from project root or necessary paths are set.")
     # Define dummy classes if import fails
     PatchEmbedding = nn.Module
     TransformerEncoderBlock = nn.Module
     logger = None

# --- Vision Transformer Model ---

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model for image classification.
    Adapts for single digit (Phase 1) or multi-digit grid (Phase 2).

    Args:
        img_size (int): Size of the input image (assumed square).
        patch_size (int): Size of each square patch.
        in_channels (int): Number of input image channels.
        num_classes (int): Number of classes *per output head*. 
            (e.g., 10 for MNIST digits).
        embed_dim (int): Dimensionality of the token/patch embeddings.
        depth (int): Number of transformer encoder layers.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        attention_dropout (float): Dropout rate for attention weights.
        mlp_dropout (float): Dropout rate for MLP layers.
        num_outputs (int): Number of separate classification outputs needed 
            (1 for Phase 1, 4 for Phase 2). Default 1.
    """
    def __init__(
        self,
        img_size: int, # Required: e.g., 28 for P1, 56 for P2
        patch_size: int,
        in_channels: int,
        num_classes: int, # Classes per digit (e.g., 10)
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        attention_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        num_outputs: int = 1, # <-- ADDED: Number of digits to predict
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_outputs
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # 1. Patch + Position Embedding (+ CLS Token)
        # Pass the correct img_size received from arguments
        self.patch_embed = PatchEmbedding(
            image_size=img_size, # Use argument
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        # 2. Transformer Encoder Blocks
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
        # --- 👇 Correct Assignment ---
        self.norm = nn.LayerNorm(embed_dim) # Assign to self.norm
        
        # 4. MLP Classifier Head
        # Output dimension depends on the number of digits we want to predict
        # Output size = num_outputs * num_classes (e.g., 4 * 10 = 40 for Phase 2)
        self.head = nn.Linear(embed_dim, num_classes * num_outputs) # <-- MODIFIED HEAD SIZE

        # Initialize weights
        self._initialize_weights()

        if logger: logger.info(f"🧠 VisionTransformer initialized: img_size={img_size}, patch_size={patch_size}, depth={depth}, heads={num_heads}, embed_dim={embed_dim}, num_outputs={num_outputs}")

    def _initialize_weights(self):
        # ... (initialization code remains the same) ...
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        nn.init.trunc_normal_(self.patch_embed.position_embedding, std=.02)
        nn.init.trunc_normal_(self.patch_embed.cls_token, std=.02)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input image batch (Batch, C, H, W).

        Returns:
            torch.Tensor: Logits output. Shape depends on num_outputs:
                          - If num_outputs=1: (Batch, NumClasses)
                          - If num_outputs>1: (Batch, NumOutputs, NumClasses) # Reshaped for clarity
        """
        # 1. Get Patch Embeddings + CLS token + Positional Embeddings
        x = self.patch_embed(x) # (B, N+1, D)

        # 2. Pass through Transformer Encoder Blocks
        for block in self.encoder_blocks:
            x = block(x)

        # 3. Apply final Layer Normalization to the CLS token output
        cls_token_output = x[:, 0] # Select CLS token -> (B, D)
        cls_token_output = self.norm(cls_token_output)

        # 4. Pass CLS token output through MLP Head
        logits = self.head(cls_token_output) # (B, NumOutputs * NumClasses)

        # 5. Reshape if multiple outputs for clarity and easier loss calculation
        # Reshape from (B, NumOutputs * NumClasses) to (B, NumOutputs, NumClasses)
        if self.num_outputs > 1:
            logits = logits.view(x.shape[0], self.num_outputs, self.num_classes) # <-- RESHAPE ADDED

        return logits

# --- Test Block ---
if __name__ == '__main__':
    if logger: logger.info("🧪 Testing VisionTransformer Model (Phase 1 & Phase 2)...")

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
    p1_num_outputs = 1 # Single digit output
    p1_mlp_ratio = 2.0 # <-- You need to define this variable

    p1_dummy_images = torch.randn(p1_batch_size, p1_in_channels, p1_img_size, p1_img_size)
    if logger: logger.info(f"Phase 1 Input shape: {p1_dummy_images.shape}")

    p1_vit_model = VisionTransformer(
        img_size=p1_img_size,
        patch_size=p1_patch_size,
        in_channels=p1_in_channels,
        num_classes=p1_num_classes,
        embed_dim=p1_embed_dim,
        depth=p1_depth,
        num_heads=p1_num_heads,
        num_outputs=p1_num_outputs,
        mlp_ratio=p1_mlp_ratio
    )

    try:
        p1_output_logits = p1_vit_model(p1_dummy_images)
        if logger: logger.info(f"✅ Phase 1 Model Forward Pass Successful!")
        if logger: logger.info(f"Phase 1 Output logits shape: {p1_output_logits.shape}")
        assert p1_output_logits.shape == (p1_batch_size, p1_num_classes), "Phase 1 Output shape mismatch!"
    except Exception as e:
        if logger: logger.error(f"❌ Error during Phase 1 model test: {e}", exc_info=True)


    # --- Test Phase 2 Config ---
    logger.info("\n--- Testing Phase 2 Configuration ---")
    p2_batch_size = 2 # Smaller batch for potentially larger model
    p2_img_size = 56 # 2x2 grid
    p2_patch_size = 7 # Keep patch size same? Or change? Let's keep 7 for now (-> 64 patches)
    p2_in_channels = 1
    p2_num_classes = 10 # Still 10 digits per output
    p2_embed_dim = 64 # Keep embed_dim same
    p2_depth = 4 # Keep depth same
    p2_num_heads = 4 # Keep heads same
    p2_num_outputs = 4 # Predict 4 digits
    p2_mlp_ratio = 2.0 # Define mlp_ratio for phase 2


    p2_dummy_images = torch.randn(p2_batch_size, p2_in_channels, p2_img_size, p2_img_size)
    if logger: logger.info(f"Phase 2 Input shape: {p2_dummy_images.shape}")

    p2_vit_model = VisionTransformer(
        img_size=p2_img_size,
        patch_size=p2_patch_size,
        in_channels=p2_in_channels,
        num_classes=p2_num_classes,
        embed_dim=p2_embed_dim,
        depth=p2_depth,
        num_heads=p2_num_heads, num_outputs=p2_num_outputs,
        mlp_ratio=p2_mlp_ratio, 
    )

    try:
        p2_output_logits = p2_vit_model(p2_dummy_images)
        if logger: logger.info(f"✅ Phase 2 Model Forward Pass Successful!")
        if logger: logger.info(f"Phase 2 Output logits shape: {p2_output_logits.shape}") # Expected: (Batch, NumOutputs, NumClasses)
        assert p2_output_logits.shape == (p2_batch_size, p2_num_outputs, p2_num_classes), "Phase 2 Output shape mismatch!"
    except Exception as e:
        if logger: logger.error(f"❌ Error during Phase 2 model test: {e}", exc_info=True)