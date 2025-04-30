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
import numpy as np

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
        TransformerDecoderBlockMLX
    )
    from utils import logger
except ImportError as e:
    print(f"❌ Error importing modules in model_mlx.py: {e}")
    # Define dummy classes if import fails
    PatchEmbeddingMLX = nn.Module
    TransformerEncoderBlockMLX = nn.Module
    import logging
    logger = logging.getLogger("model_mlx")

# --- Phase 1 & 2: Vision Transformer (Encoder Only - MLX) ---
class VisionTransformerMLX(nn.Module):
    """
    Vision Transformer (ViT) for image classification using MLX.

    This model implements the encoder-only ViT architecture for both single-output (Phase 1)
    and multi-output (Phase 2) classification tasks. It splits the input image into patches,
    embeds them, applies a stack of Transformer encoder blocks, and uses the [CLS] token's
    representation for classification.

    Args:
        img_size (int): Size of the input image (assumed square).
        patch_size (int): Size of each square patch.
        in_channels (int): Number of input image channels.
        num_classes (int): Number of classes per output head.
        embed_dim (int): Dimensionality of the patch embeddings.
        depth (int): Number of transformer encoder layers.
        num_heads (int): Number of attention heads per layer.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        dropout (float): Dropout rate for all layers.
        num_outputs (int): Number of separate classification outputs (1 for Phase 1, 4 for Phase 2).
    """
    def __init__(
        self, img_size: int = 28, patch_size: int = 7, in_channels: int = 1,
        num_classes: int = 10, embed_dim: int = 64, depth: int = 4,
        num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1,
        num_outputs: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_outputs
        self.patch_embed = PatchEmbeddingMLX(img_size, patch_size, in_channels, embed_dim)
        self.encoder_blocks = [
            TransformerEncoderBlockMLX(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)]
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes * num_outputs)
        if logger:
            logger.info(f"🧠 VisionTransformerMLX init: img={img_size}, patch={patch_size}, depth={depth}, heads={num_heads}, embed={embed_dim}, outputs={num_outputs}")

    def get_encoder_features(self, x: mx.array) -> mx.array:
        """
        Encodes the input image and returns all patch and CLS token features.

        Args:
            x (mx.array): Input image batch (B, H, W, C).
        Returns:
            mx.array: Patch and CLS token features (B, N+1, D).
        """
        x = self.patch_embed(x)
        for block in self.encoder_blocks:
            x = block(x)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass for classification using the [CLS] token.

        Args:
            x (mx.array): Input image batch (B, H, W, C).
        Returns:
            mx.array: Logits output. Shape (B, NumClasses) for Phase 1, (B, NumOutputs, NumClasses) for Phase 2.
        """
        x = self.get_encoder_features(x)
        cls_token_output = self.norm(x[:, 0])
        logits = self.head(cls_token_output)
        if self.num_outputs > 1:
            B = x.shape[0]
            logits = logits.reshape(B, self.num_outputs, self.num_classes)
        return logits

# --- Phase 3: Encoder-Decoder Vision Transformer (MLX) ---
class EncoderDecoderViTMLX(nn.Module):
    """
    Encoder-Decoder Vision Transformer for sequence generation using MLX.

    This model implements a ViT encoder for images and a Transformer decoder for
    sequence generation (e.g., digit sequence prediction). The decoder uses causal
    masking for autoregressive generation and attends to the encoder's output.

    Args:
        img_size (int): Input image size (square).
        patch_size (int): Patch size for encoder.
        in_channels (int): Number of input channels.
        encoder_embed_dim (int): Embedding dim for encoder.
        encoder_depth (int): Number of encoder layers.
        encoder_num_heads (int): Number of encoder attention heads.
        decoder_vocab_size (int): Output vocabulary size for decoder.
        decoder_embed_dim (int): Embedding dim for decoder.
        decoder_depth (int): Number of decoder layers.
        decoder_num_heads (int): Number of decoder attention heads.
        max_seq_len (int): Maximum sequence length for decoder positional embedding.
        mlp_ratio (float): MLP ratio for both encoder and decoder.
        dropout (float): Dropout rate for all layers.
    """
    def __init__(
        self, img_size: int, patch_size: int, in_channels: int,
        encoder_embed_dim: int, encoder_depth: int, encoder_num_heads: int,
        decoder_vocab_size: int, decoder_embed_dim: int, decoder_depth: int,
        decoder_num_heads: int, max_seq_len: int,
        mlp_ratio: float = 2.0, dropout: float = 0.1,
    ):
        super().__init__()
        self.decoder_vocab_size = decoder_vocab_size
        self.patch_embed = PatchEmbeddingMLX(img_size, patch_size, in_channels, encoder_embed_dim)
        self.encoder_blocks = [
            TransformerEncoderBlockMLX(encoder_embed_dim, encoder_num_heads, mlp_ratio, dropout)
            for _ in range(encoder_depth)]
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)
        self.decoder_embed = nn.Embedding(decoder_vocab_size, decoder_embed_dim)
        self.decoder_pos_embed = mx.random.normal(shape=(1, max_seq_len, decoder_embed_dim)) * 0.02
        self.decoder_embed_dropout = nn.Dropout(p=dropout)
        self.decoder_blocks = [
            TransformerDecoderBlockMLX(decoder_embed_dim, decoder_num_heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)]
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.output_head = nn.Linear(decoder_embed_dim, decoder_vocab_size)
        if logger:
            logger.info(f"🧠 EncoderDecoderViTMLX init: EncDepth={encoder_depth}, DecDepth={decoder_depth}")

    def encode(self, src_img: mx.array) -> mx.array:
        """
        Encodes the input image into memory for the decoder.

        Args:
            src_img (mx.array): Input image batch (B, H, W, C).
        Returns:
            mx.array: Encoder memory (B, N+1, D_enc).
        """
        x = self.patch_embed(src_img)
        for block in self.encoder_blocks:
            x = block(x)
        memory = self.encoder_norm(x)
        return memory

    def decode(self, tgt_seq: mx.array, memory: mx.array, tgt_mask: Optional[mx.array] = None) -> mx.array:
        """
        Decodes a target sequence using encoder memory.

        Args:
            tgt_seq (mx.array): Target sequence indices (B, T).
            memory (mx.array): Encoder memory (B, N+1, D_enc).
            tgt_mask (Optional[mx.array]): Causal mask for decoder self-attention.
        Returns:
            mx.array: Decoder output logits (B, T, VocabSize).
        """
        B, T = tgt_seq.shape
        tgt_embed = self.decoder_embed(tgt_seq)
        pos_embed = self.decoder_pos_embed[:, :T]
        x = self.decoder_embed_dropout(tgt_embed + pos_embed)
        for block in self.decoder_blocks:
            x = block(tgt=x, memory=memory, tgt_mask=tgt_mask)
        x = self.decoder_norm(x)
        logits = self.output_head(x)
        return logits

    def __call__(self, src_img: mx.array, tgt_seq: mx.array) -> mx.array:
        """
        Full forward pass for training (teacher forcing).

        Args:
            src_img (mx.array): Input image batch (B, H, W, C).
            tgt_seq (mx.array): Input target sequence (B, T).
        Returns:
            mx.array: Output logits (B, T, VocabSize).
        """
        memory = self.encode(src_img)
        T = tgt_seq.shape[1]
        causal_mask = np.tril(np.ones((T, T))).astype(bool)
        causal_mask_mlx = mx.array(causal_mask)
        logits = self.decode(tgt_seq, memory, tgt_mask=causal_mask_mlx)
        return logits

# --- Test Block ---
if __name__ == '__main__':
    if logger:
        logger.info("🧪 Testing MLX Model Architectures...")

    # --- Test Phase 1/2 ViT ---
    logger.info("\n--- Testing VisionTransformerMLX (Phase 1/2) ---")
    p1_model = VisionTransformerMLX(img_size=28, patch_size=7, num_classes=10, depth=4)
    p1_input = mx.random.normal(shape=(4, 28, 28, 1))
    mx.eval(p1_model.parameters())
    p1_out = p1_model(p1_input)
    mx.eval(p1_out)
    logger.info(f"P1 Output Shape: {p1_out.shape}")
    assert p1_out.shape == (4, 10)

    p2_model = VisionTransformerMLX(img_size=56, patch_size=7, num_classes=10, depth=4, num_outputs=4)
    p2_input = mx.random.normal(shape=(2, 56, 56, 1))
    mx.eval(p2_model.parameters())
    p2_out = p2_model(p2_input)
    mx.eval(p2_out)
    logger.info(f"P2 Output Shape: {p2_out.shape}")
    assert p2_out.shape == (2, 4, 10)
    logger.info("✅ VisionTransformerMLX tests passed.")

    # --- Test Phase 3 EncoderDecoderViT ---
    logger.info("\n--- Testing EncoderDecoderViTMLX (Phase 3) ---")
    _B = 2; _S_tgt = 10; _V_dec = 13; _img_size = 64; _patch_size = 8; _in_C = 1
    _enc_D = 64; _enc_depth = 4; _enc_H = 4
    _dec_D = 64; _dec_depth = 4; _dec_H = 4

    p3_img = mx.random.normal(shape=(_B, _img_size, _img_size, _in_C))
    p3_tgt_input = mx.random.randint(0, _V_dec, (_B, _S_tgt - 1)).astype(mx.uint32)

    p3_model = EncoderDecoderViTMLX(
        img_size=_img_size, patch_size=_patch_size, in_channels=_in_C,
        encoder_embed_dim=_enc_D, encoder_depth=_enc_depth, encoder_num_heads=_enc_H,
        decoder_vocab_size=_V_dec, decoder_embed_dim=_dec_D,
        decoder_depth=_dec_depth, decoder_num_heads=_dec_H,
        max_seq_len=_S_tgt
    )
    mx.eval(p3_model.parameters())

    try:
        output_logits = p3_model(p3_img, p3_tgt_input)
        mx.eval(output_logits)
        if logger:
            logger.info(f"✅ Phase 3 MLX Model Forward Pass Successful!")
            logger.info(f"Phase 3 Output logits shape: {output_logits.shape}")
        assert output_logits.shape == (_B, _S_tgt - 1, _V_dec), "Phase 3 Output shape mismatch!"
        logger.info("✅ EncoderDecoderViTMLX test passed.")
    except Exception as e:
        if logger:
            logger.error(f"❌ Error during Phase 3 MLX model test: {e}", exc_info=True)