# MNIST Digit Classifier (Transformer) - PyTorch Version
# File: src/mnist_transformer/model.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Defines ViT Encoder & Encoder-Decoder model architectures.
# Created: 2025-04-28
# Updated: 2025-04-30

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# --- Add project root for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent
if str(project_root) not in sys.path:
    print(f"🏗️ [model.py] Adding project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

# Import building blocks from modules.py
try:
    from src.mnist_transformer.modules import (
        PatchEmbedding,
        TransformerEncoderBlock,
        TransformerDecoderBlock
    )
    from utils import logger
except ImportError as e:
    print(f"❌ Error importing modules in model.py: {e}")
    PatchEmbedding = nn.Module
    TransformerEncoderBlock = nn.Module
    TransformerDecoderBlock = nn.Module
    import logging
    logger = logging.getLogger(__name__)

# --- Phase 1 & 2: Vision Transformer (Encoder Only) ---
class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification (Phases 1 & 2).

    Args:
        img_size (int): Input image size (height/width).
        patch_size (int): Patch size for splitting image.
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        embed_dim (int): Embedding dimension.
        depth (int): Number of encoder blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP hidden dim ratio.
        dropout (float): Dropout probability.
        num_outputs (int): Number of output heads.
    """
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.1,
        num_outputs: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_outputs
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(
            img_size,
            patch_size,
            in_channels,
            embed_dim
        )
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                dropout
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes * num_outputs)
        self._initialize_weights()
        if logger:
            logger.info(
                f"🧠 VisionTransformer initialized: img={img_size}, "
                f"patch={patch_size}, depth={depth}, heads={num_heads}, "
                f"embed={embed_dim}, outputs={num_outputs}"
            )

    def _initialize_weights(self):
        """
        Initialize model weights using Xavier and truncated normal.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'patch_embed'):
            nn.init.trunc_normal_(
                self.patch_embed.position_embedding, std=.02
            )
            nn.init.trunc_normal_(self.patch_embed.cls_token, std=.02)

    def get_encoder_features(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Pass input through patch embedding and encoder blocks.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Encoded features.
        """
        x = self.patch_embed(x)
        for block in self.encoder_blocks:
            x = block(x)
        return x

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        x = self.get_encoder_features(x)
        cls_token_output = self.norm(x[:, 0])
        logits = self.head(cls_token_output)
        if self.num_outputs > 1:
            logits = logits.view(
                x.shape[0],
                self.num_outputs,
                self.num_classes
            )
        return logits

# --- 👇 NEW: Phase 3 - Encoder-Decoder Vision Transformer ---
class EncoderDecoderViT(nn.Module):
    """
    Encoder-Decoder Vision Transformer for sequence generation (Phase 3).

    Args:
        img_size (int): Input image size.
        patch_size (int): Patch size for image.
        in_channels (int): Number of input channels.
        encoder_embed_dim (int): Encoder embedding dimension.
        encoder_depth (int): Number of encoder blocks.
        encoder_num_heads (int): Encoder attention heads.
        decoder_vocab_size (int): Decoder vocabulary size.
        decoder_embed_dim (int): Decoder embedding dimension.
        decoder_depth (int): Number of decoder blocks.
        decoder_num_heads (int): Decoder attention heads.
        mlp_ratio (float): MLP hidden dim ratio.
        dropout (float): Dropout probability.
        attention_dropout (float): Attention dropout probability.
    """
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        encoder_embed_dim: int,
        encoder_depth: int,
        encoder_num_heads: int,
        decoder_vocab_size: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.decoder_vocab_size = decoder_vocab_size

        # 1. Encoder Part
        self.patch_embed = PatchEmbedding(
            img_size,
            patch_size,
            in_channels,
            encoder_embed_dim
        )
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                encoder_embed_dim,
                encoder_num_heads,
                mlp_ratio,
                attention_dropout,
                dropout
            )
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        # 2. Decoder Part
        self.decoder_embed = nn.Embedding(
            decoder_vocab_size,
            decoder_embed_dim
        )
        self.max_seq_len = 10
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.max_seq_len, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                dropout
            )
            for _ in range(decoder_depth)
        ])
        self.output_head = nn.Linear(
            decoder_embed_dim,
            decoder_vocab_size
        )

        self._initialize_weights()
        if logger:
            logger.info(
                f"🧠 EncoderDecoderViT initialized: EncDepth={encoder_depth}, "
                f"DecDepth={decoder_depth}"
            )

    def _initialize_weights(self):
        """
        Initialize model weights for all modules.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=.02)
        if hasattr(self, 'patch_embed'):
            nn.init.trunc_normal_(
                self.patch_embed.position_embedding, std=.02
            )
            nn.init.trunc_normal_(self.patch_embed.cls_token, std=.02)
        if hasattr(self, 'decoder_pos_embed'):
            nn.init.trunc_normal_(self.decoder_pos_embed, std=.02)

    def encode(
        self,
        src_img: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode input image into memory representation.

        Args:
            src_img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Encoded memory tensor.
        """
        x = self.patch_embed(src_img)
        for block in self.encoder_blocks:
            x = block(x)
        memory = self.encoder_norm(x)
        return memory

    def decode(
        self,
        tgt_seq: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence using encoder memory.

        Args:
            tgt_seq (torch.Tensor): Target token indices.
            memory (torch.Tensor): Encoder memory tensor.
            tgt_mask (Optional[torch.Tensor]): Causal mask.

        Returns:
            torch.Tensor: Output logits for each token.
        """
        B, T = tgt_seq.shape
        tgt_embed = self.decoder_embed(tgt_seq)
        pos_embed = self.decoder_pos_embed[:, :T]
        x = tgt_embed + pos_embed
        for block in self.decoder_blocks:
            x = block(tgt=x, memory=memory, tgt_mask=tgt_mask)
        logits = self.output_head(x)
        return logits

    def forward(
        self,
        src_img: torch.Tensor,
        tgt_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for encoder-decoder model.

        Args:
            src_img (torch.Tensor): Input image tensor.
            tgt_seq (torch.Tensor): Target token indices.

        Returns:
            torch.Tensor: Output logits for each token.
        """
        memory = self.encode(src_img)
        tgt_len = tgt_seq.size(1)
        causal_mask = torch.tril(
            torch.ones(
                tgt_len,
                tgt_len,
                device=tgt_seq.device,
                dtype=torch.bool
            )
        )
        logits = self.decode(
            tgt_seq,
            memory,
            tgt_mask=causal_mask
        )
        return logits

# --- Test Block ---
if __name__ == '__main__':
    if logger:
        logger.info("🧪 Testing PyTorch Model Architectures...")

    # --- Test Phase 1/2 ViT ---
    logger.info("\n--- Testing VisionTransformer (Phase 1/2) ---")
    _mlp_ratio = 2.0

    p1_model = VisionTransformer(
        img_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        depth=4,
        num_heads=4,
        mlp_ratio=_mlp_ratio,
        num_outputs=1
    )
    p1_input = torch.randn(4, 1, 28, 28)
    p1_output = p1_model(p1_input)
    logger.info(f"Phase 1 Output Shape: {p1_output.shape}")
    assert p1_output.shape == (4, 10)

    p2_model = VisionTransformer(
        img_size=56,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        depth=4,
        num_heads=4,
        mlp_ratio=_mlp_ratio,
        num_outputs=4
    )
    p2_input = torch.randn(2, 1, 56, 56)
    p2_output = p2_model(p2_input)
    logger.info(f"Phase 2 Output Shape: {p2_output.shape}")
    assert p2_output.shape == (2, 4, 10)
    logger.info("✅ VisionTransformer tests passed.")

    # --- Test Phase 3 EncoderDecoderViT ---
    logger.info("\n--- Testing EncoderDecoderViT (Phase 3) ---")
    _B = 2
    _S_tgt = 10
    _V_dec = 13
    _img_size = 64
    _patch_size = 8
    _in_channels = 1
    _enc_D = 64
    _enc_depth = 4
    _enc_H = 4
    _dec_D = 64
    _dec_depth = 4
    _dec_H = 4

    p3_img_input = torch.randn(_B, _in_channels, _img_size, _img_size)
    p3_tgt_input = torch.randint(0, _V_dec, (_B, _S_tgt - 1))

    p3_model = EncoderDecoderViT(
        img_size=_img_size,
        patch_size=_patch_size,
        in_channels=_in_channels,
        encoder_embed_dim=_enc_D,
        encoder_depth=_enc_depth,
        encoder_num_heads=_enc_H,
        decoder_vocab_size=_V_dec,
        decoder_embed_dim=_dec_D,
        decoder_depth=_dec_depth,
        decoder_num_heads=_dec_H
    )

    try:
        output_logits = p3_model(p3_img_input, p3_tgt_input)
        logger.info(f"✅ Phase 3 Model Forward Pass Successful!")
        logger.info(
            f"Phase 3 Output logits shape: {output_logits.shape}"
        )
        assert output_logits.shape == (_B, _S_tgt - 1, _V_dec)
        logger.info("✅ EncoderDecoderViT test passed.")
    except Exception as e:
        logger.error(
            f"❌ Error during Phase 3 model test: {e}", exc_info=True
        )
