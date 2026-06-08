"""
Knowledge Encoder Implementations in PyTorch with RoPE for CLIP Pretraining

This module contains implementations of various knowledge map encoder architectures
optimized for CLIP-style contrastive learning:
1. Conv1DEncoder (Baseline) - Temporal convolutions with multi-scale pooling
2. KnowledgeViT - Vision Transformer for token-level processing
3. KnowledgePatchViT - Patch-based Vision Transformer treating (T, F) as 2D image

Key Features:
- **CLIP-Ready**: Returns both sequence and pooled representations
- **RoPE**: Uses Rotary Position Embedding for better relative position encoding
- **Flexible**: Can handle variable temporal lengths (T) and feature dimensions (F)
- **Multi-scale**: All encoders output multi-scale pooled features for robustness

Design Philosophy:
- Input format: (B, T, F) where T is temporal length, F is feature dimension
- Output format: tuple of (sequence, pooled)
  - sequence: (B, T, hidden_dim) or (B, num_patches, hidden_dim)
  - pooled: (B, hidden_dim*5) - concatenation of multi-scale temporal pooling

Input format: (batch, temporal, features)
Example input shape: (B, 30, 238)

Quick Start for CLIP:
--------------------
from knowledge_encoder import get_knowledge_encoder
import torch

# Baseline Conv1D encoder
model = get_knowledge_encoder("baseline", input_dim=238, hidden_dim=256)
x = torch.randn(2, 30, 238)
sequence, pooled = model(x)  # sequence: (2, 30, 256), pooled: (2, 1280)

# ViT encoder
model = get_knowledge_encoder("vit", input_dim=238, hidden_dim=256, depth=4)
sequence, pooled = model(x)  # sequence: (2, 30, 256), pooled: (2, 1280)

Usage examples are provided in comments after each class definition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

# Import Rotary Position Embedding from utils
try:
    from ..utils.positional_encoding import RotaryPositionEmbedding
except ImportError:
    # Fallback for different import contexts
    try:
        from utils.positional_encoding import RotaryPositionEmbedding
    except ImportError:
        # Last resort: add parent to path
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from utils.positional_encoding import RotaryPositionEmbedding

def _normalize_time_indices(indices: Optional[torch.Tensor], target_len: int) -> Optional[torch.Tensor]:
    """
    Normalize time indices to [0, 1] and resample to target_len.
    Indices < 0 are treated as padding and mapped to 0.
    """
    if indices is None:
        return None
    if indices.dim() == 1:
        indices = indices.unsqueeze(0)
    B, T = indices.shape
    indices = indices.to(dtype=torch.float32)
    normalized = torch.zeros((B, T), device=indices.device, dtype=indices.dtype)
    for b in range(B):
        valid = indices[b] >= 0
        if valid.any():
            min_val = indices[b][valid].min()
            max_val = indices[b][valid].max()
            if max_val > min_val:
                normalized[b] = (indices[b] - min_val) / (max_val - min_val)
                normalized[b][~valid] = 0.0
            else:
                normalized[b][valid] = 0.0
    if T != target_len:
        normalized = F.interpolate(
            normalized.unsqueeze(1),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
    return normalized


class DyT(nn.Module):
    """
    Dynamic Tanh-based normalization layer used as a drop-in replacement for LayerNorm.
    """
    def __init__(self, dim: int, alpha_init: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.alpha * x)
        return x * self.gamma + self.bias


# ===========================
# Helper Modules
# ===========================

class FeedForward(nn.Module):
    """GELU-based MLP for transformer blocks."""
    def __init__(self, dim, hidden_dim=None, drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiheadAttentionWithRoPE(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding (RoPE)."""
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_rope:
            self.rope = RotaryPositionEmbedding(head_dim)

        # Block-level attention cache (set on each forward, shape (B, num_heads, N, N))
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x, positions: Optional[torch.Tensor] = None):
        # x: (B, N, C) where N is sequence length
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        if self.use_rope:
            q, k = self.rope(q, k, positions)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.last_attn = attn  # store block-level attention (B, num_heads, N, N)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class KnowledgeTransformerBlock(nn.Module):
    """Standard transformer block with pre-norm and residual connections."""
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = DyT(dim)
        self.attn = MultiheadAttentionWithRoPE(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = DyT(dim)
        self.ffn = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, positions: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.ffn(self.norm2(x))
        return x


class PatchEmbed2D(nn.Module):
    """
    2D patch embedding over (T, F) treating knowledge map as a single-channel image.
    Converts (B, T, F) -> (B, num_patches, dim) via Conv2D patchify.
    """
    def __init__(self, patch_t: int, patch_f: int, dim: int):
        super().__init__()
        self.patch_t = patch_t
        self.patch_f = patch_f
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=dim,
            kernel_size=(patch_t, patch_f),
            stride=(patch_t, patch_f),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, 1, T, F)
        b, t, f = x.shape
        x_img = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.proj(x_img)    # (B, dim, T', F') where T'=T//patch_t, F'=F//patch_f
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
        return x


# ===========================
# 1. Conv1D Encoder (Baseline)
# ===========================

class Conv1DEncoder(nn.Module):
    """
    Baseline knowledge map encoder using multi-branch temporal convolutions.
    
    Architecture:
    - Linear projection to hidden_dim
    - Multi-branch temporal convolutions (kernel size 3 and 5)
    - Multi-scale pooling (small + global)
    
    Input:  (B, T, F) where T is temporal length, F is feature dimension
    Output: sequence (B, T, hidden_dim), pooled (B, hidden_dim*5)
    
    Args:
        input_dim: Input feature dimension (default: 238)
        hidden_dim: Hidden dimension for temporal features (default: 256)
    """

    def __init__(self, input_dim: int = 238, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.temporal_proj = nn.Linear(input_dim, hidden_dim)
        self.temporal_conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.temporal_conv5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, km: torch.Tensor) -> torch.Tensor:
        """
        Args:
            km: Knowledge map tensor (B, T, F)
        
        Returns:
            sequence: Temporal sequence features (B, T, hidden_dim)
        """
        # km: (B, T, F)
        x = self.temporal_proj(km)  # (B, T, hidden_dim)
        x = x.transpose(1, 2)  # (B, hidden_dim, T)

        # Multi-branch temporal convs
        branch3 = self.activation(self.bn3(self.temporal_conv3(x)))
        branch5 = self.activation(self.bn5(self.temporal_conv5(x)))
        x = (branch3 + branch5) * 0.5  # (B, hidden_dim, T)
        
        x_seq = x.transpose(1, 2)  # (B, T, hidden_dim)
        
        return x_seq  # sequence (B, T, hidden_dim)


"""
Conv1DEncoder Usage Example:
-----------------------------
import torch
from knowledge_encoder import Conv1DEncoder

# Create dummy knowledge map input: (batch, temporal, features)
km = torch.randn(2, 30, 238)

# ===== Basic Usage =====
encoder = Conv1DEncoder(input_dim=238, hidden_dim=256)
sequence, pooled = encoder(km)

print(f"Sequence shape: {sequence.shape}")  # (2, 30, 256)
print(f"Pooled shape: {pooled.shape}")      # (2, 1280) = 256 * 5

# ===== Integration with CLIP =====
# Use pooled features for contrastive learning
video_features = pooled  # (B, 1280)
# Combine with video encoder features before projection
"""


# ===========================
# 2. Knowledge ViT (Token-level Transformer)
# ===========================

class KnowledgeViT(nn.Module):
    """
    Vision Transformer (ViT) for knowledge maps with token-level processing.
    
    Paper Inspiration: "An Image is Worth 16x16 Words" (ViT)
    
    Architecture:
    - Projects each timestep's features to hidden_dim (token embedding)
    - Prepends learnable CLS token for global representation
    - Applies transformer blocks with RoPE positional encoding
    - Returns CLS token as pooled representation (standard ViT approach)
    
    Input:  (B, T, F) where T is temporal length, F is feature dimension
    Output: tuple of (sequence (B, T, hidden_dim), pooled (B, hidden_dim))
            where pooled is the CLS token output
    
    Args:
        input_dim: Input feature dimension (default: 238)
        hidden_dim: Hidden dimension for transformer (default: 256)
        depth: Number of transformer blocks (default: 4)
        num_heads: Number of attention heads (default: 4)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        drop: Dropout rate (default: 0.0)
        attn_drop: Attention dropout rate (default: 0.0)
    """
    def __init__(
        self,
        input_dim: int = 238,
        hidden_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Learnable CLS token (standard ViT approach)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            KnowledgeTransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
            )
            for _ in range(depth)
        ])

        self.norm = DyT(hidden_dim)

    def forward(self, km: torch.Tensor, km_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            km: Knowledge map tensor (B, T, F)
            km_indices: Optional (B, T) temporal indices for RoPE.

        Returns:
            sequence: (B, T, hidden_dim) — temporal tokens only (CLS is dropped).
        """
        B = km.shape[0]
        # km: (B, T, F)
        x = self.input_proj(km)  # (B, T, hidden_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, hidden_dim)

        # Transformer blocks with RoPE (aligned to temporal indices if provided)
        temporal_positions = _normalize_time_indices(km_indices, km.shape[1]) if km_indices is not None else None
        if temporal_positions is not None:
            cls_positions = torch.zeros((B, 1), device=km.device, dtype=temporal_positions.dtype)
            positions = torch.cat([cls_positions, temporal_positions], dim=1)
        else:
            positions = None
        for blk in self.blocks:
            x = blk(x, positions)

        x = self.norm(x)  # (B, T+1, hidden_dim)
        
        # Extract CLS token and sequence
        # cls_output = x[:, 0]  # (B, hidden_dim) - CLS token
        # sequence = x[:, 1:]  # (B, T, hidden_dim) - temporal tokens

        return x[:, 1:,:]

    def get_block_attention(self) -> List[Optional[torch.Tensor]]:
        """
        Return block-level attention from the last forward pass.
        Call after forward(km) so that each block's attention is populated.

        Returns:
            List of length depth; each element has shape (B, num_heads, N, N)
            where N = T+1 (CLS + temporal tokens). None for a block if not yet run.
        """
        out: List[Optional[torch.Tensor]] = []
        for blk in self.blocks:
            attn = getattr(blk.attn, "last_attn", None)
            out.append(attn)
        return out

    def get_attention_map_2d(
        self,
        T: int,
        F: int,
        use_last_layer: bool = True,
        layer_index: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Remap block-level attention to (B, T, F) for visualization.
        Call after forward(km) so that block attention is populated.

        Uses incoming attention to each temporal token (how much each timestep
        was attended to). If attention has CLS (shape T+1), uses CLS→temporal;
        otherwise uses mean incoming to each token. Result is broadcast over F
        to match KM shape.

        Args:
            T: Temporal length of the original knowledge map.
            F: Feature dimension of the original knowledge map.
            use_last_layer: If True, use only layer_index; else average over all layers.
            layer_index: Which block to use when use_last_layer is True.

        Returns:
            Tensor (B, T, F) on CPU, float, or None if no attention available.
        """
        block_attns = self.get_block_attention()
        valid = [a for a in block_attns if a is not None]
        if not valid:
            return None
        if use_last_layer:
            attn = valid[layer_index]  # (B, H, N, N)
        else:
            attn = torch.stack(valid, dim=0).mean(dim=0)  # (B, H, N, N)
        N = attn.shape[-1]
        if N == T + 1:
            # CLS + temporal: use CLS→temporal
            cls_to_tokens = attn[:, :, 0, 1:]  # (B, num_heads, T)
            temporal_attn = cls_to_tokens.mean(dim=1)  # (B, T)
        elif N == T:
            # Temporal only: use mean incoming to each token
            temporal_attn = attn.mean(dim=2).mean(dim=1)  # (B, T)
        else:
            return None
        B = temporal_attn.shape[0]
        return temporal_attn.unsqueeze(-1).expand(B, T, F)  # (B, T, F)


# ===========================
# 3. Knowledge Patch ViT (2D Patch Transformer)
# ===========================

class KnowledgePatchViT(nn.Module):
    """
    Patch-based Vision Transformer (Patch ViT) for knowledge maps.
    
    Paper Inspiration: "An Image is Worth 16x16 Words" (ViT)
    
    Architecture:
    - Treats (T, F) as a 2D image (B, 1, T, F)
    - Divides into non-overlapping patches via Conv2D
    - Applies transformer blocks with RoPE over patch sequence
    - Multi-scale pooling for final representation
    
    Input:  (B, T, F)
    Output: sequence (B, num_patches, hidden_dim), pooled (B, hidden_dim*5)
    
    Note: T must be divisible by patch_t, F must be divisible by patch_f
    
    Args:
        input_dim: Input feature dimension (default: 238)
        hidden_dim: Hidden dimension for transformer (default: 256)
        depth: Number of transformer blocks (default: 4)
        num_heads: Number of attention heads (default: 4)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        drop: Dropout rate (default: 0.0)
        attn_drop: Attention dropout rate (default: 0.0)
        patch_t: Temporal patch size (default: 4)
        patch_f: Feature patch size (default: 8)
    """
    def __init__(
        self,
        input_dim: int = 238,
        hidden_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        patch_t: int = 4,
        patch_f: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.patch_t = patch_t
        self.patch_f = patch_f
        
        # Note: input_dim is used as F dimension for patch embedding
        self.patch_embed = PatchEmbed2D(patch_t=patch_t, patch_f=patch_f, dim=hidden_dim)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            KnowledgeTransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
            )
            for _ in range(depth)
        ])

        self.norm = DyT(hidden_dim)

    def forward(self, km: torch.Tensor, km_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            km: Knowledge map tensor (B, T, F)
        
        Returns:
            sequence: Patch transformer output (B, num_patches, hidden_dim)
        """
        # km: (B, T, F)
        x = self.patch_embed(km)  # (B, num_patches, hidden_dim), where num_patches = (T//patch_t)*(F//patch_f)

        # Transformer blocks with RoPE over patches (temporal alignment if provided)
        if km_indices is not None:
            temporal_patches = max(1, km.shape[1] // self.patch_t)
            feature_patches = max(1, km.shape[2] // self.patch_f)
            temporal_positions = _normalize_time_indices(km_indices, temporal_patches)
            positions = temporal_positions.repeat_interleave(feature_patches, dim=1)
        else:
            positions = None
        for blk in self.blocks:
            x = blk(x, positions)

        x = self.norm(x)  # (B, num_patches, hidden_dim)

        return x[:, 1:,:]  # sequence (B, num_patches, hidden_dim)

    def get_block_attention(self) -> List[Optional[torch.Tensor]]:
        """
        Return block-level attention from the last forward pass.
        Call after forward(km) so that each block's attention is populated.

        Returns:
            List of length depth; each element has shape (B, num_heads, N, N)
            where N = num_patches. None for a block if not yet run.
        """
        out: List[Optional[torch.Tensor]] = []
        for blk in self.blocks:
            attn = getattr(blk.attn, "last_attn", None)
            out.append(attn)
        return out

    def get_attention_map_2d(
        self,
        T: int,
        F: int,
        use_last_layer: bool = True,
        layer_index: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Remap patch-level attention to (B, T, F) for visualization (ViT-style).
        Call after forward(km) so that block attention is populated.

        Uses patch index to align with raw (T, F) layout:
        - PatchEmbed2D outputs patches in row-major order: patch index
          p = t_idx * F_p + f_idx, where (t_idx, f_idx) is the 2D patch index.
        - Patch (t_idx, f_idx) covers raw region
          [t_idx*patch_t : (t_idx+1)*patch_t, f_idx*patch_f : (f_idx+1)*patch_f].
        We compute per-patch importance (incoming attention), reshape using the
        same p <-> (t_idx, f_idx) mapping, then bilinearly interpolate to (T, F).

        Args:
            T: Temporal length of the original knowledge map.
            F: Feature dimension of the original knowledge map.
            use_last_layer: If True, use only layer_index; else average over all layers.
            layer_index: Which block to use when use_last_layer is True.

        Returns:
            Tensor (B, T, F), or None if no attention available.
        """
        block_attns = self.get_block_attention()
        valid = [a for a in block_attns if a is not None]
        if not valid:
            return None
        if use_last_layer:
            attn = valid[layer_index]  # (B, num_heads, N, N)
        else:
            attn = torch.stack(valid, dim=0).mean(dim=0)  # (B, num_heads, N, N)
        # Per-patch importance: mean attention received (incoming)
        # attn (B, H, N, N): for each key j, mean over query i and heads -> (B, N)
        patch_importance = attn.mean(dim=2).mean(dim=1)  # (B, N)
        T_p = T // self.patch_t
        F_p = F // self.patch_f
        N = T_p * F_p
        if patch_importance.shape[1] != N:
            return None
        # Remap patch index p back to 2D: p <-> (t_idx, f_idx) with p = t_idx*F_p + f_idx
        # (same layout as PatchEmbed2D.flatten(2): (B, dim, T_p, F_p) -> last dim varies fastest)
        grid = patch_importance.reshape(-1, T_p, F_p)  # (B, T_p, F_p); grid[b,t_idx,f_idx] = patch_importance[b, t_idx*F_p + f_idx]
        # Upsample patch grid to raw (T, F) for visualization
        grid_4d = grid.unsqueeze(1)  # (B, 1, T_p, F_p)
        out = F.interpolate(
            grid_4d,
            size=(T, F),
            mode="bilinear",
            align_corners=False,
        )
        return out.squeeze(1)  # (B, T, F)


"""
KnowledgePatchViT Usage Example:
---------------------------------
import torch
from knowledge_encoder import KnowledgePatchViT

# Create dummy knowledge map input: (batch, temporal, features)
# Note: T and F must be divisible by patch_t and patch_f
km = torch.randn(2, 32, 240)  # T=32 divisible by 4, F=240 divisible by 8

# ===== Basic Usage =====
encoder = KnowledgePatchViT(
    input_dim=240,
    hidden_dim=256,
    depth=4,
    num_heads=4,
    patch_t=4,
    patch_f=8
)
sequence, pooled = encoder(km)

# num_patches = (32/4) * (240/8) = 8 * 30 = 240
print(f"Sequence shape: {sequence.shape}")  # (2, 240, 256)
print(f"Pooled shape: {pooled.shape}")      # (2, 1280) = 256 * 5

# ===== Larger Patches (fewer tokens, faster) =====
encoder_fast = KnowledgePatchViT(
    input_dim=240,
    hidden_dim=256,
    depth=4,
    num_heads=4,
    patch_t=8,
    patch_f=16
)

# ===== Smaller Patches (more tokens, more detail) =====
encoder_detailed = KnowledgePatchViT(
    input_dim=240,
    hidden_dim=256,
    depth=4,
    num_heads=4,
    patch_t=2,
    patch_f=4
)
"""


# ===========================
# Backward Compatibility Aliases
# ===========================

# Legacy names from previous implementation
KnowledgeMapEncoder = Conv1DEncoder  # Original baseline name
Conv1dEncoder = Conv1DEncoder  # Alternate capitalization
KnowledgeTransformerEncoder = KnowledgeViT  # Legacy ViT name
KnowledgePatchEncoder = KnowledgePatchViT  # Legacy Patch ViT name


# ===========================
# Factory Function
# ===========================

def get_knowledge_encoder(encoder_type: str = "baseline", **kwargs) -> nn.Module:
    """
    Factory function to build knowledge encoders by type.
    
    Args:
        encoder_type: One of ["baseline", "conv1d", "vit", "transformer", "patch_vit", "patch"]
            - "baseline" / "conv1d": Conv1D-based encoder (Conv1DEncoder)
            - "vit" / "transformer": Token-level ViT (KnowledgeViT)
            - "patch_vit" / "patch": Patch-based ViT (KnowledgePatchViT)
        **kwargs: Additional arguments passed to the encoder constructor
            Common kwargs:
                - input_dim: Input feature dimension (default: 238)
                - hidden_dim: Hidden dimension (default: 256)
            ViT-specific kwargs:
                - depth: Number of transformer blocks (default: 4)
                - num_heads: Number of attention heads (default: 4)
                - mlp_ratio: MLP expansion ratio (default: 4.0)
                - drop: Dropout rate (default: 0.0)
                - attn_drop: Attention dropout rate (default: 0.0)
            Patch ViT-specific kwargs:
                - patch_t: Temporal patch size (default: 4)
                - patch_f: Feature patch size (default: 8)
    
    Returns:
        Knowledge encoder module
    
    Example:
        >>> encoder = get_knowledge_encoder("vit", input_dim=238, hidden_dim=256, depth=4)
        >>> km = torch.randn(2, 30, 238)
        >>> sequence, pooled = encoder(km)
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type in ("baseline", "conv", "conv1d"):
        return Conv1DEncoder(**kwargs)
    elif encoder_type in ("vit", "transformer"):
        return KnowledgeViT(**kwargs)
    elif encoder_type in ("patch_vit", "patch", "patch_transformer"):
        return KnowledgePatchViT(**kwargs)
    else:
        raise ValueError(
            f"Unknown encoder_type '{encoder_type}'. "
            f"Choose from: baseline, conv1d, vit, transformer, patch_vit, patch"
        )


"""
Factory Function Usage Example:
--------------------------------
import torch
from knowledge_encoder import get_knowledge_encoder

km = torch.randn(2, 30, 238)

# ===== Baseline Conv1D Encoder =====
encoder_baseline = get_knowledge_encoder("baseline", input_dim=238, hidden_dim=256)
seq, pooled = encoder_baseline(km)
print(f"Baseline - Sequence: {seq.shape}, Pooled: {pooled.shape}")

# ===== ViT Encoder =====
encoder_vit = get_knowledge_encoder("vit", input_dim=238, hidden_dim=256, depth=4, num_heads=4)
seq, pooled = encoder_vit(km)
print(f"ViT - Sequence: {seq.shape}, Pooled: {pooled.shape}")

# ===== Patch ViT Encoder =====
# Note: Use input with dimensions divisible by patch sizes
km_patch = torch.randn(2, 32, 240)  # T=32 (div by 4), F=240 (div by 8)
encoder_patch = get_knowledge_encoder("patch_vit", input_dim=240, hidden_dim=256, 
                                      depth=4, num_heads=4, patch_t=4, patch_f=8)
seq, pooled = encoder_patch(km_patch)
print(f"Patch ViT - Sequence: {seq.shape}, Pooled: {pooled.shape}")
"""


# ===========================
# Utility Functions
# ===========================

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


"""
Complete Usage Example with All Encoders:
------------------------------------------
import torch
from knowledge_encoder import get_knowledge_encoder, count_parameters

# Input shape: (batch, temporal, features)
km = torch.randn(2, 30, 238)

print("=" * 70)
print("Knowledge Encoder Comparison")
print("=" * 70)

# 1. Baseline Conv1D Encoder
print("\n1. Conv1D Encoder (Baseline)")
encoder = get_knowledge_encoder("baseline", input_dim=238, hidden_dim=256)
seq, pooled = encoder(km)
print(f"   Sequence: {seq.shape}, Pooled: {pooled.shape}")
print(f"   Parameters: {count_parameters(encoder):,}")

# 2. Knowledge ViT
print("\n2. Knowledge ViT (Token-level Transformer)")
encoder = get_knowledge_encoder("vit", input_dim=238, hidden_dim=256, depth=4, num_heads=4)
seq, pooled = encoder(km)
print(f"   Sequence: {seq.shape}, Pooled: {pooled.shape}")
print(f"   Parameters: {count_parameters(encoder):,}")

# 3. Knowledge Patch ViT
print("\n3. Knowledge Patch ViT (2D Patch Transformer)")
km_patch = torch.randn(2, 32, 240)  # Divisible dimensions
encoder = get_knowledge_encoder("patch_vit", input_dim=240, hidden_dim=256,
                               depth=4, num_heads=4, patch_t=4, patch_f=8)
seq, pooled = encoder(km_patch)
print(f"   Sequence: {seq.shape}, Pooled: {pooled.shape}")
print(f"   Parameters: {count_parameters(encoder):,}")

print("\\n" + "=" * 70)
print("RoPE Benefits")
print("=" * 70)
benefits = '''
1. Relative Position Encoding: Naturally encodes relative positions
2. No Learned Parameters: Reduces model size compared to absolute embeddings
3. Better Extrapolation: Handles variable-length sequences better
4. Computational Efficiency: Applied during attention without lookups
5. Translation Invariance: Built-in translation properties
'''
print(benefits)
"""


# ===========================
# Exports
# ===========================

__all__ = [
    # Main encoders
    "Conv1DEncoder",
    "KnowledgeViT",
    "KnowledgePatchViT",
    
    # Backward compatibility aliases
    "KnowledgeMapEncoder",  # original baseline name (alias for Conv1DEncoder)
    "Conv1dEncoder",  # alternate capitalization
    "KnowledgeTransformerEncoder",  # alias for KnowledgeViT
    "KnowledgePatchEncoder",  # alias for KnowledgePatchViT
    
    # Factory function
    "get_knowledge_encoder",
    
    # Utilities
    "count_parameters",
    
    # Helper modules (for advanced users)
    "FeedForward",
    "MultiheadAttentionWithRoPE",
    "KnowledgeTransformerBlock",
    "PatchEmbed2D",
]
