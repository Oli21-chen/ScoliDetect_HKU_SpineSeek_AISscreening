"""
Video Encoder Implementations in PyTorch with RoPE for CLIP Pretraining

This module contains implementations of various video encoder architectures
optimized for CLIP-style contrastive learning:
0. Conv3DEncoder
1. ViVit (Video Vision Transformer)
2. TimeSformer
3. Video Swin Transformer
4. Multiscale Vision Transformer (MViT)
5. Uniformer

Key Features:
- **CLIP-Ready**: Default configuration returns patch embeddings (num_classes=0)
- **RoPE**: Uses Rotary Position Embedding for better relative position encoding
- **Flexible**: Can add classification head for supervised fine-tuning (num_classes > 0)
- **Patch Tokens**: Returns all patch tokens (excluding CLS token) for CLIP

Design Philosophy:
- Default (num_classes=0): Returns (B, num_patches, embed_dim) for contrastive learning
- Optional (num_classes>0): Returns (B, num_classes) for supervised classification
- Output is from patch tokens, not CLS token, providing richer spatial information

Input format: (batch, temporal, height, width, channel)
Example input shape: (B, 4, 256, 256, 3)

Quick Start for CLIP:
--------------------
from video_encoders import ViVit
import torch

model = ViVit(embed_dim=768, depth=12)  # num_classes=0 by default
x = torch.randn(2, 4, 256, 256, 3)
patch_embeddings = model(x)  # (2, num_patches, 768) - ready for projection & contrastive loss

Usage examples are provided in comments after each class definition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import math
from typing import Optional, Tuple

# Torchvision video CNN backbones (ResNet3D family).
# These are standard baselines (R3D / R(2+1)D) commonly used in video understanding.
try:
    import torchvision.models.video as tv_video
except Exception:
    tv_video = None

# Import RoPE from utils
try:
    from ..utils.positional_encoding import RotaryPositionEmbedding
except ImportError:
    # Fallback for different import contexts
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


class Conv3DEncoder(nn.Module):
    """
    Lightweight video encoder that preserves temporal alignment.
    Expects input (B, T, H, W, C) and keeps the temporal dimension,
    projecting per-timestep embeddings so they can be aligned with knowledge map timesteps.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, hidden_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # video: (B, T, H, W, C) -> (B, C, T, H, W)
        B, T, H, W, C = video.shape
        x = video.permute(0, 4, 1, 2, 3).contiguous()
        x = self.stem(x)

        # Spatial pooling while keeping time
        x = x.mean(dim=[3, 4])  # (B, hidden_dim, T)

        x_seq = x.transpose(1, 2)  # (B, T, hidden_dim)
        
        # Mean pooling and L2 normalization
        pooled = x.mean(dim=-1)  # (B, hidden_dim) - mean over temporal dimension
        pooled = F.normalize(pooled, p=2, dim=-1)  # L2 normalization
        
        return x_seq, pooled  # sequence (B, T, hidden_dim), normalized pooled (B, hidden_dim)


class TorchvisionVideoResNetEncoder(nn.Module):
    """
    Wrapper for torchvision 3D ResNet backbones (e.g., r3d_18, r2plus1d_18).

    Returns:
      - sequence: (B, T', hidden_dim) temporal-aligned features (spatially pooled)
      - pooled:   (B, hidden_dim) mean over time (L2-normalized)
    """

    _SUPPORTED = ("r3d_18", "r2plus1d_18")

    def __init__(
        self,
        backbone: str = "r3d_18",
        hidden_dim: int = 512,
        pretrained: bool = False,
        spatial_pool: str = "mean",
    ):
        super().__init__()
        if tv_video is None:
            raise ImportError(
                "torchvision.models.video is not available. "
                "Install torchvision with video ops support, or use Conv3DEncoder."
            )
        if backbone not in self._SUPPORTED:
            raise ValueError(f"Unsupported backbone '{backbone}'. Choose from: {', '.join(self._SUPPORTED)}")
        if spatial_pool not in ("mean",):
            raise ValueError("Only spatial_pool='mean' is supported.")

        # Note: torchvision's r3d_18 / r2plus1d_18 output 512 channels in the final stage.
        weights = None
        if pretrained:
            # Torchvision version differences: prefer explicit weights if available.
            if backbone == "r3d_18" and hasattr(tv_video, "R3D_18_Weights"):
                weights = tv_video.R3D_18_Weights.DEFAULT
            elif backbone == "r2plus1d_18" and hasattr(tv_video, "R2Plus1D_18_Weights"):
                weights = tv_video.R2Plus1D_18_Weights.DEFAULT

        if backbone == "r3d_18":
            model = tv_video.r3d_18(weights=weights)
        else:
            model = tv_video.r2plus1d_18(weights=weights)

        # Keep only the feature extractor part (drop avgpool/fc).
        self.backbone = model
        self.hidden_dim = hidden_dim
        self.backbone_name = backbone
        self.spatial_pool = spatial_pool

        self.backbone_out_dim = 512
        self.proj = nn.Identity() if self.backbone_out_dim == hidden_dim else nn.Linear(self.backbone_out_dim, hidden_dim)

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input: (B, T, H, W, C) -> (B, C, T, H, W)
        x = video.permute(0, 4, 1, 2, 3).contiguous()

        # Torchvision VideoResNet forward without classification head:
        # stem -> layer1..4 -> (B, 512, T', H', W')
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Spatial pool to keep temporal alignment
        x = x.mean(dim=[3, 4])  # (B, 512, T')
        x = x.transpose(1, 2).contiguous()  # (B, T', 512)
        x_seq = self.proj(x)  # (B, T', hidden_dim)

        pooled = x_seq.mean(dim=1)  # (B, hidden_dim)
        pooled = F.normalize(pooled, p=2, dim=-1)
        return x_seq, pooled


class _BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = None
        if stride != (1, 1, 1) or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet3DEncoder(nn.Module):
    """
    ResNet3D-18 style backbone with configurable width (base_channels).

    This is a \"well-designed\" 3D CNN baseline (ResNet family) while allowing
    parameter-count matching vs transformers by shrinking width.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        base_channels: int = 64,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_channels = base_channels

        # Stem similar to common video ResNet: spatial downsample early, keep temporal stride 1.
        self.stem = nn.Sequential(
            nn.Conv3d(3, base_channels, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        self.in_planes = base_channels
        self.layer1 = self._make_layer(base_channels * 1, layers[0], stride=(1, 1, 1))
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=(2, 2, 2))
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=(2, 2, 2))
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=(2, 2, 2))

        self.backbone_out_dim = base_channels * 8
        self.proj = nn.Identity() if self.backbone_out_dim == hidden_dim else nn.Linear(self.backbone_out_dim, hidden_dim)

    def _make_layer(self, planes: int, blocks: int, stride: Tuple[int, int, int]) -> nn.Sequential:
        layers = []
        layers.append(_BasicBlock3D(self.in_planes, planes, stride=stride))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock3D(self.in_planes, planes, stride=(1, 1, 1)))
        return nn.Sequential(*layers)

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = video.permute(0, 4, 1, 2, 3).contiguous()  # (B,C,T,H,W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, C_out, T', H', W')

        x = x.mean(dim=[3, 4])  # (B, C_out, T')
        x = x.transpose(1, 2).contiguous()  # (B, T', C_out)
        x_seq = self.proj(x)  # (B, T', hidden_dim)

        pooled = x_seq.mean(dim=1)
        pooled = F.normalize(pooled, p=2, dim=-1)
        return x_seq, pooled


# ===========================
# 1. ViVit (Video Vision Transformer)
# ===========================

class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, temporal_size=8, temporal_patch_size=2, 
                 in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.temporal_size = temporal_size
        self.temporal_patch_size = temporal_patch_size
        
        self.num_patches = (temporal_size // temporal_patch_size) * \
                          (img_size // patch_size) * (img_size // patch_size)
        
        self.proj = nn.Conv3d(in_channels, embed_dim, 
                             kernel_size=(temporal_patch_size, patch_size, patch_size),
                             stride=(temporal_patch_size, patch_size, patch_size))
    
    def forward(self, x):
        # x: (B, T, H, W, C) -> (B, C, T, H, W)
        x = rearrange(x, 'b t h w c -> b c t h w')
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_rope = use_rope
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # RoPE
        if use_rope:
            self.rope = RotaryPositionEmbedding(head_dim)
    
    def forward(self, x, positions=None):
        """
        Args:
            x: (B, N, C)
            positions: Optional position indices for RoPE (B, N)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # Apply RoPE to q and k
        if self.use_rope:
            q, k = self.rope(q, k, positions)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., use_rope=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                       attn_drop=attn_drop, proj_drop=drop, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x, positions=None):
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.mlp(self.norm2(x))
        return x


class ViVit(nn.Module):
    """
    Video Vision Transformer (ViVit)
    
    Paper: "ViViT: A Video Vision Transformer" (https://arxiv.org/abs/2103.15691)
    Model: Factorized Encoder version (spatial then temporal)
    """
    def __init__(self, img_size=256, patch_size=16, temporal_size=4, temporal_patch_size=1,
                 in_channels=3, num_classes=0, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        """
        Args:
            num_classes: Number of classes for classification.
                        If 0 (default), returns patch token embeddings for CLIP.
                        If > 0, adds classification head for supervised learning.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, 
            patch_size=patch_size,
            temporal_size=temporal_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        
        # CLS token (RoPE replaces positional embedding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                use_rope=True
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, video_indices: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, T, H, W, C) tensor
        Returns:
            - If num_classes = 0: (B, num_patches, embed_dim) patch embeddings for CLIP
            - If num_classes > 0: (B, num_classes) classification logits
        """
        B, T, H, W, _ = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = self.pos_drop(x)
        
        # Create RoPE positions aligned to temporal indices (CLS token at position 0)
        temporal_patch_size = self.patch_embed.temporal_patch_size
        temporal_patches = max(1, T // temporal_patch_size)
        spatial_patches = (H // self.patch_embed.patch_size) * (W // self.patch_embed.patch_size)
        if video_indices is not None:
            temporal_positions = _normalize_time_indices(video_indices, temporal_patches)
        else:
            temporal_positions = torch.linspace(0, 1, temporal_patches, device=x.device).unsqueeze(0).expand(B, -1)
        positions = temporal_positions.repeat_interleave(spatial_patches, dim=1)
        cls_positions = torch.zeros((B, 1), device=x.device, dtype=positions.dtype)
        positions = torch.cat([cls_positions, positions], dim=1)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x, positions)
        
        x = self.norm(x)
       
        return x[:, 1:, :]  # (B, num_patches, embed_dim)
        


# ===========================
# 2. TimeSformer
# ===========================

class DividedSpaceTimeAttention(nn.Module):
    """Divided Space-Time Attention for TimeSformer"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.temporal_attn = MultiheadAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop, use_rope=True)
        self.spatial_attn = MultiheadAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop, use_rope=True)
    
    def forward(self, x, num_temporal_patches, num_spatial_patches, temporal_positions: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, 1 + T*S, C) where 1 is CLS token, T is temporal, S is spatial
            num_temporal_patches: T
            num_spatial_patches: S (H*W patches)
        """
        B, N, C = x.shape
        
        # Temporal attention
        cls_token = x[:, 0:1, :]  # (B, 1, C)
        x_temporal = x[:, 1:, :]  # (B, T*S, C)
        
        # Reshape for temporal attention: process each spatial location across time
        x_temporal = rearrange(x_temporal, 'b (t s) c -> (b s) t c', 
                              t=num_temporal_patches, s=num_spatial_patches)
        cls_token_temporal = cls_token.repeat(num_spatial_patches, 1, 1)
        x_temporal = torch.cat([cls_token_temporal, x_temporal], dim=1)
        
        # Create temporal position indices (CLS token at position 0)
        if temporal_positions is None:
            temporal_positions = torch.linspace(
                0, 1, num_temporal_patches, device=x_temporal.device
            ).unsqueeze(0).expand(B, -1)
        cls_temporal = torch.zeros((temporal_positions.shape[0], 1), device=x_temporal.device, dtype=temporal_positions.dtype)
        temporal_positions = torch.cat([cls_temporal, temporal_positions], dim=1)
        temporal_positions = temporal_positions.repeat_interleave(num_spatial_patches, dim=0)
        
        x_temporal = self.temporal_attn(x_temporal, temporal_positions)
        
        x_temporal = x_temporal[:, 1:, :]  # Remove CLS
        x_temporal = rearrange(x_temporal, '(b s) t c -> b (t s) c', 
                              b=B, s=num_spatial_patches)
        
        # Spatial attention
        x_spatial = rearrange(x_temporal, 'b (t s) c -> (b t) s c',
                            t=num_temporal_patches, s=num_spatial_patches)
        cls_token_spatial = cls_token.repeat(num_temporal_patches, 1, 1)
        x_spatial = torch.cat([cls_token_spatial, x_spatial], dim=1)
        
        # Create spatial position indices
        spatial_positions = torch.arange(num_spatial_patches + 1, device=x_spatial.device)
        spatial_positions = spatial_positions.unsqueeze(0).expand(x_spatial.shape[0], -1)
        
        x_spatial = self.spatial_attn(x_spatial, spatial_positions)
        
        # Average CLS token across time dimension (keeping batch dimension)
        cls_token = x_spatial[:, 0:1, :]  # (B*T, 1, C)
        cls_token = rearrange(cls_token, '(b t) n c -> b t n c', b=B, t=num_temporal_patches)
        cls_token = cls_token.mean(dim=1)  # (B, 1, C) - average over time
        
        x_spatial = x_spatial[:, 1:, :]
        x_spatial = rearrange(x_spatial, '(b t) s c -> b (t s) c',
                            b=B, t=num_temporal_patches)
        
        x = torch.cat([cls_token, x_spatial], dim=1)
        
        return x


class TimeSformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DividedSpaceTimeAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                             attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x, num_temporal_patches, num_spatial_patches, temporal_positions: Optional[torch.Tensor] = None):
        x = self.attn(self.norm1(x), num_temporal_patches, num_spatial_patches, temporal_positions)
        x = x + self.mlp(self.norm2(x))
        return x


class TimeSformer(nn.Module):
    """
    TimeSformer: Is Space-Time Attention All You Need for Video Understanding?
    
    Paper: https://arxiv.org/abs/2102.05095
    Uses divided space-time attention
    """
    def __init__(self, img_size=256, patch_size=16, temporal_size=4,
                 in_channels=3, num_classes=0, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        """
        Args:
            num_classes: Number of classes for classification.
                        If 0 (default), returns patch token embeddings for CLIP.
                        If > 0, adds classification head for supervised learning.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.temporal_size = temporal_size
        
        # Calculate number of patches
        self.num_spatial_patches = (img_size // patch_size) ** 2
        self.num_temporal_patches = temporal_size
        num_patches = self.num_temporal_patches * self.num_spatial_patches
        
        # Patch embedding (2D patches across frames)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                    kernel_size=patch_size, stride=patch_size)
        
        # CLS token (RoPE replaces positional embedding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TimeSformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Optional classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else None
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, video_indices: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, T, H, W, C) tensor
        Returns:
            - If num_classes = 0: (B, num_patches, embed_dim) patch embeddings for CLIP
            - If num_classes > 0: (B, num_classes) classification logits
        """
        B, T, H, W, C = x.shape
        
        # Process each frame
        x = rearrange(x, 'b t h w c -> (b t) c h w')
        x = self.patch_embed(x)  # (B*T, embed_dim, H', W')
        x = rearrange(x, '(b t) c h w -> b (t h w) c', b=B, t=T)
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = self.pos_drop(x)
        
        # Temporal positions aligned to indices (if provided)
        temporal_positions = _normalize_time_indices(video_indices, self.num_temporal_patches) if video_indices is not None else None

        # Transformer blocks with divided space-time attention (RoPE applied inside)
        for blk in self.blocks:
            x = blk(x, self.num_temporal_patches, self.num_spatial_patches, temporal_positions)
        
        x = self.norm(x)
        
        if self.num_classes == 0:
            # For CLIP: return patch tokens (exclude CLS token)
            return x[:, 1:, :]  # (B, num_patches, embed_dim)
        else:
            # For classification: use CLS token
            x = x[:, 0]  # Take CLS token
            x = self.head(x)
            return x


"""
TimeSformer Usage Example:
--------------------------
import torch
from video_encoders import TimeSformer

# Create dummy input: (batch, temporal, height, width, channel)
x = torch.randn(2, 4, 256, 256, 3)
    
# ===== CLIP Pretraining (default) =====
model = TimeSformer(
    img_size=256,
    patch_size=16,
    temporal_size=4,
    in_channels=3,
    embed_dim=768,
    depth=12,
    num_heads=12
)

patch_embeddings = model(x)  # (2, num_patches, 768)
# num_patches = 4 * (256/16)^2 = 4 * 256 = 1024
print(f"Patch embeddings shape: {patch_embeddings.shape}")
    
# ===== Supervised Fine-tuning =====
classifier = TimeSformer(
    img_size=256,
    patch_size=16,
    temporal_size=4,
    in_channels=3,
    num_classes=10,
    embed_dim=768,
    depth=12,
    num_heads=12
)

logits = classifier(x)  # (2, 10)
print(f"Logits shape: {logits.shape}")
"""


# ===========================
# 3. Video Swin Transformer
# ===========================

class WindowAttention3D(nn.Module):
    """Window based multi-head self attention for 3D video with RoPE"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (T, H, W)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # RoPE for relative position encoding
        self.rope = RotaryPositionEmbedding(head_dim)
    
    def forward(self, x):
        """
        Args:
            x: (num_windows*B, window_size*window_size*window_size, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, num_heads, N, head_dim)
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition3d(x, window_size):
    """
    Args:
        x: (B, T, H, W, C)
        window_size: (T, H, W)
    Returns:
        windows: (num_windows*B, window_size[0]*window_size[1]*window_size[2], C)
    """
    B, T, H, W, C = x.shape
    wt, wh, ww = window_size
    
    # Check if dimensions are divisible by window size
    assert T % wt == 0, f"Temporal dimension {T} not divisible by window size {wt}"
    assert H % wh == 0, f"Height {H} not divisible by window size {wh}"
    assert W % ww == 0, f"Width {W} not divisible by window size {ww}"
    
    x = x.view(B, T // wt, wt, H // wh, wh, W // ww, ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, wt * wh * ww, C)
    return windows


def window_reverse3d(windows, window_size, T, H, W):
    """
    Args:
        windows: (num_windows*B, window_size[0]*window_size[1]*window_size[2], C)
        window_size: (T, H, W)
        T, H, W: original video dimensions
    Returns:
        x: (B, T, H, W, C)
    """
    wt, wh, ww = window_size
    B = int(windows.shape[0] / (T * H * W / wt / wh / ww))
    x = windows.view(B, T // wt, H // wh, W // ww, wt, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
    return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block for 3D video"""
    def __init__(self, dim, num_heads, window_size=(2, 8, 8), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        
        # Cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                  dims=(1, 2, 3))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition3d(shifted_x, self.window_size)
        
        # Window attention
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        shifted_x = window_reverse3d(attn_windows, self.window_size, T, H, W)
        
        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                          dims=(1, 2, 3))
        else:
            x = shifted_x
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x


class VideoSwinTransformer(nn.Module):
    """
    Video Swin Transformer
    
    Paper: "Video Swin Transformer" (https://arxiv.org/abs/2106.13230)
    
    Important: window_size must divide evenly into the patch dimensions!
    After patch embedding, dimensions become:
        T' = temporal_size // temporal_patch_size
        H' = img_size // patch_size
        W' = img_size // patch_size
    
    window_size (wt, wh, ww) must satisfy:
        T' % wt == 0, H' % wh == 0, W' % ww == 0
    
    Example: img_size=256, patch_size=4 -> H'=W'=64
             Valid window sizes: (2, 8, 8), (2, 4, 4), (2, 16, 16), (2, 32, 32)
             Invalid: (2, 7, 7) because 64 % 7 != 0
    """
    def __init__(self, img_size=256, patch_size=4, temporal_size=4, temporal_patch_size=1,
                 in_channels=3, num_classes=0, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=(2, 8, 8), mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        """
    Args:
            num_classes: Number of classes for classification.
                        If 0 (default), returns spatial patch embeddings for CLIP.
                        If > 0, adds classification head for supervised learning.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size)
        )
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock3D(
                    dim=int(embed_dim * 2 ** i_layer),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate
                )
                for i in range(depths[i_layer])
            ])
            self.layers.append(layer)
            
            # Patch merging
            if i_layer < self.num_layers - 1:
                self.layers.append(nn.Conv3d(
                    int(embed_dim * 2 ** i_layer),
                    int(embed_dim * 2 ** (i_layer + 1)),
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2)
                ))
        
        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Optional classification head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else None
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, H, W, C) tensor
        Returns:
            - If num_classes = 0: (B, num_patches, num_features) spatial embeddings for CLIP
            - If num_classes > 0: (B, num_classes) classification logits
        """
        # Convert to (B, C, T, H, W)
        x = rearrange(x, 'b t h w c -> b c t h w')
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, T', H', W')
        x = self.pos_drop(x)
        
        # Convert to (B, T', H', W', embed_dim)
        x = rearrange(x, 'b c t h w -> b t h w c')
        
        # Stages
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.ModuleList):
                # Transformer blocks
                for blk in layer:
                    x = blk(x)
            else:
                # Patch merging (downsampling)
                x = rearrange(x, 'b t h w c -> b c t h w')
                x = layer(x)
                x = rearrange(x, 'b c t h w -> b t h w c')
        
        x = self.norm(x)  # (B, T', H', W', C)
        
        if self.num_classes == 0:
            # For CLIP: return all spatial tokens
            B, T, H, W, C = x.shape
            x = rearrange(x, 'b t h w c -> b (t h w) c')  # (B, num_patches, C)
            return x
        else:
            # For classification: global average pooling
            x = rearrange(x, 'b t h w c -> b c (t h w)')
            x = self.avgpool(x)  # (B, C, 1)
            x = x.squeeze(-1)  # (B, C)
            x = self.head(x)
            return x


"""
Video Swin Transformer Usage Example:
-------------------------------------
import torch
from video_encoders import VideoSwinTransformer

# Create dummy input: (batch, temporal, height, width, channel)
x = torch.randn(2, 4, 256, 256, 3)

# ===== CLIP Pretraining (default) =====
model = VideoSwinTransformer(
    img_size=256,
    patch_size=4,
    temporal_size=4,
    in_channels=3,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=(2, 8, 8)
)

spatial_embeddings = model(x)  # (2, num_patches, 768)
# Output dim = 96 * 2^3 = 768 (from 4 stages)
print(f"Spatial embeddings shape: {spatial_embeddings.shape}")

# ===== Supervised Fine-tuning =====
classifier = VideoSwinTransformer(
    img_size=256,
    patch_size=4,
    temporal_size=4,
    in_channels=3,
    num_classes=10,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24]
)

logits = classifier(x)  # (2, 10)
"""


# ===========================
# 4. Multiscale Vision Transformer (MViT)
# ===========================

class MultiScaleAttention(nn.Module):
    """Multi-scale attention with pooling and RoPE"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 kernel_q=(1, 1, 1), kernel_kv=(1, 1, 1), stride_q=(1, 1, 1), stride_kv=(1, 1, 1)):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Pooling layers for Q, K, V
        self.pool_q = nn.Conv3d(dim, dim, kernel_q, stride=stride_q, padding=tuple(k//2 for k in kernel_q), groups=dim)
        self.pool_k = nn.Conv3d(dim, dim, kernel_kv, stride=stride_kv, padding=tuple(k//2 for k in kernel_kv), groups=dim)
        self.pool_v = nn.Conv3d(dim, dim, kernel_kv, stride=stride_kv, padding=tuple(k//2 for k in kernel_kv), groups=dim)
        
        self.stride_q = stride_q
        self.stride_kv = stride_kv
        
        # RoPE for relative position encoding
        self.rope = RotaryPositionEmbedding(head_dim)
    
    def forward(self, x, thw_shape):
        """
        Args:
            x: (B, N, C)
            thw_shape: (T, H, W) - spatial dimensions
        """
        B, N, C = x.shape
        T, H, W = thw_shape
        
        # Reshape to 3D
        x_3d = x.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        
        # Apply pooling
        q = self.pool_q(x_3d).flatten(2).transpose(1, 2)  # (B, N_q, C)
        k = self.pool_k(x_3d).flatten(2).transpose(1, 2)  # (B, N_k, C)
        v = self.pool_v(x_3d).flatten(2).transpose(1, 2)  # (B, N_v, C)
        
        # Linear projections
        q = self.q(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Apply RoPE separately to q and k (they may have different sequence lengths after pooling)
        # For q: apply RoPE with its own sequence length
        q_seq_len = q.shape[2]
        cos_q, sin_q = self.rope._compute_cos_sin(q_seq_len, q.device)
        q = (q * cos_q) + (self.rope.rotate_half(q) * sin_q)
        
        # For k: apply RoPE with its own sequence length
        k_seq_len = k.shape[2]
        cos_k, sin_k = self.rope._compute_cos_sin(k_seq_len, k.device)
        k = (k * cos_k) + (self.rope.rotate_half(k) * sin_k)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Update shape for next layer
        T_new = T // self.stride_q[0]
        H_new = H // self.stride_q[1]
        W_new = W // self.stride_q[2]
        
        return x, (T_new, H_new, W_new)


class MultiScaleBlock(nn.Module):
    """Multi-scale transformer block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 kernel_q=(1, 1, 1), kernel_kv=(1, 1, 1), stride_q=(1, 1, 1), stride_kv=(1, 1, 1)):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiScaleAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
            kernel_q=kernel_q, kernel_kv=kernel_kv,
            stride_q=stride_q, stride_kv=stride_kv
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        self.stride_q = stride_q
        
        # Dimension projection if downsampling
        if any(s > 1 for s in stride_q):
            self.proj = nn.Linear(dim, dim)
        else:
            self.proj = None
    
    def forward(self, x, thw_shape):
        """
        Args:
            x: (B, N, C)
            thw_shape: (T, H, W)
        """
        x_norm = self.norm1(x)
        x_attn, thw_new = self.attn(x_norm, thw_shape)
        
        # Handle residual connection with downsampling
        if self.proj is not None:
            x = self.proj(x_norm)
            # Downsample x to match x_attn
            B, N, C = x.shape
            T, H, W = thw_shape
            x = x.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
            x = F.avg_pool3d(x, kernel_size=self.stride_q, stride=self.stride_q)
            x = x.flatten(2).transpose(1, 2)
        
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        
        return x, thw_new


class MultiscaleVisionTransformer(nn.Module):
    """
    Multiscale Vision Transformer (MViT)
    
    Paper: "Multiscale Vision Transformers" (https://arxiv.org/abs/2104.11227)
    """
    def __init__(self, img_size=256, patch_size=16, temporal_size=4, temporal_patch_size=1,
                 in_channels=3, num_classes=0, embed_dim=96, depth=16, num_heads=1,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        """
        Args:
            num_classes: Number of classes for classification.
                        If 0 (default), returns patch token embeddings for CLIP.
                        If > 0, adds classification head for supervised learning.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size)
        )
        
        # Calculate initial spatial dimensions
        self.T = temporal_size // temporal_patch_size
        self.H = img_size // patch_size
        self.W = img_size // patch_size
        
        # RoPE replaces positional embedding
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Multi-scale stages (downsample at specific depths)
        # Note: MultiScaleBlock only changes spatial resolution, NOT channel dimension
        # All blocks use the same embed_dim throughout
        self.blocks = nn.ModuleList()
        
        for i in range(depth):
            # Downsample at certain layers (e.g., every 4 layers)
            if i > 0 and i % 4 == 0 and i < depth - 1:
                stride_q = (1, 2, 2)  # Spatial downsampling only
            else:
                stride_q = (1, 1, 1)
            
            block = MultiScaleBlock(
                dim=embed_dim,  # Same dimension for all blocks
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                kernel_q=(1, 3, 3),
                kernel_kv=(1, 3, 3),
                stride_q=stride_q,
                stride_kv=(1, 1, 1)
            )
            self.blocks.append(block)
        
        self.final_dim = embed_dim  # Final dimension same as embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        
        # Optional classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, H, W, C) tensor
        Returns:
            - If num_classes = 0: (B, num_patches, final_dim) patch embeddings for CLIP
            - If num_classes > 0: (B, num_classes) classification logits
        """
        B = x.shape[0]
        
        # Convert to (B, C, T, H, W)
        x = rearrange(x, 'b t h w c -> b c t h w')
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, T', H', W')
        
        # Flatten and transpose
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        x = self.pos_drop(x)
        
        # Multi-scale blocks (RoPE applied inside attention)
        thw = (self.T, self.H, self.W)
        for blk in self.blocks:
            x, thw = blk(x, thw)
        
        x = self.norm(x)
        
        if self.num_classes == 0:
            # For CLIP: return all patch tokens
            return x  # (B, num_patches, final_dim)
        else:
            # For classification: global average pooling
            x = x.mean(dim=1)
            x = self.head(x)
            return x


"""
Multiscale Vision Transformer Usage Example:
--------------------------------------------
import torch
from video_encoders import MultiscaleVisionTransformer

# Create dummy input: (batch, temporal, height, width, channel)
x = torch.randn(2, 4, 256, 256, 3)

# ===== CLIP Pretraining (default) =====
model = MultiscaleVisionTransformer(
    img_size=256,
    patch_size=16,
    temporal_size=4,
    in_channels=3,
    embed_dim=96,
    depth=16,
    num_heads=1
)

patch_embeddings = model(x)  # (2, num_patches, final_dim)
print(f"Patch embeddings shape: {patch_embeddings.shape}")

# ===== Supervised Fine-tuning =====
classifier = MultiscaleVisionTransformer(
    img_size=256,
    patch_size=16,
    temporal_size=4,
    in_channels=3,
    num_classes=10,
    embed_dim=96,
    depth=16,
    num_heads=1
)

logits = classifier(x)  # (2, 10)
"""


# ===========================
# 5. Uniformer
# ===========================

class LocalTemporalAttention(nn.Module):
    """Local temporal attention using convolution"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 1, 1), 
                                    stride=1, padding=(1, 0, 0), groups=dim * 3, bias=False)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=False)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        """
        b, c, t, h, w = x.shape
        
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = rearrange(q, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = rearrange(out, 'b head c (t h w) -> b (head c) t h w', head=self.num_heads, t=t, h=h, w=w)
        
        out = self.project_out(out)
        return out


class GlobalAttention(nn.Module):
    """Global attention for later stages with RoPE"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # RoPE for relative position encoding
        self.rope = RotaryPositionEmbedding(head_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) -> converted internally
        """
        # Convert to (B, N, C) for standard attention
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Convert back to (B, C, T, H, W)
        x = rearrange(x, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)
        return x


class CMlp(nn.Module):
    """Convolutional MLP"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UniformerBlock(nn.Module):
    """Uniformer block with local or global attention"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., use_local=True):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(dim)
        
        if use_local:
            self.attn = LocalTemporalAttention(dim, num_heads=num_heads)
        else:
            self.attn = GlobalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = nn.BatchNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Uniformer(nn.Module):
    """
    Uniformer: Unified Transformer for Efficient Spatiotemporal Representation Learning
    
    Paper: "Uniformer: Unified Transformer for Efficient Spatiotemporal Representation Learning"
    (https://arxiv.org/abs/2201.04676)
    
    Uses local attention in early layers and global attention in later layers.
    """
    def __init__(self, img_size=256, patch_size=16, temporal_size=4, temporal_patch_size=1,
                 in_channels=3, num_classes=0, embed_dims=[64, 128, 256, 512],
                 depths=[3, 4, 8, 3], num_heads=[1, 2, 4, 8], mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        """
        Args:
            num_classes: Number of classes for classification.
                        If 0 (default), returns spatial patch embeddings for CLIP.
                        If > 0, adds classification head for supervised learning.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_stages = len(depths)
    
        # Patch embedding (stem)
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dims[0],
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size)
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            # Use local attention in first 2 stages, global in later stages
            use_local = (i < 2)
            
            stage = nn.ModuleList([
                UniformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    use_local=use_local
                )
                for _ in range(depths[i])
            ])
            self.stages.append(stage)
        
            # Downsample between stages (except last)
            if i < self.num_stages - 1:
                self.stages.append(
                    nn.Conv3d(embed_dims[i], embed_dims[i + 1],
                             kernel_size=(1, 2, 2), stride=(1, 2, 2))
                )
        
        self.final_dim = embed_dims[-1]
        self.norm = nn.BatchNorm3d(embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Optional classification head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else None
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, H, W, C) tensor
        Returns:
            - If num_classes = 0: (B, num_patches, final_dim) spatial embeddings for CLIP
            - If num_classes > 0: (B, num_classes) classification logits
        """
        # Convert to (B, C, T, H, W)
        x = rearrange(x, 'b t h w c -> b c t h w')
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dims[0], T', H', W')
        
        # Stages
        for i, stage in enumerate(self.stages):
            if isinstance(stage, nn.ModuleList):
                # Uniformer blocks
                for blk in stage:
                    x = blk(x)
            else:
                # Downsample
                x = stage(x)
        
        # Norm
        x = self.norm(x)  # (B, C, T', H', W')
        
        if self.num_classes == 0:
            # For CLIP: return all spatial tokens
            B, C, T, H, W = x.shape
            x = rearrange(x, 'b c t h w -> b (t h w) c')  # (B, num_patches, C)
            return x
        else:
            # For classification: global average pooling
            x = self.avgpool(x)  # (B, C, 1, 1, 1)
            x = x.flatten(1)  # (B, C)
            x = self.head(x)
            return x


"""
Uniformer Usage Example:
------------------------
import torch
from video_encoders import Uniformer

# Create dummy input: (batch, temporal, height, width, channel)
x = torch.randn(2, 4, 256, 256, 3)

# ===== CLIP Pretraining (default) =====
model = Uniformer(
    img_size=256,
    patch_size=16,
    temporal_size=4,
    in_channels=3,
    embed_dims=[64, 128, 256, 512],
    depths=[3, 4, 8, 3],
    num_heads=[1, 2, 4, 8]
)

spatial_embeddings = model(x)  # (2, num_patches, 512)
# Output dim = embed_dims[-1] = 512
print(f"Spatial embeddings shape: {spatial_embeddings.shape}")

# ===== Supervised Fine-tuning =====
classifier = Uniformer(
    img_size=256,
    patch_size=16,
    temporal_size=4,
    in_channels=3,
    num_classes=10,
    embed_dims=[64, 128, 256, 512],
    depths=[3, 4, 8, 3]
)

logits = classifier(x)  # (2, 10)
"""


# ===========================
# Utility Functions
# ===========================

"""
Complete Usage Example with All Models:
---------------------------------------
import torch
from video_encoders import get_model, count_parameters

# Input shape: (batch, temporal, height, width, channel)
x = torch.randn(2, 4, 256, 256, 3)

# =============================================================================
# CLIP Pretraining Mode (num_classes=0, default)
# =============================================================================
print("=" * 70)
print("CLIP PRETRAINING MODE - Returns Patch Embeddings")
print("=" * 70)

clip_params = {
    'img_size': 256,
    'temporal_size': 4,
    'in_channels': 3,
    'num_classes': 0  # CLIP mode
}
    
    # 1. ViVit
print("\\n1. ViVit")
vivit = get_model('vivit', **clip_params, patch_size=16, embed_dim=384, depth=6, num_heads=6)
embeddings = vivit(x)
print(f"   Output shape: {embeddings.shape}")  # (2, num_patches, 384)
print(f"   Parameters: {count_parameters(vivit):,}")
    
    # 2. TimeSformer
print("\\n2. TimeSformer")
timesformer = get_model('timesformer', **clip_params, patch_size=16, embed_dim=384, depth=6, num_heads=6)
embeddings = timesformer(x)
print(f"   Output shape: {embeddings.shape}")  # (2, num_patches, 384)
print(f"   Parameters: {count_parameters(timesformer):,}")
    
    # 3. Video Swin Transformer
print("\\n3. Video Swin Transformer")
video_swin = get_model('video_swin', **clip_params, patch_size=4, embed_dim=96, 
                       depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
embeddings = video_swin(x)
print(f"   Output shape: {embeddings.shape}")  # (2, num_patches, 768)
print(f"   Parameters: {count_parameters(video_swin):,}")
    
    # 4. Multiscale Vision Transformer
print("\\n4. Multiscale Vision Transformer (MViT)")
mvit = get_model('mvit', **clip_params, patch_size=16, embed_dim=96, depth=16, num_heads=1)
embeddings = mvit(x)
print(f"   Output shape: {embeddings.shape}")  # (2, num_patches, final_dim)
print(f"   Parameters: {count_parameters(mvit):,}")
    
# 5. Uniformer
print("\\n5. Uniformer")
uniformer = get_model('uniformer', **clip_params, patch_size=16, 
                      embed_dims=[64, 128, 256, 512], depths=[3, 4, 8, 3])
embeddings = uniformer(x)
print(f"   Output shape: {embeddings.shape}")  # (2, num_patches, 512)
print(f"   Parameters: {count_parameters(uniformer):,}")

# =============================================================================
# Supervised Fine-tuning Mode (num_classes > 0)
# =============================================================================
print("\\n" + "=" * 70)
print("SUPERVISED FINE-TUNING MODE - Returns Classification Logits")
print("=" * 70)

supervised_params = {
    'img_size': 256,
    'temporal_size': 4,
    'in_channels': 3,
    'num_classes': 10  # Classification mode
}

print("\\nViVit Classifier")
classifier = get_model('vivit', **supervised_params, patch_size=16, embed_dim=384, depth=6, num_heads=6)
logits = classifier(x)
print(f"   Output shape: {logits.shape}")  # (2, 10)

# =============================================================================
# CLIP Workflow Example
# =============================================================================
print("\\n" + "=" * 70)
print("TYPICAL CLIP WORKFLOW")
print("=" * 70)
example_code = '''
# 1. Get patch embeddings from video encoder
video_encoder = ViVit(embed_dim=768, depth=12)  # num_classes=0 by default
patch_embeddings = video_encoder(video_input)  # (B, num_patches, 768)

# 2. Pool patches (mean, attention, or learnable pooling)
video_features = patch_embeddings.mean(dim=1)  # (B, 768)

# 3. Project to CLIP embedding space
projection = nn.Linear(768, clip_dim)  # e.g., clip_dim=512
video_clip_features = projection(video_features)  # (B, 512)

# 4. Compute contrastive loss with text features
# loss = contrastive_loss(video_clip_features, text_clip_features)
'''
print(example_code)


# =============================================================================
# About RoPE (Rotary Position Embedding) Implementation
# =============================================================================

RoPE Benefits:
--------------
1. Relative Position Encoding: RoPE naturally encodes relative positions between
   tokens through rotation, making it more robust for variable-length sequences.

2. No Learned Parameters: Unlike absolute positional embeddings, RoPE doesn't
   require learning position-specific parameters, reducing model size.

3. Better Extrapolation: Models with RoPE can better handle sequences longer
   than those seen during training.

4. Computational Efficiency: RoPE is applied during attention computation without
   additional position embedding lookups.

5. Translation Invariance: The rotation mechanism provides built-in translation
   invariance properties.

Changes from Original Implementations:
--------------------------------------
- Removed all `pos_embed` parameters (nn.Parameter)
- Added `RotaryPositionEmbedding` module that applies rotation to Q and K
- RoPE is applied in all attention mechanisms:
  * MultiheadAttention (used in ViVit, TimeSformer)
  * WindowAttention3D (used in Video Swin Transformer)
  * MultiScaleAttention (used in MViT)
  * GlobalAttention (used in Uniformer)
- Position indices are automatically generated or passed to attention layers

Technical Notes:
----------------
- RoPE rotates query and key vectors at different frequencies based on position
- The rotation is applied in the head dimension (head_dim)
- For video data, positions are treated sequentially (flattened T×H×W patches)
- CLS tokens receive position 0, patches receive sequential positions
"""


# ===========================
# Backward Compatibility Aliases
# ===========================

# Default video encoder (Conv3DEncoder is the baseline for SigLIP)
VideoEncoder = Conv3DEncoder


# ===========================
# Wrapper for SigLIP Compatibility
# ===========================

class SigLIPVideoEncoderWrapper(nn.Module):
    """
    Wrapper for transformer-based video encoders to match Conv3DEncoder output format.
    
    Transformer encoders (ViVit, TimeSformer, etc.) return patch embeddings: (B, num_patches, embed_dim)
    Conv3DEncoder returns: (sequence, pooled) where sequence is (B, T, hidden_dim) and pooled is (B, hidden_dim*5)
    
    This wrapper adds multi-scale temporal pooling to transformer outputs to match Conv3DEncoder format.
    """
    def __init__(self, encoder: nn.Module, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        
        # Projection to match hidden_dim.
        # For hierarchical backbones (e.g., VideoSwin), output dim is `num_features`,
        # while `embed_dim` is only the stage-0 width.
        encoder_dim = getattr(encoder, 'num_features', getattr(encoder, 'embed_dim', hidden_dim))
        if encoder_dim != hidden_dim:
            self.proj = nn.Linear(encoder_dim, hidden_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: Input video (B, T, H, W, C)
        
        Returns:
            sequence: (B, num_patches, hidden_dim) where num_patches depends on encoder
        """
        # Get patch embeddings from transformer encoder
        x = self.encoder(video)  # (B, num_patches, embed_dim)
        
        # Project to hidden_dim if needed
        x = self.proj(x)  # (B, num_patches, hidden_dim)
        
        return x  # (B, num_patches, hidden_dim)


# ===========================
# Factory Function
# ===========================

def get_video_encoder(encoder_type: str = "conv3d", **kwargs) -> nn.Module:
    """
    Factory function to build video encoders by type.
    
    Args:
        encoder_type: One of ["conv3d", "vivit", "timesformer", "video_swin", "mvit", "uniformer"]
            - "conv3d" / "baseline": Conv3D-based encoder (Conv3DEncoder)
            - "vivit": Video Vision Transformer (ViVit)
            - "timesformer": Divided Space-Time Attention Transformer
            - "video_swin": Video Swin Transformer
            - "mvit": Multiscale Vision Transformer
            - "uniformer": Unified Transformer
        **kwargs: Additional arguments passed to the encoder constructor
            Common kwargs for SigLIP baseline (Conv3D):
                - hidden_dim: Hidden dimension (default: 256)
            Common kwargs for transformers (ViVit, TimeSformer):
                - img_size: Image size (default: 256)
                - patch_size: Patch size (default: 16)
                - temporal_size: Temporal dimension (default: 4)
                - temporal_patch_size: Temporal patch size (default: 1)
                - in_channels: Input channels (default: 3)
                - num_classes: Number of classes, 0 for CLIP mode (default: 0)
                - embed_dim: Embedding dimension (default: 768)
                - depth: Number of transformer blocks (default: 12)
                - num_heads: Number of attention heads (default: 12)
                - mlp_ratio: MLP expansion ratio (default: 4.0)
                - drop_rate: Dropout rate (default: 0.0)
                - attn_drop_rate: Attention dropout rate (default: 0.0)
            Video Swin specific:
                - depths: List of depths per stage (default: [2, 2, 6, 2])
                - num_heads: List of heads per stage (default: [3, 6, 12, 24])
                - window_size: Window size (default: (2, 8, 8))
            Uniformer specific:
                - embed_dims: List of embedding dimensions (default: [64, 128, 256, 512])
                - depths: List of depths per stage (default: [3, 4, 8, 3])
                - num_heads: List of heads per stage (default: [1, 2, 4, 8])
    
    Returns:
        Video encoder module
    
    Example:
        >>> # Conv3D encoder (SigLIP baseline)
        >>> encoder = get_video_encoder("conv3d", hidden_dim=256)
        >>> video = torch.randn(2, 4, 256, 256, 3)
        >>> sequence, pooled = encoder(video)
        
        >>> # ViVit encoder (wrapped for SigLIP compatibility)
        >>> encoder = get_video_encoder("vivit", hidden_dim=256, img_size=256, embed_dim=256, depth=12)
        >>> sequence, pooled = encoder(video)
        
    Note:
        When used with SigLIP, all encoders return (sequence, pooled) tuple.
        Transformer encoders are automatically wrapped with SigLIPVideoEncoderWrapper.
    """
    encoder_type = encoder_type.lower()
    
    # Extract hidden_dim for wrapper (default: 256) and remove from kwargs
    # Transformer encoders don't use hidden_dim, they use embed_dim
    hidden_dim = kwargs.pop('hidden_dim', 256)
    
    if encoder_type in ("conv3d", "baseline", "conv", "default"):
        # Conv3DEncoder already returns (sequence, pooled) format
        # It uses hidden_dim parameter
        return Conv3DEncoder(hidden_dim=hidden_dim, **kwargs)
    
    elif encoder_type in ("r3d", "r3d_18", "resnet3d", "resnet3d_18"):
        # Standard 3D CNN baseline (3D ResNet-18).
        pretrained = bool(kwargs.pop("pretrained", False))
        spatial_pool = kwargs.pop("spatial_pool", "mean")
        encoder = TorchvisionVideoResNetEncoder(
            backbone="r3d_18",
            hidden_dim=hidden_dim,
            pretrained=pretrained,
            spatial_pool=spatial_pool,
        )
        return encoder

    elif encoder_type in ("r2plus1d", "r2plus1d_18", "resnet2plus1d", "resnet2plus1d_18"):
        # Stronger 3D CNN baseline (R(2+1)D ResNet-18).
        pretrained = bool(kwargs.pop("pretrained", False))
        spatial_pool = kwargs.pop("spatial_pool", "mean")
        encoder = TorchvisionVideoResNetEncoder(
            backbone="r2plus1d_18",
            hidden_dim=hidden_dim,
            pretrained=pretrained,
            spatial_pool=spatial_pool,
        )
        return encoder

    elif encoder_type in ("resnet3d", "resnet3d18", "resnet3d_18_scaled", "resnet3d_scaled"):
        # Width-scaled ResNet3D-18 style baseline for parameter matching.
        base_channels = int(kwargs.pop("base_channels", 48))
        encoder = ResNet3DEncoder(
            hidden_dim=hidden_dim,
            base_channels=base_channels,
            layers=(2, 2, 2, 2),
        )
        return encoder

    elif encoder_type in ("vivit", "vit"):
        # Set num_classes=0 for CLIP mode (returns patch embeddings)
        kwargs.setdefault('num_classes', 0)
        # Use hidden_dim as embed_dim if not specified
        kwargs.setdefault('embed_dim', hidden_dim)
        encoder = ViVit(**kwargs)
        return SigLIPVideoEncoderWrapper(encoder, hidden_dim)
    
    elif encoder_type in ("timesformer", "time_sformer"):
        kwargs.setdefault('num_classes', 0)
        kwargs.setdefault('embed_dim', hidden_dim)
        encoder = TimeSformer(**kwargs)
        return SigLIPVideoEncoderWrapper(encoder, hidden_dim)
    
    elif encoder_type in ("video_swin", "swin", "swin_transformer"):
        kwargs.setdefault('num_classes', 0)
        # Provide robust defaults for VideoSwin when config omits stage settings.
        kwargs.setdefault('temporal_patch_size', 1)
        kwargs.setdefault('embed_dim', 112)
        kwargs.setdefault('depths', [1, 1, 1, 1])
        kwargs.setdefault('num_heads', [4, 8, 16, 32])
        kwargs.setdefault('window_size', (2, 7, 7))
        kwargs.setdefault('mlp_ratio', 4.0)
        # Video Swin uses embed_dim that scales up through stages
        # Final dimension will be embed_dim * 2^(num_stages-1)
        encoder = VideoSwinTransformer(**kwargs)
        return SigLIPVideoEncoderWrapper(encoder, hidden_dim)
    
    elif encoder_type in ("mvit", "multiscale_vit", "multiscale"):
        kwargs.setdefault('num_classes', 0)
        kwargs.setdefault('embed_dim', hidden_dim)
        encoder = MultiscaleVisionTransformer(**kwargs)
        return SigLIPVideoEncoderWrapper(encoder, hidden_dim)
    
    elif encoder_type in ("uniformer", "uniform"):
        kwargs.setdefault('num_classes', 0)
        encoder = Uniformer(**kwargs)
        return SigLIPVideoEncoderWrapper(encoder, hidden_dim)
    
    else:
        raise ValueError(
            f"Unknown encoder_type '{encoder_type}'. "
            f"Choose from: conv3d, vivit, timesformer, video_swin, mvit, uniformer"
        )


# ===========================
# Utility Functions
# ===========================

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================
# Exports
# ===========================

__all__ = [
    # Main encoders
    "Conv3DEncoder",
    "TorchvisionVideoResNetEncoder",
    "ResNet3DEncoder",
    "ViVit",
    "TimeSformer",
    "VideoSwinTransformer",
    "MultiscaleVisionTransformer",
    "Uniformer",
    
    # Wrapper for SigLIP compatibility
    "SigLIPVideoEncoderWrapper",
    
    # Backward compatibility
    "VideoEncoder",  # alias for Conv3DEncoder
    
    # Factory function
    "get_video_encoder",
    
    # Utilities
    "count_parameters",
    
    # Helper modules (for advanced users)
    "RotaryPositionEmbedding",
    "PatchEmbed3D",
    "MultiheadAttention",
    "MLP",
    "TransformerBlock",
]