"""
Positional Encoding Utilities for Transformers

This module provides positional encoding implementations for transformer architectures,
including Rotary Position Embedding (RoPE) which is used across video and knowledge encoders.
"""

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    
    RoPE encodes absolute positions with rotation matrices and naturally incorporates
    relative position information into self-attention.
    
    Key Benefits:
    - Relative Position Encoding: Naturally encodes relative positions between tokens
    - No Learned Parameters: Reduces model size compared to absolute positional embeddings
    - Better Extrapolation: Can handle sequences longer than those seen during training
    - Computational Efficiency: Applied during attention without additional lookups
    - Translation Invariance: Built-in translation invariance properties
    
    Args:
        dim: Dimension of the positional encoding (typically head_dim in attention)
        max_seq_len: Maximum sequence length to cache (default: 2048)
    
    Example:
        >>> rope = RotaryPositionEmbedding(dim=64, max_seq_len=512)
        >>> # In attention layer:
        >>> q = torch.randn(2, 8, 100, 64)  # (B, num_heads, seq_len, head_dim)
        >>> k = torch.randn(2, 8, 100, 64)
        >>> q_rot, k_rot = rope(q, k)
    """
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        
        # Compute the inverse frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self.max_seq_len = max_seq_len
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _compute_cos_sin(self, seq_len, device):
        """
        Compute and cache cos/sin values for rotary position encoding.
        
        Args:
            seq_len: Sequence length
            device: Device to place tensors on
        
        Returns:
            cos: Cosine values (1, 1, seq_len, head_dim)
            sin: Sine values (1, 1, seq_len, head_dim)
        """
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            # Shape: (1, 1, seq_len, head_dim) for broadcasting with (B, num_heads, N, head_dim)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
        return self._cos_cached, self._sin_cached
    
    def rotate_half(self, x):
        """
        Rotates half the hidden dims of the input.
        
        This is a key operation in RoPE that creates the rotation effect.
        
        Args:
            x: Input tensor (..., dim)
        
        Returns:
            Rotated tensor with first half and second half swapped and negated
        """
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k, positions=None):
        """
        Apply rotary position embedding to query and key tensors.
        
        Args:
            q: Query tensor (B, num_heads, N, head_dim)
            k: Key tensor (B, num_heads, N, head_dim)
            positions: Optional position indices (B, N). If None, assumes sequential positions.
        
        Returns:
            q_rot: Rotated query tensor
            k_rot: Rotated key tensor
        """
        seq_len = q.shape[2]
        
        if positions is None:
            # Sequential positions
            cos, sin = self._compute_cos_sin(seq_len, q.device)
        else:
            # Custom positions
            t = positions.type_as(self.inv_freq)  # (B, N)
            freqs = torch.einsum('bi,j->bij', t, self.inv_freq)  # (B, N, head_dim//2)
            emb = torch.cat((freqs, freqs), dim=-1)  # (B, N, head_dim)
            cos = emb.cos()[:, None, :, :]  # (B, 1, N, head_dim) for broadcasting
            sin = emb.sin()[:, None, :, :]  # (B, 1, N, head_dim) for broadcasting
        
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rot, k_rot
    
    def forward(self, q, k, positions=None):
        """
        Forward pass - applies RoPE to q and k.
        
        Args:
            q: Query tensor (B, num_heads, N, head_dim)
            k: Key tensor (B, num_heads, N, head_dim)
            positions: Optional position indices (B, N)
        
        Returns:
            q_rot: Rotated query tensor
            k_rot: Rotated key tensor
        """
        return self.apply_rotary_pos_emb(q, k, positions)


__all__ = [
    "RotaryPositionEmbedding",
]

