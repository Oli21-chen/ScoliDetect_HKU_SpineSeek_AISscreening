"""
Supervised Fine-Tuning Regressor for multimodal SigLIP.
Uses video encoder, knowledge map encoder, and text encoder to predict labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from einops.layers.torch import Rearrange,Reduce

from .video_encoder import get_video_encoder
from .knowledge_encoder import get_knowledge_encoder
from .text_encoder import TextEncoder


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

class CrossAttentionBlock(nn.Module):
    """
    Cross attention: query attends to key/value from another modality.
    out = query + Dropout(Attention(norm(query), key, value))
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Lq, D) - tokens that attend
            key_value: (B, Lkv, D) - key and value source
        Returns:
            (B, Lq, D) - query + attention output
        """
        q = self.norm(query)
        attn_out, _ = self.attn(q, key_value, key_value, attn_mask=attn_mask)
        x = query + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GatedTokenPooling(nn.Module):
    """
    Sigmoid-gated weighted pooling: learns which tokens to emphasize.
    Replaces mean pooling with importance-weighted sum.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) - token sequence
        Returns:
            (B, D) - pooled representation
        """
        scores = torch.sigmoid(self.gate(x))  # (B, L, 1)
        weighted = (scores * x).sum(dim=1) / (scores.sum(dim=1) + 1e-8)
        return weighted


class SFTRegressor(nn.Module): # The Best version
    """
    Supervised Fine-Tuning Regressor for multimodal SigLIP.
    Uses video encoder, knowledge map encoder, and text encoder to predict labels.
    bidirectional cross-attention between video and km (video_to_km_cross, km_to_video_cross).
    Args:
        km_feature_dim: Knowledge map input feature dimension (default: 238)
        hidden_dim: Hidden dimension for encoders (default: 256)
        label_dim: Output label dimension (default: 1 for binary classification)
        video_encoder_type: Video encoder type (default: "conv3d")
        video_encoder_kwargs: Additional kwargs for video encoder
        km_encoder_type: Knowledge encoder type (default: "baseline")
        km_encoder_kwargs: Additional kwargs for knowledge encoder
        text_model_name: Text encoder model name (default: "sentence-transformers/all-MiniLM-L6-v2")
        text_max_length: Maximum text sequence length (default: 64)
        text_trainable: Whether to fine-tune text encoder (default: True)
        use_text: Whether to use text encoder (default: True)
    """
    
    def __init__(
        self,
        km_feature_dim: int = 238,
        hidden_dim: int = 256,
        label_dim: int = 1,
        video_encoder_type: str = "conv3d",
        video_encoder_kwargs: Optional[Dict[str, Any]] = None,
        km_encoder_type: str = "baseline",
        km_encoder_kwargs: Optional[Dict[str, Any]] = None,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_max_length: int = 128,
        text_trainable: bool = True,
        use_text: bool = True,
        use_latent_pooling: bool = False,
        latent_pool_size: int = 1,
        regressor_dropout: float = 0.1,
        use_km_video_cross_attn: bool = True,
        cross_attn_num_heads: int = 8,
        cross_attn_drop: float = 0.1,
        cross_attn_num_layers: int = 2,
        temporal_attn_bias_strength: float = 0.0,
        modality_dropout_prob: float = 0.0,
        use_bottleneck_fusion: bool = False,
        bottleneck_tokens: int = 32,
        bottleneck_layers: int = 1,
        align_loss_weight: float = 0.0,
        align_loss_temperature: float = 0.07,
        aux_loss_type: str = "infonce",
        aux_proj_dim: int = 256,
        barlow_lambda: float = 5e-3,
        vicreg_sim_coeff: float = 25.0,
        vicreg_var_coeff: float = 25.0,
        vicreg_cov_coeff: float = 1.0,
        vicreg_var_target: float = 1.0,
        vicreg_eps: float = 1e-4,
        use_gated_token_pooling: bool = True,
    ):
        super().__init__()
        
        self.use_text = use_text
        self.hidden_dim = hidden_dim
        self.use_latent_pooling = use_latent_pooling
        self.latent_pool_size = max(1, latent_pool_size)
        self.use_km_video_cross_attn = use_km_video_cross_attn
        self.cross_attn_num_layers = max(1, int(cross_attn_num_layers))
        self.temporal_attn_bias_strength = max(0.0, float(temporal_attn_bias_strength))
        self.modality_dropout_prob = max(0.0, min(1.0, float(modality_dropout_prob)))
        self.use_bottleneck_fusion = bool(use_bottleneck_fusion)
        self.bottleneck_tokens = max(1, int(bottleneck_tokens))
        self.bottleneck_layers = max(1, int(bottleneck_layers))
        self.align_loss_weight = max(0.0, float(align_loss_weight))
        self.align_loss_temperature = max(1e-3, float(align_loss_temperature))
        self.aux_loss_type = str(aux_loss_type).lower().strip()
        if self.aux_loss_type not in {"none", "infonce", "barlow", "vicreg", "hybrid"}:
            self.aux_loss_type = "infonce"
        self.aux_proj_dim = max(8, int(aux_proj_dim))
        self.barlow_lambda = max(0.0, float(barlow_lambda))
        self.vicreg_sim_coeff = max(0.0, float(vicreg_sim_coeff))
        self.vicreg_var_coeff = max(0.0, float(vicreg_var_coeff))
        self.vicreg_cov_coeff = max(0.0, float(vicreg_cov_coeff))
        self.vicreg_var_target = max(1e-3, float(vicreg_var_target))
        self.vicreg_eps = max(1e-8, float(vicreg_eps))
        self.use_gated_token_pooling = use_gated_token_pooling
        self._last_aux_loss: Optional[torch.Tensor] = None
        self._last_aux_components: Optional[Dict[str, torch.Tensor]] = None
        
        # Build video encoder
        video_kwargs = {"hidden_dim": hidden_dim}
        if video_encoder_kwargs is not None:
            video_kwargs.update(video_encoder_kwargs)
        self.video_encoder = get_video_encoder(video_encoder_type, **video_kwargs)
        
        # Build knowledge map encoder
        km_kwargs = {"input_dim": km_feature_dim, "hidden_dim": hidden_dim}
        if km_encoder_kwargs is not None:
            km_kwargs.update(km_encoder_kwargs)
        self.km_encoder = get_knowledge_encoder(km_encoder_type, **km_kwargs)
        
        # Build text encoder (optional)
        if use_text:
            self.text_encoder = TextEncoder(
                model_name=text_model_name,
                max_length=text_max_length,
                trainable=text_trainable,
            )
            text_feat_dim = self.text_encoder.model.config.hidden_size
        else:
            self.text_encoder = None
            text_feat_dim = 0
        
        # Feature dimensions for multimodal fusion
        # - We pool each modality to a single hidden_dim vector and then fuse.
        self.num_modalities = 3 if use_text else 2

        # Modality-specific normalization
        self.video_norm = DyT(hidden_dim)
        self.km_norm = DyT(hidden_dim)
        self.text_norm = DyT(hidden_dim) if use_text else None

        # Latent attention pooling (optional): learnable latents attend over modality tokens
        self.latent_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )
        self.latent_norm = DyT(hidden_dim)
        self.latent_query = nn.Parameter(torch.randn(1, self.latent_pool_size, hidden_dim))
        
        # Projection to align text features to hidden_dim (MLP when text is used)
        if text_feat_dim > 0:
            self.text_proj = nn.Sequential(
                nn.Linear(text_feat_dim, hidden_dim*2),
                DyT(hidden_dim*2),
                nn.Linear(hidden_dim*2, hidden_dim),
            )
        else:
            self.text_proj = None

        # Cross attention between km and video (bidirectional)
        if use_km_video_cross_attn:
            self.video_to_km_cross = nn.ModuleList([
                CrossAttentionBlock(
                    embed_dim=hidden_dim,
                    num_heads=cross_attn_num_heads,
                    dropout=cross_attn_drop,
                )
                for _ in range(self.cross_attn_num_layers)
            ])
            self.km_to_video_cross = nn.ModuleList([
                CrossAttentionBlock(
                    embed_dim=hidden_dim,
                    num_heads=cross_attn_num_heads,
                    dropout=cross_attn_drop,
                )
                for _ in range(self.cross_attn_num_layers)
            ])
            self.cross_attn_num_heads = cross_attn_num_heads
        else:
            self.video_to_km_cross = None
            self.km_to_video_cross = None
            self.cross_attn_num_heads = 0

        # Perceiver-style bottleneck fusion: compact latent tokens attend to both modalities.
        if self.use_bottleneck_fusion:
            self.bottleneck_query = nn.Parameter(torch.randn(1, self.bottleneck_tokens, hidden_dim))
            self.bottleneck_video_cross = nn.ModuleList([
                CrossAttentionBlock(
                    embed_dim=hidden_dim,
                    num_heads=cross_attn_num_heads,
                    dropout=cross_attn_drop,
                )
                for _ in range(self.bottleneck_layers)
            ])
            self.bottleneck_km_cross = nn.ModuleList([
                CrossAttentionBlock(
                    embed_dim=hidden_dim,
                    num_heads=cross_attn_num_heads,
                    dropout=cross_attn_drop,
                )
                for _ in range(self.bottleneck_layers)
            ])
            self.bottleneck_norm = DyT(hidden_dim)
            self.bottleneck_to_video = nn.Linear(hidden_dim, hidden_dim)
            self.bottleneck_to_km = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.bottleneck_query = None
            self.bottleneck_video_cross = None
            self.bottleneck_km_cross = None
            self.bottleneck_norm = None
            self.bottleneck_to_video = None
            self.bottleneck_to_km = None

        # Gated token pooling (sigmoid-gated weighted pooling for video and km)
        if use_gated_token_pooling:
            self.video_token_pool = GatedTokenPooling(hidden_dim)
            self.km_token_pool = GatedTokenPooling(hidden_dim)
        else:
            self.video_token_pool = None
            self.km_token_pool = None

        # Projection head for auxiliary multimodal alignment losses.
        # Keeping this small stabilizes Barlow/VICReg while preserving main-task capacity.
        self.align_proj_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.aux_proj_dim),
        )
        self.align_proj_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.aux_proj_dim),
        )

        # Regression head (dropout in head reduces overfitting)
        # Input dim: hidden_dim when latent_pooling, else num_modalities * hidden_dim (concat fusion)
        regressor_input_dim = hidden_dim if use_latent_pooling else self.num_modalities * hidden_dim
        regressor_dropout = max(0.0, min(1.0, regressor_dropout))
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(regressor_dropout),
            nn.Linear(hidden_dim * 2, label_dim),
        )

    def _build_temporal_attn_mask(
        self,
        query_indices: Optional[torch.Tensor],
        key_indices: Optional[torch.Tensor],
        query_len: int,
        key_len: int,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if (
            self.temporal_attn_bias_strength <= 0.0
            or query_indices is None
            or key_indices is None
            or query_indices.dim() != 2
            or key_indices.dim() != 2
        ):
            return None
        q_idx = query_indices.to(device=device, dtype=dtype)
        k_idx = key_indices.to(device=device, dtype=dtype)

        # Resample indices to match actual token lengths seen by attention.
        # This handles cases where encoder token sequence length differs from raw index length
        # (e.g. video patch tokens >> original temporal frame count).
        if q_idx.shape[1] != query_len:
            q_idx = F.interpolate(
                q_idx.unsqueeze(1), size=query_len, mode="linear", align_corners=False
            ).squeeze(1)
        if k_idx.shape[1] != key_len:
            k_idx = F.interpolate(
                k_idx.unsqueeze(1), size=key_len, mode="linear", align_corners=False
            ).squeeze(1)

        q_min = q_idx.min(dim=1, keepdim=True).values
        q_max = q_idx.max(dim=1, keepdim=True).values
        k_min = k_idx.min(dim=1, keepdim=True).values
        k_max = k_idx.max(dim=1, keepdim=True).values
        q_norm = (q_idx - q_min) / (q_max - q_min + 1e-6)
        k_norm = (k_idx - k_min) / (k_max - k_min + 1e-6)
        dist = torch.abs(q_norm.unsqueeze(-1) - k_norm.unsqueeze(1))  # (B, L_q, 1) - (B, 1, L_k) = (B, Lq, Lk)
        bias = -self.temporal_attn_bias_strength * dist
        return bias.repeat_interleave(num_heads, dim=0)  # (B*num_heads, Lq, Lk)

    def _apply_modality_dropout(
        self,
        video_token: torch.Tensor,
        km_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (not self.training) or self.modality_dropout_prob <= 0.0:
            return video_token, km_token
        bsz = video_token.shape[0]
        keep_video = (torch.rand(bsz, 1, device=video_token.device) > self.modality_dropout_prob).to(video_token.dtype)
        keep_km = (torch.rand(bsz, 1, device=km_token.device) > self.modality_dropout_prob).to(km_token.dtype)
        both_dropped = (keep_video + keep_km) < 0.5
        keep_video = torch.where(both_dropped, torch.ones_like(keep_video), keep_video)
        scale = 1.0 / max(1e-6, 1.0 - self.modality_dropout_prob)
        return video_token * keep_video * scale, km_token * keep_km * scale

    def _compute_alignment_loss(
        self,
        video_token: torch.Tensor,
        km_token: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # Auxiliary multimodal alignment between pooled video and KM tokens.
        self._last_aux_components = None
        if self.align_loss_weight <= 0.0 or self.aux_loss_type == "none":
            return None
        bsz = video_token.shape[0]
        if bsz < 2:
            return video_token.new_zeros(())
        v = self.align_proj_v(video_token)
        k = self.align_proj_k(km_token)

        def _infonce_loss(v_feat: torch.Tensor, k_feat: torch.Tensor) -> torch.Tensor:
            v_n = F.normalize(v_feat, dim=-1)
            k_n = F.normalize(k_feat, dim=-1)
            logits_vk = torch.matmul(v_n, k_n.transpose(0, 1)) / self.align_loss_temperature
            logits_kv = torch.matmul(k_n, v_n.transpose(0, 1)) / self.align_loss_temperature
            targets = torch.arange(v_feat.shape[0], device=v_feat.device)
            return 0.5 * (
                F.cross_entropy(logits_vk, targets) + F.cross_entropy(logits_kv, targets)
            )

        def _barlow_loss(v_feat: torch.Tensor, k_feat: torch.Tensor) -> torch.Tensor:
            v_norm = (v_feat - v_feat.mean(dim=0)) / (v_feat.std(dim=0) + 1e-6)
            k_norm = (k_feat - k_feat.mean(dim=0)) / (k_feat.std(dim=0) + 1e-6)
            c = torch.matmul(v_norm.t(), k_norm) / float(v_feat.shape[0])  # (D, D)
            eye = torch.eye(c.size(0), device=c.device, dtype=c.dtype)
            c_diff = (c - eye).pow(2)
            off_diag = ~eye.bool()
            c_diff[off_diag] = c_diff[off_diag] * self.barlow_lambda
            return c_diff.sum()

        def _vicreg_var_term(x: torch.Tensor) -> torch.Tensor:
            std = torch.sqrt(x.var(dim=0, unbiased=False) + self.vicreg_eps)
            return F.relu(self.vicreg_var_target - std).mean()

        def _vicreg_cov_term(x: torch.Tensor) -> torch.Tensor:
            x_centered = x - x.mean(dim=0, keepdim=True)
            cov = (x_centered.t() @ x_centered) / max(1, x_centered.shape[0] - 1)
            eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
            off_diag = cov * (1.0 - eye)
            return (off_diag.pow(2).sum()) / cov.size(0)

        def _vicreg_loss(v_feat: torch.Tensor, k_feat: torch.Tensor) -> torch.Tensor:
            sim = F.mse_loss(v_feat, k_feat)
            var = _vicreg_var_term(v_feat) + _vicreg_var_term(k_feat)
            cov = _vicreg_cov_term(v_feat) + _vicreg_cov_term(k_feat)
            return (
                self.vicreg_sim_coeff * sim
                + self.vicreg_var_coeff * var
                + self.vicreg_cov_coeff * cov
            )

        infonce_loss = v.new_zeros(())
        barlow_loss = v.new_zeros(())
        vicreg_loss = v.new_zeros(())
        if self.aux_loss_type == "barlow":
            barlow_loss = _barlow_loss(v, k)
            raw_total = barlow_loss
        elif self.aux_loss_type == "vicreg":
            vicreg_loss = _vicreg_loss(v, k)
            raw_total = vicreg_loss
        elif self.aux_loss_type == "hybrid":
            infonce_loss = _infonce_loss(v, k)
            vicreg_loss = _vicreg_loss(v, k)
            raw_total = 0.8 * infonce_loss + 0.2 * vicreg_loss
        else:
            infonce_loss = _infonce_loss(v, k)
            raw_total = infonce_loss

        weighted_total = raw_total * self.align_loss_weight
        self._last_aux_components = {
            "infonce": infonce_loss,
            "barlow": barlow_loss,
            "vicreg": vicreg_loss,
            "raw_total": raw_total,
            "weighted_total": weighted_total,
        }
        return weighted_total

    def consume_aux_loss(self) -> Optional[torch.Tensor]:
        aux = self._last_aux_loss
        self._last_aux_loss = None
        return aux

    def consume_aux_components(self) -> Optional[Dict[str, torch.Tensor]]:
        comps = self._last_aux_components
        self._last_aux_components = None
        return comps


    def forward(
        self,
        video: torch.Tensor,
        knowledge_map: torch.Tensor,
        texts: Optional[List[str]] = None,
        km_indices: Optional[torch.Tensor] = None,
        video_indices: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        return_embeddings: bool = False,
    ):
        """
        Forward pass through the model.
        
        Args:
            video: Video tensor (B, T, H, W, C)
            knowledge_map: Knowledge map tensor (B, T, F)
            texts: Optional list of processed text strings (B,) - processed in collate function
            km_indices: Optional knowledge-map indices (B, T_km)
            video_indices: Optional video indices (B, T_vid)
            return_aux: If True, also return auxiliary alignment loss (and component tensor).
            return_embeddings: If True, return dict with predictions and modality embeddings.
        
        Returns:
            (B, label_dim) logits, or tuple with aux, or dict if return_embeddings.
        """
        self._last_aux_loss = None
        self._last_aux_components = None
        batch_size = video.shape[0]
        device = video.device
        
        # Encode video: (B, T, H, W, C) -> (B, T, hidden_dim)
        if video_indices is not None:
            try:
                video_output = self.video_encoder(video, video_indices=video_indices)
            except TypeError:
                video_output = self.video_encoder(video)
        else:
            video_output = self.video_encoder(video)
        if isinstance(video_output, tuple):
            video_seq, _ = video_output
        else:
            video_seq = video_output

        # Encode knowledge map: (B, T, F) -> (B, T, hidden_dim)
        if km_indices is not None:
            try:
                km_output = self.km_encoder(knowledge_map, km_indices=km_indices)
            except TypeError:
                km_output = self.km_encoder(knowledge_map)
        else:
            km_output = self.km_encoder(knowledge_map)
        if isinstance(km_output, tuple):
            km_seq, _ = km_output
        else:
            km_seq = km_output

        # Cross attention between km and video (bidirectional)
        if self.video_to_km_cross is not None and self.km_to_video_cross is not None:
            v2k_mask = self._build_temporal_attn_mask(
                query_indices=video_indices,
                key_indices=km_indices,
                query_len=video_seq.shape[1],
                key_len=km_seq.shape[1],
                num_heads=self.cross_attn_num_heads,
                device=video_seq.device,
                dtype=video_seq.dtype,
            )
            k2v_mask = self._build_temporal_attn_mask(
                query_indices=km_indices,
                key_indices=video_indices,
                query_len=km_seq.shape[1],
                key_len=video_seq.shape[1],
                num_heads=self.cross_attn_num_heads,
                device=km_seq.device,
                dtype=km_seq.dtype,
            )
            for v2k, k2v in zip(self.video_to_km_cross, self.km_to_video_cross):
                video_seq = v2k(video_seq, km_seq, attn_mask=v2k_mask)  # video attends to km
                km_seq = k2v(km_seq, video_seq, attn_mask=k2v_mask)  # km attends to video

        # Optional bottleneck fusion (compact latent interaction across modalities).
        if self.use_bottleneck_fusion and self.bottleneck_query is not None:
            bottleneck = self.bottleneck_query.expand(batch_size, -1, -1)
            for b2v, b2k in zip(self.bottleneck_video_cross, self.bottleneck_km_cross):
                bottleneck = b2v(bottleneck, video_seq)
                bottleneck = b2k(bottleneck, km_seq)
            bottleneck_token = self.bottleneck_norm(bottleneck.mean(dim=1))
        else:
            bottleneck_token = None

        # Pool over temporal dimension (gated or mean) and normalize
        if self.video_token_pool is not None and self.km_token_pool is not None:
            video_pooled = self.video_token_pool(video_seq)  # (B, hidden_dim)
            km_pooled = self.km_token_pool(km_seq)  # (B, hidden_dim)
        else:
            video_pooled = video_seq.mean(dim=1)  # (B, hidden_dim)
            km_pooled = km_seq.mean(dim=1)  # (B, hidden_dim)
        video_token = self.video_norm(video_pooled)  # (B, hidden_dim)
        km_token = self.km_norm(km_pooled)  # (B, hidden_dim)
        if bottleneck_token is not None:
            video_token = video_token + self.bottleneck_to_video(bottleneck_token)
            km_token = km_token + self.bottleneck_to_km(bottleneck_token)
        # video_token, km_token = self._apply_modality_dropout(video_token, km_token)
        aux_loss = self._compute_alignment_loss(video_token, km_token)
        self._last_aux_loss = aux_loss
        aux_components = self._last_aux_components
        
        # # Modality order is fixed: [video, km, text]. Do not change; regressor weights depend on this layout
        features_list = [video_token.unsqueeze(1), km_token.unsqueeze(1)]  # each (B, 1, hidden_dim)
        # Encode text if available
        text_features = None
        if self.use_text and self.text_encoder is not None and texts is not None:
            # Handle DataParallel: texts may contain full batch, need to slice
            if len(texts) > batch_size:
                try:
                    if device.type == 'cuda':
                        device_idx = torch.cuda.current_device()
                    else:
                        device_idx = 0
                    start_idx = device_idx * batch_size
                    end_idx = start_idx + batch_size
                    if start_idx < len(texts):
                        texts_slice = texts[start_idx:min(end_idx, len(texts))]
                    else:
                        texts_slice = texts[:batch_size]
                except (AttributeError, TypeError, RuntimeError):
                    texts_slice = texts[:batch_size]
            else:
                texts_slice = texts
            
            # Pad if needed
            while len(texts_slice) < batch_size:
                texts_slice.append("")
            
            text_features = self.text_encoder(texts_slice, device=device)  # (B, text_feat_dim)
            # Adapter to fusion hidden_dim: default-init + trainable unless ckpt loads them
            # (see load_pretrained_checkpoint(..., load_text_proj_from_pretrained)).
            if self.text_proj is not None:
                text_features = self.text_proj(text_features)  # (B, hidden_dim)
            text_features = self.text_norm(text_features) if self.text_norm is not None else text_features
        
        # If text is enabled but missing, use zero token to keep modality count consistent
        if self.use_text:
            if text_features is None:
                text_features = torch.zeros(
                    batch_size,
                    self.hidden_dim,
                    device=device,
                    dtype=video_token.dtype,
                )
            features_list.append(text_features.unsqueeze(1))  # (B, 1, hidden_dim)
        
        # Stack all modality tokens: (B, num_modalities, hidden_dim)
        stacked_features = torch.cat(features_list, dim=1)  # (B, num_modalities, hidden_dim)
        
        # Latent attention pooling (optional): learnable latents attend over modality tokens
        if self.use_latent_pooling:
            latents = self.latent_query.expand(batch_size, -1, -1)  # (B, latent_pool_size, hidden_dim)
            latent_out, _ = self.latent_attention(latents, stacked_features, stacked_features)
            latent_out = self.latent_norm(latent_out)  # (B, latent_pool_size, hidden_dim)
            pooled_features = latent_out.mean(dim=1)  # (B, hidden_dim)
            fused_features = pooled_features
        else:
            # Concat with reshape to (B, num_modalities * hidden_dim)
            fused_features = stacked_features.reshape(batch_size, -1)  # (B, num_modalities * hidden_dim)

        # Predict labels
        predictions = self.regressor(fused_features)  # (B, label_dim)

        if return_embeddings:
            out: Dict[str, Any] = {
                "predictions": predictions,
                "video_emb": video_token,
                "km_emb": km_token,
                "video_align": self.align_proj_v(video_token),
                "km_align": self.align_proj_k(km_token),
            }
            if return_aux:
                aux_components_tensor = None
                if aux_components is not None:
                    aux_components_tensor = torch.stack([
                        aux_components["infonce"].detach(),
                        aux_components["barlow"].detach(),
                        aux_components["vicreg"].detach(),
                        aux_components["raw_total"].detach(),
                        aux_components["weighted_total"].detach(),
                    ], dim=0)
                if aux_loss is None:
                    aux_loss = predictions.new_zeros((1,))
                elif torch.is_tensor(aux_loss) and aux_loss.dim() == 0:
                    aux_loss = aux_loss.unsqueeze(0)
                if aux_components_tensor is None:
                    aux_components_tensor = predictions.new_zeros((5,))
                out["aux_loss"] = aux_loss
                out["aux_components"] = aux_components_tensor
            return out

        if return_aux:
            aux_components_tensor = None
            if aux_components is not None:
                aux_components_tensor = torch.stack([
                    aux_components["infonce"].detach(),
                    aux_components["barlow"].detach(),
                    aux_components["vicreg"].detach(),
                    aux_components["raw_total"].detach(),
                    aux_components["weighted_total"].detach(),
                ], dim=0)
            if aux_loss is None:
                aux_loss = predictions.new_zeros((1,))
            elif torch.is_tensor(aux_loss) and aux_loss.dim() == 0:
                # DataParallel warns when gathering pure scalars; return (1,) per replica.
                aux_loss = aux_loss.unsqueeze(0)
            if aux_components_tensor is None:
                aux_components_tensor = predictions.new_zeros((5,))
            return predictions, aux_loss, aux_components_tensor
        return predictions


class SFTRegressor_k(nn.Module):
    """
    Supervised Fine-Tuning Regressor for multimodal SigLIP.
    Uses video encoder, knowledge map encoder, and text encoder to predict labels.
    
    Args:
        km_feature_dim: Knowledge map input feature dimension (default: 238)
        hidden_dim: Hidden dimension for encoders (default: 256)
        label_dim: Output label dimension (default: 1 for binary classification)
        video_encoder_type: Video encoder type (default: "conv3d")
        video_encoder_kwargs: Additional kwargs for video encoder
        km_encoder_type: Knowledge encoder type (default: "baseline")
        km_encoder_kwargs: Additional kwargs for knowledge encoder
        text_model_name: Text encoder model name (default: "sentence-transformers/all-MiniLM-L6-v2")
        text_max_length: Maximum text sequence length (default: 64)
        text_trainable: Whether to fine-tune text encoder (default: True)
        use_text: Whether to use text encoder (default: True)
    """
    
    def __init__(
        self,
        km_feature_dim: int = 238,
        hidden_dim: int = 256,
        label_dim: int = 1,
        video_encoder_type: str = "conv3d",
        video_encoder_kwargs: Optional[Dict[str, Any]] = None,
        km_encoder_type: str = "baseline",
        km_encoder_kwargs: Optional[Dict[str, Any]] = None,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_max_length: int = 128,
        text_trainable: bool = True,
        use_text: bool = True,
        use_latent_pooling: bool = False,
        latent_pool_size: int = 1,
        regressor_dropout: float = 0.1,
    ):
        super().__init__()
        
        # Build knowledge map encoder
        km_kwargs = {"input_dim": km_feature_dim, "hidden_dim": hidden_dim}
        if km_encoder_kwargs is not None:
            km_kwargs.update(km_encoder_kwargs)
        self.km_encoder = get_knowledge_encoder(km_encoder_type, **km_kwargs)

        self.km_norm = DyT(hidden_dim)
      
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(regressor_dropout),
            nn.Linear(hidden_dim * 2, label_dim),
        )
   
    def forward(
        self,
        video: torch.Tensor,
        knowledge_map: torch.Tensor,
        texts: Optional[List[str]] = None,
        km_indices: Optional[torch.Tensor] = None,
        video_indices: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through the model.
        
        Args:
            video: Video tensor (B, T, H, W, C)
            knowledge_map: Knowledge map tensor (B, T, F)
            texts: Optional list of processed text strings (B,) - processed in collate function
            km_indices: Optional knowledge-map indices (B, T_km)
            video_indices: Optional video indices (B, T_vid)
        
        Returns:
            Predicted labels (B, label_dim)
        """
        km_output = self.km_encoder(knowledge_map, km_indices=km_indices)
        if isinstance(km_output, tuple):
            km_seq, km_pooled = km_output
        else:
            km_seq = km_output
            # Mean pooling
            km_pooled = km_seq.mean(dim=1)  # (B, hidden_dim)
        km_pooled = self.km_norm(km_pooled)
        
        # Predict labels
        predictions = self.regressor(km_pooled)  # (B, label_dim)

        return predictions


class SFTRegressor_v(nn.Module):
    """
    Supervised Fine-Tuning Regressor — video-only ablation variant.
    Uses only the video encoder to predict labels (mirrors SFTRegressor_k for KM-only).

    Args:
        hidden_dim: Hidden dimension for encoders (default: 256)
        label_dim: Output label dimension (default: 1 for binary classification)
        video_encoder_type: Video encoder type (default: "conv3d")
        video_encoder_kwargs: Additional kwargs for video encoder
        km_feature_dim: Unused; kept for API compatibility with SFTRegressor_k
        km_encoder_type: Unused; kept for API compatibility
        km_encoder_kwargs: Unused; kept for API compatibility
        text_model_name: Unused; kept for API compatibility
        text_max_length: Unused; kept for API compatibility
        text_trainable: Unused; kept for API compatibility
        use_text: Unused; kept for API compatibility
    """

    def __init__(
        self,
        km_feature_dim: int = 238,
        hidden_dim: int = 256,
        label_dim: int = 1,
        video_encoder_type: str = "conv3d",
        video_encoder_kwargs: Optional[Dict[str, Any]] = None,
        km_encoder_type: str = "baseline",
        km_encoder_kwargs: Optional[Dict[str, Any]] = None,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_max_length: int = 128,
        text_trainable: bool = True,
        use_text: bool = True,
        use_latent_pooling: bool = False,
        latent_pool_size: int = 1,
        regressor_dropout: float = 0.1,
    ):
        super().__init__()

        video_kwargs = {"hidden_dim": hidden_dim}
        if video_encoder_kwargs is not None:
            video_kwargs.update(video_encoder_kwargs)
        self.video_encoder = get_video_encoder(video_encoder_type, **video_kwargs)

        self.video_norm = DyT(hidden_dim)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(regressor_dropout),
            nn.Linear(hidden_dim * 2, label_dim),
        )

    def forward(
        self,
        video: torch.Tensor,
        knowledge_map: torch.Tensor,
        texts: Optional[List[str]] = None,
        km_indices: Optional[torch.Tensor] = None,
        video_indices: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through the model.

        Args:
            video: Video tensor (B, T, H, W, C)
            knowledge_map: Unused; kept for API compatibility (B, T, F)
            texts: Unused; kept for API compatibility
            km_indices: Unused; kept for API compatibility
            video_indices: Optional video indices (B, T_vid)

        Returns:
            Predicted labels (B, label_dim)
        """
        if video_indices is not None:
            try:
                video_output = self.video_encoder(video, video_indices=video_indices)
            except TypeError:
                video_output = self.video_encoder(video)
        else:
            video_output = self.video_encoder(video)
        if isinstance(video_output, tuple):
            video_seq, video_pooled = video_output
        else:
            video_seq = video_output
            video_pooled = video_seq.mean(dim=1)  # (B, hidden_dim)
        video_pooled = self.video_norm(video_pooled)

        predictions = self.regressor(video_pooled)  # (B, label_dim)

        return predictions

