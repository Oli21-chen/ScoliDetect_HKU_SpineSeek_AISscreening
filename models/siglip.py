import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .video_encoder import get_video_encoder
from .knowledge_encoder import get_knowledge_encoder
from .text_encoder import TextEncoder


class SigLIPBaseline(nn.Module):
    """
    SigLIP with trimodality pair-wise contrastive learning (km, video, text).
    Three pairwise objectives: km-text, video-text, video-km. Each pair uses
    diagonal positives and cross-entropy in both directions. Use
    utils.pre_utils.trimodality_contrastive_loss(outputs).
    
    Args:
        km_input_dim: Knowledge map input dimension (default: 238)
        hidden_dim: Hidden dimension for encoders (default: 256)
        projection_dim: Projection dimension for contrastive learning (default: 512)
        text_model_name: Text encoder model name (default: "sentence-transformers/all-MiniLM-L6-v2")
        text_max_length: Maximum text sequence length (default: 64)
        text_trainable: Whether to fine-tune text encoder (default: False)
        video_encoder_type: Video encoder type - "conv3d", "vivit", "timesformer", etc. (default: "conv3d")
        video_encoder_kwargs: Additional kwargs for video encoder (default: None)
        km_encoder_type: Knowledge encoder type - "baseline", "vit", or "patch_vit" (default: "baseline")
        km_encoder_kwargs: Additional kwargs for knowledge encoder (default: None)
    """

    def __init__(
        self,
        km_input_dim: int = 238,
        hidden_dim: int = 256,
        projection_dim: int = 512,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_max_length: int = 64,
        text_trainable: bool = False,
        video_encoder_type: str = "vivit",  # Options: "conv3d", "vivit", "timesformer", "video_swin", "mvit", "uniformer"
        video_encoder_kwargs: dict = None,
        km_encoder_type: str = "vit",  # Options: "baseline", "vit", "patch_vit"
        km_encoder_kwargs: dict = None,
    ):
        super().__init__()
        
        # Build video encoder with factory function
        video_kwargs = {
            "hidden_dim": hidden_dim,
        }
        if video_encoder_kwargs:
            video_kwargs.update(video_encoder_kwargs)
        self.video_encoder = get_video_encoder(video_encoder_type, **video_kwargs)
        
        # Build knowledge encoder with factory function
        km_kwargs = {
            "input_dim": km_input_dim,
            "hidden_dim": hidden_dim,
        }
        if km_encoder_kwargs:
            km_kwargs.update(km_encoder_kwargs)
        self.km_encoder = get_knowledge_encoder(km_encoder_type, **km_kwargs)
        
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            max_length=text_max_length,
            trainable=text_trainable,
        )

        # Modality-wise normalization (more stable than relying only on L2 at the end)
        self.video_norm = nn.LayerNorm(hidden_dim, bias=False)
        self.km_norm = nn.LayerNorm(hidden_dim, bias=False)
        
        # Per-modality projection heads for trimodality contrastive learning
        self.video_proj = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )
        self.km_proj = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.model.config.hidden_size, projection_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim*2, projection_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection heads with Xavier uniform initialization."""
        for module in [self.video_proj, self.km_proj, self.text_proj]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def forward(
        self,
        video: torch.Tensor,
        knowledge_map: torch.Tensor,
        text: List[str],
        km_indices: torch.Tensor = None,
        video_indices: torch.Tensor = None,
    ):
        batch_size = video.shape[0]
        device = video.device

        # Encode kinematic modalities
        if km_indices is not None:
            try:
                km_output = self.km_encoder(knowledge_map, km_indices=km_indices)
            except TypeError:
                km_output = self.km_encoder(knowledge_map)
        else:
            km_output = self.km_encoder(knowledge_map)

        if isinstance(km_output, tuple):
            km_output = km_output[0]

        if video_indices is not None:
            try:
                video_output = self.video_encoder(video, video_indices=video_indices)
            except TypeError:
                video_output = self.video_encoder(video)
        else:
            video_output = self.video_encoder(video)

        if isinstance(video_output, tuple):
            video_output = video_output[0]

        # Modality-wise LayerNorm on sequence features
        video_output = self.video_norm(video_output)
        km_output = self.km_norm(km_output)

        # Per-modality pooling and projection for trimodality contrastive learning
        video_features = self.video_proj(video_output.mean(dim=1))   # (B, projection_dim)
        km_features = self.km_proj(km_output.mean(dim=1))            # (B, projection_dim)

        # Encode text
        texts = text
        # Handle DataParallel case where full-batch texts are passed to each device
        if len(texts) > batch_size:
            try:
                if device.type == "cuda":
                    device_idx = torch.cuda.current_device()
                else:
                    device_idx = 0
                start_idx = device_idx * batch_size
                end_idx = start_idx + batch_size
                if start_idx < len(texts):
                    texts_slice = texts[start_idx : min(end_idx, len(texts))]
                else:
                    texts_slice = texts[:batch_size]
            except (AttributeError, TypeError, RuntimeError):
                texts_slice = texts[:batch_size]
        else:
            texts_slice = texts

        # Pad if needed (safety for last uneven replica)
        texts_slice = list(texts_slice)
        while len(texts_slice) < batch_size:
            texts_slice.append("")

        text_features = self.text_encoder(texts_slice, device=device)  # (B, text_feat_dim)
        text_features = self.text_proj(text_features)  # (B, projection_dim)

        # L2-normalize for pair-wise similarity contrastive learning
        video_features = F.normalize(video_features, p=2, dim=-1)  # (B, projection_dim)
        km_features = F.normalize(km_features, p=2, dim=-1)        # (B, projection_dim)
        text_features = F.normalize(text_features, p=2, dim=-1)    # (B, projection_dim)

        # Three pairwise similarity matrices (positives on diagonal)
        logit_scale = self.logit_scale.exp()
        logits_km_text = logit_scale * (km_features @ text_features.t())    # (B, B)
        logits_video_text = logit_scale * (video_features @ text_features.t())  # (B, B)
        logits_video_km = logit_scale * (video_features @ km_features.t())   # (B, B)

        return {
            "logits_km_text": logits_km_text,
            "logits_video_text": logits_video_text,
            "logits_video_km": logits_video_km,
            "video_features": video_features,
            "km_features": km_features,
            "text_features": text_features,
            "km_sequence": km_output,
            "video_sequence": video_output,
        }

        # return logits_video_km