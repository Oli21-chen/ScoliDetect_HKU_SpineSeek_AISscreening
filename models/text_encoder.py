from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """
    Pretrained text encoder wrapper.
    Uses Hugging Face transformers with mean pooling on the last hidden state.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 64,
        trainable: bool = True,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        if len(texts) == 0:
            texts = [""]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        outputs = self.model(**encoded)
        last_hidden = outputs.last_hidden_state  # (B, L, hidden)
        attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (B, L, 1)
        masked_hidden = last_hidden * attention_mask
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        pooled = masked_hidden.sum(dim=1) / lengths  # mean pooling
        return pooled

