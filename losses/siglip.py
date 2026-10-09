import torch
import torch.nn.functional as F


def siglip_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Symmetric contrastive loss used by SigLIP.
    Treats diagonal pairs as positives.
    """
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)

