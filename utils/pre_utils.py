# -*- coding: utf-8 -*-
"""
Utility functions for pretraining script.
Contains checkpoint management, GPU info, losses, and other helper functions.
"""

import os
import torch
import torch.nn.functional as F


def unwrap_model(model):
    """Return the underlying model, unwrapping DataParallel if present."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, val_loss, config, path):
    """Save all training states needed for resume."""
    # Ensure directory exists
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Convert to absolute path to avoid issues with relative paths
    abs_path = os.path.abspath(path)
    
    try:
        state_dict = unwrap_model(model).state_dict()
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "val_loss": val_loss,
                "config": config,
            },
            abs_path,
        )
    except Exception as e:
        print(f"❌ Error saving checkpoint to {abs_path}: {e}")
        print(f"   Directory exists: {os.path.exists(checkpoint_dir)}")
        print(f"   Directory is writable: {os.access(checkpoint_dir, os.W_OK) if checkpoint_dir else 'N/A'}")
        raise


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None, device='cpu'):
    """
    Load checkpoint and restore training state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        scaler: GradScaler to load state into (optional)
        device: Device to map tensors to
    
    Returns:
        start_epoch: Epoch to resume from (checkpoint epoch + 1)
        global_step: Global step count from checkpoint
        best_val_loss: Best validation loss from checkpoint
    """
    print(f"\n{'='*80}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model_to_load = unwrap_model(model)
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model state loaded")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Optimizer state loaded")
    
    # Extract training state first (needed for manual scheduler advancement)
    start_epoch = checkpoint.get('epoch', 0) + 1  # Resume from next epoch
    global_step = checkpoint.get('global_step', 0)
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    # Load scheduler state (handle incompatibilities gracefully)
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"✓ Scheduler state loaded")
        except (KeyError, RuntimeError, ValueError) as e:
            print(f"⚠️  Warning: Could not load scheduler state (scheduler type may have changed)")
            print(f"   Error: {str(e)}")
            print(f"   Manually advancing scheduler to step {global_step}...")
            # Manually step scheduler to catch up to global_step
            # This ensures LR is at the correct value for the resumed training step
            for _ in range(global_step):
                scheduler.step()
            print(f"✓ Scheduler manually advanced to step {global_step}")
    
    # Load scaler state
    if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"✓ GradScaler state loaded")
    
    print(f"✓ Resuming from epoch {start_epoch} (global step {global_step})")
    print(f"✓ Previous validation loss: {best_val_loss:.6f}")
    print(f"{'='*80}\n")
    
    return start_epoch, global_step, best_val_loss


def print_gpu_info() -> int:
    """Print detailed GPU availability info (respecting CUDA_VISIBLE_DEVICES) and return visible GPU count."""
    print("\n" + "=" * 70)
    print("🖥️  GPU DETECTION REPORT")
    print("=" * 70)

    mask = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if mask:
        print(f"🎯 CUDA_VISIBLE_DEVICES={mask}")
        print(f"   Physical GPU IDs: {mask} → Mapped to logical IDs: {', '.join(str(i) for i in range(len(mask.split(','))))}")
    else:
        print("CUDA_VISIBLE_DEVICES not set (all GPUs visible)")

    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return 0

    num_gpus = torch.cuda.device_count()
    print(f"✅ CUDA is available")
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ Number of visible GPUs: {num_gpus}")
    print("-" * 70)

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_total = props.total_memory / (1024 ** 3)  # Convert to GB
        print(f"\n  GPU {i}: {props.name}")
        print(f"    • Compute Capability: {props.major}.{props.minor}")
        print(f"    • Total Memory: {memory_total:.2f} GB")
        print(f"    • Multiprocessors: {props.multi_processor_count}")

    print("\n" + "=" * 70 + "\n")
    return num_gpus


# ---------- Contrastive losses (pair-wise similarity) ----------


def siglip_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Symmetric pair-wise contrastive loss (InfoNCE) for SigLIP.
    Treats diagonal pairs (i, i) as positives; all other pairs as negatives.
    Uses cross-entropy over the similarity matrix in both directions (kin→text, text→kin).

    Supports both:
    - Single-GPU / non-parallel: logits shape (B, B)
    - DataParallel-style multi-GPU: logits shape (B_global, B_local),
      where each GPU produced a local (B_local, B_local) block concatenated along dim=0.
    """
    b_rows, b_cols = logits.shape

    if b_rows == b_cols:
        batch_size = b_rows
        labels = torch.arange(batch_size, device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_i + loss_t)

    if b_cols == 0 or b_rows % b_cols != 0:
        raise ValueError(
            f"siglip_loss expected square logits or stacked (N*b, b) from DataParallel, "
            f"got shape {logits.shape}."
        )

    num_blocks = b_rows // b_cols
    local_b = b_cols
    labels_local = torch.arange(local_b, device=logits.device)
    logits_blocks = logits.view(num_blocks, local_b, local_b)
    loss_i = 0.0
    loss_t = 0.0
    for k in range(num_blocks):
        block = logits_blocks[k]
        loss_i = loss_i + F.cross_entropy(block, labels_local)
        loss_t = loss_t + F.cross_entropy(block.t(), labels_local)
    loss_i = loss_i / num_blocks
    loss_t = loss_t / num_blocks
    return 0.5 * (loss_i + loss_t)


def trimodality_contrastive_loss(
    outputs: dict,
    weight_km_text: float = 1.0,
    weight_video_text: float = 1.0,
    weight_video_km: float = 1.0,
    return_components: bool = True,
):
    """
    Trimodality pair-wise contrastive loss: km-text + video-text + video-km.
    Expects outputs dict with logits_km_text, logits_video_text, logits_video_km
    (each (B, B) or DataParallel-shaped). Sums symmetric InfoNCE over the three pairs.

    Returns:
        If return_components is True: (total_loss_tensor, {"loss_km_text": float, "loss_video_text": float, "loss_video_km": float})
        If return_components is False: total_loss_tensor only (backward compatible).
    """
    loss_km_text = siglip_loss(outputs["logits_km_text"])
    loss_video_text = siglip_loss(outputs["logits_video_text"])
    loss_video_km = siglip_loss(outputs["logits_video_km"])
    total = (
        weight_km_text * loss_km_text
        + weight_video_text * loss_video_text
        + weight_video_km * loss_video_km
    )
    if not return_components:
        return total
    details = {
        "loss_km_text": loss_km_text.item(),
        "loss_video_text": loss_video_text.item(),
        "loss_video_km": loss_video_km.item(),
    }
    return total, details


def pairwise_contrastive_loss(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    temperature: float = 0.07,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    Pair-wise similarity contrastive loss (InfoNCE) between two modality embeddings.
    Expects L2-normalized features; computes similarity as dot product / temperature.

    Args:
        features_a: (B, D) normalized embeddings (e.g. kinematic)
        features_b: (B, D) normalized embeddings (e.g. text)
        temperature: scale factor for logits (default 0.07)
        symmetric: if True, average loss in both directions (a→b and b→a)

    Returns:
        Scalar loss.
    """
    logits = (features_a @ features_b.t()) / max(temperature, 1e-8)
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)
    loss_ab = F.cross_entropy(logits, labels)
    if symmetric:
        loss_ba = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_ab + loss_ba)
    return loss_ab

