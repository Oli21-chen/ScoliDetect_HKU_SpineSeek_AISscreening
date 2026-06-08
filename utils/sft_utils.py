"""
Supervised Fine-Tuning utilities for SigLIP multimodal training.
Includes model definition, training functions, and helper utilities.
"""

import os
import json
import time
import warnings
import bisect

# Suppress NCCL warning on Windows (NCCL not supported, but we use DataParallel/gloo instead)
warnings.filterwarnings("ignore", message="PyTorch is not compiled with NCCL support")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

_AUC_SKLEARN_FAIL_WARNED = False


def _tensor_to_float_list(t: torch.Tensor) -> List[float]:
    """
  Convert a tensor to a Python float list for sklearn metrics.

  Prefer ``.tolist()`` over ``.numpy()`` so AUC still works when PyTorch was
  built against NumPy 1.x but the environment has NumPy 2.x (``tensor.numpy()``
  then raises "Numpy is not available" and AUC was silently reported as 0).
    """
    return [float(x) for x in t.detach().cpu().reshape(-1).tolist()]


def _roc_auc_from_tensors(probs: torch.Tensor, labels_binary: torch.Tensor) -> float:
    """ROC-AUC with safe tensor→sklearn conversion; 0.0 if undefined (single class)."""
    global _AUC_SKLEARN_FAIL_WARNED
    if not SKLEARN_AVAILABLE:
        return 0.0
    labels_flat = labels_binary.detach().cpu().reshape(-1).float()
    if labels_flat.numel() == 0 or len(torch.unique(labels_flat)) < 2:
        return 0.0
    probs_flat = probs.detach().cpu().reshape(-1).float()
    try:
        return float(
            roc_auc_score(
                _tensor_to_float_list(labels_flat),
                _tensor_to_float_list(probs_flat),
            )
        )
    except Exception as exc:
        if not _AUC_SKLEARN_FAIL_WARNED:
            warnings.warn(
                f"roc_auc_score failed ({exc!r}); AUC will show as 0.0 until fixed. "
                "If you see this after a NumPy upgrade, try `pip install 'numpy<2'` "
                "or reinstall PyTorch against your NumPy version.",
                UserWarning,
                stacklevel=2,
            )
            _AUC_SKLEARN_FAIL_WARNED = True
        return 0.0


def resolve_device(gpu_ids: Optional[List[int]] = None) -> torch.device:
    """
    Resolve the device to use for training.
    
    Args:
        gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2, 3]). If None, uses default device.
    
    Returns:
        Primary device (first GPU if multiple GPUs specified)
    """
    if torch.cuda.is_available():
        if gpu_ids is not None and len(gpu_ids) > 0:
            # Use the first GPU as primary device
            return torch.device(f"cuda:{gpu_ids[0]}")
        else:
            return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_multi_gpu(
    model: nn.Module,
    gpu_ids: Optional[List[int]] = None,
    use_distributed: bool = False,
) -> nn.Module:
    """
    Setup model for (optional) multi-GPU training/evaluation.

    This function is robust to checkpoints that were trained on more GPUs than
    are physically available on the current machine. Any invalid GPU IDs are
    filtered out so you can safely run with a single GPU even if the config
    says ``gpu_ids: [0, 1]``, etc.
    """
    if not torch.cuda.is_available():
        return model

    available_gpu_count = torch.cuda.device_count()
    available_gpu_ids = list(range(available_gpu_count))

    # If no gpu_ids are specified, default to "all available" (or just cuda:0 if one GPU).
    if gpu_ids is None or len(gpu_ids) == 0:
        if available_gpu_count > 1:
            print(f"Using DataParallel on {available_gpu_count} GPUs")
            model = model.to(torch.device("cuda:0"))
            model = nn.DataParallel(model)
        else:
            model = model.to(torch.device("cuda:0"))
        return model

    # Filter out any GPU IDs that are not available on this machine.
    filtered_gpu_ids = [gid for gid in gpu_ids if gid in available_gpu_ids]
    if len(filtered_gpu_ids) == 0:
        # Fall back to a single available GPU (typically cuda:0)
        print(
            f"Warning: requested gpu_ids={gpu_ids} are not fully available on this "
            f"machine (found {available_gpu_count} GPUs). Falling back to cuda:0."
        )
        model = model.to(torch.device("cuda:0"))
        return model

    # Single GPU after filtering → just move model, no DataParallel wrapper needed.
    if len(filtered_gpu_ids) == 1:
        device = torch.device(f"cuda:{filtered_gpu_ids[0]}")
        model = model.to(device)
        return model

    # Multiple GPUs after filtering
    if use_distributed:
        # DistributedDataParallel requires torch.distributed initialization
        # This is more complex and requires proper setup
        raise NotImplementedError(
            "DistributedDataParallel requires torch.distributed initialization. "
            "Use DataParallel for simpler multi-GPU setup."
        )
    else:
        # Use DataParallel on the subset of valid GPUs
        print(f"Using DataParallel on GPUs: {filtered_gpu_ids}")
        model = model.to(f"cuda:{filtered_gpu_ids[0]}")
        model = nn.DataParallel(model, device_ids=filtered_gpu_ids)
        return model


class BinaryFocalWithLabelSmoothingLoss(nn.Module):
    """
    Binary focal loss with optional label smoothing and logit regularization.
    
    This wraps BCE-with-logits and applies:
    - optional positive-class weight (same role as BCEWithLogitsLoss pos_weight)
    - label smoothing on targets
    - focal re-weighting (gamma, alpha)
    - optional L2 regularization on logits to discourage extreme values
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        label_smoothing: float = 0.05,
        logit_reg: float = 1e-4,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.logit_reg = logit_reg
        self.pos_weight = float(pos_weight) if pos_weight is not None and pos_weight > 0 else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure shapes are compatible (B, 1) vs (B,)
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.view(-1, 1)
            targets = targets.view_as(logits)

        # Label smoothing for binary targets in {0,1}
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            # 0 -> eps, 1 -> 1 - eps
            targets = targets * (1.0 - eps) + eps * (1.0 - targets)

        # BCE with logits (per-sample); pos_weight matches nn.BCEWithLogitsLoss (positive class scale)
        pw = None
        if self.pos_weight is not None:
            pw = logits.new_tensor(self.pos_weight)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )

        # Focal modulation
        with torch.no_grad():
            p_t = torch.sigmoid(logits)
            p_t = p_t * targets + (1.0 - p_t) * (1.0 - targets)
            focal_factor = (1.0 - p_t).pow(self.gamma)
            if self.alpha is not None:
                alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            else:
                alpha_t = 1.0

        loss = alpha_t * focal_factor * bce
        loss = loss.mean()

        # Optional logit regularization (penalize very large |logits|)
        if self.logit_reg > 0.0:
            loss = loss + self.logit_reg * logits.pow(2).mean()

        return loss


def pick_criterion(
    label: Optional[torch.Tensor],
    label_dim: int = 1,
    loss_type: str = "focal",
    pos_weight: Optional[float] = None,
    device: Optional[torch.device] = None,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = 0.25,
    focal_label_smoothing: float = 0.05,
    focal_logit_reg: float = 1e-4,
    focal_use_pos_weight: bool = False,
) -> Optional[nn.Module]:
    """
    Pick the appropriate loss criterion based on label type and dimension.
    
    Args:
        label: Sample label tensor (can be None)
        label_dim: Expected label dimension
        loss_type: For binary ("label_dim==1"), "bce" or "focal". Focal helps with class imbalance.
        pos_weight: Optional positive-class weight for BCE ("neg / pos"). For focal, used only if
            focal_use_pos_weight is True (same weighting as BCE on the underlying BCE term).
        device: Optional device for creating pos_weight tensor (BCE only; focal builds weights on logits' device).
        focal_gamma: Focal focusing parameter (default 2.0).
        focal_alpha: Focal class balance; None disables alpha_t weighting (1.0).
        focal_label_smoothing: Label smoothing epsilon for focal (0 disables).
        focal_logit_reg: L2 penalty on logits added to focal loss (0 disables).
        focal_use_pos_weight: If True and pos_weight is set, pass pos_weight into focal's BCE term.
        
    Returns:
        Loss criterion module or None if no labels
    """
    if label is None:
        return None
    
    if label_dim == 1:
        if loss_type == "focal":
            focal_pw: Optional[float] = None
            if focal_use_pos_weight and pos_weight is not None and pos_weight > 0.0:
                focal_pw = float(pos_weight)
            return BinaryFocalWithLabelSmoothingLoss(
                gamma=float(focal_gamma),
                alpha=focal_alpha,
                label_smoothing=float(focal_label_smoothing),
                logit_reg=float(focal_logit_reg),
                pos_weight=focal_pw,
            )
        bce_kwargs: Dict[str, Any] = {}
        if pos_weight is not None and pos_weight > 0.0:
            pw_tensor = torch.tensor([float(pos_weight)], dtype=torch.float32)
            if device is not None:
                pw_tensor = pw_tensor.to(device)
            bce_kwargs["pos_weight"] = pw_tensor
        return nn.BCEWithLogitsLoss(**bce_kwargs)
    elif label_dim > 1:
        # Multi-class classification
        return nn.CrossEntropyLoss()
    else:
        # Regression
        return nn.MSELoss()


def calculate_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Calculate accuracy, precision, recall, F1 score, and AUC ROC for binary classification.
    
    Args:
        predictions: Model predictions (logits for BCEWithLogitsLoss)
        labels: Ground truth labels (binary: 0 or 1)
    
    Returns:
        Dictionary with metrics: accuracy, precision, recall, f1, auc_roc
    """
    # Convert logits to probabilities and then to binary predictions
    if predictions.dim() > 1:
        predictions = predictions.squeeze(-1)
    if labels.dim() > 1:
        labels = labels.squeeze(-1)
    
    # Apply sigmoid and threshold at 0.5 for binary classification
    probs = torch.sigmoid(predictions)
    pred_binary = (probs >= 0.5).float()
    labels_binary = labels.float()
    
    # Calculate TP, TN, FP, FN
    tp = ((pred_binary == 1) & (labels_binary == 1)).sum().float()
    tn = ((pred_binary == 0) & (labels_binary == 0)).sum().float()
    fp = ((pred_binary == 1) & (labels_binary == 0)).sum().float()
    fn = ((pred_binary == 0) & (labels_binary == 1)).sum().float()
    
    # Count samples per class
    total_samples = tp + tn + fp + fn
    positive_samples = (labels_binary == 1).sum().float()
    negative_samples = (labels_binary == 0).sum().float()
    
    # Calculate overall metrics
    accuracy = (tp + tn) / (total_samples + 1e-8)
    precision = tp / (tp + fp + 1e-8)  # Positive class precision
    recall = tp / (tp + fn + 1e-8)  # Positive class recall (sensitivity)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Calculate per-class metrics
    positive_accuracy = recall  # Positive class accuracy = recall (TP / (TP + FN))
    negative_accuracy = tn / (tn + fp + 1e-8)  # Negative class accuracy (specificity)
    negative_precision = tn / (tn + fn + 1e-8)  # Negative class precision
    negative_recall = tn / (tn + fp + 1e-8)  # Negative class recall (specificity)
    specificity = negative_accuracy  # Same as negative accuracy
    
    # Class distribution
    positive_ratio = positive_samples / (total_samples + 1e-8)
    negative_ratio = negative_samples / (total_samples + 1e-8)
    
    auc_roc = _roc_auc_from_tensors(probs, labels_binary)

    return {
        # Overall metrics
        "accuracy": accuracy.item(),
        "precision": precision.item(),  # Positive class precision
        "recall": recall.item(),  # Positive class recall (sensitivity)
        "f1": f1.item(),
        "auc_roc": auc_roc,
        # Per-class metrics
        "positive_accuracy": positive_accuracy.item(),
        "negative_accuracy": negative_accuracy.item(),
        "negative_precision": negative_precision.item(),
        "negative_recall": negative_recall.item(),
        "specificity": specificity.item(),  # Same as negative_accuracy
        "sensitivity": recall.item(),  # Same as recall
        # Confusion matrix counts
        "tp": tp.item(),
        "tn": tn.item(),
        "fp": fp.item(),
        "fn": fn.item(),
        # Class distribution
        "total_samples": total_samples.item(),
        "positive_samples": positive_samples.item(),
        "negative_samples": negative_samples.item(),
        "positive_ratio": positive_ratio.item(),
        "negative_ratio": negative_ratio.item(),
    }


def _compute_binary_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, threshold: float):
    """
    Compute accuracy / precision / recall / F1 for a given probability threshold.
    Uses the same definitions as calculate_metrics in sft_utils.py, but with configurable threshold.
    """
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    if labels.dim() > 1:
        labels = labels.squeeze(-1)

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    labels = labels.float()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    tn = ((preds == 0) & (labels == 0)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / (total + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    auc_roc = _roc_auc_from_tensors(probs, labels)

    return {
        "threshold": threshold,
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "auc_roc": auc_roc,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module],
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    writer: Optional[Any] = None,
    global_step: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float], int]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss criterion (can be None)
        device: Device to use
        scheduler: Learning rate scheduler (optional)
        writer: TensorBoard SummaryWriter (optional)
        global_step: Current global step count
        config: Optional config dict for training options
    
    Returns:
        Tuple of (average_loss, metrics_dict, updated_global_step)
    """
    model.train()
    total_loss = 0.0
    steps = 0
    
    # Accumulate predictions and labels for metrics calculation
    all_predictions = []
    all_labels = []
    
    total_batches = len(dataloader)
    print(f"  Starting training: {total_batches} batches")
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start_time = time.time()
        
        video = batch["video"].to(device)
        knowledge_map = batch["knowledge_map"].to(device)
        texts = batch.get("texts", None)  # Processed text strings from collate function
        km_indices = batch.get("km_indices", None)
        video_indices = batch.get("video_indices", None)
        if km_indices is not None:
            km_indices = km_indices.to(device)
        if video_indices is not None:
            video_indices = video_indices.to(device)

        # Forward pass
        model_out = model(
            video, knowledge_map, texts,
            km_indices=km_indices, video_indices=video_indices, 
            return_aux=True,
        )
        aux_components_tensor = None
        if isinstance(model_out, tuple) and len(model_out) == 3:
            predictions, aux_loss, aux_components_tensor = model_out
        elif isinstance(model_out, tuple):
            predictions, aux_loss = model_out
        else:
            predictions, aux_loss = model_out, None
        # DataParallel gathers per-replica scalar aux losses into a 1D tensor.
        # Ensure aux_loss is a scalar before combining with main loss.
        if aux_loss is not None and torch.is_tensor(aux_loss) and aux_loss.dim() > 0:
            aux_loss = aux_loss.mean()
        if aux_components_tensor is not None and torch.is_tensor(aux_components_tensor):
            # Expected order: [infonce, barlow, vicreg, raw_total, weighted_total]
            # DataParallel may stack replica outputs into shape (n_replica, 5).
            if aux_components_tensor.dim() > 1:
                aux_components_tensor = aux_components_tensor.mean(dim=0)

        # Skip batch if predictions contain NaN (avoid NaNs propagating in loss/backward)
        if torch.isnan(predictions).any():
            nan_count = torch.isnan(predictions).sum().item()
            if predictions.dim() == 2 and predictions.shape[1] == 1:
                nan_mask = torch.isnan(predictions.squeeze(-1))
            else:
                nan_mask = torch.isnan(predictions.squeeze())
            source_files = batch.get("source_file", ["unknown"] * predictions.shape[0])
            nan_indices = torch.where(nan_mask)[0].cpu().tolist()
            nan_source_files = [source_files[i] if i < len(source_files) else "unknown" for i in nan_indices]
            print(f"Warning: Skipping batch with {nan_count} NaN predictions (step {global_step})")
            print(f"  NaN samples: {nan_source_files}")
            continue

        # Compute loss
        if criterion is not None and "label" in batch:
            labels = batch["label"].to(device)
            # Ensure predictions and labels have compatible shapes
            if predictions.dim() == 1 and labels.dim() == 1:
                predictions = predictions.unsqueeze(-1)
            elif predictions.dim() == 2 and labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            
            loss = criterion(predictions, labels)
            main_loss = loss
            if aux_loss is not None:
                loss = loss + aux_loss

            # Accumulate for metrics
            batch_preds = predictions.detach().cpu()
            batch_lbls = labels.detach().cpu()
            all_predictions.append(batch_preds)
            all_labels.append(batch_lbls)
            
            # Calculate batch-level metrics for printing
            batch_metrics = calculate_metrics(batch_preds, batch_lbls)

            # Optional debug: print labels and predictions for this batch
            if config is not None and config.get("debug_print_labels", False):
                label_list = batch_lbls.view(-1).tolist()
                # Use sigmoid to get probabilities from logits
                probs = torch.sigmoid(batch_preds.view(-1))
                prob_list = probs.tolist()
                pred_binary = (probs >= 0.5).float().tolist()
                print(f"    Labels (batch {batch_idx + 1}): {label_list}")
                print(f"    Pred probs (batch {batch_idx + 1}): {prob_list}")
                print(f"    Pred binary (>=0.5) (batch {batch_idx + 1}): {pred_binary}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            max_grad_norm = config.get("max_grad_norm") if config else None
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            steps += 1
            global_step += 1
            
            # Calculate batch time
            batch_time = time.time() - batch_start_time
            
            # Print step-by-step details for each batch (focus on AUC instead of F1/Recall)
            current_avg_loss = total_loss / steps
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else 0.0
            print_str = (
                f"  Batch {batch_idx + 1}/{total_batches} | "
                f"Loss: {loss.item():.6f} | "
                f"Avg Loss: {current_avg_loss:.6f} | "
                f"Acc: {batch_metrics.get('accuracy', 0.0):.4f} | "
                f"AUC: {batch_metrics.get('auc_roc', 0.0):.4f} | "
                f"LR: {current_lr:.6e} | "
                f"Time: {batch_time:.3f}s"
            )
            print(print_str)

            # Log to tensorboard step by step
            if writer is not None:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Loss/train_main", main_loss.item(), global_step)
                if aux_loss is not None:
                    writer.add_scalar("Loss/train_aux_align", aux_loss.item(), global_step)
                else:
                    # Keep the curve continuous when aux loss is disabled/unavailable.
                    writer.add_scalar("Loss/train_aux_align", 0.0, global_step)
                if aux_components_tensor is not None:
                    writer.add_scalar("Loss/train_aux_infonce_raw", aux_components_tensor[0].item(), global_step)
                    writer.add_scalar("Loss/train_aux_barlow_raw", aux_components_tensor[1].item(), global_step)
                    writer.add_scalar("Loss/train_aux_vicreg_raw", aux_components_tensor[2].item(), global_step)
                    writer.add_scalar("Loss/train_aux_raw_total", aux_components_tensor[3].item(), global_step)
                    writer.add_scalar("Loss/train_aux_weighted_total", aux_components_tensor[4].item(), global_step)
                if scheduler is not None:
                    writer.add_scalar("LR", current_lr, global_step)
        else:
            # Skip batch if no criterion or no labels
            print(f"  Batch {batch_idx + 1}/{total_batches}: Skipped (no labels or criterion)")
            continue
    
    # Calculate metrics
    metrics = {}
    if len(all_predictions) > 0:
        all_preds = torch.cat(all_predictions, dim=0)
        all_lbls = torch.cat(all_labels, dim=0)
        metrics = calculate_metrics(all_preds, all_lbls)
        
        # Log epoch-level metrics to tensorboard
        # Best practice: Log aggregated metrics at epoch level (more stable and meaningful)
        # For training, only track accuracy and AUC in TensorBoard.
        if writer is not None:
            writer.add_scalar("Metrics/train/accuracy", metrics["accuracy"], global_step)
            writer.add_scalar("Metrics/train/auc_roc", metrics.get("auc_roc", 0.0), global_step)
    
    avg_loss = total_loss / max(steps, 1)
    # Log epoch-averaged loss (complement to step-level loss for smoother view)
    if writer is not None:
        writer.add_scalar("Loss/train_epoch", avg_loss, global_step)
    
    # Print epoch summary (focus on accuracy and AUC for train set)
    print(f"  Training Summary:")
    print(f"    Total batches processed: {steps}/{total_batches}")
    print(f"    Average Loss: {avg_loss:.6f}")
    if len(metrics) > 0:
        print(f"    Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        print(f"    AUC-ROC: {metrics.get('auc_roc', 0.0):.4f}")
    
    return avg_loss, metrics, global_step


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
    writer: Optional[Any] = None,
    global_step: int = 0,
    verbose: bool = True,
    config: Optional[Dict[str, Any]] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate for one epoch.
    
    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        criterion: Loss criterion (can be None)
        device: Device to use
        writer: TensorBoard SummaryWriter (optional)
        global_step: Current global step count
        verbose: If False, suppress batch/summary prints (metrics still computed)
        config: Optional config dict (e.g. debug_print_labels)
        scheduler: Optional LR scheduler (for print format parity with train_one_epoch)
    
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    steps = 0
    
    # Accumulate predictions and labels for metrics calculation
    all_predictions = []
    all_labels = []
    
    total_batches = len(dataloader)
    if verbose:
        print(f"  Starting validation: {total_batches} batches")
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start_time = time.time()
        
        video = batch["video"].to(device)
        knowledge_map = batch["knowledge_map"].to(device)
        texts = batch.get("texts", None)  # Processed text strings from collate function
        km_indices = batch.get("km_indices", None)
        video_indices = batch.get("video_indices", None)
        if km_indices is not None:
            km_indices = km_indices.to(device)
        if video_indices is not None:
            video_indices = video_indices.to(device)
        
    
        # Forward pass
        predictions = model(
            video, knowledge_map, texts,
           km_indices=km_indices, video_indices=video_indices
        ) 
        
        # Skip batch if predictions contain NaN
        if torch.isnan(predictions).any():
            nan_count = torch.isnan(predictions).sum().item()
            # Handle different prediction shapes: (batch_size,) or (batch_size, 1)
            if predictions.dim() == 2 and predictions.shape[1] == 1:
                nan_mask = torch.isnan(predictions.squeeze(-1))
            else:
                nan_mask = torch.isnan(predictions.squeeze())
            source_files = batch.get("source_files", batch.get("source_file", ["unknown"] * predictions.shape[0]))
            if isinstance(source_files, str):
                source_files = [source_files]
            nan_indices = torch.where(nan_mask)[0].cpu().tolist()
            nan_source_files = [source_files[i] if i < len(source_files) else "unknown" for i in nan_indices]
            print(f"  Batch {batch_idx + 1}/{total_batches}: Warning - Skipping batch with {nan_count} NaN predictions")
            print(f"    NaN samples: {nan_source_files}")
            continue
        
        # Compute loss
        if criterion is not None and "label" in batch:
            labels = batch["label"].to(device)
            # Ensure predictions and labels have compatible shapes
            if predictions.dim() == 1 and labels.dim() == 1:
                predictions = predictions.unsqueeze(-1)
            elif predictions.dim() == 2 and labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            steps += 1
            
            # Calculate batch-level metrics for printing
            batch_preds = predictions.detach().cpu()
            batch_lbls = labels.detach().cpu()
            batch_metrics = calculate_metrics(batch_preds, batch_lbls)

            # Optional debug: print labels and predictions for this validation batch
            if config is not None and config.get("debug_print_labels", False):
                label_list = batch_lbls.view(-1).tolist()
                probs = torch.sigmoid(batch_preds.view(-1))
                prob_list = probs.tolist()
                pred_binary = (probs >= 0.5).float().tolist()
                print(f"    [VAL] Labels (batch {batch_idx + 1}): {label_list}")
                print(f"    [VAL] Pred probs (batch {batch_idx + 1}): {prob_list}")
                print(f"    [VAL] Pred binary (>=0.5) (batch {batch_idx + 1}): {pred_binary}")
            
            # Accumulate for epoch-level metrics
            all_predictions.append(batch_preds)
            all_labels.append(batch_lbls)
            
            # Calculate batch time
            batch_time = time.time() - batch_start_time
            
            # Same per-batch print format as train_one_epoch
            if verbose:
                current_avg_loss = total_loss / steps
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else 0.0
                print(
                    f"  Batch {batch_idx + 1}/{total_batches} | "
                    f"Loss: {loss.item():.6f} | "
                    f"Avg Loss: {current_avg_loss:.6f} | "
                    f"Acc: {batch_metrics.get('accuracy', 0.0):.4f} | "
                    f"AUC: {batch_metrics.get('auc_roc', 0.0):.4f} | "
                    f"LR: {current_lr:.6e} | "
                    f"Time: {batch_time:.3f}s"
                )
        else:
            # Skip batch if no criterion or no labels
            if verbose:
                print(f"  Batch {batch_idx + 1}/{total_batches}: Skipped (no labels or criterion)")
            continue
    
    # Calculate epoch-level metrics
    metrics = {}
    if len(all_predictions) > 0:
        all_preds = torch.cat(all_predictions, dim=0)
        all_lbls = torch.cat(all_labels, dim=0)
        metrics = calculate_metrics(all_preds, all_lbls)
        
        # Log epoch-level metrics to tensorboard
        # Best practice: Log aggregated validation metrics at epoch level.
        # For validation, only track accuracy and AUC in TensorBoard.
        if writer is not None:
            writer.add_scalar("Metrics/val/accuracy", metrics["accuracy"], global_step)
            writer.add_scalar("Metrics/val/auc_roc", metrics.get("auc_roc", 0.0), global_step)
    
    avg_loss = total_loss / max(steps, 1)
    # Log epoch-averaged validation loss
    if writer is not None:
        writer.add_scalar("Loss/val_epoch", avg_loss, global_step)
    
    # Same epoch summary format as train_one_epoch
    if verbose:
        print(f"  Validation Summary:")
        print(f"    Total batches processed: {steps}/{total_batches}")
        print(f"    Average Loss: {avg_loss:.6f}")
        if len(metrics) > 0:
            print(f"    Accuracy: {metrics.get('accuracy', 0.0):.4f}")
            print(f"    AUC-ROC: {metrics.get('auc_roc', 0.0):.4f}")
    
    return avg_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_path: str,
    config: Dict[str, Any],
    best_acc: Optional[float] = None,
    best_auc: Optional[float] = None,
    best_test_acc: Optional[float] = None,
    best_test_auc: Optional[float] = None,
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save (handles DataParallel wrapped models)
        optimizer: Optimizer state
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        checkpoint_path: Path to save checkpoint
        config: Configuration dictionary
        best_acc: Best validation accuracy (optional)
        best_auc: Best validation AUC-ROC (optional)
        best_test_acc: Best external test accuracy (optional)
        best_test_auc: Best external test AUC-ROC (optional)
    """
    # Handle DataParallel wrapped models
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }
    if best_acc is not None:
        checkpoint["best_acc"] = best_acc
    if best_auc is not None:
        checkpoint["best_auc"] = best_auc
    if best_test_acc is not None:
        checkpoint["best_test_acc"] = best_test_acc
    if best_test_auc is not None:
        checkpoint["best_test_auc"] = best_test_auc
    torch.save(checkpoint, checkpoint_path)


def _extract_subject_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract subject ID from CSV or video filename.
    Same logic as in data_sampler.py for consistency.
    """
    import re
    # Remove extension
    base_name = os.path.splitext(filename)[0]
    
    # Pattern 1: Direct numeric at start or after underscore
    match = re.search(r'(?:^|_)(\d+)(?:_|$)', base_name)
    if match:
        return match.group(1)
    
    # Pattern 2: Any sequence of digits
    match = re.search(r'\d+', base_name)
    if match:
        return match.group(0)
    
    return None


def _is_positive_label(label_value, binary_threshold: float = 11.0):
    """
    Determine if a label_value is positive based on binary_threshold.
    Returns (is_positive: bool, is_valid: bool)
    """
    if label_value is None:
        return False, False
    
    # Handle list/array: use max value
    if isinstance(label_value, (list, tuple)):
        try:
            max_val = max(float(x) for x in label_value)
            return max_val >= binary_threshold, True
        except (ValueError, TypeError):
            return False, False
    
    # Handle single value
    try:
        val = float(label_value)
        # If already binary (0.0 or 1.0), use it directly
        if val == 0.0 or val == 1.0:
            return val >= 0.5, True
        # Otherwise apply threshold
        return val >= binary_threshold, True
    except (ValueError, TypeError):
        return False, False


def count_dataset_pos_neg(dataset_split, binary_threshold: float = 11.0):
    """
    Count positive and negative samples in a dataset split.
    Optimized to access labels directly without loading full samples.
    
    Args:
        dataset_split: Dataset split (train or val) - can be Subset or Dataset
        binary_threshold: Threshold for binary classification
    
    Returns:
        Tuple of (pos_count, neg_count, no_label_count)
    """
    pos_count = 0
    neg_count = 0
    no_label_count = 0
    
    # Try to access underlying dataset and indices for faster label access
    # random_split creates a Subset which has .dataset and .indices attributes
    if hasattr(dataset_split, 'dataset') and hasattr(dataset_split, 'indices'):
        # Access underlying dataset
        underlying_dataset = dataset_split.dataset
        indices = dataset_split.indices
        
        # ConcatDataset of PKL datasets: resolve global index to sub-dataset + local index
        if hasattr(underlying_dataset, 'datasets') and hasattr(underlying_dataset, 'cumulative_sizes'):
            sub_datasets = underlying_dataset.datasets
            cumulative_sizes = underlying_dataset.cumulative_sizes
            if all(hasattr(ds, 'patch_metadata') for ds in sub_datasets):
                for idx in indices:
                    dataset_idx = bisect.bisect_right(cumulative_sizes, idx)
                    if dataset_idx == 0:
                        local_idx = idx
                    else:
                        local_idx = idx - cumulative_sizes[dataset_idx - 1]
                    ds = sub_datasets[dataset_idx]
                    if local_idx < len(ds.patch_metadata):
                        patch_info = ds.patch_metadata[local_idx]
                        label_value = patch_info.get('label_value')
                        if label_value is None:
                            label_value = patch_info.get('label')
                        is_positive, is_valid = _is_positive_label(label_value, binary_threshold)
                        if is_valid:
                            if is_positive:
                                pos_count += 1
                            else:
                                neg_count += 1
                        else:
                            no_label_count += 1
                    else:
                        no_label_count += 1
                return pos_count, neg_count, no_label_count
        
        # Check for PKL dataset (has patch_metadata) - single dataset
        if hasattr(underlying_dataset, 'patch_metadata'):
            # PKL dataset: use label_value with max comparison
            for idx in indices:
                if idx < len(underlying_dataset.patch_metadata):
                    patch_info = underlying_dataset.patch_metadata[idx]
                    # Prefer label_value over label for threshold comparison.
                    # Use explicit None check so valid zero values are preserved.
                    label_value = patch_info.get('label_value')
                    if label_value is None:
                        label_value = patch_info.get('label')
                    
                    is_positive, is_valid = _is_positive_label(label_value, binary_threshold)
                    if is_valid:
                        if is_positive:
                            pos_count += 1
                        else:
                            neg_count += 1
                    else:
                        no_label_count += 1
                else:
                    no_label_count += 1
            return pos_count, neg_count, no_label_count
        
        # Check if dataset has label_map for direct access
        if hasattr(underlying_dataset, 'label_map') and underlying_dataset.label_map is not None:
            # Fast path: access labels directly from label_map without loading samples
            # Check for sample_indices first (used by SigLIPFullGaitDataset_v2)
            if hasattr(underlying_dataset, 'sample_indices'):
                # Use sample_indices which already contains label information
                for idx in indices:
                    if idx < len(underlying_dataset.sample_indices):
                        sample_info = underlying_dataset.sample_indices[idx]
                        label_value = sample_info.get('label')
                        
                        is_positive, is_valid = _is_positive_label(label_value, binary_threshold)
                        if is_valid:
                            if is_positive:
                                pos_count += 1
                            else:
                                neg_count += 1
                        else:
                            no_label_count += 1
                    else:
                        no_label_count += 1
                return pos_count, neg_count, no_label_count
            elif hasattr(underlying_dataset, 'samples'):
                # Fallback to samples (used by fullgait datasets)
                for idx in indices:
                    if idx < len(underlying_dataset.samples):
                        csv_path, _ = underlying_dataset.samples[idx]
                        csv_filename = os.path.basename(csv_path)
                        
                        # Extract subject_id using same logic as dataset
                        subject_id = _extract_subject_id_from_filename(csv_filename)
                        
                        if subject_id is not None:
                            subject_key = str(int(subject_id))  # Convert to int then string (matches dataset logic)
                            label_value = underlying_dataset.label_map.get(subject_key)
                            
                            is_positive, is_valid = _is_positive_label(label_value, binary_threshold)
                            if is_valid:
                                if is_positive:
                                    pos_count += 1
                                else:
                                    neg_count += 1
                            else:
                                no_label_count += 1
                        else:
                            no_label_count += 1
                    else:
                        no_label_count += 1
                return pos_count, neg_count, no_label_count
    
    # Fallback: slower path - access samples one by one
    # Only use this if fast path doesn't work (e.g., dataset doesn't have label_map)
    print("Warning: Using slower label counting method. Consider optimizing dataset structure.")
    for idx in range(len(dataset_split)):
        try:
            sample = dataset_split[idx]
            label = sample.get("label")
            
            # Handle torch.Tensor
            if isinstance(label, torch.Tensor):
                label_value = label.item() if label.numel() == 1 else float(label)
            else:
                label_value = label
            
            is_positive, is_valid = _is_positive_label(label_value, binary_threshold)
            if is_valid:
                if is_positive:
                    pos_count += 1
                else:
                    neg_count += 1
            else:
                no_label_count += 1
        except Exception:
            no_label_count += 1
    
    return pos_count, neg_count, no_label_count


def print_label_distribution(train_dataset, val_dataset, binary_threshold: float = 11.0):
    """
    Print label distribution for train and validation sets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        binary_threshold: Threshold for binary classification
    """
    # Count for train set
    train_pos, train_neg, train_no_label = count_dataset_pos_neg(train_dataset, binary_threshold)
    print(f"\nTrain set label distribution:")
    print(f"  Positive (>= {binary_threshold}): {train_pos} ({train_pos/len(train_dataset)*100:.2f}%)")
    print(f"  Negative (< {binary_threshold}): {train_neg} ({train_neg/len(train_dataset)*100:.2f}%)")
    if train_no_label > 0:
        print(f"  No label: {train_no_label} ({train_no_label/len(train_dataset)*100:.2f}%)")
    
    # Count for val set
    val_pos, val_neg, val_no_label = count_dataset_pos_neg(val_dataset, binary_threshold)
    print(f"\nValidation set label distribution:")
    print(f"  Positive (>= {binary_threshold}): {val_pos} ({val_pos/len(val_dataset)*100:.2f}%)")
    print(f"  Negative (< {binary_threshold}): {val_neg} ({val_neg/len(val_dataset)*100:.2f}%)")
    if val_no_label > 0:
        print(f"  No label: {val_no_label} ({val_no_label/len(val_dataset)*100:.2f}%)")
    
    # Total counts
    total_pos = train_pos + val_pos
    total_neg = train_neg + val_neg
    total_samples = len(train_dataset) + len(val_dataset)
    print(f"\nTotal dataset label distribution:")
    print(f"  Positive: {total_pos} ({total_pos/total_samples*100:.2f}%)")
    print(f"  Negative: {total_neg} ({total_neg/total_samples*100:.2f}%)")
    print()

    # Return counts so callers can derive class weights (e.g., BCE pos_weight)
    return {
        "train_pos": train_pos,
        "train_neg": train_neg,
        "train_no_label": train_no_label,
        "val_pos": val_pos,
        "val_neg": val_neg,
        "val_no_label": val_no_label,
        "total_pos": total_pos,
        "total_neg": total_neg,
        "total_samples": total_samples,
        "binary_threshold": binary_threshold,
    }


def load_pretrained_checkpoint(
    model: nn.Module,
    pretrained_checkpoint_path: str,
    device: torch.device,
    text_trainable: bool = True,
    load_text_encoder_weights: bool = True,
    load_text_proj_from_pretrained: bool = True,
) -> nn.Module:
    """
    Load pretrained checkpoint weights into model (only encoder weights).
    
    Args:
        model: Model to load weights into
        pretrained_checkpoint_path: Path to pretrained checkpoint
        device: Device to load checkpoint on
        text_trainable: If False, do not load weights into text_encoder (keeps encoder frozen with its default pretrained weights).
        load_text_encoder_weights: If False, never apply checkpoint weights to ``text_encoder.*``
            (HF/SentenceTransformer defaults are kept).
        load_text_proj_from_pretrained: If False, do not load ``text_proj.*`` or ``text_norm.*`` from the
            checkpoint so they stay at freshly initialized values and train as usual in the new run.
            Set True to restore projection + DyT from the same checkpoint as the encoders.
    
    Returns:
        Model with loaded weights
    """
    if not os.path.exists(pretrained_checkpoint_path):
        print(f"Warning: Pretrained checkpoint not found: {pretrained_checkpoint_path}")
        print("  Starting training from scratch.")
        return model
    
    print(f"\nLoading pretrained checkpoint from: {pretrained_checkpoint_path}")
    pretrained_checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
    
    # Get model state dict (handle DataParallel wrapping)
    if "model_state_dict" in pretrained_checkpoint:
        pretrained_state_dict = pretrained_checkpoint["model_state_dict"]
    elif "state_dict" in pretrained_checkpoint:
        pretrained_state_dict = pretrained_checkpoint["state_dict"]
    else:
        pretrained_state_dict = pretrained_checkpoint
    
    # Remove DataParallel prefix if present
    pretrained_state_dict = {k.replace("module.", ""): v for k, v in pretrained_state_dict.items()}
    
    # Get current model state dict (before DataParallel)
    model_state_dict = model.state_dict()
    
    # Map encoder weights from pretrained model to SFTRegressor (video, km, text adapter as configured).
    # Skip text_encoder.* when frozen or when caller disables loading (keep HF default weights).
    # Optionally skip text_proj.* and text_norm.* so they stay at init (train in the new task).
    loaded_keys = []
    skipped_keys = []
    encoder_prefixes = ("video_encoder.", "km_encoder.", "text_encoder.", "text_proj.", "text_norm.")
    skip_text_encoder_ckpt = (not text_trainable) or (not load_text_encoder_weights)
    for key, value in pretrained_state_dict.items():
        if skip_text_encoder_ckpt and key.startswith("text_encoder."):
            reason = "load_text_encoder_weights=False" if not load_text_encoder_weights else "text_encoder frozen"
            skipped_keys.append(f"{key} ({reason}, not loading)")
            continue
        if not load_text_proj_from_pretrained and (
            key.startswith("text_proj.") or key.startswith("text_norm.")
        ):
            skipped_keys.append(f"{key} (text_proj/text_norm kept at init, not loading)")
            continue
        if any(key.startswith(prefix) for prefix in encoder_prefixes):
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    model_state_dict[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch: {model_state_dict[key].shape} vs {value.shape})")
            else:
                skipped_keys.append(f"{key} (not found in model)")
    
    # Load the updated state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    text_loaded = sum(
        1
        for k in loaded_keys
        if k.startswith("text_encoder.")
        or k.startswith("text_proj.")
        or k.startswith("text_norm.")
    )
    print(f"  Loaded {len(loaded_keys)} encoder weights from pretrained checkpoint" + (
        f" (including {text_loaded} text-related)" if text_loaded else ""
    ))
    if skipped_keys:
        print(f"  Skipped {len(skipped_keys)} keys:")
        for key in skipped_keys[:10]:  # Show first 10
            print(f"    - {key}")
        if len(skipped_keys) > 10:
            print(f"    ... and {len(skipped_keys) - 10} more")

    # Report module-wise checkpoint coverage so transfer quality is easy to diagnose.
    module_prefixes = (
        ("video_encoder", "video_encoder."),
        ("km_encoder", "km_encoder."),
        ("text_encoder", "text_encoder."),
        ("text_proj", "text_proj."),
        ("text_norm", "text_norm."),
    )
    print("  Pretrained coverage by module:")
    for module_name, prefix in module_prefixes:
        model_key_count = sum(1 for k in model_state_dict.keys() if k.startswith(prefix))
        loaded_key_count = sum(1 for k in loaded_keys if k.startswith(prefix))
        coverage = (100.0 * loaded_key_count / model_key_count) if model_key_count > 0 else 0.0
        print(f"    - {module_name}: {loaded_key_count}/{model_key_count} ({coverage:.1f}%)")
    
    # Print pretrained checkpoint info if available
    if "epoch" in pretrained_checkpoint:
        print(f"  Pretrained checkpoint epoch: {pretrained_checkpoint.get('epoch', 'N/A')}")
    if "loss" in pretrained_checkpoint:
        print(f"  Pretrained checkpoint loss: {pretrained_checkpoint.get('loss', 'N/A'):.6f}")
    
    return model


def print_model_info(model: nn.Module, config: Dict[str, Any]):
    """
    Print model information.
    
    Args:
        model: Model to print info for
        config: Configuration dictionary
    """
    print(f"\nModel created:")
    print(f"  Video encoder: {config.get('video_encoder_type', 'vivit')}")
    print(f"  Knowledge encoder: {config.get('km_encoder_type', 'baseline')}")
    print(f"  Text encoder: {'enabled' if config.get('use_text', True) else 'disabled'}")
    if isinstance(model, nn.DataParallel):
        print(f"  Multi-GPU: DataParallel on {len(model.device_ids)} GPUs")
        total_params = sum(p.numel() for p in model.module.parameters())
        trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: Dict[str, Any],
    steps_per_epoch: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: Model to create optimizer for
        config: Configuration dictionary
        steps_per_epoch: Number of steps per epoch
    
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Use Adam (set optimizer to "adamw" in config for decoupled weight decay)
    optimizer_name = config.get("optimizer", "adamw").lower()
    base_lr = float(config["learning_rate"])
    weight_decay = float(config.get("weight_decay", 0.01))
    use_discriminative_lr = bool(config.get("use_discriminative_lr", False))

    if use_discriminative_lr:
        encoder_lr = float(config.get("encoder_learning_rate", base_lr))
        head_lr = float(config.get("head_learning_rate", base_lr))
        text_lr = float(config.get("text_learning_rate", encoder_lr))

        named_params = list(model.named_parameters())
        encoder_prefixes = ("video_encoder.", "km_encoder.")
        text_prefixes = ("text_encoder.", "text_proj.", "text_norm.")

        encoder_params = [
            p for n, p in named_params if any(n.startswith(prefix) for prefix in encoder_prefixes)
        ]
        text_params = [
            p for n, p in named_params if any(n.startswith(prefix) for prefix in text_prefixes)
        ]
        head_params = [
            p
            for n, p in named_params
            if not any(n.startswith(prefix) for prefix in encoder_prefixes + text_prefixes)
        ]

        param_groups = []
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay})
        if text_params:
            param_groups.append({"params": text_params, "lr": text_lr, "weight_decay": weight_decay})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.AdamW(param_groups)

        print("Optimizer parameter groups (discriminative LR):")
        print(f"  Encoder LR: {encoder_lr:.2e} (video/km)")
        print(f"  Text LR:    {text_lr:.2e} (text encoder/proj)")
        print(f"  Head LR:    {head_lr:.2e} (fusion/regressor)")
        print(f"  Weight decay: {weight_decay:.2e}")
    else:
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=base_lr,
                weight_decay=config.get("weight_decay", 0.0),
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=base_lr,
                weight_decay=config.get("weight_decay", 0.01),
            )
    
    # Setup learning rate scheduler: linear warmup -> cosine decay
    total_steps = config["num_epochs"] * steps_per_epoch
    warmup_steps = max(1, int(total_steps * config.get("warmup_ratio", 0.1)))
    cosine_steps = max(1, total_steps - warmup_steps)
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,  # Start at 0.1% (1e-3) of base learning rate
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=5e-7  # Minimum learning rate
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    
    print(f"Learning rate schedule:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps} ({warmup_steps/total_steps*100:.1f}%)")
    print(f"  Cosine decay steps: {cosine_steps}")
    
    return optimizer, scheduler


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _set_requires_grad_by_prefix(model: nn.Module, prefixes: Tuple[str, ...], requires_grad: bool):
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad = requires_grad


def _set_requires_grad_for_all(model: nn.Module, requires_grad: bool):
    for param in model.parameters():
        param.requires_grad = requires_grad


def _unfreeze_last_n_blocks(module: nn.Module, n_blocks: int) -> int:
    if n_blocks <= 0:
        return 0
    candidates = []
    for attr in ("blocks", "layers"):
        if hasattr(module, attr):
            obj = getattr(module, attr)
            if isinstance(obj, nn.ModuleList):
                candidates.append(obj)
    if hasattr(module, "transformer"):
        transformer = getattr(module, "transformer")
        for attr in ("blocks", "layers"):
            if hasattr(transformer, attr):
                obj = getattr(transformer, attr)
                if isinstance(obj, nn.ModuleList):
                    candidates.append(obj)

    if not candidates:
        return 0
    blocks = candidates[0]
    n = min(len(blocks), n_blocks)
    for block in list(blocks)[-n:]:
        for p in block.parameters():
            p.requires_grad = True
    return n


def apply_staged_finetune_policy(model: nn.Module, config: Dict[str, Any], epoch: int) -> str:
    """
    Apply staged freeze/unfreeze policy for pretrained finetuning.

    Stages:
      1) head_only_epochs: train fusion/regressor heads only
      2) partial_unfreeze_until_epoch: unfreeze top encoder blocks
      3) remaining epochs: unfreeze full encoders
    """
    if not bool(config.get("enable_staged_finetune", False)):
        return "disabled"

    inner_model = _unwrap_model(model)
    head_only_epochs = int(config.get("head_only_epochs", 0))
    partial_unfreeze_until_epoch = int(config.get("partial_unfreeze_until_epoch", head_only_epochs))
    unfreeze_top_n_blocks = int(config.get("unfreeze_top_n_blocks", 1))
    keep_text_frozen = not bool(config.get("text_trainable", True))

    _set_requires_grad_for_all(inner_model, True)
    if keep_text_frozen:
        _set_requires_grad_by_prefix(inner_model, ("text_encoder.",), False)

    if epoch <= head_only_epochs:
        _set_requires_grad_by_prefix(inner_model, ("video_encoder.", "km_encoder."), False)
        stage = "head_only"
    elif epoch <= partial_unfreeze_until_epoch:
        _set_requires_grad_by_prefix(inner_model, ("video_encoder.", "km_encoder."), False)
        video_unfrozen = _unfreeze_last_n_blocks(inner_model.video_encoder, unfreeze_top_n_blocks)
        km_unfrozen = _unfreeze_last_n_blocks(inner_model.km_encoder, unfreeze_top_n_blocks)
        if video_unfrozen == 0:
            _set_requires_grad_by_prefix(inner_model, ("video_encoder.",), True)
        if km_unfrozen == 0:
            _set_requires_grad_by_prefix(inner_model, ("km_encoder.",), True)
        stage = f"partial_unfreeze(top_blocks={unfreeze_top_n_blocks})"
    else:
        _set_requires_grad_by_prefix(inner_model, ("video_encoder.", "km_encoder."), True)
        stage = "full_unfreeze"

    trainable_params = sum(p.numel() for p in inner_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in inner_model.parameters())
    print(
        f"  Finetune stage @ epoch {epoch}: {stage} | "
        f"trainable params: {trainable_params:,}/{total_params:,}"
    )
    return stage


def setup_directories(config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Setup checkpoint and log directories with timestamps.
    
    Args:
        config: Configuration dictionary
            - save_dir: Base directory for checkpoints
            - log_dir: Base directory for logs (optional, defaults to "./logs")
            - run_name: Prefix for directory names (optional, defaults to "sft")
    
    Returns:
        Tuple of (checkpoint_dir, log_dir)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = config.get("run_name", "sft")  # Default to "sft" if not specified
    
    ckpt_dir = os.path.join(config["save_dir"], f"{run_name}_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoint directory: {ckpt_dir}")
    
    log_dir = os.path.join(config.get("log_dir", "./logs"), f"{run_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"TensorBoard log dir: {log_dir}")
    
    return ckpt_dir, log_dir


def load_resume_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    resume_checkpoint_path: str,
    device: torch.device,
    steps_per_epoch: int,
    text_trainable: bool = True,
) -> Tuple[int, int, float, float, float, float, float]:
    """
    Load checkpoint for resuming training.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to advance
        resume_checkpoint_path: Path to resume checkpoint
        device: Device to load checkpoint on
        steps_per_epoch: Number of steps per epoch
        text_trainable: If False, do not load weights into text_encoder (keeps current frozen encoder state).
    
    Returns:
        Tuple of (start_epoch, global_step, best_val_loss, best_acc, best_auc, best_test_acc, best_test_auc)
    """
    if not os.path.exists(resume_checkpoint_path):
        print(f"Warning: Resume checkpoint not found: {resume_checkpoint_path}")
        print("  Starting fresh training instead.")
        return 1, 0, float('inf'), 0.0, 0.0, 0.0, 0.0
    
    print(f"\n{'='*80}")
    print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
    print(f"{'='*80}")
    
    # PyTorch 2.6 defaults to weights_only=True, which can break loading
    # training checkpoints that include non-tensor metadata (e.g., numpy scalars).
    # This resume path expects a full checkpoint; set weights_only=False.
    checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)

    # Load model weights
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        # When text_trainable is False, do not overwrite text_encoder with checkpoint weights
        if not text_trainable:
            model_state = {
                k: v for k, v in model_state.items()
                if not (k.startswith("text_encoder.") or k.startswith("module.text_encoder."))
            }
            print("  (text_encoder keys excluded from load; keeping current frozen encoder)")
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state, strict=False)
        else:
            model.load_state_dict(model_state, strict=False)
        print("✓ Model state loaded from checkpoint")
    else:
        print("⚠️ No model_state_dict in checkpoint; keeping current model weights")

    # Load optimizer state if present
    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("✓ Optimizer state loaded from checkpoint")
        except Exception as e:
            print(f"⚠️ Failed to load optimizer state, continuing with fresh optimizer: {e}")

    # Extract training state
    checkpoint_epoch = checkpoint.get("epoch", 0)
    start_epoch = checkpoint_epoch + 1  # Resume from next epoch
    best_val_loss = checkpoint.get("val_loss", float('inf'))

    # Calculate global_step from checkpoint epoch
    global_step = checkpoint_epoch * steps_per_epoch

    # Manually advance scheduler to match resumed epoch
    print(f"Advancing scheduler to match epoch {checkpoint_epoch} (step {global_step})...")
    for _ in range(global_step):
        scheduler.step()
    print(f"✓ Scheduler advanced to step {global_step}")
    
    best_acc = checkpoint.get("best_acc", checkpoint.get("best_auc", 0.0))
    best_auc = checkpoint.get("best_auc", 0.0)
    best_test_acc = checkpoint.get("best_test_acc", 0.0)
    best_test_auc = checkpoint.get("best_test_auc", 0.0)

    print(f"✓ Resuming from epoch {start_epoch} (checkpoint was at epoch {checkpoint_epoch})")
    print(f"✓ Previous validation loss: {best_val_loss:.6f}")
    print(f"✓ Best accuracy: {best_acc:.4f} | Best AUC: {best_auc:.4f}")
    print(f"✓ Best test accuracy: {best_test_acc:.4f} | Best test AUC: {best_test_auc:.4f}")
    print(f"✓ Global step: {global_step}")
    print(f"{'='*80}\n")

    return start_epoch, global_step, best_val_loss, best_acc, best_auc, best_test_acc, best_test_auc

