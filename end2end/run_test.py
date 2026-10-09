"""
Test-only script for evaluating a saved checkpoint on the test split.
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from utils import prepare_datasets, calculate_pos_weight, gait_collate_fnv1
from models_table_modality import TableViT, BinaryClassificationLoss


def load_model_config(experiment_dir):
    """Load model config from training_summary.json if available."""
    summary_path = os.path.join(experiment_dir, "training_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        return summary.get("model_config", None), summary.get("data_config", None)
    return None, None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    checkpoint_path = r"C:\Users\Olive\Desktop\Nature_Communication\code_video\checkpoints\experiment_20260125232636\best_model.pth"
    dk_file = r"C:\Users\Olive\Desktop\Nature_Communication\code_video\data\organized_data_dk.h5"
    sz_file = r"C:\Users\Olive\Desktop\Nature_Communication\code_video\data\organized_data_sz.h5"

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    experiment_dir = os.path.dirname(checkpoint_path)
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load model config + data config if available
    model_config, data_config = load_model_config(experiment_dir)
    if model_config is None:
        # Fallback to default config (adjust if needed)
        model_config = {
            "inputshape": (32, 238),
            "mlp_dim": 512,
            "att_dim": 256,
            "att_drop": 0.1,
            "head": 8,
            "drop_out": 0.0,
            "transform_layers": 8,
            "if_cls": True,
        }
        print("Warning: training_summary.json not found; using default model config.")

    if data_config is None:
        data_config = {
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
            "random_seed": 42,
            "noise_std": 0.0,
        }
        print("Warning: training_summary.json not found; using default data config.")

    # Prepare datasets (same split as training)
    train_dataset, val_dataset, test_dataset, combined_dataset, label_counts = prepare_datasets(
        dk_file,
        sz_file,
        train_ratio=data_config["train_ratio"],
        val_ratio=data_config["val_ratio"],
        test_ratio=data_config["test_ratio"],
        random_seed=data_config["random_seed"],
        noise_std=data_config.get("noise_std", 0.0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=gait_collate_fnv1,
    )

    # Build model and load weights
    model = TableViT(**model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Criterion for evaluation
    pos_weight = calculate_pos_weight(label_counts, device=device)
    criterion = BinaryClassificationLoss(device=device, l1_lambda=0.0, pos_weight=pos_weight).to(device)

    # Evaluate with probabilities for AUC calculation
    total_loss = 0.0
    all_preds = []
    all_probs = []  # Store probabilities for AUC
    all_labels = []
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            videos = batch["videos"].to(device)
            tables = batch["tables"].to(device)
            labels = batch["labels"].to(device)
            subject_id = batch["subject_id"].to(device)
            batch_size = videos.size(0)

            # Forward pass
            outputs, intermediate_features = model(tables)

            # Calculate loss
            loss = criterion(
                x=intermediate_features,
                outputs=outputs,
                targets=labels,
                model=model
            )

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Get probabilities and predictions
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels_cpu = labels.int().detach().cpu().numpy()

            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels_cpu.flatten())

    test_loss = total_loss / total_samples
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_recall = recall_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds)
    test_auc = roc_auc_score(all_labels, all_probs)

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save confusion matrix plot
    experiment_dir = os.path.dirname(checkpoint_path)
    cm_save_path = os.path.join(experiment_dir, 'test_confusion_matrix.png')
    
    # Create and save confusion matrix visualization
    plt.figure(figsize=(10, 8), dpi=200)
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                annot_kws={"size": 14},
                cbar_kws={"shrink": 0.8},
                square=True,
                linewidths=0.5,
                linecolor='grey',
                xticklabels=['Normal', 'Scoliosis'],
                yticklabels=['Normal', 'Scoliosis'])
    
    plt.title(f'Test Set Confusion Matrix\n(Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f})', 
              fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(cm_save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print("=== Test Results ===")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-score: {test_f1:.4f}")
    print(f"AUC-ROC: {test_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Scoliosis")
    print(f"True Normal      {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"True Scoliosis   {cm[1,0]:4d}      {cm[1,1]:4d}")
    print(f"\nConfusion matrix saved to: {cm_save_path}")
    print("====================")


if __name__ == "__main__":
    main()
