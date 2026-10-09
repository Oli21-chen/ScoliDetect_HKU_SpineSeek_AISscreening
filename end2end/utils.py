import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
from torch.utils.data import Subset

class GaitDatasetv1(Dataset):
    """PyTorch Dataset for gait analysis data from h5 files."""
    
    def __init__(self, 
                 h5_file_path: str, 
                 group_name: str = None,
                 data_types: list[str] = ['video', 'table', 'labels', 'gait_analysis', 'prompts'],
                 normalize_video: bool = True,
                 normalize_table: bool = False,
                 max_cobb_threshold: float = 15.0,
                 add_gaussian_noise: bool = False,
                 noise_std: float = 0.01):
        """
        Initialize the dataset.
        
        Args:
            h5_file_path: Path to the h5 file
            group_name: Group name ('dk' or 'sz'), auto-detected if None
            data_types: List of data types to load
            normalize_video: Whether to normalize video data to [0,1]
            normalize_table: Whether to normalize table data
            max_cobb_threshold: Threshold for binary classification of scoliosis
            add_gaussian_noise: Whether to add Gaussian noise to table data (for data augmentation)
            noise_std: Standard deviation of Gaussian noise to add (default: 0.01)
        """
        self.h5_file_path = h5_file_path
        self.data_types = data_types
        self.normalize_video = normalize_video
        self.normalize_table = normalize_table
        self.max_cobb_threshold = max_cobb_threshold
        self.add_gaussian_noise = add_gaussian_noise
        self.noise_std = noise_std
        
        # Load dataset info
        with h5py.File(h5_file_path, 'r') as hf:
            if group_name is None:
                # Auto-detect group name from available keys
                group_name = list(hf.keys())[0]
            
            self.group_name = group_name
            group = hf[group_name]
            
            # Load subject IDs
            self.subject_ids = [sid.decode('utf-8') for sid in group['subject_ids'][:]]
            self.num_subjects = len(self.subject_ids)
            
            # Preload video and table shapes for faster access
            self.video_shape = group['video_data']['videos'].shape[1:] if 'video_data' in group else None
            self.table_shape = group['table_data']['tables'].shape[1:] if 'table_data' in group else None

    def __len__(self):
        return self.num_subjects
    
    def __getitem__(self, idx: int):
        """Get a sample from the dataset."""
        sample = {}
        subject_id = self.subject_ids[idx]
        
        with h5py.File(self.h5_file_path, 'r') as hf:
            group = hf[self.group_name]
            
            # Load video data
            if 'video' in self.data_types and 'video_data' in group:
                video_data = group['video_data']['videos'][idx]
                video_tensor = torch.from_numpy(video_data).float()
                if self.normalize_video:
                    video_tensor = video_tensor / 255.0
                sample['video'] = video_tensor
            
            # Load table data
            if 'table' in self.data_types and 'table_data' in group:
                table_data = group['table_data']['tables'][idx]
                table_tensor = torch.from_numpy(table_data).float()
                if self.normalize_table:
                    table_tensor = (table_tensor - table_tensor.mean()) / (table_tensor.std() + 1e-8)
                
                # Add Gaussian noise for data augmentation (only during training)
                if self.add_gaussian_noise:
                    noise = torch.randn_like(table_tensor) * self.noise_std
                    table_tensor = table_tensor + noise
                
                sample['table'] = table_tensor
            
            # Initialize Cobb angles to None
            m_cobb_l = None
            m_cobb_r = None
            
            # Load labels
            if 'labels' in self.data_types and 'labels' in group:
                # Load demographic data
                if 'age' in group['labels']:
                    sample['age'] = torch.tensor(float(group['labels']['age'][idx]))

                if 'sex' in group['labels']:
                    sex_str = group['labels']['sex'][idx].decode('utf-8')
                    sample['sex'] = torch.tensor(1.0 if sex_str.lower() == 'male' else 0.0)
                    
                # Load Cobb angles
                if 'M_cobb_l' in group['labels']:
                    m_cobb_l = float(group['labels']['M_cobb_l'][idx])
                    sample['M_cobb_l'] = torch.tensor(m_cobb_l)
                if 'M_cobb_r' in group['labels']:
                    m_cobb_r = float(group['labels']['M_cobb_r'][idx])
                    sample['M_cobb_r'] = torch.tensor(m_cobb_r)
            
            # Calculate binary label based on Cobb angles
            if m_cobb_l is not None and m_cobb_r is not None:
                max_cobb = max(m_cobb_l, m_cobb_r)
                binary_label = 1 if max_cobb > self.max_cobb_threshold else 0
            else:
                # Handle missing values
                binary_label = 0
                
            # Store calculated label
            sample['labels'] = torch.tensor(binary_label, dtype=torch.long)
           
            # Load text prompts
            if 'prompts' in self.data_types and 'prompts' in group:
                gait_key = f'subject_{subject_id}'
                if gait_key in group['prompts']:
                    sample['prompt'] = group['prompts'][gait_key][()].decode('utf-8')
                else:
                    sample['prompt'] = "No prompt available"
            
            sample['subject_id'] = subject_id
        
        return sample


class GaussianNoiseDataset(Dataset):
    """
    Wrapper dataset that adds Gaussian noise to table data.
    Useful for data augmentation during training.
    """
    def __init__(self, base_dataset, noise_std=0.01):
        """
        Args:
            base_dataset: The base dataset to wrap
            noise_std: Standard deviation of Gaussian noise to add
        """
        self.base_dataset = base_dataset
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        # Add Gaussian noise to table data
        if 'table' in sample:
            noise = torch.randn_like(sample['table']) * self.noise_std
            sample['table'] = sample['table'] + noise
        
        return sample


def gait_collate_fnv1(batch):
    """
    优化的步态数据批处理函数，支持：
    1. 可变长度视频序列的智能填充
    2. 多模态数据（视频/表格/步态参数）的并行处理
    3. 自动生成序列长度掩码供RNN/Transformer使用
    
    Args:
        batch: 从Dataset获取的样本列表，每个样本是字典格式
        
    Returns:
        结构化的批次数据字典，包含填充后的张量和原始数据引用
    """
    videos = torch.stack([item['video'] for item in batch])
    tables =  torch.stack([item['table'] for item in batch])
    labels =  torch.stack([item['labels'] for item in batch])
    # prompts = [item['prompt'] for item in batch]
    subject_id = torch.tensor([int(item['subject_id']) for item in batch])
    
    return {
        "videos": videos,
        "tables": tables,
        "labels":labels,
        # "prompts":prompts,
        "subject_id": subject_id,
      
    }

def prepare_datasets(dk_file, sz_file, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, 
                     random_seed=42, noise_std=0.3):
    """
    Prepare train/val/test datasets from h5 files.
    
    Args:
        dk_file: Path to DK dataset h5 file
        sz_file: Path to SZ dataset h5 file
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for splitting
        noise_std: Gaussian noise standard deviation for training set
        
    Returns:
        train_dataset, val_dataset, test_dataset, combined_dataset, label_counts
    """
    from torch.utils.data import ConcatDataset, random_split
    from collections import Counter
    
    dk_dataset = GaitDatasetv1(h5_file_path=dk_file, group_name='dk')
    sz_dataset = GaitDatasetv1(h5_file_path=sz_file, group_name='sz')
    
    combined_dataset = ConcatDataset([dk_dataset, sz_dataset])
    
    total_size = len(combined_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    
    train_dataset, val_dataset, test_dataset = random_split(
        combined_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Add Gaussian noise to training set
    train_dataset = GaussianNoiseDataset(train_dataset, noise_std=noise_std)
    
    # Count labels for class weighting
    all_train_labels = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        label = sample['labels']
        if hasattr(label, 'item'):
            label = label.item()
        all_train_labels.append(label)
    label_counts = Counter(all_train_labels)
    
    return train_dataset, val_dataset, test_dataset, combined_dataset, label_counts


def calculate_pos_weight(label_counts, device='cuda'):
    """
    Calculate positive weight for BCEWithLogitsLoss.
    
    Args:
        label_counts: Counter object with label counts
        device: Target device
        
    Returns:
        pos_weight tensor
    """
    num_negative = label_counts[0] if 0 in label_counts else 0
    num_positive = label_counts[1] if 1 in label_counts else 0
    return torch.tensor([float(num_negative) / max(1.0, float(num_positive))], device=device)


def validate_model_config(model_config):
    """
    Validate model configuration and print information.
    
    Args:
        model_config: Dictionary with model hyperparameters
        
    Returns:
        None (prints validation info)
    """
    assert model_config['att_dim'] % model_config['head'] == 0, \
        f"att_dim ({model_config['att_dim']}) must be divisible by head ({model_config['head']})"
    
    print("=" * 60)
    print("Model Configuration:")
    print(f"  mlp_dim: {model_config['mlp_dim']} (embedding & MLP dimension)")
    print(f"  att_dim: {model_config['att_dim']} (attention dimension)")
    if model_config['mlp_dim'] != model_config['att_dim']:
        print(f"  ⚠️  Dimensions differ: Projection layers will be used")
        print(f"     mlp_dim -> att_dim (before attention)")
        print(f"     att_dim -> mlp_dim (after attention)")
        print(f"  💡 RECOMMENDATION: Consider setting att_dim = mlp_dim for standard practice")
        print(f"     This avoids projection overhead and follows BERT/GPT/ViT conventions")
    else:
        print(f"  ✓ Dimensions match: No projection needed (standard practice)")
    print(f"  head: {model_config['head']} (att_dim/head = {model_config['att_dim']//model_config['head']})")
    print("=" * 60)


def create_lr_scheduler(optimizer, initial_lr, num_epochs, num_train_batches_per_epoch, 
                       warmup_ratio=0.1, min_lr_ratio=0.1):
    """
    Create step-based learning rate scheduler with warmup and cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        initial_lr: Initial learning rate
        num_epochs: Total number of epochs
        num_train_batches_per_epoch: Number of batches per epoch
        warmup_ratio: Ratio of warmup steps (default: 0.1 = 10%)
        min_lr_ratio: Final LR ratio relative to initial LR (default: 0.1 = 10%)
        
    Returns:
        scheduler, warmup_steps, total_training_steps
    """
    import math
    from torch.optim.lr_scheduler import LambdaLR
    
    total_training_steps = num_epochs * num_train_batches_per_epoch
    warmup_steps = int(total_training_steps * warmup_ratio)
    min_lr_factor = min_lr_ratio
    
    def lr_scheduler_with_warmup_step(step, warmup_steps, total_steps, max_lr=1.0, min_factor=0.1):
        """Step-based learning rate scheduler with warmup and cosine decay."""
        if step < warmup_steps:
            return (step / warmup_steps) * max_lr
        else:
            decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
            return min_factor + (max_lr - min_factor) * cosine_decay
    
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_scheduler_with_warmup_step(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_training_steps,
            max_lr=1.0,
            min_factor=min_lr_factor
        )
    )
    
    return scheduler, warmup_steps, total_training_steps


def convert_to_json_serializable(obj):
    """
    Recursively convert numpy types and other non-JSON-serializable types to native Python types.
    
    Args:
        obj: Object to convert (dict, list, or primitive)
        
    Returns:
        JSON-serializable object
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (torch.Tensor,)):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    else:
        return obj


def save_training_summary(checkpoint_dir, current_time, model_config, training_config, 
                          data_config, loss_config, model_info, training_results, 
                          final_metrics, best_model_metrics, training_history, device):
    """
    Save training summary as JSON and text files.
    
    Args:
        checkpoint_dir: Directory to save summary
        current_time: Experiment timestamp
        model_config: Model configuration dict
        training_config: Training configuration dict
        data_config: Data configuration dict
        loss_config: Loss configuration dict
        model_info: Model info dict (parameters, etc.)
        training_results: Training results dict
        final_metrics: Final test metrics dict
        best_model_metrics: Best model metrics dict
        training_history: Training history dict
        device: Device info
        
    Returns:
        summary_path, summary_text_path
    """
    import json
    
    summary = {
        'experiment_info': {
            'timestamp': current_time,
            'checkpoint_dir': checkpoint_dir,
            'tensorboard_logdir': f'C:/Users/Olive/Desktop/Nature_Communication/code_video/runs/experiment_{current_time}',
        },
        'model_config': model_config,
        'training_config': training_config,
        'data_config': data_config,
        'loss_config': loss_config,
        'model_info': model_info,
        'training_results': training_results,
        'final_metrics': final_metrics,
        'best_model_metrics': best_model_metrics,
        'training_history': training_history,
        'device_info': {
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
        }
    }
    
    # Convert numpy types to native Python types for JSON serialization
    summary = convert_to_json_serializable(summary)
    
    # Save JSON summary
    summary_path = os.path.join(checkpoint_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save text summary
    summary_text_path = os.path.join(checkpoint_dir, 'training_summary.txt')
    with open(summary_text_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXPERIMENT INFO\n")
        f.write("-" * 80 + "\n")
        f.write(f"Timestamp: {current_time}\n")
        f.write(f"Checkpoint Directory: {checkpoint_dir}\n")
        f.write(f"TensorBoard Logdir: runs/experiment_{current_time}\n\n")
        
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        for key, value in model_config.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nTotal Parameters: {model_info['total_parameters']:,}\n")
        f.write(f"Trainable Parameters: {model_info['trainable_parameters']:,}\n\n")
        
        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Initial Learning Rate: {training_config['initial_lr']}\n")
        f.write(f"Final Learning Rate: {training_config['final_lr']:.2e}\n")
        f.write(f"Number of Epochs: {training_config['num_epochs']}\n")
        f.write(f"Batch Size: {training_config['batch_size']}\n")
        f.write(f"Warmup Steps: {training_config['warmup_steps']} ({training_config['warmup_steps']/training_config['total_training_steps']*100:.1f}%)\n")
        f.write(f"Total Training Steps: {training_config['total_training_steps']}\n\n")
        
        f.write("DATA CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Train/Val/Test Split: {data_config['train_ratio']:.1%} / {data_config['val_ratio']:.1%} / {data_config['test_ratio']:.1%}\n")
        f.write(f"Train Size: {data_config['train_size']}\n")
        f.write(f"Val Size: {data_config['val_size']}\n")
        f.write(f"Test Size: {data_config['test_size']}\n")
        f.write(f"Total Samples: {data_config['total_samples']}\n")
        f.write(f"Gaussian Noise Std: {data_config['noise_std']}\n")
        f.write(f"Class Distribution - Negative: {loss_config['class_distribution']['num_negative']}, Positive: {loss_config['class_distribution']['num_positive']}\n\n")
        
        f.write("TRAINING RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best Epoch: {training_results['best_epoch']}\n")
        f.write(f"Best Validation Accuracy: {training_results['best_val_accuracy']:.4f}\n")
        f.write(f"Total Training Time: {training_results['total_training_time_seconds']:.2f} seconds ({training_results['total_training_time_hours']:.2f} hours)\n\n")
        
        f.write("FINAL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Test Loss: {final_metrics['test_loss']:.4f}\n")
        f.write(f"Test Accuracy: {final_metrics['test_accuracy']:.4f}\n")
        f.write(f"Test Recall: {final_metrics['test_recall']:.4f}\n")
        f.write(f"Test F1-Score: {final_metrics['test_f1']:.4f}\n\n")
        
        f.write("BEST MODEL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best Test Loss: {best_model_metrics['best_test_loss']:.4f}\n")
        f.write(f"Best Test Accuracy: {best_model_metrics['best_test_accuracy']:.4f}\n")
        f.write(f"Used Best Model: {best_model_metrics['used_best_model']}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of Summary\n")
        f.write("=" * 80 + "\n")
    
    return summary_path, summary_text_path


def save_checkpoint(checkpoint_dict, checkpoint_path, epoch=None):
    """
    Save model checkpoint.
    
    Args:
        checkpoint_dict: Dictionary with checkpoint data
        checkpoint_path: Path to save checkpoint
        epoch: Epoch number (for logging)
    """
    torch.save(checkpoint_dict, checkpoint_path)
    if epoch is not None:
        print(f'Saved checkpoint at epoch {epoch+1}: {checkpoint_path}')


def create_confusion_matrix_plot(test_labels, test_preds, test_accuracy, epoch=None):
    """
    Create confusion matrix visualization.
    
    Args:
        test_labels: True labels
        test_preds: Predicted labels
        test_accuracy: Test accuracy
        epoch: Epoch number (for TensorBoard)
        
    Returns:
        image_tensor: Tensor for TensorBoard
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    import torchvision.transforms as transforms
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8), dpi=200)
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                annot_kws={"size": 12},
                cbar_kws={"shrink": 0.8},
                square=True,
                linewidths=0.5,
                linecolor='grey')
    
    plt.title(f'Confusion Matrix (Accuracy: {test_accuracy:.4f})', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = transforms.ToTensor()(image)
    plt.close()
    
    return image_tensor


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    从检查点文件加载模型
    
    Args:
        checkpoint_path: 检查点文件路径
        device: 计算设备
        
    Returns:
        model: 加载的模型
        checkpoint_info: 检查点信息字典
    """
    # Import here to avoid circular dependency
    from models_table_modality import TableViT
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型配置
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # 使用默认配置作为后备
        print("Warning: No model config found in checkpoint, using default config")
        config = {
            'inputshape': (32,238),
            'mlp_dim': 6*64,
            'att_dim': 6*64,
            'att_drop': 0.1,
            'head': 6,
            'drop_out': 0.1,
            'transform_layers': 4,
            'if_cls': False,
        }
    
    # 创建模型
    model = TableViT(**config).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 准备检查点信息
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'Unknown'),
        'val_f1': checkpoint.get('val_f1', 'Unknown'),
        'val_accuracy': checkpoint.get('val_accuracy', 'Unknown'),
        'best_val_f1': checkpoint.get('best_val_f1', 'Unknown'),
        'total_training_time': checkpoint.get('total_training_time', 'Unknown')
    }
    
    print(f"Model loaded successfully!")
    print(f"  - Epoch: {checkpoint_info['epoch']}")
    print(f"  - Validation F1: {checkpoint_info['val_f1']}")
    print(f"  - Validation Accuracy: {checkpoint_info['val_accuracy']}")
    
    return model, checkpoint_info
