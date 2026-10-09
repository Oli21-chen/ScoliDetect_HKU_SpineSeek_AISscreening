'''
tensorboard --logdir=C:/Users/Olive/Desktop/Nature_Communication/code_video/runs
'''


import torch
from torch.utils.data import DataLoader
from utils import (
    gait_collate_fnv1,
    prepare_datasets, calculate_pos_weight, validate_model_config,
    create_lr_scheduler, save_training_summary, save_checkpoint,
    create_confusion_matrix_plot
)
from models_table_modality import TableViT, BinaryClassificationLoss, evaluate_model, visualize_attention_to_table
import os
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import io
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


current_time = datetime.now().strftime("%Y%m%d%H%M%S")  # 格式：年月日时分秒

writer = SummaryWriter(log_dir=f'C:/Users/Olive/Desktop/Nature_Communication/code_video/runs/experiment_{current_time}')

# Create checkpoint folder for this experiment
checkpoint_dir = f'C:/Users/Olive/Desktop/Nature_Communication/code_video/checkpoints/experiment_{current_time}'
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"Checkpoint directory created: {checkpoint_dir}")

best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')  # Final model after all epochs


# ========== Data Preparation ==========
print("=== GAIT DATA LOADER SAMPLE USAGE ===")

dk_file = r'C:\Users\Olive\Desktop\Nature_Communication\code_video\data\organized_data_dk.h5'
sz_file = r'C:\Users\Olive\Desktop\Nature_Communication\code_video\data\organized_data_sz.h5'

train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2
noise_std = 0.5
random_seed = 42

train_dataset, val_dataset, test_dataset, combined_dataset, label_counts = prepare_datasets(
    dk_file, sz_file, train_ratio, val_ratio, test_ratio, random_seed, noise_std
)
print(f"合并后数据集总大小: {len(combined_dataset)} 样本")
print(f"Applied Gaussian noise (std={noise_std}) to training set for data augmentation")

num_negative = label_counts[0] if 0 in label_counts else 0
num_positive = label_counts[1] if 1 in label_counts else 0
pos_weight = calculate_pos_weight(label_counts, device)

initial_lr = 1e-4 #(4e-5]
num_epochs = 100
batch_size = 16

# 创建统一DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,  # 合并后可全局打乱
    collate_fn = gait_collate_fnv1  # 使用自定义的collate函数
)
val_loader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    shuffle = True,  # 合并后可全局打乱
    collate_fn = gait_collate_fnv1  # 使用自定义的collate函数
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,  # 可以使用与验证集相同的batch_size
    shuffle=False,  # 测试集通常不需要打乱顺序
    collate_fn=gait_collate_fnv1  # 使用相同的collate函数
)

# ========== Model Configuration ==========
# Model hyperparameters
# RECOMMENDATION: For best practice, use mlp_dim == att_dim (unified dimension)
# This is the standard approach used in BERT, GPT, ViT, etc.
# If you need different dimensions, projection layers will be added automatically.

MODEL_CONFIG = {
    'inputshape': (32, 238),
    'mlp_dim': 512,          # MLP dimension: used for embeddings, MLP layers, and output
    'att_dim': 256,          # Attention dimension: RECOMMENDED to match mlp_dim (standard practice)
                             # If different, projection layers will be used (adds parameters)
    'att_drop': 0.1,
    'head': 8,               # Number of attention heads (must divide att_dim evenly)
    'drop_out': 0.,
    'transform_layers': 8,
    'if_cls': True,
}

validate_model_config(MODEL_CONFIG)

model = TableViT(**MODEL_CONFIG).to(device)

# 2. 初始化损失函数
criterion = BinaryClassificationLoss(
    device=device, 
    l1_lambda=1e-6,
    pos_weight=pos_weight
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"| - Total parameters: {total_params} - |")


optimizer = optim.Adam(model.parameters(), lr=initial_lr)

num_train_batches_per_epoch = len(train_loader)
scheduler, warmup_steps, total_training_steps = create_lr_scheduler(
    optimizer, initial_lr, num_epochs, num_train_batches_per_epoch,
    warmup_ratio=0.1, min_lr_ratio=0.1
)
print(f"Total training steps: {total_training_steps} ({num_epochs} epochs × {num_train_batches_per_epoch} batches/epoch)")
print(f"Warmup steps: {warmup_steps} ({warmup_steps/total_training_steps*100:.1f}% of total steps)")
print(f"LR schedule: Warmup to {initial_lr:.2e}, then cosine decay to {initial_lr*0.1:.2e}")  

# 初始化时间统计变量

best_val_acc = 0.0
best_epoch = None  # Initialize best_epoch to avoid NameError
metrics = {
   'train_loss': [],
    'train_acc': [], 
    'val_loss': [],    
    'val_acc': [],  
}

total_start_time = time.time()
global_step = 0  # Track global step number for LR scheduling

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    # ========== 训练阶段 ==========
    model.train()
    train_loss = 0.0
    train_preds = []
    train_labels = []
    
    # 初始化Recall和F1的计算
    train_accuracy = 0.0
    num_train_batches = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        # 获取批次数据
        videos = batch["videos"].to(device)
        tables = batch["tables"].to(device)
        labels = batch["labels"].to(device)
        subject_id = batch["subject_id"].to(device)
        batch_size = videos.size(0)
        
        # 前向传播
        optimizer.zero_grad()
        outputs, intermediate_features = model(tables)  # 模型返回outputs和中间特征
        
        # 计算损失（使用您的BinaryClassificationLoss）
        loss = criterion(
            x=intermediate_features,  # 中间特征（当前未在损失中使用，但保留接口）
            outputs=outputs, 
            targets=labels, 
            model=model
        )
        
        # 反向传播和参数更新
        loss.backward()
        optimizer.step()
        
        # Update learning rate at each step
        scheduler.step()
        global_step += 1
        
        # 收集统计信息
        train_loss += loss.item()* batch_size 
       
        # 收集预测和真实标签用于指标计算
        preds = (torch.sigmoid(outputs) > 0.5).int().detach().cpu().numpy()
        labels_cpu = labels.int().detach().cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels_cpu)
        
        num_train_batches += 1
        total_samples += batch_size
    
    # 计算训练指标
    train_accuracy = accuracy_score(train_labels, train_preds)
    # 平均损失
    train_loss = train_loss / total_samples
  
    # ========== 验证阶段 ==========
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    num_val_batches = 0
    val_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            videos = batch["videos"].to(device)
            tables = batch["tables"].to(device)
            labels = batch["labels"].to(device)
            subject_id = batch["subject_id"].to(device)
            batch_size = videos.size(0)
            
            # 前向传播
            outputs, intermediate_features = model(tables)
            
            # 计算损失
            loss = criterion(
                x=intermediate_features,  # 中间特征（当前未在损失中使用，但保留接口）
                outputs=outputs, 
                targets=labels, 
                model=model
            )
            
            # 收集预测和真实标签
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            labels_cpu = labels.int().cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels_cpu)
            
            val_loss += loss.item() * batch_size
            num_val_batches += 1
            val_samples += batch_size
    
    # 计算验证指标
    val_loss = val_loss / val_samples
    val_accuracy = accuracy_score(val_labels, val_preds)
    
    # Get current learning rate (after all steps in this epoch)
    current_lr = optimizer.param_groups[0]['lr']
    
     # 记录指标
    metrics['train_loss'].append(train_loss)
    metrics['train_acc'].append(train_accuracy)
    metrics['val_loss'].append(val_loss)  # 新增验证损失
    metrics['val_acc'].append(val_accuracy)

     # ========== 模型保存 ==========
    # 如果性能更好，保存模型
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        print(f'\nSaving best model (Val acc: {val_accuracy:.4f})')
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, best_model_path)
    
    # Save periodic checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        periodic_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'train_acc': train_accuracy,
            'val_loss': val_loss,
            'val_acc': val_accuracy,
            'metrics': metrics
        }, periodic_checkpoint_path, epoch)
     
    # 计算当前epoch耗时
    epoch_total_time = time.time() - epoch_start_time
    

    print(
    f"Epoch {epoch+1}/{num_epochs} | "
    f"Train Loss: {train_loss:.4f} | "
    f"Train Acc: {train_accuracy:.4f} | "
    f"Val Loss: {val_loss:.4f} | "  
    f"Val Acc: {val_accuracy:.4f} |  "
    f"Time: {epoch_total_time:.2f}s | "
    f"LR: {current_lr:.7f} | "
    f"Step: {global_step}/{total_training_steps}"
    )
    # ========== 在每个epoch的训练和验证计算结束后，添加如下代码 ==========
    writer.add_scalar('Learning Rate', current_lr, epoch)
    writer.add_scalar('Learning Rate/Step', current_lr, global_step)  # Also log by step
    # 记录训练损失和准确率
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

    # 记录验证损失和准确率
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

    
# 最终处理
total_time = time.time() - total_start_time
print(f"Training completed in {total_time:.2f} seconds")
# Save model
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_loss,
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None
}

save_checkpoint(checkpoint, final_model_path)
if best_epoch is not None:
    print('Best val_acc saved model is at epoch:', best_epoch)
else:
    print('No model was saved during training (best_val_acc never improved)')
print("| - Training Done - |")

# 在测试集上评估
test_loss, test_accuracy, test_preds, test_labels = evaluate_model(
    model, test_loader, criterion, device
)
print(f"测试集性能: Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

# 在测试集上评估
best_model = model
best_model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
best_loss, best_accuracy, best_preds, best_labels = evaluate_model(
    best_model, test_loader, criterion, device
)

# Store which model performed better for summary
used_best_model = bool(best_accuracy > test_accuracy)  # Convert to Python bool for JSON serialization
if used_best_model:
    test_loss = best_loss
    test_accuracy = best_accuracy
    test_preds = best_preds
    test_labels = best_labels
    print('Choose Best_model at epoch:', best_epoch, best_accuracy)
else:
    print('Choose Train_model at epoch:', epoch, test_accuracy)


image_tensor = create_confusion_matrix_plot(test_labels, test_preds, test_accuracy, epoch)
writer.add_image('Validation/Confusion Matrix', image_tensor, epoch)
writer.close()

# 计算其他指标（可选）
test_recall = recall_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds)
print('--------------------------------')
print(f"Summary: Accuracy: {test_accuracy:.4f}, Recall: {test_recall:.4f}, F1-score: {test_f1:.4f}")

# ========== Attention Visualization ==========
print("\n=== Generating Attention Visualizations ===")
# Use the best model for visualization
model_for_viz = best_model if best_accuracy > test_accuracy else model
model_for_viz.eval()

# Get a sample from test set for visualization
with torch.no_grad():
    sample_batch = next(iter(test_loader))
    sample_tables = sample_batch["tables"].to(device)
    sample_labels = sample_batch["labels"].to(device)
    
    # Forward pass with attention capture
    outputs, intermediate_features, attention_weights = model_for_viz(
        sample_tables, 
        return_attention_info=True
    )
    
    # Visualize attention for first sample in batch
    attention_save_path = os.path.join(checkpoint_dir, 'attention_visualization.png')
    fig = visualize_attention_to_table(
        attention_weights, 
        model_for_viz, 
        sample_idx=0,
        save_path=attention_save_path
    )
    
    # Also save to TensorBoard
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    attention_image = Image.open(buf)
    attention_tensor = transforms.ToTensor()(attention_image)
    
    # Reopen writer if closed
    if writer is None:
        writer = SummaryWriter(log_dir=f'C:/Users/Olive/Desktop/Nature_Communication/code_video/runs/experiment_{current_time}')
    writer.add_image('Attention/Last Layer Attention', attention_tensor, 0)
    plt.close(fig)
    
    print(f"Attention visualization saved to: {attention_save_path}")
    print(f"True label: {sample_labels[0].item()}, Predicted: {torch.sigmoid(outputs[0]).item():.4f}")

writer.close()
print("| - All Visualizations Complete - |")

# ========== Save Training Summary ==========
summary_path, summary_text_path = save_training_summary(
    checkpoint_dir=checkpoint_dir,
    current_time=current_time,
    model_config=MODEL_CONFIG,
    training_config={
        'initial_lr': initial_lr,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'warmup_steps': warmup_steps,
        'total_training_steps': total_training_steps,
        'min_lr_factor': 0.1,
        'final_lr': optimizer.param_groups[0]['lr'],
    },
    data_config={
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'total_samples': len(combined_dataset),
        'random_seed': random_seed,
        'noise_std': noise_std,
    },
    loss_config={
        'l1_lambda': 1e-6,
        'pos_weight': pos_weight.item() if isinstance(pos_weight, torch.Tensor) else pos_weight,
        'class_distribution': {
            'num_negative': num_negative,
            'num_positive': num_positive,
        }
    },
    model_info={
        'total_parameters': total_params,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    },
    training_results={
        'best_epoch': best_epoch if best_epoch is not None else 'N/A',
        'best_val_accuracy': best_val_acc,
        'total_training_time_seconds': total_time,
        'total_training_time_hours': total_time / 3600,
    },
    final_metrics={
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
    },
    best_model_metrics={
        'best_test_loss': float(best_loss),
        'best_test_accuracy': float(best_accuracy),
        'final_model_test_loss': float(test_loss),
        'final_model_test_accuracy': float(test_accuracy),
        'used_best_model': used_best_model,
    },
    training_history={
        'final_train_loss': float(train_loss),
        'final_train_acc': float(train_accuracy),
        'final_val_loss': float(val_loss),
        'final_val_acc': float(val_accuracy),
        'metrics': {
            'train_loss': [float(x) for x in metrics['train_loss']],
            'train_acc': [float(x) for x in metrics['train_acc']],
            'val_loss': [float(x) for x in metrics['val_loss']],
            'val_acc': [float(x) for x in metrics['val_acc']],
        }
    },
    device=device
)
print(f"\nTraining summary saved to: {summary_path}")
print(f"Text summary saved to: {summary_text_path}")
print("| - Summary Saved - |")
