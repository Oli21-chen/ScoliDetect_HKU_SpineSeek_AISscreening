import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange,Reduce
from einops import rearrange, repeat

class DyT(nn.Module):
    def __init__(self, dim, alpha_init=0.5):
        super().__init__()
        # 可学习参数：缩放因子 α，逐通道权重 γ 和偏置 β
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)  # 初始值建议 0.5
        self.gamma = nn.Parameter(torch.ones(dim))  # 特征维度权重
        self.bias = nn.Parameter(torch.zeros(dim))  # 特征维度偏置

    def forward(self, x):
        x = torch.tanh(self.alpha * x)  # 动态缩放 + Tanh 非线性压缩
        return x * self.gamma + self.bias  # 仿射变换

class RotaryPositionEmbedding(nn.Module):
    """RoPE实现旋转位置嵌入"""
    def __init__(self, dim_model):
        super().__init__()
        assert dim_model % 2 == 0, "dim_model must be even for RoPE"
        self.dim_model = dim_model
        self.freqs_cache = None
    
    def _create_freqs(self, positions, device):
        dim_half = self.dim_model // 2  # 只需要一半的维度
        # 生成旋转频率（每个位置点对应dim_half个频率）
        i = torch.arange(0, dim_half, 1, device=device).float()
        theta = 1.0 / (10000 ** (2 * i / self.dim_model))
        
        # 创建位置-频率矩阵
        freqs = torch.outer(positions.squeeze(-1), theta)
        return freqs  # (seq_len, dim_half)
    
    def apply_rope(self, x, positions):
        """
        x: 输入张量 (batch_size, seq_len, dim_model)
        positions: 位置索引 (batch_size, seq_len, 1)
        """
        device = x.device
        batch_size, seq_len, dim_model = x.shape
        dim_half = dim_model // 2
        
        # 确保维度是偶数
        if dim_model % 2 != 0:
            raise ValueError(f"Feature dimension must be even, got {dim_model}")
        
        # 创建频率缓存
        if self.freqs_cache is None or self.freqs_cache.shape[0] < seq_len:
            pos_range = torch.arange(seq_len, device=device).view(-1, 1)
            self.freqs_cache = self._create_freqs(pos_range, device)  # (seq_len, dim_half)
        
        # 获取旋转频率 (1, seq_len, dim_half)
        freqs = self.freqs_cache[:seq_len].unsqueeze(0).to(x.dtype)
        
        # 分割x为两个部分
        x1 = x[..., :dim_half]
        x2 = x[..., dim_half:]
        
        # 应用旋转操作 (广播机制)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # 旋转变换
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        
        # 合并结果
        return torch.cat([y1, y2], dim=-1)

class StochasticDepth(nn.Module):
    """Stochastic Depth layer (https://arxiv.org/abs/1603.09382)"""
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.training:
            keep_prob = 1 - self.drop_prob
            shape = (x.size(0),) + (1,) * (x.dim() - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return (x / keep_prob) * random_tensor
        return x

class LayerScale(nn.Module):
    """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239"""
    def __init__(self, init_values: float, projection_dim: int):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(projection_dim))
        
    def forward(self, x):
        return x * self.gamma

class PatchToken(nn.Module):
    """Patch tokenization layer"""
    def __init__(self, mlp_dim: int, table_size: tuple, if_cls: bool = False):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.table_size = table_size
        self.patch_size = (6, 6)
        self.if_cls = if_cls
        
        if if_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, mlp_dim))
        
        self.project = nn.Conv2d(
            in_channels=1,
            out_channels=mlp_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0
        )
        
        self.num_patches = (table_size[0] // self.patch_size[0]) * (table_size[1] // self.patch_size[1])
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, mlp_dim))
        
    def forward(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(1)  # Add channel dimension
        
        patches = self.project(images)
        b, c, h, w = patches.size()
        patches = patches.view(b, c, h * w).transpose(1, 2)
        
        patches = patches + self.position_embedding
        
        if self.if_cls:
            cls_tokens = self.cls_token.expand(b, -1, -1)
            patches = torch.cat([cls_tokens, patches], dim=1)
            
        return patches

class TableToken(nn.Module):
    """Table tokenization layer"""
    def __init__(self, mlp_dim: int, table_size: tuple, if_cls: bool = False):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.table_size = table_size
        self.if_cls = if_cls
        
        if self.if_cls:
            self.cls_spt = nn.Parameter(torch.zeros(1, 1, mlp_dim))
            self.cls_temp = nn.Parameter(torch.zeros(1, 1, mlp_dim))
        
        # Positional embeddings
        self.pos_encoding_temp = RotaryPositionEmbedding(mlp_dim)
        self.pos_encoding_spt = RotaryPositionEmbedding(mlp_dim)

        # 列投影层 (高度方向分割)
        self.project_col =  nn.Conv2d(1, mlp_dim, kernel_size=(table_size[0], 1), 
                     stride=(table_size[0], 1), padding=0) # [B,512,16,W]
       
        # 行投影层 (宽度方向分割)
        self.project_row = nn.Conv2d(1, mlp_dim, kernel_size=(1, table_size[1]), 
                     stride=(1, table_size[1]), padding=0)  # [B,512,H,16]

        self.norm1 = DyT(mlp_dim)# nn.LayerNorm(mlp_dim)
        self.norm2 = DyT(mlp_dim)# nn.LayerNorm(mlp_dim)
        
    def forward(self, images):#[4, 512, 238]
        if images.dim() == 3:
            images = images.unsqueeze(1)  # Add channel dimension#[4,1, 512, 238]
        
        B = images.shape[0]    
        # Column projection (feature patches)
        fea_n_patches = self.project_col(images) #torch.Size([4, 512, 16, 238])
        b, c, h, w = fea_n_patches.size()
        fea_n_patches = fea_n_patches.view(b, c, h * w).transpose(1, 2)
        
         # Row projection (temporal patches)
        tem_l_patches = self.project_row(images)
        b, c, h, w = tem_l_patches.size()
        tem_l_patches = tem_l_patches.view(b, c, h * w).transpose(1, 2)

        if self.if_cls:
            # Create expanded versions without reassigning the parameters
            cls_spt_expanded = repeat(self.cls_spt, '1 1 d -> b 1 d', b=B)
            cls_temp_expanded = repeat(self.cls_temp, '1 1 d -> b 1 d', b=B)
           

            fea_n_patches = torch.cat((cls_spt_expanded, fea_n_patches), dim=1)
            tem_l_patches = torch.cat((cls_temp_expanded, tem_l_patches), dim=1)

            # fea_n_patches = fea_n_patches + self.pos_encoding_spt
            device = fea_n_patches.device  # Use the device from the patches tensor
            fea_positions = torch.cat([
                torch.zeros(fea_n_patches.shape[0], 1, device=device),
                torch.arange(1, fea_n_patches.shape[1]+1, device=device).repeat(fea_n_patches.shape[0], 1)
            ], dim=1).long().unsqueeze(-1)  # (batch_size, num_points+1, 1)

            tem_positions = torch.cat([
                torch.zeros(tem_l_patches.shape[0], 1, device=device),
                torch.arange(1, tem_l_patches.shape[1]+1, device=device).repeat(tem_l_patches.shape[0], 1)
            ], dim=1).long().unsqueeze(-1)  # (batch_size, num_points+1, 1)

        else:
            device = fea_n_patches.device  # Use the device from the patches tensor
            # When if_cls=False, positions are [0, 1, 2, ..., N-1] for N patches
            fea_positions = torch.cat([
                torch.zeros(fea_n_patches.shape[0], 1, device=device),
                torch.arange(1, fea_n_patches.shape[1], device=device).repeat(fea_n_patches.shape[0], 1)
            ], dim=1).long().unsqueeze(-1)  # (batch_size, num_patches, 1)
            # tem_l_patches = tem_l_patches + self.pos_encoding_temp
            tem_positions = torch.cat([
                torch.zeros(tem_l_patches.shape[0], 1, device=device),
                torch.arange(1, tem_l_patches.shape[1], device=device).repeat(tem_l_patches.shape[0], 1)
            ], dim=1).long().unsqueeze(-1)  # (batch_size, num_patches, 1)

        fea_n_patches = self.pos_encoding_spt.apply_rope(fea_n_patches, fea_positions)
        tem_l_patches = self.pos_encoding_temp.apply_rope(tem_l_patches, tem_positions)
            
        fea_n_patches = self.norm1(fea_n_patches)
        tem_l_patches = self.norm2(tem_l_patches)

        # patches_emb = torch.cat([ 
        #         tem_l_patches[:,1:,:],
        #         fea_n_patches[:,1:,:]
        #     ], dim=1)

        return tem_l_patches, fea_n_patches

class ImportanceLayer(nn.Module):
    def __init__(self, mlp_dim, eps=1e-10, dropout=0.1, temperature=1e-6):
        super().__init__()
        self.eps = eps
        self.temperature = temperature
        
    def forward(self, logits):
        """
        Args:
            logits: [B, N, C] - input logits (features)
        """
        # Compute entropy from logits with learnable temperature
        probs = F.softmax(logits, dim=-1)  # [B, N, C]
        log_probs = torch.log2(probs + self.eps)  # Base-2 for bits
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=False)  # [B, N]
        return entropy

class TableViT(nn.Module):
    """Vision Transformer for tabular data"""
    def __init__(
        self,
        inputshape: tuple,
        mlp_dim: int = 512,
        att_dim: int = 512,
        att_drop: float = 0.1,
        head: int = 6,
        drop_out: float = 0.1,
        transform_layers: int = 6,
        if_cls: bool = True
    ):
        super().__init__()
        self.if_cls = if_cls
        self.inputshape = inputshape  # Store for attention visualization
        
        self.table_token = TableToken(
            mlp_dim=mlp_dim,
            table_size=inputshape,
            if_cls=if_cls
        )
        
        # Store patch dimensions for attention visualization
        # Temporal patches: table_size[0] (height dimension)
        # Feature patches: table_size[1] (width dimension)
        # Note: CLS tokens (if enabled) are removed before concatenation,
        # so the effective patch counts are still inputshape[0] and inputshape[1].
        self.num_temp_patches = inputshape[0]
        self.num_feat_patches = inputshape[1]
        self.total_patches = self.num_temp_patches + self.num_feat_patches
        
        # Store dimensions
        self.mlp_dim = mlp_dim
        self.att_dim = att_dim
        
        # Projection layers to handle different mlp_dim and att_dim
        # If dimensions differ, we need to project between them
        if mlp_dim != att_dim:
            # Project from mlp_dim to att_dim before attention
            self.att_input_proj = nn.Linear(mlp_dim, att_dim)
            # Project from att_dim back to mlp_dim after attention
            self.att_output_proj = nn.Linear(att_dim, mlp_dim)
        else:
            # Identity if dimensions match
            self.att_input_proj = nn.Identity()
            self.att_output_proj = nn.Identity()
        
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": DyT(mlp_dim),# nn.LayerNorm(mlp_dim),
                "attn": nn.MultiheadAttention(
                    embed_dim = att_dim,
                    num_heads = head,
                    dropout = att_drop,
                    batch_first = True
                ),
                # "layer_scale1": LayerScale(init_values, mlp_dim),
                # "stochastic_depth": StochasticDepth(depth),
                "norm2": DyT(mlp_dim),# nn.LayerNorm(mlp_dim),
                "mlp": nn.Sequential(
                    nn.Linear(mlp_dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, mlp_dim),
                    nn.Dropout(drop_out)
                ),
                # "layer_scale2": LayerScale(init_values, mlp_dim)
            })
            for _ in range(transform_layers)
        ])
        self.norm1 = nn.LayerNorm(mlp_dim)

        self.norm_final = nn.LayerNorm(mlp_dim)# nn.LayerNorm(mlp_dim)

        self.output_layer = nn.Sequential(
            Reduce('b n c -> b c', reduction='mean'),
            nn.Linear(mlp_dim, 1),
        )
        # self.mix_weight = nn.Parameter(torch.tensor(0.5))
    def forward(self, x, return_attention_info=False):
        tem_l_patches, fea_n_patches = self.table_token(x)
        # x = self.linear_proj(x)
        x = torch.cat([ 
                tem_l_patches[:,1:,:],
                fea_n_patches[:,1:,:]
            ], dim=1)

        attention_weights = None
        
        for i, layer in enumerate(self.transformer_layers):
            # Attention branch
            residual = x  # [B, N, mlp_dim]
            x = layer["norm1"](x)  # [B, N, mlp_dim]
            
            # Project to attention dimension if needed
            x_att = self.att_input_proj(x)  # [B, N, att_dim]
            
            # Capture attention weights from the last layer only
            if return_attention_info and i == len(self.transformer_layers) - 1:
                attn_output, attn_weights = layer["attn"](x_att, x_att, x_att, need_weights=True, average_attn_weights=False)
                # attn_weights shape: [B, num_heads, seq_len, seq_len]
                # Average across heads: [B, seq_len, seq_len]
                attention_weights = attn_weights.mean(dim=1)
            else:
                attn_output, _ = layer["attn"](x_att, x_att, x_att)  # [B, N, att_dim]
            
            # Project back to mlp_dim if needed
            attn_output = self.att_output_proj(attn_output)  # [B, N, mlp_dim]
            
            x = residual + attn_output  # [B, N, mlp_dim]
            
            # MLP branch
            residual = x
            x = layer["norm2"](x)
            mlp_output = layer["mlp"](x)
            x = residual + mlp_output
        
        x = self.norm1(x)
        output = self.output_layer(x)
        
        if return_attention_info:
            return output, x, attention_weights
        return output, x
    

class BinaryClassificationLoss(nn.Module):
    def __init__(self, device='cuda', l1_lambda=1e-6, loss_weight=1.0, pos_weight: torch.Tensor = None):
        """
        二元分类损失函数 + L1正则化
        
        参数:
        - device: 计算设备 (cuda/cpu)
        - l1_lambda: L1正则化强度
        - loss_weight: 分类损失的权重（用于调整主任务损失）
        """
        super().__init__()
        self.device = device
        self.l1_lambda = l1_lambda
        # 使用BCEWithLogitsLoss (自动包含sigmoid和BCE)
        if pos_weight is not None:
            # pos_weight expects a 1D tensor of size [num_classes]; here binary -> [1]
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, x, outputs, targets, model):
        """
        计算二元分类损失 + L1正则化
        
        参数:
        - x: 中间特征 (当前未使用，保留以兼容接口)
        - outputs: 模型输出的logits (未激活)，形状为 (B, 1)
        - targets: 真实标签 (0或1)，形状为 (B,)
        - model: 模型本身 (用于获取权重进行正则化)
        
        返回:
        - total_loss: 总损失
        """
        # 确保logits和targets在同一设备
        outputs = outputs.to(self.device).view(-1)
        targets = targets.to(self.device).view(-1).float()
        
        # 计算二元分类损失 (自动包含sigmoid+BCE)
        bce_loss = self.bce_loss(outputs, targets)
        
        # 计算L1正则化 (所有非bias权重)
        l1_loss = 0.0
        for name, param in model.named_parameters():
            if 'bias' not in name and param.requires_grad:
                l1_loss += torch.sum(torch.abs(param))
        l1_loss *= self.l1_lambda

        # 总损失 = BCE损失 + L1正则化
        total_loss = bce_loss + l1_loss
        return total_loss 

from sklearn.metrics import accuracy_score, recall_score, f1_score

def evaluate_model(model, data_loader, criterion, device):
    """
    在给定数据加载器上评估模型性能

    参数:
    - model: 训练好的模型
    - data_loader: 测试集或验证集的DataLoader
    - criterion: 损失函数
    - device: 计算设备

    返回:
    - avg_loss: 平均损失
    - accuracy: 准确率
    - all_preds: 所有预测值
    - all_labels: 所有真实标签
    """
    model.eval()  # 设置模型为评估模式[3,7](@ref)
    total_loss = 0.0
    all_preds = []
    all_labels = []
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源[3,7](@ref)
        for batch_idx, batch in enumerate(data_loader):
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

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # 收集预测和真实标签
            preds = (torch.sigmoid(outputs) > 0.5).int().detach().cpu().numpy()
            labels_cpu = labels.int().detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_cpu)

    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_preds, all_labels


def visualize_attention_to_table(attention_weights, model, sample_idx=0, save_path=None):
    """
    More sophisticated visualization: map attention weights back to original table coordinates
    
    Args:
        attention_weights: [B, seq_len, seq_len] attention weights
        model: TableViT model
        sample_idx: Batch index to visualize
        save_path: Path to save figure
    
    Returns:
        fig: matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract attention for one sample
    if attention_weights.dim() == 3:
        attn = attention_weights[sample_idx].detach().cpu().numpy()
    else:
        attn = attention_weights.detach().cpu().numpy()
    
    num_temp = model.num_temp_patches
    num_feat = model.num_feat_patches
    table_height, table_width = model.inputshape
    seq_len = attn.shape[0]
    if seq_len != (num_temp + num_feat):
        raise ValueError(
            f"Attention seq_len ({seq_len}) != num_temp+num_feat ({num_temp + num_feat}). "
            "Check patch ordering or token counts."
        )
    if table_height != num_temp or table_width != num_feat:
        raise ValueError(
            f"Inputshape {model.inputshape} does not match patch counts "
            f"(num_temp={num_temp}, num_feat={num_feat})."
        )
    
    # Create attention map for table
    # Method: For each table position, compute how much attention it receives
    attention_map = np.zeros((table_height, table_width))
    
    # Each temporal patch i corresponds to table row i
    # Each feature patch j corresponds to table column j
    # Patch order in the sequence: [temporal patches..., feature patches...]
    for i in range(table_height):
        for j in range(table_width):
            # Find corresponding patch indices
            temp_patch_idx = min(i, num_temp - 1)
            feat_patch_idx = min(j, num_feat - 1)
            
            # Attention received by this table position:
            # - From temporal patch perspective: attention to temporal patch i
            # - From feature patch perspective: attention to feature patch j
            temp_attn = attn[:, temp_patch_idx].sum() if temp_patch_idx < attn.shape[0] else 0
            feat_attn = attn[:, num_temp + feat_patch_idx].sum() if (num_temp + feat_patch_idx) < attn.shape[0] else 0
            
            attention_map[i, j] = (temp_attn + feat_attn) / 2.0
    
    # Normalize for better visualization
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Full attention matrix
    im1 = axes[0].imshow(attn, cmap='viridis', aspect='auto')
    axes[0].set_title('Full Attention Matrix', fontsize=11)
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    axes[0].axvline(x=num_temp - 0.5, color='red', linestyle='--', alpha=0.7)
    axes[0].axhline(y=num_temp - 0.5, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Temporal patches attention (first num_temp rows)
    temp_attn = attn[:num_temp, :]
    im2 = axes[1].imshow(temp_attn, cmap='plasma', aspect='auto')
    axes[1].set_title(f'Temporal Patches Attention\n({num_temp} patches)', fontsize=11)
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Temporal Patch Index')
    axes[1].axvline(x=num_temp - 0.5, color='white', linestyle='--', alpha=0.7)
    plt.colorbar(im2, ax=axes[1])
    
    # Plot 3: Mapped to table structure
    im3 = axes[2].imshow(attention_map, cmap='hot', aspect='auto', interpolation='bilinear')
    axes[2].set_title(f'Attention Mapped to Table\n({table_height}×{table_width})', fontsize=11)
    axes[2].set_xlabel('Feature Dimension')
    axes[2].set_ylabel('Temporal Dimension')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    
    return fig

# 使用示例
# if __name__ == "__main__":
#     # 设置随机种子
#     SEED = 28
#     torch.manual_seed(SEED)
#     np.random.seed(SEED)
    
#     # 创建模型
#     model = TableViT(
#         inputshape=(238, 30),  # 假设输入形状为(238, 30)
#         mlp_dim=512,
#         att_dim=512,
#         att_drop=0.1,
#         head=6,
#         drop_out=0.1,
#         transform_layers=6,
#         if_cls=False,
#         depth=0.1,
#         init_values=0.1
#     )
    
#     # 测试输入
#     test_input = torch.randn(8, 238, 30)  # (batch, seq_len, features)
#     output = model(test_input)
#     print("Output shape:", output.shape)  # 期望输出: torch.Size([8, 1])

#simply summarize