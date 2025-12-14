#!/bin/bash
# solve.sh - 完成ABMIL模型实现的自动化脚本

set -e  # 遇到错误时退出
set -u  # 遇到未定义变量时退出

echo "🚀 开始完成ABMIL模型实现..."

# 1. 备份原始文件
BACKUP_DIR="/app/backups"
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp /app/abmil_assignment.py "${BACKUP_DIR}/abmil_assignment_backup_${TIMESTAMP}.py"
echo "✅ 原始文件已备份到 ${BACKUP_DIR}/abmil_assignment_backup_${TIMESTAMP}.py"

# 2. 生成完整的ABMIL实现
cat > /app/abmil_assignment.py << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import math
import warnings

class Attention_TanhSigmoidGating(nn.Module):
    """
    基于Tanh-Sigmoid门控机制的注意力模块，用于ABMIL模型
    
    参考: Ilse, M., Tomczak, J., & Welling, M. (2018). 
    Attention-based deep multiple instance learning.
    """
    
    def __init__(self, D: int, L: int, dropout: float = 0.25):
        """
        初始化注意力模块
        
        参数:
            D: 输入特征维度
            L: 注意力网络隐藏层维度
            dropout: Dropout比率
        """
        super(Attention_TanhSigmoidGating, self).__init__()
        
        # 验证输入参数
        if D <= 0 or L <= 0:
            raise ValueError("维度参数D和L必须为正整数")
        if not (0 <= dropout < 1):
            raise ValueError("Dropout比率必须在[0, 1)范围内")
        
        # 创建网络层 - 使用指定的确切属性名
        self.tanhV = nn.Sequential(
            nn.Linear(D, L),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        self.sigmU = nn.Sequential(
            nn.Linear(D, L),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        
        self.w = nn.Linear(L, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for layer in self.tanhV:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        for layer in self.sigmU:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        nn.init.xavier_normal_(self.w.weight)
        nn.init.constant_(self.w.bias, 0)
    
    def _get_eps(self, dtype: torch.dtype) -> float:
        """
        根据数据类型返回适当的epsilon值，用于数值稳定性
        
        参数:
            dtype: 张量数据类型
            
        返回:
            适当的epsilon值
        """
        if dtype == torch.float16:
            return 1e-2
        elif dtype == torch.float32:
            return 1e-6
        elif dtype == torch.float64:
            return 1e-10
        else:
            return 1e-8
    
    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算注意力权重和加权特征
        
        参数:
            H: 形状为(B, M, D)的张量，其中
               B = batch大小
               M = 袋中实例(patch)数量
               D = 特征维度
        
        返回:
            A: 形状为(B, M)的注意力权重
            M: 形状为(B, D)的加权聚合特征
        """
        # 输入验证
        if not torch.is_tensor(H):
            raise TypeError("输入必须是torch.Tensor类型")
        
        if H.dim() != 3:
            raise ValueError(f"输入张量维度应为3 (batch, instances, features), 但得到 {H.dim()}")
        
        if torch.isnan(H).any() or torch.isinf(H).any():
            warnings.warn("输入张量包含NaN或Inf值", RuntimeWarning)
        
        # 获取设备和数据类型
        device = H.device
        dtype = H.dtype
        
        # 获取批次大小和实例数量
        B, M, D = H.shape
        
        # 重塑为2D张量以进行高效处理
        H_2d = H.view(-1, D)  # (B*M, D)
        
        # 计算注意力分数
        tanh_part = self.tanhV(H_2d)  # (B*M, L)
        sigm_part = self.sigmU(H_2d)  # (B*M, L)
        gated = tanh_part * sigm_part  # (B*M, L)
        attn_scores = self.w(gated)    # (B*M, 1)
        
        # 重塑回原始批次结构
        attn_scores = attn_scores.view(B, M)  # (B, M)
        
        # 数值稳定性: 根据数据类型设置适当的epsilon
        eps = self._get_eps(dtype)
        
        # 计算注意力权重 (使用softmax)
        # 为避免数值不稳定，减去最大值
        max_scores = attn_scores.max(dim=1, keepdim=True)[0]
        exp_scores = torch.exp(attn_scores - max_scores)
        A = exp_scores / (torch.sum(exp_scores, dim=1, keepdim=True) + eps)
        
        # 计算加权聚合特征
        M = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # (B, D)
        
        # 确保输出没有NaN或Inf
        if torch.isnan(A).any() or torch.isinf(A).any():
            warnings.warn("注意力权重包含NaN或Inf值", RuntimeWarning)
        
        if torch.isnan(M).any() or torch.isinf(M).any():
            warnings.warn("聚合特征包含NaN或Inf值", RuntimeWarning)
        
        return A, M


class ABMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning (ABMIL) 模型
    
    参考: Ilse, M., Tomczak, J., & Welling, M. (2018). 
    Attention-based deep multiple instance learning.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.25):
        """
        初始化ABMIL模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            n_classes: 分类类别数
            dropout: Dropout比率
        """
        super(ABMIL, self).__init__()
        
        # 验证输入参数
        if input_dim <= 0 or hidden_dim <= 0 or n_classes <= 0:
            raise ValueError("所有维度参数必须为正整数")
        if not (0 <= dropout < 1):
            raise ValueError("Dropout比率必须在[0, 1)范围内")
        
        # 创建模型组件 - 使用指定的确切属性名
        self.inst_level_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.global_attn = Attention_TanhSigmoidGating(D=hidden_dim, L=hidden_dim, dropout=dropout)
        
        self.bag_level_classifier = nn.Linear(hidden_dim, n_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for layer in self.inst_level_fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        nn.init.xavier_normal_(self.bag_level_classifier.weight)
        nn.init.constant_(self.bag_level_classifier.bias, 0)
    
    def forward(self, x: torch.Tensor, lens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算分类结果
        
        参数:
            x: 形状为(B, M_max, D)的张量，包含填充后的bag特征
               B = batch大小
               M_max = 最大bag大小
               D = 特征维度
            lens: 形状为(B,)的张量，包含每个bag的实际长度
            
        返回:
            logits: 形状为(B, n_classes)的分类logits
            attn_weights: 形状为(B, M_max)的注意力权重
        """
        # 输入验证
        if not torch.is_tensor(x):
            raise TypeError("输入x必须是torch.Tensor类型")
        
        if x.dim() != 3:
            raise ValueError(f"输入张量x维度应为3 (batch, instances, features), 但得到 {x.dim()}")
        
        B, M_max, D = x.shape
        
        # 处理变长bags
        if lens is not None:
            if lens.shape[0] != B:
                raise ValueError(f"lens张量的batch大小({lens.shape[0]})与输入({B})不匹配")
            
            # 验证lens值是否有效
            if torch.any(lens <= 0) or torch.any(lens > M_max):
                invalid_values = lens[(lens <= 0) | (lens > M_max)]
                raise ValueError(f"无效的bag长度值: {invalid_values.tolist()}")
            
            # 创建掩码
            device = x.device
            mask = torch.arange(M_max, device=device).expand(B, M_max) < lens.unsqueeze(1)
            
            # 应用掩码 - 将填充部分设为0
            x = x * mask.unsqueeze(-1).float()
        
        # 实例级特征转换
        # 重塑为2D以便通过全连接层
        x_reshaped = x.view(-1, D)  # (B*M_max, D)
        H = self.inst_level_fc(x_reshaped)  # (B*M_max, hidden_dim)
        H = H.view(B, M_max, -1)  # (B, M_max, hidden_dim)
        
        # 全局注意力聚合
        A, M = self.global_attn(H)  # A: (B, M_max), M: (B, hidden_dim)
        
        # 应用掩码到注意力权重 (如果提供了lens)
        if lens is not None:
            # 将填充部分的注意力权重设为0
            A = A * mask.float()
            
            # 重新归一化注意力权重
            A_sum = A.sum(dim=1, keepdim=True)
            eps = self.global_attn._get_eps(A.dtype)
            A = A / (A_sum + eps)
        
        # 袋级分类
        logits = self.bag_level_classifier(M)  # (B, n_classes)
        
        return logits, A
    
    def predict(self, x: torch.Tensor, lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        预测样本的类别
        
        参数:
            x: 输入特征
            lens: 可选的bag长度
            
        返回:
            预测的类别索引
        """
        logits, _ = self.forward(x, lens)
        return torch.argmax(logits, dim=1)
    
    def get_attention(self, x: torch.Tensor, lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取注意力权重
        
        参数:
            x: 输入特征
            lens: 可选的bag长度
            
        返回:
            注意力权重
        """
        _, A = self.forward(x, lens)
        return A


def basic_test():
    """
    基本测试函数，验证ABMIL模型实现是否正确
    """
    try:
        # 测试配置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 2
        max_instances = 10
        input_dim = 512
        hidden_dim = 256
        n_classes = 2
        
        # 创建模型
        model = ABMIL(input_dim, hidden_dim, n_classes).to(device)
        
        # 创建随机输入 (batch_size, max_instances, input_dim)
        x = torch.randn(batch_size, max_instances, input_dim).to(device)
        
        # 随机bag长度 (确保至少有一个实例)
        lens = torch.randint(1, max_instances+1, (batch_size,)).to(device)
        
        # 前向传播
        logits, attn_weights = model(x, lens)
        
        # 验证输出形状
        assert logits.shape == (batch_size, n_classes), \
            f"错误的logits形状: {logits.shape}, 应为 {(batch_size, n_classes)}"
        
        assert attn_weights.shape == (batch_size, max_instances), \
            f"错误的注意力权重形状: {attn_weights.shape}, 应为 {(batch_size, max_instances)}"
        
        # 验证注意力权重总和为1 (考虑数值精度)
        attn_sum = attn_weights.sum(dim=1)
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-6), \
            f"注意力权重总和不为1: {attn_sum}"
        
        # 验证梯度流
        target = torch.randint(0, n_classes, (batch_size,)).to(device)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        
        # 检查是否有梯度
        grad_exists = False
        for param in model.parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                grad_exists = True
                break
        
        assert grad_exists, "没有检测到有效的梯度流"
        
        # 清理
        del model, x, lens, logits, attn_weights, loss
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✓ 前向传播成功完成！")
        return True
    
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行基本测试
    basic_test()
EOF

echo "✅ ABMIL模型实现已成功写入 /app/abmil_assignment.py"

# 3. 验证文件是否存在且不为空
if [ ! -s /app/abmil_assignment.py ]; then
    echo "❌ 错误：生成的文件为空"
    exit 1
fi

# 4. 运行基本测试
echo "🔍 运行基本测试验证实现..."
python3 /app/abmil_assignment.py

# 5. 检查测试结果
if [ $? -eq 0 ]; then
    echo "🎉 任务完成！ABMIL模型已成功实现并通过基本测试。"
    echo "✅ 请继续在评估环境中运行完整测试套件。"
    exit 0
else
    echo "❌ 警告：基本测试未通过。请检查实现。"
    exit 1
fi