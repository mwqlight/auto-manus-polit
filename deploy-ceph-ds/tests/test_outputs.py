"""
Use this file to define pytest tests that verify the outputs of the task.

This file will be copied to /tests/test_outputs.py and run by the /tests/test.sh file
from the working directory.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from typing import Tuple, Optional

# 导入ABMIL实现
from abmil_assignment import ABMIL, Attention_TanhSigmoidGating, basic_test


class TestABMILModel(unittest.TestCase):
    """ABMIL模型的单元测试套件"""
    
    def setUp(self):
        """在每个测试前设置"""
        # 设置随机种子确保可重现性
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 基本配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 512
        self.hidden_dim = 256
        self.n_classes = 2
        self.batch_size = 4
        self.max_instances = 20
        
        # 创建模型
        self.model = ABMIL(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_classes=self.n_classes,
            dropout=0.1
        ).to(self.device)
        
        # 将模型设置为评估模式，但保留dropout以便测试
        self.model.train()
    
    def test_basic_forward_pass(self):
        """测试1: 基本前向传播功能和输出形状验证"""
        print("\n🔍 测试1: 基本前向传播功能")
        
        # 创建随机输入
        x = torch.randn(self.batch_size, self.max_instances, self.input_dim).to(self.device)
        
        # 随机bag长度
        lens = torch.randint(1, self.max_instances + 1, (self.batch_size,)).to(self.device)
        
        # 前向传播
        logits, attn_weights = self.model(x, lens)
        
        # 验证输出形状
        self.assertEqual(logits.shape, (self.batch_size, self.n_classes),
                         f"Logits形状错误: 期望 {(self.batch_size, self.n_classes)}, 得到 {logits.shape}")
        
        self.assertEqual(attn_weights.shape, (self.batch_size, self.max_instances),
                         f"注意力权重形状错误: 期望 {(self.batch_size, self.max_instances)}, 得到 {attn_weights.shape}")
        
        # 验证注意力权重性质
        # 1. 所有权重应为非负
        self.assertTrue(torch.all(attn_weights >= 0), "发现负的注意力权重")
        
        # 2. 每个样本的注意力权重总和应接近1 (考虑填充)
        for i in range(self.batch_size):
            valid_len = lens[i].item()
            weight_sum = attn_weights[i, :valid_len].sum().item()
            self.assertAlmostEqual(weight_sum, 1.0, delta=1e-5,
                                 msg=f"样本{i}的注意力权重总和({weight_sum})不接近1.0")
        
        print("✅ 基本前向传播测试通过")
    
    def test_edge_cases_and_input_validation(self):
        """测试2: 边界情况和输入验证"""
        print("\n🔍 测试2: 边界情况和输入验证")
        
        # 测试用例1: 单实例bag
        print("  - 测试单实例bag")
        x_single = torch.randn(1, 1, self.input_dim).to(self.device)
        lens_single = torch.tensor([1]).to(self.device)
        logits, attn_weights = self.model(x_single, lens_single)
        
        self.assertEqual(logits.shape, (1, self.n_classes))
        self.assertEqual(attn_weights.shape, (1, 1))
        self.assertAlmostEqual(attn_weights[0, 0].item(), 1.0, delta=1e-6)
        
        # 测试用例2: 最大bag大小 (模拟大bag)
        print("  - 测试大bag处理 (1000个实例)")
        large_bag_size = 1000
        x_large = torch.randn(1, large_bag_size, self.input_dim).to(self.device)
        lens_large = torch.tensor([large_bag_size]).to(self.device)
        
        # 监控内存使用
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
        
        logits, attn_weights = self.model(x_large, lens_large)
        
        # 检查内存使用 (应小于500MB)
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated()
            mem_increase = peak_mem - start_mem
            self.assertLess(mem_increase, 500 * 1024 * 1024,  # 500MB
                           f"大bag内存使用过高: {mem_increase / (1024*1024):.2f} MB")
        
        self.assertEqual(logits.shape, (1, self.n_classes))
        self.assertEqual(attn_weights.shape, (1, large_bag_size))
        
        # 测试用例3: 无效输入验证
        print("  - 测试无效输入处理")
        
        # 无效维度
        with self.assertRaises(ValueError):
            x_invalid = torch.randn(1, self.input_dim).to(self.device)  # 缺少实例维度
            self.model(x_invalid)
        
        # 无效bag长度
        with self.assertRaises(ValueError):
            x_valid = torch.randn(1, 10, self.input_dim).to(self.device)
            lens_invalid = torch.tensor([15]).to(self.device)  # 长度超过实例数
            self.model(x_valid, lens_invalid)
        
        # 零长度bag
        with self.assertRaises(ValueError):
            lens_zero = torch.tensor([0]).to(self.device)
            self.model(x_valid, lens_zero)
        
        print("✅ 边界情况和输入验证测试通过")
    
    def test_gradient_flow_and_training(self):
        """测试3: 梯度流和训练能力"""
        print("\n🔍 测试3: 梯度流和训练能力")
        
        # 创建小批量数据
        x = torch.randn(self.batch_size, self.max_instances, self.input_dim).to(self.device)
        lens = torch.randint(1, self.max_instances + 1, (self.batch_size,)).to(self.device)
        target = torch.randint(0, self.n_classes, (self.batch_size,)).to(self.device)
        
        # 确保模型处于训练模式
        self.model.train()
        
        # 前向传播
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        optimizer.zero_grad()
        
        logits, attn_weights = self.model(x, lens)
        
        # 计算损失
        loss = F.cross_entropy(logits, target)
        self.assertFalse(torch.isnan(loss) or torch.isinf(loss), "损失值无效")
        
        # 反向传播
        loss.backward()
        
        # 验证梯度
        grad_norms = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                self.assertFalse(torch.isnan(param.grad).any(), f"参数{name}的梯度包含NaN")
                self.assertFalse(torch.isinf(param.grad).any(), f"参数{name}的梯度包含Inf")
                self.assertGreater(grad_norm, 1e-8, f"参数{name}的梯度范数({grad_norm})过小")
        
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        self.assertGreater(avg_grad_norm, 1e-6, f"平均梯度范数({avg_grad_norm})过小，梯度流可能存在问题")
        
        # 执行优化步骤
        optimizer.step()
        
        # 验证参数确实更新了
        new_logits, _ = self.model(x, lens)
        self.assertFalse(torch.allclose(logits, new_logits), "模型参数未更新")
        
        print(f"✅ 梯度流测试通过 (平均梯度范数: {avg_grad_norm:.6f})")
    
    def test_dtype_and_numerical_stability(self):
        """测试4: 数据类型支持和数值稳定性"""
        print("\n🔍 测试4: 数据类型支持和数值稳定性")
        
        # 测试用例1: 不同数据类型
        dtypes = [torch.float32]
        if torch.cuda.is_available():
            dtypes.extend([torch.float16, torch.float64])
        
        for dtype in dtypes:
            print(f"  - 测试数据类型: {dtype}")
            model_dtype = self.model.to(dtype)
            x = torch.randn(self.batch_size, self.max_instances, self.input_dim).to(self.device, dtype=dtype)
            lens = torch.randint(1, self.max_instances + 1, (self.batch_size,)).to(self.device)
            
            try:
                logits, attn_weights = model_dtype(x, lens)
                
                # 验证输出没有NaN或Inf
                self.assertFalse(torch.isnan(logits).any(), f"{dtype}下logits包含NaN")
                self.assertFalse(torch.isinf(logits).any(), f"{dtype}下logits包含Inf")
                self.assertFalse(torch.isnan(attn_weights).any(), f"{dtype}下注意力权重包含NaN")
                self.assertFalse(torch.isinf(attn_weights).any(), f"{dtype}下注意力权重包含Inf")
                
                # 验证注意力权重总和
                for i in range(self.batch_size):
                    valid_len = lens[i].item()
                    weight_sum = attn_weights[i, :valid_len].sum()
                    self.assertTrue(torch.isfinite(weight_sum), f"{dtype}下权重总和不为有限值")
                    self.assertAlmostEqual(weight_sum.item(), 1.0, delta=1e-3 if dtype == torch.float16 else 1e-5,
                                         msg=f"{dtype}下样本{i}的注意力权重总和不接近1.0")
                
                print(f"    ✓ {dtype} 测试通过")
                
            except Exception as e:
                print(f"    ❌ {dtype} 测试失败: {str(e)}")
                raise
        
        # 测试用例2: 数值稳定性 (极端值)
        print("  - 测试数值稳定性 (极端值)")
        model_fp32 = self.model.to(torch.float32)
        
        # 创建包含极端值的输入
        x_extreme = torch.randn(self.batch_size, self.max_instances, self.input_dim).to(self.device)
        # 添加一些极大值
        extreme_indices = torch.randint(0, self.batch_size * self.max_instances * self.input_dim, (10,))
        x_extreme.view(-1)[extreme_indices] = 1e6
        
        # 添加一些极小值
        extreme_indices = torch.randint(0, self.batch_size * self.max_instances * self.input_dim, (10,))
        x_extreme.view(-1)[extreme_indices] = -1e6
        
        lens = torch.randint(1, self.max_instances + 1, (self.batch_size,)).to(self.device)
        
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                logits, attn_weights = model_fp32(x_extreme, lens)
                
                # 检查是否有警告 (允许有警告但不能失败)
                if len(w) > 0:
                    print(f"    ⚠️ 检测到 {len(w)} 个警告，但处理成功")
                    for warning in w:
                        print(f"      - {warning.message}")
                
                # 验证没有NaN或Inf
                self.assertFalse(torch.isnan(logits).any(), "极端值导致logits包含NaN")
                self.assertFalse(torch.isinf(logits).any(), "极端值导致logits包含Inf")
                self.assertFalse(torch.isnan(attn_weights).any(), "极端值导致注意力权重包含NaN")
                self.assertFalse(torch.isinf(attn_weights).any(), "极端值导致注意力权重包含Inf")
            
            print("    ✓ 极端值测试通过")
            
        except Exception as e:
            print(f"    ❌ 数值稳定性测试失败: {str(e)}")
            raise
        
        print("✅ 数据类型和数值稳定性测试通过")
    
    @classmethod
    def tearDownClass(cls):
        """所有测试完成后清理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n🎉 所有ABMIL测试完成!")

def test_outputs():
    """Test that the outputs are correct."""
    # 首先运行提供的basic_test
    print("🔧 首先运行基本功能测试...")
    if not basic_test():
        print("❌ 基本功能测试失败，跳过详细单元测试")
        exit(1)
    
    print("\n🧪 开始详细单元测试...")
    unittest.main(verbosity=2)
