## 任务概述
  您需要完成ABMIL模型类中的前向传播实现。该模型使用基于注意力的池化机制，通过计算图像块嵌入(patch embeddings)的加权平均值进行分类。

  ## 数学背景
  给定一个包含M个维度为D的图像块嵌入的包 H = {h₁, ..., hₘ} ∈ ℝᴹˣᴰ，ABMIL计算：

  z = Σᵢ₌₁ᴹ aᵢ * hᵢ

  其中注意力权重aᵢ通过以下方式计算：
  aᵢ = exp{wᵀ(tanh(Vhᵢᵀ) ⊙ sigm(Uhᵢᵀ))} / Σⱼ exp{wᵀ(tanh(Vhⱼᵀ) ⊙ sigm(Uhⱼᵀ))}
  
  参考文献：Ilse, M., Tomczak, J., & Welling, M. (2018). Attention-based deep multiple instance learning. 
  In International Conference on Machine Learning (pp. 2127-2136). PMLR.
  
  ## 实现要求
  您需要完成两个神经网络类的实现：

  Attention_TanhSigmoidGating 类：
  - 在__init__中使用以下确切属性名初始化神经网络层：
    - self.tanhV: 顺序层(Sequential)，包含Linear(D, L)、Tanh()和Dropout
    - self.sigmU: 顺序层(Sequential)，包含Linear(D, L)、Sigmoid()和Dropout  
    - self.w: 从L到1维的线性层(Linear)
  - 实现前向传播，使用softmax计算注意力权重

  ABMIL 类：
  - 在__init__中使用以下确切属性名初始化模型组件：
    - self.inst_level_fc: 顺序层(Sequential)，包含Linear(input_dim, hidden_dim)、ReLU()和Dropout
    - self.global_attn: Attention_TanhSigmoidGating模块，参数D=hidden_dim, L=hidden_dim
    - self.bag_level_classifier: 从hidden_dim到n_classes的线性层(Linear)
  - 实现带有适当注意力池化的前向传播

  实现说明：
  - 必须实现模型初始化(__init__)和前向传播(forward)方法
  - 需要正确构建PyTorch的nn.Sequential和nn.Linear层
  - 必须包含输入验证、错误处理和数值稳定性检查
  - 应能处理可变大小的样本包(bag)和边缘情况
  - 根据数据类型(dtype)确保注意力权重的适当数值精度（例如：float16为1e-2，float32为1e-6，float64为1e-10）
  - 仔细关注上面提供的数学公式

  ## 提供的文件
  - `/app/abmil_assignment.py`: 包含不完整的ABMIL实现，仅有最小化占位符
  - 需要完成注意力机制和ABMIL模型的前向传播实现
  - 测试框架期望健壮的实现，能够通过包含性能要求在内的综合测试

  ## 性能要求
  您的实现必须满足严格的性能要求：

  内存效率：
  - 能够处理大型样本包（10,000+图像块），内存使用不过度增加（<2GB）
  - 支持超大样本包（50,000图像块）而不出现内存错误
  - 适当的内存清理和垃圾回收

  速度与可扩展性：
  - 模型应随输入大小合理扩展（时间复杂度不差于O(n)，且常数较小）
  - 时间复杂度不应超过线性扩展的2倍
  - 大型样本包（10,000图像块）必须在合理超时内完成（<30秒）
  - 保持所有输入尺寸下的数值准确性

  数值精度：
  - 支持多种数据类型：float16、float32、float64
  - 为每种数据类型维护适当的精度容差
  - 处理混合精度场景（fp16输入，fp32模型）
  - 确保float32和float64数据类型的梯度为有限值

  ## 成功标准
  - 通过替换所有pass语句完成两个类的实现
  - 通过包含边缘情况和数值稳定性在内的综合测试套件
  - 实现适当的错误处理和输入验证
  - 确保梯度流动和训练能力
  - 满足所有性能要求（内存、速度、精度）
  - 实现必须通过直接模型执行测试，且basic_test()函数必须成功执行无错误，返回True，并打印"✓ 前向传播成功完成！"