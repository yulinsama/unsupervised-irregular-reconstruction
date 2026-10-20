"""
DIP网络（Deep Iterative Prior）及其Attention模块

实现论文中的纯全连接网络拓扑，包含两个Attention模块，
用于处理地震数据Patch的重建任务。

网络结构严格按照论文图1的拓扑实现：
- 输入：展平的Patch (size,)
- 隐藏层：FC_2000 → ReLU → FC_1000 → ReLU → Attention1
         → FC_500 → ReLU → Attention2 → FC_{2*size} → ReLU → Linear_{size}
- 包含跳跃连接和注意力加权
"""

import torch
import torch.nn as nn
from torch import Tensor


class Attention(nn.Module):
    """
    注意力模块 (图中虚线框部分)

    Global Average → FC+ReLU → FC+Sigmoid

    输入: (batch_size, channels) 的1D张量
    输出: (batch_size, channels) 的注意力权重，与输入形状相同
    """

    def __init__(self, channels: int):
        """
        初始化注意力模块

        Args:
            channels: 输入张量的通道数，也决定输出注意力权重的形状
        """
        super().__init__()
        self.channels = channels

        # 全连接层1：Global Average → FC+ReLU
        # 输入是全局平均值，形状为 (batch_size, 1)，所以 fc1 的输入维度是 1
        self.fc1 = nn.Linear(1, channels)
        self.relu = nn.ReLU()

        # 全连接层2：FC+Sigmoid (输出注意力权重)
        # 输出维度是 channels，与原始输入形状匹配
        self.fc2 = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, channels)

        Returns:
            attention_weights: 注意力权重张量，形状为 (batch_size, channels)
        """
        # Step 1: Global Average Pooling
        # torch.mean 会自动处理不同 batch_size，输出 (batch_size, 1)
        global_avg = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1)

        # Step 2: FC+ReLU
        # 将 (batch_size, 1) 的 global_avg 作为输入，全连接层会将其扩展到 channels 维度
        fc_out = self.fc1(global_avg)  # (batch_size, channels)
        relu_out = self.relu(fc_out)    # (batch_size, channels)

        # Step 3: FC+Sigmoid
        attention_weights = self.sigmoid(self.fc2(relu_out))  # (batch_size, channels)

        return attention_weights
    
class OptimizedAttention(nn.Module):
    """
    修正后的注意力模块 (1D Feature Attention)
    
    移除了会导致维度崩塌的 Global Average，
    采用标准的瓶颈结构 (降维 -> 升维) 提取通道注意力权重。
    """
    def __init__(self, channels: int, reduction_ratio: int = 4):
        super().__init__()
        self.channels = channels
        
        # 引入降维比例，减少参数量并强迫网络提取核心特征 (经典设计)
        bottleneck_dim = max(1, channels // reduction_ratio)
        
        # FC1: 降维 + ReLU
        self.fc1 = nn.Linear(channels, bottleneck_dim)
        self.relu = nn.ReLU()
        
        # FC2: 升维还原 + Sigmoid
        self.fc2 = nn.Linear(bottleneck_dim, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch_size, channels)
        """
        # (batch_size, channels) -> (batch_size, bottleneck_dim)
        out = self.relu(self.fc1(x))
        
        # (batch_size, bottleneck_dim) -> (batch_size, channels)
        attention_weights = self.sigmoid(self.fc2(out))
        
        return attention_weights


class DipNetwork(nn.Module):
    """
    DIP网络 (Deep Iterative Prior) - 严格按照论文图1实现

    网络拓扑严格遵循：
    x1 = ReLU(FC_2000(x))
    x2 = ReLU(FC_1000(x1))
    c1 = torch.cat([x1, x2], dim=-1)  (拼接)
    m1 = c1 * Attention1(c1) (注意力加权)
    x3 = ReLU(FC_500(m1))
    c2 = torch.cat([m1, x3], dim=-1)
    m2 = c2 * Attention2(c2)
    c3 = torch.cat([m1, m2], dim=-1)  (跳跃连接拼接)
    x4 = ReLU(FC_{2*size}(c3))
    out = Linear_{size}(x4)
    """

    def __init__(self, size: int):
        """
        初始化DIP网络

        Args:
            size: 输入Patch的展平维度 (px * py * pt)
        """
        super().__init__()
        self.size = size

        # 网络层定义
        self.fc1 = nn.Linear(size, 2000)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(2000, 1000)
        self.relu2 = nn.ReLU()

        # 第一个注意力模块作用于 c1 (x1+x2)
        self.attention1 = Attention(2000 + 1000)  # c1 的维度是 3000

        self.fc3 = nn.Linear(3000, 500)  # FC_500(m1)，m1 = c1 * att1 的维度
        self.relu3 = nn.ReLU()

        # 第二个注意力模块作用于 c2 (m1+x3)
        self.attention2 = Attention(3000 + 500)  # c2 的维度是 3500

        # 输出层 - 注意：FC4 的输入是 c3 (m1+m2)
        # m1: (batch_size, 3000), m2: (batch_size, 3500) -> c3: (batch_size, 6500)
        # FC4 输出维度：2 * size
        self.fc4 = nn.Linear(3000 + 3500, 2 * size)
        self.relu4 = nn.ReLU()

        # 最终输出层
        self.fc5 = nn.Linear(2 * size, size)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播，严格按照图1的拓扑结构

        Args:
            x: 输入张量，形状为 (batch_size, size)，size = px * py * pt

        Returns:
            out: 输出张量，形状为 (batch_size, size)
        """
        # Step 1: x1 = ReLU(FC_2000(x))
        x1 = self.relu1(self.fc1(x))  # (batch_size, 2000)

        # Step 2: x2 = ReLU(FC_1000(x1))
        x2 = self.relu2(self.fc2(x1))  # (batch_size, 1000)

        # Step 3: c1 = torch.cat([x1, x2], dim=-1)  (拼接)
        # x1: (batch_size, 2000), x2: (batch_size, 1000)
        # c1: (batch_size, 3000)
        c1 = torch.cat([x1, x2], dim=-1)

        # Step 4: m1 = c1 * Attention1(c1) (注意力加权)
        attention_weights1 = self.attention1(c1)  # (batch_size, 3000)
        m1 = c1 * attention_weights1  # (batch_size, 3000)

        # Step 5: x3 = ReLU(FC_500(m1))
        # m1: (batch_size, 3000) → FC_500 → (batch_size, 500)
        x3 = self.relu3(self.fc3(m1))  # (batch_size, 500)

        # Step 6: c2 = torch.cat([m1, x3], dim=-1)
        # m1: (batch_size, 3000), x3: (batch_size, 500)
        # c2: (batch_size, 3500)
        c2 = torch.cat([m1, x3], dim=-1)

        # Step 7: m2 = c2 * Attention2(c2) (注意力加权)
        attention_weights2 = self.attention2(c2)  # (batch_size, 3500)
        m2 = c2 * attention_weights2  # (batch_size, 3500)

        # Step 8: c3 = torch.cat([m1, m2], dim=-1) (跳跃连接拼接)
        # m1: (batch_size, 3000), m2: (batch_size, 3500)
        # c3: (batch_size, 6500)
        c3 = torch.cat([m1, m2], dim=-1)

        # Step 9: x4 = ReLU(FC_{2*size}(c3))
        # c3: (batch_size, 6500) → FC_{2*size} → (batch_size, 2*size)
        x4 = self.relu4(self.fc4(c3))  # (batch_size, 2*size)

        # Step 10: out = Linear_{size}(x4)
        # x4: (batch_size, 2*size) → Linear_{size} → (batch_size, size)
        out = self.fc5(x4)  # (batch_size, size)

        return out

    def get_layer_shapes(self, input_shape: tuple) -> dict:
        """
        获取网络各层的输出形状，用于调试

        Args:
            input_shape: 输入形状 (batch_size, size)

        Returns:
            shapes: 各层的输出形状字典
        """
        batch_size, size = input_shape
        shapes = {}

        shapes['input'] = (batch_size, size)

        x1 = torch.randn(batch_size, size)
        shapes['fc1'] = x1.shape
        x1 = self.relu1(self.fc1(x1))
        shapes['x1'] = x1.shape

        x2 = self.relu2(self.fc2(x1))
        shapes['x2'] = x2.shape

        c1 = torch.cat([x1, x2], dim=-1)
        shapes['c1'] = c1.shape

        m1 = c1 * self.attention1(c1)
        shapes['m1'] = m1.shape

        x3 = self.relu3(self.fc3(m1))
        shapes['x3'] = x3.shape

        c2 = torch.cat([m1, x3], dim=-1)
        shapes['c2'] = c2.shape

        m2 = c2 * self.attention2(c2)
        shapes['m2'] = m2.shape

        c3 = torch.cat([m1, m2], dim=-1)
        shapes['c3'] = c3.shape

        x4 = self.relu4(self.fc4(c3))
        shapes['x4'] = x4.shape

        out = self.fc5(x4)
        shapes['output'] = out.shape

        return shapes