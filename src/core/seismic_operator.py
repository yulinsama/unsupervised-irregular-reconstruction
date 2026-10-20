"""
非规则网格双线性插值算子及其伴随算子

实现论文中的正演算子 E 和伴随算子 E^T，用于非规则网格地震数据的
正演建模和伴随计算。

正演算子 E: 从规则网格 (Nx, Ny, Nt) 双线性插值到非规则坐标 (N_traces, Nt)
伴随算子 E^T: 从非规则观测 (N_traces, Nt) 加权散布回规则网格 (Nx, Ny, Nt)
"""

import torch
import torch.nn.functional as F
from typing import Tuple


class BilinearOperator:
    """
    非规则网格双线性插值算子及其伴随算子

    双线性插值公式:
    g(a,b) ≈ g(0,0)h1 + g(0,1)h2 + g(1,0)h3 + g(1,1)h4

    其中:
    - (a, b) 是归一化的相对坐标 [0, 1] x [0, 1]
    - g(0,0), g(0,1), g(1,0), g(1,1) 是四个角点的值
    - h1, h2, h3, h4 是对应的权重
    """

    def __init__(self, grid_shape: Tuple[int, int, int], otg_coords: torch.Tensor):
        """
        初始化双线性插值算子

        Args:
            grid_shape: 规则网格的形状 (Nx, Ny, Nt)
            otg_coords: 非规则地震道的 (x, y) 坐标，形状为 (N_traces, 2)
                       坐标值范围: [0, Nx-1] x [0, Ny-1] (连续值域)
        """
        self.Nx, self.Ny, self.Nt = grid_shape
        self.otg_coords = otg_coords  # (N_traces, 2), 坐标范围 [0, Nx-1] x [0, Ny-1]
        self.N_traces = otg_coords.shape[0]

        # 预计算插值权重和索引（用于伴随算子）
        self._prepare_interpolation_weights()

    def _prepare_interpolation_weights(self) -> None:
        """
        预计算每个非规则坐标点对应的四个网格点的索引和双线性插值权重

        对于每个坐标 (x, y)，找到其四个最近的网格点:
        - (x0, y0): 左下角 (floor, floor)
        - (x0, y1): 左上角 (floor, ceil)
        - (x1, y0): 右下角 (ceil, floor)
        - (x1, y1): 右上角 (ceil, ceil)
        """
        # 获取四个角点的整数坐标
        x_floor = torch.floor(self.otg_coords[:, 0:1]).long()  # (N_traces, 1)
        y_floor = torch.floor(self.otg_coords[:, 1:2]).long()  # (N_traces, 1)

        self.x0 = x_floor.squeeze(-1)  # (N_traces,)
        self.y0 = y_floor.squeeze(-1)  # (N_traces,)
        self.x1 = self.x0 + 1          # (N_traces,)
        self.y1 = self.y0 + 1          # (N_traces,)

        # 计算相对偏移量（双线性插值权重）
        dx = self.otg_coords[:, 0:1] - x_floor.float()  # (N_traces, 1), x方向的偏移
        dy = self.otg_coords[:, 1:2] - y_floor.float()  # (N_traces, 1), y方向的偏移

        # 双线性插值权重
        # w00: 左下角 (x0, y0) 权重
        # w01: 左上角 (x0, y1) 权重
        # w10: 右下角 (x1, y0) 权重
        # w11: 右上角 (x1, y1) 权重
        w00 = (1.0 - dx) * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w10 = dx * (1.0 - dy)
        w11 = dx * dy
        print(f"x0 range: {self.x0.min().item()} to {self.x0.max().item()}")
        print(f"w00 range: {w00.min().item()} to {w00.max().item()}")

        # 堆叠所有权重，形状: (N_traces, 4)
        self.weights = torch.cat([w00, w01, w10, w11], dim=1).squeeze(-1)  # (N_traces, 4)

        # 边界处理：确保索引不超出网格范围
        self.x0 = torch.clamp(self.x0, 0, self.Nx - 1)
        self.x1 = torch.clamp(self.x1, 0, self.Nx - 1)
        self.y0 = torch.clamp(self.y0, 0, self.Ny - 1)
        self.y1 = torch.clamp(self.y1, 0, self.Ny - 1)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        正演算子 E (Forward Operator)

        从规则网格 (Nx, Ny, Nt) 通过双线性插值提取非规则坐标点的地震道

        Args:
            d: 规则网格数据，形状为 (Nx, Ny, Nt)

        Returns:
            d_obs: 非规则坐标点的观测数据，形状为 (N_traces, Nt)

        Note:
            使用 torch.nn.functional.grid_sample 实现高效并行计算
            grid_sample 要求坐标归一化到 [-1, 1] 范围
        """
        # device = d.device
        # dtype = d.dtype

        # # 将坐标归一化到 [-1, 1] 范围 (grid_sample 的要求)
        # # 从 [0, Nx-1] 和 [0, Ny-1] 映射到 [-1, 1]
        # x_norm = 2.0 * self.otg_coords[:, 0] / (self.Nx - 1) - 1.0  # (N_traces,)
        # y_norm = 2.0 * self.otg_coords[:, 1] / (self.Ny - 1) - 1.0  # (N_traces,)

        # # 构造 grid_sample 需要的坐标格式: (N_traces, 2)
        # # grid = torch.stack([x_norm, y_norm], dim=1)  # (N_traces, 2)
        # grid = torch.stack([y_norm, x_norm], dim=1)  # (N_traces, 2)

        # # 准备输入数据: grid_sample 期望 (batch, channels, height, width) 格式
        # # d: (Nx, Ny, Nt) -> (1, Nt, Nx, Ny)
        # # permute: 将时间维度移到通道维度
        # d_expanded = d.permute(2, 0, 1).unsqueeze(0)  # (1, Nt, Nx, Ny)

        # # 构造 grid 形状为 (1, N_traces, 1, 2)
        # # grid_sample 会自动处理 Nt (Channels)
        # grid_compact = grid.unsqueeze(0).unsqueeze(2)

        # # 使用双线性插值提取非规则坐标点的值
        # d_obs = F.grid_sample(
        #     d_expanded,
        #     grid_compact,
        #     mode='bilinear',
        #     padding_mode='border',  # 边界外推，使用边界值
        #     align_corners=False     # 像素中心对齐，更精确的插值
        # )  # 输出形状: (1, Nt, N_traces, 1)

        # # 调整输出形状: (1, Nt, N_traces, 1) -> (N_traces, Nt)
        # d_obs = d_obs.squeeze(0).squeeze(-1).permute(1, 0)

        d00 = d[self.x0, self.y0, :] 
        d01 = d[self.x0, self.y1, :]
        d10 = d[self.x1, self.y0, :]
        d11 = d[self.x1, self.y1, :]
        
        # 广播相乘并相加
        d_obs = (d00 * self.weights[:, 0:1] + 
                 d01 * self.weights[:, 1:2] + 
                 d10 * self.weights[:, 2:3] + 
                 d11 * self.weights[:, 3:4])

        return d_obs

    def adjoint(self, d_obs: torch.Tensor) -> torch.Tensor:
        """
        伴随算子 E^T (Adjoint Operator)

        将非规则观测数据 (N_traces, Nt) 根据插值权重散布回规则网格 (Nx, Ny, Nt)
        使用可导的 scatter-add 机制实现

        Args:
            d_obs: 非规则观测数据，形状为 (N_traces, Nt)

        Returns:
            d_recon: 散布回规则网格的数据，形状为 (Nx, Ny, Nt)

        Note:
            使用 torch.index_add 实现可导的散布操作
            每个观测点的值按双线性权重分配到其四个最近的网格点
        """
        device = d_obs.device
        dtype = d_obs.dtype

        # 初始化规则网格输出
        d_recon = torch.zeros(
            (self.Nx, self.Ny, self.Nt),
            device=device,
            dtype=dtype
        )

        # 将规则网格展平为一维: (Nx, Ny, Nt) -> (Nx * Ny, Nt)
        # 这样可以使用一维索引进行散布
        grid_flat = d_recon.reshape(-1, self.Nt)  # (Nx * Ny, Nt)

        # 计算四个角点的一维线性索引
        # 索引 = x * Ny + y (row-major order)
        idx00 = self.x0 * self.Ny + self.y0  # (N_traces,)
        idx01 = self.x0 * self.Ny + self.y1  # (N_traces,)
        idx10 = self.x1 * self.Ny + self.y0  # (N_traces,)
        idx11 = self.x1 * self.Ny + self.y1  # (N_traces,)

        # 将观测数据按权重加权后散布到四个角点
        # 使用 index_add 实现可导的累加操作
        # 注意：torch.index_add 返回新张量，需要赋值回去或使用原地操作 index_add_
        grid_flat = torch.index_add(grid_flat, 0, idx00, d_obs * self.weights[:, 0:1])
        grid_flat = torch.index_add(grid_flat, 0, idx01, d_obs * self.weights[:, 1:2])
        grid_flat = torch.index_add(grid_flat, 0, idx10, d_obs * self.weights[:, 2:3])
        grid_flat = torch.index_add(grid_flat, 0, idx11, d_obs * self.weights[:, 3:4])

        # 恢复形状: (Nx * Ny, Nt) -> (Nx, Ny, Nt)
        d_recon = grid_flat.reshape(self.Nx, self.Ny, self.Nt)

        return d_recon

    def __call__(self, d: torch.Tensor, adjoint: bool = False) -> torch.Tensor:
        """
        便捷调用方法

        Args:
            d: 输入数据
            adjoint: 是否使用伴随算子（默认 False，使用正演算子）

        Returns:
            如果 adjoint=False: 正演结果 E[d]
            如果 adjoint=True: 伴随结果 E^T[d]
        """
        if adjoint:
            return self.adjoint(d)
        else:
            return self.forward(d)


def run_dot_product_test():
    """执行内积测试以验证 E 和 E^T 的伴随性"""
    print("=" * 60)
    print("Starting Dot-Product Test for BilinearOperator...")
    print("=" * 60)
    
    # 测试参数设置
    Nx, Ny, Nt = 50, 50, 256
    N_traces = 1500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 强制使用 float64 (double precision) 进行数值稳定测试
    dtype = torch.float64
    
    # 1. 构造随机非规则坐标点，映射到 [0, Nx-1]x[0, Ny-1]
    otg_coords = torch.rand(N_traces, 2, dtype=dtype, device=device)
    otg_coords[:, 0] *= (Nx - 1)
    otg_coords[:, 1] *= (Ny - 1)
    
    # 初始化算子
    operator = BilinearOperator((Nx, Ny, Nt), otg_coords)
    
    # 2. 生成随机张量 d (规则网格模型空间) 和 d_obs (非规则观测数据空间)
    d_model = torch.randn(Nx, Ny, Nt, dtype=dtype, device=device)
    d_data = torch.randn(N_traces, Nt, dtype=dtype, device=device)
    
    # 3. 计算正演和伴随
    # d_fwd = E * d_model
    d_fwd = operator.forward(d_model)
    # d_adj = E^T * d_data
    d_adj = operator.adjoint(d_data)
    
    # 4. 计算两侧内积
    # inner1 = <E(d), d_data>
    inner1 = torch.sum(d_fwd * d_data)
    # inner2 = <d, E^T(d_data)>
    inner2 = torch.sum(d_model * d_adj)
    
    # 5. 分析误差
    abs_diff = torch.abs(inner1 - inner2).item()
    rel_diff = abs_diff / (torch.abs(inner1).item() + 1e-15)
    
    print(f"Device           : {device}")
    print(f"Data type        : {dtype}")
    print(f"<E d, d_obs>     = {inner1.item():.15f}")
    print(f"<d, E^T d_obs>   = {inner2.item():.15f}")
    print(f"Absolute Error   = {abs_diff:.5e}")
    print(f"Relative Error   = {rel_diff:.5e}")
    
    print("-" * 60)
    # 一般认为浮点双精度的误差在 1e-12 以内即代表数学上完美相等
    if rel_diff < 1e-12:
        print("✅ TEST PASSED: The operator is strictly adjoint!")
    else:
        print("❌ TEST FAILED: The operator violates adjointness.")

if __name__ == "__main__":
    run_dot_product_test()