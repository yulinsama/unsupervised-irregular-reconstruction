"""
DLOTG 算法主训练循环

实现论文中的 Algorithm 1，用于非规则网格地震数据重建。

算法流程：
1. 初始化规则网格数据 d_0 = zeros
2. For l in range(L):
   a. Data Update: s_l = d_{l-1} - beta * E^T(E(d_{l-1}) - d_obs)
   b. Data Separation: p_l = P(s_l)
   c. DIP-guided Reconstruction: 训练 DIP 网络最小化 ||p_l - R(p_l, theta)||^2
   d. Data Integration: d_l = P^{-1}(r_l, grid_shape)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import sys
import os
import numpy as np
import h5py
import scipy.io
import argparse
import matplotlib.pyplot as plt
# import torch.nn.functional as F

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.seismic_operator import BilinearOperator
from src.data.patching import PatchOperator
from src.models.dip_network import DipNetwork


def normalize_coordinates(otg_coords: torch.Tensor, grid_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    将物理坐标映射到 [0, Nx-1] x [0, Ny-1] 的网格索引空间
    """
    Nx, Ny, _ = grid_shape
    
    # 提取 X 和 Y 坐标
    x_coords = otg_coords[:, 0]
    y_coords = otg_coords[:, 1]
    
    # 获取物理坐标的真实边界
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # 映射到 [0, Nx-1] 和 [0, Ny-1]
    # 公式: (x - min) / (max - min) * (N - 1)
    x_grid = (x_coords - x_min) / (x_max - x_min) * (Nx - 1)
    y_grid = (y_coords - y_min) / (y_max - y_min) * (Ny - 1)
    
    # 重新拼合
    mapped_coords = torch.stack([x_grid, y_grid], dim=1)
    
    return mapped_coords

def dlotg_algorithm(
    d_obs: torch.Tensor,
    otg_coords: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int] = (15, 15, 15),
    shift: Tuple[int, int, int] = (1, 1, 1),
    L: int = 15,
    inner_epochs: int = 50,
    lr: float = 0.001,
    beta: float = 0.1,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
    save_intermediate: bool = True,
    intermediate_dir: str = None,
    init_type: str = 'gaussian',
    init_std_scale: float = 1,
    save_frequency: int = 10
) -> torch.Tensor:
    """
    DLOTG 算法主函数 (Algorithm 1)

    Args:
        d_obs: 非规则观测数据，形状为 (N_traces, Nt)
        otg_coords: 非规则地震道的 (x, y) 坐标，形状为 (N_traces, 2)
        grid_shape: 规则网格的形状 (Nx, Ny, Nt)
        patch_size: 每个patch的大小 (px, py, pt)
        shift: 滑动窗口的步长 (sx, sy, st)
        L: 外层 POCS 迭代次数
        inner_epochs: 内层 DIP 网络训练的 epoch 数
        lr: 学习率
        beta: 梯度下降步长
        device: 计算设备 ('cuda' 或 'cpu')
        verbose: 是否打印训练信息
        save_intermediate: 是否保存中间结果
        intermediate_dir: 中间结果保存目录
        init_type: 初始化类型 ('zeros' 或 'gaussian')
        init_std_scale: 高斯分布标准差比例
        save_frequency: 保存中间结果的频率（每N个epoch保存一次）

    Returns:
        d_final: 重建后的规则网格数据，形状为 (Nx, Ny, Nt)
    """
    # 将数据移动到指定设备
    d_obs = d_obs.to(device)
    otg_coords = otg_coords.to(device)

    # 修复越界问题
    otg_coords_mapped = normalize_coordinates(otg_coords, grid_shape)

    # 初始化算子
    bilinear_op = BilinearOperator(grid_shape, otg_coords_mapped)
    # bilinear_op = BilinearOperator(grid_shape, otg_coords)
    patch_op = PatchOperator(patch_size, shift)

    # 阶段 0: 初始化规则网格数据
    if init_type == 'zeros':
        d_current = torch.zeros(grid_shape, device=device, dtype=d_obs.dtype)
        if verbose:
            print(f"Initialization: Zeros")
    else:  # gaussian
        # 使用高斯分布初始化，标准差基于观测数据的统计特性
        obs_std = d_obs.std()
        d_current = torch.randn(grid_shape, device=device, dtype=d_obs.dtype) * (obs_std * init_std_scale)

        if verbose:
            print(f"Initialization:")
            print(f"  Type: Gaussian")
            print(f"  Observed data std: {obs_std:.6f}")
            print(f"  Initial std scaling factor: {init_std_scale}")
            print(f"  Initial data range: [{d_current.min():.6f}, {d_current.max():.6f}]")
            print(f"  Initial data std: {d_current.std():.6f}")

    # 记录原始形状，用于 unpatch
    original_shape = grid_shape

    # 初始化 DIP 网络（在 POCS 循环外，让网络能够记住学到的先验结构）
    # 首先需要计算 patch 的维度
    dummy_patch = patch_op.patch(d_current)
    patch_dim = dummy_patch.shape[1]  # px * py * pt
    dip_net = DipNetwork(patch_dim).to(device)

    # 初始化优化器
    optimizer = optim.Adam(dip_net.parameters(), lr=lr)

    # 创建中间结果保存目录
    if save_intermediate and intermediate_dir:
        os.makedirs(intermediate_dir, exist_ok=True)
        if verbose:
            print(f"Intermediate results will be saved to: {intermediate_dir}")

    # 外层 POCS 迭代
    for l in range(L):
        if verbose:
            print(f"\n=== POCS Iteration {l+1}/{L} ===")

        # ========== 阶段 1: Data Update ==========
        # 计算正演: E(d_{l-1})
        d_forward = bilinear_op.forward(d_current)  # (N_traces, Nt)
        # print(f"d_forward range: {d_forward.min().item()} to {d_forward.max().item()}")

        # 计算残差: E(d_{l-1}) - d_obs
        residual = d_forward - d_obs  # (N_traces, Nt)
        # print(f"residual range: {residual.min().item()} to {residual.max().item()}")

        # 计算伴随: E^T(residual)
        grad = bilinear_op.adjoint(residual)  # (Nx, Ny, Nt)
        # print(f"grad range: {grad.min().item()} to {grad.max().item()}")

        # 梯度下降步: s_l = d_{l-1} - beta * grad
        s_l = d_current - beta * grad  # (Nx, Ny, Nt)
        # print(f"s_l range: {s_l.min().item()} to {s_l.max().item()}")
        if save_intermediate and intermediate_dir:
            s_l_filename = f"s_l_iteration_{l+1}.mat"
            s_l_path = os.path.join(intermediate_dir, s_l_filename)
            scipy.io.savemat(s_l_path, {
                's_l': s_l.detach().cpu().numpy(),
                'iteration': l + 1
            })


        # 计算当前的目标函数值 (数据保真度)
        data_fidelity = torch.norm(residual, p=2).item()
        if verbose:
            print(f"Data Update: Data Fidelity = {data_fidelity:.4f}")

        # ========== 阶段 2: Data Separation ==========
        # Patching: p_l = P(s_l)
        p_l = patch_op.patch(s_l)  # (N_patches, px*py*pt)

        if verbose:
            print(f"Data Separation: {p_l.shape[0]} patches created")

        # ========== 阶段 3: DIP-guided Reconstruction ==========
        # 训练 DIP 网络（网络在 POCS 循环外初始化，能够记住学到的先验结构）
        if verbose:
            print(f"DIP Training: Training for {inner_epochs} epochs...")

        for epoch in range(inner_epochs):
            # 前向传播: r = R(p, theta)
            # print(f"p_l range: {p_l.min().item()} to {p_l.max().item()}")
            r = dip_net(p_l)  # (N_patches, px*py*pt)
            # print(f"r range: {r.min().item()} to {r.max().item()}")

            # 计算损失: L = ||p_l - r||^2 (无监督自编码损失)
            loss = torch.norm(r - p_l, p=2) ** 2
            # loss = nn.functional.l1_loss(r, p_l)
            # loss = nn.functional.mse_loss(r, p_l)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印损失
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{inner_epochs}, Loss = {loss.item():.4f}")

            # 保存中间结果（每 save_frequency 个 epoch 保存一次）
            if save_intermediate and intermediate_dir and (epoch + 1) % save_frequency == 0:
                # 重建当前时刻的网格
                r_l_current = r.detach()
                d_current_intermediate = patch_op.unpatch(r_l_current, original_shape)

                # 保存文件名格式: reconstruction_result_n_iteration_m_epoch.mat
                filename = f"reconstruction_result_{l+1}_iteration_{epoch+1}_epoch.mat"
                save_path = os.path.join(intermediate_dir, filename)
                scipy.io.savemat(save_path, {
                    'd_recon': d_current_intermediate.cpu().numpy(),
                    'iteration': l + 1,
                    'epoch': epoch + 1,
                    'loss': loss.item()
                })

        # 获取重建的 patches
        r_l = dip_net(p_l).detach()  # (N_patches, px*py*pt)

        # 计算重建损失
        reconstruction_loss = torch.norm(r_l - p_l, p=2).item()
        if verbose:
            print(f"DIP Reconstruction: Loss = {reconstruction_loss:.4f}")

        # ========== 阶段 4: Data Integration ==========
        # Unpatch: d_l = P^{-1}(r_l, grid_shape)
        d_current = patch_op.unpatch(r_l, original_shape)  # (Nx, Ny, Nt)

        # 计算当前 POCS 迭代的目标函数值
        current_forward = bilinear_op.forward(d_current)
        current_residual = current_forward - d_obs
        objective = torch.norm(current_residual, p=2).item()
        if verbose:
            print(f"Data Integration: Objective = {objective:.4f}")

        # 保存每个 POCS 迭代完成后的结果
        if save_intermediate and intermediate_dir:
            filename = f"reconstruction_result_{l+1}_iteration_final.mat"
            save_path = os.path.join(intermediate_dir, filename)
            scipy.io.savemat(save_path, {
                'd_recon': d_current.detach().cpu().numpy(),  # 加上 .detach() 最安全
                's_l': s_l.detach().cpu().numpy(),
                'grad': grad.detach().cpu().numpy(),
                'iteration': l + 1,
                'epoch': inner_epochs,
                'objective': objective
            })

    return d_current


def generate_dummy_data(
    grid_shape: Tuple[int, int, int] = (50, 50, 100),
    missing_ratio: float = 0.3,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成带有缺失值的 Dummy 数据用于测试

    Args:
        grid_shape: 规则网格的形状 (Nx, Ny, Nt)
        missing_ratio: 缺失数据的比例
        device: 计算设备

    Returns:
        d_true: 真实的规则网格数据，形状为 (Nx, Ny, Nt)
        d_obs: 非规则观测数据，形状为 (N_traces, Nt)
        otg_coords: 非规则地震道的 (x, y) 坐标，形状为 (N_traces, 2)
    """
    Nx, Ny, Nt = grid_shape

    # 生成虚假的规则网格数据 (使用随机噪声 + 简单的信号)
    torch.manual_seed(42)
    d_true = torch.randn(Nx, Ny, Nt, device=device)

    # 添加一些简单的结构 (例如：在空间上有变化的信号)
    x_grid, y_grid, t_grid = torch.meshgrid(
        torch.arange(Nx, device=device) / Nx,
        torch.arange(Ny, device=device) / Ny,
        torch.arange(Nt, device=device) / Nt,
        indexing='ij'
    )
    signal = 0.5 * torch.sin(2 * 3.14159 * (x_grid + y_grid) * 2) * torch.exp(-t_grid)
    d_true = d_true + signal

    # 生成非规则坐标 (随机采样)
    N_traces = int((1 - missing_ratio) * Nx * Ny)
    total_positions = Nx * Ny

    # 随机选择位置
    all_indices = torch.randperm(total_positions)
    selected_indices = all_indices[:N_traces]

    # 转换为 (x, y) 坐标
    x_coords = selected_indices // Ny
    y_coords = selected_indices % Ny

    otg_coords = torch.stack([x_coords, y_coords], dim=1).float()  # (N_traces, 2)

    # 提取对应的地震道 (使用简单的最近邻提取)
    d_obs = torch.zeros(N_traces, Nt, device=device)
    for i, (x, y) in enumerate(otg_coords):
        d_obs[i] = d_true[int(x), int(y), :]

    return d_true, d_obs, otg_coords


if __name__ == "__main__":
    """
    主函数：加载真实数据并运行 DLOTG 算法

    使用真实的非规则网格地震数据进行重建任务。
    """
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='DLOTG Algorithm for Seismic Data Reconstruction')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (e.g., cuda:0, cuda:1, cuda:2, cpu)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the data file (default: data/curve_3D_irregular_clean_2.mat)')
    parser.add_argument('--grid-shape', type=int, nargs=3, default=[20, 20, 256],
                        help='Grid shape (Nx, Ny, Nt) (default: 50 50 256)')
    parser.add_argument('--patch-size', type=int, nargs=3, default=[15, 15, 15],
                        help='Patch size (px, py, pt) (default: 15 15 15)')
    parser.add_argument('--shift', type=int, nargs=3, default=[5, 5, 5],
                        help='Shift step (sx, sy, st) (default: 5 5 5)')
    parser.add_argument('--L', type=int, default=25,
                        help='Number of POCS iterations (default: 15)')
    parser.add_argument('--inner-epochs', type=int, default=50,
                        help='Number of DIP training epochs per iteration (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--beta', type=float, default=0.2,
                        help='Gradient descent step size (default: 0.1)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the result (default: data/reconstructed_result.mat)')
    parser.add_argument('--pause-after-load', action='store_true',
                        help='Pause after loading data to visualize before running algorithm')
    parser.add_argument('--transpose-data', action='store_true',
                        help='Transpose the data (swap time and trace dimensions)')
    parser.add_argument('--save-intermediate',type=bool, default=True,
                        help='Save intermediate results (each iteration and epoch)')
    parser.add_argument('--intermediate-dir', type=str, default='/home/data1/yuhao/seismic-dlotg/data/intermediate_results_0.2b_linear',
                        help='Directory to save intermediate results (default: data/intermediate_results)')
    parser.add_argument('--save-frequency', type=int, default=10,
                        help='Save intermediate results every N epochs (default: 10)')
    parser.add_argument('--init-type', type=str, default='gaussian',
                        choices=['zeros', 'gaussian'],
                        help='Initialization type: zeros or gaussian (default: gaussian)')
    parser.add_argument('--init-std-scale', type=float, default=0.1,
                        help='Standard deviation scale for Gaussian initialization relative to observed data std (default: 0.1)')

    args = parser.parse_args()

    print("=" * 60)
    print("DLOTG Algorithm - Real Data Task")
    print("=" * 60)

    # 显示使用的设备
    device = args.device
    print(f"\nUsing device: {device}")

    # 验证设备可用性
    if device.startswith('cuda'):
        gpu_id = int(device.split(':')[1]) if ':' in device else 0
        if not torch.cuda.is_available():
            print(f"Warning: CUDA is not available, falling back to CPU")
            device = 'cpu'
        elif gpu_id >= torch.cuda.device_count():
            print(f"Warning: GPU {gpu_id} not available, falling back to cuda:0")
            device = 'cuda:0'

    # 1. 路径设置
    if args.data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 'data', 'curve_3D_irregular_clean_linear.mat')
    else:
        data_path = args.data_path

    # 2. 加载 Matlab 数据 (使用 h5py 支持 v7.3 格式)
    print(f"Loading data from {data_path}...")
    try:
        # 使用 h5py 加载 MATLAB v7.3 格式的文件
        with h5py.File(data_path, 'r') as f:
            # 查看可用变量
            print(f"  Available variables: {list(f.keys())}")

            # 加载数据
            # d: 非规则观测数据 (通常形状为 (Nt, N_traces) 或 (N_traces, Nt))
            # sx_ir, sy_ir: 非规则地震道的 (x, y) 坐标
            d_obs = torch.from_numpy(f['d'][:]).float()
            sx_ir = torch.from_numpy(f['sx_ir'][:]).float()
            sy_ir = torch.from_numpy(f['sy_ir'][:]).float()

            # # 1. 检查数据类型并转为 NumPy
            # if isinstance(d_obs, torch.Tensor):
            #     # 如果数据在 GPU 上，需要加 .cpu()
            #     data_np = d_obs.cpu().detach().numpy()
            # else:
            #     data_np = d_obs

            # print(f"NumPy 数据维度: {data_np.shape}")

            # # 2. 地震数据常用色彩映射
            # # 'seismic' (红白蓝) 或 'gray' (灰度)
            # cmap = 'seismic' 

            # # 3. 绘制巨大的二维图
            # plt.figure(figsize=(16, 6)) # 设宽一点，因为道数很多

            # # imshow 默认垂直轴是第 0 维 (2500)，水平轴是第 1 维 (256)
            # # 这不符合地震习惯（时间轴通常向下）。我们需要转置(.T)。
            # plt.imshow(data_np.T, cmap=cmap, aspect='auto')

            # plt.title('二维显示: 所有 2500 道排列')
            # plt.ylabel('时间/深度轴 (Index)')
            # plt.xlabel('道号 (Trace Index)')
            # plt.colorbar(label='振幅')

            # plt.tight_layout()
            # plt.show()

            print(f"\n  Original d_obs shape: {d_obs.shape}")
            print(f"  Original sx_ir shape: {sx_ir.shape}")
            print(f"  Original sy_ir shape: {sy_ir.shape}")

            # 处理坐标形状 (从 (1, N_traces) 调整为 (N_traces,))
            sx_ir = sx_ir.squeeze()  # (N_traces,)
            sy_ir = sy_ir.squeeze()  # (N_traces,)

            # 根据 sx_ir, sy_ir 的形状推断 N_traces
            N_traces = sx_ir.shape[0]

            # 检查数据形状是否正确
            # dlotg_algorithm 期望 d_obs 形状为 (N_traces, Nt)
            # 我们根据坐标数量来推断哪个维度是时间维度
            if d_obs.shape[0] == N_traces:
                # 数据已经是 (N_traces, Nt)，不需要转置
                print(f"  Data shape is correct: (N_traces={d_obs.shape[0]}, Nt={d_obs.shape[1]})")
            elif d_obs.shape[1] == N_traces:
                # 数据是 (Nt, N_traces)，需要转置
                print(f"  Transposing data from {d_obs.shape} to {(d_obs.shape[1], d_obs.shape[0])}")
                d_obs = d_obs.t()
            elif args.transpose_data:
                # 用户指定要转置
                print(f"  Transposing data as requested by --transpose-data flag")
                d_obs = d_obs.t()
            else:
                # 无法自动判断，给出警告
                print(f"\n  [WARNING] Cannot automatically determine data orientation!")
                print(f"  Coordinate data suggests N_traces = {N_traces}")
                print(f"  Current d_obs shape: {d_obs.shape}")
                print(f"  Neither dimension matches N_traces")
                print(f"  Use --transpose-data flag if data needs transposition\n")

            # 合并 x 和 y 坐标
            otg_coords = torch.stack([sx_ir, sy_ir], dim=1)  # (N_traces, 2)

            print(f"\n  Loaded successfully!")
            print(f"  d_obs shape: {d_obs.shape}")
            print(f"  otg_coords shape: {otg_coords.shape}")
            print(f"  Data range: [{d_obs.min():.4f}, {d_obs.max():.4f}]")
            print(f"  Data mean: {d_obs.mean():.4f}")
            print(f"  Data std: {d_obs.std():.4f}")

            # # 取第 128 个时间采样点的所有道数据
            # # d_obs 形状是 [2500, 256]
            # slice_data = d_obs[:, 128].cpu().numpy()
            # x_coords = otg_coords[:, 0].cpu().numpy()
            # y_coords = otg_coords[:, 1].cpu().numpy()

            # plt.figure(figsize=(8, 6))
            # # 画散点图，c 指定颜色，s 指定点的大小
            # scatter = plt.scatter(x_coords, y_coords, c=slice_data, cmap='seismic', s=15)
            # plt.colorbar(scatter, label='Amplitude')
            # plt.title('Time Slice at idx=128 (Direct Scatter Plot)')
            # plt.xlabel('X Coordinate')
            # plt.ylabel('Y Coordinate')
            # plt.axis('equal')
            # plt.show()

            # 定义规则网格形状 (根据数据调整)
            # 根据坐标范围和文件信息设定
            grid_shape = tuple(args.grid_shape)
            # 如果没有指定，根据数据维度自动设置
            if grid_shape[2] != d_obs.shape[1]:
                grid_shape = (grid_shape[0], grid_shape[1], d_obs.shape[1])
            print(f"  Grid shape: {grid_shape}")

            # 可视化加载的数据（如果用户选择暂停）
            # if args.pause_after_load:
            #     print("\n" + "=" * 60)
            #     print("Visualizing loaded data...")
            #     print("=" * 60)

            #     try:
            #         import matplotlib.pyplot as plt

            #         # 创建子图
            #         fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            #         # 1. 显示前几条地震道
            #         n_traces_show = min(5, d_obs.shape[0])
            #         for i in range(n_traces_show):
            #             axes[0, 0].plot(d_obs[i, :].numpy(), label=f'Trace {i}')
            #         axes[0, 0].set_xlabel('Time')
            #         axes[0, 0].set_ylabel('Amplitude')
            #         axes[0, 0].set_title(f'Sample Traces (First {n_traces_show})')
            #         axes[0, 0].legend()
            #         axes[0, 0].grid(True, alpha=0.3)

            #         # 2. 显示坐标分布
            #         axes[0, 1].scatter(otg_coords[:, 0].numpy(), otg_coords[:, 1].numpy(),
            #                           s=10, alpha=0.6, c='blue')
            #         axes[0, 1].set_xlabel('X coordinate')
            #         axes[0, 1].set_ylabel('Y coordinate')
            #         axes[0, 1].set_title('Trace Coordinate Distribution')
            #         axes[0, 1].grid(True, alpha=0.3)
            #         axes[0, 1].set_aspect('equal')

            #         # 3. 显示数据的平均值随时间变化
            #         mean_trace = d_obs.mean(dim=0).numpy()
            #         axes[1, 0].plot(mean_trace, linewidth=2, color='red')
            #         axes[1, 0].set_xlabel('Time')
            #         axes[1, 0].set_ylabel('Amplitude')
            #         axes[1, 0].set_title('Mean Trace')
            #         axes[1, 0].grid(True, alpha=0.3)

            #         # 4. 显示数据的标准差随时间变化
            #         std_trace = d_obs.std(dim=0).numpy()
            #         axes[1, 1].plot(std_trace, linewidth=2, color='green')
            #         axes[1, 1].set_xlabel('Time')
            #         axes[1, 1].set_ylabel('Amplitude')
            #         axes[1, 1].set_title('Standard Deviation Over Time')
            #         axes[1, 1].grid(True, alpha=0.3)

            #         plt.tight_layout()
            #         plt.savefig('data_loading_visualization.png', dpi=150, bbox_inches='tight')
            #         print("  Saved visualization to 'data_loading_visualization.png'")
            #         plt.show()

            #     except ImportError:
            #         print("  [WARNING] matplotlib not installed, skipping visualization")
            #         print("  Install with: pip install matplotlib")

            #     print("\n" + "=" * 60)
            #     input("Press Enter to continue running the algorithm, or Ctrl+C to exit...")
            #     print("=" * 60 + "\n")

    except Exception as e:
        print(f"Error loading .mat file: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # 3. 配置算法参数 (从命令行参数获取)
    patch_size = tuple(args.patch_size)
    shift = tuple(args.shift)
    L = args.L
    inner_epochs = args.inner_epochs
    lr = args.lr
    beta = args.beta

    config = {
        'grid_shape': grid_shape,
        'patch_size': patch_size,
        'shift': shift,
        'L': L,
        'inner_epochs': inner_epochs,
        'lr': lr,
        'beta': beta,
        'device': device
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 运行 DLOTG 算法
    print(f"\nRunning DLOTG algorithm...")

    # 设置中间结果保存目录
    if args.save_intermediate:
        if args.intermediate_dir is None:
            intermediate_dir = os.path.join(project_root, 'data', 'intermediate_results')
        else:
            intermediate_dir = args.intermediate_dir
        print(f"Intermediate results will be saved to: {intermediate_dir}")
    else:
        intermediate_dir = None

    try:
        d_reconstructed = dlotg_algorithm(
            d_obs=d_obs,
            otg_coords=otg_coords,
            grid_shape=grid_shape,
            patch_size=patch_size,
            shift=shift,
            L=L,
            inner_epochs=inner_epochs,
            lr=lr,
            beta=beta,
            device=device,
            verbose=True,
            save_intermediate=True ,
            intermediate_dir=intermediate_dir,
            init_type=args.init_type,
            init_std_scale=1,
            save_frequency=args.save_frequency
        )

        print(f"\n" + "=" * 60)
        print("DLOTG Algorithm Completed Successfully!")
        print("=" * 60)
        print(f"  Reconstructed grid shape: {d_reconstructed.shape}")

        # 5. 保存结果
        if args.save_path is None:
            save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'data', 'reconstructed_result.mat')
        else:
            save_path = args.save_path
        scipy.io.savemat(save_path, {'d_recon': d_reconstructed.cpu().numpy()})
        print(f"  Result saved to {save_path}")

        print("\n[OK] Pipeline completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Algorithm failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
