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


def normalize_coordinates(
    otg_coords: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    reg_coords: Tuple[torch.Tensor, torch.Tensor] = None,
) -> torch.Tensor:
    """
    将物理坐标映射到 [0, Nx-1] x [0, Ny-1] 的网格索引空间

    Args:
        otg_coords: 不规则道坐标 (N_traces, 2)，物理坐标单位
        grid_shape: 规则网格形状 (Nx, Ny, Nt)
        reg_coords: 规则网格的物理坐标 (sx_r, sy_r)，形状分别为 (Nx,) / (Ny,)。
            若提供，则用真实网格原点与间距做映射（正确做法）；
            若为 None，则退回到按不规则点 min/max 拉伸（旧逻辑，会扭曲位置，不推荐）。
    """
    Nx, Ny, _ = grid_shape

    # 提取 X 和 Y 坐标
    x_coords = otg_coords[:, 0]
    y_coords = otg_coords[:, 1]

    if reg_coords is not None:
        # 用规则网格的真实原点与间距做映射: index = (x - x0) / dx
        # dx 取 (端点距) / (N-1)，对均匀网格成立（本数据网格间距固定为 30）
        sx_r, sy_r = reg_coords
        sx_r = sx_r.to(device=x_coords.device, dtype=x_coords.dtype)
        sy_r = sy_r.to(device=y_coords.device, dtype=y_coords.dtype)

        dx = (sx_r[-1] - sx_r[0]) / (Nx - 1)
        dy = (sy_r[-1] - sy_r[0]) / (Ny - 1)

        x_grid = (x_coords - sx_r[0]) / dx
        y_grid = (y_coords - sy_r[0]) / dy
    else:
        # [回退/旧逻辑] 按不规则点自身范围拉伸到 [0, N-1]
        # 注意: 这会按不规则点的张成范围缩放, 与真实网格间距不一致, 会扭曲道的定位
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        x_grid = (x_coords - x_min) / (x_max - x_min) * (Nx - 1)
        y_grid = (y_coords - y_min) / (y_max - y_min) * (Ny - 1)

    # 重新拼合
    mapped_coords = torch.stack([x_grid, y_grid], dim=1)

    return mapped_coords

def dlotg_algorithm(
    d_obs: torch.Tensor,
    otg_coords: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    reg_coords: Tuple[torch.Tensor, torch.Tensor] = None,
    patch_size: Tuple[int, int, int] = (15, 15, 15),
    shift: Tuple[int, int, int] = (1, 1, 1),
    L: int = 25,
    inner_epochs: int = 50,
    lr: float = 1e-4,
    beta: float = 0.5,
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
    if reg_coords is not None:
        reg_coords = (reg_coords[0].to(device), reg_coords[1].to(device))

    # 用规则网格坐标把物理坐标映射到网格索引空间（正确的插值定位）
    otg_coords_mapped = normalize_coordinates(otg_coords, grid_shape, reg_coords)

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
            # DIP 重建 (论文公式 12): min_theta ||p_l - R(p_l)||^2, 普通自编码。
            r = dip_net(p_l)  # (N_patches, px*py*pt)

            # 损失: 与公式 12 的 ||.||^2 (SSE) 等价, 仅差一个常数因子 (patch 元素数, 被 lr 吸收)。
            # 用 mse 量级稳定、lr 可移植 (原始代码用的是 SSE: torch.norm(r - p_l, p=2) ** 2)。
            loss = nn.functional.mse_loss(r, p_l)

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
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use (e.g., cuda:0, cuda:1, cuda:2, cpu)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the data file (default: data/curve_3D_irregular_clean_2.mat)')
    parser.add_argument('--grid-shape', type=int, nargs=3, default=[100, 100, 512],
                        help='Grid shape (Nx, Ny, Nt) (default: 50 50 256)')
    parser.add_argument('--patch-size', type=int, nargs=3, default=[10, 10, 64],
                        help='Patch size (px, py, pt) (default: 10 10 10). '
                             'Should be well below the spatial grid size, else too few patches.')
    parser.add_argument('--shift', type=int, nargs=3, default=[5, 5, 32],
                        help='Shift step (sx, sy, st) (default: 2 2 2)')
    parser.add_argument('--L', type=int, default=50,
                        help='Number of POCS iterations (default: 25)')
    parser.add_argument('--inner-epochs', type=int, default=25,
                        help='DIP training epochs per POCS iteration (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Data-update step size (default: 0.5).')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the result (default: data/reconstructed_result.mat)')
    parser.add_argument('--pause-after-load', action='store_true',
                        help='Pause after loading data to visualize before running algorithm')
    parser.add_argument('--transpose-data', action='store_true',
                        help='Transpose the data (swap time and trace dimensions)')
    parser.add_argument('--save-intermediate', action=argparse.BooleanOptionalAction, default=True,
                        help='Save intermediate results (use --no-save-intermediate to disable)')
    parser.add_argument('--intermediate-dir', type=str, default='/home/data1/yuhao/seismic-dlotg/data/intermediate_results_0.5b_real_0.5miss_0.3jit_shift_10',
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
                                 'data', 'real_3D_irregular_clean_0.5mis_0.3jit.mat')
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

            # 加载规则网格的物理坐标（用于正确的 物理坐标 -> 网格索引 映射）
            sx_r = torch.from_numpy(f['sx_r'][:]).float().squeeze() if 'sx_r' in f else None
            sy_r = torch.from_numpy(f['sy_r'][:]).float().squeeze() if 'sy_r' in f else None

            

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

            # 只保留真实观测道，丢弃被抽掉的缺失道(全 0)。
            # 否则缺失道的 0 会被保真项 E(d)-d_obs 当成"在该位置观测到 0"，
            # 把重建强制拉向 0，与"插值缺失道"的目标冲突。
            obs_mask = d_obs.norm(dim=1) > 1e-8
            print(f"  Observed nonzero traces: {obs_mask.sum().item()} / {d_obs.shape[0]}")
            d_obs = d_obs[obs_mask]
            sx_ir = sx_ir[obs_mask]
            sy_ir = sy_ir[obs_mask]

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
            Nx_g, Ny_g, _ = grid_shape
            # 空间维以规则网格坐标 sx_r/sy_r 的长度为准（这才是规则化的目标网格）
            if sx_r is not None and sy_r is not None:
                Nx_g, Ny_g = sx_r.shape[0], sy_r.shape[0]
            # 时间维以数据为准
            grid_shape = (Nx_g, Ny_g, d_obs.shape[1])
            print(f"  Grid shape: {grid_shape}")
            if sx_r is not None and sy_r is not None:
                print(f"  Using regular grid coords: "
                      f"x in [{sx_r.min():.1f}, {sx_r.max():.1f}] ({sx_r.shape[0]} pts), "
                      f"y in [{sy_r.min():.1f}, {sy_r.max():.1f}] ({sy_r.shape[0]} pts)")
            else:
                print(f"  [WARNING] sx_r/sy_r not found in file; "
                      f"falling back to min/max coordinate stretching (less accurate)")

            

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
            reg_coords=(sx_r, sy_r) if sx_r is not None and sy_r is not None else None,
            patch_size=patch_size,
            shift=shift,
            L=L,
            inner_epochs=inner_epochs,
            lr=lr,
            beta=beta,
            device=device,
            verbose=True,
            save_intermediate=args.save_intermediate,
            intermediate_dir=intermediate_dir,
            init_type=args.init_type,
            init_std_scale=args.init_std_scale,
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
