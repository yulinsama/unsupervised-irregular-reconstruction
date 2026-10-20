"""
3D 数据体的分块 (Patching) 和重组机制

实现论文中的 3D Patching 算子 P 和 P^{-1}，用于将 3D 地震数据体
切分为重叠的小块，并在重组时处理重叠区域的平均。

Patching 算子 P: 将 (Nx, Ny, Nt) 切分为 (N_patches, px*py*pt)
重组算子 P^{-1}: 将 patches 重组回原始形状，重叠区域求平均值
"""

import torch
from typing import Tuple


class PatchOperator:
    """
    3D 数据体分块和重组算子

    支持步长小于块大小的重叠分块，重组时在重叠区域计算平均值。
    """

    def __init__(self, patch_size: Tuple[int, int, int], shift: Tuple[int, int, int]):
        """
        初始化 Patch 算子

        Args:
            patch_size: 每个patch的大小 (px, py, pt)
            shift: 滑动窗口的步长 (sx, sy, st)，当 shift < patch_size 时产生重叠
        """
        self.patch_size = patch_size  # (px, py, pt)
        self.shift = shift  # (sx, sy, st)

    def patch(self, d: torch.Tensor) -> torch.Tensor:
        """
        将 3D 张量切分为 patches (Patching 算子 P)

        Args:
            d: 输入 3D 张量，形状为 (Nx, Ny, Nt)

        Returns:
            patches: 切分后的 patches，形状为 (N_patches, px * py * pt)

        Note:
            使用滑动窗口提取 patches，当 shift < patch_size 时会产生重叠
            patches 是展平后的，便于后续的全连接网络处理
        """
        Nx, Ny, Nt = d.shape
        px, py, pt = self.patch_size
        sx, sy, st = self.shift

        # 计算 patch 的数量
        num_patches_x = (Nx - px) // sx + 1
        num_patches_y = (Ny - py) // sy + 1
        num_patches_t = (Nt - pt) // st + 1
        num_patches = num_patches_x * num_patches_y * num_patches_t

        # 使用 unfold 方法高效提取 patches
        # unfold 只能在单个维度上操作，需要分步进行

        # 1. 在 x 方向上 unfold
        d_unfolded_x = d.unfold(0, px, sx)  # (num_patches_x, Ny, Nt, px)

        # 2. 在 y 方向上 unfold
        d_unfolded_xy = d_unfolded_x.unfold(1, py, sy)  # (num_patches_x, num_patches_y, Nt, px, py)

        # 3. 在 t 方向上 unfold
        d_unfolded_xyt = d_unfolded_xy.unfold(2, pt, st)  # (num_patches_x, num_patches_y, num_patches_t, px, py, pt)

        # 重塑为 (num_patches, px, py, pt)
        patches = d_unfolded_xyt.permute(3, 4, 5, 0, 1, 2).reshape(px, py, pt, num_patches)
        patches = patches.permute(3, 0, 1, 2)  # (num_patches, px, py, pt)

        # 确保张量连续后再展平
        patches_flat = patches.contiguous().reshape(num_patches, -1)

        return patches_flat

    def unpatch(self, patches: torch.Tensor, original_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        将 patches 重组回原始 3D 数据体 (重组算子 P^{-1})

        Args:
            patches: 展平的 patches，形状为 (N_patches, px * py * pt)
            original_shape: 原始数据的形状 (Nx, Ny, Nt)

        Returns:
            d_recon: 重组后的 3D 张量，形状为 (Nx, Ny, Nt)

        Note:
            当 shift < patch_size 时，在重叠区域计算平均值
            使用 count map 记录每个体素被覆盖的次数
        """
        Nx, Ny, Nt = original_shape
        px, py, pt = self.patch_size
        sx, sy, st = self.shift
        num_patches = patches.shape[0]

        # 计算 patch 的数量
        num_patches_x = (Nx - px) // sx + 1
        num_patches_y = (Ny - py) // sy + 1
        num_patches_t = (Nt - pt) // st + 1

        # 恢复 patches 的形状: (N_patches, px * py * pt) -> (N_patches, px, py, pt)
        patches_reshaped = patches.reshape(num_patches, px, py, pt)

        # 初始化输出张量和 count map
        device = patches.device
        dtype = patches.dtype

        d_recon = torch.zeros((Nx, Ny, Nt), device=device, dtype=dtype)
        count_map = torch.zeros((Nx, Ny, Nt), device=device, dtype=dtype)

        # 使用循环将每个 patch 放回原位（因为需要处理重叠）
        # 对于大规模数据，可以考虑使用更高效的实现（如 scatter_add）
        patch_idx = 0
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                for k in range(num_patches_t):
                    # 计算当前 patch 在原图中的位置
                    start_x = i * sx
                    end_x = start_x + px

                    start_y = j * sy
                    end_y = start_y + py

                    start_t = k * st
                    end_t = start_t + pt

                    # 将 patch 累加到对应位置
                    d_recon[start_x:end_x, start_y:end_y, start_t:end_t] += patches_reshaped[patch_idx]

                    # 更新 count map
                    count_map[start_x:end_x, start_y:end_y, start_t:end_t] += 1.0

                    patch_idx += 1

        # 避免除零：count_map 中为 0 的位置保持为 0（这些位置没有被任何 patch 覆盖）
        count_map[count_map == 0] = 1.0

        # 在重叠区域计算平均值
        d_recon = d_recon / count_map

        return d_recon

    def __call__(self, d: torch.Tensor, unpatch: bool = False) -> torch.Tensor:
        """
        便捷调用方法

        Args:
            d: 输入数据（patch时为原始数据，unpatch时为patches）
            unpatch: 是否执行 unpatch 操作（默认 False，执行 patch）

        Returns:
            如果 unpatch=False: patches
            如果 unpatch=True: 重组后的数据
        """
        if unpatch:
            raise ValueError(
                "unpatch() requires explicit original_shape parameter. "
                "Please call unpatch() directly with original_shape."
            )
        else:
            return self.patch(d)

