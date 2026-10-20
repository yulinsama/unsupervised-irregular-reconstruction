"""
批量运行 DLOTG 实验脚本

测试不同的 iteration (L) 和 inner_epochs 组合
结果保存到 data/experiments_results 文件夹
"""

import os
import sys
import torch
import argparse
from typing import List, Tuple

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.main_dlotg import dlotg_algorithm, normalize_coordinates, generate_dummy_data
import h5py
import scipy.io


def load_data(data_path, device='cpu'):
    """加载地震数据"""
    with h5py.File(data_path, 'r') as f:
        d_obs = torch.from_numpy(f['d'][:]).float()
        sx_ir = torch.from_numpy(f['sx_ir'][:]).float()
        sy_ir = torch.from_numpy(f['sy_ir'][:]).float()

    sx_ir = sx_ir.squeeze()
    sy_ir = sy_ir.squeeze()
    otg_coords = torch.stack([sx_ir, sy_ir], dim=1)

    return d_obs, otg_coords


def run_single_experiment(
    d_obs,
    otg_coords,
    grid_shape,
    L,
    inner_epochs,
    lr,
    beta,
    patch_size,
    shift,
    device,
    init_type,
    init_std_scale,
    save_dir,
    verbose=True
):
    """运行单个实验"""
    exp_name = f"L{L}_epochs{inner_epochs}"

    if verbose:
        print("=" * 80)
        print(f"Running Experiment: {exp_name}")
        print("=" * 80)
        print(f"  L (POCS iterations): {L}")
        print(f"  inner_epochs: {inner_epochs}")
        print(f"  lr: {lr}")
        print(f"  beta: {beta}")
        print(f"  patch_size: {patch_size}")
        print(f"  shift: {shift}")
        print(f"  device: {device}")
        print(f"  init_type: {init_type}")
        print(f"  init_std_scale: {init_std_scale}")

    # 运行 DLOTG 算法
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
        verbose=verbose,
        save_intermediate=False,  # 批量实验不保存中间结果
        intermediate_dir=None,
        init_type=init_type,
        init_std_scale=init_std_scale,
        save_frequency=10
    )

    # 保存结果
    save_path = os.path.join(save_dir, f"result_{exp_name}.mat")
    scipy.io.savemat(save_path, {
        'd_recon': d_reconstructed.cpu().numpy(),
        'L': L,
        'inner_epochs': inner_epochs,
        'lr': lr,
        'beta': beta,
        'patch_size': patch_size,
        'shift': shift
    })

    if verbose:
        print(f"\n[OK] Experiment {exp_name} completed!")
        print(f"  Result saved to: {save_path}")

    return d_reconstructed


def main():
    parser = argparse.ArgumentParser(description='Batch run DLOTG experiments')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the data file (default: data/curve_3D_irregular_clean_2.mat)')
    parser.add_argument('--device', type=str, default='cuda:2',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--grid-shape', type=int, nargs=3, default=[50, 50, 256],
                        help='Grid shape (default: 50 50 256)')
    parser.add_argument('--patch-size', type=int, nargs=3, default=[15, 15, 15],
                        help='Patch size (default: 15 15 15)')
    parser.add_argument('--shift', type=int, nargs=3, default=[5, 5, 5],
                        help='Shift step (default: 5 5 5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Beta (default: 0.01)')
    parser.add_argument('--init-type', type=str, default='gaussian',
                        choices=['zeros', 'gaussian'], help='Initialization type (default: gaussian)')
    parser.add_argument('--init-std-scale', type=float, default=0.1,
                        help='Initialization std scale (default: 0.1)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save results (default: data/experiments_results)')

    args = parser.parse_args()

    # 设置数据路径
    if args.data_path is None:
        data_path = os.path.join(project_root, 'data', 'curve_3D_irregular_clean_2.mat')
    else:
        data_path = args.data_path

    # 设置保存目录
    if args.save_dir is None:
        save_dir = os.path.join(project_root, 'data', 'experiments_results_L2_b0.5')
    else:
        save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    print("=" * 80)
    print("Batch DLOTG Experiments")
    print("=" * 80)
    print(f"\nLoading data from {data_path}...")

    d_obs, otg_coords = load_data(data_path, args.device)

    print(f"  d_obs shape: {d_obs.shape}")
    print(f"  otg_coords shape: {otg_coords.shape}")
    print(f"  Data range: [{d_obs.min():.6f}, {d_obs.max():.6f}]")

    # 定义实验配置
    grid_shape = tuple(args.grid_shape)
    patch_size = tuple(args.patch_size)
    shift = tuple(args.shift)

    # 定义不同的 (L, inner_epochs) 组合
    experiment_configs = [
        # (L, inner_epochs)
        (15, 10),
        (15, 15),
        (15, 20),
        (15, 25),
        (15, 30),
        (15, 35),
        (15, 40),
        (15, 45),
        (15, 50),
        (15, 55),
        (15, 60),
        (20, 10),
        (20, 15),
        (20, 20),
        (20, 25),
        (20, 30),
        (20, 35),
        (20, 40),
        (20, 45),
        (20, 50),
        (20, 55),
        (20, 60),
        (25, 10),
        (25, 15),
        (25, 20),
        (25, 25),
        (25, 30),
        (25, 35),
        (25, 40),
        (25, 45),
        (25, 50),
        (25, 55),
        (25, 60),
    ]

    print(f"\n" + "=" * 80)
    print(f"Experiment Configurations ({len(experiment_configs)} total)")
    print("=" * 80)

    for i, (L, inner_epochs) in enumerate(experiment_configs, 1):
        print(f"{i:2d}. L={L:2d}, inner_epochs={inner_epochs:2d}")

    print("=" * 80)

    # 运行所有实验
    print(f"\nStarting experiments...")
    print(f"Results will be saved to: {save_dir}")
    print("=" * 80 + "\n")

    results = []

    for i, (L, inner_epochs) in enumerate(experiment_configs, 1):
        print(f"\n[{i}/{len(experiment_configs)}] ", end="")

        try:
            d_recon = run_single_experiment(
                d_obs=d_obs,
                otg_coords=otg_coords,
                grid_shape=grid_shape,
                L=L,
                inner_epochs=inner_epochs,
                lr=args.lr,
                beta=args.beta,
                patch_size=patch_size,
                shift=shift,
                device=args.device,
                init_type=args.init_type,
                init_std_scale=args.init_std_scale,
                save_dir=save_dir,
                verbose=True
            )

            results.append({
                'L': L,
                'inner_epochs': inner_epochs,
                'success': True,
                'error': None
            })

        except Exception as e:
            print(f"\n[ERROR] Experiment L={L}, epochs={inner_epochs} failed!")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'L': L,
                'inner_epochs': inner_epochs,
                'success': False,
                'error': str(e)
            })

    # 打印总结
    print("\n" + "=" * 80)
    print("Experiment Summary")
    print("=" * 80)

    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"\nTotal experiments: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    if failed > 0:
        print(f"\nFailed experiments:")
        for r in results:
            if not r['success']:
                print(f"  - L={r['L']}, epochs={r['inner_epochs']}: {r['error']}")

    print("\n" + "=" * 80)
    print(f"All results saved to: {save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
