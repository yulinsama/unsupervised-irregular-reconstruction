# Unsupervised irregular seismic reconstruction

基于 DLOTG 的非规则网格地震数据无监督重建与去噪研究项目。

当前仓库保存已有实现与需求基线；算法改进及效果验证尚未完成。完整需求、数据约定、研究阶段和待定实验细则见 [proposal.md](proposal.md)。

## 代码入口

- `scripts/main_dlotg_yingjie.py`：当前主要研究入口。
- `src/core/seismic_operator.py`：双线性正演与伴随算子。
- `src/data/patching.py`：三维切块与重组。
- `src/models/dip_network.py`：现有全连接网络及注意力模块。

其他主脚本为已有版本；`scripts/run_experiments.py` 当前调用 `main_dlotg.py`，不应与主要入口混为同一基线。部分配置及评估模块为空，尚不可视为已完成工具。

## 数据与训练约束

首批本地输入为 `data/linear_3D_irregular.mat` 和 `data/curve_3D_irregular.mat`，其字段与旧入口不同，尚需适配。两者目标网格为 `128×128×256`，各有 8192 条非规则观测道。

本期逐数据集从头拟合，不使用预训练。完整参考 `d_regular` 仅用于最终评价，不参与训练或自动选参。研究先使用无噪声数据，再逐步加入噪声。

大型数据、训练输出、权重及缓存通过 `.gitignore` 排除，继续留在本地。仓库不包含输入数据或论文 PDF。

## 依赖与参考

现有依赖列表见 [requirements.txt](requirements.txt)。本次仅同步代码与文档，未验证完整训练环境及 GPU 兼容性。

参考论文：Li、Saad、Chen，*Unsupervised deep learning for off-the-grid seismic reconstruction and denoising*，[DOI: 10.1190/geo2024-0189.1](https://doi.org/10.1190/geo2024-0189.1)。
