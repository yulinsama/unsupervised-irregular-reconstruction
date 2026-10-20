# import h5py
# import numpy as np
# import matplotlib.pyplot as plt

# mat_path = '/home/yuhao/seismic-dlotg/data/reconstructed_result.mat'

# # 使用 h5py 读取 v7.3 格式
# with h5py.File(mat_path, 'r') as f:
#     # 打印文件中所有的 key，确认是不是 'd_recon'
#     print("Keys in mat file:", list(f.keys())) 
    
#     # 读取数据，并转置回原来的维度顺序 [Z, Y, X] 
#     d_recon = np.array(f['d_recon']).transpose()

# print(f"数据加载成功，矩阵形状为: {d_recon.shape}")

# # ------------ 后面继续接之前的 Matplotlib 或 PyVista 可视化代码 ------------
# class IndexTracker:
#     def __init__(self, ax, X):
#         self.ax = ax
#         self.X = X
#         self.ind = X.shape[0] // 2
#         self.im = ax.imshow(self.X[self.ind, :, :], cmap='seismic', 
#                             vmin=np.min(X)*0.8, vmax=np.max(X)*0.8)
#         self.update()

#     def on_scroll(self, event):
#         if event.button == 'up':
#             self.ind = min(self.ind + 1, self.X.shape[0] - 1)
#         else:
#             self.ind = max(self.ind - 1, 0)
#         self.update()

#     def update(self):
#         self.im.set_data(self.X[self.ind, :, :])
#         self.ax.set_title(f'Slice {self.ind}/{self.X.shape[0]-1}')
#         self.im.axes.figure.canvas.draw()

# fig, ax = plt.subplots(figsize=(8, 6))
# tracker = IndexTracker(ax, d_recon)
# fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
# fig.colorbar(tracker.im, ax=ax)
# plt.show()

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

mat_path = '/home/yuhao/seismic-dlotg/data/reconstructed_result.mat'
# mat_path = '/home/yuhao/seismic-dlotg/data/curve_3D_irregular_clean_2.mat'
data = scipy.io.loadmat(mat_path)['d_recon']
Nx, Ny, Nt = data.shape

# 强制使用无界面后端
plt.switch_backend('Agg')

print("正在生成抽样时间切片拼图...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# 均匀抽样 8 个时间点
sample_indices = np.linspace(0, Nt - 1, 8, dtype=int)
v_limit = np.max(np.abs(data)) * 0.7

for i, t_idx in enumerate(sample_indices):
    im = axes[i].imshow(data[:, :, t_idx].T, cmap='seismic', vmin=-v_limit, vmax=v_limit, aspect='auto')
    axes[i].set_title(f'Time Step: {t_idx}')
    axes[i].set_xlabel('Nx')
    axes[i].set_ylabel('Ny')

plt.tight_layout()
output_path = '/home/yuhao/seismic-dlotg/reconstruction_slices.png'
plt.savefig(output_path, dpi=200)
print(f"📸 拼图已成功保存至: {output_path}，你可以通过本地查看该图片！")