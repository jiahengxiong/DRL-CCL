import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

# 设置样式
sns.set(style="whitegrid")
palette = sns.color_palette("tab10")

# 横坐标标签
x_labels = [1, 4, 16, 64, 256]
x = np.arange(len(x_labels))

# 新数据集：不同传播时延（单位：秒）对应的结果
data_sets = {
    "WAN delay = 50 µs (10 km)": {
        'SCALE-CCL': [403.6625, 1458.35, 5677.1, 22552.1, 90052.1],
        'TE-CCL': [403.6625, 1408.35, 5627.1, 22502.1, 90052.1],
        'SPH': [1034.2625, 3963.95, 15682.7, 62557.7, 250057.7],
        'NCCL': [910.775, 3488.9, 13801.4, 55051.4, 220051.4]
    },
    "WAN delay = 500 µs (100 km)": {
        'SCALE-CCL': [853.66, 1908.35, 6127.1, 23002.1, 90502.1],
        'TE-CCL': [815.3, 1752.8, 5627.1, 22502.1, 90002.1],
        'SPH': [1712.225, 4413.95, 16132.7, 63007.7, 250507.7],
        'NCCL': [1670.3625, 3938.9, 14251.4, 55501.4, 220501.4]
    },
    "WAN delay = 1000 µs (200 km)": {
        'SCALE-CCL': [1353.6625, 2408.35, 6627.1, 23502.1, 91002.1],
        'TE-CCL': [1315.3, 2252.8, 6627.1, 22502.1, 90002.1],
        'SPH': [2712.225, 4913.95, 16632.7, 63507.7, 251007.7],
        'NCCL': [2670.3625, 4662.55, 14751.4, 56001.4, 221001.4]
    }
}

# 图形设置
fig = plt.figure(figsize=(30, 6), constrained_layout=True)
gs = fig.add_gridspec(1, 3)

bar_width = 0.18

for idx, (title, dataset) in enumerate(data_sets.items()):
    ax = fig.add_subplot(gs[0, idx])
    keys = list(dataset.keys())
    if 'NCCL' in keys:
        keys.remove('NCCL')
        keys.insert(2, 'NCCL')
    for i, alg in enumerate(keys):
        positions = x + (i - 1) * (bar_width - 0.04)
        ax.bar(positions, dataset[alg], width=bar_width, label=alg, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels([f'{v}' for v in x_labels], fontsize=36)
    ax.set_title(title, fontsize=36, fontweight='bold')
    ax.set_ylabel('Completion time (μs)', fontsize=30, fontweight='bold')
    ax.tick_params(axis='y', labelsize=36)
    ax.set_xlabel('Chunk size (MB)', fontsize=30, fontweight='bold', labelpad=2)
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.7, which='both')
    if idx == 0:
        ax.legend(fontsize=26, frameon=True, columnspacing=0.3, handletextpad=0.2, labelspacing=0.2, ncol=2, loc='upper left', bbox_to_anchor=(-0.02,1.05))
    else:
        ax.legend().remove()

sns.despine()
output_path = "pro_copletion.jpg"
plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.04)
plt.show()