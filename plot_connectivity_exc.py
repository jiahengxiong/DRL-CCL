import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

# 设置样式
sns.set(style="whitegrid")
palette = sns.color_palette("deep")

# 横坐标标签
x_labels = [1, 4, 16, 64, 256]
x = np.arange(len(x_labels))

# 数据集
data_sets = {
    "Connectivity = 0.3": {
        'LTCCL': [0.038444, 0.037667, 0.037187, 0.037152, 0.037020],
        'TECCL': [2.38, 1.95, 1.83, 1.95, 1.95],
        'Shortest-path': [0.064272, 0.062085, 0.061631, 0.062381, 0.06288]
    },
    "Connectivity = 0.5": {
        'LTCCL': [0.043178, 0.054990, 0.045330, 0.045225, 0.044211],
        'TECCL': [1.94, 2.02, 1.83, 1.95, 1.95],
        'Shortest-path': [0.081406, 0.084858, 0.082436, 0.083185, 0.083972]
    },
    "Connectivity = 0.7": {
        'LTCCL': [0.041155, 0.040406, 0.040587, 0.040643, 0.040119],
        'TECCL': [2.78, 2.83, 2.49, 2.45, 2.43],
        'Shortest-path': [0.115097, 0.110642, 0.110831, 0.109418, 0.110384]
    }
}

# 图形设置
fig = plt.figure(figsize=(15, 2.5))
gs = GridSpec(1, 3, figure=fig, wspace=0.25)

bar_width = 0.25

for idx, (title, dataset) in enumerate(data_sets.items()):
    ax = fig.add_subplot(gs[0, idx])
    for i, alg in enumerate(dataset.keys()):
        positions = x + (i - 1) * (bar_width - 0.02)
        ax.bar(positions, dataset[alg], width=bar_width, label=alg, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels([f'{v}' for v in x_labels], fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Execution time (s)', fontsize=12)
    ax.set_xlabel('Chunk size (MB)', fontsize=12)
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.7, which='both')
    if idx == 0:
        ax.legend(fontsize=10, frameon=True, columnspacing=0.5)
    else:
        ax.legend().remove()

sns.despine()
plt.subplots_adjust(bottom=0.25, wspace=0.25)  # 手动增加底部空间
output_path = "subchunk_completion_time_connectivity.png"
plt.savefig(output_path, dpi=300)
plt.show()
