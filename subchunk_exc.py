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
    "Num. Subchunk = 1": {
        'LTCCL': [0.040813, 0.04008, 0.044292, 0.043763, 0.044935],
        'TECCL':  [4.96, 2.07, 1.63, 1.91, 2.01],
        'Shortest-path': [0.082652, 0.081944, 0.083011, 0.084864, 0.084722]
    },
    "Num. Subchunk = 2": {
        'LTCCL': [0.087601, 0.097403, 0.092261, 0.086734, 0.087733],
        'TECCL': [31.73, 18.48, 15.27, 17.75, 17.25],
        'Shortest-path': [0.185245, 0.176668, 0.173668, 0.17227, 0.173708]
    },
    "Num. Subchunk = 4": {
        'LTCCL': [0.224312, 0.264627, 0.265771, 0.225817, 0.225914],
        'TECCL': [700.65, 529.96, 549.49, 760.57, 757.29],
        'Shortest-path': [0.392255, 0.396197, 0.356457, 0.329598, 0.32908]
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
