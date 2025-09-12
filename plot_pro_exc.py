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

# 新数据集：不同传播时延（单位：秒）对应的结果
data_sets = {
    "WAN delay = 50 µs (10 km)": {
        'LTCCL': [0.043178, 0.044990, 0.045330, 0.045225, 0.044211],
        'TECCL': [1.94, 2.02, 1.83, 1.95, 1.95],
        'Shortest-path': [0.081406, 0.084858, 0.082436, 0.083185, 0.083972]
    },
    "WAN delay = 250 µs (50 km)": {
        'LTCCL': [0.041428, 0.040946, 0.040939, 0.040722, 0.041822],
        'TECCL': [4.01, 1.78, 2.06, 1.83, 1.94],
        'Shortest-path': [.081406, 0.084858, 0.082436, 0.083185, 0.083972]
    },
    "WAN delay = 500 µs (100 km)": {
        'LTCCL': [0.040813, 0.04008, 0.044292, 0.043763, 0.044935],
        'TECCL': [4.96, 2.07, 1.63, 1.91, 2.01],
        'Shortest-path': [0.082652, 0.081944, 0.083011, 0.084864, 0.084722]
    },
    "WAN delay = 750 µs (150 km)": {
        'LTCCL': [0.039974, 0.039161, 0.043624, 0.045191, 0.044869],
        'TECCL': [4.61, 2.34, 1.96, 1.82, 2.01],
        'Shortest-path': [0.082332, 0.081191, 0.080977, 0.083076, 0.084874]
    },
    "WAN delay = 1000 µs (200 km)": {
        'LTCCL': [0.040893, 0.041644, 0.045885, 0.046102, 0.044867],
        'TECCL': [6.17, 4.07, 1.77, 2.05, 1.86],
        'Shortest-path': [0.084242, 0.088829, 0.08327, 0.085903, 0.086065]
    }
}

# 图形设置
fig = plt.figure(figsize=(25, 3))
gs = GridSpec(1, 5, figure=fig, wspace=0.25)

bar_width = 0.25

for idx, (title, dataset) in enumerate(data_sets.items()):
    ax = fig.add_subplot(gs[0, idx])
    for i, alg in enumerate(dataset.keys()):
        positions = x + (i - 1) * (bar_width - 0.02)
        ax.bar(positions, dataset[alg], width=bar_width, label=alg, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels([f'{v}' for v in x_labels], fontsize=20)
    ax.set_title(title, fontsize=20, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Execution time (s)', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel('Chunk size (MB)', fontsize=20)
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.7, which='both')
    if idx == 0:
        ax.legend(fontsize=15, frameon=True, columnspacing=0.5, ncol = 1, loc='upper right')
    else:
        ax.legend().remove()

sns.despine()
plt.subplots_adjust(bottom=0.25, wspace=0.4)  # 手动增加底部空间
output_path = "subchunk_completion_time_connectivity.png"
plt.savefig(output_path, dpi=300)
plt.show()