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
        'LTCCL': [679.2, 2554.2, 10054.2, 40054.2, 160054.2],
        'TECCL': [679.2, 1927.8, 7552.8, 30052.8, 120052.8],
        'SPH': [1149.35, 4430.6, 17555.6, 70055.6, 280055.6]
    },
    "Connectivity = 0.5": {
        'LTCCL': [403.6625, 1458.35, 5677.1, 22552.1, 90052.1],
        'TECCL': [403.6625, 1408.35, 5627.1, 22502.1, 90052.1],
        'SPH': [1034.2625, 3963.95, 15682.7, 62557.7, 250057.7],
        'NCCL': [1670.3625, 3938.9, 14251.4, 55501.4, 220501.4]
    },
    "Connectivity = 0.7": {
        'LTCCL': [363.9, 1301.4, 5051.4, 20051.4, 80051.4],
        'TECCL': [325.5375, 1145.85, 4427.1, 17552.1, 70052.1],
        'SPH': [882.9125, 3343.85, 13187.6, 52562.6, 210062.6]
    }
}

# 图形设置
fig = plt.figure(figsize=(18, 5))
gs = GridSpec(1, 3, figure=fig, wspace=0.2)

bar_width = 0.2

for idx, (title, dataset) in enumerate(data_sets.items()):
    ax = fig.add_subplot(gs[0, idx])
    for i, alg in enumerate(dataset.keys()):
        positions = x + (i - 1) * (bar_width - 0.02)
        ax.bar(positions, dataset[alg], width=bar_width, label=alg, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels([f'{v}' for v in x_labels], fontsize=16)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_ylabel('Completion time (μs)', fontsize=18)
    ax.set_xlabel('Chunk size (MB)', fontsize=18)
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.7, which='both')
    ax.tick_params(axis='y', labelsize=16)
    if idx == 0:
        ax.legend(fontsize=14, frameon=True, columnspacing=0.5)
    else:
        ax.legend().remove()

sns.despine()
plt.subplots_adjust(bottom=0.25, wspace=0.2)  # 手动增加底部空间
output_path = "subchunk_completion_time_connectivity.png"
plt.savefig(output_path, dpi=300)
plt.show()
