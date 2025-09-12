import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

# 设置样式
sns.set(style="whitegrid")
palette = sns.color_palette("tab10")

# 横坐标标签
x_labels = [1, 4, 16, 64, 256]
x = np.arange(len(x_labels))

# 数据集
data_sets = {
    "Num. Subchunk = 1": {
        'SCALE-CCL': [853.66, 1908.35, 6127.1, 23002.1, 90502.1],
        'TE-CCL': [815.3, 1752.8, 5627.1, 22502.1, 90002.1],
        'SPH': [1712.225, 4413.95, 16132.7, 63007.7, 250507.7],
        'NCCL': [1670.3625, 3938.9, 14251.4, 55501.4, 220501.4]
    },
    "Num. Subchunk = 2": {
        'SCALE-CCL': [795.07, 1673.98, 5313.2, 21250.7, 85000.7],
        'TE-CCL': [737.175, 1519.825, 5001.4, 18752.1, 75002.1],
        'SPH': [1477.15, 3206.625, 11566, 43753.5, 175003.5],
        'NCCL': [1533.64375, 3938.9, 14251.4, 55501.4, 220501.4]
    },
    "Num. Subchunk = 4": {
        'SCALE-CCL': [726.71, 1362.18, 5188.9, 20625.7, 82500.7],
        'TE-CCL': [687.646875, 1322.4125, 4720.85, 18752.8, 75002.8],
        'SPH': [1457.61875, 2855.0625, 9534.75, 35628.5, 142503.5],
        'NCCL': [1465.284375, 3938.9, 14251.4, 55501.4, 220501.4]
    }
}

# 图形设置
# Golden ratio width per subplot ≈ 1.618 * height; for 3 subplots, width ≈ 9.7*3=29, height=4.5
fig = plt.figure(figsize=(29, 4.5))
gs = GridSpec(1, 3, figure=fig, wspace=0.1)

bar_width = 0.18

for idx, (title, dataset) in enumerate(data_sets.items()):
    ax = fig.add_subplot(gs[0, idx])
    for i, alg in enumerate(dataset.keys()):
        total_width = bar_width * len(dataset)
        positions = x - total_width/2 + i * bar_width + bar_width/2
        ax.bar(positions, dataset[alg], width=bar_width, label=alg, color=palette[i], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{v}' for v in x_labels], fontsize=30)
    ax.set_title(title, fontsize=28, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Completion time (μs)', fontsize=24, fontweight='bold', labelpad=20)
    ax.set_xlabel('Chunk size (MB)', fontsize=24, fontweight='bold')
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(axis='y', linestyle='--', alpha=0.7, which='both')
    ax.tick_params(axis='y', labelsize=30)
    if idx == 0:
        ax.legend(fontsize=24, frameon=True, ncol=2, loc='upper left', columnspacing=0.5, handletextpad=0.4, labelspacing=0.2)
    else:
        ax.legend().remove()
    # Remove bar value labels for cleaner bars per requirements
    # for container in ax.containers:
    #     ax.bar_label(container, fontsize=12, padding=2, rotation=90)

sns.despine()
plt.subplots_adjust(bottom=0.25, wspace=0.25)  # 手动增加底部空间
output_path = "subchunk_com.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.01)
plt.show()
