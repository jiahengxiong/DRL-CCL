import pandas as pd
import matplotlib.pyplot as plt
import os

# 使用指定的数据文件
csv_path = '/home/ubuntu/Education/results/2GPU_exp/latency_result_2880.csv'
df = pd.read_csv(csv_path)

plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 24
    })
plt.figure(figsize=(10, 6))

# Theory: 无标记实线，作为"底层"参考，不与 Our Simulator 的标记争夺视觉
# 注意：CSV 中的列名是 Latency_ms, SimAI, Theory
plt.plot(df['Latency_ms'], df['Theory'],
         label='Theory',
         marker='x',
         color='blue',
         linestyle='-',
         linewidth=2,
         markeredgewidth=1.5,
         markersize=6,
        )

# SimAi
plt.plot(df['Latency_ms'], df['SimAI'],
         label='SimAi',
         marker='s',
         markerfacecolor='none',
         markeredgecolor='red',
         markeredgewidth=1.5,
         color='red',
         linestyle='--',
         linewidth=2,
         markersize=6,
        )

# Our Simulator: 注释掉的部分
"""
plt.plot(df['Latency_ms'], df['CCL_Simulator'],
         label='Our Simulator',
         marker='o',
         markerfacecolor='none',
         markeredgecolor='green',
         markeredgewidth=1.5,
         color='green',
         linestyle=':',
         linewidth=2,
         markersize=10,
        )
"""

plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 16
    })

plt.xscale('log')
# 仅选择代表性的关键点作为刻度，避免重叠
key_ticks = [0.001, 0.01, 0.1, 0.5]
plt.xticks(key_ticks, [str(t) for t in key_ticks])
plt.xlabel('Latency (ms)')
plt.ylabel('Communication Time (us)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存结果
output_dir = '/home/ubuntu/Education/results/2GPU_exp'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'simai_latency_result_2880.png'))
# plt.show() # 如果在无 GUI 环境运行，通常注释掉 show
