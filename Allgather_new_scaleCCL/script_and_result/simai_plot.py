import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./script_and_result/low_latency.csv')
df = df.drop(index=0).reset_index(drop=True)
df_high = pd.read_csv('./script_and_result/high_latency.csv')
df = pd.concat([df, df_high], ignore_index=True)

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(10, 6))

# Theory: 无标记实线，作为"底层"参考，不与 Our Simulator 的标记争夺视觉
plt.plot(df['Latency'], df['Theory'],
         label='Theory',
         marker='x',              # 去掉标记，避免与 Our Simulator 重叠
         color='blue',
         linestyle='-',
         linewidth=2,
         markeredgewidth=1.5,
         markersize=6,
        )

# SimAi
plt.plot(df['Latency'], df['SimAi'],
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

# Our Simulator: 只有标记、无连线，"浮"在 Theory 实线上
# 空心大标记，可以透过看到下面的蓝线，同时自身清晰可辨
plt.plot(df['Latency'], df['CCL_Simulator'],
         label='Our Simulator',
         marker='o',
         markerfacecolor='none',
         markeredgecolor='green',
         markeredgewidth=1.5,
         color='green',
         linestyle=':',           # 去掉连线，线由 Theory 代替
         linewidth=2,
         markersize=10,
        )

plt.xscale('log')
plt.xlabel('Latency (ms)')
plt.ylabel('Communication Time (us)')
plt.legend()
plt.grid(True)
plt.savefig('script_and_result/simai_latency_result.png')
plt.show()