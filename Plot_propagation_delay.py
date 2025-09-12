import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Seaborn 风格
sns.set(style="whitegrid", font_scale=1.3, palette="deep")

# 数据
TECCL_summary = {
    10: 1.97, 200: 6.33, 400: 8.88, 600: 11.62, 800: 12.83,
    1000: 25.23, 1200: 22.15, 1400: 30.38, 1600: 38.46,
    1800: 42.02, 2000: 49.36
}
LTCCL_summary = {
    10: 0.044, 200: 0.041, 400: 0.043, 600: 0.049, 800: 0.041,
    1000: 0.039, 1200: 0.039, 1400: 0.039, 1600: 0.039,
    1800: 0.039, 2000: 0.040
}

distances = list(TECCL_summary.keys())
teccl_values = [TECCL_summary[d] for d in distances]
ltccl_values = [LTCCL_summary[d] for d in distances]

# 绘图
plt.figure(figsize=(12, 7))
sns.lineplot(x=distances, y=teccl_values, marker='o', label='TECCL', linewidth=3)
sns.lineplot(x=distances, y=ltccl_values, marker='s', label='LTCCL', linewidth=3)

plt.title('Execution Time vs Distance', fontsize=18, weight='bold')
plt.xlabel('Distance (km)', fontsize=15)
plt.ylabel('Execution Time (s)', fontsize=15)
plt.legend(title='Algorithm', fontsize=13)
plt.tight_layout()
plt.show()