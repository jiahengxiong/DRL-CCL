import re
import pandas as pd
import matplotlib.pyplot as plt

# 改成你的第二个 txt 文件路径
file_path = "E:\MySpace\Experiment\DRL-CCL\Allgather_new_scaleCCL\prediction_error_analysis.txt"

# 解析每一行：
# 格式示例：0,(0, 2),0.463962,0.021181,0.442781
pattern = re.compile(
    r'^\s*(\d+),\((\d+)\s*,\s*(\d+)\),([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$'
)

rows = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("时间步"):
            continue

        m = pattern.match(line)
        if m:
            timestep = int(m.group(1))
            link_pair = f"({m.group(2)}, {m.group(3)})"
            pred = float(m.group(4))
            true = float(m.group(5))
            abs_err = float(m.group(6))

            rows.append([timestep, link_pair, pred, true, abs_err])

# 构造 DataFrame
df = pd.DataFrame(rows, columns=["时间步", "链路对", "预测值", "真实值", "绝对差值"])

# 只取前100步：0~99
df_100 = df[df["时间步"] < 100].copy()

# 聚合方式：sum / mean 都可以，这里默认用 sum
agg_method = "sum"

grouped = (
    df_100.groupby("时间步", as_index=False)
    .agg({
        "预测值": agg_method,
        "真实值": agg_method
    })
    .sort_values("时间步")
)
# 每个时间步聚合后除以 8，得到平均值
grouped["预测值"] = grouped["预测值"] / 8
grouped["真实值"] = grouped["真实值"] / 8

# 聚合后的预测值和真实值的绝对差
grouped["聚合绝对差值"] = (grouped["预测值"] - grouped["真实值"]).abs()

# ====== 画图：仿照图一风格 ======
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# 背景和网格风格
ax.set_facecolor("#ffffff")
ax.grid(True, which="major", linestyle="-", linewidth=0.8, color="#c2c1c1", alpha=0.8)

# 线条风格：参考图一的绿色虚线圆点风格
ax.plot(
    grouped["时间步"],
    grouped["聚合绝对差值"],
    linestyle="-",
    linewidth=2.2,
    color="green",
    label="Aggregated Absolute Error"
)

ax.set_ylim(-0.05,0.6)

ax.set_xlabel("Time Steps", fontsize=22, fontname="Times New Roman")
ax.set_ylabel("Average Traffic Prediction \n Error in WAN Paths", fontsize=22,fontname="Times New Roman")
ax.legend(
    loc="upper right",
    frameon=True,
    prop={"family": "Times New Roman", "size": 18},
    borderpad=0.8,
    labelspacing=0.6,
    handlelength=2.8
)
ax.tick_params(axis="both", labelsize=20)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname("Times New Roman")
plt.rcParams["font.family"] = "Times New Roman"

plt.tight_layout()

# 如果要保存图片，取消下一行注释
#plt.savefig("./agg_abs_error_first_100_steps.png", dpi=300, bbox_inches="tight")

plt.show()