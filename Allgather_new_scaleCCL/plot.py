import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def visualize_csv_data(file_path):
    # 1. 读取数据
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'，请检查文件名或路径。")
        return

    # 2. 数据处理：将单位从 us 转换为 ms
    y_labels = df.columns[1:]
    df[y_labels] = df[y_labels] / 1000

    # 3. 提取 X 轴原始数据
    x_col_name = df.columns[0]
    x_values = df[x_col_name] # 这里保留原始的小数 (如 0.1, 0.2)
    
    # 4. 设置绘图参数
    x_indices = np.arange(len(x_values)) 
    num_bars = len(y_labels)
    width = 0.18 # 柱子宽度
    
    # 定义纹理列表
    patterns = ['/', '\\\\', 'xxx', '...', '---'] 
    # 获取颜色循环
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    # 存储自定义图例句柄
    legend_elements = []

    # 5. 循环绘制每一组数据
    for i, attribute in enumerate(y_labels):
        # 居中对齐逻辑
        offset = (i - (num_bars - 1) / 2) * width
        pos = x_indices + offset
        
        hatch_style = patterns[i % len(patterns)]
        current_color = colors[i % len(colors)]
        
        ax.bar(pos, 
               df[attribute], 
               width, 
               label=attribute, 
               hatch=hatch_style, 
               color=current_color, 
               edgecolor='black', 
               linewidth=0.8)
        
        legend_elements.append(Patch(facecolor=current_color, 
                                     edgecolor='black', 
                                     hatch=hatch_style, 
                                     label=attribute))

    # 6. 修饰图表
    # 严格按照您的要求设置 X 轴标签
    ax.set_xlabel('Original data ratio') 
    ax.set_ylabel('Cumulative time (ms)')
    ax.set_title('')
    
    # 7. 设置 X 轴刻度：直接使用原始数据（保留小数点）
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_values) 
    
    # 8. 优化图例
    ax.legend(handles=legend_elements, 
              loc='upper right', 
              frameon=True, 
              edgecolor='gray',
              handlelength=2.5, 
              handleheight=1.5)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # 9. 保存并显示
    plt.savefig('alpha_result.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    target_csv = "alpha_result.csv"
    visualize_csv_data(target_csv)