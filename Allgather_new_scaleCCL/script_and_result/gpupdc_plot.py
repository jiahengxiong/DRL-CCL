import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def visualize_gpu_data(file_path):
    # 1. 读取数据
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'，请检查文件名或路径。")
        return

    # 2. 数据处理：保持原始单位 s
    y_labels = df.columns[1:]

    # 3. 提取 X 轴原始数据
    x_col_name = df.columns[0]
    x_values = df[x_col_name]
    
    # 4. 设置绘图参数
    x_indices = np.arange(len(x_values)) 
    num_bars = len(y_labels)
    width = 0.18 
    
    patterns = ['/', '\\\\', 'xxx', '...', '---'] 
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 24
    })
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    legend_elements = []

    # 5. 循环绘制每一组数据
    for i, attribute in enumerate(y_labels):
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
    ax.set_xlabel(x_col_name)
    ax.set_ylabel('Cumulative time (s)') 
    ax.set_title('')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 16
    })
    # --- 新增：设置纵坐标范围 ---
    ax.set_ylim(0, 600) 
    
    # 7. 设置 X 轴刻度
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
    plt.savefig('gpu_result.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 使用 r 前缀防止路径转义
    target_csv = r"script_and_result\gpu_result_linux.csv"
    visualize_gpu_data(target_csv)