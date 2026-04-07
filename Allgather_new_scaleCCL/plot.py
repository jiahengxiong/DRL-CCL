import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def visualize_csv_data(df, output_path):
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 22,
            "axes.labelsize": 24,
            "axes.titlesize": 24,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 20,
        }
    )
    y_labels = df.columns[1:]
    df[y_labels] = df[y_labels] / 1000

    x_col_name = df.columns[0]
    x_values = df[x_col_name].tolist()
    
    x_indices = np.arange(len(x_values))
    num_bars = len(y_labels)
    width = 0.18
    
    patterns = ['/', '\\\\', 'xxx', '...', '---']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    legend_elements = []

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

    ax.set_xlabel('Propagation delay (us)')
    ax.set_ylabel('Cumulative time (ms)')
    ax.set_title('')
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_values) 
    
    ax.legend(handles=legend_elements, 
              loc='upper left', 
              frameon=True, 
              edgecolor='gray',
              handlelength=2.5, 
              handleheight=1.5)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.savefig(output_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "Propagation_delay": ["100us", "150us", "200us", "250us"],
            "Baseline": [694145.95, 717635.23, 739242.32, 761104.11],
            "Slow Pace": [458308.54, 481520.83, 504886.59, 528358.15],
            "Fast pace": [273771.83, 298934.33, 320728.89, 344628.08],
            "Fast + Slow Pace": [169292.28, 193892.28, 220728.89, 242839.16],
        }
    )
    visualize_csv_data(df, output_path="propagation_delay_result.png")
