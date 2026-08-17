import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# Exact paper typography and styling
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 150

def plot_multiple_bp_vs_osd(datasets, plot_title, save_filename):
    """
    Generates a log-log plot comparing pure BP vs BP-OSD across multiple datasets.
    
    datasets: list of dicts with keys 'label', 'filepath', and 'color'
    """
    fig, ax = plt.subplots(figsize=(7.5, 6)) # Made slightly wider to accommodate a larger legend
    
    global_min_val = float('inf')
    
    # Loop through all datasets and plot them on the same axis
    for data_info in datasets:
        filepath = data_info['filepath']
        code_label = data_info['label']
        color = data_info['color']
        
        if not os.path.exists(filepath):
            print(f"Warning: Could not find {filepath}. Skipping this dataset.")
            continue
            
        # Load the 3-column data: p_rates, wer_bp, wer_osd
        data = np.loadtxt(filepath)
        p_rates = data[:, 0]
        wer_bp = data[:, 1]
        wer_osd = data[:, 2]
        
        # Plot pure BP (Open Circles - markerfacecolor='none')
        ax.plot(p_rates, wer_bp, color=color, marker='o', linestyle='-', 
                linewidth=1, markersize=6, markerfacecolor='none', 
                label=f'{code_label}, BP')
                
        # Plot BP-OSD (Filled Circles)
        ax.plot(p_rates, wer_osd, color=color, marker='o', linestyle='-', 
                linewidth=1, markersize=6, 
                label=f'{code_label}, BP-OSD')

        # Track the lowest non-zero data point across ALL files for dynamic y-scaling
        valid_wer = np.concatenate([wer_bp[wer_bp > 0], wer_osd[wer_osd > 0]])
        if len(valid_wer) > 0:
            local_min = np.min(valid_wer)
            if local_min < global_min_val:
                global_min_val = local_min

    # Apply Log Scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set exact physical error rate limits from the paper image
    ax.set_xlim([1e-2, 2e-1])
    
    # Set y-axis bounds dynamically based on the global lowest non-zero data
    if global_min_val != float('inf'):
        ax.set_ylim([max(1e-8, global_min_val * 0.1), 1.2e0])
    else:
        ax.set_ylim([1e-8, 1.2e0]) # Fallback if all files are empty or zero

    # Add exact grid styling (dashed gray/lightgray)
    ax.grid(True, which='major', linestyle='--', color='gray', alpha=0.6)
    ax.grid(True, which='minor', linestyle='--', color='lightgray', alpha=0.4)

    # Style ticks (pointing inwards on all sides)
    ax.tick_params(which='both', direction='in', top=True, right=True, length=4)
    ax.tick_params(which='major', length=6)

    # Add Labels and Title
    ax.set_xlabel('Physical error rate')
    ax.set_ylabel('WER')
    ax.set_title(plot_title, pad=15, fontsize=14)

    # Add Legend (Black square box, loc='best' prevents covering data points)
    ax.legend(ncol=1, edgecolor='black', framealpha=1, fancybox=False, loc='best')

    # Save and display
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plot_path = f'results/{save_filename}.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved successfully to {plot_path}")
    plt.show()

if __name__ == "__main__":
    # Define the datasets you want to plot together
    # Ensure simulate.py has been executed for all of these to generate the data
    datasets_to_plot = [
        {
            'label': 'Regular Codes', 
            'filepath': 'results/tile_288_8_14_x_biased_comparison.txt', 
            'color': 'black'
        },
        {
            'label': 'Bias Tailored Codes', 
            'filepath': 'results/bt_tile_288_8_12_x_biased_comparison.txt', 
            'color': 'blue'
        }
        # {
        #     'label': 'X bias', 
        #     'filepath': 'results/tile_288_8_14_x_biased_comparison.txt', 
        #     'color': 'red'
        # }
    ]
    
    plot_multiple_bp_vs_osd(
        datasets=datasets_to_plot,
        plot_title='Comparison of [[288, 8, 14]] Tile Codes X bias',
        save_filename='BT vs Reg X Bias Comparison'
    )