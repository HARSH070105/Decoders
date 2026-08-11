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

def plot_bp_vs_osd(code_label, filepath, color):
    """Generates the log-log plot comparing pure BP vs BP-OSD."""
    # Load the 3-column data: p_rates, wer_bp, wer_osd
    data = np.loadtxt(filepath)
    p_rates = data[:, 0]
    wer_bp = data[:, 1]
    wer_osd = data[:, 2]
    
    fig, ax = plt.subplots(figsize=(6.5, 6))
    
    # Plot pure BP (Open Circles - markerfacecolor='none')
    ax.plot(p_rates, wer_bp, color=color, marker='o', linestyle='-', 
            linewidth=1, markersize=6, markerfacecolor='none', 
            label=f'{code_label}, BP')
            
    # Plot BP-OSD (Filled Circles)
    ax.plot(p_rates, wer_osd, color=color, marker='o', linestyle='-', 
            linewidth=1, markersize=6, 
            label=f'{code_label}, BP-OSD')

    # Apply Log Scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set exact physical error rate limits from the paper image
    ax.set_xlim([1e-2, 2e-1])
    
    # Set y-axis bounds dynamically based on lowest non-zero data
    min_val = min(np.min(wer_bp[wer_bp>0]), np.min(wer_osd[wer_osd>0]))
    ax.set_ylim([max(1e-8, min_val * 0.1), 1.2e0])

    # Add exact grid styling (dashed gray/lightgray)
    ax.grid(True, which='major', linestyle='--', color='gray', alpha=0.6)
    ax.grid(True, which='minor', linestyle='--', color='lightgray', alpha=0.4)

    # Style ticks (pointing inwards on all sides)
    ax.tick_params(which='both', direction='in', top=True, right=True, length=4)
    ax.tick_params(which='major', length=6)

    # Add Labels and Title
    ax.set_xlabel('Physical error rate')
    ax.set_ylabel('WER')
    ax.set_title(f'The effect of OSD-0 on {code_label}', pad=15, fontsize=14)

    # Add Legend (Black square box)
    ax.legend(ncol=1, edgecolor='black', framealpha=1, fancybox=False, loc='upper left')

    # Save and display
    plt.tight_layout()
    plot_path = f'results/{code_label.replace(" ", "_").replace("[", "").replace("]", "").replace(",", "")}_bp_vs_osd.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved successfully to {plot_path}")
    plt.show()

if __name__ == "__main__":
    # Ensure simulate.py has been executed to generate the data
    file_path = "results/gb_254_28_comparison.txt"
    
    if os.path.exists(file_path):
        # Using black to perfectly replicate the A1 code style from the image
        plot_bp_vs_osd(code_label="GB [254, 28]", filepath=file_path, color='red')
    else:
        print(f"Error: Could not find {file_path}.")
        print("Please run simulate.py first to generate the required Monte Carlo data.")
