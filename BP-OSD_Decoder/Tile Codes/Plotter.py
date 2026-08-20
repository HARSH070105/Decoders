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


def plot_multiple_bp_vs_osd(code_name):

    # ---------------------------------------------------------
    # Results folder
    # ---------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir = os.path.join(
        script_dir,
        f"Results_{code_name}"
    )

    # ---------------------------------------------------------
    # Automatically find the three result files
    # ---------------------------------------------------------
    datasets = [
        {
            'label': 'UnBiased Noise',
            'filepath': os.path.join(
                results_dir,
                f'{code_name}_depolarizing_comparison.txt'
            ),
            'color': 'black'
        },
        {
            'label': 'X-Biased Noise',
            'filepath': os.path.join(
                results_dir,
                f'{code_name}_pure_x_comparison.txt'
            ),
            'color': 'red'
        },
        {
            'label': 'Z-Biased Noise',
            'filepath': os.path.join(
                results_dir,
                f'{code_name}_pure_z_comparison.txt'
            ),
            'color': 'blue'
        }
    ]

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6))

    global_min_val = float('inf')

    # ---------------------------------------------------------
    # Plot datasets
    # ---------------------------------------------------------
    for data_info in datasets:

        filepath = data_info['filepath']
        code_label = data_info['label']
        color = data_info['color']

        if not os.path.exists(filepath):
            print(f"WARNING: Could not find {filepath}")
            continue

        data = np.loadtxt(filepath)

        p_rates = data[:, 0]
        wer_bp = data[:, 1]
        wer_osd = data[:, 2]

        # BP - open circles
        ax.plot(
            p_rates,
            wer_bp,
            color=color,
            marker='o',
            linestyle='-',
            linewidth=1,
            markersize=6,
            markerfacecolor='none',
            label=f'{code_label}, BP'
        )

        # BP-OSD - filled circles
        ax.plot(
            p_rates,
            wer_osd,
            color=color,
            marker='o',
            linestyle='-',
            linewidth=1,
            markersize=6,
            label=f'{code_label}, BP-OSD'
        )

        # Minimum non-zero WER
        valid_wer = np.concatenate([
            wer_bp[wer_bp > 0],
            wer_osd[wer_osd > 0]
        ])

        if len(valid_wer) > 0:
            global_min_val = min(
                global_min_val,
                np.min(valid_wer)
            )

    # ---------------------------------------------------------
    # Axes
    # ---------------------------------------------------------
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim([1e-2, 2e-1])

    if global_min_val != float('inf'):
        ax.set_ylim([
            max(1e-8, global_min_val * 0.1),
            1.2e0
        ])
    else:
        ax.set_ylim([1e-8, 1.2e0])

    # ---------------------------------------------------------
    # Grid
    # ---------------------------------------------------------
    ax.grid(
        True,
        which='major',
        linestyle='--',
        color='gray',
        alpha=0.6
    )

    ax.grid(
        True,
        which='minor',
        linestyle='--',
        color='lightgray',
        alpha=0.4
    )

    # ---------------------------------------------------------
    # Ticks
    # ---------------------------------------------------------
    ax.tick_params(
        which='both',
        direction='in',
        top=True,
        right=True,
        length=4
    )

    ax.tick_params(which='major', length=6)

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------
    ax.set_xlabel('Physical error rate')
    ax.set_ylabel('WER')

    ax.set_title(
        f'Comparison of {code_name} under Different Noise Models',
        pad=15,
        fontsize=14
    )

    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------
    ax.legend(
        ncol=1,
        edgecolor='black',
        framealpha=1,
        fancybox=False,
        loc='best'
    )

    # ---------------------------------------------------------
    # Save INSIDE Results_{code_name}
    # ---------------------------------------------------------
    plt.tight_layout()

    save_path = os.path.join(
        results_dir,
        f'{code_name}_comparison_quick.png'
    )

    plt.savefig(
        save_path,
        bbox_inches='tight'
    )

    print(f"Plot saved successfully to:")
    print(save_path)

    plt.show()


# =============================================================
# ONLY CHANGE THIS
# =============================================================

code_name = "tile_288_8_14"

plot_multiple_bp_vs_osd(code_name)