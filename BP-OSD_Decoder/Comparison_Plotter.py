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

def plot_bt_vs_regular(code_suffix):
    # ---------------------------------------------------------
    # Directory Setup
    # ---------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    bt_dir = os.path.join(script_dir, "HaRT Codes", f"Results_BT_{code_suffix}")
    reg_dir = os.path.join(script_dir, "Tile Codes", f"Results_{code_suffix}")

    noise_models = [
        {'label': 'UnBiased Noise', 'suffix': 'depolarizing'},
        {'label': 'X-Biased Noise', 'suffix': 'pure_x'},
        {'label': 'Z-Biased Noise', 'suffix': 'pure_z'}
    ]

    # Create a 1x3 subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ---------------------------------------------------------
    # Plotting Loop
    # ---------------------------------------------------------
    for ax, noise in zip(axes, noise_models):
        method = noise['suffix']
        
        # Target exact filenames
        bt_file = os.path.join(bt_dir, f"BT_{code_suffix}_{method}_comparison.txt")
        reg_file = os.path.join(reg_dir, f"{code_suffix}_{method}_comparison.txt")
        
        # 1. Plot Regular Code
        if os.path.exists(reg_file):
            data_reg = np.loadtxt(reg_file)
            p_reg, bp_reg, osd_reg = data_reg[:, 0], data_reg[:, 1], data_reg[:, 2]
            
            # BP - open circles
            ax.plot(p_reg, bp_reg, color='black', marker='o', linestyle='-', 
                    linewidth=1, markersize=6, markerfacecolor='none', label='Reg BP')
            # BP-OSD - filled circles
            ax.plot(p_reg, osd_reg, color='black', marker='o', linestyle='-', 
                    linewidth=1, markersize=6, label='Reg BP-OSD')
        else:
            print(f"WARNING: Missing {reg_file}")

        # 2. Plot Bias Tailored Code
        if os.path.exists(bt_file):
            data_bt = np.loadtxt(bt_file)
            p_bt, bp_bt, osd_bt = data_bt[:, 0], data_bt[:, 1], data_bt[:, 2]
            
            # BP - open squares (to distinguish from regular codes)
            ax.plot(p_bt, bp_bt, color='red', marker='s', linestyle='--', 
                    linewidth=1, markersize=6, markerfacecolor='none', label='BT BP')
            # BP-OSD - filled squares
            ax.plot(p_bt, osd_bt, color='red', marker='s', linestyle='--', 
                    linewidth=1, markersize=6, label='BT BP-OSD')
        else:
            print(f"WARNING: Missing {bt_file}")

        # 3. Axis Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1e-2, 2e-1])
        ax.set_ylim([1e-8, 1.2e0]) 

        ax.grid(True, which='major', linestyle='--', color='gray', alpha=0.6)
        ax.grid(True, which='minor', linestyle='--', color='lightgray', alpha=0.4)
        ax.tick_params(which='both', direction='in', top=True, right=True, length=4)
        ax.tick_params(which='major', length=6)

        ax.set_xlabel('Physical error rate')
        if ax == axes[0]:
            ax.set_ylabel('WER')

        ax.set_title(f'{noise["label"]}', pad=15, fontsize=14)
        ax.legend(ncol=1, edgecolor='black', framealpha=1, fancybox=False, loc='best')

    # ---------------------------------------------------------
    # Save Output
    # ---------------------------------------------------------
    plt.tight_layout()
    save_path = os.path.join(script_dir, f"{code_suffix}_architecture_comparison.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved successfully to: {save_path}")
    plt.show()

# =============================================================
# Execution
# =============================================================
code_suffix = "tile_toric"  
plot_bt_vs_regular(code_suffix)