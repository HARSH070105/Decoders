"""
Simulate_BT.py — Monte Carlo simulation for bias-tailored tile codes.

Mirrors the structure of Simulate.py exactly so the two are easy to compare.
The only differences are:
  - load_bt_code() also reads num_horizontal_qubits from the .npz
  - errors are drawn from Channel_BT.generate_bt_error() (sector-2 swap)
  - decoders come from Decoders_BT.setup_bt_decoders() (per-qubit priors)
  - OSD path applies the channel update (Decoders_BT.channel_update_x_to_z)
  - Failure.check_logical_failure() and Logicals.build_canonical_logicals()
    are reused unchanged

Usage
-----
Point code_name at an .npz built by Build_Tile_Codes.build_and_save()
(which now saves num_horizontal_qubits).  Run this file directly or
import run_bt_monte_carlo() for use in notebooks.
"""

import numpy as np
import os
from tqdm import tqdm
import scipy.sparse as sp

from Logicals import build_canonical_logicals
from Channel_BT import generate_bt_error
from Decoders_BT import setup_bt_decoders, channel_update_x_to_z
from Failure import check_logical_failure


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_bt_code(filepath: str):
    """
    Load a tile code .npz and return (HX, HZ, n_h).

    n_h is num_horizontal_qubits — the index where sector-2 begins.
    Build_Tile_Codes.build_and_save() always writes this field now.
    """
    data = np.load(filepath)
    HX = sp.csr_matrix(
        (data['HX_data'], data['HX_indices'], data['HX_indptr']),
        shape=data['HX_shape'],
    ).toarray()
    HZ = sp.csr_matrix(
        (data['HZ_data'], data['HZ_indices'], data['HZ_indptr']),
        shape=data['HZ_shape'],
    ).toarray()
    n_h = int(data['num_horizontal_qubits'])
    return HX, HZ, n_h


# ---------------------------------------------------------------------------
# Monte Carlo loop
# ---------------------------------------------------------------------------

def run_bt_monte_carlo(filepath: str, error_rates, max_trials: int):
    """
    Run bias-tailored Monte Carlo decoding over depolarizing, pure_x, pure_z.

    Parameters
    ----------
    filepath    : path to .npz produced by Build_Tile_Codes.build_and_save()
    error_rates : iterable of physical error rates
    max_trials  : number of trials per (model, p) combination

    Returns
    -------
    dict mapping noise_model -> (list_wer_bp, list_wer_osd)
    """
    HX, HZ, n_h = load_bt_code(filepath)
    n = HX.shape[1]

    print(f"Loaded: {filepath}")
    print(f"  n={n}, sector-1={n_h}, sector-2={n - n_h}")
    print("Computing canonical logicals ...")
    LX, LZ = build_canonical_logicals(HX, HZ)
    k = LX.shape[0]
    print(f"  k={k} logical qubits\n")

    noise_models = ['depolarizing', 'pure_x', 'pure_z']
    all_results = {}

    for model in noise_models:
        print(f"--- BT {model.upper()} ---")
        results_bp  = []
        results_osd = []

        for p in error_rates:
            bp_x, bp_z, osd_x, osd_z = setup_bt_decoders(HX, HZ, p, n_h,
                                                           noise_model=model)
            fails_bp  = 0
            fails_osd = 0

            pbar = tqdm(range(max_trials),
                        desc=f"BT {model} | p={p:.4f}", unit="trial")

            for trial in pbar:
                # ---- error (sector-2 swap applied) ----
                e_x, e_z = generate_bt_error(n, p, n_h, noise_model=model)

                # ---- syndromes (original CSS matrices, unchanged) ----
                s_x = (HZ @ e_x) % 2
                s_z = (HX @ e_z) % 2

                # ---- pure BP ----
                gx_bp = bp_x.decode(s_x)
                gz_bp = bp_z.decode(s_z)
                if check_logical_failure(e_x, e_z, gx_bp, gz_bp,
                                         HX, HZ, LX, LZ):
                    fails_bp += 1

                # ---- BP-OSD with channel update ----
                gx_osd = osd_x.decode(s_x)
                # Update Z-priors based on X-decoder output (Algorithm 2)
                pz_updated = channel_update_x_to_z(gx_osd, p, noise_model=model)
                osd_z.update_channel_probs(pz_updated)
                gz_osd = osd_z.decode(s_z)
                if check_logical_failure(e_x, e_z, gx_osd, gz_osd,
                                         HX, HZ, LX, LZ):
                    fails_osd += 1

                if trial % 100 == 0:
                    pbar.set_postfix({"BP": fails_bp, "OSD": fails_osd})

            pbar.close()

            wer_bp  = fails_bp  / max_trials
            wer_osd = fails_osd / max_trials
            results_bp.append(wer_bp)
            results_osd.append(wer_osd)
            print(f"  p={p:.4f} | WER BP={wer_bp:.6f} | WER OSD={wer_osd:.6f}")

        all_results[model] = (results_bp, results_osd)
        print()

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    error_rates = [0.01, 0.02, 0.03, 0.04, 0.05,
                   0.06, 0.07, 0.08, 0.09, 0.10,
                   0.11, 0.12, 0.13, 0.14, 0.15]
    trials = 50000

    code_name = "tile_288_8_14"      # same .npz as used in Simulate.py
    print(f"=== Bias-tailored simulation: {code_name} ===\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path   = os.path.join(script_dir, "Codes", f"{code_name}.npz")

    results_dict = run_bt_monte_carlo(npz_path, error_rates, trials)

    results_dir = os.path.join(script_dir, f"Results_BT_{code_name}")
    os.makedirs(results_dir, exist_ok=True)

    for model, (wer_bp, wer_osd) in results_dict.items():
        save_path = os.path.join(results_dir,
                                 f"{code_name}_BT_{model}_comparison.txt")
        np.savetxt(save_path,
                   np.column_stack((error_rates, wer_bp, wer_osd)),
                   header="p  wer_bp  wer_osd", fmt="%.6f")
        print(f"Saved: {save_path}")
