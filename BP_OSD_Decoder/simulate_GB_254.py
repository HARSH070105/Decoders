import numpy as np
import os
from tqdm import tqdm

from logicals import build_canonical_logicals
from channel import generate_depolarizing_error
from decoder import setup_all_decoders
from failure import check_logical_failure

def load_code(filepath):
    data = np.load(filepath)
    import scipy.sparse as sp
    HX = sp.csr_matrix((data['HX_data'], data['HX_indices'], data['HX_indptr']), shape=data['HX_shape']).toarray()
    HZ = sp.csr_matrix((data['HZ_data'], data['HZ_indices'], data['HZ_indptr']), shape=data['HZ_shape']).toarray()
    return HX, HZ

def run_monte_carlo(filepath, error_rates, max_trials):
    HX, HZ = load_code(filepath)
    n = HX.shape[1]
    
    print(f"Loading {filepath} and computing Canonical Logicals...")
    LX, LZ = build_canonical_logicals(HX, HZ)
    print(f"Logicals generated: k = {LX.shape[0]}")
    
    results_bp = []
    results_osd = []
    
    for p in error_rates:
        bp_x, bp_z, osd_x, osd_z = setup_all_decoders(HX, HZ, 2*p/3) #needed this adjustment as the p given to decoder is also scaled (p/3 for X and Z and P/3 added due to Y)
        fails_bp = 0
        fails_osd = 0
        
        pbar = tqdm(range(max_trials), desc=f"p={p:.4f}", unit="trial")
        
        for trial in pbar:
            e_x, e_z = generate_depolarizing_error(n, p)
            s_x = (HZ @ e_x) % 2
            s_z = (HX @ e_z) % 2
            
            # Pure BP
            gx_bp = bp_x.decode(s_x)
            gz_bp = bp_z.decode(s_z)
            if check_logical_failure(e_x, e_z, gx_bp, gz_bp, HX, HZ, LX, LZ):
                fails_bp += 1
                
            # BP-OSD
            gx_osd = osd_x.decode(s_x)
            gz_osd = osd_z.decode(s_z)
            if check_logical_failure(e_x, e_z, gx_osd, gz_osd, HX, HZ, LX, LZ):
                fails_osd += 1
                
            if trial % 100 == 0:
                pbar.set_postfix({"BP": fails_bp, "OSD": fails_osd})
                
        pbar.close()
                
        wer_bp = fails_bp / max_trials
        wer_osd = fails_osd / max_trials
        results_bp.append(wer_bp)
        results_osd.append(wer_osd)
        print(f"FINAL p = {p:.4f} | WER BP = {wer_bp:.6f} | WER OSD = {wer_osd:.6f}\n")
        
    return results_bp, results_osd


if __name__ == "__main__":
    error_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13]
    trials = 50000000
    
    code_name = "gb_254_28"
    print(f"--- Simulating Code: {code_name} ---")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    wer_bp, wer_osd = run_monte_carlo(os.path.join(script_dir, f"codes/{code_name}.npz"), error_rates, trials)
    
    # Save a 3-column file: p, wer_bp, wer_osd
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    np.savetxt(os.path.join(results_dir, f"{code_name}_comparison.txt"), np.column_stack((error_rates, wer_bp, wer_osd)))