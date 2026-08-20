import numpy as np
import os
from tqdm import tqdm

from Logicals import build_canonical_logicals
from Channel import generate_error
from Decoders import setup_all_decoders
from Failure import check_logical_failure

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
    
    noise_models = ['depolarizing', 'pure_x', 'pure_z']
    all_results = {}
    
    for model in noise_models:
        print(f"\n--- Running {model.upper()} noise model ---")
        results_bp = []
        results_osd = []
        
        for p in error_rates:
            bp_x, bp_z, osd_x, osd_z = setup_all_decoders(HX, HZ, p, noise_model=model)
            fails_bp = 0
            fails_osd = 0
            
            pbar = tqdm(range(max_trials), desc=f"{model} | p={p:.4f}", unit="trial")
            
            for trial in pbar:
                e_x, e_z = generate_error(n, p, noise_model=model)
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
            print(f"FINAL {model} p = {p:.4f} | WER BP = {wer_bp:.6f} | WER OSD = {wer_osd:.6f}")
            
        all_results[model] = (results_bp, results_osd)
        
    return all_results

if __name__ == "__main__":
    error_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    trials = 50000
    
    code_name = "tile_288_8_14" 
    print(f"--- Simulating Code: {code_name} ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run the consolidated Monte Carlo loop
    results_dict = run_monte_carlo(os.path.join(script_dir, f"Codes/{code_name}.npz"), error_rates, trials)
    
    # Save a 3-column file for each noise model, including code dimensions in the folder name
    results_dir = os.path.join(script_dir, f"Results_{code_name}")
    os.makedirs(results_dir, exist_ok=True)
    
    for model, (wer_bp, wer_osd) in results_dict.items():
        save_path = os.path.join(results_dir, f"{code_name}_{model}_comparison.txt")
        np.savetxt(save_path, np.column_stack((error_rates, wer_bp, wer_osd)))
        print(f"Saved {model} results to {save_path}")