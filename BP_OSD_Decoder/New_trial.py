import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy import sparse

# Import matrices (Adjust paths to match your directory structure)
from GHGP.ghgp_882_24 import HX as HX_B1, HZ as HZ_B1
from GB.gb_254_28 import HX as HX_A1, HZ as HZ_A1
# --- Configuration ---
p_list = np.linspace(0.01, 0.12, 8)
max_iter = 32
nms_factor = 0.625  # Normalized Min-Sum factor

def generate_pauli_error(n, p):
    r = np.random.rand(n)
    X, Z = np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if r[i] < p:
            t = np.random.randint(3)
            if t == 0: X[i] = 1        
            elif t == 1: Z[i] = 1      
            else: X[i], Z[i] = 1, 1    
    return X, Z

# --- Core Solvers ---
def gf2_solve(H, s):
    """Efficient Gaussian elimination over GF(2)."""
    m, n = H.shape
    H = H.copy()
    s = s.copy()
    pivots = []
    row = 0
    for col in range(n):
        if row >= m: break
        # Fast pivot search in GF(2)
        pivot = np.argmax(H[row:, col]) + row
        if H[pivot, col] == 0: continue
        
        # Swap rows if necessary
        if pivot != row:
            H[[row, pivot]] = H[[pivot, row]]
            s[[row, pivot]] = s[[pivot, row]]
            
        # Eliminate
        for r in range(m):
            if r != row and H[r, col]:
                H[r] ^= H[row]
                s[r] ^= s[row]
        
        pivots.append(col)
        row += 1
    return pivots, s[:row]

def osd0_decode(H, syndrome, llr, hard_decisions):
    """OSD-0 utilizing the residual syndrome logic."""
    n = H.shape[1]
    
    # Calculate unsatisfied parity checks (residual syndrome)
    s_res = (syndrome ^ (H @ hard_decisions)) % 2
    if not np.any(s_res):
        return hard_decisions
        
    # Sort by reliability (magnitudes near 0 are least reliable)
    perm = np.argsort(np.abs(llr))
    Hs = H.toarray()[:, perm] if sparse.issparse(H) else H[:, perm]
    
    # Solve for the basis
    pivots, s_red = gf2_solve(Hs, s_res)
    
    # Map back to original indices
    e_corr = np.zeros(n, dtype=np.uint8)
    for i, col in enumerate(pivots):
        if i < len(s_red):
            e_corr[perm[col]] = s_red[i]
            
    # Flip the necessary bits in the original BP hard decisions
    return (hard_decisions ^ e_corr) % 2

# --- Decoders ---
class LayeredBPDecoder:
    """BP Decoder using Layered Scheduling for faster convergence on quantum codes."""
    def __init__(self, H, p, max_iter=32, nms_factor=0.625):
        self.H = H.tocsr() if sparse.issparse(H) else sparse.csr_matrix(H)
        self.m, self.n = self.H.shape
        self.max_iter = max_iter
        self.p = p
        self.nms_factor = nms_factor
        self.check_to_var = [self.H.getrow(i).indices for i in range(self.m)]

    def decode(self, syndrome):
        p_eff = 2 * self.p / 3 
        llr = np.full(self.n, np.log((1 - p_eff) / p_eff))
        msg_cv = np.zeros((self.m, self.n))

        for _ in range(self.max_iter):
            for i in range(self.m):
                idxs = self.check_to_var[i]
                if len(idxs) == 0: continue

                # Gather Variable-to-Check messages
                msg_vc = llr[idxs] - msg_cv[i, idxs]

                # Min-Sum processing
                sign = -1 if syndrome[i] else 1
                min_vals = np.abs(msg_vc)
                signs = np.sign(msg_vc)
                signs[signs == 0] = 1 

                total_sign = sign * np.prod(signs)

                # Update Check-to-Variable messages and LLRs immediately
                for idx_j, j in enumerate(idxs):
                    other_mins = np.delete(min_vals, idx_j)
                    m_val = np.min(other_mins) if len(other_mins) > 0 else 0.0
                    
                    new_msg = total_sign * signs[idx_j] * m_val * self.nms_factor
                    
                    # Layered update: Apply immediately to LLR
                    llr[j] += new_msg - msg_cv[i, j]
                    msg_cv[i, j] = new_msg

            # Check convergence
            guess = (llr < 0).astype(np.uint8)
            if not np.any((self.H @ guess) % 2 != syndrome):
                return guess, llr

        return (llr < 0).astype(np.uint8), llr

# --- Simulation Pipeline ---
def run_single_trial(args):
    HX, HZ, p, apply_osd = args
    n = HX.shape[1]
    X_err, Z_err = generate_pauli_error(n, p)
    
    # Z-type errors
    sX = (HX @ Z_err) % 2
    bpZ = LayeredBPDecoder(HX, p, max_iter, nms_factor)
    z_hat, llr_z = bpZ.decode(sX)
    
    if apply_osd and not np.all((HX @ z_hat) % 2 == sX):
        z_hat = osd0_decode(HX, sX, llr_z, z_hat)

    # X-type errors
    sZ = (HZ @ X_err) % 2
    bpX = LayeredBPDecoder(HZ, p, max_iter, nms_factor)
    x_hat, llr_x = bpX.decode(sZ)
    
    if apply_osd and not np.all((HZ @ x_hat) % 2 == sZ):
        x_hat = osd0_decode(HZ, sZ, llr_x, x_hat)

    success_Z = np.all((HX @ (Z_err ^ z_hat)) % 2 == 0)
    success_X = np.all((HZ @ (X_err ^ x_hat)) % 2 == 0)
    
    return success_Z and success_X

def simulate_parallel(HX, HZ, apply_osd, target_errors=25, max_trials=100000):
    results = []
    # Increased batch size for better parallelization efficiency
    batch_size = 2500 
    
    for p in p_list:
        errors = 0
        trials = 0
        
        with ProcessPoolExecutor() as executor:
            with tqdm(total=target_errors, desc=f"p={p:.3f}") as pbar:
                while errors < target_errors and trials < max_trials:
                    args = [(HX, HZ, p, apply_osd) for _ in range(batch_size)]
                    trial_results = list(executor.map(run_single_trial, args))
                    
                    trials += batch_size
                    new_errors = batch_size - sum(trial_results)
                    errors += new_errors
                    
                    pbar.update(new_errors)
                    pbar.set_postfix({'Trials': trials, 'WER': f"{errors/trials:.2e}"})
        
        wer = errors / trials if trials > 0 else 1.0
        if wer == 0: wer = 1 / max_trials 
        results.append(wer)
        
    return np.array(results)

if __name__ == "__main__":
    codes = {
        "A1 (GB)": (HX_A1, HZ_A1), 
        "B1 (GHP)": (HX_B1, HZ_B1)
    }
    
    plt.figure(figsize=(8, 6))
    for name, (HX, HZ) in codes.items():
        print(f"\nSimulating {name}...")
        
        print("Running Layered BP only...")
        res_bp = simulate_parallel(HX, HZ, apply_osd=False)
        
        print("Running Layered BP + OSD-0...")
        res_osd = simulate_parallel(HX, HZ, apply_osd=True)
        
        plt.semilogy(p_list, res_bp, 'o-', label=f"{name}, BP")
        plt.semilogy(p_list, res_osd, 'o--', label=f"{name}, BP+OSD-0")

    plt.xlabel("Physical error rate (p)")
    plt.ylabel("Word Error Rate (WER)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig("performance.png", dpi=300)
    print("\nSimulation complete. Results saved to performance.png")