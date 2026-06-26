import os
import numpy as np
import heapq
from scipy import sparse
import matplotlib.pyplot as plt

# --- Your Provided Loading Logic ---
N = 254
K = 28
L = 127

_dir_path = os.path.dirname(os.path.realpath(__file__))
# HX = sparse.load_npz(os.path.join(_dir_path, 'gb_254_28_hx.npz'))
# HZ = sparse.load_npz(os.path.join(_dir_path, 'gb_254_28_hz.npz'))

# Placeholder for testing HX = sparse.random(127, 254, density=0.05, format='csr', data_rvs=np.ones)

def calculate_heuristic(residual_syndrome, H_matrix, weights, J_forbidden):
   
    cost = 0
    # Iterate over active detectors (1s in the residual syndrome)
    active_detectors = np.where(residual_syndrome == 1)[0]
    
    for d in active_detectors:
        min_cost = float('inf')
        # Find all errors (columns) incident to this detector (row d)
        incident_errors = H_matrix[d].indices
        
        for e in incident_errors:
            if e not in J_forbidden:
                # GetDetCost logic: w(e) / |x \cap D(e)|
                # For simplicity in this skeleton, we just use the error weight
                min_cost = min(min_cost, weights[e])
        if min_cost != float('inf'):
            cost += min_cost
            
    return cost

def tesseract_decode(syndrome, H_matrix, p, beam_width=20):
    """
    A conceptual implementation of the Tesseract A* search decoder.
    """
    N_errors = H_matrix.shape[1]
    
    # Calculate error weights: w(e) = -log(p / (1-p))
    weights = -np.log(p / (1 - p)) * np.ones(N_errors)
    
    # Priority Queue for A* Search. 
    # Store tuples: (f_cost, g_cost, error_set_tuple, residual_syndrome_tuple)
    pq = []
    
    start_syndrome = tuple(syndrome)
    start_f = calculate_heuristic(np.array(start_syndrome), H_matrix, weights, set())
    
    # Push START node (empty set of errors)
    heapq.heappush(pq, (start_f, 0, (), start_syndrome))
    
    min_residual_weight = np.sum(syndrome)
    visited_residuals = set()
    
    nodes_explored = 0
    pq_limit = 200000 # To avoid infinite loops as per the paper
    
    while pq and nodes_explored < pq_limit:
        f_cost, g_cost, F, res_synd_tuple = heapq.heappop(pq)
        nodes_explored += 1
        res_synd = np.array(res_synd_tuple)
        
        # Check if EXIT node (residual syndrome is all zeros)
        if np.sum(res_synd) == 0:
            return list(F) # This is our Most-Likely Error
            
        # Beam Search Pruning
        current_res_weight = np.sum(res_synd)
        min_residual_weight = min(min_residual_weight, current_res_weight)
        if current_res_weight > min_residual_weight + beam_width:
            continue
            
        # No-revisit detections heuristic
        if res_synd_tuple in visited_residuals:
            continue
        visited_residuals.add(res_synd_tuple)
        
        # Expand node (Canonical ordering: only add errors incident to the lowest-index active detector)
        active_detectors = np.where(res_synd == 1)[0]
        if len(active_detectors) == 0: continue
        
        d_min = active_detectors[0] 
        incident_errors = H_matrix[d_min].indices
        
        # Forbid errors already in F
        J = set(F) 
        
        for e in incident_errors:
            if e in J: continue
            
            # Create new state
            new_F = tuple(sorted(list(F) + [e]))
            new_g = g_cost + weights[e]
            
            # Update syndrome: S_new = S_old XOR H[:, e]
            e_syndrome = H_matrix[:, e].toarray().flatten()
            new_res_synd = (res_synd + e_syndrome) % 2
            
            # Calculate A* heuristic
            h_cost = calculate_heuristic(new_res_synd, H_matrix, weights, J)
            new_f = new_g + h_cost
            
            heapq.heappush(pq, (new_f, new_g, new_F, tuple(new_res_synd)))

    return [] # Decoder failure

def run_simulation(H_matrix, physical_error_rates, shots=1000):
    logical_error_rates = []
    
    for p in physical_error_rates:
        errors = 0
        for _ in range(shots):
            # 1. Generate random error vector
            true_error = (np.random.rand(H_matrix.shape[1]) < p).astype(int)
            
            # 2. Compute Syndrome
            syndrome = H_matrix.dot(true_error) % 2
            
            # Skip trivial cases
            if np.sum(syndrome) == 0:
                continue
                
            # 3. Decode
            guessed_error_indices = tesseract_decode(syndrome, H_matrix, p)
            guessed_error = np.zeros(H_matrix.shape[1], dtype=int)
            guessed_error[guessed_error_indices] = 1
            
            # 4. Check success (Residual error must be in the stabilizer space)
            residual_error = (true_error + guessed_error) % 2
            
            # Note: For full verification, you need the Logical matrices (Lx, Lz) to check
            # if the residual error commutes with all logicals. 
            # For this simplified script, we assume failure if true_error != guessed_error
            if not np.array_equal(true_error, guessed_error):
                errors += 1
                
        ler = errors / shots
        logical_error_rates.append(ler)
        print(f"p = {p:.4f} | LER = {ler:.4f}")
        
    return logical_error_rates

# --- Execution ---
if __name__ == "__main__":
    p_vals = np.linspace(0.01, 0.1, 5)
    # Testing with HX
    ler_vals = run_simulation(HX, p_vals, shots=50)
    
    plt.figure(figsize=(8, 6))
    plt.plot(p_vals, ler_vals, marker='o', linestyle='-', label='Tesseract (Python)')
    plt.yscale('log')
    plt.xlabel('Physical Error Rate ($p$)')
    plt.ylabel('Logical Error Rate')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()