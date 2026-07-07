import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy import sparse
from itertools import product

# Import matrices 
from GB.gb_254_28 import HX as HX_A1, HZ as HZ_A1
from GHGP.ghgp_882_24 import HX as HX_B1, HZ as HZ_B1

# --- Configuration ---
p_list = np.linspace(0.01, 0.14, 8)
max_iter = 32
nms_factor = 0.625
OSD_ORDER = 0          # 0 = original OSD-0 behavior. Try 6-12 to push past the error floor.
TARGET_ERRORS = 50
MAX_TRIALS = 50000
BATCH_SIZE = 1000
N_WORKERS = None        # None -> os.cpu_count()


# ============================== Error channel ==============================

def generate_pauli_error(n, p, rng):
    """Vectorized depolarizing-channel error draw. Same statistics as the
    original per-qubit loop: P(X)=P(Z)=P(Y)=p/3 each."""
    r = rng.random(n)
    err_mask = r < p
    t = rng.integers(3, size=n)
    X = np.zeros(n, dtype=np.uint8)
    Z = np.zeros(n, dtype=np.uint8)
    X[err_mask & (t != 1)] = 1   # t==0 (X) or t==2 (Y)
    Z[err_mask & (t != 0)] = 1   # t==1 (Z) or t==2 (Y)
    return X, Z


# ============================== GF(2) linear algebra ==============================

def gf2_rref(H, s):
    """
    Full Gauss-Jordan reduction over GF(2), vectorized row-clearing.
    Returns: pivots (pivot column indices in the order found),
             M (the fully reduced augmented matrix),
             s_full (the fully reduced RHS, full length, not truncated),
             rank (= len(pivots)).
    Row i of M, for i < rank, is the pivot row for pivots[i].
    """
    M = H.copy()
    s_full = s.copy()
    m, n = M.shape
    pivots = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        col_data = M[row:, col]
        if not col_data.any():
            continue
        pivot = int(np.argmax(col_data)) + row
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
            s_full[[row, pivot]] = s_full[[pivot, row]]
        clear_mask = M[:, col].astype(bool)
        clear_mask[row] = False
        if clear_mask.any():
            M[clear_mask] ^= M[row]
            s_full[clear_mask] ^= s_full[row]
        pivots.append(col)
        row += 1
    return pivots, M, s_full, row


def gf2_solve(H, s):
    """Drop-in replacement for the original gf2_solve (same signature/output)."""
    pivots, M, s_full, rank = gf2_rref(H, s)
    return pivots, s_full[:rank]


def gf2_nullspace(A):
    """Drop-in replacement for the original gf2_nullspace."""
    pivots, M, _, rank = gf2_rref(A.copy(), np.zeros(A.shape[0], dtype=np.uint8))
    n = A.shape[1]
    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]
    null_basis = np.zeros((len(free_cols), n), dtype=np.uint8)
    for i, f in enumerate(free_cols):
        null_basis[i, f] = 1
        for j, pc in enumerate(pivots):
            null_basis[i, pc] = M[j, f]
    return null_basis


# ============================== OSD ==============================

def osd_decode(H, syndrome, llr, hard_decisions, order=0, max_combos_cap=4096):
    """
    OSD-0 (order=0): least-reliable columns are greedily chosen as the
    pivot/solved set (Gaussian elimination naturally picks the leftmost
    available column at each step, and columns are pre-sorted least-reliable
    first) -- so the correction needed to match the residual syndrome is
    pushed onto the bits you trust least, while reliable bits default to
    "no extra flip". This is the same convention the original used; only the
    elimination internals are vectorized here.

    order=k>0: additionally searches all 2^k assignments of the k
    least-reliable bits *within the free (already-trusted) set* -- the free
    positions sitting closest to the reliability boundary, where "trust it,
    don't flip" is weakest -- and keeps whichever syndrome-consistent
    candidate has the lowest reliability-weighted Hamming cost. This is
    standard order-w OSD reprocessing, adapted to syndrome decoding.
    """
    if sparse.issparse(H):
        H = H.toarray()
    n = H.shape[1]
    s_res = (syndrome ^ (H @ hard_decisions)) % 2
    if not np.any(s_res):
        return hard_decisions

    perm = np.argsort(np.abs(llr))          # least reliable first
    Hs = H[:, perm]
    pivots, M, s_full, rank = gf2_rref(Hs, s_res)
    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]

    e_corr = np.zeros(n, dtype=np.uint8)
    base_s = s_full[:rank]
    for i, col in enumerate(pivots):
        e_corr[perm[col]] = base_s[i]

    if order <= 0 or len(free_cols) == 0:
        return (hard_decisions ^ e_corr) % 2

    k = min(order, len(free_cols))
    while 2 ** k > max_combos_cap and k > 0:
        k -= 1
    if k == 0:
        return (hard_decisions ^ e_corr) % 2

    test_free = free_cols[:k]                       # k least-reliable free columns
    test_free_orig = [perm[c] for c in test_free]
    M_test = M[:rank, test_free]

    abs_llr = np.abs(llr)
    best_e = e_corr
    best_cost = np.sum(abs_llr[e_corr.astype(bool)])

    for combo in product([0, 1], repeat=k):
        combo_arr = np.array(combo, dtype=np.uint8)
        if not combo_arr.any():
            continue  # already evaluated as the order-0 baseline above
        s_candidate = base_s ^ ((M_test @ combo_arr) % 2)
        cand = np.zeros(n, dtype=np.uint8)
        for i, col in enumerate(pivots):
            cand[perm[col]] = s_candidate[i]
        for j, oc in enumerate(test_free_orig):
            cand[oc] = combo_arr[j]
        cost = np.sum(abs_llr[cand.astype(bool)])
        if cost < best_cost:
            best_cost = cost
            best_e = cand

    return (hard_decisions ^ best_e) % 2


# ============================== Vectorized BP (flooding / normalized min-sum) ==============================

def build_bp_structures(H):
    """Precompute padded check->variable neighbor tables ONCE per code
    (not per trial, not per p -- this used to be rebuilt on every single
    decode() call via LayeredBPDecoder.__init__)."""
    Hc = H.tocsr() if sparse.issparse(H) else sparse.csr_matrix(H)
    m, n = Hc.shape
    indptr, indices = Hc.indptr, Hc.indices
    degrees = np.diff(indptr)
    dmax = int(degrees.max()) if m > 0 else 0

    col_idx = np.zeros((m, dmax), dtype=np.int64)
    mask = np.zeros((m, dmax), dtype=bool)
    for i in range(m):
        d = degrees[i]
        col_idx[i, :d] = indices[indptr[i]:indptr[i + 1]]
        mask[i, :d] = True

    return {
        "H": Hc, "m": m, "n": n, "dmax": dmax,
        "col_idx": col_idx, "mask": mask,
        "col_arange": np.arange(dmax)[None, :],
    }


class VectorizedBPDecoder:
    """Flooding-schedule normalized min-sum BP. See module docstring for the
    layered -> flooding rationale."""

    def __init__(self, structs, max_iter=32, nms_factor=0.625):
        self.s = structs
        self.max_iter = max_iter
        self.nms_factor = nms_factor

    def decode(self, syndrome, p):
        s = self.s
        m, n, dmax = s["m"], s["n"], s["dmax"]
        col_idx, mask, col_arange = s["col_idx"], s["mask"], s["col_arange"]
        H = s["H"]

        p_eff = 2 * p / 3
        channel_llr = np.full(n, np.log((1 - p_eff) / p_eff))
        var_llr = channel_llr.copy()
        msg_c2v = np.zeros((m, dmax))
        row_sign = np.where(syndrome == 1, -1.0, 1.0)
        guess = (var_llr < 0).astype(np.uint8)

        for _ in range(self.max_iter):
            if not np.any((H @ guess) % 2 != syndrome):
                return guess, var_llr

            msg_v2c = var_llr[col_idx] - msg_c2v
            msg_v2c[~mask] = np.inf   # padding never wins the min, contributes sign +1

            signs = np.sign(msg_v2c)
            signs[signs == 0] = 1.0
            mags = np.abs(msg_v2c)

            argmin1 = np.argmin(mags, axis=1)
            min1 = mags[np.arange(m), argmin1]
            mags_masked = mags.copy()
            mags_masked[np.arange(m), argmin1] = np.inf
            min2 = np.min(mags_masked, axis=1)

            is_argmin = (col_arange == argmin1[:, None])
            min_excl_self = np.where(is_argmin, min2[:, None], min1[:, None])

            total_sign = row_sign * np.prod(signs, axis=1)
            sign_excl_self = total_sign[:, None] * signs

            new_msg = self.nms_factor * sign_excl_self * min_excl_self
            new_msg[~mask] = 0.0

            delta = new_msg - msg_c2v
            delta[~mask] = 0.0
            np.add.at(var_llr, col_idx[mask], delta[mask])
            msg_c2v = new_msg

            guess = (var_llr < 0).astype(np.uint8)

        return guess, var_llr


# ============================== Worker-process state ==============================
# Each worker process builds its decoder structures ONCE (via the pool
# initializer) instead of every trial, and never repickles HX/HZ/PX/PZ
# per task -- only the small (p, apply_osd) tuple crosses the IPC boundary.

_W = {}

def _init_worker(HX, HZ, PX, PZ, max_iter, nms_factor, osd_order):
    global _W
    _W['HX'], _W['HZ'], _W['PX'], _W['PZ'] = HX, HZ, PX, PZ
    _W['decZ'] = VectorizedBPDecoder(build_bp_structures(HX), max_iter, nms_factor)  # Z-errors via HX
    _W['decX'] = VectorizedBPDecoder(build_bp_structures(HZ), max_iter, nms_factor)  # X-errors via HZ
    _W['osd_order'] = osd_order
    _W['rng'] = np.random.default_rng()   # OS-entropy seed -> independent stream per worker


def _run_trial(args):
    p, apply_osd = args
    W = _W
    HX, HZ, PX, PZ = W['HX'], W['HZ'], W['PX'], W['PZ']
    n = HX.shape[1]
    rng = W['rng']
    X_err, Z_err = generate_pauli_error(n, p, rng)

    sX = (HX @ Z_err) % 2
    z_hat, llr_z = W['decZ'].decode(sX, p)
    if apply_osd and not np.all((HX @ z_hat) % 2 == sX):
        z_hat = osd_decode(HX, sX, llr_z, z_hat, order=W['osd_order'])

    sZ = (HZ @ X_err) % 2
    x_hat, llr_x = W['decX'].decode(sZ, p)
    if apply_osd and not np.all((HZ @ x_hat) % 2 == sZ):
        x_hat = osd_decode(HZ, sZ, llr_x, x_hat, order=W['osd_order'])

    # Success iff the residual error is a stabilizer (in the row space of the
    # opposing check matrix), equivalently orthogonal to its nullspace --
    # this part of the original's logic was already correct.
    success_Z = np.all((PZ @ (Z_err ^ z_hat)) % 2 == 0)
    success_X = np.all((PX @ (X_err ^ x_hat)) % 2 == 0)
    return success_Z and success_X


# ============================== Simulation driver ==============================

def simulate(executor, p_list, apply_osd, target_errors, max_trials, batch_size, label=""):
    results = []
    for p in p_list:
        errors, trials = 0, 0
        with tqdm(total=target_errors, desc=f"{label} p={p:.3f}") as pbar:
            while errors < target_errors and trials < max_trials:
                args = [(p, apply_osd)] * batch_size
                trial_results = list(executor.map(_run_trial, args, chunksize=50))
                trials += batch_size
                new_errors = batch_size - sum(trial_results)
                errors += new_errors
                pbar.update(new_errors)
                pbar.set_postfix({'Trials': trials, 'WER': f"{errors/trials:.2e}"})
        wer = errors / trials if trials > 0 else 1.0
        if wer == 0:
            wer = 1 / max_trials
        results.append(wer)
    return np.array(results)


if __name__ == "__main__":
    codes = {
        "A1 (GB)": (HX_A1, HZ_A1),
        "B1 (GHP)": (HX_B1, HZ_B1),
    }

    plt.figure(figsize=(8, 6))
    for name, (HX, HZ) in codes.items():
        print(f"\nSetting up {name}...")
        HX = HX.tocsr() if sparse.issparse(HX) else sparse.csr_matrix(HX)
        HZ = HZ.tocsr() if sparse.issparse(HZ) else sparse.csr_matrix(HZ)

        print("Computing nullspaces for logical error checking...")
        PX = gf2_nullspace(HX.toarray())
        PZ = gf2_nullspace(HZ.toarray())

        print(f"Simulating {name}...")
        # ONE pool per code, reused for every p value and both sweeps below --
        # this used to be 16 pool spin-ups per code (8 p-values x 2 sweeps).
        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=_init_worker,
            initargs=(HX, HZ, PX, PZ, max_iter, nms_factor, OSD_ORDER),
        ) as executor:
            print("Running vectorized BP only...")
            res_bp = simulate(executor, p_list, apply_osd=False,
                               target_errors=TARGET_ERRORS, max_trials=MAX_TRIALS,
                               batch_size=BATCH_SIZE, label="BP")

            print(f"Running vectorized BP + OSD-{OSD_ORDER}...")
            res_osd = simulate(executor, p_list, apply_osd=True,
                                target_errors=TARGET_ERRORS, max_trials=MAX_TRIALS,
                                batch_size=BATCH_SIZE, label="BP+OSD")

        plt.semilogy(p_list, res_bp, 'o-', label=f"{name}, BP")
        plt.semilogy(p_list, res_osd, 'o--', label=f"{name}, BP+OSD-{OSD_ORDER}")

    plt.xlabel("Physical error rate (p)")
    plt.ylabel("Word Error Rate (WER)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig("hpc_decoder_performance_optimized.png", dpi=300)
    print("\nSimulation complete. Results saved to hpc_decoder_performance_optimized.png")