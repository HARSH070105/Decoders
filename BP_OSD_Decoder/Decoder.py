import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy import sparse
from itertools import product

from Load_codes import load_code

# Load the matrices directly from the generated .npz files
HX_A1, HZ_A1 = load_code("codes/gb_254_28.npz")
HX_B1, HZ_B1 = load_code("codes/ghp_882_24.npz")

# --- Configuration ---
p_list = np.linspace(0.01, 0.14, 8)
max_iter = 32
nms_factor = 0.625
OSD_ORDER = 0

TARGET_ERRORS = 50          # stop early once we have enough errors for a good estimate
BATCH_SIZE = 2000

# --- Adaptive trial budget ---
# Low-p / high-performing configs (e.g. B1+OSD) can have WER far below 1e-5.
# A fixed MAX_TRIALS=50000 can NEVER see a single error there, which is exactly
# why the previous plot pinned to a flat floor. Instead we let each point run
# until it either finds TARGET_ERRORS, or hits a per-point ceiling that scales
# with how rare we expect errors to be at that p.
MIN_TRIALS_PER_POINT = 5_000
GLOBAL_MAX_TRIALS = 5_000_000   # hard safety ceiling per point (tune to your time budget)
NO_ERROR_GROWTH_FACTOR = 4      # if zero errors seen, keep multiplying the budget
CONFIDENCE_Z = 1.96              # ~95% for Wilson interval


def wilson_upper_bound(k, n, z=CONFIDENCE_Z):
    """Wilson score upper bound for a binomial rate, valid even when k=0.
    Used to report a meaningful *upper bound* on WER instead of silently
    clamping to 1/max_trials, which was the root bug before."""
    if n == 0:
        return 1.0
    phat = k / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    adj = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    return (centre + adj) / denom


# ============================== Error channel ==============================

def generate_pauli_error(n, p, rng):
    r = rng.random(n)
    err_mask = r < p
    t = rng.integers(3, size=n)
    X = np.zeros(n, dtype=np.uint8)
    Z = np.zeros(n, dtype=np.uint8)
    X[err_mask & (t != 1)] = 1
    Z[err_mask & (t != 0)] = 1
    return X, Z


# ============================== GF(2) linear algebra ==============================

def gf2_rref(H, s):
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
    pivots, M, s_full, rank = gf2_rref(H, s)
    return pivots, s_full[:rank]


def gf2_nullspace(A):
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
    if sparse.issparse(H):
        H = H.toarray()
    n = H.shape[1]
    s_res = (syndrome ^ (H @ hard_decisions)) % 2
    if not np.any(s_res):
        return hard_decisions

    perm = np.argsort(np.abs(llr))
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

    test_free = free_cols[:k]
    test_free_orig = [perm[c] for c in test_free]
    M_test = M[:rank, test_free]

    abs_llr = np.abs(llr)
    best_e = e_corr
    best_cost = np.sum(abs_llr[e_corr.astype(bool)])

    for combo in product([0, 1], repeat=k):
        combo_arr = np.array(combo, dtype=np.uint8)
        if not combo_arr.any():
            continue
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
            msg_v2c[~mask] = np.inf

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

_W = {}

def _init_worker(HX, HZ, PX, PZ, max_iter, nms_factor, osd_order):
    global _W
    _W['HX'], _W['HZ'], _W['PX'], _W['PZ'] = HX, HZ, PX, PZ
    _W['decZ'] = VectorizedBPDecoder(build_bp_structures(HX), max_iter, nms_factor)
    _W['decX'] = VectorizedBPDecoder(build_bp_structures(HZ), max_iter, nms_factor)
    _W['osd_order'] = osd_order
    _W['rng'] = np.random.default_rng()


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

    success_Z = np.all((PZ @ (Z_err ^ z_hat)) % 2 == 0)
    success_X = np.all((PX @ (X_err ^ x_hat)) % 2 == 0)
    return success_Z and success_X


# ============================== Simulation driver (adaptive budget) ==============================

def simulate(executor, p_list, apply_osd, target_errors, batch_size,
             min_trials, global_max_trials, label=""):
    """
    Runs batches until either `target_errors` logical errors are seen, or the
    per-point trial ceiling is exhausted. The ceiling GROWS while zero errors
    have been observed (NO_ERROR_GROWTH_FACTOR), instead of being a fixed
    MAX_TRIALS that silently clamps the WER estimate to a flat floor.

    Returns:
        wer_est   : point estimate (errors/trials), or Wilson upper bound
                    when errors == 0 (so the point is still meaningful and
                    not an artificial constant).
        is_censored: True where wer_est is an upper bound, not a direct hit.
    """
    wer_results = []
    censored_flags = []

    for p in p_list:
        errors, trials = 0, 0
        ceiling = min_trials
        pbar = tqdm(desc=f"{label} p={p:.4f}", unit="trial")

        while errors < target_errors and trials < ceiling:
            args = [(p, apply_osd)] * batch_size
            trial_results = list(executor.map(_run_trial, args, chunksize=50))
            trials += batch_size
            new_errors = batch_size - sum(trial_results)
            errors += new_errors
            pbar.update(batch_size)
            pbar.set_postfix({'errors': errors, 'ceiling': ceiling,
                               'WER~': f"{errors/trials:.2e}"})

            # Still zero errors and we're near the ceiling -> grow the budget
            # (this is what lets very-low-WER points, e.g. B1+OSD, actually
            # resolve instead of hitting a fixed MAX_TRIALS wall).
            if errors == 0 and trials >= ceiling and ceiling < global_max_trials:
                ceiling = min(ceiling * NO_ERROR_GROWTH_FACTOR, global_max_trials)

        pbar.close()

        if errors > 0:
            wer_results.append(errors / trials)
            censored_flags.append(False)
        else:
            # No errors even at the ceiling: report a statistically honest
            # upper bound instead of an arbitrary 1/max_trials constant.
            wer_results.append(wilson_upper_bound(0, trials))
            censored_flags.append(True)

    return np.array(wer_results), np.array(censored_flags)


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
        with ProcessPoolExecutor(
            max_workers=None,
            initializer=_init_worker,
            initargs=(HX, HZ, PX, PZ, max_iter, nms_factor, OSD_ORDER),
        ) as executor:
            print("Running vectorized BP only...")
            res_bp, cens_bp = simulate(
                executor, p_list, apply_osd=False,
                target_errors=TARGET_ERRORS, batch_size=BATCH_SIZE,
                min_trials=MIN_TRIALS_PER_POINT,
                global_max_trials=GLOBAL_MAX_TRIALS, label="BP")

            print(f"Running vectorized BP + OSD-{OSD_ORDER}...")
            res_osd, cens_osd = simulate(
                executor, p_list, apply_osd=True,
                target_errors=TARGET_ERRORS, batch_size=BATCH_SIZE,
                min_trials=MIN_TRIALS_PER_POINT,
                global_max_trials=GLOBAL_MAX_TRIALS, label="BP+OSD")

        # Plot: solid markers for real hits, open/hollow markers for censored
        # (upper-bound) points, so you can visually tell floor-limited points
        # apart from genuinely measured ones.
        p_bp_solid = p_list[~cens_bp]
        p_bp_hollow = p_list[cens_bp]
        p_osd_solid = p_list[~cens_osd]
        p_osd_hollow = p_list[cens_osd]

        line_bp, = plt.semilogy(p_list, res_bp, '-', color=None, label=f"{name}, BP")
        plt.semilogy(p_bp_solid, res_bp[~cens_bp], 'o', color=line_bp.get_color())
        plt.semilogy(p_bp_hollow, res_bp[cens_bp], 'o', mfc='none', color=line_bp.get_color())

        line_osd, = plt.semilogy(p_list, res_osd, '--', label=f"{name}, BP+OSD-{OSD_ORDER}")
        plt.semilogy(p_osd_solid, res_osd[~cens_osd], 'o', color=line_osd.get_color())
        plt.semilogy(p_osd_hollow, res_osd[cens_osd], 'o', mfc='none', color=line_osd.get_color())

    plt.xlabel("Physical error rate (p)")
    plt.ylabel("Word Error Rate (WER)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig("hpc_decoder_performance_optimized.png", dpi=300)
    print("\nSimulation complete. Results saved to hpc_decoder_performance_optimized.png")