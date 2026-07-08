import numpy as np
from scipy import sparse
import os

OUT_DIR = "codes"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================== GF(2) helpers ==============================

def circ_from_poly(exps, ell):
    """ell x ell circulant matrix for polynomial with 1s at given exponents."""
    c = np.zeros(ell, dtype=np.uint8)
    for e in exps:
        c[e % ell] = 1
    C = np.zeros((ell, ell), dtype=np.uint8)
    for i in range(ell):
        for j in range(ell):
            C[i, j] = c[(i - j) % ell]
    return C


def zero_block(ell):
    return np.zeros((ell, ell), dtype=np.uint8)


def build_block_matrix(entries, ell):
    """entries: 2D list, each cell None or list-of-exponents."""
    rows = []
    for r in range(len(entries)):
        row_blocks = []
        for c in range(len(entries[0])):
            e = entries[r][c]
            row_blocks.append(zero_block(ell) if e is None else circ_from_poly(e, ell))
        rows.append(np.hstack(row_blocks))
    return np.vstack(rows)


def build_diag_block(block, count):
    n = block.shape[0]
    M = np.zeros((n * count, n * count), dtype=np.uint8)
    for i in range(count):
        M[i * n:(i + 1) * n, i * n:(i + 1) * n] = block
    return M


def transpose_block_entries(entries, ell):
    """Block-transpose + negate each circulant's exponents mod ell
    (since circulant(p(x))^T = circulant(p(x^-1 mod ell)))."""
    n_rows = len(entries)
    n_cols = len(entries[0])
    new_entries = [[None] * n_rows for _ in range(n_cols)]
    for r in range(n_rows):
        for c in range(n_cols):
            e = entries[r][c]
            if e is not None:
                new_e = [(-x) % ell for x in e]
                new_entries[c][r] = new_e
    return new_entries


def gf2_rank(A):
    M = A.copy().astype(np.uint8)
    m, n = M.shape
    rank = 0
    for col in range(n):
        pivot = None
        for row in range(rank, m):
            if M[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            M[[rank, pivot]] = M[[pivot, rank]]
        for row in range(m):
            if row != rank and M[row, col]:
                M[row] ^= M[rank]
        rank += 1
    return rank


def verify_and_save(name, HX, HZ, expected_k, expected_n=None):
    HX = HX.astype(np.uint8)
    HZ = HZ.astype(np.uint8)
    comm = (HX @ HZ.T) % 2
    assert np.abs(comm).max() == 0, f"{name}: CSS commutativity HX@HZ.T=0 FAILED"

    n = HX.shape[1]
    rk_HX = gf2_rank(HX)
    rk_HZ = gf2_rank(HZ)
    k = n - rk_HX - rk_HZ

    print(f"{name}: n={n}, rank(HX)={rk_HX}, rank(HZ)={rk_HZ}, k={k} "
          f"(expected k={expected_k})")
    if expected_n is not None:
        assert n == expected_n, f"{name}: length mismatch, got {n}, expected {expected_n}"
    assert k == expected_k, f"{name}: dimension mismatch! got k={k}, expected k={expected_k}"

    path = os.path.join(OUT_DIR, f"{name}.npz")
    np.savez(
        path,
        HX_data=sparse.csr_matrix(HX).data,
        HX_indices=sparse.csr_matrix(HX).indices,
        HX_indptr=sparse.csr_matrix(HX).indptr,
        HX_shape=np.array(HX.shape),
        HZ_data=sparse.csr_matrix(HZ).data,
        HZ_indices=sparse.csr_matrix(HZ).indices,
        HZ_indptr=sparse.csr_matrix(HZ).indptr,
        HZ_shape=np.array(HZ.shape),
        n=n, k=k,
    )
    print(f"  -> saved to {path}")


# ============================== A1: GB [[254,28]] ==============================
# a(x) = 1 + x^15 + x^20 + x^28 + x^66
# b(x) = 1 + x^58 + x^59 + x^100 + x^121
# ell = 127
def build_A1():
    ell = 127
    a_exps = [0, 15, 20, 28, 66]
    b_exps = [0, 58, 59, 100, 121]
    A = circ_from_poly(a_exps, ell)
    B = circ_from_poly(b_exps, ell)
    HX = np.hstack([A, B])
    HZ = np.hstack([B.T, A.T])
    return HX, HZ


# ============================== A2: GB [[126,28,8]] ==============================
# a(x) = 1 + x + x^14 + x^16 + x^22
# b(x) = 1 + x^3 + x^13 + x^20 + x^42
# ell = 63
def build_A2():
    ell = 63
    a_exps = [0, 1, 14, 16, 22]
    b_exps = [0, 3, 13, 20, 42]
    A = circ_from_poly(a_exps, ell)
    B = circ_from_poly(b_exps, ell)
    HX = np.hstack([A, B])
    HZ = np.hstack([B.T, A.T])
    return HX, HZ


# ============================== B1: GHP [[882,24]] ==============================
# ell = 63, b(x) = 1 + x + x^6, B = b(x) I_7
# A is the 7x7 block matrix given in Appendix B
def build_B1():
    ell = 63
    A_entries = [
        [[27], None, None, None, None, [0], [54]],
        [[54], [27], None, None, None, None, [0]],
        [[0], [54], [27], None, None, None, None],
        [None, [0], [54], [27], None, None, None],
        [None, None, [0], [54], [27], None, None],
        [None, None, None, [0], [54], [27], None],
        [None, None, None, None, [0], [54], [27]],
    ]
    b_exps = [0, 1, 6]

    A = build_block_matrix(A_entries, ell)
    B = build_diag_block(circ_from_poly(b_exps, ell), 7)
    HX = np.hstack([A, B])

    A_star_entries = transpose_block_entries(A_entries, ell)
    A_star = build_block_matrix(A_star_entries, ell)
    B_T = circ_from_poly(b_exps, ell).T.copy()
    HZ = np.hstack([build_diag_block(B_T, 7), A_star])

    return HX, HZ


if __name__ == "__main__":
    HX, HZ = build_A1()
    verify_and_save("gb_254_28", HX, HZ, expected_k=28, expected_n=254)

    HX, HZ = build_A2()
    verify_and_save("gb_126_28", HX, HZ, expected_k=28, expected_n=126)

    HX, HZ = build_B1()
    verify_and_save("ghp_882_24", HX, HZ, expected_k=24, expected_n=882)

    print("\nAll codes built and verified against paper dimensions.")