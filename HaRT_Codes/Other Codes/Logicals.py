import numpy as np

def gf2_rref(A):
    """Computes Row Reduced Echelon Form over GF(2)."""
    M = A.copy() % 2
    rows, cols = M.shape
    r = 0
    pivot_cols = []
    for c in range(cols):
        if r >= rows: break
        pivot = np.argmax(M[r:, c]) + r
        if M[pivot, c] == 0: continue
        M[[r, pivot]] = M[[pivot, r]]
        pivot_cols.append(c)
        for i in range(rows):
            if i != r and M[i, c] == 1:
                M[i] ^= M[r]
        r += 1
    return M, pivot_cols

def gf2_rank(A):
    if len(A) == 0: return 0
    return len(gf2_rref(A)[1])

def gf2_nullspace(A):
    """Computes a basis for the right nullspace of A over GF(2)."""
    A = A % 2
    m, n = A.shape
    R, pivot_cols = gf2_rref(A)
    free_cols = [c for c in range(n) if c not in pivot_cols]
    
    if not free_cols: return np.zeros((0, n), dtype=int)
    
    null_basis = np.zeros((len(free_cols), n), dtype=int)
    for i, free_col in enumerate(free_cols):
        null_basis[i, free_col] = 1
        for r, p_col in enumerate(pivot_cols):
            if R[r, free_col] == 1:
                null_basis[i, p_col] = 1
    return null_basis

def gf2_inv(A):
    """Inverts a square matrix A over GF(2)."""
    A = A % 2
    n = A.shape[0]
    augmented = np.hstack([A, np.eye(n, dtype=int)])
    R, pivot_cols = gf2_rref(augmented)
    if len(pivot_cols) < n or pivot_cols[:n] != list(range(n)):
        raise ValueError("Matrix is singular over GF(2) and cannot be inverted.")
    return R[:, n:]

def get_quotient_basis(K, S):
    """
    Finds basis for K / S over GF(2).
    """
    if len(S) == 0: return K.copy()
    rank_S = gf2_rank(S)
    basis = []
    current_span = list(S)
    current_rank = rank_S
    
    for v in K:
        test_span = np.vstack([current_span, v])
        new_rank = gf2_rank(test_span)
        if new_rank > current_rank:
            basis.append(v)
            current_span.append(v)
            current_rank = new_rank
            
    return np.array(basis)

def build_canonical_logicals(HX, HZ):
    """
    Computes exact canonical LX and LZ.
    Ensures LX @ LZ^T = I_k (mod 2).
    """
    HX = HX.astype(int) % 2
    HZ = HZ.astype(int) % 2
    
    # 1. Compute Kernels
    ker_HZ = gf2_nullspace(HZ)
    ker_HX = gf2_nullspace(HX)

    # 2. Extract independent logicals (quotient space)
    L_X_tilde = get_quotient_basis(ker_HZ, HX)
    L_Z_tilde = get_quotient_basis(ker_HX, HZ)

    k = L_X_tilde.shape[0]
    if k == 0:
        return np.zeros((0, HX.shape[1]), dtype=int), np.zeros((0, HX.shape[1]), dtype=int)

    # 3. Canonicalize via Symplectic Matrix Inverse
    Omega = (L_X_tilde @ L_Z_tilde.T) % 2
    Omega_inv = gf2_inv(Omega)

    LX = L_X_tilde
    LZ = (Omega_inv.T @ L_Z_tilde) % 2

    return LX, LZ