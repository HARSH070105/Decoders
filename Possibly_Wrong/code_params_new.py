import numpy as np
from scipy.sparse import csr_matrix
import galois

GF2 = galois.GF(2)

def rank_gf2(M):
    return int(np.linalg.matrix_rank(GF2(np.array(M, dtype=int) % 2)))

def load_hxhz(path):
    d = np.load(path)
    Hx = csr_matrix((d['HX_data'], d['HX_indices'], d['HX_indptr']), shape=tuple(d['HX_shape'])).toarray()
    Hz = csr_matrix((d['HZ_data'], d['HZ_indices'], d['HZ_indptr']), shape=tuple(d['HZ_shape'])).toarray()
    return Hx, Hz

def symplectic_code_nk(Hx, Hz):
    """
    For codes where HX/HZ are the symplectic X-part and Z-part of the SAME
    checks (row i of HX and row i of HZ together = one Pauli check) --
    e.g. after a Hadamard rotation of a CSS code, or any general stabilizer
    code stored this way. This is DIFFERENT from CSS Hx/Hz, where rows of
    Hx and Hz are separate, independent generators.

    H = [HX | HZ] side-by-side, shape (m, 2n). One row per check.
    """
    Hx = np.array(Hx, dtype=int) % 2
    Hz = np.array(Hz, dtype=int) % 2
    assert Hx.shape == Hz.shape, "HX and HZ must be the same shape (one X-part/Z-part pair per check)"
    m, n = Hx.shape

    Xg, Zg = GF2(Hx), GF2(Hz)
    # commutator: C[i,j] = x_i.z_j + z_i.x_j (mod 2); rows commute iff C=0 everywhere
    C = np.array((Xg @ Zg.T) + (Zg @ Xg.T))
    commute = bool(np.all(C == 0))

    H = np.hstack([Hx, Hz])   # (m, 2n)
    r = rank_gf2(H)
    k = n - r

    print(f"n = {n}")
    print(f"num checks (rows) = {m}")
    print(f"rank(H) [symplectic, {H.shape[0]}x{H.shape[1]}] = {r}")
    print(f"All checks mutually commute: {commute}")
    if not commute:
        print(f"  -> NOT a valid stabilizer group. Bad row pairs:\n  {np.argwhere(C != 0)}")
    k_str = str(k) if commute else "undefined (invalid stabilizer group)"
    print(f"k = n - rank(H) = {k_str}")

    return {"n": n, "k": k if commute else None, "valid_stabilizer_group": commute}

if __name__ == "__main__":
    path = 'codes/bt_tile_288_8_12_new.npz'
    Hx, Hz = load_hxhz(path)
    symplectic_code_nk(Hx, Hz)