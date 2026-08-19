"""
bias_tailored_tile_code.py

Fits "tile codes" (Steffan, Choe, Breuckmann, Pereira, Eberhardt,
arXiv:2504.09171) into the bias-tailored / XZZX framework of
Roffe, Cohen, Quintavalle, Chandra, Campbell (arXiv:2202.01702).

------------------------------------------------------------------------
THE IDEA
------------------------------------------------------------------------
Bias-tailoring (Sec 4.3, Eq. 26 of the Roffe et al. paper) takes a CSS code

    H_CSS = [  0     0   | H_Z1  H_Z2 ]
            [ H_X1  H_X2 |   0     0  ]

that is naturally split into two qubit "sectors" (sector 1, sector 2), and
applies a Hadamard gate to every qubit in sector 2. Conjugating by H swaps
X<->Z on those qubits for every stabiliser. Because H is Clifford, this
preserves commutation of the whole stabiliser group and leaves n, k
unchanged (only the code's response to biased noise changes: sector-2
qubits now contribute their X-type errors to what used to be a purely
Z-type check, and vice versa).

Tile codes are already built from two qubit sectors -- horizontal-edge
qubits and vertical-edge qubits -- exactly the "sector 1 / sector 2"
structure the bias-tailored construction needs. This module Hadamard-rotates
the vertical-edge qubits (sector 2) of a tile code, producing a non-CSS,
XZZX-style bias-tailored tile code.

Per check row, post-rotation:

    X-check (originally pure X-type):
        X-part = old X-support restricted to sector-1 (horizontal) qubits
        Z-part = old X-support restricted to sector-2 (vertical) qubits

    Z-check (originally pure Z-type):
        Z-part = old Z-support restricted to sector-1 (horizontal) qubits
        X-part = old Z-support restricted to sector-2 (vertical) qubits

Every check keeps its own row (no merging of X/Z checks) -- it just becomes
a genuinely mixed Pauli (both an X-part and a Z-part) exactly like the XZZX
stabiliser in Fig. 4 of the bias-tailoring paper.

Commutation is checked with the general (non-CSS) symplectic form:
for checks g=(gx|gz), h=(hx|hz),

    g and h commute  <=>  gx.hz^T + gz.hx^T = 0 (mod 2)

This is verified for every pair before anything is saved.

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
    from tile_code_builder import build_tile_code
    from bias_tailored_tile_code import bias_tailor_and_save

    result = build_tile_code(B=3, X_h={0,5,8}, X_v={2,6,7},
                              bulk_cols=10, bulk_rows=10)
    bias_tailor_and_save("bt_tile_288_8_12", result)

------------------------------------------------------------------------
OUTPUT
------------------------------------------------------------------------
For a code named e.g. "bt_tile_288_8_12", writes into ./codes/:

    bt_tile_288_8_12.npz  -- sparse GX, GZ (per-check symplectic parts,
                              scipy csr components) + n, k, num_checks
    bt_tile_288_8_12.txt   -- human readable: GX and GZ as separate 0/1
                              matrices (one row per check, space-separated),
                              i.e. the same "separate X and Z" format as
                              tile_code_builder.py's CSS output, except here
                              GX/GZ are per-CHECK columns of a single mixed
                              stabiliser list rather than disjoint X-checks
                              and Z-checks (see note in the file header).
"""

from typing import Dict, Optional
import os
import numpy as np
from scipy import sparse

from tile_code_builder import gf2_rank, OUT_DIR

__all__ = ["hadamard_rotate_sector2", "bias_tailor_and_save"]


def hadamard_rotate_sector2(tile_result: Dict, verbose: bool = True) -> Dict:
    """
    Apply the bias-tailoring Hadamard rotation to the vertical-edge qubit
    sector of a tile code produced by tile_code_builder.build_tile_code().

    Parameters
    ----------
    tile_result : dict
        The dict returned by build_tile_code() -- must contain
        'HX', 'HZ', 'n', 'num_horizontal_qubits', 'num_vertical_qubits'.
    verbose : bool

    Returns
    -------
    dict with:
        GX, GZ : (num_checks, n) uint8 arrays -- symplectic X-part and
                 Z-part of every check (rows 0..nX-1 are the rotated
                 X-checks, rows nX..nX+nZ-1 are the rotated Z-checks)
        n, k, num_checks, num_horizontal_qubits, num_vertical_qubits
    """
    HX = tile_result["HX"].astype(np.uint8)
    HZ = tile_result["HZ"].astype(np.uint8)
    n_h = tile_result["num_horizontal_qubits"]
    n = tile_result["n"]

    sec1 = slice(0, n_h)      # horizontal qubits: sector 1 (unrotated)
    sec2 = slice(n_h, n)      # vertical qubits:   sector 2 (Hadamard-rotated)

    nX, nZ = HX.shape[0], HZ.shape[0]

    GX = np.zeros((nZ + nX, n), dtype=np.uint8)
    GZ = np.zeros((nZ + nX, n), dtype=np.uint8)

    # Z-checks FIRST (rows 0..nZ-1): Z-part keeps sector-1 (h) support,
    # sector-2 (v) support moves to the X-part.
    # gX = [0 | Hzv],  gZ = [Hzh | 0]
    GZ[:nZ, sec1] = HZ[:, sec1]   # Hzh stays as Z on h-qubits
    GX[:nZ, sec2] = HZ[:, sec2]   # Hzv flips to X on v-qubits

    # X-checks SECOND (rows nZ..nZ+nX-1): X-part keeps sector-1 (h) support,
    # sector-2 (v) support moves to the Z-part.
    # gX = [Hxh | 0],  gZ = [0 | Hxv]
    GX[nZ:, sec1] = HX[:, sec1]   # Hxh stays as X on h-qubits
    GZ[nZ:, sec2] = HX[:, sec2]   # Hxv flips to Z on v-qubits

    if verbose:
        print(f"[hadamard-rotate] sector1(h)={n_h} qubits unchanged, "
              f"sector2(v)={n - n_h} qubits rotated")
        print(f"[hadamard-rotate] rows 0..{nZ-1}=Z-checks, "
              f"rows {nZ}..{nZ+nX-1}=X-checks")

    # ---- verify: every pair of checks must commute (general symplectic) ----
    comm = (GX.astype(int) @ GZ.astype(int).T + GZ.astype(int) @ GX.astype(int).T) % 2
    if comm.any():
        bad = np.argwhere(comm == 1)
        msg = [f"Bias-tailored checks do NOT commute: {len(bad)} anticommuting pairs.",
               "First offending row-index pairs (post-rotation check indices):"]
        for r, c in bad[:20]:
            msg.append(f"  check {r} vs check {c}")
        msg.append("This should not happen for a Hadamard rotation of a valid "
                    "CSS code -- check that the input tile_result actually came "
                    "from a verified-commuting CSS build_tile_code() call.")
        raise ValueError("\n".join(msg))

    full_rank_matrix = np.hstack([GX, GZ])
    k = n - gf2_rank(full_rank_matrix)

    if verbose:
        print(f"[hadamard-rotate] n={n}  num_checks={nX+nZ}  k={k}")

    return dict(
        GX=GX, GZ=GZ, n=n, k=k,
        num_checks=nX + nZ,
        num_X_checks=nX, num_Z_checks=nZ,
        num_horizontal_qubits=n_h,
        num_vertical_qubits=n - n_h,
    )


def save_bias_tailored_code(name: str, bt_result: Dict, out_dir: str = OUT_DIR):
    """
    Saves using the SAME key convention as tile_code_builder.save_code()
    (HX_data/HX_indices/HX_indptr/HX_shape, HZ_... likewise) so existing
    decoder scripts that load '{code}.npz' expecting HX/HZ (e.g.
    simulate_p_adjusted.py's load_code()) work unchanged on bias-tailored
    codes too. For a non-CSS code, HX/HZ here are the per-check symplectic
    X-part/Z-part (each row is one check's full Pauli support split into
    its X- and Z-components) -- same shapes/semantics a decoder needs,
    just not required to be disjoint per-row as in a CSS code.
    """
    os.makedirs(out_dir, exist_ok=True)
    HX, HZ = bt_result["GX"], bt_result["GZ"]
    n, k = bt_result["n"], bt_result["k"]

    HX_csr = sparse.csr_matrix(HX)
    HZ_csr = sparse.csr_matrix(HZ)

    npz_path = os.path.join(out_dir, f"{name}.npz")
    np.savez(
        npz_path,
        HX_data=HX_csr.data, HX_indices=HX_csr.indices, HX_indptr=HX_csr.indptr, HX_shape=np.array(HX.shape),
        HZ_data=HZ_csr.data, HZ_indices=HZ_csr.indices, HZ_indptr=HZ_csr.indptr, HZ_shape=np.array(HZ.shape),
        n=n, k=k,
        num_X_checks=bt_result["num_X_checks"],
        num_Z_checks=bt_result["num_Z_checks"],
        num_horizontal_qubits=bt_result["num_horizontal_qubits"],
        num_vertical_qubits=bt_result["num_vertical_qubits"],
    )

    txt_path = os.path.join(out_dir, f"{name}.txt")
    with open(txt_path, "w") as f:
        f.write(f"# Bias-tailored (Hadamard-rotated) tile code: {name}\n")
        f.write(f"# n={n}  k={k}  num_checks={bt_result['num_checks']}\n")
        f.write(f"# sector1 (horizontal, unrotated) qubits = {bt_result['num_horizontal_qubits']}\n")
        f.write(f"# sector2 (vertical, Hadamard-rotated) qubits = {bt_result['num_vertical_qubits']}\n")
        f.write(f"# rows 0..{bt_result['num_Z_checks']-1} = rotated Z-checks, "
                f"rows {bt_result['num_Z_checks']}..{bt_result['num_checks']-1} = rotated X-checks\n")
        f.write("# HX: symplectic X-part per check, HZ: symplectic Z-part per check\n")
        f.write(f"# HX: {HX.shape[0]} x {HX.shape[1]}\n")
        f.write("HX\n")
        for row in HX:
            f.write(" ".join(str(b) for b in row) + "\n")
        f.write(f"# HZ: {HZ.shape[0]} x {HZ.shape[1]}\n")
        f.write("HZ\n")
        for row in HZ:
            f.write(" ".join(str(b) for b in row) + "\n")

    print(f"n={n}  k={k}  num_checks={bt_result['num_checks']}")
    print(f"saved -> {npz_path}")
    print(f"saved -> {txt_path}")
    return dict(n=n, k=k, npz_path=npz_path, txt_path=txt_path)


def bias_tailor_and_save(
    name: str,
    tile_result: Dict,
    expected_n: Optional[int] = None,
    expected_k: Optional[int] = None,
    out_dir: str = OUT_DIR,
    verbose: bool = True,
):
    """Convenience wrapper: Hadamard-rotate a build_tile_code() result and
    save it. See module docstring for usage."""
    bt = hadamard_rotate_sector2(tile_result, verbose=verbose)

    if expected_n is not None and bt["n"] != expected_n:
        print(f"NOTE: n={bt['n']} does not match expected_n={expected_n}.")
    if expected_k is not None and bt["k"] != expected_k:
        print(f"NOTE: k={bt['k']} does not match expected_k={expected_k}.")

    return save_bias_tailored_code(name, bt, out_dir=out_dir)


if __name__ == "__main__":
    from tile_code_builder import build_tile_code

    # Example: bias-tailor the [[288,8,12]] tile code.
    tile_result = build_tile_code(
        B=3, X_h={0, 5, 8}, X_v={2, 6, 7}, bulk_cols=10, bulk_rows=10
    )
    bias_tailor_and_save(
        "bt_tile_288_8_12_new", tile_result,
        expected_n=288, expected_k=8,
    )