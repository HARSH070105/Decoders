from typing import Dict, Optional
import os
import numpy as np
from scipy import sparse

from tile_code_builder import gf2_rank, OUT_DIR

__all__ = ["hadamard_rotate_sector2", "bias_tailor_and_save"]


def hadamard_rotate_sector2(tile_result: Dict, verbose: bool = True) -> Dict:
    
    HX = tile_result["HX"].astype(np.uint8)
    HZ = tile_result["HZ"].astype(np.uint8)
    n_h = tile_result["num_horizontal_qubits"]
    n = tile_result["n"]

    sec1 = slice(0, n_h)      # horizontal qubits: sector 1 (unrotated)
    sec2 = slice(n_h, n)      # vertical qubits:   sector 2 (Hadamard-rotated)

    nX, nZ = HX.shape[0], HZ.shape[0]

    GX = np.zeros((nX + nZ, n), dtype=np.uint8)
    GZ = np.zeros((nX + nZ, n), dtype=np.uint8)

    # Rotated X-check rows: X-part keeps sector-1 support, sector-2 support
    # moves to the Z-part.
    GX[:nX, sec1] = HX[:, sec1]
    GZ[:nX, sec2] = HX[:, sec2]

    # Rotated Z-check rows: Z-part keeps sector-1 support, sector-2 support
    # moves to the X-part.
    GZ[nX:, sec1] = HZ[:, sec1]
    GX[nX:, sec2] = HZ[:, sec2]

    if verbose:
        print(f"[hadamard-rotate] sector1(h)={n_h} qubits unchanged, "
              f"sector2(v)={n - n_h} qubits rotated")
        print(f"[hadamard-rotate] {nX} rotated X-checks, {nZ} rotated Z-checks")

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
        f.write(f"# rows 0..{bt_result['num_X_checks']-1} = rotated X-checks, "
                f"rows {bt_result['num_X_checks']}..{bt_result['num_checks']-1} = rotated Z-checks\n")
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
        "bt_tile_288_8_12", tile_result,
        expected_n=288, expected_k=8,
    )