from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional, Iterable
import os
import numpy as np
from scipy import sparse

OUT_DIR = "Codes"


# ============================================================
# Tile geometry
# ============================================================

def flat_to_xy(a: int, B: int) -> Tuple[int, int]:
    return (a % B, a // B)


def derive_z_from_x(X_h: Set[int], X_v: Set[int], B: int) -> Tuple[Set[int], Set[int]]:
    """Apply (T2): X horizontal a -> Z vertical B^2-1-a ; X vertical a -> Z horizontal B^2-1-a."""
    Z_h = {B * B - 1 - a for a in X_v}
    Z_v = {B * B - 1 - a for a in X_h}
    return Z_h, Z_v


def offsets_from_flat(indices: Iterable[int], B: int) -> List[Tuple[int, int]]:
    return [flat_to_xy(a, B) for a in indices]


def _edges_at_anchor(h_offsets, v_offsets, anchor):
    ax, ay = anchor
    hs = {('h', ax + dx, ay + dy) for (dx, dy) in h_offsets}
    vs = {('v', ax + dx, ay + dy) for (dx, dy) in v_offsets}
    return hs | vs


# ============================================================
# GF(2) helpers
# ============================================================

def gf2_rank(A: np.ndarray) -> int:
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


def find_anticommuting_pairs(HX, HZ, x_anchors, z_anchors, max_report=20):
    prod = (HX.astype(int) @ HZ.astype(int).T) % 2
    bad = np.argwhere(prod == 1)
    out = [(x_anchors[r], z_anchors[c]) for r, c in bad[:max_report]]
    return out, len(bad)


# ============================================================
# Core construction
# ============================================================

def build_tile_code(
    B: int,
    X_h: Set[int],
    X_v: Set[int],
    bulk_cols: int,
    bulk_rows: int,
    Z_h: Optional[Set[int]] = None,
    Z_v: Optional[Set[int]] = None,
    verbose: bool = True,
):
    """
    Returns dict with HX, HZ (np.uint8 arrays), qubit_index, n, and anchor
    lists, after verifying commutation.
    """
    if Z_h is None or Z_v is None:
        Z_h, Z_v = derive_z_from_x(X_h, X_v, B)

    x_h_off = offsets_from_flat(X_h, B)
    x_v_off = offsets_from_flat(X_v, B)
    z_h_off = offsets_from_flat(Z_h, B)
    z_v_off = offsets_from_flat(Z_v, B)

    if verbose:
        print(f"X-tile: h={sorted(x_h_off)} v={sorted(x_v_off)} (weight {len(X_h)+len(X_v)})")
        print(f"Z-tile: h={sorted(z_h_off)} v={sorted(z_v_off)} (weight {len(Z_h)+len(Z_v)})")

    # ---- bulk anchors ----
    bulk_anchors = [(i, j) for i in range(bulk_cols) for j in range(bulk_rows)]

    if verbose:
        bulk_qubit_preview: Set[Tuple[str, int, int]] = set()
        for a in bulk_anchors:
            bulk_qubit_preview |= _edges_at_anchor(x_h_off, x_v_off, a)
            bulk_qubit_preview |= _edges_at_anchor(z_h_off, z_v_off, a)
        print(f"[bulk] {len(bulk_anchors)} anchors -> {len(bulk_qubit_preview)} qubits (bulk only, before boundary)")

    extra = B - 1
    x_anchors = list(bulk_anchors)
    z_anchors = list(bulk_anchors)

    # X gets extra rows above/below, spanning only the ORIGINAL column range
    # (not the Z-extended range) -- avoids corner double-extension.
    for j in list(range(-extra, 0)) + list(range(bulk_rows, bulk_rows + extra)):
        for i in range(0, bulk_cols):
            x_anchors.append((i, j))
    # Z gets extra columns left/right, spanning only the ORIGINAL row range.
    for i in list(range(-extra, 0)) + list(range(bulk_cols, bulk_cols + extra)):
        for j in range(0, bulk_rows):
            z_anchors.append((i, j))

    if verbose:
        print(f"[boundary] extended to {len(x_anchors)} X-anchors, {len(z_anchors)} Z-anchors")

    # ---- qubit_set MUST be built from ALL anchors (bulk + boundary) ----
    # Building it from bulk anchors only (as an earlier version of this code
    # did) truncates boundary stabilizer supports asymmetrically: a qubit
    # that only appears near a boundary check (never in the bulk) gets
    # dropped from qubit_set, so it silently disappears from that check's
    # support -- but NOT necessarily from the neighboring check of the
    # opposite type at the same anchor, since X- and Z-anchors are extended
    # along different axes. That parity mismatch is exactly what breaks
    # commutation for tile shapes that aren't a simple axis-aligned
    # "staircase" (the hypergraph-product case from Appendix A always
    # happened to avoid it). Including boundary anchors here fixes this
    # for arbitrary (T1)+(T2)-valid tiles.
    qubit_set: Set[Tuple[str, int, int]] = set()
    for a in x_anchors:
        qubit_set |= _edges_at_anchor(x_h_off, x_v_off, a)
    for a in z_anchors:
        qubit_set |= _edges_at_anchor(z_h_off, z_v_off, a)

    if verbose:
        print(f"[qubit_set] {len(qubit_set)} qubits (bulk + boundary anchors)")

    # ---- truncate supports to qubit_set ----
    x_stabs = []
    for a in x_anchors:
        supp = _edges_at_anchor(x_h_off, x_v_off, a) & qubit_set
        if supp:
            x_stabs.append((a, supp))

    z_stabs = []
    for a in z_anchors:
        supp = _edges_at_anchor(z_h_off, z_v_off, a) & qubit_set
        if supp:
            z_stabs.append((a, supp))

    # ---- prune qubits touched by no X-stab or no Z-stab ----
    x_touched = set().union(*[s for _, s in x_stabs]) if x_stabs else set()
    z_touched = set().union(*[s for _, s in z_stabs]) if z_stabs else set()
    kept = x_touched & z_touched

    x_stabs = [(a, s & kept) for (a, s) in x_stabs]
    x_stabs = [(a, s) for (a, s) in x_stabs if s]
    z_stabs = [(a, s & kept) for (a, s) in z_stabs]
    z_stabs = [(a, s) for (a, s) in z_stabs if s]

    if verbose:
        print(f"[prune] qubits kept={len(kept)}  X-checks={len(x_stabs)}  Z-checks={len(z_stabs)}")

    # ---- fix qubit ordering: horizontal block, then vertical block ----
    h_qubits = sorted({(x, y) for (k, x, y) in kept if k == 'h'})
    v_qubits = sorted({(x, y) for (k, x, y) in kept if k == 'v'})
    qubit_index: Dict[Tuple[str, int, int], int] = {}
    for idx, (x, y) in enumerate(h_qubits):
        qubit_index[('h', x, y)] = idx
    off = len(h_qubits)
    for idx, (x, y) in enumerate(v_qubits):
        qubit_index[('v', x, y)] = off + idx

    n = len(h_qubits) + len(v_qubits)

    HX = np.zeros((len(x_stabs), n), dtype=np.uint8)
    for r, (a, supp) in enumerate(x_stabs):
        for key in supp:
            HX[r, qubit_index[key]] = 1

    HZ = np.zeros((len(z_stabs), n), dtype=np.uint8)
    for r, (a, supp) in enumerate(z_stabs):
        for key in supp:
            HZ[r, qubit_index[key]] = 1

    x_anchor_list = [a for a, _ in x_stabs]
    z_anchor_list = [a for a, _ in z_stabs]

    # ---- verify commutation before returning anything ----
    prod = (HX.astype(int) @ HZ.astype(int).T) % 2
    if prod.any():
        pairs, n_bad = find_anticommuting_pairs(HX, HZ, x_anchor_list, z_anchor_list)
        msg = [f"Stabilizers do NOT commute: {n_bad} anticommuting (X,Z) check pairs.",
               "First offending anchor pairs (X-anchor, Z-anchor):"]
        for xa, za in pairs:
            msg.append(f"  X@{xa}  vs  Z@{za}")
        msg.append("This tile's boundary anchor placement (Appendix A extension) "
                    "is not guaranteed for arbitrary tiles -- only for tiles whose "
                    "extra weight sits in a hypergraph-product-style layout. "
                    "Inspect the anticommuting pairs above (they usually cluster at "
                    "corners) and adjust boundary anchors for this tile shape.")
        raise ValueError("\n".join(msg))

    return dict(
        HX=HX, HZ=HZ, n=n,
        num_horizontal_qubits=len(h_qubits),
        num_vertical_qubits=len(v_qubits),
        qubit_index=qubit_index,
        x_anchors=x_anchor_list,
        z_anchors=z_anchor_list,
    )


def save_code(name: str, HX: np.ndarray, HZ: np.ndarray, out_dir: str = OUT_DIR,
              num_horizontal_qubits: Optional[int] = None):
    os.makedirs(out_dir, exist_ok=True)
    n = HX.shape[1]
    rk_x = gf2_rank(HX)
    rk_z = gf2_rank(HZ)
    k = n - rk_x - rk_z

    HX_csr = sparse.csr_matrix(HX)
    HZ_csr = sparse.csr_matrix(HZ)

    n_h = num_horizontal_qubits if num_horizontal_qubits is not None else n

    npz_path = os.path.join(out_dir, f"{name}.npz")
    np.savez(
        npz_path,
        HX_data=HX_csr.data, HX_indices=HX_csr.indices, HX_indptr=HX_csr.indptr, HX_shape=np.array(HX.shape),
        HZ_data=HZ_csr.data, HZ_indices=HZ_csr.indices, HZ_indptr=HZ_csr.indptr, HZ_shape=np.array(HZ.shape),
        n=n, k=k, num_horizontal_qubits=n_h,
    )

    txt_path = os.path.join(out_dir, f"{name}.txt")
    with open(txt_path, "w") as f:
        f.write(f"# Tile code: {name}\n")
        f.write(f"# n={n}  k={k}  rank(HX)={rk_x}  rank(HZ)={rk_z}\n")
        f.write(f"# HX: {HX.shape[0]} x {HX.shape[1]}\n")
        f.write("HX\n")
        for row in HX:
            f.write(" ".join(str(b) for b in row) + "\n")
        f.write(f"# HZ: {HZ.shape[0]} x {HZ.shape[1]}\n")
        f.write("HZ\n")
        for row in HZ:
            f.write(" ".join(str(b) for b in row) + "\n")

    print(f"n={n}  k={k}  rank(HX)={rk_x}  rank(HZ)={rk_z}")
    print(f"saved -> {npz_path}")
    print(f"saved -> {txt_path}")
    return dict(n=n, k=k, npz_path=npz_path, txt_path=txt_path)


def search_tile(
    B: int,
    weight: int,
    bulk_cols: int,
    bulk_rows: int,
    expected_n: int,
    expected_k: int,
    max_tries: int = 5000,
    seed: int = 0,
    also_check_bulk: Optional[Tuple[int, int]] = None,
    also_expected: Optional[Tuple[int, int]] = None,
):
    """
    Randomly search for an (X_h, X_v) tile pair of the given weight (split
    evenly between horizontal/vertical support) in a B x B box that:
      1. satisfies the commutation check in build_tile_code() (i.e. the
         Appendix A-style boundary extension used here is actually valid
         for this tile shape), and
      2. produces the requested [[expected_n, expected_k, ...]] code.

    Returns (X_h, X_v) on success, or None if max_tries is exhausted.
    """
    import random
    rng = random.Random(seed)
    half = weight // 2
    rest = weight - half

    for _ in range(max_tries):
        X_h = frozenset(rng.sample(range(B * B), half))
        X_v = frozenset(rng.sample(range(B * B), rest))
        try:
            r = build_tile_code(B=B, X_h=X_h, X_v=X_v,
                                bulk_cols=bulk_cols, bulk_rows=bulk_rows,
                                verbose=False)
        except ValueError:
            continue
        n = r["n"]
        k = n - gf2_rank(r["HX"]) - gf2_rank(r["HZ"])
        if (n, k) != (expected_n, expected_k):
            continue

        if also_check_bulk is not None:
            try:
                r2 = build_tile_code(B=B, X_h=X_h, X_v=X_v,
                                     bulk_cols=also_check_bulk[0],
                                     bulk_rows=also_check_bulk[1],
                                     verbose=False)
            except ValueError:
                continue
            n2 = r2["n"]
            k2 = n2 - gf2_rank(r2["HX"]) - gf2_rank(r2["HZ"])
            if (n2, k2) != also_expected:
                continue

        return X_h, X_v

    return None


def build_and_save(
    name: str,
    B: int,
    X_h: Set[int],
    X_v: Set[int],
    bulk_cols: int,
    bulk_rows: int,
    Z_h: Optional[Set[int]] = None,
    Z_v: Optional[Set[int]] = None,
    expected_n: Optional[int] = None,
    expected_k: Optional[int] = None,
    out_dir: str = OUT_DIR,
    verbose: bool = True,
):
    result = build_tile_code(B, X_h, X_v, bulk_cols, bulk_rows, Z_h, Z_v, verbose=verbose)
    HX, HZ, n = result["HX"], result["HZ"], result["n"]

    if expected_n is not None and n != expected_n:
        print(f"NOTE: n={n} does not match expected_n={expected_n}. "
              f"Try different bulk_cols/bulk_rows.")

    info = save_code(name, HX, HZ, out_dir=out_dir,
                     num_horizontal_qubits=result["num_horizontal_qubits"])

    if expected_k is not None and info["k"] != expected_k:
        print(f"NOTE: k={info['k']} does not match expected_k={expected_k}.")

    return info


if __name__ == "__main__":

    print("=" * 70)
    print("Tile Toric Code")
    print("=" * 70)
    build_and_save(
        name="tile_toric", 
        B=2,
        X_h={2, 3}, 
        X_v={1, 3},
        bulk_cols=10, 
        bulk_rows=10,
        expected_n=221, 
        expected_k=1,
    )

    # print("=" * 70)
    # print("[[288, 8, 12]]  B=3  weight-6")
    # print("=" * 70)
    # build_and_save(
    #     name="tile_288_8_12", 
    #     B=3,
    #     X_h={0, 5, 8}, 
    #     X_v={2, 6, 7},
    #     bulk_cols=10, 
    #     bulk_rows=10,
    #     expected_n=288, 
    #     expected_k=8,
    # )

    # print("\n" + "=" * 70)
    # print("[[288, 8, 14]]  B=3  weight-8")
    # print("=" * 70)
    # build_and_save(
    #     name="tile_288_8_14", 
    #     B=3,
    #     X_h={0, 2, 3, 6}, 
    #     X_v={0, 4, 6, 8},
    #     bulk_cols=10, 
    #     bulk_rows=10,
    #     expected_n=288, 
    #     expected_k=8,
    # )

    # print("\n" + "=" * 70)
    # print("[[288, 18, 13]]  B=4  weight-8")
    # print("=" * 70)
    # build_and_save(
    #     name="tile_288_18_13", 
    #     B=4,
    #     X_h={0, 3, 10, 12},
    #     X_v={1, 4, 5, 15},
    #     bulk_cols=9,
    #     bulk_rows=9,
    #     expected_n=288, 
    #     expected_k=18,
    # )

    # print("\n" + "=" * 70)
    # print("[[512, 18, 19]]  B=4  weight-8  (same tile as above, bulk=13x13)")
    # print("=" * 70)
    # build_and_save(
    #     name="tile_512_18_19",
    #     B=4,
    #     X_h={0, 3, 10, 12},
    #     X_v={1, 4, 5, 15},
    #     bulk_cols=13, 
    #     bulk_rows=13,
    #     expected_n=512, 
    #     expected_k=18,
    # )

    # ------------------------------------------------------------------
    # search_tile() is available for exploring other tile shapes /
    # (n,k) targets, e.g.:
    #
    #   X_h, X_v = search_tile(
    #       B=5, weight=10, bulk_cols=9, bulk_rows=9,
    #       expected_n=512, expected_k=18,
    #   )
    #   print(X_h, X_v)
    # ------------------------------------------------------------------