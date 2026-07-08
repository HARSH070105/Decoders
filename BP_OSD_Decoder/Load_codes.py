import numpy as np
from scipy import sparse


def load_code(npz_path):
    d = np.load(npz_path)
    HX = sparse.csr_matrix(
        (d["HX_data"], d["HX_indices"], d["HX_indptr"]),
        shape=tuple(d["HX_shape"]),
    )
    HZ = sparse.csr_matrix(
        (d["HZ_data"], d["HZ_indices"], d["HZ_indptr"]),
        shape=tuple(d["HZ_shape"]),
    )
    return HX, HZ


def load_code_info(npz_path):
    """Also returns (n, k) metadata stored alongside the matrices."""
    d = np.load(npz_path)
    HX, HZ = load_code(npz_path)
    return HX, HZ, int(d["n"]), int(d["k"])


if __name__ == "__main__":
    # quick smoke test
    for name in ["codes/gb_254_28.npz", "codes/gb_126_28.npz", "codes/ghp_882_24.npz"]:
        HX, HZ, n, k = load_code_info(name)
        comm = (HX @ HZ.T).toarray() % 2
        print(f"{name}: HX{HX.shape}, HZ{HZ.shape}, n={n}, k={k}, "
              f"CSS check ok={np.abs(comm).max() == 0}")