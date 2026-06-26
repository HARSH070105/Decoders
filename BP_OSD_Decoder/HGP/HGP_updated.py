import numpy as np
from scipy import sparse

code_name = "hgp"
l = 63
k = 578
h_indices = [0, 3, 34, 41, 57]

def sparse_circulant(indices, l):
    data = np.ones(len(indices) * l, dtype=np.uint8)
    rows = []
    cols = []
    for i in range(l):
        for idx in indices:
            rows.append(i)
            cols.append((i + idx) % l)
    return sparse.csr_matrix((data, (rows, cols)), shape=(l, l))

H = sparse_circulant(h_indices, l)
I = sparse.eye(l, dtype=np.uint8, format="csr")

HX_left  = sparse.kron(I, H)
HX_right = sparse.kron(H.transpose(), I)
HX = sparse.hstack([HX_left, HX_right], format="csr")

HZ_left  = sparse.kron(H, I)
HZ_right = sparse.kron(I, H.transpose())
HZ = sparse.hstack([HZ_left, HZ_right], format="csr")

n = HX.shape[1]
print(f"Code parameters: n={n}, k={k}, l={l}")

comm = HX.dot(HZ.transpose())
comm.data = comm.data % 2
comm.eliminate_zeros()
if comm.nnz != 0:
    print("CSS Condition Failed")
else:
    print("CSS Condition Passed")

filename_py = f"{code_name}_{n}_{k}.py"
filename_hx = f"{code_name}_{n}_{k}_hx.npz"
filename_hz = f"{code_name}_{n}_{k}_hz.npz"

sparse.save_npz(filename_hx, HX)
sparse.save_npz(filename_hz, HZ)

with open(filename_py, "w") as f:
    f.write("import os\n")
    f.write("from scipy import sparse\n\n")
    f.write(f"N = {n}\n")
    f.write(f"K = {k}\n")
    f.write(f"L = {l}\n\n")

    f.write(f"# HP parameters\n")
    f.write(f"H_POLY = {h_indices}\n\n")

    f.write("# Load parity check matrices from sparse .npz files\n")
    f.write(f"_dir_path = os.path.dirname(os.path.realpath(__file__))\n")
    f.write(f"HX = sparse.load_npz(os.path.join(_dir_path, '{filename_hx}'))\n")
    f.write(f"HZ = sparse.load_npz(os.path.join(_dir_path, '{filename_hz}'))\n")

print(f"Generated: {filename_py} and accompanying .npz files")