import numpy as np
from scipy import sparse

code_name = "gb"
l = 127
k = 28
a_indices = [0, 15, 20, 28, 66]
b_indices = [0, 58, 59, 100, 121]

def sparse_circulant(indices, l):
    data = np.ones(len(indices) * l, dtype=np.uint8)
    rows = []
    cols = []
    for i in range(l):
        for idx in indices:
            rows.append(i)
            cols.append((i + idx) % l)
    return sparse.csr_matrix((data, (rows, cols)), shape=(l, l))

A = sparse_circulant(a_indices, l)
B = sparse_circulant(b_indices, l)

HX = sparse.hstack([A, B], format="csr")
HZ = sparse.hstack([B.transpose(), A.transpose()], format="csr")

n = HX.shape[1]
print(f"Code parameters: n={n}, k={k}, l={l}")

# CSS Check
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

# Save sparse matrices
sparse.save_npz(filename_hx, HX)
sparse.save_npz(filename_hz, HZ)

# Generate the .py configuration file
with open(filename_py, "w") as f:
    f.write("import os\n")
    f.write("from scipy import sparse\n\n")
    f.write(f"N = {n}\n")
    f.write(f"K = {k}\n")
    f.write(f"L = {l}\n\n")

    f.write(f"# Polynomial supports\n")
    f.write(f"A_INDICES = {a_indices}\n")
    f.write(f"B_INDICES = {b_indices}\n\n")

    f.write("# Load parity check matrices from sparse .npz files\n")
    f.write(f"_dir_path = os.path.dirname(os.path.realpath(__file__))\n")
    f.write(f"HX = sparse.load_npz(os.path.join(_dir_path, '{filename_hx}'))\n")
    f.write(f"HZ = sparse.load_npz(os.path.join(_dir_path, '{filename_hz}'))\n")

print(f"Generated: {filename_py} and accompanying .npz files")