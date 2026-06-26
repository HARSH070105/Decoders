import numpy as np
from scipy import sparse

code_name = "ghgp"
l = 63
k = 24
b_poly = [0, 1, 6]
m = 7

# Corrected shifts for Image B1
A_shifts = [
    [27  , 0   , 54  , None, None, None, None],
    [None, 27  , 0   , 54  , None, None, None],
    [None, None, 27  , 0   , 54  , None, None],
    [None, None, None, 27  , 0   , 54  , None],
    [None, None, None, None, 27  , 0   , 54  ],
    [54  , None, None, None, None, 27  , 0   ],
    [0   , 54  , None, None, None, None, 27  ],
]

def sparse_circulant_shift(shift, l):
    if shift is None:
        return sparse.csr_matrix((l, l), dtype=np.uint8)
    data = np.ones(l, dtype=np.uint8)
    rows = np.arange(l)
    cols = (np.arange(l) + shift) % l
    return sparse.csr_matrix((data, (rows, cols)), shape=(l, l))

def sparse_circulant_poly(indices, l):
    M = sparse.csr_matrix((l, l), dtype=np.uint8)
    for shift in indices:
        M = M + sparse_circulant_shift(shift, l)
    M.data = M.data % 2
    M.eliminate_zeros()
    return M

A_blocks = []
for i in range(m):
    row_blocks = []
    for j in range(m):
        row_blocks.append(sparse_circulant_shift(A_shifts[i][j], l))
    A_blocks.append(row_blocks)

A = sparse.bmat(A_blocks, format="csr")
B_block = sparse_circulant_poly(b_poly, l)
B = sparse.kron(sparse.eye(m, dtype=np.uint8), B_block, format="csr")

HX = sparse.hstack([A, B], format="csr")
HZ = sparse.hstack([B.transpose(), A.transpose()], format="csr")

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

    f.write(f"# GHP parameters\n")
    f.write(f"M = {m}\n")
    f.write(f"A_SHIFTS = {A_shifts[0]}\n")
    f.write(f"B_POLY = {b_poly}\n\n")

    f.write("# Load parity check matrices from sparse .npz files\n")
    f.write(f"_dir_path = os.path.dirname(os.path.realpath(__file__))\n")
    f.write(f"HX = sparse.load_npz(os.path.join(_dir_path, '{filename_hx}'))\n")
    f.write(f"HZ = sparse.load_npz(os.path.join(_dir_path, '{filename_hz}'))\n")

print(f"Generated: {filename_py} and accompanying .npz files")