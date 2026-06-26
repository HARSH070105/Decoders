import numpy as np

code_name = "ghgp"

l = 63
k = 24

b_poly = [0, 1, 6]

# Size of A (7x7 blocks)
m = 7

# None → zero matrix
# 0 → identity (x^0)
# need to use circulant shifts insead of hardcoded (T_T), but lite
A_shifts = [
    [27  , None, None, None, None, 0   , 54  ],
    [54  , 27  , None, None, None, None, 0   ],
    [0   , 54  , 27  , None, None, None, None],
    [None, 0   , 54  , 27  , None, None, None],
    [None, None, 0   , 54  , 27  , None, None],
    [None, None, None, 0   , 54  , 27  , None],
    [None, None, None, None, 0   , 54  , 27  ],
]

def circulant_shift(shift, l):
    row = np.zeros(l, dtype=np.uint8)
    row[shift % l] = 1
    M = np.zeros((l, l), dtype=np.uint8)
    for i in range(l):
        M[i] = np.roll(row, i)
    return M

def circulant_poly(indices, l):
    M = np.zeros((l, l), dtype=np.uint8)
    for shift in indices:
        M ^= circulant_shift(shift, l)
    return M

A_blocks = []
for i in range(m):
    row_blocks = []
    for j in range(m):
        shift = A_shifts[i][j]

        if shift is None:
            row_blocks.append(np.zeros((l, l), dtype=np.uint8))
        else:
            row_blocks.append(circulant_shift(shift, l))

    A_blocks.append(row_blocks)

A = np.block(A_blocks)

B_block = circulant_poly(b_poly, l)
B = np.kron(np.eye(m, dtype=np.uint8), B_block)

HX = np.hstack([A, B])
HZ = np.hstack([B.T, A.T])

n = HX.shape[1]

print(f"Code parameters: n={n}, k={k}, l={l}")

if n != 2 * m * l:
    print("Unexpected size!")

comm = (HX @ HZ.T) % 2
if np.any(comm):
    print("CSS Condition Failed")
else:
    print("CSS Condition Passed")


filename = f"{code_name}_{n}_{k}.py"

def matrix_to_list(M):
    return M.astype(int).tolist()

with open(filename, "w") as f:
    f.write(f"N = {n}\n")
    f.write(f"K = {k}\n")
    f.write(f"L = {l}\n\n")

    f.write(f"# GHP parameters\n")
    f.write(f"M = {m}\n")
    f.write(f"A_SHIFTS = {A_shifts[0]}\n")
    f.write(f"B_POLY = {b_poly}\n\n")

    f.write("# Parity check matrices\n")
    f.write("HX = [\n")
    for row in matrix_to_list(HX):
        f.write(f"    {row},\n")
    f.write("]\n\n")

    f.write("HZ = [\n")
    for row in matrix_to_list(HZ):
        f.write(f"    {row},\n")
    f.write("]\n\n")

print(f"Generated: {filename}")