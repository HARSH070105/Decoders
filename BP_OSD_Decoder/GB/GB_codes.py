import numpy as np

code_name = "gb"

l = 127
k = 28

# Need an automatic way, baad mein dekheinge
a_indices = [0, 15, 20, 28, 66]
b_indices = [0, 58, 59, 100, 121]

def circulant_from_poly(indices, l):
    row = np.zeros(l, dtype=np.uint8)
    for i in indices:
        row[i % l] = 1

    M = np.zeros((l, l), dtype=np.uint8)
    for i in range(l):
        M[i] = np.roll(row, i)
    return M

A = circulant_from_poly(a_indices, l)
B = circulant_from_poly(b_indices, l)

HX = np.hstack([A, B])
HZ = np.hstack([B.T, A.T])

n = HX.shape[1]
if n==2*l:
    print(f"Code parameters: n={n}, k={k}, l={l}")
else:
    print("Unexpected code length")

commutator = (HX @ HZ.T) % 2
if np.any(commutator != 0):
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

    f.write(f"# Polynomial supports\n")
    f.write(f"A_INDICES = {a_indices}\n")
    f.write(f"B_INDICES = {b_indices}\n\n")

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