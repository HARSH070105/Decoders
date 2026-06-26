import numpy as np

code_name = "hgp"

l = 63
k = 578

h_indices = [0, 3, 34, 41, 57]

def circulant_from_poly(indices, l):
    row = np.zeros(l, dtype=np.uint8)
    for i in indices:
        row[i % l] = 1

    M = np.zeros((l, l), dtype=np.uint8)
    for i in range(l):
        M[i] = np.roll(row, i)
    return M

H = circulant_from_poly(h_indices, l)

I = np.eye(l, dtype=np.uint8)

HX_left  = np.kron(I, H)
HX_right = np.kron(H.T, I)
HX = np.hstack([HX_left, HX_right])

HZ_left  = np.kron(H, I)
HZ_right = np.kron(I, H.T)
HZ = np.hstack([HZ_left, HZ_right])

n = HX.shape[1]

print(f"Code parameters: n={n}, k={k}, l={l}")

if n != 2 * l * l:
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

    f.write(f"# HP parameters\n")
    f.write(f"H_POLY = {h_indices}\n\n")

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