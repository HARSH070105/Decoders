import numpy as np
from scipy.sparse import csr_matrix
import galois

GF2 = galois.GF(2)

def rank_gf2(M):
    return int(np.linalg.matrix_rank(GF2(np.array(M, dtype=int) % 2)))

d = np.load("codes/tile_288_8_12.npz")

Hx = csr_matrix((d['HX_data'], d['HX_indices'], d['HX_indptr']), shape=tuple(d['HX_shape'])).toarray()
Hz = csr_matrix((d['HZ_data'], d['HZ_indices'], d['HZ_indptr']), shape=tuple(d['HZ_shape'])).toarray()

n = Hx.shape[1]
kx = n - rank_gf2(Hx)
kz = n - rank_gf2(Hz)

print(f"n = {n}")
print(f"kx = {kx}")
print(f"kz = {kz}")