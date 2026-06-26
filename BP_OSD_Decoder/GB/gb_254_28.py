import os
from scipy import sparse

N = 254
K = 28
L = 127

# Polynomial supports
A_INDICES = [0, 15, 20, 28, 66]
B_INDICES = [0, 58, 59, 100, 121]

# Load parity check matrices from sparse .npz files
_dir_path = os.path.dirname(os.path.realpath(__file__))
HX = sparse.load_npz(os.path.join(_dir_path, 'gb_254_28_hx.npz'))
HZ = sparse.load_npz(os.path.join(_dir_path, 'gb_254_28_hz.npz'))
