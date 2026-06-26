import os
from scipy import sparse

N = 882
K = 24
L = 63

# GHP parameters
M = 7
A_SHIFTS = [27, 0, 54, None, None, None, None]
B_POLY = [0, 1, 6]

# Load parity check matrices from sparse .npz files
_dir_path = os.path.dirname(os.path.realpath(__file__))
HX = sparse.load_npz(os.path.join(_dir_path, 'ghgp_882_24_hx.npz'))
HZ = sparse.load_npz(os.path.join(_dir_path, 'ghgp_882_24_hz.npz'))
