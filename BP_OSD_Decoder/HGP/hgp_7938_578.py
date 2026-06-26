import os
from scipy import sparse

N = 7938
K = 578
L = 63

# HP parameters
H_POLY = [0, 3, 34, 41, 57]

# Load parity check matrices from sparse .npz files
_dir_path = os.path.dirname(os.path.realpath(__file__))
HX = sparse.load_npz(os.path.join(_dir_path, 'hgp_7938_578_hx.npz'))
HZ = sparse.load_npz(os.path.join(_dir_path, 'hgp_7938_578_hz.npz'))
