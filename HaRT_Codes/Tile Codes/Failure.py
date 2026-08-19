import numpy as np

def check_logical_failure(e_x, e_z, guess_x, guess_z, HX, HZ, LX, LZ):
    """
    Verifies decoding success using syndrome and canonical logical checks.
    """
    r_x = (e_x + guess_x) % 2
    r_z = (e_z + guess_z) % 2
    
    # 1. Syndrome check (Did it return a valid stabilizer state?)
    if np.any((HZ @ r_x) % 2) or np.any((HX @ r_z) % 2):
        return True
        
    # 2. Canonical Logical check
    # Due to LX @ LZ^T = I, an X-residual triggers if it overlaps with LZ
    if np.any((LZ @ r_x) % 2):
        return True
        
    if np.any((LX @ r_z) % 2):
        return True
        
    return False