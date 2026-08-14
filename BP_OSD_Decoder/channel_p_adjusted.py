import numpy as np

def generate_depolarizing_error(n, p):
    choices = [0, 1, 2, 3]
    probs = [1 - p, p, 0, 0]
    
    errors = np.random.choice(choices, size=n, p=probs)
    
    e_x = np.zeros(n, dtype=np.uint8)
    e_z = np.zeros(n, dtype=np.uint8)
    
    # X errors are present in X (1) and Y (3)
    e_x[(errors == 1) | (errors == 3)] = 1
    # Z errors are present in Z (2) and Y (3)
    e_z[(errors == 2) | (errors == 3)] = 1
    
    return e_x, e_z