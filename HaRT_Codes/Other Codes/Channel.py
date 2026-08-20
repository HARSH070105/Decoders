import numpy as np

def generate_error(n, p, noise_model='depolarizing'):
    
    choices = [0, 1, 2, 3] # I, X, Z, Y
    
    if noise_model == 'depolarizing':
        probs = [1 - p, p/3, p/3, p/3]
    elif noise_model == 'pure_x':
        probs = [1 - p, p, 0, 0]
    elif noise_model == 'pure_z':
        probs = [1 - p, 0, p, 0]
    elif noise_model == 'pure_y':
        probs = [1 - p, 0, 0, p]
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")
    
    errors = np.random.choice(choices, size=n, p=probs)
    
    e_x = np.zeros(n, dtype=np.uint8)
    e_z = np.zeros(n, dtype=np.uint8)
    
    # X errors are present in X (1) and Y (3)
    e_x[(errors == 1) | (errors == 3)] = 1
    # Z errors are present in Z (2) and Y (3)
    e_z[(errors == 2) | (errors == 3)] = 1
    
    return e_x, e_z