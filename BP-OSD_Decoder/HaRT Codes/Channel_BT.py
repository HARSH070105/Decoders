"""
Channel_BT.py — error generation for bias-tailored (Hadamard-rotated) tile codes.

The bias-tailored construction keeps the original CSS check matrices HX, HZ
unchanged and instead applies the Hadamard rotation to the *error model*
(paper Appendix D, Algorithm 1, lines 5-12).

For sector-1 qubits (indices 0 .. n_h-1) the assignment is standard:
    e_x[j] = 1  iff Pauli X or Y occurred
    e_z[j] = 1  iff Pauli Z or Y occurred

For sector-2 qubits (indices n_h .. n-1) X and Z are swapped, because a
Hadamard gate maps X -> Z and Z -> X:
    e_x[j] = 1  iff Pauli Z or Y occurred   (physical Z -> effective X)
    e_z[j] = 1  iff Pauli X or Y occurred   (physical X -> effective Z)

This function supports all four noise models from Channel.py so that
Simulate_BT.py can iterate over the same set of models.
"""

import numpy as np


def generate_bt_error(n: int, p: float, n_h: int, noise_model: str = 'depolarizing'):
    """
    Sample a Pauli error and apply the sector-2 Hadamard swap.

    Parameters
    ----------
    n          : total number of qubits
    p          : physical error rate
    n_h        : sector-1 size (sector-2 = qubits n_h .. n-1)
    noise_model: 'depolarizing' | 'pure_x' | 'pure_z' | 'pure_y'

    Returns
    -------
    e_x, e_z : np.ndarray uint8, shape (n,)
        Effective X- and Z-error vectors in the Hadamard-rotated frame.
        These are decoded directly with the original CSS matrices HX, HZ.
    """
    choices = [0, 1, 2, 3]   # I, X, Z, Y  (note: Z=2, Y=3 — matches Channel.py)

    if noise_model == 'depolarizing':
        probs = [1 - p, p/3, p/3, p/3]
    elif noise_model == 'pure_x':
        probs = [1 - p, p, 0.0, 0.0]
    elif noise_model == 'pure_z':
        probs = [1 - p, 0.0, p, 0.0]
    elif noise_model == 'pure_y':
        probs = [1 - p, 0.0, 0.0, p]
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

    errors = np.random.choice(choices, size=n, p=probs)

    e_x = np.zeros(n, dtype=np.uint8)
    e_z = np.zeros(n, dtype=np.uint8)

    # ---- sector-1: standard Pauli -> (e_x, e_z) mapping ----
    s1 = slice(0, n_h)
    e_x[s1] = ((errors[s1] == 1) | (errors[s1] == 3)).astype(np.uint8)  # X or Y
    e_z[s1] = ((errors[s1] == 2) | (errors[s1] == 3)).astype(np.uint8)  # Z or Y

    # ---- sector-2: X <-> Z swapped (Hadamard rotation) ----
    s2 = slice(n_h, n)
    e_x[s2] = ((errors[s2] == 2) | (errors[s2] == 3)).astype(np.uint8)  # Z or Y -> eff. X
    e_z[s2] = ((errors[s2] == 1) | (errors[s2] == 3)).astype(np.uint8)  # X or Y -> eff. Z

    return e_x, e_z
