"""
Decoders_BT.py — decoder setup for bias-tailored (Hadamard-rotated) tile codes.

Two differences from Decoders.py:

1.  Per-qubit channel_probs instead of a scalar error_rate.
    The Hadamard swap means sector-1 and sector-2 qubits have different
    effective error rates when the noise is not symmetric (pure_x, pure_z).
    Under depolarising noise everything is uniform (2p/3), but the per-qubit
    array is built in all cases so biased models are handled correctly.

2.  channel_update_x_to_z() implements the Bayesian update from the paper's
    Algorithm 2: after the X-decoding round is done, the Z-decoder's priors
    are refined to account for Y-error correlations.  This gives a measurable
    WER improvement under depolarising noise (cf. paper Fig. 13).

Decoder matrix convention (unchanged from Decoders.py):
    X-error decoder  uses HZ  (s_x = HZ @ e_x)
    Z-error decoder  uses HX  (s_z = HX @ e_z)
"""

import numpy as np
from ldpc.bp_decoder import BpDecoder
from ldpc.bposd_decoder import BpOsdDecoder

# Small floor to avoid log(0) in LLR computations
_EPS = 1e-10


def _bt_priors(n: int, p: float, n_h: int, noise_model: str):
    """
    Return (px_priors, pz_priors) — per-qubit float arrays of length n.

    px_priors[j] = Pr[e_x[j] = 1]  used by the X-error decoder (factor graph HZ)
    pz_priors[j] = Pr[e_z[j] = 1]  used by the Z-error decoder (factor graph HX)

    Sector-1 (j < n_h): standard assignment.
    Sector-2 (j >= n_h): X and Z swapped due to Hadamard rotation.
    """
    if noise_model == 'depolarizing':
        p_X = p_Z = p_Y = p / 3.0
    elif noise_model == 'pure_x':
        p_X, p_Z, p_Y = p, 0.0, 0.0
    elif noise_model == 'pure_z':
        p_X, p_Z, p_Y = 0.0, p, 0.0
    elif noise_model == 'pure_y':
        p_X, p_Z, p_Y = 0.0, 0.0, p
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

    px_priors = np.empty(n, dtype=float)
    pz_priors = np.empty(n, dtype=float)

    # sector-1: Pr[e_x] = p_X + p_Y,  Pr[e_z] = p_Z + p_Y
    px_priors[:n_h] = np.clip(p_X + p_Y, _EPS, 1 - _EPS)
    pz_priors[:n_h] = np.clip(p_Z + p_Y, _EPS, 1 - _EPS)

    # sector-2: X and Z swapped
    #   Pr[eff. e_x] = Pr[physical Z or Y] = p_Z + p_Y
    #   Pr[eff. e_z] = Pr[physical X or Y] = p_X + p_Y
    px_priors[n_h:] = np.clip(p_Z + p_Y, _EPS, 1 - _EPS)
    pz_priors[n_h:] = np.clip(p_X + p_Y, _EPS, 1 - _EPS)

    return px_priors, pz_priors


def setup_bt_decoders(HX, HZ, p: float, n_h: int,
                      noise_model: str = 'depolarizing', osd_order: int = 0):
    """
    Build BP and BP-OSD decoders for a bias-tailored tile code.

    Parameters
    ----------
    HX, HZ      : original CSS check matrices (not permuted)
    p           : physical error rate
    n_h         : number of sector-1 qubits (sector-2 starts at index n_h)
    noise_model : 'depolarizing' | 'pure_x' | 'pure_z' | 'pure_y'
    osd_order   : 0 -> OSD-0,  >0 -> OSD-CS with that search order

    Returns
    -------
    bp_x, bp_z, osd_x, osd_z
    """
    n = HX.shape[1]
    osd_method_str = 'OSD_0' if osd_order == 0 else 'OSD_CS'

    px_priors, pz_priors = _bt_priors(n, p, n_h, noise_model)

    bp_x = BpDecoder(
        HZ,
        channel_probs=px_priors,
        bp_method='minimum_sum',
        ms_scaling_factor=0.625,
        max_iter=32,
        schedule='serial',
    )
    bp_z = BpDecoder(
        HX,
        channel_probs=pz_priors,
        bp_method='minimum_sum',
        ms_scaling_factor=0.625,
        max_iter=32,
        schedule='serial',
    )
    osd_x = BpOsdDecoder(
        HZ,
        channel_probs=px_priors,
        bp_method='minimum_sum',
        ms_scaling_factor=0.625,
        max_iter=32,
        schedule='serial',
        osd_method=osd_method_str,
        osd_order=osd_order,
    )
    osd_z = BpOsdDecoder(
        HX,
        channel_probs=pz_priors,
        bp_method='minimum_sum',
        ms_scaling_factor=0.625,
        max_iter=32,
        schedule='serial',
        osd_method=osd_method_str,
        osd_order=osd_order,
    )

    return bp_x, bp_z, osd_x, osd_z


def channel_update_x_to_z(rX: np.ndarray, p: float,
                           noise_model: str = 'depolarizing') -> np.ndarray:
    """
    Bayesian channel update (paper Algorithm 2): given the X-decoder output rX,
    compute updated Z-decoder priors that account for Y-error correlations.

    For each qubit j:
        if rX[j] == 0  (no X-error detected):
            pZ_new[j] = p_Z / (1 - p_X - p_Y)
        if rX[j] == 1  (X-error detected):
            pZ_new[j] = p_Y / (p_X + p_Y)

    These are the effective Z-priors; for sector-2 qubits the swap has already
    been applied upstream (in _bt_priors and Channel_BT), so no extra sector
    logic is needed here — the update formula is uniform across all qubits.

    Parameters
    ----------
    rX         : X-decoder output, shape (n,), values 0 or 1
    p          : physical error rate
    noise_model: same model used for the X-decoder

    Returns
    -------
    pz_updated : np.ndarray float, shape (n,), clamped away from 0/1
    """
    if noise_model == 'depolarizing':
        p_X = p_Z = p_Y = p / 3.0
    elif noise_model == 'pure_x':
        p_X, p_Z, p_Y = p, 0.0, 0.0
    elif noise_model == 'pure_z':
        p_X, p_Z, p_Y = 0.0, p, 0.0
    elif noise_model == 'pure_y':
        p_X, p_Z, p_Y = 0.0, 0.0, p
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

    denom_no_x  = 1.0 - p_X - p_Y   # Pr[no X-type error]
    denom_yes_x = p_X + p_Y          # Pr[X-type error]

    # Avoid division by zero for pure channels
    val_if_0 = (p_Z / denom_no_x)  if denom_no_x  > 0 else _EPS
    val_if_1 = (p_Y / denom_yes_x) if denom_yes_x > 0 else _EPS

    pz_updated = np.where(rX == 0, val_if_0, val_if_1)
    return np.clip(pz_updated, _EPS, 1 - _EPS)
