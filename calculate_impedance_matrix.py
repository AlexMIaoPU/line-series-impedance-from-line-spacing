"""
Assume, 

- untransposed distribution lines (self & mutual inductance cannot be separated)
- symmetric conductor (with respect to sending and receiving ends) so that Zab = Zba

given the 

- location coordinates (x, y) relative to a point on the ground, this is used in IEC cim:WirePosition
- resistance of conductors in Ω/m
- gmr in m (if not given can be calculated from radius, r*e^(1/4))

calculate the series impedance matrix of the lines.

This is a simple implementation using Carson's method and Kron reduction.

This is meant to be interfaced with pandapower's line creation

- Alex M

"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MU0  = 4 * np.pi * 1e-7         # H/m   - free-space permeability
EPS0 = 8.854187817e-12          # F/m   - free-space permittivity
FREQ = 60                       # Hz    - country dependent, use 60 for US
RHO_EARTH = 100                 # Ω·m   - Soil resistivity, IEEE Std 80 default

# ═════════════════════════════════════════════════════════════════════════════
# CORE CALCULATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
 
def build_primitive_impedance(pos_x, pos_y, gmr, r_ac, freq=FREQ, rho=RHO_EARTH):
    """
    Carson's equations: primitive impedance matrix Z_prim (Ω/m).
    Works for any n conductors (n=2 for 1-phase+N, n=4 for 3-phase+N).
    """
    omega   = 2 * np.pi * freq
    r_earth = omega * MU0 / 8                       # Ω/m — earth return resistance
    De      = 658.5 * np.sqrt(rho / freq)           # m   — equivalent earth return depth
    k_Z     = omega * MU0 / (2 * np.pi)             # Ω/m — reactance scaling factor
    n       = len(pos_x)
 
    Z = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            if i == j:
                Z[i, i] = (r_ac[i] + r_earth) + 1j * k_Z * np.log(De / gmr[i])
            else:
                D_ij   = np.hypot(pos_x[i] - pos_x[j], pos_y[i] - pos_y[j])
                D_ij_p = np.hypot(pos_x[i] - pos_x[j], pos_y[i] + pos_y[j])
                Z[i, j] = r_earth + 1j * k_Z * np.log(De / D_ij)
    return Z

def kron_reduce(M, elim_idx):
    """
    Kron reduction: eliminate conductor at elim_idx (grounded, V=0).
 
    For impedance Z: Z_red = Z_pp - Z_pn * Z_nn⁻¹ * Z_np
    """
    keep = [i for i in range(M.shape[0]) if i != elim_idx]
    M_pp = M[np.ix_(keep,       keep)]
    M_pn = M[np.ix_(keep,       [elim_idx])]
    M_np = M[np.ix_([elim_idx], keep)]
    M_nn = M[np.ix_([elim_idx], [elim_idx])]
    return M_pp - M_pn @ np.linalg.inv(M_nn) @ M_np


def calc_three_phase(wires, freq=FREQ, rho=RHO_EARTH):
    """Calculate the 3-phase impedance from wires ordered as A, B, C, N.

    Each wire is a mapping with ``location`` as an ``(x, y)`` pair, ``gmr``
    in metres, and ``r_ac`` in ohms per metre.
    """
    if len(wires) != 4:
        raise ValueError("wires must contain four entries ordered as A, B, C, N")

    pos_x  = np.array([wire['location'][0] for wire in wires])
    pos_y  = np.array([wire['location'][1] for wire in wires])
    gmr    = np.array([wire['gmr']         for wire in wires])
    r_ac   = np.array([wire['r_ac']        for wire in wires])
 
    Z_prim_km = build_primitive_impedance(pos_x, pos_y, gmr, r_ac, freq, rho) * 1000
 
    Z_abc = kron_reduce(Z_prim_km, 3)

    return Z_abc