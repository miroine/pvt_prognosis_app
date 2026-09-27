"""
Lohrenz-Bray-Clark (1964) viscosity for hydrocarbon mixtures.
Inputs:
  comp_names : list of component names
  comp_frac  : mole fractions in the phase (x or y)
  density    : phase density [lb/ft3]
  T          : [°R]
  c7_props   : optional dict for C7+
Output:
  viscosity  [cP]
"""

import numpy as np
from components import get_props


def lbc_viscosity(comp_names, comp_frac, rho, T, c7_props=None):
    n = len(comp_names)
    Tc = np.array([get_props(c, c7_props)["Tc"] for c in comp_names])
    Pc = np.array([get_props(c, c7_props)["Pc"] for c in comp_names])
    MW = np.array([get_props(c, c7_props)["MW"] for c in comp_names])
    Vc = np.array([get_props(c, c7_props)["Vc"] for c in comp_names])  # ft3/lbmol

    x = np.asarray(comp_frac, dtype=float); x = x / x.sum()

    # Dilute-gas viscosity, Stiel-Thodos
    Tr = T / Tc
    # zeta_i = Tc^(1/6) / (MW^(1/2) * Pc^(2/3))   (Field-unit version)
    zeta = Tc ** (1 / 6) / (MW ** 0.5 * Pc ** (2 / 3))
    mu_star = np.where(Tr <= 1.5,
                       3.4e-4 * Tr ** 0.94 / zeta,
                       1.778e-4 * (4.58 * Tr - 1.67) ** (5 / 8) / zeta)

    # Mixture dilute-gas viscosity (Herning-Zipperer)
    sqrtMW = np.sqrt(MW)
    mu_mix_star = np.sum(x * mu_star * sqrtMW) / np.sum(x * sqrtMW)

    # Mixture parameters
    Tcm = float(np.dot(x, Tc))
    Pcm = float(np.dot(x, Pc))
    MWm = float(np.dot(x, MW))
    Vcm = float(np.dot(x, Vc))

    zeta_m = Tcm ** (1 / 6) / (MWm ** 0.5 * Pcm ** (2 / 3))

    # Reduced density
    rho_m = rho / MWm           # lbmol/ft3
    rho_r = rho_m * Vcm

    # LBC polynomial
    poly = (0.1023 + 0.023364 * rho_r + 0.058533 * rho_r ** 2
            - 0.040758 * rho_r ** 3 + 0.0093324 * rho_r ** 4)
    mu = mu_mix_star + (poly ** 4 - 1e-4) / zeta_m
    return float(max(mu, 1e-5))
