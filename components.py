"""
Component library for compositional PVT.
Critical properties (Tc, Pc, omega), molecular weight, parachors,
and Peng-Robinson binary interaction coefficients (kij).
Sources: Reid-Prausnitz-Poling, Whitson-Brule, Ahmed.
Units: Tc [R], Pc [psia], MW [lb/lb-mol], parachor [-].
"""

import numpy as np

# Tc [°R], Pc [psia], omega, MW [lb/lbmol], parachor
COMPONENTS = {
    "N2":   {"Tc": 227.16, "Pc": 492.84, "omega": 0.0403, "MW":  28.013, "P_ch":  41.0, "Vc": 1.443},
    "CO2":  {"Tc": 547.42, "Pc": 1069.51,"omega": 0.2236, "MW":  44.010, "P_ch":  78.0, "Vc": 1.505},
    "H2S":  {"Tc": 672.12, "Pc": 1299.97,"omega": 0.0827, "MW":  34.082, "P_ch":  80.1, "Vc": 1.564},
    "C1":   {"Tc": 343.00, "Pc": 667.78, "omega": 0.0115, "MW":  16.043, "P_ch":  77.0, "Vc": 1.590},
    "C2":   {"Tc": 549.59, "Pc": 707.78, "omega": 0.0908, "MW":  30.070, "P_ch": 108.0, "Vc": 2.370},
    "C3":   {"Tc": 665.69, "Pc": 616.40, "omega": 0.1454, "MW":  44.097, "P_ch": 150.3, "Vc": 3.250},
    "iC4":  {"Tc": 734.13, "Pc": 527.94, "omega": 0.1760, "MW":  58.123, "P_ch": 181.5, "Vc": 4.208},
    "nC4":  {"Tc": 765.29, "Pc": 550.56, "omega": 0.1928, "MW":  58.123, "P_ch": 189.9, "Vc": 4.080},
    "iC5":  {"Tc": 828.69, "Pc": 490.37, "omega": 0.2273, "MW":  72.150, "P_ch": 225.0, "Vc": 4.899},
    "nC5":  {"Tc": 845.47, "Pc": 488.60, "omega": 0.2510, "MW":  72.150, "P_ch": 231.5, "Vc": 4.870},
    "C6":   {"Tc": 913.27, "Pc": 436.62, "omega": 0.2957, "MW":  86.177, "P_ch": 271.0, "Vc": 5.929},
}

# Symmetric kij matrix (PR). Most hydrocarbon-hydrocarbon = 0.
# Non-zero entries are HC/non-HC interactions (Reid-Prausnitz, GPA).
KIJ_TABLE = {
    ("N2", "CO2"):  -0.020,
    ("N2", "H2S"):   0.130,
    ("N2", "C1"):    0.025,
    ("N2", "C2"):    0.010,
    ("N2", "C3"):    0.090,
    ("N2", "iC4"):   0.095,
    ("N2", "nC4"):   0.095,
    ("N2", "iC5"):   0.100,
    ("N2", "nC5"):   0.100,
    ("N2", "C6"):    0.110,
    ("N2", "C7+"):   0.115,
    ("CO2", "H2S"):  0.097,
    ("CO2", "C1"):   0.103,
    ("CO2", "C2"):   0.130,
    ("CO2", "C3"):   0.135,
    ("CO2", "iC4"):  0.130,
    ("CO2", "nC4"):  0.130,
    ("CO2", "iC5"):  0.125,
    ("CO2", "nC5"):  0.125,
    ("CO2", "C6"):   0.125,
    ("CO2", "C7+"):  0.115,
    ("H2S", "C1"):   0.085,
    ("H2S", "C2"):   0.084,
    ("H2S", "C3"):   0.075,
    ("H2S", "iC4"):  0.050,
    ("H2S", "nC4"):  0.060,
    ("H2S", "iC5"):  0.060,
    ("H2S", "nC5"):  0.065,
    ("H2S", "C6"):   0.050,
    ("H2S", "C7+"):  0.030,
}


def characterize_c7plus(MW_c7, SG_c7):
    """
    Whitson / Riazi-Daubert correlations for C7+ pseudo-component.
    Inputs:
        MW_c7  : molecular weight of C7+ (lb/lbmol), typical 140-300
        SG_c7  : specific gravity (water=1) of C7+, typical 0.78-0.92
    Returns dict with Tc [°R], Pc [psia], omega, MW, parachor, Vc.
    """
    # Riazi-Daubert (1980) for Tb [°R]
    Tb = 4.5579 * MW_c7 ** 0.15178 * SG_c7 ** 0.15427  # in K initially
    Tb_R = Tb * 1.8 * 100  # scale factor — convert Riazi K to °R approx

    # Use Kesler-Lee (1976) — more reliable
    # Tb correlation (Soreide):
    Tb_R = 1928.3 - (1.695e5) * MW_c7 ** -0.03522 * SG_c7 ** 3.266 \
            * np.exp(-4.922e-3 * MW_c7 - 4.7685 * SG_c7 + 3.462e-3 * MW_c7 * SG_c7)

    # Kesler-Lee Tc, Pc
    Tc = (341.7 + 811 * SG_c7
          + (0.4244 + 0.1174 * SG_c7) * Tb_R
          + (0.4669 - 3.2623 * SG_c7) * 1e5 / Tb_R)
    lnPc = (8.3634 - 0.0566 / SG_c7
            - (0.24244 + 2.2898 / SG_c7 + 0.11857 / SG_c7 ** 2) * 1e-3 * Tb_R
            + (1.4685 + 3.648 / SG_c7 + 0.47227 / SG_c7 ** 2) * 1e-7 * Tb_R ** 2
            - (0.42019 + 1.6977 / SG_c7 ** 2) * 1e-10 * Tb_R ** 3)
    Pc = np.exp(lnPc)

    # Kesler-Lee omega (Lee-Kesler with Tbr = Tb/Tc)
    Tbr = Tb_R / Tc
    if Tbr < 0.8:
        omega = (-np.log(Pc / 14.7) - 5.92714 + 6.09648 / Tbr
                 + 1.28862 * np.log(Tbr) - 0.169347 * Tbr ** 6) / \
                (15.2518 - 15.6875 / Tbr - 13.4721 * np.log(Tbr) + 0.43577 * Tbr ** 6)
    else:
        Kw = Tb_R ** (1 / 3) / SG_c7
        omega = -7.904 + 0.1352 * Kw - 0.007465 * Kw ** 2 + 8.359 * Tbr \
                + (1.408 - 0.01063 * Kw) / Tbr

    # Critical volume (Riazi-Daubert)
    Vc = 7.0434e-7 * Tb_R ** 2.3829 * SG_c7 ** -1.683  # ft3/lbmol

    # Parachor (Firoozabadi)
    P_ch = -11.4 + 3.23 * MW_c7 - 0.0022 * MW_c7 ** 2

    return {"Tc": float(Tc), "Pc": float(Pc), "omega": float(omega),
            "MW": float(MW_c7), "P_ch": float(P_ch), "Vc": float(Vc)}


def kij(comp_i, comp_j):
    """Get binary interaction coefficient (symmetric, default 0 for HC-HC)."""
    if comp_i == comp_j:
        return 0.0
    key = (comp_i, comp_j) if (comp_i, comp_j) in KIJ_TABLE else (comp_j, comp_i)
    return KIJ_TABLE.get(key, 0.0)


def get_props(comp_name, c7_props=None):
    """Return component property dict, falling back to C7+ characterization."""
    if comp_name == "C7+" and c7_props is not None:
        return c7_props
    return COMPONENTS[comp_name]
