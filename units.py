"""
Unit conversion between Field and SI for PVT calculations.

Internal calculations always use FIELD units (psia, °F, scf/STB, rb/STB, cp, lb/ft3).
This module converts user inputs from SI -> Field on entry,
and Field -> SI on display.

SI conventions used here (petroleum-industry SI, not strict SI):
    Pressure     bara              (1 bara = 14.5038 psia)
    Temperature  °C
    GOR / Rs     Sm3/Sm3            (1 Sm3/Sm3 = 5.6146 scf/STB)
    Rv / CGR     Sm3/Sm3
    Bo           rm3/Sm3 = Bo (rb/STB) (dimensionless, identical)
    Bg           rm3/Sm3
    viscosity    cP                 (same in both systems)
    density      kg/m3              (1 lb/ft3 = 16.0185 kg/m3)
    salinity     ppm (same)
"""

# Conversion factors (multiply field -> SI; divide for reverse)
PSIA_PER_BAR = 14.50377
F_TO_C_SCALE = 5.0 / 9.0
SCF_PER_SM3 = 35.3147 / 1.0  # actually 1 Sm3 = 35.3147 scf? No: 1 Sm3 ≈ 35.314 scf
LB_FT3_PER_KG_M3 = 1.0 / 16.01846

# Better: explicit conversion functions
def psia_to_bar(p):  return p / PSIA_PER_BAR
def bar_to_psia(p):  return p * PSIA_PER_BAR
def F_to_C(T):        return (T - 32.0) * F_TO_C_SCALE
def C_to_F(T):        return T / F_TO_C_SCALE + 32.0
def scfSTB_to_Sm3Sm3(R): return R / 5.6146     # 1 Sm3/Sm3 = 5.6146 scf/STB
def Sm3Sm3_to_scfSTB(R): return R * 5.6146
def lbft3_to_kgm3(rho):  return rho * 16.01846
def kgm3_to_lbft3(rho):  return rho / 16.01846
def rbMscf_to_rm3Sm3(B): return B / 5.6146     # Bg conversion same factor

UNIT_LABELS = {
    "Field": {
        "P": "psia", "T": "°F", "Rs": "scf/STB", "Bo": "rb/STB",
        "Bg": "rb/Mscf", "Rv": "STB/Mscf",
        "mu": "cP", "rho": "lb/ft³", "Cw": "1/psi",
    },
    "SI": {
        "P": "bara", "T": "°C", "Rs": "Sm³/Sm³", "Bo": "rm³/Sm³",
        "Bg": "rm³/Sm³", "Rv": "Sm³/Sm³",
        "mu": "cP", "rho": "kg/m³", "Cw": "1/bar",
    },
}


def label(units, key):
    return UNIT_LABELS[units][key]


def to_field_P(P_user, units):
    return bar_to_psia(P_user) if units == "SI" else P_user

def to_user_P(P_field, units):
    return psia_to_bar(P_field) if units == "SI" else P_field

def to_field_T(T_user, units):
    return C_to_F(T_user) if units == "SI" else T_user

def to_user_T(T_field, units):
    return F_to_C(T_field) if units == "SI" else T_field

def to_field_Rs(R_user, units):
    return Sm3Sm3_to_scfSTB(R_user) if units == "SI" else R_user

def to_user_Rs(R_field, units):
    return scfSTB_to_Sm3Sm3(R_field) if units == "SI" else R_field

def to_user_Bg(Bg_field_rbMscf, units):
    """Bg from rb/Mscf (Field) to rm3/Sm3 (SI)."""
    return rbMscf_to_rm3Sm3(Bg_field_rbMscf) if units == "SI" else Bg_field_rbMscf

def to_user_rho(rho_field, units):
    return lbft3_to_kgm3(rho_field) if units == "SI" else rho_field

def to_user_Cw(Cw_field, units):
    # 1/psi -> 1/bar:  multiply by 14.5038
    return Cw_field * PSIA_PER_BAR if units == "SI" else Cw_field
