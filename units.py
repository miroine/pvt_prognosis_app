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

def to_field_Cw(Cw_user, units):
    """Reverse: Cw from 1/bar (SI) to 1/psi (field)."""
    return Cw_user / PSIA_PER_BAR if units == "SI" else Cw_user


def to_field_rho(rho_user, units):
    """Reverse: kg/m3 -> lb/ft3."""
    return kgm3_to_lbft3(rho_user) if units == "SI" else rho_user


def to_field_Bg(Bg_user, units):
    """Reverse: rm3/Sm3 (SI) -> rb/Mscf (field)."""
    return Bg_user * 5.6146 if units == "SI" else Bg_user


def to_field_Rv(Rv_user, units):
    """Reverse: Sm3/Sm3 (SI) -> STB/Mscf (field)."""
    return Rv_user * 5.6146 if units == "SI" else Rv_user


def to_user_Rv(Rv_field_STBMscf, units):
    """Field STB/Mscf -> SI Sm3/Sm3."""
    return Rv_field_STBMscf / 5.6146 if units == "SI" else Rv_field_STBMscf


# ----------------------------------------------------------------
# CGR (condensate-gas ratio) conversion
# ----------------------------------------------------------------
# The app's internal model (WetGasCorrelations) requires CGR strictly in
# STB/MMscf. The SI display unit is Sm3/MSm3. The conversion uses the same
# 5.6146 factor the rest of the GOR-family conversions use, so CGR scales
# consistently with Rs and Rv. Centralizing it here means the factor is
# defined in exactly one place.
def to_field_cgr(cgr_user, units):
    """CGR from the display unit to the internal field unit (STB/MMscf)."""
    return cgr_user * 5.6146 if units == "SI" else cgr_user


def to_user_cgr(cgr_field, units):
    """CGR from the internal field unit (STB/MMscf) to the display unit."""
    return cgr_field / 5.6146 if units == "SI" else cgr_field


# ----------------------------------------------------------------
# ΔT conversion (temperature delta, not absolute T)
# ----------------------------------------------------------------
def to_field_deltaT(dT_user, units):
    """Convert a temperature DIFFERENCE from user to field units.
    Uses the scale factor 9/5 only — NO 32° offset."""
    return dT_user / F_TO_C_SCALE if units == "SI" else dT_user

def to_user_deltaT(dT_field, units):
    return dT_field * F_TO_C_SCALE if units == "SI" else dT_field


# ----------------------------------------------------------------
# Robust lab-measurement conversion for tuning workflows
# ----------------------------------------------------------------
# Each lab measurement has 'type', 'P' (display units, except for Pb/Pdew),
# and 'value' (display units appropriate to the type).
#
# These two helpers do the conversions consistently across all branches.
# ----------------------------------------------------------------

def lab_to_field(lab_list, units):
    """Convert a list of display-unit lab measurements to FIELD units.

    Supported types:
        - 'Pb', 'Pdew': value is pressure (psia/bara), P is ignored
        - 'Rs':         value is Rs (scf/STB or Sm3/Sm3), P is pressure
        - 'Rv':         value is Rv (STB/Mscf or Sm3/Sm3), P is pressure
        - 'Bo', 'Bg':   value is dimensionless ratio (same in both systems)
        - 'mu_o', 'mu_g': value is viscosity in cP (same in both systems)
        - 'Z':          dimensionless
    """
    out = []
    for m in lab_list:
        t = m.get("type", "")
        fm = {"type": t, "weight": m.get("weight", 1.0)}
        v = float(m.get("value", 0.0))
        P = float(m.get("P", 0.0))
        if t in ("Pb", "Pdew"):
            fm["P"] = 0.0
            fm["value"] = to_field_P(v, units)
        elif t == "Rs":
            fm["P"] = to_field_P(P, units)
            fm["value"] = to_field_Rs(v, units)
        elif t == "Rv":
            # Note: Field Rv unit is STB/Mscf, SI is Sm3/Sm3
            fm["P"] = to_field_P(P, units)
            fm["value"] = to_field_Rv(v, units)
        elif t in ("Bo", "Bg", "Z", "mu_o", "mu_g"):
            # Dimensionless or cP - same in both systems
            fm["P"] = to_field_P(P, units)
            fm["value"] = v
        else:
            # Unknown type - pass through with P conversion
            fm["P"] = to_field_P(P, units)
            fm["value"] = v
        out.append(fm)
    return out


def field_pred_to_user(pred_array, lab_list, units):
    """Convert predictions from FIELD units back to display units.
    Uses the lab_list types to know which conversion to apply to each prediction.
    """
    import numpy as np
    out = []
    for val, m in zip(pred_array, lab_list):
        t = m.get("type", "")
        if t in ("Pb", "Pdew"):
            out.append(to_user_P(val, units))
        elif t == "Rs":
            out.append(to_user_Rs(val, units))
        elif t == "Rv":
            out.append(to_user_Rv(val, units))
        else:
            out.append(val)
    return np.array(out)


# ----------------------------------------------------------------------
# Flowline / thermal-hydraulic unit conversions
# ----------------------------------------------------------------------
# All flowline physics in hydrate.py works in FIELD units (inch, ft,
# psia, F, lb/ft3, cP, BTU/(hr.ft2.F), STB/d, Mscf/d). These helpers let
# the UI accept and display SI equivalents.

MM_PER_INCH = 25.4
FT_PER_M = 3.280839895
# U-value: 1 BTU/(hr.ft2.F) = 5.678263 W/(m2.K)
WM2K_PER_BTUHRFT2F = 5.678263
# Heat capacity: 1 BTU/(lb.F) = 4.1868 kJ/(kg.K)
KJKGK_PER_BTULBF = 4.1868
# Gas rate: 1 Mscf = 28.31685 Sm3 (standard cubic metres)
SM3_PER_MSCF = 28.31685
# Liquid rate: 1 STB = 0.1589873 Sm3
SM3_PER_STB = 0.1589873


def to_field_diameter(d_user, units):
    """Pipe diameter: mm (SI) or inch (Field) -> inch."""
    return d_user / MM_PER_INCH if units == "SI" else d_user

def to_user_diameter(d_inch, units):
    """inch -> mm (SI) or inch (Field)."""
    return d_inch * MM_PER_INCH if units == "SI" else d_inch


def to_field_length(L_user, units):
    """Pipe length: m (SI) or ft (Field) -> ft."""
    return L_user * FT_PER_M if units == "SI" else L_user

def to_user_length(L_ft, units):
    """ft -> m (SI) or ft (Field)."""
    return L_ft / FT_PER_M if units == "SI" else L_ft


def to_field_Uvalue(U_user, units):
    """Heat-transfer coefficient: W/(m2.K) (SI) or BTU/(hr.ft2.F)
    (Field) -> BTU/(hr.ft2.F)."""
    return U_user / WM2K_PER_BTUHRFT2F if units == "SI" else U_user

def to_user_Uvalue(U_field, units):
    """BTU/(hr.ft2.F) -> W/(m2.K) (SI) or BTU/(hr.ft2.F) (Field)."""
    return U_field * WM2K_PER_BTUHRFT2F if units == "SI" else U_field


def to_field_cp(cp_user, units):
    """Heat capacity: kJ/(kg.K) (SI) or BTU/(lb.F) (Field) ->
    BTU/(lb.F)."""
    return cp_user / KJKGK_PER_BTULBF if units == "SI" else cp_user

def to_user_cp(cp_field, units):
    """BTU/(lb.F) -> kJ/(kg.K) (SI) or BTU/(lb.F) (Field)."""
    return cp_field * KJKGK_PER_BTULBF if units == "SI" else cp_field


def to_field_qgas(q_user, units):
    """Gas rate: Sm3/d (SI) or Mscf/d (Field) -> Mscf/d."""
    return q_user / SM3_PER_MSCF if units == "SI" else q_user

def to_user_qgas(q_mscfd, units):
    """Mscf/d -> Sm3/d (SI) or Mscf/d (Field)."""
    return q_mscfd * SM3_PER_MSCF if units == "SI" else q_mscfd


def to_field_qliq(q_user, units):
    """Liquid rate: Sm3/d (SI) or STB/d (Field) -> STB/d."""
    return q_user / SM3_PER_STB if units == "SI" else q_user

def to_user_qliq(q_stbd, units):
    """STB/d -> Sm3/d (SI) or STB/d (Field)."""
    return q_stbd * SM3_PER_STB if units == "SI" else q_stbd


# Short unit-label helpers for the flowline UI.
def flowline_labels(units):
    """Return a dict of display unit strings for the active system."""
    if units == "SI":
        return {"D": "mm", "L": "m", "U": "W/(m²·K)",
                "cp": "kJ/(kg·K)", "rho": "kg/m³", "mu": "cP",
                "qgas": "Sm³/d", "qliq": "Sm³/d", "v": "m/s"}
    return {"D": "inch", "L": "ft", "U": "BTU/(hr·ft²·°F)",
            "cp": "BTU/(lb·°F)", "rho": "lb/ft³", "mu": "cP",
            "qgas": "Mscf/d", "qliq": "STB/d", "v": "ft/s"}
