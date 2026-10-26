"""
PVT Studio — Nodal Analysis & Lift Curves
==========================================

Multiphase pressure-drop along a tubing or flowline, marched stepwise
from outlet to inlet to build a Tubing/Flowline Performance Curve
(inlet pressure required vs. flow rate). Inputs come from the UI in
field units.

Correlations implemented:

  * Beggs & Brill (1973, revised 1979) — the workhorse correlation for
    horizontal/inclined/uphill/downhill pipes carrying any fluid mix.
    Implements all four flow patterns (segregated, intermittent,
    distributed, transition), the inclination-angle correction, the
    Payne et al. (1979) holdup correction factor, and a no-slip
    friction factor with a two-phase multiplier.

  * Hagedorn & Brown (1965) with Griffith bubble-flow correction — a
    correlation tuned for VERTICAL flow of oil + gas, especially at
    high liquid loading where slug flow dominates. The Griffith
    correction handles the bubble-flow regime that H-B alone gets
    wrong at low gas rates.

  * Gray (1974) — specifically for VERTICAL wet-gas / gas-condensate
    wells where the gas dominates and a modest amount of liquid is
    carried up as mist or film. Simpler than B-B, more accurate for
    its target regime.

  * Duns-Ros and Orkiszewski are NOT implemented (they are large,
    multi-regime correlations and a partial implementation would be
    worse than B-B). The UI recommends them where they would apply
    but reports which correlation actually ran.

The march
---------
The pipe is divided into N segments. We march from the OUTLET (where
pressure is known — wellhead or separator) toward the INLET, computing
the local pressure gradient (dP/dL) at each segment from the current
P, T and fluid properties and integrating. Temperature is linearly
interpolated between the inlet-T and outlet-T inputs (a simple but
robust assumption — a coupled thermal model is the next step but is
out of scope here).

Caveats
-------
Mechanistic two-phase models (PIPESIM-style) are more accurate than
the empirical correlations here, especially in transition regions.
These results are SCREENING quality — use them to size a turndown
limit, find a stable operating point, or compare lift options, not as
the final design number.
"""

APP_VERSION = "1.4.7"   # must match pvt_app.py (deployment check)

import math
import numpy as np


# ---- Constants ----
G_C = 32.174              # ft/s^2
PSI_TO_PSF = 144.0


# ----------------------------------------------------------------------
# HELPERS — basic in-situ properties
# ----------------------------------------------------------------------
def _superficial_velocities(q_liq_bbl_d, q_gas_mscf_d, D_in_inch,
                              P_psia, T_F, Z_gas, Bo_rb_stb):
    """Superficial liquid and gas velocities at LINE conditions, ft/s.

    Liquid is taken at reservoir conditions via Bo (so the in-situ
    liquid rate is Bo*q_stb). Gas is converted from standard to actual
    via PV=ZnRT.
    """
    D_ft = D_in_inch / 12.0
    area = math.pi * D_ft ** 2 / 4.0
    # Liquid (STB/d -> rb/d -> ft3/s)
    q_l_ft3_s = q_liq_bbl_d * Bo_rb_stb * 5.6146 / 86400.0
    # Gas — convert standard to actual
    T_R = T_F + 459.67
    P_sc, T_sc = 14.696, 519.67
    q_g_std_ft3_s = q_gas_mscf_d * 1000.0 / 86400.0
    q_g_ft3_s = (q_g_std_ft3_s * (P_sc / max(P_psia, 1.0))
                  * (T_R / T_sc) * max(Z_gas, 0.1))
    v_sl = q_l_ft3_s / area if area > 0 else 0.0
    v_sg = q_g_ft3_s / area if area > 0 else 0.0
    return v_sl, v_sg


def _haaland_friction(reynolds, rel_roughness):
    """Darcy friction factor (Haaland explicit approximation)."""
    if reynolds <= 0:
        return 0.02
    if reynolds < 2300.0:
        return 64.0 / reynolds
    inv_sqrt = -1.8 * math.log10(
        (rel_roughness / 3.7) ** 1.11 + 6.9 / reynolds)
    if inv_sqrt == 0:
        return 0.02
    return (1.0 / inv_sqrt) ** 2


def _gas_density_at(P_psia, T_F, gas_sg, Z):
    """Real-gas density at P,T from EOS, lb/ft3."""
    T_R = T_F + 459.67
    # M = 28.964 * SG; PV = nZRT -> rho = P M / (Z R T)
    M = 28.9647 * gas_sg
    R = 10.7316    # psia.ft3/(lbmol.R)
    return P_psia * M / (max(Z, 0.1) * R * T_R)


def _gas_z_simple(P_psia, T_F, gas_sg):
    """A simple Hall-Yarborough-style Z; screening accuracy only.

    Uses Standing-Katz pseudo-criticals and a Brill-Beggs explicit fit.
    """
    # Pseudo-critical (Sutton, sweet)
    Tpc = 169.2 + 349.5 * gas_sg - 74.0 * gas_sg ** 2
    Ppc = 756.8 - 131.0 * gas_sg - 3.6 * gas_sg ** 2
    T_R = T_F + 459.67
    Tpr = T_R / Tpc
    Ppr = P_psia / Ppc
    # Brill-Beggs explicit correlation
    A = 1.39 * (Tpr - 0.92) ** 0.5 - 0.36 * Tpr - 0.101
    B = ((0.62 - 0.23 * Tpr) * Ppr
          + (0.066 / max(Tpr - 0.86, 0.05) - 0.037) * Ppr ** 2
          + 0.32 * Ppr ** 6 / max(10 ** min(9 * (Tpr - 1), 300.0), 1e-12))
    C = 0.132 - 0.32 * math.log10(Tpr)
    D = 10 ** (0.3106 - 0.49 * Tpr + 0.1824 * Tpr ** 2)
    Z = A + (1 - A) / math.exp(min(B, 50.0)) + C * Ppr ** D
    return max(0.3, min(Z, 1.4))


def _gas_viscosity_lee(P_psia, T_F, gas_sg, Z):
    """Lee-Gonzalez-Eakin gas viscosity, cP."""
    T_R = T_F + 459.67
    M = 28.9647 * gas_sg
    rho_g = _gas_density_at(P_psia, T_F, gas_sg, Z)   # lb/ft3
    rho_cgs = rho_g * 16.018 / 1000.0   # g/cm3
    K = (9.4 + 0.02 * M) * T_R ** 1.5 / (209.0 + 19.0 * M + T_R)
    X = 3.5 + 986.0 / T_R + 0.01 * M
    Y = 2.4 - 0.2 * X
    return 1e-4 * K * math.exp(X * rho_cgs ** Y)


def _oil_viscosity_simple(api, T_F):
    """Beggs-Robinson dead-oil viscosity, cP. A screening estimate."""
    T_F = max(T_F, 50.0)
    Z = 3.0324 - 0.02023 * api
    Y = 10 ** Z
    X = Y * T_F ** (-1.163)
    return max(0.1, 10 ** X - 1.0)


def _oil_density(api, gas_sg, P_psia, Rs_scf_stb, Bo_rb_stb):
    """Live-oil density, lb/ft3 (mass balance from SG_oil, Rs, Bo)."""
    rho_o_stock = 141.5 / (api + 131.5) * 62.428    # lb/ft3 stock-tank
    # Mass of stock-tank oil + mass of dissolved gas per STB, in lb,
    # divided by reservoir volume Bo*5.615 ft3/STB.
    mass_per_stb = rho_o_stock * 5.6146 + Rs_scf_stb * gas_sg * 0.0764
    vol_per_stb = Bo_rb_stb * 5.6146
    return mass_per_stb / max(vol_per_stb, 0.01)


def _water_density(salinity_wt_pct, T_F):
    """Brine density, lb/ft3 (McCain-style simple form)."""
    rho_w_pure = 62.4 * (1.0 - 0.0001 * (T_F - 60.0))
    return rho_w_pure * (1.0 + 0.00747 * salinity_wt_pct)


# ----------------------------------------------------------------------
# BEGGS & BRILL (revised) — horizontal/inclined two-phase ΔP
# ----------------------------------------------------------------------
def _beggs_brill_holdup(v_sl, v_sg, D_ft, theta_deg, rho_l, sigma=30.0):
    """Beggs-Brill in-situ liquid holdup with inclination correction.

    Returns a dict with no-slip holdup, flow pattern, holdup at
    horizontal, the inclination correction factor C and B(angle), and
    the final inclined holdup.
    """
    v_m = v_sl + v_sg
    if v_m <= 0:
        return {"lambda_l": 0.0, "pattern": "no-flow",
                "H_L0": 0.0, "C": 0.0, "Bangle": 1.0, "H_L": 0.0,
                "Fr": 0.0, "v_m": 0.0}
    lambda_l = v_sl / v_m
    Fr = v_m ** 2 / (G_C * D_ft)
    # Flow-pattern boundaries (Beggs-Brill 1973)
    L1 = 316.0 * lambda_l ** 0.302
    L2 = 0.0009252 * lambda_l ** -2.4684
    L3 = 0.10 * lambda_l ** -1.4516
    L4 = 0.5 * lambda_l ** -6.738
    # (A blanket "lambda < 0.01 -> segregated" rule used to sit here; the
    # published map makes very dry flow DISTRIBUTED once Fr >= L1.)
    if (lambda_l < 0.01 and Fr < L1) or (lambda_l >= 0.01 and Fr < L2):
        pattern = "segregated"
    elif lambda_l >= 0.01 and L2 <= Fr <= L3:
        pattern = "transition"
    elif ((0.01 <= lambda_l < 0.4 and L3 < Fr <= L1)
           or (lambda_l >= 0.4 and L3 < Fr <= L4)):
        pattern = "intermittent"
    else:
        pattern = "distributed"
    # H_L(0) — horizontal holdup
    coefs = {"segregated": (0.98, 0.4846, 0.0868),
             "intermittent": (0.845, 0.5351, 0.0173),
             "distributed": (1.065, 0.5824, 0.0609)}
    if pattern in coefs:
        a, b, c = coefs[pattern]
        H_L0 = a * lambda_l ** b / max(Fr, 1e-6) ** c
    elif pattern == "transition":
        a1, b1, c1 = coefs["segregated"]
        a2, b2, c2 = coefs["intermittent"]
        H_seg = a1 * lambda_l ** b1 / max(Fr, 1e-6) ** c1
        H_int = a2 * lambda_l ** b2 / max(Fr, 1e-6) ** c2
        A = (L3 - Fr) / (L3 - L2) if L3 != L2 else 0.5
        A = max(0.0, min(A, 1.0))
        H_L0 = A * H_seg + (1.0 - A) * H_int
    else:
        H_L0 = lambda_l
    H_L0 = max(H_L0, lambda_l)   # cannot be less than no-slip
    # Liquid velocity number for the inclination correction. Field form
    # (ft/s, lb/ft3, dyn/cm): Nlv = 1.938 v_sl (rho_l / sigma)^0.25 — the
    # same number used by H-B and M-B below. The previous expression
    # divided by g_c*sigma*1e-3 and came out ~22 % too high.
    Nlv = 1.938 * v_sl * (rho_l / max(sigma, 1e-3)) ** 0.25
    # C and B(angle) by flow pattern
    if theta_deg >= 0.0:
        # Uphill
        if pattern == "segregated":
            d_, e_, f_, g_ = 0.011, -3.768, 3.539, -1.614
        elif pattern == "intermittent":
            d_, e_, f_, g_ = 2.96, 0.305, -0.4473, 0.0978
        elif pattern == "distributed":
            d_, e_, f_, g_ = 0.0, 0.0, 0.0, 0.0   # C = 0
        else:
            d_, e_, f_, g_ = 0.011, -3.768, 3.539, -1.614
    else:
        # Downhill — all patterns use the same C
        d_, e_, f_, g_ = 4.70, -0.3692, 0.1244, -0.5056
    if d_ == 0.0:
        C = 0.0
    else:
        try:
            C = (1.0 - lambda_l) * math.log(
                max(d_ * lambda_l ** e_ * Nlv ** f_
                     * max(Fr, 1e-6) ** g_, 1e-12))
        except (ValueError, OverflowError):
            C = 0.0
    C = max(C, 0.0)
    theta_rad = math.radians(theta_deg)
    Bangle = 1.0 + C * (math.sin(1.8 * theta_rad)
                         - (1.0 / 3.0) * math.sin(1.8 * theta_rad) ** 3)
    H_L = H_L0 * Bangle
    H_L = max(min(H_L, 1.0), lambda_l)
    return {"lambda_l": lambda_l, "pattern": pattern, "H_L0": H_L0,
            "C": C, "Bangle": Bangle, "H_L": H_L, "Fr": Fr, "v_m": v_m}


def _beggs_brill_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g,
                       D_in_inch, theta_deg, rel_roughness, P_psia,
                       sigma=30.0):
    """Beggs-Brill pressure gradient (psi/ft) for a pipe segment.

    Returns a dict with gradient components and the holdup info.
    """
    D_ft = D_in_inch / 12.0
    hd = _beggs_brill_holdup(v_sl, v_sg, D_ft, theta_deg, rho_l, sigma)
    H_L = hd["H_L"]
    lam = hd["lambda_l"]
    v_m = hd["v_m"]
    rho_ns = rho_l * lam + rho_g * (1.0 - lam)
    rho_s = rho_l * H_L + rho_g * (1.0 - H_L)
    mu_ns = mu_l * lam + mu_g * (1.0 - lam)
    # Two-phase Reynolds (no-slip)
    Re_ns = 1488.0 * rho_ns * v_m * D_ft / max(mu_ns, 1e-9)
    f_n = _haaland_friction(Re_ns, rel_roughness)
    # Two-phase friction factor multiplier
    if H_L > lam > 0:
        # Beggs & Brill (1973): S = ln(2.2y - 1.2) for 1 < y < 1.2
        # (the general fit's denominator vanishes near y = 1.017), the
        # general fit elsewhere. (The branches were swapped, and the
        # max(denom, 1e-6) guard turned a negative denominator into
        # exp(huge) -> "math range error" -> NaN points on the VLP.)
        y_term = lam / (H_L ** 2)
        if 1.0 < y_term < 1.2:
            s = math.log(2.2 * y_term - 1.2)
        else:
            x = math.log(max(y_term, 1e-6))
            denom = (-0.0523 + 3.182 * x - 0.8725 * x ** 2
                      + 0.01853 * x ** 4)
            s = x / denom if abs(denom) > 1e-9 else 0.0
        f_tp = f_n * math.exp(min(s, 5.0))
    else:
        f_tp = f_n
    # Pressure-gradient components (psi/ft)
    # Friction: dP/dL = f * rho_ns * v_m^2 / (2 * g_c * D)  -> lb/ft3 * ft/s^2 = lb/(ft^2.s^2)/gc -> lbf/ft3
    dpdL_fric_psf_ft = f_tp * rho_ns * v_m ** 2 / (2.0 * G_C * D_ft)
    dpdL_fric = dpdL_fric_psf_ft / PSI_TO_PSF
    # Hydrostatic
    dpdL_hydro = rho_s * math.sin(math.radians(theta_deg)) / PSI_TO_PSF
    # Acceleration — small at most conditions, included for completeness
    Ek = v_m * v_sg * rho_ns / (G_C * max(P_psia, 1.0) * PSI_TO_PSF)
    Ek = min(Ek, 0.95)
    dpdL_total = (dpdL_fric + dpdL_hydro) / max(1.0 - Ek, 0.05)
    return {"dpdL_psi_ft": dpdL_total,
            "dpdL_fric": dpdL_fric, "dpdL_hydro": dpdL_hydro,
            "Ek": Ek, "H_L": H_L, "lambda_l": lam,
            "pattern": hd["pattern"], "rho_s": rho_s, "v_m": v_m,
            "Fr": hd["Fr"], "Re_ns": Re_ns, "f_tp": f_tp}


# ----------------------------------------------------------------------
# HAGEDORN-BROWN (with Griffith bubble correction) — VERTICAL pipes
# ----------------------------------------------------------------------
def _griffith_bubble(v_sl, v_sg, D_ft):
    """Return True if Griffith bubble-flow correction should be used.

    Bubble flow exists when the gas fraction is small AND velocities are
    low. The Griffith test compares L_B = max(1.071 - 0.2218*v_m^2/D, 0.13)
    to the in-situ gas volume fraction.
    """
    v_m = v_sl + v_sg
    if v_m <= 0:
        return False
    L_B = max(1.071 - 0.2218 * v_m ** 2 / D_ft, 0.13)
    lambda_g = v_sg / v_m
    return lambda_g < L_B


def _hagedorn_brown_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g,
                          D_in_inch, theta_deg, rel_roughness,
                          P_psia, sigma=30.0):
    """Hagedorn-Brown vertical-flow gradient (psi/ft).

    Uses a Griffith bubble-flow correction at low gas rates and the
    classic H-B holdup correlation for slug/mist flow above it. The
    holdup is a function of four dimensionless numbers (Nlv, Ngv, Nd,
    Nl); we use simplified curve fits.

    For an inclined or horizontal pipe, falls back to a Beggs-Brill
    call (this is what commercial tools do).
    """
    if abs(theta_deg - 90.0) > 30.0:
        # Not vertical enough — defer to B-B
        return _beggs_brill_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g,
                                  D_in_inch, theta_deg, rel_roughness,
                                  P_psia, sigma)
    D_ft = D_in_inch / 12.0
    v_m = v_sl + v_sg
    if v_m <= 0:
        return {"dpdL_psi_ft": rho_l / PSI_TO_PSF,
                "dpdL_fric": 0.0, "dpdL_hydro": rho_l / PSI_TO_PSF,
                "Ek": 0.0, "H_L": 1.0, "lambda_l": 1.0,
                "pattern": "no-flow", "rho_s": rho_l, "v_m": 0.0,
                "Fr": 0.0, "Re_ns": 0.0, "f_tp": 0.0}
    # Griffith bubble-flow regime
    if _griffith_bubble(v_sl, v_sg, D_ft):
        # Griffith slip velocity ~ 0.8 ft/s
        v_s = 0.8
        # In-situ liquid holdup from Griffith
        H_L = 1.0 - 0.5 * (1.0 + v_m / v_s
                            - math.sqrt((1.0 + v_m / v_s) ** 2
                                         - 4.0 * v_sg / v_s))
        H_L = max(min(H_L, 1.0), 1e-3)
        rho_s = rho_l * H_L + rho_g * (1.0 - H_L)
        v_l_actual = v_sl / max(H_L, 1e-3)
        Re_l = 1488.0 * rho_l * v_l_actual * D_ft / max(mu_l, 1e-9)
        f = _haaland_friction(Re_l, rel_roughness)
        dpdL_fric = f * rho_l * v_l_actual ** 2 / (2.0 * G_C * D_ft) / PSI_TO_PSF
        pattern = "bubble (Griffith)"
    else:
        # H-B regime — slug / mist
        # Dimensionless numbers
        Nlv = 1.938 * v_sl * (rho_l / sigma) ** 0.25
        Ngv = 1.938 * v_sg * (rho_l / sigma) ** 0.25
        Nd = 120.872 * D_ft * (rho_l / sigma) ** 0.5
        Nl = 0.15726 * mu_l * (1.0 / (rho_l * sigma ** 3)) ** 0.25
        # Rational curve fits of the Hagedorn-Brown charts (Trick, 2003):
        # CNL(NL), HL/psi(X) and psi(Y). The earlier quartic-in-log fit
        # went negative below X ~ 1e-4 and clamped H_L to the no-slip
        # value, so H-B collapsed onto the homogeneous model.
        CNl = ((0.0019 + 0.0322 * Nl - 0.6642 * Nl ** 2 + 4.9951 * Nl ** 3)
               / (1.0 - 10.0147 * Nl + 33.8696 * Nl ** 2
                  + 277.2817 * Nl ** 3))
        CNl = max(min(CNl, 0.02), 0.0019)
        group = Nlv / max(Ngv ** 0.575, 1e-6) * (P_psia / 14.7) ** 0.1 \
                * CNl / max(Nd, 1e-6)
        X = max(group, 0.0)
        HL_psi = math.sqrt(max((0.0047 + 1123.32 * X + 729489.64 * X ** 2)
                               / (1.0 + 1097.1566 * X + 722153.97 * X ** 2),
                               0.0))
        HL_psi = max(min(HL_psi, 1.0), 0.0)
        group2 = Ngv * Nl ** 0.380 / max(Nd ** 2.14, 1e-6)
        Yp = min(max(group2, 0.0), 0.09)
        psi = ((1.0886 - 69.9473 * Yp + 2334.3497 * Yp ** 2
                - 12896.683 * Yp ** 3)
               / (1.0 - 53.4401 * Yp + 1517.9369 * Yp ** 2
                  - 8419.8115 * Yp ** 3))
        psi = max(min(psi, 1.82), 1.0)
        H_L = HL_psi * psi
        H_L = max(min(H_L, 1.0), v_sl / max(v_m, 1e-6))
        rho_s = rho_l * H_L + rho_g * (1.0 - H_L)
        rho_ns = rho_l * (v_sl / v_m) + rho_g * (v_sg / v_m)
        # Friction uses no-slip Reynolds with mixture properties
        mu_ns = mu_l * (v_sl / v_m) + mu_g * (v_sg / v_m)
        # H-B uses rho_ns^2 / rho_s in the friction term
        Re = 1488.0 * rho_ns * v_m * D_ft / max(mu_ns, 1e-9)
        f = _haaland_friction(Re, rel_roughness)
        dpdL_fric = f * rho_ns ** 2 * v_m ** 2 \
            / (2.0 * G_C * D_ft * max(rho_s, 0.1)) / PSI_TO_PSF
        pattern = "slug/mist (H-B)"
    # Hydrostatic (vertical: sin = 1)
    dpdL_hydro = rho_s * math.sin(math.radians(theta_deg)) / PSI_TO_PSF
    # Acceleration
    Ek = v_m * v_sg * rho_s / (G_C * max(P_psia, 1.0) * PSI_TO_PSF)
    Ek = min(Ek, 0.95)
    dpdL_total = (dpdL_fric + dpdL_hydro) / max(1.0 - Ek, 0.05)
    return {"dpdL_psi_ft": dpdL_total,
            "dpdL_fric": dpdL_fric, "dpdL_hydro": dpdL_hydro,
            "Ek": Ek, "H_L": H_L,
            "lambda_l": v_sl / v_m if v_m > 0 else 0.0,
            "pattern": pattern, "rho_s": rho_s, "v_m": v_m,
            "Fr": v_m ** 2 / (G_C * D_ft), "Re_ns": 0.0, "f_tp": 0.0}


# ----------------------------------------------------------------------
# GRAY — wet-gas / gas-condensate vertical wells
# ----------------------------------------------------------------------
def _gray_dpdL(v_sl, v_sg, rho_l, rho_g, mu_g, D_in_inch, theta_deg,
                rel_roughness, P_psia, sigma=30.0):
    """Gray (1974) correlation for vertical wet-gas wells.

    Reasonable when v_sg > ~15 ft/s and lambda_l < ~0.1 (gas-dominated).
    Returns the same dict shape as the others.
    """
    if abs(theta_deg - 90.0) > 30.0:
        return _beggs_brill_dpdL(v_sl, v_sg, rho_l, rho_g, mu_g, mu_g,
                                  D_in_inch, theta_deg, rel_roughness,
                                  P_psia, sigma)
    D_ft = D_in_inch / 12.0
    v_m = v_sl + v_sg
    if v_m <= 0:
        return {"dpdL_psi_ft": rho_l / PSI_TO_PSF, "dpdL_fric": 0.0,
                "dpdL_hydro": rho_l / PSI_TO_PSF, "Ek": 0.0,
                "H_L": 1.0, "lambda_l": 1.0, "pattern": "no-flow",
                "rho_s": rho_l, "v_m": 0.0, "Fr": 0.0,
                "Re_ns": 0.0, "f_tp": 0.0}
    R = v_sl / v_sg if v_sg > 0 else 1e6
    # Gray dimensionless numbers — guard against rho_g >= rho_l (a
    # heavy condensate near its critical point can have a gas density
    # that approaches or exceeds the in-situ liquid density).
    drho = max(rho_l - rho_g, 0.1)
    Nv = rho_g ** 2 * v_m ** 4 / (G_C * sigma * drho + 1e-9)
    Nd = G_C * drho * D_ft ** 2 / max(sigma, 1e-6)
    A_term = -2.5142 * (1 + 730.0 * R / (R + 1.0)) / max(Nd ** 0.5, 1e-6)
    H_L = 1.0 - (1.0 - math.exp(min(A_term, 50.0))) \
            * ((R + 1.0) / (R + 1.0001))
    H_L = max(min(H_L, 1.0), v_sl / v_m)
    rho_s = rho_l * H_L + rho_g * (1.0 - H_L)
    # Effective roughness for Gray (function of liquid loading)
    e_e = max(rel_roughness * D_ft,
               (28.5 * sigma / (G_C * drho + 1e-9)
                * (1.0 + R)) ** 0.5)
    e_e_rel = e_e / D_ft if D_ft > 0 else rel_roughness
    # Cap to a physically meaningful range — Haaland is undefined for
    # e/D > ~0.05 and unreliable above that.
    e_e_rel = min(e_e_rel, 0.05)
    Re = 1488.0 * rho_g * v_m * D_ft / max(mu_g, 1e-9)
    f = _haaland_friction(Re, e_e_rel)
    dpdL_fric = f * rho_g * v_m ** 2 / (2.0 * G_C * D_ft) / PSI_TO_PSF
    dpdL_hydro = rho_s * math.sin(math.radians(theta_deg)) / PSI_TO_PSF
    Ek = v_m * v_sg * rho_g / (G_C * max(P_psia, 1.0) * PSI_TO_PSF)
    Ek = min(Ek, 0.95)
    dpdL_total = (dpdL_fric + dpdL_hydro) / max(1.0 - Ek, 0.05)
    return {"dpdL_psi_ft": dpdL_total, "dpdL_fric": dpdL_fric,
            "dpdL_hydro": dpdL_hydro, "Ek": Ek, "H_L": H_L,
            "lambda_l": v_sl / v_m,
            "pattern": "mist/annular (Gray)",
            "rho_s": rho_s, "v_m": v_m,
            "Fr": v_m ** 2 / (G_C * D_ft),
            "Re_ns": Re, "f_tp": f}


# ----------------------------------------------------------------------
# NO-SLIP HOMOGENEOUS (Fancher & Brown lower bound)
# ----------------------------------------------------------------------
def _result(dpdL_fric, dpdL_hydro, Ek, H_L, lam, pattern, rho_s, v_m,
            D_ft, Re=0.0, f=0.0):
    Ek = min(max(Ek, 0.0), 0.95)
    return {"dpdL_psi_ft": (dpdL_fric + dpdL_hydro) / max(1.0 - Ek, 0.05),
            "dpdL_fric": dpdL_fric, "dpdL_hydro": dpdL_hydro, "Ek": Ek,
            "H_L": H_L, "lambda_l": lam, "pattern": pattern,
            "rho_s": rho_s, "v_m": v_m,
            "Fr": v_m ** 2 / (G_C * D_ft) if D_ft > 0 else 0.0,
            "Re_ns": Re, "f_tp": f}


def _no_flow(rho_l, D_ft):
    return _result(0.0, rho_l / PSI_TO_PSF, 0.0, 1.0, 1.0, "no-flow",
                   rho_l, 0.0, D_ft)


def _noslip_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g, D_in_inch,
                 theta_deg, rel_roughness, P_psia, sigma=30.0):
    """Homogeneous no-slip model: H_L = lambda_L, Moody friction on
    no-slip mixture properties. Physically the LOWER bound of the
    pressure drop in upward flow (Fancher & Brown use it the same way)."""
    D_ft = D_in_inch / 12.0
    v_m = v_sl + v_sg
    if v_m <= 0:
        return _no_flow(rho_l, D_ft)
    lam = v_sl / v_m
    rho_n = rho_l * lam + rho_g * (1 - lam)
    mu_n = mu_l * lam + mu_g * (1 - lam)
    Re = 1488.0 * rho_n * v_m * D_ft / max(mu_n, 1e-9)
    f = _haaland_friction(Re, rel_roughness)
    fric = f * rho_n * v_m ** 2 / (2 * G_C * D_ft) / PSI_TO_PSF
    hydro = rho_n * math.sin(math.radians(theta_deg)) / PSI_TO_PSF
    Ek = v_m * v_sg * rho_n / (G_C * max(P_psia, 1.0) * PSI_TO_PSF)
    return _result(fric, hydro, Ek, lam, lam, "homogeneous (no-slip)",
                   rho_n, v_m, D_ft, Re, f)


# ----------------------------------------------------------------------
# MUKHERJEE & BRILL (1985) — all inclinations
# ----------------------------------------------------------------------
_MB_UP = (-0.380113, 0.129875, -0.119788, 2.343227, 0.475686, 0.288657)
_MB_DN_STRAT = (-1.330282, 4.808139, 4.171584, 56.262268, 0.079951, 0.504887)
_MB_DN_OTHER = (-0.516644, 0.789805, 0.551627, 15.519214, 0.371771, 0.393952)
_MB_FR = ((0.01, 1.00), (0.20, 0.98), (0.30, 1.20), (0.40, 1.25),
          (0.50, 1.30), (0.70, 1.25), (1.00, 1.00), (10.0, 1.00))


def _mukherjee_brill_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g, D_in_inch,
                          theta_deg, rel_roughness, P_psia, sigma=30.0):
    """Mukherjee & Brill (1985): explicit flow-pattern map and holdup
    H_L = exp[(C1 + C2 sin + C3 sin^2 + C4 NL^2) NGV^C5 / NLV^C6].
    Bubble / slug: gravity on slip density, Moody friction (no-slip Re).
    Annular: friction-factor ratio table vs lambda/H_L. The stratified
    (downhill) momentum balance is approximated with the slug form."""
    D_ft = D_in_inch / 12.0
    v_m = v_sl + v_sg
    if v_m <= 0:
        return _no_flow(rho_l, D_ft)
    lam = v_sl / v_m
    sg = max(sigma, 1e-3)
    k4 = (rho_l / sg) ** 0.25
    NLV, NGV = max(1.938 * v_sl * k4, 1e-9), max(1.938 * v_sg * k4, 1e-9)
    NL = 0.15726 * mu_l * (1.0 / (rho_l * sg ** 3)) ** 0.25
    th = math.radians(theta_deg)
    st_, s2 = math.sin(th), math.sin(th) ** 2
    lgv, llv = math.log10(NGV), math.log10(NLV)
    # flow pattern
    NGVSM = 10 ** (1.401 - 2.694 * NL + 0.521 * NLV ** 0.329)
    if NGV > NGVSM:
        pattern = "annular"
    elif theta_deg >= 0:
        NLVBS = 10 ** (lgv + 0.940 + 0.074 * st_ - 0.855 * s2 + 3.695 * NL)
        pattern = "bubble" if NLV > NLVBS else "slug"
    else:
        NLVST = 10 ** (0.321 - 0.017 * NGV - 4.267 * st_ - 2.972 * NL
                       - 0.033 * lgv ** 2 - 3.925 * s2)
        if NLV < NLVST:
            pattern = "stratified"
        else:
            NGVBS = 10 ** (llv + 0.431 - 3.003 * NL - 1.138 * llv * st_
                           - 0.429 * llv ** 2 * st_ + 1.132 * st_)
            pattern = "bubble" if NGV < NGVBS else "slug"
    C = (_MB_UP if theta_deg >= 0 else
         _MB_DN_STRAT if pattern == "stratified" else _MB_DN_OTHER)
    expo = (C[0] + C[1] * st_ + C[2] * s2 + C[3] * NL ** 2) \
        * NGV ** C[4] / NLV ** C[5]
    H_L = math.exp(max(min(expo, 0.0), -50.0))
    H_L = min(max(H_L, lam if theta_deg >= 0 else 0.0), 1.0)
    rho_s = rho_l * H_L + rho_g * (1 - H_L)
    rho_n = rho_l * lam + rho_g * (1 - lam)
    mu_n = mu_l * lam + mu_g * (1 - lam)
    Re = 1488.0 * rho_n * v_m * D_ft / max(mu_n, 1e-9)
    f = _haaland_friction(Re, rel_roughness)
    if pattern == "annular":
        HR = lam / max(H_L, 1e-9)
        xs, ys = zip(*_MB_FR)
        f = f * float(np.interp(HR, xs, ys))
        fric = f * rho_n * v_m ** 2 / (2 * G_C * D_ft) / PSI_TO_PSF
    else:
        fric = f * rho_s * v_m ** 2 / (2 * G_C * D_ft) / PSI_TO_PSF
    hydro = rho_s * st_ / PSI_TO_PSF
    Ek = rho_s * v_m * v_sg / (G_C * max(P_psia, 1.0) * PSI_TO_PSF)
    return _result(fric, hydro, Ek, H_L, lam, pattern + " (M-B)", rho_s,
                   v_m, D_ft, Re, f)


# ----------------------------------------------------------------------
# HASAN & KABIR (1988) — drift-flux, vertical and deviated wells
# ----------------------------------------------------------------------
def _hasan_kabir_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g, D_in_inch,
                      theta_deg, rel_roughness, P_psia, sigma=30.0):
    """Hasan & Kabir mechanistic drift-flux model.
    Bubble: alpha = v_sg / (1.2 v_m + v_inf,b), Harmathy rise velocity.
    Slug / churn: Taylor-bubble rise 0.35 sqrt(gD drho/rho_l), corrected
    for deviation by sqrt(sin)(1 + cos)^1.2; C0 = 1.2 (slug), 1.15 (churn).
    Annular (Kutateladze, v_sg > 3.1 [g sigma drho / rho_g^2]^0.25):
    homogeneous core. Horizontal / downhill pipes defer to Beggs & Brill.
    """
    if theta_deg < 15.0:
        return _beggs_brill_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g,
                                 D_in_inch, theta_deg, rel_roughness,
                                 P_psia, sigma)
    D_ft = D_in_inch / 12.0
    v_m = v_sl + v_sg
    if v_m <= 0:
        return _no_flow(rho_l, D_ft)
    lam = v_sl / v_m
    # SI for the mechanistic closures
    g = 9.80665
    rl, rg = rho_l * 16.0185, rho_g * 16.0185
    drho = max(rl - rg, 1.0)
    sg = max(sigma, 1e-3) * 1e-3
    D = D_ft * 0.3048
    vsl, vsg, vm = v_sl * 0.3048, v_sg * 0.3048, v_m * 0.3048
    th = math.radians(theta_deg)
    v_b = 1.53 * (g * sg * drho / rl ** 2) ** 0.25
    v_T = 0.35 * math.sqrt(g * D * drho / rl) \
        * math.sqrt(max(math.sin(th), 0.0)) * (1 + math.cos(th)) ** 1.2
    v_ann = 3.1 * (g * sg * drho / max(rg, 1e-3) ** 2) ** 0.25
    if vsg > v_ann:
        alpha, pattern = 1 - lam, "annular"
    else:
        a_b = vsg / (1.2 * vm + v_b)
        # dispersed bubble at high mixture velocity (Hasan-Kabir)
        vm_disp = 4.68 * (D ** 0.48) * ((g * drho / sg) ** 0.5
                                        * (sg / rl) ** 0.6
                                        * (rl / max(mu_l * 1e-3, 1e-6)) ** 0.08)
        if a_b < 0.25:
            alpha, pattern = a_b, "bubble"
        elif vm > vm_disp and a_b < 0.52:
            alpha, pattern = vsg / (1.2 * vm), "dispersed bubble"
        else:
            a_s = vsg / (1.2 * vm + v_T)
            if vm > vm_disp:          # churn
                a_s = vsg / (1.15 * vm + v_T)
                pattern = "churn"
            else:
                pattern = "slug"
            alpha = a_s
    alpha = min(max(alpha, 0.0), 1 - lam if pattern == "annular" else 0.999)
    H_L = max(1 - alpha, lam)
    rho_s = rho_l * H_L + rho_g * (1 - H_L)
    mu_m = mu_l * H_L + mu_g * (1 - H_L)
    Re = 1488.0 * rho_s * v_m * D_ft / max(mu_m, 1e-9)
    f = _haaland_friction(Re, rel_roughness)
    if pattern == "slug":
        # friction acts on the liquid-slug part of the unit
        fric = f * rho_s * v_m ** 2 / (2 * G_C * D_ft) / PSI_TO_PSF * H_L
    else:
        fric = f * rho_s * v_m ** 2 / (2 * G_C * D_ft) / PSI_TO_PSF
    hydro = rho_s * math.sin(th) / PSI_TO_PSF
    Ek = rho_s * v_m * v_sg / (G_C * max(P_psia, 1.0) * PSI_TO_PSF)
    return _result(fric, hydro, Ek, H_L, lam, pattern + " (H-K)", rho_s,
                   v_m, D_ft, Re, f)


# ----------------------------------------------------------------------
# ORKISZEWSKI (1967) — vertical wells
# ----------------------------------------------------------------------
def _orkiszewski_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g, D_in_inch,
                      theta_deg, rel_roughness, P_psia, sigma=30.0,
                      water_frac=0.0):
    """Orkiszewski: Griffith bubble flow, Griffith-Wallis slug flow with
    the liquid-distribution coefficient Gamma, Duns & Ros mist flow,
    linear interpolation across the slug-mist transition.
    Simplification: bubble-rise velocity in slug flow from the Nicklin /
    Dumitrescu form 0.35 sqrt(gD) instead of the Griffith-Wallis chart.
    Non-vertical pipes defer to Beggs & Brill."""
    if abs(theta_deg - 90.0) > 30.0:
        return _beggs_brill_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g,
                                 D_in_inch, theta_deg, rel_roughness,
                                 P_psia, sigma)
    D = D_in_inch / 12.0
    v_m = v_sl + v_sg
    if v_m <= 0:
        return _no_flow(rho_l, D)
    lam = v_sl / v_m
    sin_t = math.sin(math.radians(theta_deg))
    k4 = (rho_l / max(sigma, 1e-3)) ** 0.25
    NLV, NGV = 1.938 * v_sl * k4, 1.938 * v_sg * k4
    L_B = max(1.071 - 0.2218 * v_m ** 2 / D, 0.13)
    L_S = 50.0 + 36.0 * NLV
    L_M = 75.0 + 84.0 * NLV ** 0.75

    def _bubble():
        v_s = 0.8
        disc = (1 + v_m / v_s) ** 2 - 4 * v_sg / v_s
        H_g = 0.5 * (1 + v_m / v_s - math.sqrt(max(disc, 0.0)))
        H_L = min(max(1 - H_g, lam), 1.0)
        rho_s = rho_l * H_L + rho_g * (1 - H_L)
        v_L = v_sl / max(H_L, 1e-6)
        Re = 1488.0 * rho_l * v_L * D / max(mu_l, 1e-9)
        f = _haaland_friction(Re, rel_roughness)
        fric = f * rho_l * v_L ** 2 / (2 * G_C * D) / PSI_TO_PSF
        return fric, rho_s * sin_t / PSI_TO_PSF, H_L, rho_s, Re, f

    def _slug():
        v_b = 0.35 * math.sqrt(G_C * D)
        lmu = math.log10(mu_l + 1.0)
        lv, lD = math.log10(max(v_m, 1e-6)), math.log10(D)
        if water_frac >= 0.75:      # water continuous
            if v_m < 10:
                G = 0.013 * math.log10(max(mu_l, 1e-6)) / D ** 1.38 - 0.681 \
                    + 0.232 * lv - 0.428 * lD
            else:
                G = 0.045 * math.log10(max(mu_l, 1e-6)) / D ** 0.799 - 0.709 \
                    - 0.162 * lv - 0.888 * lD
        else:                       # oil continuous
            if v_m < 10:
                G = 0.0127 * lmu / D ** 1.415 - 0.284 + 0.167 * lv + 0.113 * lD
            else:
                X = -lv * (0.01 * lmu / D ** 1.571 + 0.397 + 0.63 * lD)
                G = 0.0274 * lmu / D ** 1.371 + 0.161 + 0.569 * lD + X
        rho_s0 = (rho_l * (v_sl + v_b) + rho_g * v_sg) / (v_m + v_b)
        if v_m < 10:
            G = max(G, -0.065 * v_m)
        else:
            G = max(G, -v_b / (v_m + v_b) * (1 - rho_s0 / rho_l))
        rho_s = min(max(rho_s0 + G * rho_l, rho_g), rho_l)
        Re = 1488.0 * rho_l * v_m * D / max(mu_l, 1e-9)
        f = _haaland_friction(Re, rel_roughness)
        fric = f * rho_l * v_m ** 2 / (2 * G_C * D) / PSI_TO_PSF \
            * max((v_sl + v_b) / (v_m + v_b) + G, 0.0)
        H_L = min(max((rho_s - rho_g) / max(rho_l - rho_g, 1e-6), lam), 1.0)
        return fric, rho_s * sin_t / PSI_TO_PSF, H_L, rho_s, Re, f

    def _mist():
        rho_n = rho_l * lam + rho_g * (1 - lam)
        Re = 1488.0 * rho_g * v_sg * D / max(mu_g, 1e-9)
        f = _haaland_friction(Re, rel_roughness)
        fric = f * rho_g * v_sg ** 2 / (2 * G_C * D) / PSI_TO_PSF
        return fric, rho_n * sin_t / PSI_TO_PSF, lam, rho_n, Re, f

    if v_sg / v_m < L_B:
        (fric, hyd, H_L, rho_s, Re, f), pat = _bubble(), "bubble"
    elif NGV < L_S:
        (fric, hyd, H_L, rho_s, Re, f), pat = _slug(), "slug"
    elif NGV > L_M:
        (fric, hyd, H_L, rho_s, Re, f), pat = _mist(), "mist"
    else:
        a = (L_M - NGV) / max(L_M - L_S, 1e-9)
        s_, m_ = _slug(), _mist()
        fric, hyd, H_L, rho_s, Re, f = [a * x + (1 - a) * y
                                        for x, y in zip(s_, m_)]
        pat = "transition"
    # Physical bound: in upward flow the slip density cannot fall below
    # the no-slip density (the Gamma term can push it there at low v_m).
    rho_n = rho_l * lam + rho_g * (1 - lam)
    if rho_s < rho_n:
        rho_s, H_L = rho_n, lam
        hyd = rho_s * sin_t / PSI_TO_PSF
    Ek = 0.0 if pat in ("bubble", "slug") else \
        rho_s * v_m * v_sg / (G_C * max(P_psia, 1.0) * PSI_TO_PSF)
    return _result(fric, hyd, Ek, H_L, lam, pat + " (Ork.)", rho_s, v_m,
                   D, Re, f)


# ----------------------------------------------------------------------
# CORRELATION ROUTER
# ----------------------------------------------------------------------
CORRELATIONS = ("Beggs & Brill", "Hagedorn & Brown", "Gray",
                "Mukherjee & Brill", "Hasan & Kabir (mechanistic)",
                "Orkiszewski", "No-slip homogeneous")

CORRELATION_NOTES = {
    "Beggs & Brill": "Empirical, all inclinations (1973/77 map).",
    "Hagedorn & Brown": "Vertical oil wells, Griffith bubble correction.",
    "Gray": "Vertical wet-gas / condensate wells.",
    "Mukherjee & Brill": "Empirical, all inclinations incl. downhill.",
    "Hasan & Kabir (mechanistic)": "Drift-flux model, vertical/deviated wells.",
    "Orkiszewski": "Vertical oil wells, slug flow with liquid distribution.",
    "No-slip homogeneous": "Lower bound (no slip), single-phase / mist.",
}



def recommend_correlation(fluid_type, theta_deg, GLR_scf_stb=None):
    """Pick the best of {B-B, H-B, Gray} for the given geometry/fluid.

    Returns a dict with `recommended`, `rationale` and any
    `alternatives` the user might want to compare against.
    """
    vertical = abs(theta_deg - 90.0) < 15.0
    horizontal_ish = abs(theta_deg) < 15.0
    rec = "Beggs & Brill"
    alts = []
    rationale = "Beggs-Brill is the general-purpose default."
    if fluid_type in ("Wet Gas", "Gas-Condensate", "Dry Gas") and vertical:
        rec = "Gray"
        alts = ["Hasan & Kabir (mechanistic)", "No-slip homogeneous",
                "Beggs & Brill"]
        rationale = ("Gray was developed for vertical wet-gas / "
                      "condensate wells and outperforms Beggs-Brill in "
                      "that regime.")
    elif fluid_type in ("Light Oil", "Heavy Oil", "Water") and vertical:
        rec = "Hagedorn & Brown"
        alts = ["Orkiszewski", "Hasan & Kabir (mechanistic)",
                "Mukherjee & Brill", "Beggs & Brill"]
        rationale = ("Hagedorn-Brown with the Griffith bubble-flow "
                      "correction is the long-standing default for "
                      "vertical oil wells with slug flow. Compare with "
                      "Orkiszewski and the Hasan-Kabir mechanistic model, "
                      "and calibrate against well tests.")
        if GLR_scf_stb is not None and GLR_scf_stb > 5000:
            rationale += (" At this high GLR, mist/annular flow is likely; "
                           "check Hasan & Kabir and Gray as well.")
            alts = ["Hasan & Kabir (mechanistic)", "Gray"] + alts[:2]
    elif fluid_type in ("Light Oil", "Heavy Oil", "Water") and theta_deg >= 15:
        rec = "Mukherjee & Brill"
        alts = ["Hasan & Kabir (mechanistic)", "Beggs & Brill"]
        rationale = ("Deviated well: Mukherjee-Brill was developed on "
                      "inclined pipe (all angles); the Hasan-Kabir drift-"
                      "flux model includes a deviation correction.")
    elif horizontal_ish or theta_deg < 0:
        rec = "Beggs & Brill"
        alts = ["Mukherjee & Brill"]
        rationale = ("Beggs-Brill is the standard for horizontal, "
                      "inclined and downhill flowlines; it handles "
                      "the inclination correction properly.")
    return {"recommended": rec, "rationale": rationale,
            "alternatives": alts}


def _select_dpdL(method, v_sl, v_sg, rho_l, rho_g, mu_l, mu_g,
                  D_in_inch, theta_deg, rel_roughness, P_psia, sigma,
                  water_frac=0.0):
    if method == "Mukherjee & Brill":
        return _mukherjee_brill_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g,
                                     D_in_inch, theta_deg, rel_roughness,
                                     P_psia, sigma)
    if method.startswith("Hasan & Kabir"):
        return _hasan_kabir_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g,
                                 D_in_inch, theta_deg, rel_roughness,
                                 P_psia, sigma)
    if method == "Orkiszewski":
        return _orkiszewski_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g,
                                 D_in_inch, theta_deg, rel_roughness,
                                 P_psia, sigma, water_frac)
    if method == "No-slip homogeneous":
        return _noslip_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g,
                            D_in_inch, theta_deg, rel_roughness, P_psia,
                            sigma)
    if method == "Hagedorn & Brown":
        return _hagedorn_brown_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l,
                                     mu_g, D_in_inch, theta_deg,
                                     rel_roughness, P_psia, sigma)
    if method == "Gray":
        return _gray_dpdL(v_sl, v_sg, rho_l, rho_g, mu_g, D_in_inch,
                           theta_deg, rel_roughness, P_psia, sigma)
    return _beggs_brill_dpdL(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g,
                              D_in_inch, theta_deg, rel_roughness,
                              P_psia, sigma)


# ----------------------------------------------------------------------
# MARCH — outlet to inlet, building the pressure profile
# ----------------------------------------------------------------------
def march_pressure(P_outlet_psia, T_outlet_F, T_inlet_F, segments,
                    q_liq_bbl_d, q_gas_mscf_d,
                    api, gas_sg, water_cut_frac, salinity_wt_pct,
                    D_in_inch, rel_roughness, method,
                    n_seg_inner=10, sigma=30.0,
                    hold_mult=1.0, fric_mult=1.0, pvt=None,
                    gl_inj_mscf_d=0.0, gl_inj_md_ft=None, gl_gas_sg=None):
    """March from outlet (known P) to inlet, returning the full
    pressure profile.

    Continuous gas lift: `gl_inj_mscf_d` of lift gas (specific gravity
    `gl_gas_sg`, default = formation gas) enters the tubing at measured
    depth `gl_inj_md_ft` (from the outlet). Above that point the free-gas
    rate is formation gas + lift gas (free-gas properties from the mixed
    gravity); below it, formation gas only.

    segments : list of (length_ft, inclination_deg) tuples ordered from
               OUTLET to INLET. For a vertical well being lifted, the
               outlet is the wellhead and the inlet is the bottomhole.

    Returns a dict with:
        stations_ft     : cumulative measured distance (from outlet)
        P_psia          : pressure at each station
        T_F             : temperature at each station (linear interp)
        H_L             : in-situ liquid holdup at each station
        pattern         : flow pattern label at each station
        dpdL_fric       : friction component
        dpdL_hydro      : hydrostatic component
        P_inlet         : inlet pressure (the lift-curve result)
    """
    stations = [0.0]
    Ps = [P_outlet_psia]
    Ts = [T_outlet_F]
    HLs = [np.nan]
    patterns = ["(outlet)"]
    fric = [0.0]
    hydro = [0.0]
    total_len = sum(s[0] for s in segments)
    gl_on = (gl_inj_mscf_d or 0.0) > 0 and gl_inj_md_ft is not None \
        and 0.0 < gl_inj_md_ft < total_len
    if gl_on:
        # split the segment that contains the injection point
        _segs, _c = [], 0.0
        for L_, th_ in segments:
            if _c < gl_inj_md_ft < _c + L_:
                _segs += [(gl_inj_md_ft - _c, th_), (_c + L_ - gl_inj_md_ft, th_)]
            else:
                _segs.append((L_, th_))
            _c += L_
        segments = _segs
        sg_lift = gas_sg if gl_gas_sg is None else gl_gas_sg
    q_gas_formation = q_gas_mscf_d
    cum = 0.0
    for seg_len, theta in segments:
        dl = seg_len / n_seg_inner
        above_inj = gl_on and (cum + 0.5 * seg_len) < gl_inj_md_ft
        q_gas_seg = q_gas_formation + (gl_inj_mscf_d if above_inj else 0.0)
        sg_free = ((q_gas_formation * gas_sg + gl_inj_mscf_d * sg_lift)
                   / max(q_gas_seg, 1e-12)) if above_inj else gas_sg
        for _ in range(n_seg_inner):
            cum += dl
            # Linear T interp from outlet to inlet over total length
            frac = cum / total_len if total_len > 0 else 0.0
            T_here = T_outlet_F + (T_inlet_F - T_outlet_F) * frac
            def _grad(P_here, q_gas_mscf_d=q_gas_seg, sg_free=sg_free):
                # Properties at current P, T (free gas: lift + formation
                # gas mixture above the injection point)
                Z = _gas_z_simple(P_here, T_here, sg_free)
                mu_g = _gas_viscosity_lee(P_here, T_here, sg_free, Z)
                rho_g = _gas_density_at(P_here, T_here, sg_free, Z)
                # Solution GOR (Standing-like; very simple screening)
                # Phase split. Only the gas NOT dissolved in the oil is free
                # gas; water carries no dissolved gas and has Bw ~ 1.
                wc = max(0.0, min(water_cut_frac, 1.0))
                q_oil = q_liq_bbl_d * (1.0 - wc)
                q_wat = q_liq_bbl_d * wc
                GOR = (q_gas_mscf_d * 1000.0 / q_oil) if q_oil > 1e-9 else 0.0
                # Standing Rs: Rs = gg * [(P/18.2 + 1.4) * 10^x]^1.2048
                x_st = 0.0125 * api - 0.00091 * T_here
                Rs = gas_sg * max((P_here / 18.2 + 1.4) * 10 ** x_st, 0.0) ** 1.2048
                Rs = min(Rs, GOR)
                gamma_o = 141.5 / (api + 131.5)
                Bo = 0.972 + 0.000147 * (Rs * (gas_sg / gamma_o) ** 0.5
                                         + 1.25 * T_here) ** 1.175
                q_gas_free = max(q_gas_mscf_d - Rs * q_oil / 1000.0, 0.0)
                # Live-oil viscosity (Beggs-Robinson): mu = a * mu_od^b
                mu_od = _oil_viscosity_simple(api, T_here)
                mu_o = (10.715 * (Rs + 100.0) ** -0.515
                        * mu_od ** (5.44 * (Rs + 150.0) ** -0.338))
                rho_o = _oil_density(api, gas_sg, P_here, Rs, Bo)
                rho_w = _water_density(salinity_wt_pct, T_here)
                mu_w = max(0.3 + 0.005 * (T_here - 60.0), 0.1)
                q_oil_free, Bw, gas_rb = q_oil, 1.0, None
                if pvt is not None:
                    # Imported ECLIPSE tables: black-oil split with Rs and
                    # Rv, table FVFs / viscosities / densities. Missing
                    # tables keep the correlation values above.
                    fl = pvt.flash(P_here, T_here, q_oil, q_gas_mscf_d)
                    q_oil_free, q_gas_free = fl["O_f"], fl["G_f"]
                    if fl["Bo"] is not None:
                        Bo, mu_o = fl["Bo"], fl["mu_o"]
                        if fl["rho_o"] is not None:
                            rho_o = fl["rho_o"]
                    if fl["Bg"] is not None:
                        gas_rb = q_gas_free * fl["Bg"]
                        mu_g = fl["mu_g"]
                        if fl["rho_g"] is not None:
                            rho_g = fl["rho_g"]
                    if pvt.water is not None:
                        Bw, mu_w = pvt.water_props(P_here)
                        if pvt.rho_sc:
                            rho_w = pvt.rho_sc[1] / Bw
                # Liquid mixture weighted by IN-SITU volumes
                v_o, v_w = q_oil_free * Bo, q_wat * Bw
                f_o = v_o / (v_o + v_w) if (v_o + v_w) > 0 else 1.0
                rho_l = rho_o * f_o + rho_w * (1.0 - f_o)
                mu_l = mu_o * f_o + mu_w * (1.0 - f_o)
                # In-situ liquid volume already includes Bo -> pass Bo = 1
                v_sl, v_sg = _superficial_velocities(
                    v_o + v_w, q_gas_free, D_in_inch, P_here, T_here, Z, 1.0)
                if gas_rb is not None:
                    _area = math.pi * (D_in_inch / 12.0) ** 2 / 4.0
                    v_sg = gas_rb * 5.614583 / 86400.0 / _area
                dpdL = _select_dpdL(method, v_sl, v_sg, rho_l, rho_g,
                                      mu_l, mu_g, D_in_inch, theta,
                                      rel_roughness, P_here, sigma,
                                      water_frac=1.0 - f_o)
                if hold_mult != 1.0 or fric_mult != 1.0:
                    # Calibration multipliers (PROSPER-style "parameter
                    # 1 / parameter 2"): gravity and friction terms.
                    dpdL = dict(dpdL)
                    dpdL["dpdL_hydro"] *= hold_mult
                    dpdL["dpdL_fric"] *= fric_mult
                    dpdL["dpdL_psi_ft"] = (
                        (dpdL["dpdL_fric"] + dpdL["dpdL_hydro"])
                        / max(1.0 - dpdL["Ek"], 0.05))
                return dpdL

            # Adaptive sub-stepping: limit the pressure change per step to
            # 5 % of the local pressure. A fixed explicit step let a steep
            # gradient (high-rate gas, low outlet P) jump hundreds of psi in
            # one step and evaluate the next step at the wrong density.
            P_next, rem, it = Ps[-1], dl, 0
            dpdL = _grad(P_next)          # also covers zero-length segments
            while rem > 1e-9 and it < 400:
                if it:
                    dpdL = _grad(P_next)
                g = dpdL["dpdL_psi_ft"]
                h = rem if g == 0 else min(rem, 0.05 * max(P_next, 1.0) / abs(g))
                h = max(h, dl / 200.0)
                h = min(h, rem)
                P_next = max(P_next + g * h, 1.0)
                rem -= h
                it += 1
            stations.append(cum)
            Ps.append(P_next)
            Ts.append(T_here)
            HLs.append(dpdL["H_L"])
            patterns.append(dpdL["pattern"])
            fric.append(dpdL["dpdL_fric"])
            hydro.append(dpdL["dpdL_hydro"])
    return {"stations_ft": np.array(stations),
            "P_psia": np.array(Ps),
            "T_F": np.array(Ts),
            "H_L": np.array(HLs),
            "pattern": patterns,
            "dpdL_fric": np.array(fric),
            "dpdL_hydro": np.array(hydro),
            "P_inlet_psia": Ps[-1],
            "gl_inj_md_ft": gl_inj_md_ft if gl_on else None,
            "P_inj_psia": (float(np.interp(gl_inj_md_ft, stations, Ps))
                           if gl_on else None)}


def lift_curve(rates_bbl_d, GLR_scf_stb,
                P_outlet_psia, T_outlet_F, T_inlet_F, segments,
                api, gas_sg, water_cut_frac, salinity_wt_pct,
                D_in_inch, rel_roughness, method,
                hold_mult=1.0, fric_mult=1.0, pvt=None,
                gl_inj_mscf_d=0.0, gl_inj_md_ft=None, gl_gas_sg=None,
                gl_iglr_scf_stb=None):
    """Build a Tubing/Flowline Performance Curve.

    Returns a list of dicts, one per rate:
        q_liq_bbl_d, q_gas_mscf_d, P_inlet_psia, dominant_pattern,
        H_L_avg, dP_total
    """
    out = []
    for q in rates_bbl_d:
        q_gas = q * GLR_scf_stb / 1000.0
        try:
            r = march_pressure(
                P_outlet_psia, T_outlet_F, T_inlet_F, segments,
                q_liq_bbl_d=q, q_gas_mscf_d=q_gas, api=api,
                gas_sg=gas_sg, water_cut_frac=water_cut_frac,
                salinity_wt_pct=salinity_wt_pct,
                D_in_inch=D_in_inch, rel_roughness=rel_roughness,
                method=method, hold_mult=hold_mult, fric_mult=fric_mult,
                pvt=pvt,
                gl_inj_mscf_d=(q * gl_iglr_scf_stb / 1000.0
                               if gl_iglr_scf_stb is not None
                               else gl_inj_mscf_d),
                gl_inj_md_ft=gl_inj_md_ft, gl_gas_sg=gl_gas_sg)
        except Exception as e:
            out.append({"q_liq_bbl_d": q, "q_gas_mscf_d": q_gas,
                        "P_inlet_psia": np.nan,
                        "dominant_pattern": f"error: {e}",
                        "H_L_avg": np.nan,
                        "dP_total_psi": np.nan})
            continue
        # Dominant pattern: most-common label (skip 'outlet')
        labels = [p for p in r["pattern"] if p != "(outlet)"]
        if labels:
            from collections import Counter
            dominant = Counter(labels).most_common(1)[0][0]
        else:
            dominant = "—"
        out.append({"q_liq_bbl_d": q, "q_gas_mscf_d": q_gas,
                    "P_inlet_psia": r["P_inlet_psia"],
                    "dominant_pattern": dominant,
                    "H_L_avg": float(np.nanmean(r["H_L"])),
                    "dP_total_psi": r["P_inlet_psia"] - P_outlet_psia,
                    "P_inj_psia": r.get("P_inj_psia")})
    return out


# ======================================================================
# Inflow performance (IPR) and IPR / VLP operating point
# ======================================================================
# All pressures psia, liquid rates STB/d, gas rates Mscf/d. Every IPR is
# referenced to the SAME node as the VLP inlet (bottom of the last pipe
# segment), so the two curves can be intersected directly.

IPR_MODELS = ("Straight-line PI (undersaturated oil)",
              "Vogel (saturated oil)",
              "Composite PI + Vogel (Pr > Pb)",
              "Gas back-pressure (C, n)",
              "Darcy radial — oil (pseudo-steady state)",
              "Darcy radial — gas (pseudo-steady state, non-Darcy D)")
IPR_GAS_MODELS = (IPR_MODELS[3], IPR_MODELS[5])
IPR_DARCY_MODELS = (IPR_MODELS[4], IPR_MODELS[5])


def _vogel_frac(x):
    """q/qmax for Vogel at x = Pwf/Pr (or Pwf/Pb)."""
    x = min(max(x, 0.0), 1.0)
    return 1.0 - 0.2 * x - 0.8 * x * x


def _vogel_x(frac):
    """Inverse Vogel: x = Pwf/Pr for a given q/qmax."""
    frac = min(max(frac, 0.0), 1.0)
    return (-0.2 + np.sqrt(0.04 + 3.2 * (1.0 - frac))) / 1.6


def _pi_or_composite(model, Pr, J, Pb=None):
    """Parameter dict for a straight-line / composite IPR with index J."""
    p = {"model": model, "Pr": float(Pr), "J": float(J)}
    if Pb and 0.0 < Pb < Pr:
        qb = J * (Pr - Pb)
        p.update(kind="composite", Pb=float(Pb), qb=qb, aof=qb + J * Pb / 1.8)
    else:
        p.update(kind="pi", aof=J * Pr)
    return p


def _composite_basis(Pr, Pb, pwf):
    """q / J for the composite IPR at pwf."""
    if pwf >= Pb:
        return Pr - pwf
    return (Pr - Pb) + Pb / 1.8 * _vogel_frac(pwf / Pb)


def calibrate_ipr(model, Pr, q_test, pwf_test, Pb=None, n=1.0):
    """Fit an empirical IPR to one well-test point (q_test at pwf_test).
    Returns the parameter dict (with `aof` and `J`)."""
    if q_test <= 0 or not (0.0 <= pwf_test < Pr):
        raise ValueError("Test point must have q > 0 and 0 <= Pwf < Pr.")
    return fit_ipr(model, Pr, [(q_test, pwf_test)], Pb=Pb, n=n)["params"]


# ---- Darcy (radial inflow) ----------------------------------------------
def darcy_oil_J(k_md, h_ft, mu_cp, Bo, re_ft, rw_ft, skin):
    """Pseudo-steady-state productivity index, STB/d/psi:
    J = k h / [141.2 Bo mu (ln(re/rw) - 0.75 + S)]."""
    den = 141.2 * Bo * mu_cp * (math.log(re_ft / rw_ft) - 0.75 + skin)
    if den <= 0:
        raise ValueError("ln(re/rw) - 0.75 + S must be > 0 (skin too "
                         "negative for this drainage radius).")
    return k_md * h_ft / den


def make_darcy_oil(Pr, k_md, h_ft, mu_cp, Bo, re_ft, rw_ft, skin, Pb=None):
    """Darcy radial oil IPR; Vogel below Pb when 0 < Pb < Pr."""
    J = darcy_oil_J(k_md, h_ft, mu_cp, Bo, re_ft, rw_ft, skin)
    p = _pi_or_composite(IPR_MODELS[4], Pr, J, Pb)
    p.update(k=k_md, h=h_ft, mu=mu_cp, Bo=Bo, re=re_ft, rw=rw_ft, S=skin)
    return p


def _mu_z(P, T_F, gas_sg, pvt=None):
    if pvt is not None and getattr(pvt, "gas", None):
        return pvt.mu_z(max(P, 14.7), T_F)
    Z = _gas_z_simple(max(P, 14.7), T_F, gas_sg)
    return _gas_viscosity_lee(max(P, 14.7), T_F, gas_sg, Z) * Z


def make_darcy_gas(Pr, k_md, h_ft, T_F, gas_sg, re_ft, rw_ft, skin,
                   D_nd=0.0, pvt=None):
    """Darcy radial gas IPR (pressure-squared form, mu*z at the average
    pressure) with rate-dependent skin D (1 / (Mscf/d)):
        Pr^2 - Pwf^2 = a mu z q + b mu z q^2,
        a = 1422 T (ln(re/rw) - 0.75 + S) / (k h),  b = 1422 T D / (k h)."""
    T_R = T_F + 459.67
    lnr = math.log(re_ft / rw_ft) - 0.75 + skin
    if lnr <= 0:
        raise ValueError("ln(re/rw) - 0.75 + S must be > 0.")
    p = {"model": IPR_MODELS[5], "kind": "darcy_gas", "Pr": float(Pr),
         "a": 1422.0 * T_R * lnr / (k_md * h_ft),
         "b": 1422.0 * T_R * max(D_nd, 0.0) / (k_md * h_ft),
         "T_F": T_F, "gas_sg": gas_sg, "k": k_md, "h": h_ft,
         "re": re_ft, "rw": rw_ft, "S": skin, "D": max(D_nd, 0.0),
         "pvt": pvt}
    p["aof"] = ipr_rate(p, 0.0)
    p["J"] = ipr_rate(p, 0.9 * Pr) / (0.1 * Pr)
    return p


# ---- evaluation ---------------------------------------------------------
def ipr_rate(p, pwf):
    """Inflow rate at a flowing bottom-hole pressure (STB/d or Mscf/d)."""
    Pr, kd = p["Pr"], p["kind"]
    pwf = min(max(float(pwf), 0.0), Pr)
    if kd == "pi":
        return p["J"] * (Pr - pwf)
    if kd == "vogel":
        return p["qmax"] * _vogel_frac(pwf / Pr)
    if kd == "composite":
        return p["J"] * _composite_basis(Pr, p["Pb"], pwf)
    if kd == "backpressure":
        return p["C"] * (Pr ** 2 - pwf ** 2) ** p["n"]
    if kd == "darcy_gas":
        mz = _mu_z(0.5 * (Pr + pwf), p["T_F"], p["gas_sg"], p.get("pvt"))
        dp2 = Pr ** 2 - pwf ** 2
        A, B = p["a"] * mz, p["b"] * mz
        if B <= 1e-15:
            return dp2 / A
        return (-A + math.sqrt(A * A + 4 * B * dp2)) / (2 * B)
    raise ValueError(f"Unknown IPR kind {kd!r}")


def ipr_pwf(p, q):
    """Flowing bottom-hole pressure delivering rate q (NaN beyond AOF)."""
    Pr, kd = p["Pr"], p["kind"]
    q = float(q)
    if q < 0 or q > p["aof"] * (1 + 1e-9):
        return float("nan")
    if kd == "pi":
        return Pr - q / p["J"]
    if kd == "vogel":
        return Pr * _vogel_x(q / p["qmax"])
    if kd == "composite":
        if q <= p["qb"]:
            return Pr - q / p["J"]
        return p["Pb"] * _vogel_x((q - p["qb"]) / (p["J"] * p["Pb"] / 1.8))
    if kd == "backpressure":
        return float(np.sqrt(max(Pr ** 2 - (q / p["C"]) ** (1.0 / p["n"]),
                                 0.0)))
    # monotone in pwf -> bisection (darcy_gas)
    lo, hi = 0.0, Pr
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ipr_rate(p, mid) > q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---- calibration against one or more well tests --------------------------
def fit_ipr(model, Pr, points, Pb=None, n=1.0, darcy=None):
    """Least-squares fit of an IPR to well-test points [(q, pwf), ...]
    (q: STB/d liquid, or Mscf/d for the gas models; pwf at the node).

    Empirical models: J (PI, composite), qmax (Vogel), C and n
    (back-pressure; n is fitted when >= 2 points, else held).
    Darcy models (darcy = reservoir properties dict): the skin S is
    solved; for gas with >= 2 points the non-Darcy D is fitted too.
    Returns {"params", "fitted": {name: value}, "rmse_pwf", "n_points"}.
    """
    pts = [(float(q), float(w)) for q, w in points
           if q and q > 0 and w is not None and 0 <= w < Pr]
    if not pts:
        raise ValueError("No valid test point (need q > 0 and Pwf < Pr).")
    q = np.array([a for a, _ in pts])
    w = np.array([b for _, b in pts])
    fitted = {}
    if model == IPR_MODELS[0]:
        x = Pr - w
        J = float((q @ x) / (x @ x))
        p = _pi_or_composite(model, Pr, J)
        fitted["J"] = J
    elif model == IPR_MODELS[1]:
        f = np.array([_vogel_frac(v / Pr) for v in w])
        qmax = float((q @ f) / (f @ f))
        p = {"model": model, "kind": "vogel", "Pr": float(Pr), "qmax": qmax,
             "aof": qmax, "J": 1.8 * qmax / Pr}
        fitted["qmax"] = qmax
    elif model == IPR_MODELS[2]:
        if not (Pb and 0.0 < Pb < Pr):
            raise ValueError("Composite IPR needs 0 < Pb < Pr.")
        g = np.array([_composite_basis(Pr, Pb, v) for v in w])
        J = float((q @ g) / (g @ g))
        p = _pi_or_composite(model, Pr, J, Pb)
        fitted["J"] = J
    elif model == IPR_MODELS[3]:
        dp2 = Pr ** 2 - w ** 2
        if len(pts) >= 2 and np.ptp(np.log(dp2)) > 1e-6:
            nn, lc = np.polyfit(np.log(dp2), np.log(q), 1)
            nn = float(min(max(nn, 0.5), 1.0))
            C = float(np.exp(np.mean(np.log(q) - nn * np.log(dp2))))
            fitted["n"] = nn
        else:
            nn = float(n)
            C = float(np.exp(np.mean(np.log(q) - nn * np.log(dp2))))
        p = {"model": model, "kind": "backpressure", "Pr": float(Pr),
             "C": C, "n": nn, "aof": C * Pr ** (2 * nn)}
        p["J"] = ipr_rate(p, 0.9 * Pr) / (0.1 * Pr)
        fitted["C"] = C
    elif model == IPR_MODELS[4]:
        d = darcy
        x = np.array([(_composite_basis(Pr, Pb, v) if (Pb and 0 < Pb < Pr)
                       else Pr - v) for v in w])
        J = float((q @ x) / (x @ x))
        S = (d["k"] * d["h"] / (141.2 * d["Bo"] * d["mu"] * J)
             - math.log(d["re"] / d["rw"]) + 0.75)
        p = make_darcy_oil(Pr, d["k"], d["h"], d["mu"], d["Bo"], d["re"],
                           d["rw"], S, Pb)
        fitted["S"] = S
    elif model == IPR_MODELS[5]:
        d = darcy
        T_R = d["T_F"] + 459.67
        mz = np.array([_mu_z(0.5 * (Pr + v), d["T_F"], d["gas_sg"],
                             d.get("pvt")) for v in w])
        y = (Pr ** 2 - w ** 2) / mz              # = a q + b q^2
        kh = d["k"] * d["h"]
        if len(pts) >= 2 and np.ptp(q) > 1e-6 * q.max():
            A = np.c_[q, q ** 2]
            (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
            if b < 0:
                a, b = float((q @ y) / (q @ q)), 0.0
            Dn = b * kh / (1422.0 * T_R)
            fitted["D"] = float(Dn)
        else:
            b = 1422.0 * T_R * d.get("D", 0.0) / kh
            a = float(np.mean((y - b * q ** 2) / q))
            Dn = d.get("D", 0.0)
        S = a * kh / (1422.0 * T_R) - math.log(d["re"] / d["rw"]) + 0.75
        p = make_darcy_gas(Pr, d["k"], d["h"], d["T_F"], d["gas_sg"],
                           d["re"], d["rw"], S, Dn, pvt=d.get("pvt"))
        fitted["S"] = S
    else:
        raise ValueError(f"Unknown IPR model {model!r}")
    pred = np.array([ipr_pwf(p, a) for a in q])
    rm = float(np.sqrt(np.nanmean((pred - w) ** 2))) if len(q) else 0.0
    return {"params": p, "fitted": fitted, "rmse_pwf": rm,
            "n_points": len(pts)}


# ---- VLP calibration ------------------------------------------------------
def _vlp_at_tests(tests, common, method, hold_mult=1.0, fric_mult=1.0):
    """Predicted node pressure for each test: dicts with q (STB/d), thp
    (psia), glr (scf/STB), wc. `common` holds the march_pressure kwargs
    that do not change between tests."""
    out = []
    for t in tests:
        try:
            r = march_pressure(
                t["thp"], common["T_outlet_F"], common["T_inlet_F"],
                common["segments"], t["q"], t["q"] * t["glr"] / 1000.0,
                common["api"], common["gas_sg"], t["wc"],
                common["salinity_wt_pct"], common["D_in_inch"],
                common["rel_roughness"], method,
                hold_mult=hold_mult, fric_mult=fric_mult,
                pvt=common.get("pvt"),
                gl_inj_mscf_d=t.get("q_inj", common.get("gl_inj_mscf_d", 0.0)),
                gl_inj_md_ft=common.get("gl_inj_md_ft"),
                gl_gas_sg=common.get("gl_gas_sg"))
            out.append(float(r["P_inlet_psia"]))
        except Exception:
            out.append(float("nan"))
    return np.array(out)


def rank_correlations(tests, common, methods=None):
    """Score every correlation against measured node pressures.
    Returns rows sorted by mean absolute % error."""
    meas = np.array([t["pwf"] for t in tests], float)
    rows = []
    for m in (methods or CORRELATIONS):
        pred = _vlp_at_tests(tests, common, m)
        err = pred - meas
        ok = np.isfinite(err)
        if not ok.any():
            continue
        rows.append({"method": m, "pred": pred,
                     "mae": float(np.mean(np.abs(err[ok]))),
                     "mape": float(np.mean(np.abs(err[ok] / meas[ok])) * 100),
                     "bias": float(np.mean(err[ok])),
                     "n_ok": int(ok.sum())})
    return sorted(rows, key=lambda r: r["mape"])


def tune_vlp(tests, common, method, tune_friction=None,
             hold_bounds=(0.5, 1.5), fric_bounds=(0.3, 3.0)):
    """Fit gravity (holdup) and friction multipliers so the chosen
    correlation matches the measured node pressures (least squares on
    relative error). With a single test only the gravity multiplier is
    fitted (two unknowns cannot be identified from one point)."""
    from scipy.optimize import least_squares
    meas = np.array([t["pwf"] for t in tests], float)
    if tune_friction is None:
        tune_friction = len(tests) >= 2

    def resid(x):
        hm = x[0]
        fm = x[1] if tune_friction else 1.0
        pr = _vlp_at_tests(tests, common, method, hm, fm)
        r = (pr - meas) / meas
        return np.where(np.isfinite(r), r, 1.0)

    x0 = [1.0, 1.0] if tune_friction else [1.0]
    lb = [hold_bounds[0]] + ([fric_bounds[0]] if tune_friction else [])
    ub = [hold_bounds[1]] + ([fric_bounds[1]] if tune_friction else [])
    sol = least_squares(resid, x0, bounds=(lb, ub), diff_step=1e-3,
                        xtol=1e-6, ftol=1e-8)
    hm = float(sol.x[0])
    fm = float(sol.x[1]) if tune_friction else 1.0
    before = _vlp_at_tests(tests, common, method)
    after = _vlp_at_tests(tests, common, method, hm, fm)
    at_bound = any(abs(v - b) < 1e-3 for v, b in
                   ((hm, hold_bounds[0]), (hm, hold_bounds[1]),
                    (fm, fric_bounds[0]), (fm, fric_bounds[1]))
                   if (tune_friction or v is hm))
    return {"method": method, "hold_mult": hm, "fric_mult": fm,
            "tuned_friction": bool(tune_friction),
            "pred_before": before, "pred_after": after, "meas": meas,
            "mape_before": float(np.nanmean(np.abs(before - meas) / meas) * 100),
            "mape_after": float(np.nanmean(np.abs(after - meas) / meas) * 100),
            "at_bound": bool(at_bound)}


def correct_pressure_to_node(P_datum, datum_tvd_ft, node_tvd_ft, grad_psi_ft):
    """Shift a static reservoir pressure from its datum to the node depth
    with a static fluid gradient (deeper node -> higher pressure)."""
    return P_datum + grad_psi_ft * (node_tvd_ft - datum_tvd_ft)


def operating_points(q_vlp, p_vlp, ipr_pwf_fn):
    """Intersections of a VLP (required node pressure vs rate) with an
    IPR given as Pwf(q).

    Returns a list (low -> high rate) of dicts {q, pwf, stable}. A crossing
    is stable when the VLP rises faster than the IPR there
    (dVLP/dq > dIPR/dq); on the gravity-dominated (left) branch it is not.
    """
    q = np.asarray(q_vlp, float)
    pv = np.asarray(p_vlp, float)
    pi = np.array([ipr_pwf_fn(x) for x in q])
    d = pv - pi
    out = []
    for i in range(len(q) - 1):
        a, b = d[i], d[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0 or a * b < 0:
            t = 0.0 if a == 0 else a / (a - b)
            qx = q[i] + t * (q[i + 1] - q[i])
            px = pv[i] + t * (pv[i + 1] - pv[i])
            s_vlp = (pv[i + 1] - pv[i]) / (q[i + 1] - q[i])
            s_ipr = (pi[i + 1] - pi[i]) / (q[i + 1] - q[i])
            out.append({"q": float(qx), "pwf": float(px),
                        "stable": bool(s_vlp > s_ipr)})
    return out


def well_trajectory(segments):
    """(horizontal displacement, TVD, MD) arrays from the WELLHEAD down,
    for segments listed OUTLET -> INLET with inclination measured from
    horizontal, positive when the flow goes up (producer)."""
    x, z, md = [0.0], [0.0], [0.0]
    for L, th in segments:
        r = np.radians(th)
        x.append(x[-1] + L * np.cos(r))
        z.append(z[-1] + L * np.sin(r))       # TVD increases downward
        md.append(md[-1] + L)
    return np.array(x), np.array(z), np.array(md)


# ======================================================================
# Continuous gas lift: injection-pressure check and performance curve
# ======================================================================
def gas_column_pressure(P_top_psia, tvd_ft, gas_sg, T_top_F, T_bot_F,
                        n_steps=40):
    """Static gas-column pressure at depth tvd_ft below a surface pressure
    P_top (casing annulus, no flow): dP/dz = rho_g g, rho_g from the
    real-gas law with the Brill-Beggs Z. Returns psia."""
    P = float(P_top_psia)
    if tvd_ft <= 0:
        return P
    dz = tvd_ft / n_steps
    for i in range(n_steps):
        T = T_top_F + (T_bot_F - T_top_F) * (i + 0.5) / n_steps
        # midpoint (RK2) step
        Pm = P + 0.5 * dz * _gas_density_at(P, T, gas_sg,
                                            _gas_z_simple(P, T, gas_sg)) / PSI_TO_PSF
        rho = _gas_density_at(Pm, T, gas_sg, _gas_z_simple(Pm, T, gas_sg))
        P += rho * dz / PSI_TO_PSF
    return P


def required_surface_injection_pressure(P_tubing_at_valve_psia, valve_tvd_ft,
                                        gas_sg, T_surf_F, T_valve_F,
                                        dP_valve_psi=100.0):
    """Casing-head pressure needed so that the annulus gas at the valve
    exceeds the tubing pressure by dP_valve. Solved by bisection on the
    static gas column."""
    target = P_tubing_at_valve_psia + dP_valve_psi
    lo, hi = 14.7, target
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if gas_column_pressure(mid, valve_tvd_ft, gas_sg, T_surf_F,
                               T_valve_F) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def point_of_balance(traverse, md_to_tvd, P_inj_surf_psia, gas_sg, T_surf_F,
                     T_bot_F, dP_valve_psi=100.0):
    """Deepest MD where the annulus gas pressure (from the available
    surface injection pressure) minus the valve differential still
    exceeds the tubing pressure of the given traverse. None if the
    injection pressure is insufficient even at the top."""
    st_ = np.asarray(traverse["stations_ft"])
    Pt = np.asarray(traverse["P_psia"])
    tvd_end = max(md_to_tvd(st_[-1]), 1.0)
    best = None
    for md, pt in zip(st_, Pt):
        tvd = md_to_tvd(md)
        Tz = T_surf_F + (T_bot_F - T_surf_F) * min(max(tvd / tvd_end, 0), 1)
        pc = gas_column_pressure(P_inj_surf_psia, tvd, gas_sg, T_surf_F, Tz,
                                 n_steps=20)
        if pc - dP_valve_psi >= pt:
            best = float(md)
        else:
            break
    return best


def gas_lift_performance(inj_rates_mscf_d, rates_bbl_d, ipr_pwf_liq, lc_kwargs):
    """Gas-lift performance curve: for each lift-gas rate build the VLP
    (lift_curve with lc_kwargs) and intersect it with the IPR (Pwf as a
    function of LIQUID rate). Returns rows {q_inj, q_liq, pwf, stable}
    (q_liq None when the well does not flow)."""
    out = []
    for qi in inj_rates_mscf_d:
        kw = dict(lc_kwargs, gl_inj_mscf_d=float(qi))
        res = lift_curve(rates_bbl_d, **kw)
        qv = [r["q_liq_bbl_d"] for r in res if r["P_inlet_psia"] == r["P_inlet_psia"]]
        pv = [r["P_inlet_psia"] for r in res if r["P_inlet_psia"] == r["P_inlet_psia"]]
        ops = [o for o in operating_points(qv, pv, ipr_pwf_liq) if o["stable"]]
        if ops:
            o = max(ops, key=lambda x: x["q"])
            out.append({"q_inj": float(qi), "q_liq": o["q"], "pwf": o["pwf"],
                        "stable": True})
        else:
            out.append({"q_inj": float(qi), "q_liq": None, "pwf": None,
                        "stable": False})
    return out
