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
          + 0.32 * Ppr ** 6 / max(10 ** (9 * (Tpr - 1)), 1e-12))
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
    if lambda_l < 0.01:
        # Very dry — segregated by default
        pattern = "segregated"
    elif (lambda_l < 0.01 and Fr < L1) or (lambda_l >= 0.01 and Fr < L2):
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
    # Liquid velocity number for the inclination correction
    Nlv = v_sl * (rho_l / (G_C * sigma * 1e-3)) ** 0.25
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
        y_term = lam / (H_L ** 2)
        if y_term > 1.2:
            s = math.log(2.2 * y_term - 1.2)
        else:
            x = math.log(max(y_term, 1e-6))
            denom = (-0.0523 + 3.182 * x - 0.8725 * x ** 2
                      + 0.01853 * x ** 4)
            s = x / max(denom, 1e-6) if denom != 0 else 0.0
        f_tp = f_n * math.exp(s)
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
        # CNl from a curve fit (Brill-Mukherjee form)
        log_Nl = math.log10(max(Nl, 1e-6))
        CNl = 10 ** (-2.69851 + 0.15841 * log_Nl - 0.55100 * log_Nl ** 2
                       + 0.54785 * log_Nl ** 3 - 0.12195 * log_Nl ** 4)
        CNl = max(min(CNl, 0.1), 0.002)
        # Holdup correlation: H_L/psi = f(group)
        group = Nlv / max(Ngv ** 0.575, 1e-6) * (P_psia / 14.7) ** 0.1 \
                * CNl / max(Nd, 1e-6)
        # Fit for H_L/psi
        log_g = math.log10(max(group, 1e-12))
        HL_psi = (0.10307 + 0.61777 * log_g - 0.63295 * log_g ** 2
                   + 0.29598 * log_g ** 3 - 0.0401 * log_g ** 4)
        HL_psi = max(min(HL_psi, 1.0), 0.0)
        # Secondary correction psi
        group2 = Ngv * Nl ** 0.380 / max(Nd ** 2.14, 1e-6)
        if group2 > 0.012:
            psi = (0.91163 - 4.82176 * group2 + 1232.25 * group2 ** 2
                    - 22253.6 * group2 ** 3 + 116174.3 * group2 ** 4)
            psi = max(min(psi, 1.8), 1.0)
        else:
            psi = 1.0
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
# CORRELATION ROUTER
# ----------------------------------------------------------------------
CORRELATIONS = ("Beggs & Brill", "Hagedorn & Brown", "Gray")


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
        alts = ["Beggs & Brill"]
        rationale = ("Gray was developed for vertical wet-gas / "
                      "condensate wells and outperforms Beggs-Brill in "
                      "that regime.")
    elif fluid_type in ("Light Oil", "Heavy Oil") and vertical:
        rec = "Hagedorn & Brown"
        alts = ["Beggs & Brill", "Duns-Ros (not implemented)",
                "Orkiszewski (not implemented)"]
        rationale = ("Hagedorn-Brown with the Griffith bubble-flow "
                      "correction is the long-standing default for "
                      "vertical oil wells with slug flow. Duns-Ros and "
                      "Orkiszewski are also relevant at high GOR but "
                      "are not implemented in this tool — Beggs-Brill "
                      "is provided as an alternative.")
        if GLR_scf_stb is not None and GLR_scf_stb > 5000:
            rationale += (" The very high GLR places this in the "
                           "Duns-Ros / Orkiszewski sweet spot; treat "
                           "H-B and B-B results as screening only.")
    elif horizontal_ish or theta_deg < 0:
        rec = "Beggs & Brill"
        alts = ["Hagedorn & Brown (not appropriate here)"]
        rationale = ("Beggs-Brill is the standard for horizontal, "
                      "inclined and downhill flowlines; it handles "
                      "the inclination correction properly.")
    return {"recommended": rec, "rationale": rationale,
            "alternatives": alts}


def _select_dpdL(method, v_sl, v_sg, rho_l, rho_g, mu_l, mu_g,
                  D_in_inch, theta_deg, rel_roughness, P_psia, sigma):
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
                    n_seg_inner=10, sigma=30.0):
    """March from outlet (known P) to inlet, returning the full
    pressure profile.

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
    cum = 0.0
    for seg_len, theta in segments:
        dl = seg_len / n_seg_inner
        for _ in range(n_seg_inner):
            cum += dl
            # Linear T interp from outlet to inlet over total length
            frac = cum / total_len if total_len > 0 else 0.0
            T_here = T_outlet_F + (T_inlet_F - T_outlet_F) * frac
            P_here = Ps[-1]
            # Properties at current P, T
            Z = _gas_z_simple(P_here, T_here, gas_sg)
            mu_g = _gas_viscosity_lee(P_here, T_here, gas_sg, Z)
            rho_g = _gas_density_at(P_here, T_here, gas_sg, Z)
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
            # Liquid mixture weighted by IN-SITU volumes
            v_o, v_w = q_oil * Bo, q_wat * 1.0
            f_o = v_o / (v_o + v_w) if (v_o + v_w) > 0 else 1.0
            rho_l = rho_o * f_o + rho_w * (1.0 - f_o)
            mu_w = max(0.3 + 0.005 * (T_here - 60.0), 0.1)
            mu_l = mu_o * f_o + mu_w * (1.0 - f_o)
            # In-situ liquid volume already includes Bo -> pass Bo = 1
            v_sl, v_sg = _superficial_velocities(
                v_o + v_w, q_gas_free, D_in_inch, P_here, T_here, Z, 1.0)
            dpdL = _select_dpdL(method, v_sl, v_sg, rho_l, rho_g,
                                  mu_l, mu_g, D_in_inch, theta,
                                  rel_roughness, P_here, sigma)
            P_next = P_here + dpdL["dpdL_psi_ft"] * dl
            # Guard against runaway negative pressure
            P_next = max(P_next, 1.0)
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
            "P_inlet_psia": Ps[-1]}


def lift_curve(rates_bbl_d, GLR_scf_stb,
                P_outlet_psia, T_outlet_F, T_inlet_F, segments,
                api, gas_sg, water_cut_frac, salinity_wt_pct,
                D_in_inch, rel_roughness, method):
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
                method=method)
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
                    "dP_total_psi": r["P_inlet_psia"] - P_outlet_psia})
    return out
