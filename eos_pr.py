"""
Peng-Robinson EOS (1976) with two-phase isothermal flash.

Field units throughout: T [°R], P [psia], R = 10.732 psia·ft3/(lbmol·°R).

Pipeline:
    fugacity_coeffs(z, P, T, phase) -> ln(phi_i)
    flash(z, P, T)                  -> {V, x, y, K, ln_phi_L, ln_phi_V, status}
    saturation_pressure(z, T)       -> Pb (oil) or Pdew (gas)
"""

import numpy as np
from components import get_props, kij

R = 10.732  # psia·ft3 / (lbmol·°R)


# ------------------------------------------------------------
# PR cubic root and ln-fugacity coefficients for a phase
# ------------------------------------------------------------
def pr_phase(comp_names, x_or_y, P, T, c7_props, want="liquid"):
    """
    Compute PR Z-root and ln(phi_i) for one phase given composition `x_or_y`.
    want = 'liquid' selects the smaller real root, 'vapor' the larger.
    When only one real root exists, that root is used regardless of `want`.
    """
    n = len(comp_names)
    Tc = np.array([get_props(c, c7_props)["Tc"]    for c in comp_names])
    Pc = np.array([get_props(c, c7_props)["Pc"]    for c in comp_names])
    om = np.array([get_props(c, c7_props)["omega"] for c in comp_names])

    m = np.where(om <= 0.49,
                 0.37464 + 1.54226 * om - 0.26992 * om ** 2,
                 0.379642 + 1.48503 * om - 0.164423 * om ** 2 + 0.016666 * om ** 3)
    Tr = T / Tc
    alpha = (1 + m * (1 - np.sqrt(Tr))) ** 2

    a_i = 0.45724 * (R * Tc) ** 2 / Pc * alpha
    b_i = 0.07780 * R * Tc / Pc

    A_i = a_i * P / (R * T) ** 2
    B_i = b_i * P / (R * T)

    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kij(comp_names[i], comp_names[j])

    sqrtA = np.sqrt(A_i)
    Aij = (1 - K) * np.outer(sqrtA, sqrtA)

    z = np.asarray(x_or_y, dtype=float)
    z = z / z.sum()

    A_mix = float(z @ Aij @ z)
    B_mix = float(z @ B_i)

    coeffs = [1.0,
              -(1 - B_mix),
              A_mix - 3 * B_mix ** 2 - 2 * B_mix,
              -(A_mix * B_mix - B_mix ** 2 - B_mix ** 3)]
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    real_roots = real_roots[real_roots > B_mix + 1e-12]
    if len(real_roots) == 0:
        real_roots = np.array([B_mix + 1e-6])
    real_roots = np.sort(real_roots)

    sqrt2 = np.sqrt(2.0)
    sum_xj_Aij = Aij @ z

    def lnphi_for(Zr):
        return (B_i / B_mix * (Zr - 1)
                - np.log(max(Zr - B_mix, 1e-30))
                - A_mix / (2 * sqrt2 * B_mix) *
                  (2 * sum_xj_Aij / A_mix - B_i / B_mix) *
                  np.log((Zr + (1 + sqrt2) * B_mix) /
                         (Zr + (1 - sqrt2) * B_mix)))

    if len(real_roots) == 1:
        Zphase = real_roots[0]
        ln_phi = lnphi_for(Zphase)
    else:
        # Use the requested root
        Zphase = real_roots[0] if want == "liquid" else real_roots[-1]
        ln_phi = lnphi_for(Zphase)

    return Zphase, ln_phi, A_mix, B_mix, b_i, a_i


# ------------------------------------------------------------
# Michelsen-style stability test (TPD)
# Returns True if a 2nd phase is favored, plus initial K-values.
# ------------------------------------------------------------
def stability_test(z, comp_names, P, T, c7_props=None, tol=1e-9, maxit=80):
    """
    Two-sided TPD test:
      - vapor-like trial Y = z * K_w  (gas-down)
      - liquid-like trial Y = z / K_w (liquid-down)
    If either trial reduces TPD significantly, mixture is unstable.
    """
    z = np.asarray(z, dtype=float); z = z / z.sum()
    K_w = wilson_k(comp_names, P, T, c7_props)
    K_w = np.clip(K_w, 1e-15, 1e15)

    # Reference (feed) ln-fugacity. Try both phase roots and use lower-G one.
    Z_l, lnphi_l, *_ = pr_phase(comp_names, z, P, T, c7_props, want="liquid")
    Z_v, lnphi_v, *_ = pr_phase(comp_names, z, P, T, c7_props, want="vapor")
    G_l = float(np.dot(z, np.log(np.maximum(z, 1e-30)) + lnphi_l))
    G_v = float(np.dot(z, np.log(np.maximum(z, 1e-30)) + lnphi_v))
    if G_v < G_l:
        lnphi_z = lnphi_v
        ref_phase = "V"
    else:
        lnphi_z = lnphi_l
        ref_phase = "L"
    di = np.log(np.maximum(z, 1e-30)) + lnphi_z

    unstable = False
    K_init = K_w.copy()

    for trial_dir in ("vapor", "liquid"):
        Y = z * K_w if trial_dir == "vapor" else z / K_w
        for _ in range(maxit):
            Sy = Y.sum()
            y = Y / Sy
            want = "vapor" if trial_dir == "vapor" else "liquid"
            _, lnphi_y, *_ = pr_phase(comp_names, y, P, T, c7_props, want=want)
            Y_new = np.exp(di - lnphi_y)
            err = np.max(np.abs(Y_new - Y))
            Y = Y_new
            if err < tol:
                break
        if Y.sum() > 1.0 + 1e-6:
            unstable = True
            # Build K-values from converged trial
            y = Y / Y.sum()
            if trial_dir == "vapor":
                K_init = y / z
            else:
                K_init = z / y
            K_init = np.clip(K_init, 1e-12, 1e12)
            break

    return unstable, K_init, ref_phase


# ------------------------------------------------------------
# Wilson initial K-values
# ------------------------------------------------------------
def wilson_k(comp_names, P, T, c7_props):
    Tc = np.array([get_props(c, c7_props)["Tc"]    for c in comp_names])
    Pc = np.array([get_props(c, c7_props)["Pc"]    for c in comp_names])
    om = np.array([get_props(c, c7_props)["omega"] for c in comp_names])
    return (Pc / P) * np.exp(5.373 * (1 + om) * (1 - Tc / T))


# ------------------------------------------------------------
# Rachford-Rice solver for vapor fraction V given K and z
# ------------------------------------------------------------
def rachford_rice(z, K, tol=1e-10, maxit=80):
    Kmin, Kmax = K.min(), K.max()
    if Kmin > 1.0 or Kmax < 1.0:
        # single phase based on sum of (K-1)*z signs
        s = np.sum(z * (K - 1))
        return 0.0 if s <= 0 else 1.0
    V_lo = 1.0 / (1.0 - Kmax) + 1e-9
    V_hi = 1.0 / (1.0 - Kmin) - 1e-9
    V = 0.5
    for _ in range(maxit):
        f  = np.sum(z * (K - 1) / (1 + V * (K - 1)))
        df = -np.sum(z * (K - 1) ** 2 / (1 + V * (K - 1)) ** 2)
        if abs(df) < 1e-30:
            break
        V_new = V - f / df
        if not (V_lo < V_new < V_hi):
            V_new = 0.5 * (V_lo + V_hi)
        if f > 0: V_lo = V
        else:     V_hi = V
        if abs(V_new - V) < tol:
            return float(np.clip(V_new, 0.0, 1.0))
        V = V_new
    return float(np.clip(V, 0.0, 1.0))


# ------------------------------------------------------------
# Two-phase flash (successive substitution)
# ------------------------------------------------------------
def flash(z, comp_names, P, T, c7_props=None, K0=None, tol=1e-9, maxit=200):
    z = np.asarray(z, dtype=float); z = z / z.sum()

    # Stability test first (skip if K0 is supplied)
    if K0 is None:
        unstable, K_init, ref_phase = stability_test(z, comp_names, P, T, c7_props)
        if not unstable:
            # Single-phase — identify L vs V using a robust heuristic.
            # Strategy: Wilson K-values describe the *direction* a 2-phase
            # would split. If z is closer to a typical-vapor composition
            # (i.e. high mole fraction of light components, where Wilson K > 1)
            # then z is vapor; otherwise liquid.
            Z_L, _, A_, B_, b_i_, _ = pr_phase(comp_names, z, P, T, c7_props, want="liquid")
            Z_V, _, *_ = pr_phase(comp_names, z, P, T, c7_props, want="vapor")
            K_w = wilson_k(comp_names, P, T, c7_props)
            # Sum of mole fractions of "light" components (Wilson Ki > 1)
            light_frac = float(np.sum(z[K_w > 1.0]))
            # Heuristic: > 0.5 light fraction => vapor
            if light_frac > 0.5:
                Z_use = Z_V if not np.isnan(Z_V) else Z_L
                return {"V": 1.0, "x": z * 0, "y": z, "K": np.ones_like(z),
                        "Z_L": np.nan, "Z_V": float(Z_use), "phase": "V"}
            else:
                Z_use = Z_L if not np.isnan(Z_L) else Z_V
                return {"V": 0.0, "x": z, "y": z * 0, "K": np.ones_like(z),
                        "Z_L": float(Z_use), "Z_V": np.nan, "phase": "L"}
        K = K_init
    else:
        K = np.clip(K0, 1e-15, 1e15)

    Z_L = Z_V = np.nan
    for it in range(maxit):
        V = rachford_rice(z, K)
        if V <= 0.0:
            x = z.copy(); y = K * x
        elif V >= 1.0:
            y = z.copy(); x = y / K
        else:
            x = z / (1 + V * (K - 1))
            y = K * x

        sum_x = x.sum(); sum_y = y.sum()
        if sum_x < 1e-12 or sum_y < 1e-12 or not np.isfinite(sum_x) or not np.isfinite(sum_y):
            # Trivial collapse: one phase. Identify by Wilson-K direction.
            break
        x = x / sum_x; y = y / sum_y

        Z_L, lnphi_L, *_ = pr_phase(comp_names, x, P, T, c7_props, want="liquid")
        Z_V, lnphi_V, *_ = pr_phase(comp_names, y, P, T, c7_props, want="vapor")
        K_new = np.exp(lnphi_L - lnphi_V)

        # Trivial-solution check: K -> 1 means we're collapsing to single phase
        if np.max(np.abs(K_new - 1)) < 1e-6:
            # Single phase — classify via density
            Z_single, _, A_, B_, *_ = pr_phase(comp_names, z, P, T, c7_props, want="liquid")
            Z_single_v, _, *_ = pr_phase(comp_names, z, P, T, c7_props, want="vapor")
            if abs(Z_single - Z_single_v) < 1e-6:
                if Z_single / B_ < 3.0:
                    return {"V": 0.0, "x": z, "y": z * 0, "K": np.ones_like(z),
                            "Z_L": float(Z_single), "Z_V": np.nan, "phase": "L"}
                return {"V": 1.0, "x": z * 0, "y": z, "K": np.ones_like(z),
                        "Z_L": np.nan, "Z_V": float(Z_single), "phase": "V"}

        err = np.max(np.abs(np.log(K_new / K)))
        K = K_new
        if err < tol:
            break

    if V <= 1e-8 or not np.isfinite(V):
        Z_L, _, *_ = pr_phase(comp_names, z, P, T, c7_props, want="liquid")
        return {"V": 0.0, "x": z, "y": z * 0,
                "K": K, "Z_L": float(Z_L), "Z_V": np.nan, "phase": "L"}
    if V >= 1 - 1e-8:
        Z_V, _, *_ = pr_phase(comp_names, z, P, T, c7_props, want="vapor")
        return {"V": 1.0, "x": z * 0, "y": z,
                "K": K, "Z_L": np.nan, "Z_V": float(Z_V), "phase": "V"}

    return {"V": float(V), "x": x, "y": y, "K": K,
            "Z_L": float(Z_L), "Z_V": float(Z_V), "phase": "LV"}


# ------------------------------------------------------------
# Saturation pressure (bubble or dew) — bisection on Σy_i or Σx_i = 1
# ------------------------------------------------------------
def saturation_pressure(z, comp_names, T, c7_props=None,
                        kind="auto", P_lo=14.7, P_hi=15000.0):
    """
    Find Pb (kind='bubble') or Pdew (kind='dew') at temperature T.
    'auto' tries to detect which is appropriate.
    Strategy: bracket the pressure where the flash transitions L/V.
    """
    def is_two_phase(P):
        r = flash(z, comp_names, P, T, c7_props)
        return r["V"] not in (0.0, 1.0)

    # Coarse scan for two-phase window
    Ps = np.geomspace(max(P_lo, 14.7), P_hi, 30)
    flags = [is_two_phase(P) for P in Ps]
    if not any(flags):
        return None  # no two-phase region in range
    first = flags.index(True)
    last  = len(flags) - 1 - flags[::-1].index(True)
    P_low_2p = Ps[first]
    P_high_2p = Ps[last]

    if kind == "auto":
        # If we lose the V phase at high P -> bubble (oil); lose L at high P -> dew (gas)
        r_top = flash(z, comp_names, Ps[-1], T, c7_props)
        if r_top["phase"] == "L":
            kind = "bubble"
        elif r_top["phase"] == "V":
            kind = "dew"
        else:
            kind = "bubble"

    # Refine the upper boundary of the 2-phase region
    if kind == "bubble":
        # Bubble = highest P at which V > 0 (just barely)
        lo, hi = P_low_2p, P_hi
        # Bracket: first single-phase pressure above P_high_2p
        P_test = P_high_2p
        while P_test < P_hi:
            P_test *= 1.05
            if not is_two_phase(P_test):
                hi = P_test
                lo = P_test / 1.05
                break
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if is_two_phase(mid):
                lo = mid
            else:
                hi = mid
            if hi - lo < 0.5:
                break
        return 0.5 * (lo + hi)

    else:  # dew
        # Dew = highest P at which gas exists (single-phase V at P > Pdew)
        # so we look for the upper bound of the two-phase region similarly
        lo, hi = P_low_2p, P_hi
        P_test = P_high_2p
        while P_test < P_hi:
            P_test *= 1.05
            if not is_two_phase(P_test):
                hi = P_test
                lo = P_test / 1.05
                break
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if is_two_phase(mid):
                lo = mid
            else:
                hi = mid
            if hi - lo < 0.5:
                break
        return 0.5 * (lo + hi)


# ------------------------------------------------------------
# Phase molar volume and density from Z-root
# ------------------------------------------------------------
def molar_volume(Z, P, T):
    """Returns ft3/lbmol."""
    return Z * R * T / P


def phase_density(comp_names, x_or_y, Z, P, T, c7_props=None):
    """Returns lb/ft3."""
    MW = np.array([get_props(c, c7_props)["MW"] for c in comp_names])
    Mavg = float(np.dot(x_or_y, MW))
    v = molar_volume(Z, P, T)  # ft3/lbmol
    return Mavg / v
