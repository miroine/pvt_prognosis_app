"""
Hydrate formation prediction.

Provides estimates of the hydrate-formation P-T equilibrium curve for natural
gas mixtures. Two approaches are supported:

  1. KATZ (1945) gas gravity method:
       log10(P) = a0 + a1*T + a2*T^2 + ... where coefficients depend on gas SG.
       The classical Katz K-value charts; we use the polynomial fit by
       Bahadori & Vuthaluru (2009).

  2. MAKOGON (1981) correlation:
       log10(P) = beta + 0.0497*(T - T0) + 0.00034*(T - T0)^2
       where beta = 2.681 - 3.811*g + 1.679*g^2 (g = gas SG) and T in °C.

The Makogon method is simpler and reasonably accurate for sweet natural gases
with SG 0.55-1.0 in the temperature range 0-30 °C; it's the industry default
for a first-pass hydrate screening.

For sour gas (H2S, CO2), hydrate-forming conditions shift to higher
temperatures — H2S strongly promotes hydrates. We add a small correction
based on H2S and CO2 content.

Inputs/outputs are in FIELD units at the boundary (psia, °F), with internal
calculations in metric.
"""

import numpy as np


def hydrate_pressure_makogon(T_F, gas_sg, H2S_frac=0.0, CO2_frac=0.0):
    """
    Makogon hydrate formation pressure for natural gas.

    Args:
        T_F      : temperature (°F)
        gas_sg   : gas specific gravity (air = 1)
        H2S_frac : mole fraction of H2S (0-1)
        CO2_frac : mole fraction of CO2 (0-1)

    Returns:
        P_hydrate (psia) — minimum pressure at which hydrate forms at this T.
        Returns NaN if T is out of range (T < 32°F or T > 80°F).
    """
    T_C = (T_F - 32.0) * 5.0 / 9.0
    if T_C < -2.0 or T_C > 30.0:
        return np.nan

    # Limit gas_sg to validity range
    g = max(0.55, min(gas_sg, 1.00))

    # Makogon (1981) correlation, with P in MPa and T in degC:
    #   log10(P) = beta + 0.0497*(T + kappa*T^2) - 1
    # The trailing "-1" offset and the kappa quadratic term are part of
    # the published form; omitting the "-1" overstates the hydrate
    # pressure by a factor of ten.
    beta = 2.681 - 3.811 * g + 1.679 * g ** 2
    kappa = -0.006 + 0.011 * g + 0.011 * g ** 2
    log10_P_MPa = beta + 0.0497 * (T_C + kappa * T_C ** 2) - 1.0
    P_MPa = 10 ** log10_P_MPa
    P_psia = P_MPa * 145.0377

    # Sour-gas correction: H2S strongly promotes hydrates (lowers required P)
    # Empirical: each 1% H2S lowers P_hyd by ~5%
    # CO2: each 1% CO2 lowers P_hyd by ~1.5%
    sour_factor = 1.0 - 5.0 * H2S_frac - 1.5 * CO2_frac
    sour_factor = max(0.4, sour_factor)
    P_psia *= sour_factor

    return P_psia


def hydrate_temperature_makogon(P_psia, gas_sg, H2S_frac=0.0, CO2_frac=0.0,
                                  T_low=33.0, T_high=75.0):
    """
    Inverse: find the hydrate-formation T at given P.
    Bisection over T to find P_hydrate(T) = P_psia.
    Returns T_F or NaN if not in the valid range.
    """
    # Check bounds
    P_low_T = hydrate_pressure_makogon(T_low, gas_sg, H2S_frac, CO2_frac)
    P_high_T = hydrate_pressure_makogon(T_high, gas_sg, H2S_frac, CO2_frac)
    if np.isnan(P_low_T) or np.isnan(P_high_T):
        return np.nan
    if P_psia < P_low_T or P_psia > P_high_T:
        # Outside validity range
        if P_psia > P_high_T:
            return T_high  # operating above hydrate at any T < T_high
        return np.nan
    for _ in range(50):
        T_mid = 0.5 * (T_low + T_high)
        P_mid = hydrate_pressure_makogon(T_mid, gas_sg, H2S_frac, CO2_frac)
        if np.isnan(P_mid):
            return np.nan
        if P_mid < P_psia:
            T_low = T_mid
        else:
            T_high = T_mid
        if T_high - T_low < 0.05:
            break
    return 0.5 * (T_low + T_high)


def hydrate_curve(gas_sg, H2S_frac=0.0, CO2_frac=0.0, n_points=40):
    """
    Trace the full hydrate P-T envelope curve.

    Returns (T_F_array, P_psia_array) — the hydrate locus.
    Points where P is NaN are dropped.
    """
    T_arr = np.linspace(32.0, 75.0, n_points)
    P_arr = np.array([hydrate_pressure_makogon(T, gas_sg, H2S_frac, CO2_frac)
                       for T in T_arr])
    mask = ~np.isnan(P_arr)
    return T_arr[mask], P_arr[mask]


def assess_hydrate_risk(T_F, P_psia, gas_sg, H2S_frac=0.0, CO2_frac=0.0,
                        safety_margin_psia=200.0):
    """
    Assess the hydrate-formation risk at a single operating point.

    Returns dict with:
        in_hydrate_zone : bool — True if (T, P) is in the hydrate-forming region
        P_hydrate       : hydrate-formation pressure at this T (psia)
        T_hydrate       : hydrate-formation temperature at this P (°F)
        margin_psia     : P - P_hydrate (positive = safe, negative = risky)
        margin_F        : T - T_hydrate (positive = safe, negative = risky)
        risk_level      : 'safe' / 'marginal' / 'in_zone' / 'unknown'
        message         : descriptive text
    """
    P_hyd = hydrate_pressure_makogon(T_F, gas_sg, H2S_frac, CO2_frac)
    T_hyd = hydrate_temperature_makogon(P_psia, gas_sg, H2S_frac, CO2_frac)

    result = {
        "P_hydrate": P_hyd,
        "T_hydrate": T_hyd,
        "in_hydrate_zone": False,
        "margin_psia": np.nan,
        "margin_F":    np.nan,
        "risk_level":  "unknown",
        "message":     "",
    }

    if np.isnan(P_hyd):
        result["risk_level"] = "unknown"
        result["message"] = (f"T = {T_F:.1f} °F is outside the Makogon "
                              f"correlation's valid range (32–80 °F). "
                              f"Use a more rigorous flash-based hydrate model.")
        return result

    margin_P = P_psia - P_hyd        # positive: operating ABOVE hydrate P
    margin_T = T_F - T_hyd if not np.isnan(T_hyd) else np.nan
    result["margin_psia"] = margin_P
    result["margin_F"] = margin_T

    # If the predicted hydrate P is very high (e.g. > 12000 psia, beyond
    # typical correlation validity), warn but report safe
    if P_hyd > 12000:
        result["risk_level"] = "safe"
        result["message"] = (
            f"✓ Hydrate formation pressure at T = {T_F:.1f} °F exceeds the "
            f"Makogon correlation's reliable range (~12,000 psia limit). "
            f"At normal operating pressures, hydrate formation is not expected "
            f"at this temperature. For high-P sour-gas systems consider a "
            f"rigorous flash-based hydrate model."
        )
        return result

    # In hydrate zone: P > P_hyd at this T (i.e., high enough P to form hydrates
    # at the current low T)
    in_zone = margin_P > 0
    result["in_hydrate_zone"] = in_zone

    if in_zone:
        if margin_P < safety_margin_psia:
            result["risk_level"] = "marginal"
            result["message"] = (
                f"⚠️ Operating just inside the hydrate zone. "
                f"P = {P_psia:.0f} psia is {margin_P:.0f} psia above the "
                f"hydrate formation pressure ({P_hyd:.0f} psia) at this T. "
                f"To exit the hydrate zone: warm to T > {T_hyd:.1f} °F "
                f"or reduce P below {P_hyd:.0f} psia, or inject inhibitor "
                f"(methanol / MEG)."
            )
        else:
            result["risk_level"] = "in_zone"
            result["message"] = (
                f"🛑 Deep in the hydrate-forming zone. "
                f"P = {P_psia:.0f} psia is {margin_P:.0f} psia above the "
                f"hydrate formation pressure ({P_hyd:.0f} psia) at T = "
                f"{T_F:.1f} °F. Hydrate plugging is likely. "
                f"Mitigation: inhibitor injection (methanol or MEG), "
                f"heating, depressurization, or insulation upgrade."
            )
    else:
        if -margin_P < safety_margin_psia:
            result["risk_level"] = "marginal"
            result["message"] = (
                f"⚠️ Operating just outside the hydrate zone. "
                f"P = {P_psia:.0f} psia is only {-margin_P:.0f} psia below "
                f"the hydrate formation pressure ({P_hyd:.0f} psia). "
                f"A modest pressure increase or temperature drop could "
                f"initiate hydrate formation."
            )
        else:
            result["risk_level"] = "safe"
            result["message"] = (
                f"✓ Outside hydrate-forming conditions. "
                f"P = {P_psia:.0f} psia is {-margin_P:.0f} psia below the "
                f"hydrate formation pressure ({P_hyd:.0f} psia) at this T. "
                f"To enter the hydrate zone would require pressuring up by "
                f"≥ {-margin_P:.0f} psia or cooling to T < {T_hyd:.1f} °F."
            )

    return result


def cooldown_time_to_hydrate(T_op_F, P_op_psia, T_ambient_F,
                                gas_sg, H2S_frac=0.0, CO2_frac=0.0,
                                U_pipe=2.0, D_outer_ft=0.667,
                                rho_fluid=50.0, cp_fluid=0.5):
    """
    Estimate the time for a subsea flowline to cool from operating T to the
    hydrate-formation T at the operating P after shutdown.

    Lumped-capacitance heat transfer model (no flow):
        T(t) = T_ambient + (T_op - T_ambient) * exp(-k * t / (rho * cp * D/4))

    Args:
        T_op_F      : Initial fluid temperature (°F)
        P_op_psia   : Operating pressure (psia)
        T_ambient_F : Surrounding seawater temperature (°F), typically 36-40°F
                      at subsea depth.
        gas_sg, H2S_frac, CO2_frac : gas properties for hydrate-T lookup
        U_pipe      : Overall heat-transfer coefficient (BTU/hr/ft2/°F),
                      typical 1-3 for insulated pipe, 5-15 for bare.
        D_outer_ft  : Pipe outer diameter (ft); 0.667 ≈ 8".
        rho_fluid   : Fluid density (lb/ft3)
        cp_fluid    : Fluid specific heat (BTU/lb/°F), ~0.5 for crude.

    Returns dict with:
        T_hydrate_F  : Hydrate-formation T at operating P
        delta_T_F    : T_op - T_hydrate
        time_hours   : Hours from shutdown to reach T_hydrate
        time_minutes : Same in minutes (handy for cooldown protocol)
    """
    # Hydrate temperature at operating P
    T_hyd_F = hydrate_temperature_makogon(P_op_psia, gas_sg, H2S_frac, CO2_frac)
    if np.isnan(T_hyd_F):
        return {
            "T_hydrate_F": np.nan,
            "delta_T_F":   np.nan,
            "time_hours":  np.nan,
            "time_minutes": np.nan,
            "message": ("Hydrate temperature outside the Makogon correlation's "
                         "valid range — use a rigorous hydrate model."),
        }

    if T_op_F <= T_hyd_F:
        return {
            "T_hydrate_F": T_hyd_F,
            "delta_T_F":   T_op_F - T_hyd_F,
            "time_hours":  0.0,
            "time_minutes": 0.0,
            "message": "Already at or below hydrate T — inject inhibitor now.",
        }

    if T_op_F <= T_ambient_F + 1.0:
        # No driving force for cooldown
        return {
            "T_hydrate_F": T_hyd_F,
            "delta_T_F":   T_op_F - T_hyd_F,
            "time_hours":  float('inf'),
            "time_minutes": float('inf'),
            "message": "Fluid is already at ambient — won't cool further.",
        }

    # Lumped-capacitance: T(t) = T_amb + (T_op - T_amb) * exp(-t/tau)
    # tau = rho * cp * V / (U * A) = rho * cp * D / (4 * U)  [for a pipe per unit length]
    # Convert units: rho [lb/ft3], cp [BTU/lb/°F], D [ft], U [BTU/hr/ft2/°F]
    tau_hr = rho_fluid * cp_fluid * D_outer_ft / (4.0 * U_pipe)

    # Solve T(t) = T_hyd for t:
    # T_hyd = T_amb + (T_op - T_amb) * exp(-t/tau)
    # exp(-t/tau) = (T_hyd - T_amb) / (T_op - T_amb)
    ratio = (T_hyd_F - T_ambient_F) / (T_op_F - T_ambient_F)
    if ratio <= 0:
        # Hydrate T is below ambient — pipe cools to ambient but never reaches
        # hydrate-zone T, so it's "infinite" cooldown — actually safe.
        return {
            "T_hydrate_F": T_hyd_F,
            "delta_T_F":   T_op_F - T_hyd_F,
            "time_hours":  float('inf'),
            "time_minutes": float('inf'),
            "message": (f"Hydrate T ({T_hyd_F:.1f}°F) is below ambient "
                         f"({T_ambient_F:.1f}°F). Pipe cannot cool below ambient — "
                         f"no hydrate risk from cooldown alone."),
        }

    t_hr = -tau_hr * np.log(ratio)
    return {
        "T_hydrate_F": T_hyd_F,
        "delta_T_F":   T_op_F - T_hyd_F,
        "tau_hours":   tau_hr,
        "time_hours":  t_hr,
        "time_minutes": t_hr * 60.0,
        "message": (f"Cooldown time to hydrate zone: {t_hr:.2f} hours "
                     f"({t_hr*60:.0f} minutes). Pipe time constant τ = "
                     f"{tau_hr:.2f} hr."),
    }


def cooldown_curve(T_op_F, T_ambient_F, U_pipe, D_outer_ft, rho_fluid,
                    cp_fluid, t_end_hours=24.0, n_points=100):
    """
    Generate T vs time curve for the lumped-capacitance cooldown.

    Returns (times_hours, temps_F) for plotting.
    """
    tau_hr = rho_fluid * cp_fluid * D_outer_ft / (4.0 * U_pipe)
    times = np.linspace(0, t_end_hours, n_points)
    temps = T_ambient_F + (T_op_F - T_ambient_F) * np.exp(-times / tau_hr)
    return times, temps


def inhibitor_concentration_hammerschmidt(T_shift_F, inhibitor="methanol"):
    """
    Hammerschmidt equation: concentration of inhibitor (wt %) needed to depress
    the hydrate-formation temperature by `T_shift_F` °F.

        d = (K_H * W) / (M * (100 - W))    where W = wt% inhibitor

    Rearranged to solve for W:
        W = (d * M * 100) / (K_H + d * M)
    where d = T_shift in °F, M = MW of inhibitor, K_H = Hammerschmidt constant.

    Typical K_H values:
        methanol     : 2335
        ethylene glycol (EG)  : 2222
        triethylene glycol (TEG) : 4000  (used mainly as desiccant, less common as inhibitor)
        MEG (mono-EG): 2222

    Returns the weight % of inhibitor required.
    """
    inhibitors = {
        "methanol":  {"M": 32.04,  "K_H": 2335},
        "MEG":       {"M": 62.07,  "K_H": 2222},
        "DEG":       {"M": 106.12, "K_H": 4000},
        "TEG":       {"M": 150.17, "K_H": 5400},
    }
    if inhibitor not in inhibitors:
        return np.nan
    M = inhibitors[inhibitor]["M"]
    K_H = inhibitors[inhibitor]["K_H"]
    W = (T_shift_F * M * 100.0) / (K_H + T_shift_F * M)
    return W


# ======================================================================
# SALINITY — produced-water as a natural hydrate inhibitor
# ======================================================================
def salinity_hydrate_shift(salinity_wt_pct):
    """Hydrate-temperature depression (deg F) from dissolved salt.

    Formation brine is a natural thermodynamic inhibitor: dissolved
    NaCl-equivalent salt lowers the hydrate-formation temperature, just
    as methanol or glycol does. The Hammerschmidt equation applies with
    the salt treated as the inhibitor:

        d = K_H * W / (M * (100 - W))

    with W the salt weight percent. For NaCl, M = 58.44 and a commonly
    used Hammerschmidt constant is K_H = 2335 (the same magnitude as
    methanol — salt is a comparably effective inhibitor per mole).

    Returns the temperature shift in deg F (a positive number — the
    hydrate temperature is LOWERED by this much). Capped at a sensible
    maximum because the Hammerschmidt form is only valid for modest
    concentrations (roughly up to 20 wt%).
    """
    W = max(0.0, min(salinity_wt_pct, 25.0))
    if W <= 0.0:
        return 0.0
    M_salt = 58.44      # NaCl
    K_H = 2335.0
    d = K_H * W / (M_salt * (100.0 - W))
    return d


def hydrate_temperature_with_salinity(P_psia, gas_sg, H2S_frac=0.0,
                                       CO2_frac=0.0, salinity_wt_pct=0.0):
    """Hydrate-formation temperature corrected for brine salinity.

    Computes the base Makogon hydrate temperature, then subtracts the
    salinity depression. A salty produced-water stream is genuinely less
    hydrate-prone than fresh water, so ignoring salinity makes a tool
    conservative — it would over-predict inhibitor demand.

    Returns the corrected hydrate temperature (deg F), or NaN if the base
    correlation is out of range.
    """
    T_base = hydrate_temperature_makogon(P_psia, gas_sg, H2S_frac, CO2_frac)
    if np.isnan(T_base):
        return np.nan
    return T_base - salinity_hydrate_shift(salinity_wt_pct)


# ======================================================================
# INHIBITOR INJECTION RATE — turn a required wt% into a volume rate
# ======================================================================
# Density of common inhibitors, lb/gal (at ~60 F).
_INHIBITOR_DENSITY_LB_GAL = {
    "methanol": 6.59,
    "MEG":      9.28,
    "DEG":      9.34,
    "TEG":      9.40,
}


def inhibitor_injection_rate(T_shift_F, water_rate_bbl_d,
                              inhibitor="methanol",
                              salinity_wt_pct=0.0):
    """Required inhibitor injection rate (gal/day) for a flowline.

    The Hammerschmidt equation gives the inhibitor weight percent W
    needed in the AQUEOUS phase. To turn that into an injection rate we
    do a mass balance on the water + inhibitor mixture: the inhibitor
    must make up fraction W of the combined aqueous stream.

        m_inhibitor / (m_water + m_inhibitor) = W / 100

    so   m_inhibitor = m_water * W / (100 - W).

    T_shift_F        : required hydrate-temperature depression (deg F)
    water_rate_bbl_d : produced free-water rate (STB/d)
    inhibitor        : 'methanol', 'MEG', 'DEG' or 'TEG'
    salinity_wt_pct  : salt already present in the produced water — it
                       contributes to inhibition, so the inhibitor only
                       has to supply the REMAINING shift.

    Returns a dict:
        wt_pct_required : inhibitor wt% in the aqueous phase
        rate_gal_day    : inhibitor injection rate, gallons/day
        rate_bbl_day    : same, in barrels/day
        net_shift_F     : the shift the inhibitor must supply after
                          crediting salinity
        note            : explanation / caveat
    """
    if inhibitor not in _INHIBITOR_DENSITY_LB_GAL:
        return {"wt_pct_required": np.nan, "rate_gal_day": np.nan,
                "rate_bbl_day": np.nan, "net_shift_F": np.nan,
                "note": f"Unknown inhibitor '{inhibitor}'."}

    # Credit any salinity already present.
    salt_shift = salinity_hydrate_shift(salinity_wt_pct)
    net_shift = max(0.0, T_shift_F - salt_shift)
    if net_shift <= 0.0:
        return {"wt_pct_required": 0.0, "rate_gal_day": 0.0,
                "rate_bbl_day": 0.0, "net_shift_F": 0.0,
                "note": ("The produced-water salinity alone supplies the "
                         "required hydrate-temperature depression — no "
                         "inhibitor is needed for this duty.")}

    W = inhibitor_concentration_hammerschmidt(net_shift, inhibitor)
    if np.isnan(W) or W >= 99.0:
        return {"wt_pct_required": W, "rate_gal_day": np.nan,
                "rate_bbl_day": np.nan, "net_shift_F": net_shift,
                "note": ("The required depression is too large for the "
                         "Hammerschmidt model — consider a different "
                         "strategy (heating, insulation).")}

    # Mass balance. Water density ~ 350 lb/bbl (62.4 lb/ft3 * 5.615).
    rho_water_lb_bbl = 350.0
    m_water = water_rate_bbl_d * rho_water_lb_bbl          # lb/day
    m_inhibitor = m_water * W / (100.0 - W)                # lb/day
    rho_inh = _INHIBITOR_DENSITY_LB_GAL[inhibitor]         # lb/gal
    rate_gal = m_inhibitor / rho_inh
    rate_bbl = rate_gal / 42.0

    return {"wt_pct_required": W, "rate_gal_day": rate_gal,
            "rate_bbl_day": rate_bbl, "net_shift_F": net_shift,
            "note": ("Injection rate to keep the aqueous phase at the "
                     "required inhibitor concentration. Add a design "
                     "margin and account for inhibitor lost to the gas "
                     "and hydrocarbon-liquid phases.")}


# ======================================================================
# ALTERNATIVE HYDRATE CORRELATION — Towler & Mokhatab (cross-check)
# ======================================================================
def hydrate_temperature_towler(P_psia, gas_sg):
    """Hydrate-formation temperature from the Towler & Mokhatab (2005)
    correlation — an explicit gas-gravity correlation, useful as an
    independent cross-check on the Makogon result.

        T(F) = 13.47 ln(P) + 34.27 ln(SG) - 1.675 [ln(P) ln(SG)] - 20.35

    with P in psia. Valid for sweet natural gas over roughly 0.55-0.90
    gravity and typical pipeline pressures. Returns deg F, or NaN for
    non-physical input.
    """
    if P_psia <= 1.0 or gas_sg <= 0.0:
        return np.nan
    g = max(0.55, min(gas_sg, 0.90))
    lnP = float(np.log(P_psia))
    lnG = float(np.log(g))
    T = 13.47 * lnP + 34.27 * lnG - 1.675 * lnP * lnG - 20.35
    return T


def hydrate_temperature_consensus(P_psia, gas_sg, H2S_frac=0.0,
                                    CO2_frac=0.0, salinity_wt_pct=0.0):
    """Return both correlations and their spread, so the user sees the
    model uncertainty rather than a single false-precision number.

    Returns a dict:
        T_makogon, T_towler : the two correlation results (deg F),
                              both salinity-corrected
        T_mean              : their average
        spread_F            : absolute difference (a rough uncertainty
                              band)
    """
    salt = salinity_hydrate_shift(salinity_wt_pct)
    t_mak = hydrate_temperature_makogon(P_psia, gas_sg, H2S_frac, CO2_frac)
    t_tow = hydrate_temperature_towler(P_psia, gas_sg)
    if not np.isnan(t_mak):
        t_mak -= salt
    if not np.isnan(t_tow):
        # Towler is sweet-gas only; apply the same salinity credit.
        t_tow -= salt
    vals = [v for v in (t_mak, t_tow) if not np.isnan(v)]
    t_mean = sum(vals) / len(vals) if vals else np.nan
    spread = (abs(t_mak - t_tow)
              if (not np.isnan(t_mak) and not np.isnan(t_tow))
              else np.nan)
    return {"T_makogon": t_mak, "T_towler": t_tow,
            "T_mean": t_mean, "spread_F": spread}


# ======================================================================
# SUBCOOLING — hydrate-formation DRIVING FORCE, not just in/out
# ======================================================================
def subcooling_risk(operating_T_F, hydrate_T_F):
    """Classify hydrate risk by SUBCOOLING — how far below the hydrate
    temperature the fluid sits — rather than a binary inside/outside.

    Subcooling (delta-T_sub = T_hydrate - T_operating) is the
    thermodynamic driving force for hydrate formation. A fluid 1-2 F
    inside the envelope nucleates slowly and may never plug; a fluid
    15-20 F inside forms hydrates fast. Reporting subcooling gives a far
    more useful picture than 'in' or 'out'.

    Returns a dict: {subcooling_F, level, message}.
    A negative subcooling means the fluid is OUTSIDE the hydrate region.
    """
    if np.isnan(hydrate_T_F):
        return {"subcooling_F": np.nan, "level": "Unknown",
                "message": "Hydrate temperature could not be evaluated."}
    sub = hydrate_T_F - operating_T_F
    if sub <= 0.0:
        level = "None"
        msg = ("The fluid is above the hydrate-formation temperature — "
               "outside the hydrate region, no driving force.")
    elif sub < 3.0:
        level = "Low"
        msg = ("Only a few degrees of subcooling — hydrates are "
               "thermodynamically possible but nucleation is slow. "
               "Often tolerable for short exposures, but not a steady-"
               "state design condition.")
    elif sub < 10.0:
        level = "Moderate"
        msg = ("Moderate subcooling — hydrates will form at a "
               "meaningful rate. Inhibition or heat retention is "
               "needed for continuous operation.")
    else:
        level = "High"
        msg = ("Large subcooling — strong driving force, rapid hydrate "
               "formation and a real plugging risk. Active mitigation "
               "is required.")
    return {"subcooling_F": sub, "level": level, "message": msg}


# ======================================================================
# STEADY-STATE FLOWLINE HYDRATE CHECK
# ======================================================================
#
# A producing flowline loses heat to its surroundings as the fluid travels
# from inlet to outlet. For steady-state plug flow the bulk temperature
# decays exponentially with distance:
#
#     T(x) = T_amb + (T_in - T_amb) * exp( -U * P_perim * x / (m_dot * cp) )
#
# where U is the overall heat-transfer coefficient, P_perim = pi * D_outer
# is the heat-transfer perimeter, m_dot is the mass flow rate and cp the
# fluid heat capacity. A pipeline that climbs (TVD change) also sees the
# ambient temperature change with depth via a geothermal/sea gradient.
#
# The hydrate question: at the operating pressure the fluid must stay
# ABOVE the hydrate-formation temperature everywhere along the line.
# Because lower flow rate => longer residence time => more cooling, there
# is a MINIMUM flow rate below which the outlet (or some midpoint) crosses
# into the hydrate region. These functions find that minimum.
#
# This is a screening calculation: single-phase plug flow, constant U and
# cp, no Joule-Thomson term, no transient/shut-in behaviour. It is meant
# to size a turndown limit, not to replace a transient flow-assurance
# (e.g. OLGA) study.

import math as _math


def flowline_temperature_profile(stations, T_in_F, m_dot_lbhr, cp,
                                  U, D_outer_ft, ambient_profile):
    """March the steady-state temperature along a flowline.

    stations        : list of cumulative distances along the line (ft),
                      increasing, starting at 0 (the inlet).
    T_in_F          : inlet fluid temperature, deg F.
    m_dot_lbhr      : mass flow rate, lb/hr.
    cp              : fluid heat capacity, BTU/(lb.F).
    U               : overall heat-transfer coefficient,
                      BTU/(hr.ft^2.F).
    D_outer_ft      : pipe outer diameter used for the heat-transfer
                      area, ft.
    ambient_profile : list of ambient temperatures (deg F), one per
                      station — lets the ambient vary with TVD.

    Returns a list of fluid temperatures (deg F), one per station.
    Uses the exact exponential solution over each segment, with the
    segment ambient taken as the mean of its two endpoints.
    """
    n = len(stations)
    if n == 0:
        return []
    if m_dot_lbhr <= 0 or cp <= 0:
        return [T_in_F] * n
    perim = _math.pi * D_outer_ft           # ft of perimeter per ft length
    T = [T_in_F]
    for i in range(1, n):
        dx = stations[i] - stations[i - 1]
        T_amb_seg = 0.5 * (ambient_profile[i] + ambient_profile[i - 1])
        # Exponential decay toward the segment ambient.
        k = U * perim * dx / (m_dot_lbhr * cp)
        T_next = T_amb_seg + (T[-1] - T_amb_seg) * _math.exp(-k)
        T.append(T_next)
    return T


def flowline_hydrate_margin(stations, T_profile, P_op_psia, gas_sg,
                             H2S_frac=0.0, CO2_frac=0.0):
    """Compare a flowline temperature profile to the hydrate temperature.

    Returns a dict:
        T_hyd        : hydrate-formation temperature at P_op (deg F)
        margins      : T_fluid - T_hyd at each station (deg F)
        min_margin   : the worst (smallest) margin along the line
        min_station  : distance at which the worst margin occurs (ft)
        safe         : True if the whole line stays above the hydrate T
    A negative margin means the fluid is inside the hydrate region.
    """
    T_hyd = hydrate_temperature_makogon(P_op_psia, gas_sg,
                                         H2S_frac, CO2_frac)
    if np.isnan(T_hyd):
        return {"T_hyd": np.nan, "margins": [np.nan] * len(stations),
                "min_margin": np.nan, "min_station": None, "safe": None}
    margins = [t - T_hyd for t in T_profile]
    i_min = int(np.argmin(margins))
    return {"T_hyd": T_hyd, "margins": margins,
            "min_margin": margins[i_min],
            "min_station": stations[i_min],
            "safe": margins[i_min] > 0.0}


def minimum_flow_no_hydrate(stations, T_in_F, cp, U, D_outer_ft,
                            ambient_profile, P_op_psia, gas_sg,
                            H2S_frac=0.0, CO2_frac=0.0,
                            m_lo=1.0e2, m_hi=1.0e8):
    """Find the minimum mass flow rate (lb/hr) that keeps the whole
    flowline above the hydrate-formation temperature.

    Higher flow => warmer line, so the 'safe' condition is monotonic in
    flow rate and a bisection on m_dot is well posed.

    Returns a dict:
        m_dot_min   : minimum safe mass flow rate (lb/hr), or None if even
                      the highest tested rate is unsafe, or 0.0 if even a
                      very low rate is already safe.
        feasible    : True if a finite minimum was found.
        note        : a short explanation.
    The search brackets [m_lo, m_hi]; widen them via the arguments if a
    line is extreme.
    """
    def _safe(m):
        prof = flowline_temperature_profile(stations, T_in_F, m, cp,
                                             U, D_outer_ft,
                                             ambient_profile)
        res = flowline_hydrate_margin(stations, prof, P_op_psia, gas_sg,
                                       H2S_frac, CO2_frac)
        return res["safe"]

    hyd_T = hydrate_temperature_makogon(P_op_psia, gas_sg,
                                         H2S_frac, CO2_frac)
    if np.isnan(hyd_T):
        return {"m_dot_min": None, "feasible": False,
                "note": ("No hydrate temperature could be evaluated at "
                         "this pressure — outside the correlation range.")}
    # If the inlet itself is below the hydrate T the line can never be
    # made safe by flowing faster.
    if T_in_F <= hyd_T:
        return {"m_dot_min": None, "feasible": False,
                "note": ("The inlet temperature is already at or below "
                         "the hydrate-formation temperature — no flow "
                         "rate can keep this line out of the hydrate "
                         "region. Inlet heating or inhibitor is needed.")}
    if _safe(m_lo):
        return {"m_dot_min": 0.0, "feasible": True,
                "note": ("Even a very low flow rate stays above the "
                         "hydrate temperature — the line is not "
                         "turndown-limited by hydrates.")}
    if not _safe(m_hi):
        return {"m_dot_min": None, "feasible": False,
                "note": ("Even a very high flow rate does not keep the "
                         "line safe — check the U-value, length, or "
                         "consider inhibitor / insulation.")}
    # Bisection: find the lowest m_dot that is still safe.
    for _ in range(60):
        m_mid = _math.sqrt(m_lo * m_hi)   # geometric midpoint
        if _safe(m_mid):
            m_hi = m_mid
        else:
            m_lo = m_mid
        if m_hi / m_lo < 1.0005:
            break
    return {"m_dot_min": m_hi, "feasible": True,
            "note": ("Minimum flow rate to keep the whole line above the "
                     "hydrate-formation temperature.")}


def interpolate_ambient_from_survey(stations, survey_md, survey_tvd,
                                     T_surface_F, geo_gradient_F_per_ft):
    """Build an ambient-temperature profile from a pipeline survey.

    survey_md   : measured depths / cumulative distances of survey points
    survey_tvd  : true vertical depth at each survey point (ft, positive
                  downward)
    T_surface_F : ambient temperature at TVD = 0 (seabed mudline or
                  surface)
    geo_gradient_F_per_ft : geothermal / sea-temperature gradient with
                  depth (deg F per ft of TVD)

    Returns an ambient temperature for each entry in `stations` by linear
    interpolation of TVD against measured depth, then applying the
    gradient. This lets a climbing or descending line see a depth-varying
    ambient.
    """
    amb = []
    for s in stations:
        tvd = float(np.interp(s, survey_md, survey_tvd))
        amb.append(T_surface_F + geo_gradient_F_per_ft * tvd)
    return amb


# ======================================================================
# FLOWLINE PRESSURE LOSS  &  SLUG-FLOW SCREENING
# ======================================================================
#
# The temperature model above tells you whether a line stays warm enough
# to avoid hydrates. Two more questions matter for a real flowline:
#
#   1. PRESSURE LOSS — how much pressure is consumed friction + elevation
#      along the line. If the available drawdown cannot overcome it, the
#      line will not flow at the assumed rate.
#
#   2. SLUGGING — in two-phase flow the liquid and gas can separate into
#      alternating slugs of liquid and gas pockets. Slugs cause pressure
#      and rate surges, hydrate-prone cold spots, and separator upsets.
#
# The models here are screening-level:
#   * single-phase friction via the Darcy-Weisbach equation with a
#     Colebrook-White (Haaland-approximated) friction factor;
#   * a homogeneous two-phase option using a mixture density/viscosity;
#   * a Froude-number based flow-pattern indicator and a simple terrain-
#     slugging severity index.
# For design work a Beggs-Brill or OLGA calculation is the proper tool;
# these are meant to flag whether a detailed study is needed.


def _haaland_friction_factor(reynolds, rel_roughness):
    """Darcy friction factor via the Haaland (1983) explicit
    approximation to the Colebrook-White equation. Valid for turbulent
    flow; laminar flow falls back to 64/Re."""
    if reynolds <= 0:
        return 0.0
    if reynolds < 2300.0:
        return 64.0 / reynolds
    inv_sqrt = -1.8 * _math.log10(
        (rel_roughness / 3.7) ** 1.11 + 6.9 / reynolds)
    if inv_sqrt == 0:
        return 0.02
    return (1.0 / inv_sqrt) ** 2


def single_phase_pressure_drop(q_bbl_d, rho_lb_ft3, mu_cp,
                                D_in_inch, length_ft, rel_roughness=0.0006,
                                elevation_change_ft=0.0):
    """Single-phase liquid pressure drop along a flowline (psi).

    q_bbl_d             : volumetric flow rate, STB/d
    rho_lb_ft3          : fluid density, lb/ft3
    mu_cp               : fluid viscosity, cP
    D_in_inch           : inner diameter, inch
    length_ft           : pipe length, ft
    rel_roughness       : pipe roughness / diameter (default ~commercial
                          steel)
    elevation_change_ft : outlet TVD minus inlet TVD (negative if the
                          line climbs); adds a hydrostatic term

    Returns a dict with the friction, elevation and total pressure drop,
    plus the flow velocity, Reynolds number and friction factor.
    """
    D_ft = D_in_inch / 12.0
    area = _math.pi * D_ft ** 2 / 4.0
    # STB/d -> ft3/s
    q_ft3_s = q_bbl_d * 5.615 / 86400.0
    v = q_ft3_s / area if area > 0 else 0.0          # ft/s

    # Reynolds number (field units): with rho in lb/ft3, v in ft/s,
    # D in ft and mu in cP, Re = 1488 * rho * v * D / mu.
    Re = 1488.0 * rho_lb_ft3 * v * D_ft / max(mu_cp, 1e-9)
    f = _haaland_friction_factor(Re, rel_roughness)

    # Darcy-Weisbach: dP = f * (L/D) * rho * v^2 / 2  ; convert to psi
    # rho in lb/ft3, v in ft/s -> dP in lbf/ft2 then /144 -> psi
    g_c = 32.174
    dP_fric_psf = f * (length_ft / D_ft) * rho_lb_ft3 * v ** 2 / (2.0 * g_c)
    dP_fric = dP_fric_psf / 144.0

    # Hydrostatic term: positive elevation_change (going down) adds
    # pressure at the outlet, i.e. helps; here we report it as the
    # pressure consumed (so a climb costs pressure).
    dP_elev = rho_lb_ft3 * elevation_change_ft / 144.0

    return {"v_ft_s": v, "reynolds": Re, "friction_factor": f,
            "dP_friction_psi": dP_fric, "dP_elevation_psi": dP_elev,
            "dP_total_psi": dP_fric + dP_elev}


def two_phase_pressure_drop(q_liq_bbl_d, q_gas_mscf_d, rho_liq, rho_gas,
                             mu_liq, mu_gas, D_in_inch, length_ft,
                             P_op_psia, T_op_F, Z_gas=0.9,
                             rel_roughness=0.0006, elevation_change_ft=0.0):
    """Homogeneous (no-slip) two-phase pressure drop along a flowline.

    A screening model: gas and liquid are treated as a single mixture
    with volume-averaged density and viscosity. This is reasonable for
    dispersed-bubble or mist flow but UNDER-predicts the drop when the
    flow is slugging — see slug_flow_assessment for that warning.

    q_liq_bbl_d   : liquid rate, STB/d
    q_gas_mscf_d  : gas rate, Mscf/d (standard conditions)
    rho_liq/gas   : phase densities at LINE conditions, lb/ft3
    mu_liq/gas    : phase viscosities, cP
    P_op_psia     : operating pressure — used to convert the gas rate
                    from standard to actual (line) volume.
    T_op_F        : operating temperature, deg F.
    Z_gas         : gas compressibility factor at line conditions.

    The gas standard volume is converted to actual volume with the real-
    gas law: V_act = V_std * (P_sc/P) * (T/T_sc) * Z. Without this the
    gas volume — and the velocity — would be hugely overstated.

    Returns the pressure-drop dict plus the no-slip holdup, mixture
    density and mixture velocity.
    """
    D_ft = D_in_inch / 12.0
    area = _math.pi * D_ft ** 2 / 4.0

    # Liquid: STB/d -> ft3/s
    q_l = q_liq_bbl_d * 5.615 / 86400.0

    # Gas: Mscf/d (standard) -> actual ft3/s at line P, T.
    # Standard conditions: 14.696 psia, 519.67 R (60 F).
    T_op_R = T_op_F + 459.67
    P_sc, T_sc = 14.696, 519.67
    q_g_std = q_gas_mscf_d * 1000.0 / 86400.0          # std ft3/s
    q_g = q_g_std * (P_sc / max(P_op_psia, 1.0)) * (T_op_R / T_sc) * Z_gas

    q_tot = q_l + q_g
    lam_l = q_l / q_tot if q_tot > 0 else 0.0
    v_m = q_tot / area if area > 0 else 0.0

    rho_m = rho_liq * lam_l + rho_gas * (1.0 - lam_l)
    mu_m = mu_liq * lam_l + mu_gas * (1.0 - lam_l)

    Re = 1488.0 * rho_m * v_m * D_ft / max(mu_m, 1e-9)
    f = _haaland_friction_factor(Re, rel_roughness)

    g_c = 32.174
    dP_fric_psf = f * (length_ft / D_ft) * rho_m * v_m ** 2 / (2.0 * g_c)
    dP_fric = dP_fric_psf / 144.0
    dP_elev = rho_m * elevation_change_ft / 144.0

    return {"v_ft_s": v_m, "reynolds": Re, "friction_factor": f,
            "no_slip_holdup": lam_l, "mixture_density": rho_m,
            "q_gas_actual_ft3_s": q_g,
            "dP_friction_psi": dP_fric, "dP_elevation_psi": dP_elev,
            "dP_total_psi": dP_fric + dP_elev}


def slug_flow_assessment(q_liq_bbl_d, q_gas_mscf_d, rho_liq, rho_gas,
                          D_in_inch, P_op_psia, T_op_F, Z_gas=0.9,
                          pipe_inclination_deg=0.0):
    """Screening assessment of slug-flow likelihood and severity.

    Uses superficial velocities and a Froude number to place the flow on
    a simplified pattern map, and flags terrain/hydrodynamic slugging.

    q_liq_bbl_d         : liquid rate, STB/d
    q_gas_mscf_d        : gas rate, Mscf/d (standard conditions)
    rho_liq/gas         : phase densities, lb/ft3
    D_in_inch           : inner diameter, inch
    P_op_psia, T_op_F   : operating pressure / temperature — the gas
                          rate is converted from standard to actual
                          (line) volume, without which the superficial
                          gas velocity would be hugely overstated.
    Z_gas               : gas compressibility factor at line conditions.
    pipe_inclination_deg: pipe inclination from horizontal (deg). Uphill
                          sections are far more slug-prone.

    Returns a dict:
        v_sl, v_sg      : superficial liquid / gas velocities (ft/s)
        v_mix           : mixture velocity (ft/s)
        froude          : mixture Froude number
        regime          : screening flow-pattern label
        slug_risk       : 'Low' / 'Moderate' / 'High'
        slug_length_est : rough slug length estimate (pipe diameters)
        message         : explanation
    """
    D_ft = D_in_inch / 12.0
    area = _math.pi * D_ft ** 2 / 4.0
    q_l = q_liq_bbl_d * 5.615 / 86400.0
    # Gas: standard Mscf/d -> actual ft3/s at line conditions.
    T_op_R = T_op_F + 459.67
    P_sc, T_sc = 14.696, 519.67
    q_g_std = q_gas_mscf_d * 1000.0 / 86400.0
    q_g = q_g_std * (P_sc / max(P_op_psia, 1.0)) * (T_op_R / T_sc) * Z_gas
    v_sl = q_l / area if area > 0 else 0.0
    v_sg = q_g / area if area > 0 else 0.0
    v_m = v_sl + v_sg

    g = 32.174  # ft/s2
    froude = v_m / _math.sqrt(g * D_ft) if D_ft > 0 else 0.0
    lam_l = v_sl / v_m if v_m > 0 else 0.0

    # Simplified flow-pattern screening (horizontal-to-slightly-inclined):
    #   - very low mixture velocity + appreciable liquid -> stratified
    #   - moderate velocity with both phases present     -> slug / intermittent
    #   - high gas fraction + high velocity              -> annular / mist
    #   - high liquid fraction + high velocity           -> dispersed bubble
    if v_m < 1.0 and lam_l > 0.1:
        regime = "Stratified"
    elif lam_l > 0.9:
        regime = "Dispersed bubble / single-phase liquid"
    elif lam_l < 0.05 and v_sg > 15.0:
        regime = "Annular / mist"
    elif 0.1 <= froude <= 30.0 and 0.05 < lam_l < 0.9:
        regime = "Slug / intermittent"
    else:
        regime = "Transitional"

    # Slug risk: intermittent regime is the obvious one, but an uphill
    # inclination strongly promotes terrain slugging even at low Froude.
    uphill = pipe_inclination_deg > 1.0
    steep_uphill = pipe_inclination_deg > 5.0

    if regime.startswith("Slug"):
        if steep_uphill:
            risk = "High"
            msg = ("Intermittent (slug) flow on a steeply rising line — "
                   "both hydrodynamic and terrain slugging are likely. "
                   "Expect pressure/rate surges and cold spots; a "
                   "transient (OLGA) study and slug-catcher sizing are "
                   "warranted.")
        else:
            risk = "Moderate"
            msg = ("Intermittent (slug) flow predicted. Slugs will cause "
                   "pressure and rate fluctuations at the outlet — size "
                   "downstream equipment for the surge.")
    elif regime == "Stratified" and steep_uphill:
        risk = "High"
        msg = ("Low-velocity stratified flow on a steep uphill section — "
               "the classic terrain-slugging geometry. Liquid can "
               "accumulate in the dip and discharge as severe slugs.")
    elif regime == "Stratified" and uphill:
        risk = "Moderate"
        msg = ("Stratified flow with some uphill inclination — terrain "
               "slugging is possible at low rates. Watch turndown.")
    elif regime == "Transitional":
        risk = "Moderate"
        msg = ("Flow is near a regime boundary — small changes in rate "
               "could push it into slugging. Treat with caution.")
    else:
        risk = "Low"
        msg = (f"{regime} flow — slugging is unlikely at these "
               f"conditions.")

    # Rough slug-length estimate (Scott et al. style: slug length grows
    # with pipe diameter; typical 10-30 D for hydrodynamic slugs).
    if regime.startswith("Slug"):
        slug_len = 20.0 * (1.0 + min(froude / 10.0, 2.0))
    else:
        slug_len = 0.0

    return {"v_sl_ft_s": v_sl, "v_sg_ft_s": v_sg, "v_mix_ft_s": v_m,
            "froude": froude, "no_slip_holdup": lam_l,
            "regime": regime, "slug_risk": risk,
            "slug_length_est_D": slug_len, "message": msg}
