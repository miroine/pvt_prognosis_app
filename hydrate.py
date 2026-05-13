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

    # Beta from Makogon
    beta = 2.681 - 3.811 * g + 1.679 * g ** 2
    # T0 = 273.15 K reference; T_C already relative to 0°C => Makogon form
    # log10(P) at T_C:
    log10_P_MPa = beta + 0.0497 * T_C + 0.00034 * T_C ** 2
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
