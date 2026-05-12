"""
Phase envelope computation for a fluid of fixed composition.

Strategy:
  For a range of temperatures, find both the bubble pressure (highest P with
  V>0 in flash) and the dew pressure (lowest P with V<1, going from above).
  As T rises toward the cricondentherm/critical, the two loci converge.

  We use the existing eos_pr.saturation_pressure with a smart bracketing
  scheme that does separate searches for the bubble (high-pressure 2-phase
  boundary) and the dew (high-pressure 2-phase boundary from the gas side).

  Note: for a single composition the bubble and dew points coincide with
  the same 2-phase boundary at a given T — what we call "bubble vs dew"
  depends on which side (oil-rich vs gas-rich) the fluid is on at the
  saturation pressure. The envelope is a single closed loop in (P,T) space.
"""

import numpy as np
from eos_pr import flash, saturation_pressure


def trace_envelope(z, comp_names, c7_props=None,
                   T_min=None, T_max=None, n_points=30, P_max=15000.0):
    """
    Trace the phase envelope.

    For each T, find a single saturation pressure (bubble or dew depending on
    where the feed sits). To build the closed loop, we sample two branches:
      - Low-T branch: search bubble side
      - High-T branch: search dew side
    They meet at the critical / cricondentherm region.

    Returns:
        dict with keys:
            T_bubble, P_bubble  – bubble locus
            T_dew,    P_dew     – dew locus
            T_critical_est, P_critical_est – approximate critical point
    """
    z = np.asarray(z, dtype=float); z = z / z.sum()

    # Pick a reasonable T range from the components
    from components import get_props
    Tc_arr = np.array([get_props(c, c7_props)["Tc"] for c in comp_names])
    Tc_min = Tc_arr.min()
    Tc_max = Tc_arr.max()
    if T_min is None:
        T_min = max(0.5 * Tc_min, 360.0)   # at least ~−100°F
    if T_max is None:
        T_max = min(0.95 * Tc_max, 1500.0)

    temperatures = np.linspace(T_min, T_max, n_points)

    P_sat_arr = np.full(len(temperatures), np.nan)
    phase_at_sat = []

    for i, T in enumerate(temperatures):
        try:
            P_sat = _trace_saturation(z, comp_names, T, c7_props, P_max=P_max)
            if P_sat is not None and 14.7 < P_sat < P_max:
                P_sat_arr[i] = P_sat
                # Determine bubble vs dew by flashing just above and below
                r_above = flash(z, comp_names, P_sat * 1.02, T, c7_props)
                # If above is L => bubble (single-phase liquid above sat-P, gas appears below)
                # If above is V => dew
                phase_at_sat.append("bubble" if r_above["phase"] == "L" else "dew")
            else:
                phase_at_sat.append(None)
        except Exception:
            phase_at_sat.append(None)

    # Split into bubble and dew arrays
    T_bubble = []; P_bubble = []
    T_dew = []; P_dew = []
    for i, T in enumerate(temperatures):
        if not np.isfinite(P_sat_arr[i]):
            continue
        if phase_at_sat[i] == "bubble":
            T_bubble.append(T); P_bubble.append(P_sat_arr[i])
        elif phase_at_sat[i] == "dew":
            T_dew.append(T); P_dew.append(P_sat_arr[i])

    # Estimate critical point as where bubble and dew loci meet/converge
    T_crit = P_crit = None
    if T_bubble and T_dew:
        T_b_max = max(T_bubble); idx_b = T_bubble.index(T_b_max)
        T_d_min = min(T_dew);    idx_d = T_dew.index(T_d_min)
        # Critical lies between the two branches; midpoint is a fair estimate
        T_crit = 0.5 * (T_b_max + T_d_min)
        P_crit = 0.5 * (P_bubble[idx_b] + P_dew[idx_d])

    return {
        "T_bubble": np.array(T_bubble),
        "P_bubble": np.array(P_bubble),
        "T_dew": np.array(T_dew),
        "P_dew": np.array(P_dew),
        "T_critical_est": T_crit,
        "P_critical_est": P_crit,
        "T_range": (T_min, T_max),
    }


def _trace_saturation(z, comp_names, T, c7_props, P_max=15000.0):
    """
    Find the saturation pressure at a single T using the existing solver.
    Wraps saturation_pressure with two-sided kind="auto".
    """
    return saturation_pressure(z, comp_names, T, c7_props=c7_props,
                                kind="auto", P_lo=14.7, P_hi=P_max)
