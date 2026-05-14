"""
CCE and CVD experiments based on black-oil correlations (not EOS).

Black-oil CCE for OIL:
    At P >= Pb:  V/Vsat = Bo(P) / Bo(Pb)         (under-saturated compression)
    At P <  Pb:  V_total = Bo(P) + (Rsi - Rs(P)) * Bg(P)
                  Liquid_volume = Bo(P)
                  V/Vsat = V_total / Bo(Pb)

Black-oil CVD for WET GAS:
    Treat gas with vaporized oil Rv(P).
    Below Pdew, condensate drops out; integrating gives liquid volume per unit
    of original gas. For wet-gas correlations the dropout = (Rv_max - Rv(P))
    * Bo_cond(P) / Bg(Pdew) (approximate).
"""

import numpy as np
from correlations import OilCorrelations, GasCorrelations, WetGasCorrelations


def cce_blackoil(oil_corr, gas_corr, Rsi, Pb, pressures):
    """
    CCE simulation using oil + gas correlations.

    Args:
        oil_corr  : OilCorrelations instance
        gas_corr  : GasCorrelations instance (for free-gas Bg)
        Rsi       : initial solution GOR (scf/STB)
        Pb        : bubble point pressure (psia)
        pressures : array of pressures to evaluate at (psia)

    Returns list of rows with V/Vsat, liquid dropout %, phase identification.
    """
    Bo_sat = oil_corr.formation_volume_factor(Pb, Rsi, saturated=True)
    rows = []
    for P in sorted(pressures):
        if P >= Pb:
            # Under-saturated oil only
            Bo = oil_corr.formation_volume_factor(P, Rsi, saturated=False, Pb=Pb)
            V_rel = Bo / Bo_sat
            L_dropout = 100.0  # all liquid (treat Vsat as the reference)
            phase = "L"
            Rs = Rsi
        else:
            # Saturated: liberate gas
            Rs = oil_corr.solution_gor(P)
            Bo = oil_corr.formation_volume_factor(P, Rs, saturated=True)
            # Gas Bg at this pressure in rb/scf
            Z = gas_corr.z_factor(P) if hasattr(gas_corr, 'z_factor') else 1.0
            Bg = gas_corr.formation_volume_factor(P, Z) if hasattr(gas_corr, 'formation_volume_factor') else 0.001
            # Liberated gas per STB = (Rsi - Rs) scf/STB
            # Volume of liberated gas at (P,T) = (Rsi - Rs) * Bg rb/STB
            V_gas_liberated_rb_per_STB = (Rsi - Rs) * Bg
            V_total_rb_per_STB = Bo + V_gas_liberated_rb_per_STB
            V_rel = V_total_rb_per_STB / Bo_sat
            L_dropout = 100.0 * Bo / Bo_sat   # liquid as % of Vsat
            phase = "LV"

        # Y-function below Pb
        if P < Pb - 1 and V_rel > 1.001:
            Y = ((Pb / P) - 1.0) / (V_rel - 1.0)
        else:
            Y = np.nan

        rows.append({
            "P": P, "V_rel": V_rel, "L_dropout_pct": L_dropout,
            "Y_function": Y, "phase": phase, "Rs": Rs, "Bo": Bo,
        })
    return rows


def cvd_wetgas(wet_corr, Pdew, pressures):
    """
    CVD simulation for wet gas using the wet-gas correlation.

    Below Pdew, liquid dropout increases as condensate falls out of the gas.
    We approximate the dropout from the Rv-vs-P curve: as Rv drops, the
    "lost" condensate becomes liquid in the cell.

    Args:
        wet_corr  : WetGasCorrelations instance
        Pdew      : dew-point pressure (psia)
        pressures : array of pressures

    Returns list of rows.
    """
    # Reference: at Pdew, V_cell = V_sat (all gas)
    Z_dew = wet_corr.z_factor(Pdew)
    Bg_dew = wet_corr.formation_volume_factor(Pdew, Z_dew)  # rb/scf
    Rv_max = wet_corr.Rv_max   # STB/scf
    rows = []
    cum_produced = 0.0   # CVD removes excess; for correlation mode we just track liquid

    for P in sorted(pressures, reverse=True):
        if P >= Pdew:
            Z = wet_corr.z_factor(P)
            Bg = wet_corr.formation_volume_factor(P, Z)
            Rv = wet_corr.rv(P)
            mu = wet_corr.viscosity(P, Z)
            rows.append({
                "P": P, "Z": Z, "Bg": Bg * 1000.0,
                "Rv_produced": Rv * 1000.0,
                "L_dropout_pct": 0.0, "cum_produced_pct": cum_produced,
                "mu_g": mu, "phase": "V",
            })
        else:
            Z = wet_corr.z_factor(P)
            Bg = wet_corr.formation_volume_factor(P, Z)
            Rv = wet_corr.rv(P)
            mu = wet_corr.viscosity(P, Z)
            # Condensate lost per scf of original gas:
            # ΔRv = (Rv_max - Rv); volume of condensate at standard conditions = ΔRv STB
            # Convert STB to reservoir bbl using typical condensate FVF ~ 1.1
            Bo_cond = 1.1
            dropout_bbl_per_scf = (Rv_max - Rv) * Bo_cond
            # As % of Vsat (= Bg_dew per scf)
            L_dropout = 100.0 * dropout_bbl_per_scf / Bg_dew
            # Cumulative produced: approximate as (1 - Bg_dew/Bg) × some factor
            cum_produced += 0.1  # placeholder rate; real CVD needs proper bookkeeping
            rows.append({
                "P": P, "Z": Z, "Bg": Bg * 1000.0,
                "Rv_produced": Rv * 1000.0,
                "L_dropout_pct": L_dropout, "cum_produced_pct": min(cum_produced, 100),
                "mu_g": mu, "phase": "LV",
            })

    return sorted(rows, key=lambda r: r["P"])


# ============================================================
# CVD for black-oil (depletion below Pb)
# ============================================================
def cvd_blackoil(oil_corr, gas_corr, Rsi, Pb, pressures):
    """
    Constant-volume depletion approximation for a black oil.

    Tracks the cell as pressure is lowered below Pb. Liberated gas is
    'produced' (removed) to keep the cell volume constant; we report the
    cumulative produced gas and the remaining liquid volume fraction.

    Returns list of rows with P, Rs_remaining, liquid_frac, cum_gas_produced.
    """
    Bo_b = oil_corr.formation_volume_factor(Pb, Rsi, saturated=True)
    rows = []
    cum_gas_scf = 0.0   # cumulative gas removed, scf per STB of original oil
    prev_Rs = Rsi
    for P in sorted(pressures, reverse=True):
        if P >= Pb:
            Bo = oil_corr.formation_volume_factor(min(P, Pb), Rsi,
                                                    saturated=(P <= Pb), Pb=Pb)
            rows.append({
                "P": P, "Rs": Rsi, "Bo": Bo,
                "liquid_frac": 1.0,
                "cum_gas_produced_scfSTB": 0.0,
                "phase": "L",
            })
        else:
            Rs = oil_corr.solution_gor(P)
            Bo = oil_corr.formation_volume_factor(P, Rs, saturated=True)
            # gas liberated in this step
            d_gas = max(prev_Rs - Rs, 0.0)
            cum_gas_scf += d_gas
            prev_Rs = Rs
            liquid_frac = Bo / Bo_b
            rows.append({
                "P": P, "Rs": Rs, "Bo": Bo,
                "liquid_frac": liquid_frac,
                "cum_gas_produced_scfSTB": cum_gas_scf,
                "phase": "LV",
            })
    return sorted(rows, key=lambda r: r["P"])


# ============================================================
# Single-stage flash for a black oil (P_res -> standard conditions)
# ============================================================
def flash_blackoil(oil_corr, Rsi, Pb, P_initial):
    """
    Flash a black oil from reservoir P to standard conditions (single stage).

    For a black oil this is well-defined: at standard conditions all Rs
    comes out of solution.

    Returns dict with GOR, Bo_initial, shrinkage factor, API.
    """
    if P_initial >= Pb:
        Bo_initial = oil_corr.formation_volume_factor(P_initial, Rsi,
                                                        saturated=False, Pb=Pb)
    else:
        Rs_init = oil_corr.solution_gor(P_initial)
        Bo_initial = oil_corr.formation_volume_factor(P_initial, Rs_init,
                                                        saturated=True)
    # At standard conditions Bo = 1.0 by definition; all gas liberated
    GOR = Rsi  # scf/STB
    shrinkage = 1.0 / Bo_initial   # STB at SC per rb at reservoir
    return {
        "GOR_scfSTB": GOR,
        "Bo_initial": Bo_initial,
        "shrinkage": shrinkage,
        "Rs_initial": Rsi if P_initial >= Pb else oil_corr.solution_gor(P_initial),
    }


# ============================================================
# Multi-stage separator for a black oil (correlation-based)
# ============================================================
def multistage_separator_blackoil(oil_corr, gas_corr, Rsi, Pb, T_res, stages):
    """
    Multi-stage separator flash approximation for a black oil.

    For a correlation black oil we don't have a true composition, so we use
    the empirical observation that a multi-stage train reduces total GOR
    versus a single flash by 5-15% depending on stage count and pressures.
    Each stage is assigned a fraction of the total Rs based on stage pressure.

    stages: list of (P_psia, T_F) tuples, HP -> ST.

    Returns dict with per-stage GOR breakdown and total GOR.
    """
    if not stages:
        stages = [(14.7, 60.0)]

    # Total gas to be liberated from reservoir oil down to stock tank
    total_Rs = Rsi

    # Assign gas release per stage. Higher-pressure stages release the
    # lighter, larger fraction. Use a simple geometric split by pressure ratio.
    P_values = [s[0] for s in stages]
    # Weight: more gas released at higher pressure stages
    weights = []
    for i, P_s in enumerate(P_values):
        # Each stage releases gas proportional to the log-pressure drop into it
        if i == 0:
            w = np.log(max(Pb, P_s + 1) / P_s)
        else:
            w = np.log(max(P_values[i-1], P_s + 1) / P_s)
        weights.append(max(w, 0.05))
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    # Multi-stage efficiency factor: more stages -> slightly lower total GOR
    # (more liquid retained). 1 stage = 1.0, 2 stages ~0.95, 3 stages ~0.90
    n_st = len(stages)
    efficiency = {1: 1.0, 2: 0.95, 3: 0.91}.get(n_st, max(0.85, 1.0 - 0.05 * n_st))
    effective_total_GOR = total_Rs * efficiency

    stage_results = []
    for i, (P_s, T_s) in enumerate(stages):
        stage_GOR = effective_total_GOR * weights[i]
        # Stage liquid density estimate via API (rough — assumes stock tank API)
        stage_results.append({
            "stage": i + 1, "P": P_s, "T_F": T_s,
            "stage_GOR_scfSTB": stage_GOR,
            "fraction_of_total": weights[i],
        })

    # Stock tank oil API ~ from the dead-oil density
    api = oil_corr.api
    rho_st = 141.5 / (131.5 + api) * 62.428  # lb/ft3

    return {
        "stage_results": stage_results,
        "total_GOR_scfSTB": effective_total_GOR,
        "single_stage_GOR_scfSTB": total_Rs,
        "GOR_reduction_pct": (1 - efficiency) * 100,
        "st_oil_API": api,
        "st_oil_density": rho_st,
    }


# ============================================================
# CCE / CVD / flash for dry gas
# ============================================================
def cce_drygas(gas_corr, pressures):
    """
    CCE for dry gas — simply Z, Bg, and the gas expansion factor vs P.
    A dry gas has no liquid dropout so 'CCE' is just the expansion curve.
    """
    rows = []
    for P in sorted(pressures):
        Z = gas_corr.z_factor(P)
        Bg = gas_corr.formation_volume_factor(P, Z)
        mu = gas_corr.viscosity(P, Z)
        rows.append({
            "P": P, "Z": Z, "Bg": Bg * 1000.0, "mu_g": mu,
        })
    # Expansion factor relative to the lowest pressure
    if rows:
        Bg_max = max(r["Bg"] for r in rows)
        for r in rows:
            r["E_factor"] = Bg_max / r["Bg"] if r["Bg"] > 0 else np.nan
    return rows


def cvd_drygas(gas_corr, pressures, P_initial):
    """
    CVD for dry gas — cumulative gas recovery as P depletes.

    Recovery factor at pressure P is RF = 1 - (P/Z) / (P_i/Z_i)
    from the gas material balance for a volumetric dry-gas reservoir.
    """
    Z_i = gas_corr.z_factor(P_initial)
    pz_i = P_initial / Z_i
    rows = []
    for P in sorted(pressures, reverse=True):
        if P > P_initial:
            continue
        Z = gas_corr.z_factor(P)
        pz = P / Z
        RF = 1.0 - pz / pz_i
        rows.append({
            "P": P, "Z": Z, "P_over_Z": pz,
            "recovery_factor_pct": RF * 100.0,
            "Bg": gas_corr.formation_volume_factor(P, Z) * 1000.0,
        })
    return sorted(rows, key=lambda r: r["P"])


def flash_drygas(gas_corr, P_initial):
    """Flash dry gas to standard conditions — just the expansion factor."""
    Z_i = gas_corr.z_factor(P_initial)
    Bg_i = gas_corr.formation_volume_factor(P_initial, Z_i)  # rb/scf
    # Expansion: 1 rb reservoir gas -> 1/Bg scf at standard conditions
    expansion = 1.0 / Bg_i if Bg_i > 0 else np.nan
    return {
        "Z_initial": Z_i,
        "Bg_initial_rb_per_scf": Bg_i,
        "expansion_scf_per_rb": expansion,
    }
