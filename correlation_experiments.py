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
