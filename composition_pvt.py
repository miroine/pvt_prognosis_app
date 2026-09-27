"""
Black-oil table generation from a compositional fluid.

For an oil:
    Differential Liberation Experiment (DLE) at reservoir T:
        - At P >= Pb, fluid is single-phase liquid; properties from PR EOS
        - Below Pb, flash z at (P, T_res), strip the gas, keep the liquid as new feed
        - Bo = V_oil(P) / V_oil_sc, Rs = (cumulative gas removed) / V_oil_sc

For a gas condensate:
    Constant Volume Depletion (CVD):
        - At P >= Pdew, single-phase gas
        - Below Pdew, flash, vapor fraction = current gas; remove enough gas to keep V constant
        - Track Rv = liquid (STB) / gas (Mscf) of the produced gas re-flashed at standard conditions
"""

import numpy as np
from eos_pr import flash, pr_phase, phase_density
from lbc import lbc_viscosity
from components import get_props

T_SC = 60.0 + 460.0   # 520 R
P_SC = 14.7
SCF_PER_LBMOL = 379.49  # at standard conditions


def standard_conditions_split(comp_frac, comp_names, c7_props=None):
    """
    Flash a stream at standard conditions (60°F, 14.7 psia) and return
    (n_oil_lbmol, n_gas_lbmol, V_oil_bbl, V_gas_scf, x_oil, y_gas)
    per 1 lbmol of feed.
    """
    cf = np.asarray(comp_frac, dtype=float)
    s = cf.sum()
    if s <= 1e-12:
        # Empty stream
        n = len(comp_names)
        return 0.0, 0.0, 0.0, 0.0, np.zeros(n), np.zeros(n)
    cf = cf / s
    r = flash(cf, comp_names, P_SC, T_SC, c7_props)

    if r["phase"] == "L":
        rho = phase_density(comp_names, r["x"], r["Z_L"], P_SC, T_SC, c7_props)
        MW = np.array([get_props(c, c7_props)["MW"] for c in comp_names])
        Mavg = float(np.dot(r["x"], MW))
        V_oil_bbl = Mavg / rho / 5.615
        return 1.0, 0.0, V_oil_bbl, 0.0, r["x"], np.zeros_like(r["x"])

    if r["phase"] == "V":
        V_gas_scf = SCF_PER_LBMOL
        return 0.0, 1.0, 0.0, V_gas_scf, np.zeros_like(r["y"]), r["y"]

    V = r["V"]
    n_gas = V
    n_oil = 1.0 - V

    rho_oil = phase_density(comp_names, r["x"], r["Z_L"], P_SC, T_SC, c7_props)
    MW = np.array([get_props(c, c7_props)["MW"] for c in comp_names])
    M_oil = float(np.dot(r["x"], MW))
    V_oil_bbl = n_oil * M_oil / rho_oil / 5.615
    V_gas_scf = n_gas * SCF_PER_LBMOL
    return n_oil, n_gas, V_oil_bbl, V_gas_scf, r["x"], r["y"]


def black_oil_table_from_composition(z, comp_names, T_res, pressures,
                                     c7_props=None, fluid_kind="oil"):
    """
    Generate a black-oil table by depletion flash.
    Returns dict of arrays indexed by pressures.
    For oil: Rs[scf/STB], Bo[rb/STB], mu_o[cP], rho_o[lb/ft3]
    For gas: Rv[STB/Mscf], Bg[rb/Mscf], mu_g[cP], rho_g[lb/ft3], Z[-]
    Also returns saturation pressure (Pb or Pdew).
    """
    z = np.asarray(z, dtype=float); z = z / z.sum()

    # First reflect feed to surface to get the reference 1 STB / 1 Mscf basis
    n_oil_sc, n_gas_sc, V_oil_sc, V_gas_sc, _, _ = standard_conditions_split(
        z, comp_names, c7_props)

    # Sort pressures descending for depletion logic
    pressures = np.sort(pressures)[::-1]

    rows = []
    cumulative_gas_lbmol = 0.0      # for DLE (gas removed)
    feed = z.copy()                  # current (in-place) liquid composition (oil DLE)
    feed_lbmol = 1.0

    if fluid_kind == "oil":
        # Find Pb as the highest pressure where flash gives non-zero V
        # We'll compute it adaptively
        for P in pressures:
            r = flash(feed, comp_names, P, T_res, c7_props)
            if r["phase"] == "L" or r["V"] < 1e-6:
                # Single-phase oil: properties of liquid at P
                Z_L, _, *_ = pr_phase(comp_names, feed, P, T_res, c7_props, want="liquid")
                rho_oil = phase_density(comp_names, feed, Z_L, P, T_res, c7_props)
                MW = np.array([get_props(c, c7_props)["MW"] for c in comp_names])
                M_oil = float(np.dot(feed, MW))
                V_oil_res_bbl = feed_lbmol * M_oil / rho_oil / 5.615
                # Reflect *current feed* (after any prior gas removal in DLE)
                # to standard conditions for Bo reference
                n_o_sc, _, V_o_sc, _, _, _ = standard_conditions_split(
                    feed, comp_names, c7_props)
                Bo = V_oil_res_bbl / (feed_lbmol * V_o_sc) if V_o_sc > 0 else np.nan
                # Rs = remaining solution gas (lbmol of gas that would liberate at SC)
                _, n_g_sc, _, V_g_sc, _, _ = standard_conditions_split(
                    feed, comp_names, c7_props)
                # Rs in scf/STB referenced to *original* STB
                Rs = (n_g_sc * SCF_PER_LBMOL * feed_lbmol) / (V_oil_sc) if V_oil_sc > 0 else 0.0
                mu_o = lbc_viscosity(comp_names, feed, rho_oil, T_res, c7_props)
                rows.append({"P": P, "Rs": Rs, "Bo": Bo, "mu_o": mu_o,
                             "rho_o": rho_oil, "phase": "L"})
            else:
                # Two-phase: flash, remove gas (DLE), keep liquid as new feed
                V = r["V"]
                x = r["x"]; y = r["y"]
                Z_L = r["Z_L"]
                rho_oil = phase_density(comp_names, x, Z_L, P, T_res, c7_props)
                MW = np.array([get_props(c, c7_props)["MW"] for c in comp_names])
                M_oil = float(np.dot(x, MW))
                liquid_lbmol = feed_lbmol * (1 - V)
                V_oil_res_bbl = liquid_lbmol * M_oil / rho_oil / 5.615
                # Rs at this stage = remaining dissolved gas in oil at this P
                _, n_g_sc, _, V_g_sc, _, _ = standard_conditions_split(
                    x, comp_names, c7_props)
                # Bo on original-STB basis
                n_o_sc_x, _, V_o_sc_x, _, _, _ = standard_conditions_split(
                    x, comp_names, c7_props)
                Bo = V_oil_res_bbl / (liquid_lbmol * V_o_sc_x) if V_o_sc_x > 0 else np.nan
                Rs = (n_g_sc * SCF_PER_LBMOL * liquid_lbmol) / V_oil_sc if V_oil_sc > 0 else 0.0
                mu_o = lbc_viscosity(comp_names, x, rho_oil, T_res, c7_props)

                # Update feed for next (lower) pressure step (DLE)
                feed = x
                feed_lbmol = liquid_lbmol

                rows.append({"P": P, "Rs": Rs, "Bo": Bo, "mu_o": mu_o,
                             "rho_o": rho_oil, "phase": "LV"})
        # Sort ascending again
        rows = sorted(rows, key=lambda r: r["P"])
        return {"rows": rows, "fluid": "oil"}

    else:  # gas condensate (CVD)
        rows = []
        for P in pressures:
            r = flash(z, comp_names, P, T_res, c7_props)
            if r["phase"] == "V":
                Z_V = r["Z_V"]
                rho_g = phase_density(comp_names, z, Z_V, P, T_res, c7_props)
                # Bg = V_res / V_sc per lbmol
                # 1 lbmol gas at (P,T) takes Z*R*T/P ft3; at SC takes 379.49 scf
                Bg_rb_per_scf = (Z_V * 10.732 * T_res / P) / SCF_PER_LBMOL / 5.615
                Bg_rb_per_Mscf = Bg_rb_per_scf * 1000.0

                # Rv: re-flash z at SC to get STB/Mscf
                n_o_sc, n_g_sc, V_o_sc, V_g_sc, *_ = standard_conditions_split(
                    z, comp_names, c7_props)
                if V_g_sc > 0:
                    Rv_STB_per_Mscf = (V_o_sc / V_g_sc) * 1000.0
                else:
                    Rv_STB_per_Mscf = 0.0
                mu_g = lbc_viscosity(comp_names, z, rho_g, T_res, c7_props)
                rows.append({"P": P, "Z": Z_V, "Bg": Bg_rb_per_Mscf,
                             "Rv": Rv_STB_per_Mscf, "mu_g": mu_g,
                             "rho_g": rho_g, "phase": "V"})
            else:
                # Two-phase below Pdew
                Z_V = r["Z_V"]
                y = r["y"]
                rho_g = phase_density(comp_names, y, Z_V, P, T_res, c7_props)
                Bg_rb_per_scf = (Z_V * 10.732 * T_res / P) / SCF_PER_LBMOL / 5.615
                Bg_rb_per_Mscf = Bg_rb_per_scf * 1000.0

                # Rv from gas-phase composition re-flashed at SC
                n_o_sc, n_g_sc, V_o_sc, V_g_sc, *_ = standard_conditions_split(
                    y, comp_names, c7_props)
                Rv_STB_per_Mscf = (V_o_sc / V_g_sc) * 1000.0 if V_g_sc > 0 else 0.0
                mu_g = lbc_viscosity(comp_names, y, rho_g, T_res, c7_props)
                rows.append({"P": P, "Z": Z_V, "Bg": Bg_rb_per_Mscf,
                             "Rv": Rv_STB_per_Mscf, "mu_g": mu_g,
                             "rho_g": rho_g, "phase": "LV"})
        rows = sorted(rows, key=lambda r: r["P"])
        return {"rows": rows, "fluid": "gas"}
