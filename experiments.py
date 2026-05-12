"""
PVT laboratory experiments simulated via PR EOS.

Four experiments are supported:

  1. FLASH  – Single-stage flash at each (P, T_res). Composition z is fixed.
              Reports: phase amounts, Z, density, viscosity.

  2. CCE   – Constant Composition Expansion (also "CCT", "P-V relation").
              Closed cell, total composition = z at all P. As P drops below
              the saturation point, both phases coexist in the cell.
              Reports: relative volume V/Vsat, liquid drop-out, Y-function.

  3. CVD   – Constant Volume Depletion (gas condensate standard).
              At each step below Pdew, vapor is removed to bring the cell
              volume back to V_sat. Cumulative produced gas is tracked.
              Reports: cumulative produced moles, liquid drop-out, Z_gas,
              and an effective Rv vs P for the produced gas.

  4. DLE   – Differential Liberation (oil standard, also "DV").
              At each step below Pb, ALL liberated gas is removed and the
              remaining oil is the new feed for the next step.
              Reports: Rs, Bo, mu_o, rho_o, liberated-gas SG per stage.

Inputs to every experiment:
    z          – feed mole fractions, list
    comp_names – list of component identifiers (e.g. ['C1', 'C2', 'C7+'])
    T_R        – reservoir temperature, °R
    pressures  – list/array of psia, will be sorted high->low internally
    c7_props   – optional dict for C7+
"""

import numpy as np
from eos_pr import flash, pr_phase, phase_density
from lbc import lbc_viscosity
from components import get_props

T_SC = 60.0 + 460.0
P_SC = 14.7
SCF_PER_LBMOL = 379.49
BBL_PER_FT3 = 1.0 / 5.615


# ----------------------------------------------------------------
# Helper: phase volume in ft3 given moles and (Z, P, T)
# ----------------------------------------------------------------
def phase_volume_ft3(n_lbmol, Z, P, T_R):
    if n_lbmol <= 0 or not np.isfinite(Z):
        return 0.0
    return n_lbmol * Z * 10.732 * T_R / P


def stream_density_lbft3(comp_names, x_or_y, Z, P, T_R, c7_props=None):
    if not np.isfinite(Z) or x_or_y.sum() < 1e-12:
        return np.nan
    return phase_density(comp_names, x_or_y, Z, P, T_R, c7_props)


def stream_viscosity_cP(comp_names, x_or_y, rho, T_R, c7_props=None):
    if not np.isfinite(rho) or x_or_y.sum() < 1e-12:
        return np.nan
    return lbc_viscosity(comp_names, x_or_y, rho, T_R, c7_props)


def split_to_surface(comp_frac, comp_names, c7_props=None):
    """Flash at standard conditions, return (V_oil_bbl, V_gas_scf) per 1 lbmol."""
    cf = np.asarray(comp_frac, dtype=float)
    s = cf.sum()
    if s < 1e-12:
        return 0.0, 0.0
    cf = cf / s
    r = flash(cf, comp_names, P_SC, T_SC, c7_props)
    MW = np.array([get_props(c, c7_props)["MW"] for c in comp_names])
    if r["phase"] == "L":
        rho = phase_density(comp_names, r["x"], r["Z_L"], P_SC, T_SC, c7_props)
        Mavg = float(np.dot(r["x"], MW))
        return Mavg / rho * BBL_PER_FT3, 0.0
    if r["phase"] == "V":
        return 0.0, SCF_PER_LBMOL
    n_oil = 1.0 - r["V"]; n_gas = r["V"]
    rho_oil = phase_density(comp_names, r["x"], r["Z_L"], P_SC, T_SC, c7_props)
    M_oil = float(np.dot(r["x"], MW))
    V_oil = n_oil * M_oil / rho_oil * BBL_PER_FT3
    V_gas = n_gas * SCF_PER_LBMOL
    return V_oil, V_gas


# ================================================================
# 1. FLASH – simple single-stage at each P
# ================================================================
def run_flash(z, comp_names, T_R, pressures, c7_props=None):
    rows = []
    for P in sorted(pressures):
        r = flash(z, comp_names, P, T_R, c7_props)
        V = r["V"]
        phase = r["phase"]
        Z_L = r.get("Z_L", np.nan); Z_V = r.get("Z_V", np.nan)

        rho_L = stream_density_lbft3(comp_names, r["x"], Z_L, P, T_R, c7_props) \
                if phase != "V" else np.nan
        rho_V = stream_density_lbft3(comp_names, r["y"], Z_V, P, T_R, c7_props) \
                if phase != "L" else np.nan
        mu_L  = stream_viscosity_cP(comp_names, r["x"], rho_L, T_R, c7_props) \
                if phase != "V" and np.isfinite(rho_L) else np.nan
        mu_V  = stream_viscosity_cP(comp_names, r["y"], rho_V, T_R, c7_props) \
                if phase != "L" and np.isfinite(rho_V) else np.nan

        rows.append({"P": P, "phase": phase, "V_mol_frac": V,
                     "Z_L": Z_L, "Z_V": Z_V,
                     "rho_L": rho_L, "rho_V": rho_V,
                     "mu_L": mu_L, "mu_V": mu_V})
    return rows


# ================================================================
# 2. CCE – Constant Composition Expansion
# ================================================================
def run_cce(z, comp_names, T_R, pressures, c7_props=None, P_sat=None):
    """
    Returns rows with V_total/V_sat, liquid drop-out (% of V_sat),
    and Y-function = (P_sat/P - 1)/(V/V_sat - 1) below saturation.
    """
    z = np.asarray(z, dtype=float); z = z / z.sum()
    pressures = sorted(set(list(pressures) + ([P_sat] if P_sat else [])))[::-1]

    # First find V_sat: volume at saturation pressure
    if P_sat is None:
        # If not given, just use the highest pressure as reference
        P_ref = pressures[0]
    else:
        P_ref = P_sat

    # Volume per lbmol of feed at reference (single-phase)
    r_ref = flash(z, comp_names, P_ref, T_R, c7_props)
    if r_ref["phase"] == "V":
        Z_ref = r_ref["Z_V"]
    elif r_ref["phase"] == "L":
        Z_ref = r_ref["Z_L"]
    else:
        # Two-phase at "saturation" — use total volume
        v_L = phase_volume_ft3(1 - r_ref["V"], r_ref["Z_L"], P_ref, T_R)
        v_V = phase_volume_ft3(r_ref["V"], r_ref["Z_V"], P_ref, T_R)
        V_sat_ft3 = v_L + v_V
        Z_ref = V_sat_ft3 * P_ref / (10.732 * T_R)

    V_sat_ft3 = phase_volume_ft3(1.0, Z_ref, P_ref, T_R) \
                if not np.isnan(Z_ref) else 1.0

    rows = []
    for P in pressures:
        r = flash(z, comp_names, P, T_R, c7_props)
        if r["phase"] == "L":
            V_tot = phase_volume_ft3(1.0, r["Z_L"], P, T_R)
            V_liq = V_tot; V_vap = 0.0
        elif r["phase"] == "V":
            V_tot = phase_volume_ft3(1.0, r["Z_V"], P, T_R)
            V_liq = 0.0; V_vap = V_tot
        else:
            V_liq = phase_volume_ft3(1 - r["V"], r["Z_L"], P, T_R)
            V_vap = phase_volume_ft3(r["V"],     r["Z_V"], P, T_R)
            V_tot = V_liq + V_vap

        V_rel = V_tot / V_sat_ft3 if V_sat_ft3 > 0 else np.nan
        L_dropout = 100.0 * V_liq / V_sat_ft3 if V_sat_ft3 > 0 else 0.0

        # Y-function: only meaningful below Psat in 2-phase
        if P_sat and P < P_sat - 1 and V_rel > 1.001:
            Y = ((P_sat / P) - 1.0) / (V_rel - 1.0)
        else:
            Y = np.nan

        rows.append({"P": P, "V_rel": V_rel, "L_dropout_pct": L_dropout,
                     "Y_function": Y, "phase": r["phase"]})

    return sorted(rows, key=lambda r: r["P"])


# ================================================================
# 3. CVD – Constant Volume Depletion (gas condensate)
# ================================================================
def run_cvd(z, comp_names, T_R, pressures, c7_props=None, P_dew=None):
    """
    Standard CVD experiment.
    At P >= P_dew: single-phase gas, no removal.
    At P < P_dew: flash, find vapor and liquid volumes. Compute total volume.
                  Determine excess vapor moles to remove so V_cell = V_sat.
                  Update cell composition (= remaining vapor + all liquid).
    Reports: cumulative produced (mol % of feed), liquid dropout % of V_sat,
             Z of produced gas, two-phase Z of cell (V_cell * P / nRT).
    """
    z = np.asarray(z, dtype=float); z = z / z.sum()
    pressures = sorted(pressures)[::-1]

    if P_dew is None:
        P_dew = pressures[0]

    # Reference V_sat at Pdew (or initial cell volume) for 1 lbmol of feed
    r_dew = flash(z, comp_names, P_dew, T_R, c7_props)
    if r_dew["phase"] == "V":
        Z_dew = r_dew["Z_V"]
    elif r_dew["phase"] == "LV":
        v_L = phase_volume_ft3(1 - r_dew["V"], r_dew["Z_L"], P_dew, T_R)
        v_V = phase_volume_ft3(r_dew["V"], r_dew["Z_V"], P_dew, T_R)
        V_sat = v_L + v_V
        Z_dew = V_sat * P_dew / (10.732 * T_R * 1.0)
    else:
        Z_dew = r_dew["Z_L"]
    V_sat_ft3 = phase_volume_ft3(1.0, Z_dew, P_dew, T_R)

    rows = []
    cell_n = 1.0                       # total moles in cell (starts at 1)
    cell_z = z.copy()                  # cell composition
    cumulative_produced = 0.0          # moles of gas removed (cumulative)
    cumulative_produced_comp = np.zeros_like(z)   # composition flow weighted

    for P in pressures:
        r = flash(cell_z, comp_names, P, T_R, c7_props)

        if r["phase"] in ("V", "L"):
            # Single phase. No removal needed unless V exceeds V_sat (shouldn't above Pdew).
            Z = r["Z_V"] if r["phase"] == "V" else r["Z_L"]
            V_now = phase_volume_ft3(cell_n, Z, P, T_R)
            L_dropout = 0.0
            n_removed = 0.0
            y_removed = np.zeros_like(z)
            Z_gas = Z if r["phase"] == "V" else np.nan
            Z_2phase = Z
        else:
            V = r["V"]
            n_L = cell_n * (1 - V); n_V = cell_n * V
            v_L = phase_volume_ft3(n_L, r["Z_L"], P, T_R)
            v_V = phase_volume_ft3(n_V, r["Z_V"], P, T_R)
            V_tot = v_L + v_V
            L_dropout = 100.0 * v_L / V_sat_ft3

            # Two-phase Z
            Z_2phase = V_tot * P / (10.732 * T_R * cell_n)

            # Remove enough vapor to restore V_cell = V_sat
            excess_V_volume = V_tot - V_sat_ft3
            if excess_V_volume > 0 and r["Z_V"] > 0:
                # Moles of vapor to remove
                molar_v_V = phase_volume_ft3(1.0, r["Z_V"], P, T_R)
                n_removed = excess_V_volume / molar_v_V
                n_removed = min(n_removed, n_V)
            else:
                n_removed = 0.0
            y_removed = r["y"]
            Z_gas = r["Z_V"]

            # Update cell: remaining = liquid + remaining vapor
            n_V_new = n_V - n_removed
            new_cell_n = n_L + n_V_new
            if new_cell_n > 1e-12:
                cell_z = (n_L * r["x"] + n_V_new * r["y"]) / new_cell_n
                cell_z = cell_z / cell_z.sum()
            cell_n = new_cell_n
            cumulative_produced += n_removed
            cumulative_produced_comp += n_removed * y_removed
            V_now = V_sat_ft3   # by construction

        # Compute Rv of the produced gas (surface flash)
        if y_removed.sum() > 0:
            V_oil_bbl_per_mol, V_gas_scf_per_mol = split_to_surface(
                y_removed, comp_names, c7_props)
            Rv_STBperMscf = (V_oil_bbl_per_mol / V_gas_scf_per_mol * 1000.0) \
                            if V_gas_scf_per_mol > 0 else 0.0
        else:
            Rv_STBperMscf = 0.0

        rows.append({
            "P": P,
            "cum_produced_pct": 100.0 * cumulative_produced,
            "L_dropout_pct": L_dropout,
            "Z_2phase": Z_2phase,
            "Z_gas": Z_gas if np.isfinite(Z_gas) else np.nan,
            "Rv_produced": Rv_STBperMscf,
            "phase": r["phase"],
        })

    return sorted(rows, key=lambda r: r["P"])


# ================================================================
# 4. DLE – Differential Liberation (oil)
# ================================================================
def run_dle(z, comp_names, T_R, pressures, c7_props=None, P_b=None):
    """
    Differential liberation:
      P >= P_b: single-phase oil, properties from EOS
      P <  P_b: flash, REMOVE ALL GAS, new feed = liquid composition
    Reports: Rs, Bo, mu_o, rho_o (all referenced to original 1 STB).
    """
    z = np.asarray(z, dtype=float); z = z / z.sum()
    pressures = sorted(pressures)[::-1]

    # Reference STB from original z at SC
    V_oil_sc_orig, _ = split_to_surface(z, comp_names, c7_props)

    feed = z.copy()
    feed_lbmol = 1.0

    rows = []
    for P in pressures:
        r = flash(feed, comp_names, P, T_R, c7_props)
        MW = np.array([get_props(c, c7_props)["MW"] for c in comp_names])

        if r["phase"] == "L" or r["V"] < 1e-6:
            # Single-phase oil
            Z_L = r["Z_L"] if r["phase"] == "L" else \
                  pr_phase(comp_names, feed, P, T_R, c7_props, want="liquid")[0]
            rho = phase_density(comp_names, feed, Z_L, P, T_R, c7_props)
            M = float(np.dot(feed, MW))
            V_res = feed_lbmol * M / rho * BBL_PER_FT3
            # Dissolved-gas content (re-flash at SC)
            V_o_sc, V_g_sc = split_to_surface(feed, comp_names, c7_props)
            Rs = (V_g_sc * feed_lbmol) / V_oil_sc_orig if V_oil_sc_orig > 0 else 0.0
            Bo = V_res / (feed_lbmol * V_o_sc) if V_o_sc > 0 else np.nan
            mu = lbc_viscosity(comp_names, feed, rho, T_R, c7_props)
            rows.append({"P": P, "phase": "L", "Rs": Rs, "Bo": Bo,
                         "mu_o": mu, "rho_o": rho})
        else:
            V = r["V"]; x = r["x"]; y = r["y"]
            n_L = feed_lbmol * (1 - V)
            rho = phase_density(comp_names, x, r["Z_L"], P, T_R, c7_props)
            M = float(np.dot(x, MW))
            V_res = n_L * M / rho * BBL_PER_FT3
            V_o_sc_x, V_g_sc_x = split_to_surface(x, comp_names, c7_props)
            Rs = (V_g_sc_x * n_L) / V_oil_sc_orig if V_oil_sc_orig > 0 else 0.0
            Bo = V_res / (n_L * V_o_sc_x) if V_o_sc_x > 0 else np.nan
            mu = lbc_viscosity(comp_names, x, rho, T_R, c7_props)
            rows.append({"P": P, "phase": "LV", "Rs": Rs, "Bo": Bo,
                         "mu_o": mu, "rho_o": rho})

            # Update: remove gas, keep liquid
            feed = x
            feed_lbmol = n_L

    return sorted(rows, key=lambda r: r["P"])
