"""
Multi-stage separator flash for surface processing of reservoir fluids.

Typical surface train:
    Stage 1 (HP)  : 400-1000 psia, 100-150 F
    Stage 2 (LP)  : 50-200 psia, 80-100 F
    Stage 3 (ST)  : 14.7 psia, 60 F  (stock tank)

At each stage, the LIQUID from the previous stage is flashed. The vapor stream
from each stage is collected separately; cumulative produced gas determines GOR
while the final stage liquid is the stock-tank oil.
"""

import numpy as np
from eos_pr import flash, phase_density
from components import get_props

T_SC = 60.0 + 460.0
P_SC = 14.7
SCF_PER_LBMOL = 379.49
BBL_PER_FT3 = 1.0 / 5.615


def default_separator_train(low_gor=False):
    """Standard separator train as list of (P_psia, T_F) tuples, HP -> ST."""
    if low_gor:
        return [(100.0, 80.0), (14.7, 60.0)]
    return [(800.0, 100.0), (100.0, 80.0), (14.7, 60.0)]


def run_separator_train(z, comp_names, train, c7_props=None):
    """
    Run multi-stage separator flash on feed z.

    Returns dict with per-stage results, total gas, ST oil, combined gas SG,
    and effective GOR.
    """
    z = np.asarray(z, dtype=float); z = z / z.sum()
    MW = np.array([get_props(c, c7_props)["MW"] for c in comp_names])

    current_feed = z.copy()
    current_lbmol = 1.0

    stage_results = []
    gas_streams = []
    total_gas_scf = 0.0
    total_gas_lbmol = 0.0
    total_gas_comp = np.zeros_like(z)

    for i, (P_s, T_s_F) in enumerate(train):
        T_s_R = T_s_F + 460.0
        r = flash(current_feed, comp_names, P_s, T_s_R, c7_props)
        V = r.get("V", 0.0)

        if r["phase"] == "L" or V <= 1e-9:
            x = current_feed; y = np.zeros_like(z)
            n_oil = current_lbmol; n_gas = 0.0
            Z_L = r.get("Z_L", np.nan); Z_V = np.nan
        elif r["phase"] == "V" or V >= 1 - 1e-9:
            x = np.zeros_like(z); y = current_feed
            n_oil = 0.0; n_gas = current_lbmol
            Z_L = np.nan; Z_V = r.get("Z_V", np.nan)
        else:
            x = r["x"]; y = r["y"]
            n_oil = current_lbmol * (1 - V); n_gas = current_lbmol * V
            Z_L = r["Z_L"]; Z_V = r["Z_V"]

        V_gas_scf_this = n_gas * SCF_PER_LBMOL
        total_gas_scf += V_gas_scf_this
        total_gas_lbmol += n_gas
        if n_gas > 0:
            total_gas_comp += n_gas * y

        # Per-stage gas SG and densities
        if n_gas > 1e-12:
            gas_MW_stage = float(np.dot(y, MW))
            gas_SG_stage = gas_MW_stage / 28.97
            # gas density at stage P,T using Z_V
            if not np.isnan(Z_V):
                rho_gas_stage = phase_density(comp_names, y, Z_V,
                                                P_s, T_s_R, c7_props)
            else:
                rho_gas_stage = np.nan
        else:
            gas_MW_stage = np.nan; gas_SG_stage = np.nan
            rho_gas_stage = np.nan
        # Stage liquid (oil) density
        if n_oil > 1e-12 and not np.isnan(Z_L):
            rho_oil_stage = phase_density(comp_names, x, Z_L,
                                            P_s, T_s_R, c7_props)
        else:
            rho_oil_stage = np.nan

        gas_streams.append({"stage": i + 1, "n_gas": n_gas, "y": y,
                             "V_scf": V_gas_scf_this, "P": P_s, "T_F": T_s_F})
        stage_results.append({
            "stage": i + 1, "P": P_s, "T_F": T_s_F,
            "n_in": current_lbmol, "n_oil_out": n_oil, "n_gas_out": n_gas,
            "x": x, "y": y, "Z_L": Z_L, "Z_V": Z_V, "V_frac": V,
            "gas_scf_this_stage": V_gas_scf_this,
            "gas_SG_stage": gas_SG_stage,
            "rho_gas_stage": rho_gas_stage,
            "rho_oil_stage": rho_oil_stage,
        })

        current_feed = x
        current_lbmol = n_oil
        if current_lbmol < 1e-12:
            break

    # Stock-tank oil
    st_oil_comp = current_feed
    st_oil_lbmol = current_lbmol
    if st_oil_lbmol > 0:
        last = stage_results[-1]
        Z_st = last["Z_L"]
        if not np.isnan(Z_st):
            rho_st = phase_density(comp_names, st_oil_comp, Z_st,
                                     last["P"], last["T_F"] + 460.0, c7_props)
        else:
            rho_st = 50.0
        M_st = float(np.dot(st_oil_comp, MW))
        V_oil_bbl = st_oil_lbmol * M_st / rho_st * BBL_PER_FT3
    else:
        rho_st = np.nan; M_st = np.nan; V_oil_bbl = 0.0

    # Combined surface-gas
    if total_gas_lbmol > 0:
        total_gas_comp_norm = total_gas_comp / total_gas_lbmol
        gas_MW = float(np.dot(total_gas_comp_norm, MW))
    else:
        total_gas_comp_norm = np.zeros_like(z); gas_MW = np.nan

    GOR = (total_gas_scf / V_oil_bbl) if V_oil_bbl > 0 else 0.0

    # Per-stage GOR: gas released at that stage per STB of final stock-tank oil
    for s in stage_results:
        s["stage_GOR_scfSTB"] = (s["gas_scf_this_stage"] / V_oil_bbl
                                  if V_oil_bbl > 0 else 0.0)

    # API gravity of stock-tank oil
    api = (141.5 / (rho_st / 62.428) - 131.5) if (rho_st and rho_st > 0
                                                    and not np.isnan(rho_st)) else np.nan

    return {
        "stage_results": stage_results,
        "gas_streams": gas_streams,
        "total_gas_scf": total_gas_scf,
        "total_gas_lbmol": total_gas_lbmol,
        "st_oil_bbl": V_oil_bbl,
        "st_oil_lbmol": st_oil_lbmol,
        "st_oil_comp": st_oil_comp,
        "st_oil_density": rho_st,
        "st_oil_MW": M_st,
        "st_oil_API": api,
        "gas_comp": total_gas_comp_norm,
        "gas_MW": gas_MW,
        "gas_SG": gas_MW / 28.97 if not np.isnan(gas_MW) else np.nan,
        "GOR_scfSTB": GOR,
        "oil_yield_STB_per_lbmol": V_oil_bbl,
    }
