"""
ECLIPSE VFPPROD export from a lift-curve calculation.

ECLIPSE's VFPPROD keyword tabulates the bottomhole (or inlet) pressure
required at a node as a function of:

    FLO    — flow-rate variable (LIQ, OIL, GAS, WAT)
    THP    — tubing-head pressure (the outlet pressure)
    WFR    — water-phase ratio (WCT, WGR, WOR)
    GFR    — gas-phase ratio (GOR, GLR, OGR)
    ALQ    — artificial-lift quantity (we leave as 0)

The simulator interpolates this table during a run to get inlet P as a
function of operating conditions. Our `nodal.lift_curve` produces the
P_bhp vs rate curve for one combination of THP/WCT/GLR — to build a
proper VFP we sweep over THP, WCT, GLR values and generate the 4-D
table.

This implementation generates:
- A SINGLE-DIMENSION-SWEEP VFP (1 THP, 1 WCT, 1 GLR) — the most basic
  table, suitable for sensitivity-free studies.
- A MULTI-THP VFP with optional WCT and GLR sweeps — the standard
  production-engineering VFP.

Both write a syntactically valid VFPPROD block. Units default to FIELD
(matching the rest of the ECLIPSE export in this app); a METRIC option
converts at write time.

References:
    Schlumberger ECLIPSE Reference Manual, VFPPROD keyword
    OPM Flow keyword reference (open-source equivalent)
"""

APP_VERSION = "1.4.2"   # must match pvt_app.py (deployment check)

import numpy as np
from nodal import lift_curve, march_pressure


# ----------------------------------------------------------------------
# Unit conversion helpers (field -> metric for the VFP body)
# ----------------------------------------------------------------------
_PSIA_TO_BARA = 1.0 / 14.5038
_BBL_TO_M3 = 0.158987
_MSCF_TO_SM3 = 28.3168   # 1 Mscf = 28.3168 Sm³
_FT_TO_M = 0.3048


def _conv_rate(rate_bbl_d, unit_kind, system):
    """Convert a field rate to the chosen ECLIPSE unit system."""
    if system == "METRIC":
        return rate_bbl_d * _BBL_TO_M3       # STB/d -> Sm³/d
    return rate_bbl_d


def _conv_gas_rate(mscf_d, system):
    if system == "METRIC":
        return mscf_d * _MSCF_TO_SM3 * 1000.0 / 1000.0   # Mscf/d -> Sm³/d
        # = 28.3168 * Mscf/d Sm³/d... but actually Mscf is thousand scf.
        # 1 Mscf = 1000 scf = 1000/35.3147 Sm³ = 28.3168 Sm³. So Sm³/d = Mscf/d * 28.3168.
    return mscf_d


def _conv_P(psia, system):
    if system == "METRIC":
        return psia * _PSIA_TO_BARA
    return psia


def _conv_ratio(scf_per_stb, system):
    """GLR / GOR: scf/STB -> Sm³/Sm³ (divide by 5.6146)."""
    if system == "METRIC":
        return scf_per_stb / 5.6146
    return scf_per_stb


def _flo_to_liquid(flo_value, flo_kind, wct, glr_scf_stb):
    """Liquid rate (STB/d) that corresponds to one FLO-axis value."""
    if flo_kind == "OIL":
        return flo_value / (1.0 - wct) if wct < 0.999 else None
    if flo_kind == "WAT":
        return flo_value / wct if wct > 0.001 else None
    if flo_kind == "GAS":            # flo_value in Mscf/d
        return flo_value * 1000.0 / glr_scf_stb if glr_scf_stb > 0 else None
    return flo_value                 # LIQ


# ----------------------------------------------------------------------
# Single VFP table from a sweep over THP / WCT / GLR
# ----------------------------------------------------------------------
def build_vfp_table(
    flow_rates_bbl_d,
    thp_values_psia,
    wct_values,
    glr_values_scf_stb,
    depth_ft,
    T_outlet_F,
    T_inlet_F,
    segments,
    api,
    gas_sg,
    salinity_wt_pct,
    D_in_inch,
    rel_roughness,
    method,
    vfp_table_no=1,
    flo_kind="LIQ",
    units="FIELD",
    progress_callback=None,
):
    """Build a VFPPROD table from a sweep of nodal calculations.

    Returns a dict with:
        rates_disp, thp_disp, wfr_disp, gfr_disp : axis arrays in the
            requested units (FIELD or METRIC),
        bhp_table_disp : a 4-D numpy array indexed
            [thp][wfr][gfr][rate] giving the bottomhole/inlet pressure
            in the requested units,
        n_evals        : number of nodal calculations done (for the UI
            progress bar).

    Note: ALQ (artificial-lift quantity) is held at zero — we don't
    model gas-lift here. The table body still has the ALQ dimension as
    required by ECLIPSE; it just has a single value.
    """
    nq = len(flow_rates_bbl_d)
    nt = len(thp_values_psia)
    nw = max(1, len(wct_values))
    ng = max(1, len(glr_values_scf_stb))

    bhp = np.full((nt, nw, ng, nq), np.nan)

    n_eval = 0
    total = nt * nw * ng
    for it, P_thp in enumerate(thp_values_psia):
        for iw, wct in enumerate(wct_values):
            for ig, glr in enumerate(glr_values_scf_stb):
                # The FLO axis is in FLO units (STB/d for LIQ/OIL/WAT,
                # Mscf/d for GAS). Convert each FLO value to the liquid
                # rate the lift-curve march needs. Previously the axis was
                # always liquid rate and FLO only changed the header label,
                # so FLO=GAS/OIL/WAT tables were wrong.
                q_liq = [_flo_to_liquid(q, flo_kind, wct, glr)
                         for q in flow_rates_bbl_d]
                ok = [i for i, q in enumerate(q_liq)
                      if q is not None and q > 0]
                res_ok = lift_curve(
                    rates_bbl_d=[q_liq[i] for i in ok],
                    GLR_scf_stb=glr,
                    P_outlet_psia=P_thp,
                    T_outlet_F=T_outlet_F,
                    T_inlet_F=T_inlet_F,
                    segments=segments,
                    api=api,
                    gas_sg=gas_sg,
                    water_cut_frac=wct,
                    salinity_wt_pct=salinity_wt_pct,
                    D_in_inch=D_in_inch,
                    rel_roughness=rel_roughness,
                    method=method,
                ) if ok else []
                res = [{"P_inlet_psia": np.nan} for _ in flow_rates_bbl_d]
                for i, r in zip(ok, res_ok):
                    res[i] = r
                for iq, r in enumerate(res):
                    bhp[it, iw, ig, iq] = r["P_inlet_psia"]
                n_eval += 1
                if progress_callback is not None:
                    progress_callback(n_eval / total)

    # Display-unit copies
    if units == "METRIC":
        rates_disp = [(_conv_gas_rate(q, units) if flo_kind == "GAS"
                        else _conv_rate(q, flo_kind, units))
                       for q in flow_rates_bbl_d]
        thp_disp = [_conv_P(p, units) for p in thp_values_psia]
        gfr_disp = [_conv_ratio(g, units) for g in glr_values_scf_stb]
        bhp_disp = bhp * _PSIA_TO_BARA
        depth_disp = depth_ft * _FT_TO_M
    else:
        rates_disp = list(flow_rates_bbl_d)
        thp_disp = list(thp_values_psia)
        gfr_disp = list(glr_values_scf_stb)
        bhp_disp = bhp.copy()
        depth_disp = depth_ft

    wfr_disp = list(wct_values)
    return {
        "rates_disp": rates_disp,
        "thp_disp": thp_disp,
        "wfr_disp": wfr_disp,
        "gfr_disp": gfr_disp,
        "bhp_table_disp": bhp_disp,
        "depth_disp": depth_disp,
        "n_evals": n_eval,
        "units": units,
        "method": method,
        "flo_kind": flo_kind,
        "vfp_table_no": vfp_table_no,
    }


# ----------------------------------------------------------------------
# Format a VFP table as an ECLIPSE-compatible string
# ----------------------------------------------------------------------
def format_vfpprod(table, comment=""):
    """Return the VFPPROD keyword block as a string.

    Layout (per the ECLIPSE manual):

        VFPPROD
        -- table_no  datum_depth  rate_type  wfr_type  gfr_type
        --   ALQ_type  units      tab_type
        1  8000.0  'LIQ'  'WCT'  'GOR'  ' '  'FIELD'  'BHP' /
        -- FLO values
        100 500 1000 2000 / 
        -- THP values
        200 400 800 /
        -- WFR values
        0.0 0.5 /
        -- GFR values
        500 1000 /
        -- ALQ values
        0 /
        -- BHP table body (one record per THP/WFR/GFR/ALQ combination):
        -- thp_idx wfr_idx gfr_idx alq_idx  bhp_at_each_rate
        1 1 1 1  3000 3200 3500 4100 /
        ...
        /
    """
    units = table["units"]
    unit_tag = "'METRIC'" if units == "METRIC" else "'FIELD'"
    rates = table["rates_disp"]
    thps = table["thp_disp"]
    wfrs = table["wfr_disp"]
    gfrs = table["gfr_disp"]
    bhp = table["bhp_table_disp"]
    depth = table["depth_disp"]
    tno = table["vfp_table_no"]
    flo = table["flo_kind"]

    out = []
    if comment:
        for line in comment.splitlines():
            out.append("-- " + line)
    out.append("VFPPROD")
    out.append(
        "-- table_no datum   FLO   WFR    GFR    ALQ  UNITS    TAB")
    out.append(
        f"   {tno}    {depth:9.2f}  '{flo}'  'WCT'  'GLR'  ' '   "
        f"{unit_tag}  'BHP' /")

    def _fmt_row(values, width=10, ncols=8):
        s = []
        for i, v in enumerate(values):
            s.append(f"{v:{width}.4g}")
            if (i + 1) % ncols == 0 and i < len(values) - 1:
                s.append("\n  ")
        return "  " + " ".join(s)

    out.append("-- FLO (rate) values")
    out.append(_fmt_row(rates) + " /")
    out.append("-- THP values")
    out.append(_fmt_row(thps) + " /")
    out.append("-- WFR (water-fraction) values")
    out.append(_fmt_row(wfrs) + " /")
    out.append("-- GFR (gas ratio) values")
    out.append(_fmt_row(gfrs) + " /")
    out.append("-- ALQ values")
    out.append("   0.0 /")

    out.append("-- BHP table body: thp wfr gfr alq <BHP at each rate>")
    nt, nw, ng, nq = bhp.shape
    for it in range(nt):
        for iw in range(nw):
            for ig in range(ng):
                bvals = [bhp[it, iw, ig, iq] for iq in range(nq)]
                # Replace NaN with a large sentinel so the deck loads;
                # the simulator will see the missing-data flag.
                bvals = [v if v == v else 99999.0 for v in bvals]
                row = f"   {it+1:2d} {iw+1:2d} {ig+1:2d}  1 "
                row += " ".join(f"{v:10.4g}" for v in bvals)
                row += " /"
                out.append(row)
    out.append("/")
    return "\n".join(out) + "\n"
