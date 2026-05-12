"""
PVT Application — Equinor-themed Streamlit app
Black oil, dry gas, wet gas, water, compositional (Peng-Robinson EOS).
Features: lab experiments (Flash, CCE, CVD, DLE), phase envelope,
standalone flash calculator, ECLIPSE export (PVTO/PVDG/PVTG/PVTW).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from correlations import (OilCorrelations, GasCorrelations,
                          WaterCorrelations, WetGasCorrelations)
from eclipse_export import (build_pvto, build_pvdg, build_pvtg,
                            build_pvtw, build_pvtw_from_table,
                            build_density, build_full_deck,
                            build_pvto_from_compositional,
                            build_pvtg_from_compositional)
import units as U
import theme as TH

# ----------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------
st.set_page_config(page_title="PVT Studio", page_icon="●", layout="wide")
st.markdown(TH.CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(TH.header_banner(
    "PVT Studio",
    "Black oil • Dry gas • Wet gas • Water • Compositional (EOS) — "
    "with phase envelopes, lab experiments, and ECLIPSE export"),
    unsafe_allow_html=True)


# ----------------------------------------------------------------
# Sidebar — global controls
# ----------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    unit_system = st.radio("Unit system", ["Field", "SI"], horizontal=True,
                            help="Internal calculations use field units; SI converts at I/O. "
                                 "ECLIPSE export is always FIELD (the keyword spec).")
    L = U.UNIT_LABELS[unit_system]

    fluid = st.selectbox("Fluid type",
                         ["Oil (Black Oil)", "Dry Gas", "Wet Gas / Condensate",
                          "Water", "Compositional (EOS)"])

    st.markdown("### Reservoir Conditions")
    if unit_system == "Field":
        T_user = st.number_input(f"Temperature ({L['T']})",
                                  value=200.0, min_value=60.0, max_value=400.0)
        P_res_user = st.number_input(f"Pressure ({L['P']})",
                                      value=3500.0, min_value=14.7, max_value=15000.0)
    else:
        T_user = st.number_input(f"Temperature ({L['T']})",
                                  value=93.0, min_value=15.0, max_value=200.0)
        P_res_user = st.number_input(f"Pressure ({L['P']})",
                                      value=240.0, min_value=1.0, max_value=1000.0)
    T_res = U.to_field_T(T_user, unit_system)
    P_res = U.to_field_P(P_res_user, unit_system)

    st.markdown("### Pressure Range for Tables")
    if unit_system == "Field":
        P_min_user = st.number_input(f"P min ({L['P']})", value=14.7, min_value=14.7)
        P_max_user = st.number_input(f"P max ({L['P']})", value=6000.0, min_value=100.0)
    else:
        P_min_user = st.number_input(f"P min ({L['P']})", value=1.0, min_value=1.0)
        P_max_user = st.number_input(f"P max ({L['P']})", value=414.0, min_value=10.0)
    P_min = U.to_field_P(P_min_user, unit_system)
    P_max = U.to_field_P(P_max_user, unit_system)
    n_points = st.slider("Number of pressure points", 5, 40, 15)

    include_water = st.checkbox("Add PVTW to ECLIPSE export", value=True)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def line_chart_plotly(df, x_col, y_cols, title="", height=320, ymode="linear"):
    """Replacement for st.line_chart using Plotly with Equinor styling."""
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    fig = go.Figure()
    for i, c in enumerate(y_cols):
        fig.add_trace(TH.line_trace(df[x_col].values, df[c].values, c, color_idx=i))
    fig.update_layout(**TH.plotly_layout(
        title=title, xtitle=x_col, ytitle=(y_cols[0] if len(y_cols) == 1 else "Value"),
        height=height, ymode=ymode, showlegend=(len(y_cols) > 1)))
    st.plotly_chart(fig, use_container_width=True)


def styled_dataframe(df, height=380):
    """Display a dataframe with Equinor styling applied."""
    fmts = {c: ("{:.3e}" if "Cw" in str(c) else "{:.4f}")
            for c in df.columns if df[c].dtype != "object"}
    st.dataframe(df.style.format(fmts), use_container_width=True, height=height)


pressures = np.linspace(P_min, P_max, n_points)


# ================================================================
# OIL — Black Oil branch
# ================================================================
if fluid == "Oil (Black Oil)":
    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.markdown("### Oil Properties")
        api = st.number_input("Oil API gravity", value=35.0, min_value=10.0, max_value=60.0)
        gas_sg = st.number_input("Gas SG (air=1)", value=0.75, min_value=0.55, max_value=1.5)
        if unit_system == "Field":
            Rsi_user = st.number_input(f"Solution GOR at Pb ({L['Rs']})", value=600.0, min_value=0.0)
            Pb_user  = st.number_input(f"Bubble point ({L['P']}, 0=calc)", value=0.0, min_value=0.0)
        else:
            Rsi_user = st.number_input(f"Solution GOR at Pb ({L['Rs']})", value=107.0, min_value=0.0)
            Pb_user  = st.number_input(f"Bubble point ({L['P']}, 0=calc)", value=0.0, min_value=0.0)
        Rsi = U.to_field_Rs(Rsi_user, unit_system)
        Pb_input = U.to_field_P(Pb_user, unit_system) if Pb_user > 0 else 0.0

        st.markdown("### Correlations")
        rs_corr = st.selectbox("Rs / Pb", ["Standing", "Vasquez-Beggs", "Glaso", "Lasater"])
        bo_corr = st.selectbox("Bo", ["Standing", "Vasquez-Beggs", "Glaso"])
        mu_corr = st.selectbox("Dead-oil viscosity", ["Beggs-Robinson", "Beal", "Glaso"])

    oil = OilCorrelations(api=api, gas_sg=gas_sg, T=T_res,
                          rs_corr=rs_corr, bo_corr=bo_corr, mu_corr=mu_corr)
    Pb = Pb_input if Pb_input > 0 else oil.bubble_point(Rsi)

    rows = []
    for P in pressures:
        if P <= Pb:
            Rs = oil.solution_gor(P)
            Bo = oil.formation_volume_factor(P, Rs, saturated=True)
            mu = oil.viscosity(P, Rs, Pb, saturated=True)
        else:
            Rs = Rsi
            Bo = oil.formation_volume_factor(P, Rsi, saturated=False, Pb=Pb)
            mu = oil.viscosity(P, Rsi, Pb, saturated=False)
        rows.append({"P_field": P, "Rs_field": Rs, "Bo": Bo, "mu": mu})

    df = pd.DataFrame([{
        f"P ({L['P']})":   U.to_user_P(r["P_field"], unit_system),
        f"Rs ({L['Rs']})": U.to_user_Rs(r["Rs_field"], unit_system),
        f"Bo ({L['Bo']})": r["Bo"],
        f"μo ({L['mu']})": r["mu"],
    } for r in rows])

    with col_out:
        st.markdown(f"### Computed Properties — Pb = "
                     f"{U.to_user_P(Pb, unit_system):,.1f} {L['P']}")
        styled_dataframe(df)
        c1, c2, c3 = st.columns(3)
        pcol = f"P ({L['P']})"
        with c1: line_chart_plotly(df, pcol, f"Rs ({L['Rs']})", title="Solution GOR")
        with c2: line_chart_plotly(df, pcol, f"Bo ({L['Bo']})", title="Oil FVF")
        with c3: line_chart_plotly(df, pcol, f"μo ({L['mu']})", title="Oil Viscosity")

    st.markdown("---")
    st.markdown("### ECLIPSE Export (FIELD units)")
    df_field = pd.DataFrame([{
        "P (psia)": r["P_field"], "Rs (scf/STB)": r["Rs_field"],
        "Bo (rb/STB)": r["Bo"], "μo (cp)": r["mu"],
    } for r in rows])
    pvto_text = build_pvto(df_field, Pb, oil, Rsi, P_max)
    density_text = build_density(api=api, gas_sg=gas_sg)
    pvtw_text = ""
    if include_water:
        c_sal, c_corr = st.columns(2)
        with c_sal: salinity = st.number_input("Salinity (ppm)", value=30000.0, key="oil_sal")
        with c_corr: bw_corr = st.selectbox("Water correlation",
                                              ["McCain", "Meehan", "Numbere", "Spivey"], key="oil_wcorr")
        water = WaterCorrelations(salinity_ppm=salinity, T=T_res, corr=bw_corr)
        pvtw_text = build_pvtw_from_table(pressures, water, P_res)
    st.code(pvto_text + ("\n" + pvtw_text if pvtw_text else ""), language="text")
    deck = build_full_deck(pvto=pvto_text, pvtw=pvtw_text, density=density_text)
    st.download_button("Download PVT deck (.INC)", deck,
                        file_name="PVT_BLACKOIL.INC", mime="text/plain", type="primary")


# ================================================================
# DRY GAS
# ================================================================
elif fluid == "Dry Gas":
    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.markdown("### Gas Properties")
        gas_sg = st.number_input("Gas SG (air=1)", value=0.70, min_value=0.55, max_value=1.5)
        N2 = st.number_input("N2 mol fraction", value=0.0, min_value=0.0, max_value=0.3)
        CO2 = st.number_input("CO2 mol fraction", value=0.0, min_value=0.0, max_value=0.5)
        H2S = st.number_input("H2S mol fraction", value=0.0, min_value=0.0, max_value=0.3)
        st.markdown("### Correlations")
        z_corr = st.selectbox("Z-factor", ["Hall-Yarborough", "Dranchuk-Abou-Kassem"])
        mug_corr = st.selectbox("Gas viscosity", ["Lee-Gonzalez-Eakin", "Carr-Kobayashi-Burrows"])

    gas = GasCorrelations(gas_sg=gas_sg, T=T_res, N2=N2, CO2=CO2, H2S=H2S,
                           z_corr=z_corr, mu_corr=mug_corr)
    rows = []
    for P in pressures:
        if P < 14.7: continue
        Z = gas.z_factor(P)
        Bg = gas.formation_volume_factor(P, Z)
        rows.append({"P_field": P, "Z": Z, "Bg_rbscf": Bg,
                     "Bg_rbMscf": Bg * 1000.0, "mu": gas.viscosity(P, Z)})
    df_field = pd.DataFrame([{
        "P (psia)": r["P_field"], "Z": r["Z"],
        "Bg (rb/scf)": r["Bg_rbscf"], "μg (cp)": r["mu"]} for r in rows])
    df = pd.DataFrame([{
        f"P ({L['P']})":   U.to_user_P(r["P_field"], unit_system),
        "Z":               r["Z"],
        f"Bg ({L['Bg']})": U.to_user_Bg(r["Bg_rbMscf"], unit_system),
        f"μg ({L['mu']})": r["mu"]} for r in rows])

    with col_out:
        st.markdown("### Computed Gas Properties")
        styled_dataframe(df)
        c1, c2, c3 = st.columns(3)
        pcol = f"P ({L['P']})"
        with c1: line_chart_plotly(df, pcol, "Z", title="Z-factor")
        with c2: line_chart_plotly(df, pcol, f"Bg ({L['Bg']})", title="Gas FVF")
        with c3: line_chart_plotly(df, pcol, f"μg ({L['mu']})", title="Gas Viscosity")

    st.markdown("---")
    st.markdown("### ECLIPSE Export — PVDG (FIELD units)")
    pvdg_text = build_pvdg(df_field)
    density_text = build_density(api=35.0, gas_sg=gas_sg)
    pvtw_text = ""
    if include_water:
        c_sal, c_corr = st.columns(2)
        with c_sal: salinity = st.number_input("Salinity (ppm)", value=30000.0, key="gas_sal")
        with c_corr: bw_corr = st.selectbox("Water correlation",
                                             ["McCain", "Meehan", "Numbere", "Spivey"], key="gas_wcorr")
        water = WaterCorrelations(salinity_ppm=salinity, T=T_res, corr=bw_corr)
        pvtw_text = build_pvtw_from_table(pressures, water, P_res)
    st.code(pvdg_text + ("\n" + pvtw_text if pvtw_text else ""), language="text")
    deck = build_full_deck(pvdg=pvdg_text, pvtw=pvtw_text, density=density_text)
    st.download_button("Download PVT deck (.INC)", deck,
                        file_name="PVT_DRYGAS.INC", mime="text/plain", type="primary")


# ================================================================
# WET GAS / CONDENSATE
# ================================================================
elif fluid == "Wet Gas / Condensate":
    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.markdown("### Wet Gas Properties")
        gas_sg = st.number_input("Separator gas SG", value=0.72, min_value=0.55, max_value=1.5)
        api_cond = st.number_input("Condensate API", value=55.0, min_value=40.0, max_value=80.0)
        if unit_system == "Field":
            cgr_user = st.number_input("CGR (STB/MMscf)", value=80.0, min_value=1.0, max_value=300.0)
            Pdew_user = st.number_input(f"Dew point ({L['P']})", value=4500.0, min_value=500.0)
        else:
            cgr_user = st.number_input("CGR (Sm³/MSm³)", value=14.2, min_value=0.1, max_value=60.0)
            Pdew_user = st.number_input(f"Dew point ({L['P']})", value=310.0, min_value=30.0)
        cgr = cgr_user * 5.6146 if unit_system == "SI" else cgr_user
        Pdew = U.to_field_P(Pdew_user, unit_system)
        N2 = st.number_input("N2", value=0.0, min_value=0.0, max_value=0.3, key="wg_n2")
        CO2 = st.number_input("CO2", value=0.0, min_value=0.0, max_value=0.5, key="wg_co2")
        H2S = st.number_input("H2S", value=0.0, min_value=0.0, max_value=0.3, key="wg_h2s")
        st.markdown("### Correlations")
        z_corr = st.selectbox("Z-factor", ["Hall-Yarborough", "Dranchuk-Abou-Kassem"], key="wg_z")
        mug_corr = st.selectbox("Gas viscosity", ["Lee-Gonzalez-Eakin", "Carr-Kobayashi-Burrows"], key="wg_mu")
        rv_corr = st.selectbox("Rv vs P model", ["Linear-Pdew", "Constant"])

    wet = WetGasCorrelations(gas_sg=gas_sg, api_cond=api_cond,
                              cgr_stb_per_mmscf=cgr, T=T_res, N2=N2, CO2=CO2, H2S=H2S,
                              z_corr=z_corr, mu_corr=mug_corr,
                              rv_corr=rv_corr, Pdew=Pdew)
    rows = []
    for P in pressures:
        if P < 14.7: continue
        Z = wet.z_factor(P)
        rows.append({"P_field": P, "Z": Z,
                     "Bg_field": wet.formation_volume_factor(P, Z) * 1000.0,
                     "Rv_field": wet.rv(P) * 1000.0,
                     "mu": wet.viscosity(P, Z)})
    df = pd.DataFrame([{
        f"P ({L['P']})":   U.to_user_P(r["P_field"], unit_system),
        "Z":               r["Z"],
        f"Bg ({L['Bg']})": U.to_user_Bg(r["Bg_field"], unit_system),
        f"Rv ({L['Rv']})": U.to_user_Rs(r["Rv_field"], unit_system),
        f"μg ({L['mu']})": r["mu"]} for r in rows])

    with col_out:
        st.markdown(f"### Wet Gas Properties — recombined SG = {wet.gamma_g_res:.3f}")
        styled_dataframe(df)
        c1, c2 = st.columns(2)
        pcol = f"P ({L['P']})"
        with c1: line_chart_plotly(df, pcol, [f"Bg ({L['Bg']})", f"Rv ({L['Rv']})"],
                                    title="Bg and Rv vs Pressure")
        with c2: line_chart_plotly(df, pcol, ["Z", f"μg ({L['mu']})"],
                                    title="Z and Viscosity")

    st.markdown("---")
    st.markdown("### ECLIPSE Export — PVTG (FIELD units)")
    pvtg_text = build_pvtg(pressures, wet)
    density_text = build_density(api=api_cond, gas_sg=gas_sg)
    pvtw_text = ""
    if include_water:
        c_sal, c_corr = st.columns(2)
        with c_sal: salinity = st.number_input("Salinity (ppm)", value=30000.0, key="wg_sal")
        with c_corr: bw_corr = st.selectbox("Water correlation",
                                             ["McCain", "Meehan", "Numbere", "Spivey"], key="wg_wcorr")
        water = WaterCorrelations(salinity_ppm=salinity, T=T_res, corr=bw_corr)
        pvtw_text = build_pvtw_from_table(pressures, water, P_res)
    st.code(pvtg_text + ("\n" + pvtw_text if pvtw_text else ""), language="text")
    deck = build_full_deck(pvtg=pvtg_text, pvtw=pvtw_text, density=density_text)
    st.download_button("Download PVT deck (.INC)", deck,
                        file_name="PVT_WETGAS.INC", mime="text/plain", type="primary")


# ================================================================
# WATER
# ================================================================
elif fluid == "Water":
    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.markdown("### Water / Brine")
        salinity = st.number_input("Salinity (ppm NaCl-eq)", value=30000.0, min_value=0.0)
        bw_corr = st.selectbox("Correlation", ["McCain", "Meehan", "Numbere", "Spivey"])
        include_gas = st.checkbox("Include dissolved gas (Rsw)", value=False)
        Pb_water = 0.0
        if include_gas:
            Pb_water_user = st.number_input(
                f"Bubble-point of gas-saturated water ({L['P']})",
                value=U.to_user_P(P_res, unit_system),
                min_value=U.to_user_P(14.7, unit_system))
            Pb_water = U.to_field_P(Pb_water_user, unit_system)

    water = WaterCorrelations(salinity_ppm=salinity, T=T_res, corr=bw_corr,
                               include_gas=include_gas, Pb=Pb_water)
    rows = []
    for P in pressures:
        rows.append({
            "P_field": P, "Bw": water.bw(P), "Cw_field": water.compressibility(P),
            "mu": water.viscosity(P), "Rsw": water.rsw(P) if include_gas else 0.0,
            "rho": water.density(P)})
    df = pd.DataFrame([{
        f"P ({L['P']})":    U.to_user_P(r["P_field"], unit_system),
        f"Bw ({L['Bo']})":  r["Bw"],
        f"Cw ({L['Cw']})":  U.to_user_Cw(r["Cw_field"], unit_system),
        f"μw ({L['mu']})":  r["mu"],
        f"Rsw ({L['Rs']})": U.to_user_Rs(r["Rsw"], unit_system),
        f"ρw ({L['rho']})": U.to_user_rho(r["rho"], unit_system),
    } for r in rows])

    with col_out:
        st.markdown("### Computed Water Properties")
        styled_dataframe(df)
        c1, c2, c3 = st.columns(3)
        pcol = f"P ({L['P']})"
        with c1: line_chart_plotly(df, pcol, f"Bw ({L['Bo']})", title="Water FVF")
        with c2: line_chart_plotly(df, pcol, f"Cw ({L['Cw']})", title="Compressibility")
        with c3: line_chart_plotly(df, pcol, f"μw ({L['mu']})", title="Water Viscosity")

    st.markdown("---")
    st.markdown("### ECLIPSE Export — PVTW (FIELD units)")
    pvtw_text = build_pvtw_from_table(pressures, water, P_res)
    st.code(pvtw_text, language="text")
    deck = build_full_deck(pvtw=pvtw_text)
    st.download_button("Download PVTW (.INC)", deck,
                        file_name="PVTW.INC", mime="text/plain", type="primary")


# ================================================================
# COMPOSITIONAL (Peng-Robinson EOS) — uses tabs
# ================================================================
elif fluid == "Compositional (EOS)":
    from components import COMPONENTS, characterize_c7plus, get_props
    from eos_pr import saturation_pressure, flash
    from composition_pvt import (black_oil_table_from_composition,
                                  standard_conditions_split)
    from experiments import run_flash, run_cce, run_cvd, run_dle
    from phase_envelope import trace_envelope

    DEFAULT_COMP = {
        "N2":  0.002, "CO2": 0.009, "H2S": 0.000,
        "C1":  0.365, "C2":  0.097, "C3":  0.070,
        "iC4": 0.014, "nC4": 0.039, "iC5": 0.014, "nC5": 0.014,
        "C6":  0.043, "C7+": 0.333,
    }
    if "comp_state" not in st.session_state:
        st.session_state["comp_state"] = dict(DEFAULT_COMP)

    # ---- Composition input panel ----
    with st.expander("Composition input", expanded=True):
        cols_top = st.columns([1, 1, 1, 3])
        with cols_top[0]:
            if st.button("Normalize Σz → 1", use_container_width=True, type="primary"):
                s = sum(st.session_state["comp_state"].values())
                if s > 0:
                    for k in st.session_state["comp_state"]:
                        st.session_state["comp_state"][k] = \
                            st.session_state["comp_state"][k] / s
                    st.rerun()
        with cols_top[1]:
            if st.button("Reset defaults", use_container_width=True):
                st.session_state["comp_state"] = dict(DEFAULT_COMP)
                st.rerun()
        with cols_top[2]:
            current_sum = sum(st.session_state["comp_state"].values())
            color = "green" if abs(current_sum - 1.0) < 1e-3 else "orange"
            st.markdown(f"<div style='padding: 0.45rem 0.6rem; background:#F4F4F4; "
                        f"border-radius:4px; text-align:center;'>"
                        f"<b style='color:{color};'>Σz = {current_sum:.4f}</b></div>",
                        unsafe_allow_html=True)

        comp_inputs = {}
        cols2 = st.columns(4)
        for i, name in enumerate(DEFAULT_COMP.keys()):
            with cols2[i % 4]:
                comp_inputs[name] = st.number_input(
                    name, value=float(st.session_state["comp_state"][name]),
                    min_value=0.0, max_value=1.0, step=0.001,
                    format="%.4f", key=f"comp_input_{name}")
                st.session_state["comp_state"][name] = comp_inputs[name]

        c_c7 = st.columns(3)
        with c_c7[0]:
            MW_c7 = st.number_input("C7+ molecular weight", value=218.0,
                                    min_value=80.0, max_value=400.0)
        with c_c7[1]:
            SG_c7 = st.number_input("C7+ specific gravity", value=0.852,
                                    min_value=0.70, max_value=0.95)
        with c_c7[2]:
            comp_fluid_kind = st.selectbox(
                "Reservoir-fluid type",
                ["Oil (bubble point)", "Gas / Condensate (dew point)"])

    comp_names = [k for k, v in comp_inputs.items() if v > 0]
    z_raw = np.array([comp_inputs[k] for k in comp_names])
    if z_raw.sum() <= 0:
        st.error("All compositions are zero — enter at least one component.")
        st.stop()
    z_arr = z_raw / z_raw.sum()
    c7_props = characterize_c7plus(MW_c7=MW_c7, SG_c7=SG_c7) if "C7+" in comp_names else None
    T_R = T_res + 460.0
    fluid_kind = "oil" if "Oil" in comp_fluid_kind else "gas"

    # ---- Saturation point + C7+ summary metrics ----
    kind = "bubble" if fluid_kind == "oil" else "dew"
    Psat = None
    try:
        with st.spinner("Computing saturation pressure..."):
            Psat = saturation_pressure(z_arr, comp_names, T_R,
                                        c7_props=c7_props, kind=kind)
    except Exception as e:
        st.error(f"Saturation search failed: {e}")

    m1, m2, m3, m4, m5 = st.columns(5)
    sat_label = "Pb" if kind == "bubble" else "Pdew"
    if Psat is not None:
        m1.metric(f"{sat_label}", f"{U.to_user_P(Psat, unit_system):,.1f} {L['P']}")
    else:
        m1.metric(f"{sat_label}", "—")
    m2.metric(f"T_res", f"{T_user:.1f} {L['T']}")
    if c7_props:
        m3.metric(f"C7+ Tc",
                    f"{U.to_user_T(c7_props['Tc'] - 460.0, unit_system):.0f} {L['T']}")
        m4.metric(f"C7+ Pc",
                    f"{U.to_user_P(c7_props['Pc'], unit_system):.1f} {L['P']}")
        m5.metric(f"C7+ ω", f"{c7_props['omega']:.3f}")

    # ---- Tabbed analysis ----
    tab_exp, tab_env, tab_flash, tab_export = st.tabs(
        ["📊 Lab Experiments", "🔵 Phase Envelope",
         "⚡ Flash Calculator", "💾 ECLIPSE Export"])

    # ============================================================
    # TAB 1 — Lab experiments
    # ============================================================
    with tab_exp:
        experiment = st.selectbox(
            "Lab experiment to simulate",
            ["Black-oil table (DLE oil / depletion gas)",
             "Single-stage Flash",
             "CCE — Constant Composition Expansion",
             "CVD — Constant Volume Depletion",
             "DLE — Differential Liberation"])

        experiment_rows = []
        bot_rows = []
        try:
            with st.spinner("Running experiment..."):
                if experiment.startswith("Black-oil"):
                    result = black_oil_table_from_composition(
                        z_arr, comp_names, T_R, pressures,
                        c7_props=c7_props, fluid_kind=fluid_kind)
                    bot_rows = result["rows"]
                    experiment_rows = bot_rows
                elif experiment.startswith("Single-stage"):
                    experiment_rows = run_flash(z_arr, comp_names, T_R, pressures, c7_props)
                elif experiment.startswith("CCE"):
                    experiment_rows = run_cce(z_arr, comp_names, T_R, pressures, c7_props, P_sat=Psat)
                elif experiment.startswith("CVD"):
                    experiment_rows = run_cvd(z_arr, comp_names, T_R, pressures, c7_props, P_dew=Psat)
                elif experiment.startswith("DLE"):
                    experiment_rows = run_dle(z_arr, comp_names, T_R, pressures, c7_props, P_b=Psat)
        except Exception as e:
            st.error(f"Experiment failed: {e}")

        if experiment_rows:
            # Build display dataframe by experiment type
            if experiment.startswith("Black-oil"):
                if fluid_kind == "oil":
                    df = pd.DataFrame([{
                        f"P ({L['P']})":   U.to_user_P(r["P"], unit_system),
                        "Phase":           r["phase"],
                        f"Rs ({L['Rs']})": U.to_user_Rs(r["Rs"], unit_system),
                        f"Bo ({L['Bo']})": r["Bo"],
                        f"μo ({L['mu']})": r["mu_o"],
                        f"ρo ({L['rho']})": U.to_user_rho(r["rho_o"], unit_system),
                    } for r in bot_rows])
                else:
                    df = pd.DataFrame([{
                        f"P ({L['P']})":    U.to_user_P(r["P"], unit_system),
                        "Phase":            r["phase"], "Z": r["Z"],
                        f"Bg ({L['Bg']})":  U.to_user_Bg(r["Bg"], unit_system),
                        f"Rv ({L['Rv']})":  U.to_user_Rs(r["Rv"], unit_system),
                        f"μg ({L['mu']})":  r["mu_g"],
                        f"ρg ({L['rho']})": U.to_user_rho(r["rho_g"], unit_system),
                    } for r in bot_rows])
            elif experiment.startswith("Single-stage"):
                df = pd.DataFrame([{
                    f"P ({L['P']})":    U.to_user_P(r["P"], unit_system),
                    "Phase":            r["phase"], "V (mol frac)": r["V_mol_frac"],
                    "Z_L": r["Z_L"], "Z_V": r["Z_V"],
                    f"ρL ({L['rho']})": U.to_user_rho(r["rho_L"], unit_system),
                    f"ρV ({L['rho']})": U.to_user_rho(r["rho_V"], unit_system),
                    f"μL ({L['mu']})":  r["mu_L"], f"μV ({L['mu']})": r["mu_V"],
                } for r in experiment_rows])
            elif experiment.startswith("CCE"):
                df = pd.DataFrame([{
                    f"P ({L['P']})":   U.to_user_P(r["P"], unit_system),
                    "Phase":           r["phase"], "V / Vsat": r["V_rel"],
                    "Liquid dropout (% Vsat)": r["L_dropout_pct"],
                    "Y-function":      r["Y_function"],
                } for r in experiment_rows])
            elif experiment.startswith("CVD"):
                df = pd.DataFrame([{
                    f"P ({L['P']})":   U.to_user_P(r["P"], unit_system),
                    "Phase":           r["phase"],
                    "Cum. produced (mol %)":   r["cum_produced_pct"],
                    "Liquid dropout (% Vsat)": r["L_dropout_pct"],
                    "Z (2-phase)":     r["Z_2phase"],
                    "Z (gas)":         r["Z_gas"],
                    f"Rv produced ({L['Rv']})":
                        U.to_user_Rs(r["Rv_produced"], unit_system),
                } for r in experiment_rows])
            elif experiment.startswith("DLE"):
                df = pd.DataFrame([{
                    f"P ({L['P']})":   U.to_user_P(r["P"], unit_system),
                    "Phase":           r["phase"],
                    f"Rs ({L['Rs']})": U.to_user_Rs(r["Rs"], unit_system),
                    f"Bo ({L['Bo']})": r["Bo"],
                    f"μo ({L['mu']})": r["mu_o"],
                    f"ρo ({L['rho']})": U.to_user_rho(r["rho_o"], unit_system),
                } for r in experiment_rows])

            styled_dataframe(df)

            # Charts: pick a few key numeric columns
            numeric_cols = [c for c in df.columns
                            if c != "Phase" and df[c].dtype != "object"]
            if len(numeric_cols) >= 2:
                pcol = numeric_cols[0]
                others = numeric_cols[1:]
                # Render up to 3 charts
                chart_cols = st.columns(min(3, len(others)))
                for i, c in enumerate(others[:3]):
                    with chart_cols[i % len(chart_cols)]:
                        line_chart_plotly(df, pcol, c, title=c.split("(")[0].strip())

    # ============================================================
    # TAB 2 — Phase envelope
    # ============================================================
    with tab_env:
        st.markdown("Trace the bubble and dew loci over a temperature range. "
                    "The two branches meet at the (estimated) critical point.")

        c_env = st.columns(3)
        with c_env[0]:
            if unit_system == "Field":
                Tmin_env = st.number_input(f"T min ({L['T']})", value=-100.0,
                                            min_value=-200.0, max_value=500.0)
                Tmax_env = st.number_input(f"T max ({L['T']})", value=700.0,
                                            min_value=0.0, max_value=1500.0)
            else:
                Tmin_env = st.number_input(f"T min ({L['T']})", value=-70.0,
                                            min_value=-130.0, max_value=250.0)
                Tmax_env = st.number_input(f"T max ({L['T']})", value=370.0,
                                            min_value=-20.0, max_value=800.0)
        with c_env[1]:
            n_env = st.slider("Envelope sampling points", 10, 60, 25, key="env_n")
        with c_env[2]:
            show_reservoir = st.checkbox("Show reservoir (P, T) point", value=True)
            run_envelope = st.button("Trace envelope", type="primary",
                                      use_container_width=True)

        if run_envelope:
            T_min_R = U.to_field_T(Tmin_env, unit_system) + 460.0
            T_max_R = U.to_field_T(Tmax_env, unit_system) + 460.0
            try:
                with st.spinner("Tracing envelope (this can take a minute)..."):
                    env = trace_envelope(z_arr, comp_names, c7_props=c7_props,
                                          T_min=T_min_R, T_max=T_max_R,
                                          n_points=n_env, P_max=15000.0)
            except Exception as e:
                st.error(f"Envelope failed: {e}")
                env = None

            if env is not None and (len(env["T_bubble"]) > 0 or len(env["T_dew"]) > 0):
                # Plot
                fig = go.Figure()
                if len(env["T_bubble"]) > 0:
                    T_b_user = [U.to_user_T(t - 460.0, unit_system) for t in env["T_bubble"]]
                    P_b_user = [U.to_user_P(p, unit_system) for p in env["P_bubble"]]
                    fig.add_trace(go.Scatter(
                        x=T_b_user, y=P_b_user, name="Bubble locus",
                        mode="lines+markers",
                        line=dict(color=TH.COLOR_BUBBLE, width=3),
                        marker=dict(size=8, color=TH.COLOR_BUBBLE),
                        hovertemplate="<b>Bubble</b><br>T=%{x:.1f}<br>P=%{y:.1f}<extra></extra>",
                    ))
                if len(env["T_dew"]) > 0:
                    T_d_user = [U.to_user_T(t - 460.0, unit_system) for t in env["T_dew"]]
                    P_d_user = [U.to_user_P(p, unit_system) for p in env["P_dew"]]
                    fig.add_trace(go.Scatter(
                        x=T_d_user, y=P_d_user, name="Dew locus",
                        mode="lines+markers",
                        line=dict(color=TH.COLOR_DEW, width=3),
                        marker=dict(size=8, color=TH.COLOR_DEW),
                        hovertemplate="<b>Dew</b><br>T=%{x:.1f}<br>P=%{y:.1f}<extra></extra>",
                    ))
                if env["T_critical_est"] is not None:
                    Tc_user = U.to_user_T(env["T_critical_est"] - 460.0, unit_system)
                    Pc_user = U.to_user_P(env["P_critical_est"], unit_system)
                    fig.add_trace(go.Scatter(
                        x=[Tc_user], y=[Pc_user], name="Critical (est.)",
                        mode="markers+text",
                        marker=dict(size=15, color=TH.COLOR_CRITICAL, symbol="star",
                                    line=dict(color=TH.DARK_NAVY, width=1.5)),
                        text=[f"  CP ({Tc_user:.0f}, {Pc_user:.0f})"],
                        textposition="middle right",
                        textfont=dict(color=TH.DARK_NAVY, size=11),
                        hovertemplate="<b>Critical</b><br>T=%{x:.1f}<br>P=%{y:.1f}<extra></extra>",
                    ))
                if show_reservoir:
                    fig.add_trace(go.Scatter(
                        x=[T_user], y=[P_res_user], name="Reservoir",
                        mode="markers+text",
                        marker=dict(size=14, color="#FF8FA8", symbol="diamond",
                                    line=dict(color=TH.DARK_NAVY, width=1.5)),
                        text=[f"  Res ({T_user:.0f}, {P_res_user:.0f})"],
                        textposition="middle right",
                        textfont=dict(color=TH.DARK_NAVY, size=11),
                    ))

                fig.update_layout(**TH.plotly_layout(
                    title="Phase Envelope",
                    xtitle=f"Temperature ({L['T']})",
                    ytitle=f"Pressure ({L['P']})",
                    height=520))
                st.plotly_chart(fig, use_container_width=True)

                # Diagnostic metrics
                cm = st.columns(4)
                if len(env["P_bubble"]) > 0:
                    cm[0].metric("Cricondenbar (Pbub max)",
                                  f"{U.to_user_P(max(env['P_bubble']), unit_system):.0f} {L['P']}")
                if len(env["P_dew"]) > 0:
                    cm[1].metric("Cricondenbar (Pdew max)",
                                  f"{U.to_user_P(max(env['P_dew']), unit_system):.0f} {L['P']}")
                    cm[2].metric("Cricondentherm (max T on dew)",
                                  f"{U.to_user_T(max(env['T_dew']) - 460, unit_system):.0f} {L['T']}")
                if env["T_critical_est"] is not None:
                    cm[3].metric("T_critical (est.)",
                                  f"{U.to_user_T(env['T_critical_est']-460, unit_system):.0f} {L['T']}")
            else:
                st.warning("No saturation points found in this T range. "
                           "Try widening the temperature window.")

    # ============================================================
    # TAB 3 — Standalone flash calculator
    # ============================================================
    with tab_flash:
        st.markdown("Run a single-stage isothermal flash at any (P, T). "
                    "Useful for quick checks without generating a full table.")

        c_flash = st.columns(3)
        with c_flash[0]:
            if unit_system == "Field":
                T_flash_user = st.number_input(f"Flash T ({L['T']})",
                                                value=T_user, key="flash_T")
                P_flash_user = st.number_input(f"Flash P ({L['P']})",
                                                value=P_res_user, key="flash_P")
            else:
                T_flash_user = st.number_input(f"Flash T ({L['T']})",
                                                value=T_user, key="flash_T")
                P_flash_user = st.number_input(f"Flash P ({L['P']})",
                                                value=P_res_user, key="flash_P")
        with c_flash[1]:
            run_flash_btn = st.button("Run flash", type="primary", use_container_width=True)
            st.caption("Uses the composition above.")
        with c_flash[2]:
            st.caption(" ")

        if run_flash_btn:
            T_flash_R = U.to_field_T(T_flash_user, unit_system) + 460.0
            P_flash_field = U.to_field_P(P_flash_user, unit_system)
            from eos_pr import flash, pr_phase
            from eos_pr import phase_density
            from lbc import lbc_viscosity

            try:
                r = flash(z_arr, comp_names, P_flash_field, T_flash_R, c7_props)
            except Exception as e:
                st.error(f"Flash failed: {e}")
                r = None

            if r is not None:
                # Summary metrics
                phase_name = {"L": "Single-phase Liquid",
                              "V": "Single-phase Vapor",
                              "LV": "Two-phase (L + V)"}.get(r["phase"], r["phase"])
                cmf = st.columns(4)
                cmf[0].metric("Phase state", phase_name)
                cmf[1].metric("Vapor mol fraction", f"{r['V']:.4f}")
                if not np.isnan(r["Z_L"]):
                    cmf[2].metric("Z (Liquid)", f"{r['Z_L']:.4f}")
                if not np.isnan(r["Z_V"]):
                    cmf[3].metric("Z (Vapor)", f"{r['Z_V']:.4f}")

                # Phase compositions table
                comp_table_rows = []
                for i, c in enumerate(comp_names):
                    comp_table_rows.append({
                        "Component": c,
                        "z (feed)": z_arr[i],
                        "x (liquid)": r["x"][i] if r["phase"] != "V" else np.nan,
                        "y (vapor)": r["y"][i] if r["phase"] != "L" else np.nan,
                        "K = y/x":   r["K"][i] if r["phase"] == "LV" else np.nan,
                    })
                df_comp = pd.DataFrame(comp_table_rows)
                st.markdown("#### Phase compositions and K-values")
                st.dataframe(df_comp.style.format({
                    "z (feed)": "{:.4f}", "x (liquid)": "{:.4f}",
                    "y (vapor)": "{:.4f}", "K = y/x": "{:.4g}"}),
                    use_container_width=True, height=420)

                # Phase properties
                MW_arr = np.array([get_props(c, c7_props)["MW"] for c in comp_names])
                cols_prop = st.columns(2)

                if r["phase"] != "V" and r["x"].sum() > 0 and np.isfinite(r["Z_L"]):
                    rho_L = phase_density(comp_names, r["x"], r["Z_L"],
                                            P_flash_field, T_flash_R, c7_props)
                    mu_L = lbc_viscosity(comp_names, r["x"], rho_L,
                                            T_flash_R, c7_props)
                    M_L = float(np.dot(r["x"], MW_arr))
                    with cols_prop[0]:
                        st.markdown("##### Liquid Phase")
                        cl = st.columns(3)
                        cl[0].metric(f"ρ ({L['rho']})",
                                      f"{U.to_user_rho(rho_L, unit_system):.2f}")
                        cl[1].metric(f"μ ({L['mu']})", f"{mu_L:.4f}")
                        cl[2].metric("MW", f"{M_L:.1f}")

                if r["phase"] != "L" and r["y"].sum() > 0 and np.isfinite(r["Z_V"]):
                    rho_V = phase_density(comp_names, r["y"], r["Z_V"],
                                            P_flash_field, T_flash_R, c7_props)
                    mu_V = lbc_viscosity(comp_names, r["y"], rho_V,
                                            T_flash_R, c7_props)
                    M_V = float(np.dot(r["y"], MW_arr))
                    with cols_prop[1]:
                        st.markdown("##### Vapor Phase")
                        cv = st.columns(3)
                        cv[0].metric(f"ρ ({L['rho']})",
                                      f"{U.to_user_rho(rho_V, unit_system):.3f}")
                        cv[1].metric(f"μ ({L['mu']})", f"{mu_V:.5f}")
                        cv[2].metric("MW", f"{M_V:.1f}")

                # Surface flash for STB/Mscf info
                n_o, n_g, V_o, V_g, x_oil_sc, y_gas_sc = \
                    standard_conditions_split(z_arr, comp_names, c7_props)
                if V_g > 0 and V_o > 0:
                    GOR = (V_g / V_o)   # scf/STB
                    st.info(f"**Surface flash of feed:** "
                            f"GOR = {U.to_user_Rs(GOR, unit_system):.1f} {L['Rs']}, "
                            f"oil yield = {V_o:.4f} STB/lbmol, "
                            f"gas yield = {V_g:.1f} scf/lbmol")

    # ============================================================
    # TAB 4 — ECLIPSE export (only for black-oil table experiment)
    # ============================================================
    with tab_export:
        st.markdown("ECLIPSE PVTO/PVTG generated from the EOS black-oil table. "
                    "Run the 'Black-oil table' experiment in the Lab Experiments tab "
                    "to populate this section.")

        if bot_rows and Psat is not None:
            if fluid_kind == "oil":
                kw_text = build_pvto_from_compositional(bot_rows, Psat, P_max)
            else:
                kw_text = build_pvtg_from_compositional(bot_rows, Psat)

            n_o, n_g, V_o, V_g, x_oil_sc, y_gas_sc = standard_conditions_split(
                z_arr, comp_names, c7_props)
            MW_arr = np.array([get_props(c, c7_props)["MW"] for c in comp_names])
            rho_o_sc = ((n_o * float(np.dot(x_oil_sc, MW_arr))) / (V_o * 5.615)
                         if (V_o > 0 and n_o > 0) else 50.0)
            rho_g_sc = (0.0764 * float(np.dot(y_gas_sc, MW_arr)) / 28.97
                         if y_gas_sc.sum() > 0 else 0.05)
            density_text = ("DENSITY\n"
                            "-- Oil       Water     Gas      (lb/ft3)\n"
                            f"   {rho_o_sc:7.3f}   {62.428*1.02:7.3f}   "
                            f"{rho_g_sc:7.4f}  /\n\n")

            pvtw_text = ""
            if include_water:
                c_sal, c_corr = st.columns(2)
                with c_sal: salinity = st.number_input("Salinity (ppm)", value=30000.0,
                                                       key="comp_sal")
                with c_corr: bw_corr = st.selectbox("Water correlation",
                                                     ["McCain", "Meehan", "Numbere", "Spivey"],
                                                     key="comp_wcorr")
                water = WaterCorrelations(salinity_ppm=salinity, T=T_res, corr=bw_corr)
                pvtw_text = build_pvtw_from_table(pressures, water, P_res)

            st.code(kw_text + ("\n" + pvtw_text if pvtw_text else ""), language="text")
            if fluid_kind == "oil":
                deck = build_full_deck(pvto=kw_text, pvtw=pvtw_text, density=density_text)
                fname = "PVT_COMPOSITIONAL_OIL.INC"
            else:
                deck = build_full_deck(pvtg=kw_text, pvtw=pvtw_text, density=density_text)
                fname = "PVT_COMPOSITIONAL_GAS.INC"
            st.download_button("Download PVT deck (.INC)", deck,
                                file_name=fname, mime="text/plain", type="primary")
        else:
            st.info("Run the **Black-oil table** experiment first to generate ECLIPSE keywords.")


# ----------------------------------------------------------------
# Footer note
# ----------------------------------------------------------------
st.markdown("---")
with st.expander("About the correlations and methods"):
    st.markdown("""
**Oil (correlation):** Standing, Vasquez-Beggs, Glaso, Lasater (Rs/Pb/Bo) ·
Beggs-Robinson, Beal, Glaso (μ).

**Gas (correlation):** Sutton pseudo-criticals + Wichert-Aziz sour-gas correction ·
Z by Hall-Yarborough or Dranchuk-Abou-Kassem · Viscosity by Lee-Gonzalez-Eakin
or Carr-Kobayashi-Burrows.

**Wet gas / condensate (correlation):** McCain recombination · Linear or
constant Rv vs P.

**Compositional (EOS):** Peng-Robinson (1976) with Michelsen-style two-sided
stability test, Wilson K, Rachford-Rice solver, successive-substitution flash ·
C7+ characterized by Kesler-Lee · Lohrenz-Bray-Clark viscosity ·
Experiments: Flash, CCE, CVD, DLE.

**Water:** McCain · Meehan · Numbere · Spivey-Valko-McCain (2004) ·
Optional dissolved-gas Rsw (McCain) + Dodson-Standing Cw correction.
""")
