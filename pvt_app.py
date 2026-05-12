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

with st.expander("ℹ️ Quick help — how to use this app"):
    st.markdown("""
**Workflow:**
1. **Choose a fluid type** in the sidebar (Oil, Dry Gas, Wet Gas, Water, or
   Compositional).
2. **Set units** (Field / SI) — affects display only; ECLIPSE export is selected
   separately.
3. **Enter reservoir T, P, and pressure range** for the PVT tables.
4. **Configure fluid properties** in the main panel (correlation choice or composition).
5. **View results** in the auto-generated tables and charts.
6. For **Compositional** fluids, use the tabs to access:
    - *Lab Experiments* — DLE, CCE, CVD, single-stage flash
    - *Phase Envelope* — trace bubble and dew loci over T
    - *Flash Calculator* — single-stage flash at any (P, T)
    - *Separator Train* — multi-stage surface processing
    - *EOS Tuning* — fit parameters to lab data
    - *Multi-Region* — PVTNUM > 1 stacked PVT tables
    - *ECLIPSE Export* — PVTO/PVTG + RSVD/RVVD vs depth
7. **Monte Carlo uncertainty** is available in the Oil branch under its own expander.
8. **Equations and references** are in the Docs tab (Compositional fluid).

**Tips:**
- The composition panel has a **Normalize** button to rescale Σz → 1.0.
- ECLIPSE keywords must use *consistent* units throughout your simulator deck.
  Set the `FIELD` or `METRIC` keyword in RUNSPEC to match what you download here.
- The Tuning tab can perturb C7+ Pc/Tc/ω plus the C1-C7+ and N2-C7+ binary
  interaction coefficients.
- Phase envelope tracing can take 10–30 s for 20–30 temperature points.
""")



# ----------------------------------------------------------------
# Sidebar — global controls
# ----------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    unit_system = st.radio("Unit system", ["Field", "SI"], horizontal=True,
                            help="Internal calculations use field units; SI converts at I/O. "
                                 "ECLIPSE export is always FIELD (the keyword spec).")
    L = U.UNIT_LABELS[unit_system]

    fluid = st.selectbox("Fluid type / Analysis",
                         ["Oil (Black Oil)", "Dry Gas", "Wet Gas / Condensate",
                          "Water", "Compositional (EOS)",
                          "❄️ Hydrate Likelihood"])

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
    """Display a dataframe with safe per-column number formatting.

    Uses Streamlit's `column_config` rather than `df.style.format`, which is
    more robust against mixed-type columns, NaN values, and string columns
    (pandas may report dtype as 'str' rather than 'object' in newer versions).
    """
    if df is None or len(df) == 0:
        st.info("No data to display.")
        return

    column_config = {}
    for c in df.columns:
        # Robust numeric check
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        c_str = str(c)
        if "Cw" in c_str:
            fmt = "%.3e"
        elif c_str.strip() == "Z" or c_str.startswith("Z "):
            fmt = "%.4f"
        elif "μ" in c_str or "mu" in c_str.lower():
            fmt = "%.4f"
        elif "Rs" in c_str or "Rv" in c_str:
            fmt = "%.2f"
        elif c_str.startswith("P "):
            fmt = "%.1f"
        elif "Bo" in c_str or "Bg" in c_str or "Bw" in c_str:
            fmt = "%.4f"
        elif "ρ" in c_str or "rho" in c_str.lower():
            fmt = "%.2f"
        elif "%" in c_str or "pct" in c_str.lower() or "dropout" in c_str.lower():
            fmt = "%.2f"
        else:
            fmt = "%.4g"
        column_config[c] = st.column_config.NumberColumn(c, format=fmt)

    try:
        st.dataframe(df, use_container_width=True, height=height,
                     column_config=column_config, hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, height=height)


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

    # -------- Optional companion PVDG for the dissolved gas --------
    with st.expander("📑 Add companion PVDG (for the dissolved gas phase)"):
        st.markdown(
            "ECLIPSE black-oil with live oil also needs gas-phase properties "
            "(PVDG) for free gas — gas that comes out of solution and flows "
            "separately. This block builds PVDG using a dry-gas correlation "
            "for the *solution gas* (using the gas SG you provided)."
        )
        if st.button("Build companion PVDG", key="pvdg_companion_oil"):
            gas_companion = GasCorrelations(gas_sg=gas_sg, T=T_res)
            pvdg_rows = []
            for P in pressures:
                if P < 14.7: continue
                Zg = gas_companion.z_factor(P)
                pvdg_rows.append({
                    "P (psia)": P, "Z": Zg,
                    "Bg (rb/scf)": gas_companion.formation_volume_factor(P, Zg),
                    "μg (cp)": gas_companion.viscosity(P, Zg)})
            pvdg_df = pd.DataFrame(pvdg_rows)
            pvdg_companion_text = build_pvdg(pvdg_df)
            st.code(pvdg_companion_text, language="text")
            deck_with_pvdg = build_full_deck(
                pvto=pvto_text, pvdg=pvdg_companion_text,
                pvtw=pvtw_text, density=density_text)
            st.download_button(
                "Download deck with PVTO + PVDG (.INC)",
                deck_with_pvdg,
                file_name="PVT_BLACKOIL_with_PVDG.INC",
                mime="text/plain", type="primary",
                key="dl_oil_pvdg")

    # -------- Monte Carlo uncertainty --------
    st.markdown("---")
    with st.expander("🎲 Monte Carlo uncertainty analysis"):
        st.markdown(
            "Sample input parameters from normal distributions, run the chosen "
            "correlation at each draw, and view the distribution of resulting "
            "$P_b$, $B_o$, $R_s$, and $\\mu_o$ at the reservoir pressure."
        )
        mc_cols = st.columns(4)
        with mc_cols[0]:
            mc_std_api = st.number_input("σ_API", value=2.0, min_value=0.0, max_value=20.0)
        with mc_cols[1]:
            mc_std_sg = st.number_input("σ_gas_SG", value=0.05,
                                        min_value=0.0, max_value=0.5, format="%.3f")
        with mc_cols[2]:
            mc_std_Rsi = st.number_input("σ_Rsi (in field units)",
                                          value=50.0, min_value=0.0, max_value=500.0)
        with mc_cols[3]:
            mc_std_T = st.number_input("σ_T (in field units, °F)",
                                        value=10.0, min_value=0.0, max_value=100.0)

        mc_cols2 = st.columns(3)
        with mc_cols2[0]:
            n_samples = st.slider("Number of samples", 100, 5000, 1000, step=100)
        with mc_cols2[1]:
            mc_seed = st.number_input("RNG seed", value=42, min_value=0)
        with mc_cols2[2]:
            run_mc = st.button("Run Monte Carlo", type="primary",
                                use_container_width=True)

        if run_mc:
            from monte_carlo import run_monte_carlo_oil, summary_stats, tornado_sensitivity
            base = {"api": api, "gas_sg": gas_sg, "Rsi": Rsi, "T": T_res,
                    "rs_corr": rs_corr, "bo_corr": bo_corr, "mu_corr": mu_corr}
            unc = {"api": mc_std_api, "gas_sg": mc_std_sg,
                    "Rsi": mc_std_Rsi, "T": mc_std_T}
            with st.spinner(f"Running {n_samples} Monte Carlo draws..."):
                mc_result = run_monte_carlo_oil(
                    base, unc, n_samples=int(n_samples),
                    correlation_class=OilCorrelations,
                    target_P=P_res, seed=int(mc_seed))

            stat_Pb = summary_stats(mc_result["Pb"])
            stat_Bo = summary_stats(mc_result["Bo"])
            stat_Rs = summary_stats(mc_result["Rs"])
            stat_mu = summary_stats(mc_result["mu"])

            # Summary metrics
            sm = st.columns(4)
            sm[0].metric(f"Pb mean ({L['P']})",
                          f"{U.to_user_P(stat_Pb['mean'], unit_system):.0f}",
                          delta=f"±{U.to_user_P(stat_Pb['std'], unit_system) - U.to_user_P(0, unit_system):.0f}")
            sm[1].metric(f"Bo mean", f"{stat_Bo['mean']:.4f}",
                          delta=f"±{stat_Bo['std']:.4f}")
            sm[2].metric(f"Rs mean ({L['Rs']})",
                          f"{U.to_user_Rs(stat_Rs['mean'], unit_system):.1f}",
                          delta=f"±{U.to_user_Rs(stat_Rs['std'], unit_system) - U.to_user_Rs(0, unit_system):.1f}")
            sm[3].metric(f"μo mean ({L['mu']})", f"{stat_mu['mean']:.4f}",
                          delta=f"±{stat_mu['std']:.4f}")

            # Percentile summary table
            pct_df = pd.DataFrame([
                {"Property": "Pb",
                 "P10": U.to_user_P(stat_Pb["P10"], unit_system),
                 "P50": U.to_user_P(stat_Pb["P50"], unit_system),
                 "P90": U.to_user_P(stat_Pb["P90"], unit_system),
                 "Mean": U.to_user_P(stat_Pb["mean"], unit_system),
                 "Std": U.to_user_P(stat_Pb["std"], unit_system) -
                        U.to_user_P(0, unit_system)},
                {"Property": "Bo", "P10": stat_Bo["P10"], "P50": stat_Bo["P50"],
                 "P90": stat_Bo["P90"], "Mean": stat_Bo["mean"], "Std": stat_Bo["std"]},
                {"Property": "Rs",
                 "P10": U.to_user_Rs(stat_Rs["P10"], unit_system),
                 "P50": U.to_user_Rs(stat_Rs["P50"], unit_system),
                 "P90": U.to_user_Rs(stat_Rs["P90"], unit_system),
                 "Mean": U.to_user_Rs(stat_Rs["mean"], unit_system),
                 "Std": U.to_user_Rs(stat_Rs["std"], unit_system) -
                        U.to_user_Rs(0, unit_system)},
                {"Property": "μo", "P10": stat_mu["P10"], "P50": stat_mu["P50"],
                 "P90": stat_mu["P90"], "Mean": stat_mu["mean"], "Std": stat_mu["std"]},
            ])
            styled_dataframe(pct_df, height=200)

            # Histograms
            import plotly.graph_objects as go
            hc1, hc2 = st.columns(2)
            with hc1:
                fig = go.Figure(go.Histogram(
                    x=[U.to_user_P(p, unit_system) for p in mc_result["Pb"]
                        if not np.isnan(p)],
                    nbinsx=30, marker_color=TH.TORCH_RED, opacity=0.85,
                ))
                fig.add_vline(x=U.to_user_P(stat_Pb["mean"], unit_system),
                              line_dash="dash", line_color=TH.DARK_NAVY,
                              annotation_text="Mean")
                fig.update_layout(**TH.plotly_layout(
                    title=f"Pb distribution",
                    xtitle=f"Pb ({L['P']})", ytitle="Count",
                    height=320, showlegend=False))
                st.plotly_chart(fig, use_container_width=True)
            with hc2:
                fig = go.Figure(go.Histogram(
                    x=mc_result["Bo"][~np.isnan(mc_result["Bo"])],
                    nbinsx=30, marker_color=TH.DARK_NAVY, opacity=0.85,
                ))
                fig.add_vline(x=stat_Bo["mean"], line_dash="dash",
                              line_color=TH.TORCH_RED, annotation_text="Mean")
                fig.update_layout(**TH.plotly_layout(
                    title=f"Bo distribution at reservoir P",
                    xtitle=f"Bo", ytitle="Count",
                    height=320, showlegend=False))
                st.plotly_chart(fig, use_container_width=True)

            hc3, hc4 = st.columns(2)
            with hc3:
                fig = go.Figure(go.Histogram(
                    x=[U.to_user_Rs(r, unit_system) for r in mc_result["Rs"]
                        if not np.isnan(r)],
                    nbinsx=30, marker_color=TH.PISTACHIO, opacity=0.85,
                ))
                fig.update_layout(**TH.plotly_layout(
                    title="Rs distribution",
                    xtitle=f"Rs ({L['Rs']})", ytitle="Count",
                    height=320, showlegend=False))
                st.plotly_chart(fig, use_container_width=True)
            with hc4:
                fig = go.Figure(go.Histogram(
                    x=mc_result["mu"][~np.isnan(mc_result["mu"])],
                    nbinsx=30, marker_color="#3A6E96", opacity=0.85,
                ))
                fig.update_layout(**TH.plotly_layout(
                    title="μo distribution",
                    xtitle=f"μo ({L['mu']})", ytitle="Count",
                    height=320, showlegend=False))
                st.plotly_chart(fig, use_container_width=True)

            # Tornado for Pb
            tornado = tornado_sensitivity(base, unc, OilCorrelations,
                                            target_P=P_res, output="Pb")
            if tornado["rows"]:
                st.markdown("##### Tornado sensitivity (Pb)")
                tor_df = pd.DataFrame([{
                    "Parameter": param,
                    "Low":   U.to_user_P(lo, unit_system),
                    "High":  U.to_user_P(hi, unit_system),
                    "Range": U.to_user_P(rng, unit_system) -
                             U.to_user_P(0, unit_system),
                } for param, lo, hi, rng in tornado["rows"]])
                styled_dataframe(tor_df, height=200)
                fig = go.Figure()
                for j, (param, lo, hi, rng) in enumerate(tornado["rows"]):
                    base_v = tornado["base_value"]
                    fig.add_trace(go.Bar(
                        y=[param], x=[U.to_user_P(hi, unit_system) -
                                       U.to_user_P(base_v, unit_system)],
                        base=U.to_user_P(base_v, unit_system),
                        orientation="h", name=f"{param} high",
                        marker_color=TH.TORCH_RED,
                        showlegend=(j == 0), legendgroup="hi"))
                    fig.add_trace(go.Bar(
                        y=[param], x=[U.to_user_P(lo, unit_system) -
                                       U.to_user_P(base_v, unit_system)],
                        base=U.to_user_P(base_v, unit_system),
                        orientation="h", name=f"{param} low",
                        marker_color=TH.DARK_NAVY,
                        showlegend=(j == 0), legendgroup="lo"))
                fig.add_vline(x=U.to_user_P(tornado["base_value"], unit_system),
                              line_dash="dash", line_color="black")
                fig.update_layout(**TH.plotly_layout(
                    title="Tornado: Pb sensitivity to ±1σ parameter perturbation",
                    xtitle=f"Pb ({L['P']})", ytitle="",
                    height=320, showlegend=True))
                st.plotly_chart(fig, use_container_width=True)

    # -------- CCE / CVD experiments for correlation oil --------
    with st.expander("🧪 CCE / CVD experiments (correlation-based)"):
        st.markdown(
            "Run a black-oil CCE simulation using the chosen correlation. "
            "Below Pb, the cell contains both liberated gas and remaining oil; "
            "V/Vsat grows rapidly and Y-function is reported."
        )
        from correlation_experiments import cce_blackoil
        from correlations import GasCorrelations as GasCorrForCce

        gas_corr_for_cce = GasCorrForCce(gas_sg=gas_sg, T=T_res)
        if st.button("Run CCE", key="run_cce_oil"):
            cce_rows = cce_blackoil(oil, gas_corr_for_cce, Rsi, Pb, pressures)
            cce_df = pd.DataFrame([{
                f"P ({L['P']})": U.to_user_P(r["P"], unit_system),
                "Phase": r["phase"],
                "V / Vsat": r["V_rel"],
                "Liquid dropout (% Vsat)": r["L_dropout_pct"],
                "Y-function": r["Y_function"],
            } for r in cce_rows])
            styled_dataframe(cce_df, height=320)
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(TH.line_trace(cce_df[f"P ({L['P']})"], cce_df["V / Vsat"],
                                         "V / Vsat", color_idx=0))
            fig.update_layout(**TH.plotly_layout(
                title="CCE — V/Vsat vs P",
                xtitle=f"P ({L['P']})", ytitle="V / Vsat",
                height=340))
            st.plotly_chart(fig, use_container_width=True)


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

    # -------- Optional companion PVTO for the dropped-out condensate --------
    with st.expander("📑 Add companion PVTO (for the condensate phase)"):
        st.markdown(
            "ECLIPSE wet-gas with vaporized oil also needs oil-phase properties "
            "(PVTO) for condensate that drops out and flows as a separate "
            "liquid phase. This block builds a PVTO using a black-oil "
            "correlation on the condensate (using the condensate API and "
            "the gas SG you provided)."
        )
        if st.button("Build companion PVTO", key="pvto_companion_wg"):
            oil_companion = OilCorrelations(
                api=api_cond, gas_sg=gas_sg, T=T_res,
                rs_corr="Standing", bo_corr="Standing",
                mu_corr="Beggs-Robinson")
            # Use a typical condensate Rsi based on CGR (heavy ends -> low Rs)
            Rsi_cond = max(50.0, cgr * 0.5)  # rough estimate
            Pb_cond = oil_companion.bubble_point(Rsi_cond)
            cond_rows = []
            for P in pressures:
                if P <= Pb_cond:
                    Rs_c = oil_companion.solution_gor(P)
                    Bo_c = oil_companion.formation_volume_factor(P, Rs_c, saturated=True)
                    mu_c = oil_companion.viscosity(P, Rs_c, Pb_cond, saturated=True)
                else:
                    Rs_c = Rsi_cond
                    Bo_c = oil_companion.formation_volume_factor(
                        P, Rsi_cond, saturated=False, Pb=Pb_cond)
                    mu_c = oil_companion.viscosity(P, Rsi_cond, Pb_cond, saturated=False)
                cond_rows.append({"P (psia)": P, "Rs (scf/STB)": Rs_c,
                                  "Bo (rb/STB)": Bo_c, "μo (cp)": mu_c})
            cond_df = pd.DataFrame(cond_rows)
            pvto_cond_text = build_pvto(cond_df, Pb_cond, oil_companion, Rsi_cond, P_max)
            st.code(pvto_cond_text, language="text")
            deck_with_pvto = build_full_deck(
                pvto=pvto_cond_text, pvtg=pvtg_text,
                pvtw=pvtw_text, density=density_text)
            st.download_button(
                "Download deck with PVTG + PVTO (.INC)",
                deck_with_pvto,
                file_name="PVT_WETGAS_with_PVTO.INC",
                mime="text/plain", type="primary",
                key="dl_wg_pvto")


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
    (tab_exp, tab_env, tab_flash,
     tab_separator, tab_tuning, tab_multireg,
     tab_mc, tab_docs, tab_export) = st.tabs(
        ["📊 Lab Experiments", "🔵 Phase Envelope",
         "⚡ Flash Calculator", "🏭 Separator Train",
         "🎯 EOS Tuning", "🗂️ Multi-Region",
         "🎲 Monte Carlo", "📖 Docs", "💾 ECLIPSE Export"])

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
    # TAB — Separator Train
    # ============================================================
    with tab_separator:
        st.markdown(
            "Simulate **multi-stage surface processing**: the reservoir fluid is "
            "flashed through a series of separators at decreasing pressure, with "
            "the liquid from each stage feeding the next. The cumulative produced "
            "gas determines the field GOR; only the final stage liquid is the "
            "stock-tank oil."
        )
        from separator import run_separator_train, default_separator_train

        if "sep_train" not in st.session_state:
            st.session_state["sep_train"] = list(default_separator_train())

        c_train = st.columns([2, 1])
        with c_train[0]:
            st.markdown("##### Separator stages (HP → ST)")
            new_train = []
            for i, (Ps, Ts) in enumerate(st.session_state["sep_train"]):
                cs = st.columns([1, 2, 2, 1])
                with cs[0]: st.markdown(f"**S{i+1}**")
                with cs[1]:
                    Ps_user = st.number_input(
                        f"P_{i+1} ({L['P']})",
                        value=U.to_user_P(Ps, unit_system),
                        min_value=U.to_user_P(14.7, unit_system),
                        key=f"sepP_{i}")
                with cs[2]:
                    Ts_user = st.number_input(
                        f"T_{i+1} ({L['T']})",
                        value=U.to_user_T(Ts, unit_system),
                        key=f"sepT_{i}")
                new_train.append((U.to_field_P(Ps_user, unit_system),
                                  U.to_field_T(Ts_user, unit_system)))
            st.session_state["sep_train"] = new_train
        with c_train[1]:
            st.markdown("##### Presets")
            preset = st.selectbox(
                "Load preset",
                ["(keep current)", "3-stage (typical)", "2-stage (low-GOR)",
                 "1-stage (standard conditions only)"], key="sep_preset")
            if st.button("Apply preset", use_container_width=True):
                if preset == "3-stage (typical)":
                    st.session_state["sep_train"] = [(800.0, 100.0),
                                                     (100.0, 80.0),
                                                     (14.7, 60.0)]
                elif preset == "2-stage (low-GOR)":
                    st.session_state["sep_train"] = [(100.0, 80.0),
                                                     (14.7, 60.0)]
                elif preset == "1-stage (standard conditions only)":
                    st.session_state["sep_train"] = [(14.7, 60.0)]
                st.rerun()

        if st.button("Run separator train", type="primary"):
            with st.spinner("Running multi-stage flash..."):
                try:
                    sep_result = run_separator_train(
                        z_arr, comp_names, st.session_state["sep_train"], c7_props)
                except Exception as e:
                    st.error(f"Separator train failed: {e}")
                    sep_result = None

            if sep_result:
                m = st.columns(4)
                m[0].metric(f"GOR ({L['Rs']})",
                            f"{U.to_user_Rs(sep_result['GOR_scfSTB'], unit_system):.1f}")
                m[1].metric("ST Oil API",
                            f"{sep_result['st_oil_API']:.1f}" if not np.isnan(sep_result['st_oil_API']) else "—")
                m[2].metric("Gas SG",
                            f"{sep_result['gas_SG']:.4f}" if not np.isnan(sep_result['gas_SG']) else "—")
                m[3].metric(f"ST oil density ({L['rho']})",
                            f"{U.to_user_rho(sep_result['st_oil_density'], unit_system):.2f}"
                            if not np.isnan(sep_result['st_oil_density']) else "—")

                st.markdown("##### Per-stage breakdown")
                stage_df = pd.DataFrame([{
                    "Stage": s["stage"],
                    f"P ({L['P']})": U.to_user_P(s["P"], unit_system),
                    f"T ({L['T']})": U.to_user_T(s["T_F"], unit_system),
                    "Vapor mol frac": s["V_frac"],
                    "n_oil_out (lbmol)": s["n_oil_out"],
                    "n_gas_out (lbmol)": s["n_gas_out"],
                } for s in sep_result["stage_results"]])
                styled_dataframe(stage_df, height=240)

    # ============================================================
    # TAB — EOS Tuning
    # ============================================================
    with tab_tuning:
        st.markdown(
            "**Regress EOS parameters to lab measurements.** Provide measurements "
            "(saturation pressure, Rs, Bo, etc.), choose which C7+ parameters and "
            "kij values to free, then run the Levenberg-Marquardt optimizer."
        )
        from eos_tuning import tune_eos

        if "tuning_meas" not in st.session_state:
            st.session_state["tuning_meas"] = [
                {"type": "Psat", "value": Psat if Psat else 2500.0,
                 "kind": "bubble" if kind == "bubble" else "dew", "weight": 2.0}
            ]

        st.markdown("##### Measurements")
        meas_to_remove = []
        for i, m in enumerate(st.session_state["tuning_meas"]):
            mc = st.columns([2, 2, 2, 1, 1])
            with mc[0]:
                m["type"] = st.selectbox(
                    f"Type #{i+1}",
                    ["Psat", "Rs", "Bo", "GOR", "rho_st_oil"],
                    index=["Psat", "Rs", "Bo", "GOR", "rho_st_oil"].index(m["type"])
                    if m["type"] in ["Psat", "Rs", "Bo", "GOR", "rho_st_oil"] else 0,
                    key=f"meas_type_{i}")
            with mc[1]:
                if m["type"] in ("Rs", "Bo"):
                    Pval_user = st.number_input(
                        f"P ({L['P']})",
                        value=U.to_user_P(m.get("P", Psat or 2000), unit_system),
                        key=f"meas_P_{i}")
                    m["P"] = U.to_field_P(Pval_user, unit_system)
                else:
                    st.write(" ")
            with mc[2]:
                m["value"] = st.number_input(
                    "Measured value", value=float(m["value"]),
                    key=f"meas_val_{i}", format="%.4f")
            with mc[3]:
                m["weight"] = st.number_input(
                    "Weight", value=float(m.get("weight", 1.0)),
                    min_value=0.0, key=f"meas_w_{i}")
            with mc[4]:
                if st.button("✕", key=f"meas_rm_{i}", help="Remove this measurement"):
                    meas_to_remove.append(i)

        if meas_to_remove:
            for idx in sorted(meas_to_remove, reverse=True):
                st.session_state["tuning_meas"].pop(idx)
            st.rerun()

        if st.button("➕ Add measurement"):
            st.session_state["tuning_meas"].append(
                {"type": "Rs", "value": 500.0, "P": P_res, "weight": 1.0})
            st.rerun()

        st.markdown("##### Free parameters")
        free_param_opts = ["Pc_C7+", "Tc_C7+", "omega_C7+",
                            "kij_C1_C7+", "kij_N2_C7+"]
        free_selected = st.multiselect(
            "Parameters to tune",
            free_param_opts,
            default=["Pc_C7+", "Tc_C7+", "omega_C7+", "kij_C1_C7+"])

        if st.button("Run tuning", type="primary"):
            if not st.session_state["tuning_meas"]:
                st.error("Add at least one measurement first.")
            elif not free_selected:
                st.error("Select at least one parameter to tune.")
            else:
                with st.spinner("Optimizing EOS parameters (this can take a minute)..."):
                    try:
                        tune_result = tune_eos(
                            z_arr, comp_names, T_R, c7_props,
                            st.session_state["tuning_meas"],
                            free_params=free_selected)
                    except Exception as e:
                        st.error(f"Tuning failed: {e}")
                        tune_result = None

                if tune_result:
                    cm = st.columns(3)
                    cm[0].metric("Initial RMS", f"{tune_result['rms_initial']:.4f}")
                    cm[1].metric("Final RMS",   f"{tune_result['rms_final']:.4f}")
                    cm[2].metric("Iterations",  f"{tune_result['n_iter']}")

                    st.markdown("##### Parameter changes")
                    param_df = pd.DataFrame({
                        "Parameter": tune_result["param_names"],
                        "Initial":   tune_result["x_full_init"],
                        "Final":     tune_result["x_full_final"],
                        "Change %":  100.0 * (tune_result["x_full_final"] -
                                                tune_result["x_full_init"]) /
                                      np.maximum(np.abs(tune_result["x_full_init"]), 1e-6),
                    })
                    styled_dataframe(param_df, height=240)

                    st.markdown("##### Fit quality")
                    fit_df = pd.DataFrame({
                        "Measurement": [m["type"] for m in st.session_state["tuning_meas"]],
                        "Observed":    tune_result["observed"],
                        "Initial pred.": tune_result["predicted_initial"],
                        "Final pred.":   tune_result["predicted_final"],
                    })
                    styled_dataframe(fit_df, height=240)

    # ============================================================
    # TAB — Multi-Region
    # ============================================================
    with tab_multireg:
        st.markdown(
            "Generate a **multi-region PVT** include file (PVTNUM > 1). "
            "Each region uses the *current* composition and a region-specific "
            "saturation-point offset; useful when you have layered reservoirs "
            "with similar fluids but different bubble/dew points."
        )
        from multi_region import build_multi_region_deck

        n_regions = st.number_input("Number of regions", value=2, min_value=1, max_value=8)

        region_specs = []
        for i in range(int(n_regions)):
            with st.expander(f"Region {i+1}", expanded=(i == 0)):
                cr = st.columns(3)
                with cr[0]:
                    psat_offset_user = st.number_input(
                        f"Psat offset for region {i+1} ({L['P']})",
                        value=(i * 200.0 if unit_system == "Field"
                                else i * 13.8),
                        key=f"reg_offset_{i}",
                        help="Added to the base Psat to perturb this region")
                with cr[1]:
                    region_kind = st.selectbox(
                        f"Region {i+1} fluid kind",
                        ["oil", "gas-wet"],
                        index=(0 if fluid_kind == "oil" else 1),
                        key=f"reg_kind_{i}")
                region_specs.append({
                    "psat_offset": U.to_field_P(psat_offset_user, unit_system)
                                    if unit_system == "SI"
                                    else psat_offset_user,
                    "kind": region_kind,
                })

        if st.button("Build multi-region deck", type="primary"):
            if not bot_rows:
                st.error("Run the Black-oil table experiment first to populate "
                            "the base table.")
            else:
                # Compute real surface densities from the EOS standard-conditions split
                n_o, n_g, V_o, V_g, x_oil_sc, y_gas_sc = \
                    standard_conditions_split(z_arr, comp_names, c7_props)
                MW_arr = np.array([get_props(c, c7_props)["MW"]
                                    for c in comp_names])
                rho_o_sc = ((n_o * float(np.dot(x_oil_sc, MW_arr))) /
                              (V_o * 5.615)
                              if (V_o > 0 and n_o > 0) else 50.0)
                rho_g_sc = (0.0764 * float(np.dot(y_gas_sc, MW_arr)) / 28.97
                              if y_gas_sc.sum() > 0 else 0.05)
                # Water density: depends on salinity; use a default brine
                rho_w_sc = 62.428 * 1.02   # ~63.7 lb/ft3 typical brine

                regions_data = []
                for i, spec in enumerate(region_specs):
                    if spec["kind"] == "oil":
                        kw_text = build_pvto_from_compositional(
                            bot_rows, Psat + spec["psat_offset"], P_max)
                        region_kind_tag = "oil"
                    else:
                        kw_text = build_pvtg_from_compositional(
                            bot_rows, Psat + spec["psat_offset"])
                        region_kind_tag = "gas-wet"
                    regions_data.append({
                        "kind":     region_kind_tag,
                        "pvt_text": kw_text,
                        "density":  (rho_o_sc, rho_w_sc, rho_g_sc),
                    })
                deck = build_multi_region_deck(regions_data)
                st.code(deck, language="text")
                st.download_button("Download multi-region deck (.INC)", deck,
                                    file_name="PVT_MULTIREGION.INC",
                                    mime="text/plain", type="primary")

    # ============================================================
    # TAB — Monte Carlo
    # ============================================================
    with tab_mc:
        st.markdown(
            "**Monte Carlo uncertainty analysis** for oil-correlation parameters. "
            "Currently runs against the **correlation-based** oil model, not the EOS. "
            "Switch to the *Oil (Black Oil)* fluid type for a pure-correlation MC analysis."
        )
        st.info("Compositional MC requires the **Oil (Black Oil)** fluid type. "
                "Use that branch for the full MC tornado + histogram view.")

    # ============================================================
    # TAB — Docs
    # ============================================================
    with tab_docs:
        st.markdown("### Equations and references")

        with st.expander("Peng-Robinson Equation of State (1976)"):
            st.markdown(
                r"The Peng-Robinson EOS for a single component is:"
            )
            st.latex(r"P = \frac{RT}{v-b} - \frac{a\,\alpha(T)}{v(v+b)+b(v-b)}")
            st.markdown(r"with parameters:")
            st.latex(r"a = 0.45724 \frac{R^2 T_c^2}{P_c},\quad b = 0.07780 \frac{RT_c}{P_c}")
            st.latex(r"\alpha(T) = \left[1 + m\left(1 - \sqrt{T/T_c}\right)\right]^2")
            st.latex(r"m = 0.37464 + 1.54226\omega - 0.26992\omega^2 \quad (\omega \leq 0.49)")
            st.markdown(
                "For mixtures, van der Waals one-fluid mixing rules with binary "
                "interaction coefficients $k_{ij}$ are used."
            )

        with st.expander("Two-phase Flash (Rachford-Rice)"):
            st.markdown(
                "Given feed composition $z_i$ and equilibrium constants $K_i$, "
                "find vapor fraction $V$ satisfying:"
            )
            st.latex(r"\sum_i \frac{z_i (K_i - 1)}{1 + V(K_i - 1)} = 0")
            st.markdown(
                "Solved by Newton iteration with bracketing. Phase compositions:"
            )
            st.latex(r"x_i = \frac{z_i}{1 + V(K_i - 1)},\quad y_i = K_i x_i")

        with st.expander("Standing bubble-point correlation"):
            st.latex(r"P_b = 18.2 \left[\left(\frac{R_s}{\gamma_g}\right)^{0.83}"
                     r"\cdot 10^{0.00091\,T - 0.0125\,\text{API}} - 1.4\right]")
            st.markdown("Valid for $T$ in °F, $P_b$ in psia.")

        with st.expander("Vasquez-Beggs Rs / Pb correlation"):
            st.latex(r"R_s = C_1 \gamma_g P^{C_2} \exp\left(C_3 \frac{\text{API}}{T+460}\right)")
            st.markdown(
                "Coefficients $(C_1, C_2, C_3)$ depend on whether API ≤ 30 or > 30."
            )

        with st.expander("Hall-Yarborough Z-factor (1973)"):
            st.markdown(
                "Implicit equation for reduced density $y$ given reduced pressure $P_r$ "
                "and reduced temperature $T_r = 1/t$:"
            )
            st.latex(r"-AP_r + \frac{y + y^2 + y^3 - y^4}{(1-y)^3} - By^2 + Cy^D = 0")
            st.markdown("Solved by Newton iteration; $Z = A P_r / y$.")

        with st.expander("Lohrenz-Bray-Clark viscosity (1964)"):
            st.markdown(
                "Polynomial in reduced density $\\rho_r = \\rho \\cdot V_c$:"
            )
            st.latex(r"\left[(\mu - \mu^*)\xi + 10^{-4}\right]^{1/4} = "
                     r"0.1023 + 0.023364\rho_r + 0.058533\rho_r^2 "
                     r"- 0.040758\rho_r^3 + 0.0093324\rho_r^4")
            st.markdown(
                "where $\\xi$ is the viscosity reducing parameter, "
                "$\\mu^*$ is the dilute-gas viscosity (Stiel-Thodos)."
            )

        with st.expander("Kesler-Lee C7+ characterization"):
            st.markdown("Given molecular weight $M$ and specific gravity $\\gamma_o$ of "
                         "the heavy fraction:")
            st.latex(r"T_c = 341.7 + 811\gamma_o + (0.4244 + 0.1174\gamma_o)T_b + "
                     r"\frac{(0.4669 - 3.2623\gamma_o) \cdot 10^5}{T_b}")
            st.markdown("Pc and acentric factor follow similar empirical forms.")

        with st.expander("Multi-stage separator flash"):
            st.markdown(
                "At each stage $i$, the **liquid** from stage $i-1$ is flashed at "
                "$(P_i, T_i)$. Total surface GOR is:"
            )
            st.latex(r"\text{GOR} = \frac{\sum_i n_{g,i} \cdot V_{m,SC}}"
                     r"{V_{\text{oil},\text{ST}}}")
            st.markdown(
                "where $V_{m,SC} \\approx 379.5$ scf/lbmol is the molar volume at "
                "standard conditions and $V_{\\text{oil},\\text{ST}}$ is the "
                "stock-tank oil volume."
            )

        with st.expander("References"):
            st.markdown("""
- Peng, D.Y. & Robinson, D.B. (1976). *A New Two-Constant Equation of State*. Ind. Eng. Chem. Fund.
- Whitson, C.H. & Brulé, M.R. (2000). *Phase Behavior*. SPE Monograph Vol. 20.
- Ahmed, T. (2010). *Reservoir Engineering Handbook* (4th ed.). Gulf Professional Publishing.
- McCain, W.D. (1990). *Properties of Petroleum Fluids* (2nd ed.). PennWell.
- Coats, K.H. (1985). *Simulation of Gas Condensate Reservoir Performance*. JPT.
- Schlumberger ECLIPSE Reference Manual.
""")

    # ============================================================
    # TAB — ECLIPSE Export
    # ============================================================
    with tab_export:
        st.markdown("ECLIPSE PVTO/PVTG generated from the EOS black-oil table. "
                    "Run the 'Black-oil table' experiment in the Lab Experiments tab "
                    "to populate this section.")

        # ECLIPSE units selector
        eclipse_units = st.radio(
            "ECLIPSE deck unit system",
            ["FIELD", "METRIC"],
            horizontal=True,
            help=("ECLIPSE PVT keywords must be in *consistent* units throughout "
                  "the deck — set with the matching RUNSPEC keyword. The app "
                  "internally builds tables in FIELD then converts to METRIC "
                  "if requested. The sidebar Field/SI toggle controls the "
                  "*display* units in the rest of the app — the two are decoupled.")
        )

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

            # Convert to metric if requested
            if eclipse_units == "METRIC":
                from eclipse_export import convert_deck_to_metric
                conv = convert_deck_to_metric(
                    pvto=(kw_text if fluid_kind == "oil" else ""),
                    pvtg=(kw_text if fluid_kind != "oil" else ""),
                    pvtw=pvtw_text, density=density_text)
                kw_text_out = (conv["pvto"] if fluid_kind == "oil" else conv["pvtg"])
                pvtw_out = conv["pvtw"]
                density_out = conv["density"]
            else:
                kw_text_out = kw_text
                pvtw_out = pvtw_text
                density_out = density_text

            st.code(kw_text_out + ("\n" + pvtw_out if pvtw_out else ""), language="text")
            if fluid_kind == "oil":
                deck = build_full_deck(pvto=kw_text_out, pvtw=pvtw_out,
                                        density=density_out, units=eclipse_units)
                fname = f"PVT_COMPOSITIONAL_OIL_{eclipse_units}.INC"
            else:
                deck = build_full_deck(pvtg=kw_text_out, pvtw=pvtw_out,
                                        density=density_out, units=eclipse_units)
                fname = f"PVT_COMPOSITIONAL_GAS_{eclipse_units}.INC"
            st.download_button("Download PVT deck (.INC)", deck,
                                file_name=fname, mime="text/plain", type="primary")

            # ---- RSVD / RVVD vs depth ----
            st.markdown("---")
            st.markdown("### RSVD / RVVD — composition vs depth")
            st.markdown(
                "ECLIPSE supports Rs-vs-depth (`RSVD` for live oils) and "
                "Rv-vs-depth (`RVVD` for gas condensates) to represent "
                "compositional grading with depth."
            )
            from eclipse_export import build_rsvd, build_rvvd

            if "depth_grading" not in st.session_state:
                Rsi_now = (bot_rows[-1].get("Rs", 600) if fluid_kind == "oil"
                           else bot_rows[-1].get("Rv", 0.08))
                base_d = 8000.0 if unit_system == "Field" else 2440.0
                st.session_state["depth_grading"] = [
                    {"depth": base_d, "value": Rsi_now},
                    {"depth": base_d + 200.0, "value": Rsi_now * 1.05},
                    {"depth": base_d + 500.0, "value": Rsi_now * 1.10},
                ]

            depth_label = "ft" if unit_system == "Field" else "m"
            grading_to_remove = []
            for i, pt in enumerate(st.session_state["depth_grading"]):
                gc = st.columns([2, 2, 1])
                with gc[0]:
                    pt["depth"] = st.number_input(
                        f"Depth ({depth_label}) #{i+1}",
                        value=float(pt["depth"]), key=f"depth_d_{i}")
                with gc[1]:
                    label = f"Rs ({L['Rs']})" if fluid_kind == "oil" else f"Rv ({L['Rv']})"
                    pt["value"] = st.number_input(
                        label, value=float(pt["value"]),
                        key=f"depth_v_{i}", format="%.4f")
                with gc[2]:
                    if st.button("✕", key=f"depth_rm_{i}"):
                        grading_to_remove.append(i)

            if grading_to_remove:
                for idx in sorted(grading_to_remove, reverse=True):
                    st.session_state["depth_grading"].pop(idx)
                st.rerun()

            if st.button("➕ Add depth point"):
                st.session_state["depth_grading"].append(
                    {"depth": 8500.0 if unit_system == "Field" else 2590.0,
                     "value": 600.0 if fluid_kind == "oil" else 0.08})
                st.rerun()

            if st.button("Generate RSVD/RVVD keyword", type="primary"):
                pairs = [(p["depth"], p["value"])
                         for p in st.session_state["depth_grading"]]
                if fluid_kind == "oil":
                    grad_text = build_rsvd(pairs, units=eclipse_units)
                else:
                    grad_text = build_rvvd(pairs, units=eclipse_units)
                st.code(grad_text, language="text")
                st.download_button(
                    f"Download {'RSVD' if fluid_kind == 'oil' else 'RVVD'} keyword",
                    grad_text,
                    file_name=f"{'RSVD' if fluid_kind == 'oil' else 'RVVD'}.INC",
                    mime="text/plain")

        else:
            st.info("Run the **Black-oil table** experiment first to generate ECLIPSE keywords.")


# ================================================================
# HYDRATE LIKELIHOOD
# ================================================================
elif fluid == "❄️ Hydrate Likelihood":
    from hydrate import (hydrate_pressure_makogon, hydrate_temperature_makogon,
                          assess_hydrate_risk, hydrate_curve,
                          inhibitor_concentration_hammerschmidt)
    import plotly.graph_objects as go

    st.markdown("## Hydrate Formation Likelihood")
    st.markdown(
        "Predict whether the current operating (P, T) point is inside the "
        "**hydrate-forming envelope** for the gas being produced or transported. "
        "Uses the **Makogon (1981)** correlation with sour-gas corrections. "
        "For deeper analysis, a flash-based hydrate equilibrium model is needed — "
        "this tool is a fast screening for surface flowlines, subsea tiebacks, "
        "and choke / wellhead conditions."
    )

    col_h_in, col_h_out = st.columns([1, 2])

    with col_h_in:
        st.markdown("### Operating Point")
        T_op_user = st.number_input(
            f"Operating temperature ({L['T']})",
            value=U.to_user_T(40.0, unit_system),
            help="Temperature at the point of interest (subsea seabed, choke, "
                 "flowline midpoint, etc.).")
        P_op_user = st.number_input(
            f"Operating pressure ({L['P']})",
            value=U.to_user_P(3000.0, unit_system),
            min_value=U.to_user_P(14.7, unit_system),
            help="Pressure at the point of interest.")
        T_op_F = U.to_field_T(T_op_user, unit_system)
        P_op_psia = U.to_field_P(P_op_user, unit_system)

        st.markdown("### Gas Properties")
        gas_sg_h = st.number_input(
            "Gas specific gravity (air = 1)",
            value=0.65, min_value=0.55, max_value=1.0, step=0.01,
            help="Specific gravity of the gas phase. Heavier (richer) gas "
                 "forms hydrates at lower P / higher T.")
        H2S_h = st.number_input(
            "H2S mol fraction",
            value=0.0, min_value=0.0, max_value=0.3, step=0.01, format="%.4f",
            help="H2S strongly promotes hydrate formation: each 1% lowers the "
                 "hydrate-formation pressure by ~5%.")
        CO2_h = st.number_input(
            "CO2 mol fraction",
            value=0.0, min_value=0.0, max_value=0.5, step=0.01, format="%.4f",
            help="CO2 modestly promotes hydrates: each 1% lowers P_hyd by ~1.5%.")

        st.markdown("### Safety Margin")
        margin_user = st.number_input(
            f"Warning margin ({L['P']})",
            value=U.to_user_P(200.0, unit_system),
            min_value=0.0,
            help="A point this far from the hydrate boundary is flagged "
                 "'marginal' rather than fully 'safe' or 'in-zone'.")
        margin_psia = U.to_field_P(margin_user, unit_system) - U.to_field_P(0.0, unit_system) if unit_system == "SI" else margin_user

    # Run the assessment
    risk = assess_hydrate_risk(T_op_F, P_op_psia, gas_sg_h, H2S_h, CO2_h,
                                safety_margin_psia=margin_psia)

    with col_h_out:
        st.markdown("### Risk Assessment")

        # Big traffic-light banner
        if risk["risk_level"] == "in_zone":
            banner_color = "#EB0037"     # red
            icon = "🛑"
            level_text = "IN HYDRATE ZONE"
        elif risk["risk_level"] == "marginal":
            banner_color = "#C58B00"     # amber
            icon = "⚠️"
            level_text = "MARGINAL — near boundary"
        elif risk["risk_level"] == "safe":
            banner_color = "#9DBA00"     # pistachio
            icon = "✓"
            level_text = "SAFE — outside hydrate zone"
        else:
            banner_color = "#888888"
            icon = "?"
            level_text = "UNKNOWN"

        st.markdown(
            f"<div style='background-color:{banner_color}; padding:1rem 1.2rem; "
            f"border-radius:4px; color:white; font-size:1.4rem; font-weight:600; "
            f"text-align:center; margin-bottom:0.8rem;'>"
            f"{icon} &nbsp; {level_text}</div>",
            unsafe_allow_html=True)

        st.info(risk["message"])

        # Key metrics
        if not np.isnan(risk["P_hydrate"]):
            cm = st.columns(3)
            cm[0].metric(
                f"P_hydrate at T = {T_op_user:.1f} {L['T']}",
                f"{U.to_user_P(risk['P_hydrate'], unit_system):.0f} {L['P']}",
                help="Pressure above which hydrates form at the operating temperature.")
            if not np.isnan(risk["T_hydrate"]):
                cm[1].metric(
                    f"T_hydrate at P = {P_op_user:.0f} {L['P']}",
                    f"{U.to_user_T(risk['T_hydrate'], unit_system):.1f} {L['T']}",
                    help="Temperature above which hydrates do NOT form at the "
                         "operating pressure.")
            else:
                cm[1].metric(f"T_hydrate", "—")
            margin_p_user = U.to_user_P(risk["margin_psia"], unit_system) - \
                            U.to_user_P(0, unit_system) if unit_system == "SI" \
                            else risk["margin_psia"]
            cm[2].metric(
                f"P margin ({L['P']})",
                f"{margin_p_user:+.0f}",
                help="P_operating - P_hydrate. Positive = inside hydrate zone, "
                     "negative = outside.")

    # ---- Hydrate envelope plot ----
    st.markdown("---")
    st.markdown("### Hydrate Envelope")
    st.markdown(
        "Below: the hydrate-formation P-T curve for this gas composition. "
        "Operating points **above** the curve (high P, low T) are in the "
        "hydrate-forming region; **below** the curve (low P, high T) are safe."
    )

    T_curve_F, P_curve_psia = hydrate_curve(gas_sg_h, H2S_h, CO2_h, n_points=60)

    fig = go.Figure()
    # Hydrate locus
    T_curve_user = [U.to_user_T(t, unit_system) for t in T_curve_F]
    P_curve_user = [U.to_user_P(p, unit_system) for p in P_curve_psia]
    fig.add_trace(go.Scatter(
        x=T_curve_user, y=P_curve_user,
        name="Hydrate formation locus",
        mode="lines",
        line=dict(color="#3A6E96", width=3),
        fill='tozeroy',
        fillcolor='rgba(58, 110, 150, 0.08)',
        hovertemplate="<b>Hydrate boundary</b><br>"
                      f"T=%{{x:.1f}} {L['T']}<br>P=%{{y:.0f}} {L['P']}<extra></extra>",
    ))

    # Operating point
    op_color = ("#EB0037" if risk["risk_level"] == "in_zone"
                 else "#C58B00" if risk["risk_level"] == "marginal"
                 else "#9DBA00" if risk["risk_level"] == "safe"
                 else "#888888")
    fig.add_trace(go.Scatter(
        x=[T_op_user], y=[P_op_user],
        name="Operating point",
        mode="markers+text",
        marker=dict(size=18, color=op_color, symbol="diamond",
                    line=dict(color="#00243D", width=2)),
        text=[f"  ({T_op_user:.1f}, {P_op_user:.0f})"],
        textposition="middle right",
        textfont=dict(color="#00243D", size=12, family="Inter, sans-serif"),
        hovertemplate=f"<b>Operating point</b><br>T={T_op_user:.1f} {L['T']}"
                      f"<br>P={P_op_user:.0f} {L['P']}<extra></extra>",
    ))

    # Shade the unsafe (above-curve) region — approximate it with a top band
    P_max_chart = max(P_curve_user) * 1.3 if P_curve_user else P_op_user * 1.5
    fig.add_annotation(
        x=T_curve_user[0] if T_curve_user else T_op_user,
        y=P_max_chart * 0.85,
        text="<b>HYDRATE ZONE</b>",
        showarrow=False,
        font=dict(size=14, color="#EB0037"),
        align="left",
        xanchor="left",
    )
    fig.add_annotation(
        x=T_curve_user[-1] if T_curve_user else T_op_user,
        y=P_max_chart * 0.1,
        text="<b>SAFE — no hydrates</b>",
        showarrow=False,
        font=dict(size=14, color="#9DBA00"),
        align="right",
        xanchor="right",
    )

    fig.update_layout(**TH.plotly_layout(
        title="Hydrate Formation P-T Envelope (Makogon)",
        xtitle=f"Temperature ({L['T']})",
        ytitle=f"Pressure ({L['P']})",
        height=520))
    fig.update_yaxes(range=[0, P_max_chart])
    st.plotly_chart(fig, use_container_width=True)

    # ---- Inhibitor selection ----
    st.markdown("---")
    st.markdown("### Inhibitor Requirement (Hammerschmidt)")
    st.markdown(
        "If operating in the hydrate zone, an inhibitor (methanol or glycol) "
        "can suppress the hydrate-formation temperature by binding water and "
        "depressing its freezing point. The Hammerschmidt equation estimates "
        "the required inhibitor concentration:"
    )
    st.latex(r"W = \frac{\Delta T \cdot M \cdot 100}{K_H + \Delta T \cdot M}")
    st.caption(
        "where $W$ = inhibitor concentration (wt %), $\\Delta T$ = required "
        "temperature suppression (°F), $M$ = inhibitor molecular weight, "
        "and $K_H$ = Hammerschmidt constant (2335 for methanol, 2222 for MEG)."
    )

    # Required temperature suppression
    if not np.isnan(risk["T_hydrate"]) and risk["in_hydrate_zone"]:
        delta_T_default = (risk["T_hydrate"] - T_op_F) + 5.0  # 5 °F safety margin
    else:
        delta_T_default = 0.0

    cinh = st.columns(2)
    with cinh[0]:
        delta_T_user = st.number_input(
            f"Required ΔT suppression ({L['T']})",
            value=max(delta_T_default, 0.0) if unit_system == "Field"
                   else max(delta_T_default * 5.0/9.0, 0.0),
            min_value=0.0, step=1.0,
            help="How many degrees the hydrate-formation T must be depressed "
                 "to clear the operating point with margin.")
        delta_T_F = delta_T_user if unit_system == "Field" else delta_T_user * 9.0/5.0
    with cinh[1]:
        inhibitor = st.selectbox(
            "Inhibitor",
            ["methanol", "MEG", "DEG", "TEG"],
            help="Methanol is the most common offshore inhibitor (cheap, "
                 "effective, but volatile). MEG is preferred when recovery "
                 "and regeneration is feasible.")

    if delta_T_F > 0:
        W_required = inhibitor_concentration_hammerschmidt(delta_T_F, inhibitor)
        st.success(
            f"**{inhibitor.upper()} concentration required: "
            f"{W_required:.1f} wt%** "
            f"(to suppress hydrate formation T by {delta_T_F:.1f} °F)"
        )
        # Also display all four inhibitors for comparison
        comp_rows = []
        for inh in ["methanol", "MEG", "DEG", "TEG"]:
            W = inhibitor_concentration_hammerschmidt(delta_T_F, inh)
            comp_rows.append({"Inhibitor": inh, "Concentration (wt %)": W})
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No inhibitor needed — the current operating point is already "
                "outside the hydrate-forming region.")

    # ---- Notes ----
    st.markdown("---")
    with st.expander("📖 About the Makogon correlation and limitations"):
        st.markdown(r"""
The Makogon hydrate equilibrium curve is:

""")
        st.latex(r"\log_{10}(P) = \beta + 0.0497 \cdot (T - T_0) + 0.00034 \cdot (T - T_0)^2")
        st.markdown(r"""
where $P$ is in MPa, $T$ in °C, $T_0$ = 273.15 K (0 °C), and
""")
        st.latex(r"\beta = 2.681 - 3.811 \cdot \gamma_g + 1.679 \cdot \gamma_g^2")
        st.markdown(r"""
with $\gamma_g$ the gas specific gravity (air = 1).

**Validity:**
- Temperature: 32 °F (0 °C) to ~75 °F (24 °C)
- Gas SG: 0.55 to 1.0
- Sweet natural gas (correction applied for moderate H2S, CO2)

**Limitations:**
- Single-component (methane-dominated) model — real multi-component
  natural gas can deviate by 100–300 psia.
- The sour-gas corrections here are empirical first-order; for systems with
  > 15% H2S use a rigorous flash-based hydrate model (CSMHyd, PVTsim,
  Multiflash, etc.).
- Does not account for salt content of co-produced water (salt depresses
  hydrate formation T by ~1–2 °F per 1 wt% NaCl).
- The Hammerschmidt inhibitor equation is valid for ΔT < 40 °F. For deeper
  suppression, use the Nielsen-Bucklin equation.

**References:**
- Makogon, Y.F. (1981). *Hydrates of Natural Gas*. PennWell.
- Sloan, E.D. & Koh, C.A. (2007). *Clathrate Hydrates of Natural Gases*.
- Hammerschmidt, E.G. (1934). *Formation of Gas Hydrates in Natural Gas
  Transmission Lines*. Ind. Eng. Chem.
- Bahadori, A. & Vuthaluru, H.B. (2009). New correlation for hydrate
  formation conditions. *J. Nat. Gas Sci. Eng.*
""")


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
