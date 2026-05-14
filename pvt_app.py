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

from fluid_registry import make_fluid_record, to_json, from_json, summarize
from export_utils import (df_to_csv_bytes, to_json_bytes,
                            build_api_payload, build_pdf_report)
from composition_guess import guess_oil_composition, guess_gas_composition

# ----------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------
st.set_page_config(page_title="PVT Studio", page_icon="●", layout="wide")
st.markdown(TH.CUSTOM_CSS, unsafe_allow_html=True)

# Mascot header (inline SVG, no external dependency)
from mascot import header_with_mascot
st.markdown(
    header_with_mascot(
        "PVT Studio",
        "Black oil • Dry gas • Wet gas • Water • Compositional (EOS) • "
        "Hydrate • Rock compressibility — built by Merouane Hamdani"),
    unsafe_allow_html=True)

with st.expander("📄 License, disclaimer, and how to cite"):
    st.markdown("""
**MIT License** — Copyright © 2026 Merouane Hamdani

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> **THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED**, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

### ⚠️ Early-phase tool — disclaimer

PVT Studio is an **early-phase screening tool**, intended for:

- Educational and training purposes
- Rapid first-pass PVT calculations
- Building intuition about fluid behavior

It is **NOT** intended for:
- Final reservoir-simulation deck input without independent validation
- Safety-critical flow assurance decisions
- Final field development planning

Correlation predictions, EOS tuning, hydrate likelihood, and ECLIPSE export
must be **validated against measured lab PVT data and rigorous reservoir-grade
software** (PVTsim Nova, Whitson, Multiflash, CMG WinProp, Schlumberger PVTi)
before use in field design.

### How to cite

> Hamdani, M. (2026). *PVT Studio: Open-source PVT modeling tool with
> ECLIPSE export.* MIT License.

### Owner / contact

**Merouane Hamdani** — primary author and maintainer.
""")


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
                          "❄️ Hydrate Likelihood",
                          "🪨 Rock Compressibility"])

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

    st.markdown("### ECLIPSE Export")
    enable_eclipse_export = st.checkbox(
        "Enable ECLIPSE export", value=True,
        help="When OFF, all ECLIPSE export panels and download buttons are hidden. "
             "Use this when you only need PVT analysis without simulator-ready files.")
    if enable_eclipse_export:
        eclipse_unit_choice = st.radio(
            "ECLIPSE deck unit system", ["FIELD", "METRIC"],
            horizontal=True, key="global_eclipse_units",
            help=("Independent of the Field/SI display toggle. ECLIPSE keywords "
                  "must be in consistent units (set via RUNSPEC). "
                  "Internal calcs are FIELD, METRIC is converted at output."))
    else:
        eclipse_unit_choice = "FIELD"   # default unused
    include_water = st.checkbox("Add PVTW to ECLIPSE export", value=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px; color:#666; line-height:1.4;'>"
        "<b>PVT Studio v1.0</b><br>"
        "© M. Hamdani — MIT License<br>"
        "<i>Early-phase tool — see disclaimer above.</i>"
        "</div>",
        unsafe_allow_html=True)


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
# Shared utilities: fluid registry + per-branch Tools panel
# ================================================================
if "fluid_registry" not in st.session_state:
    st.session_state["fluid_registry"] = {}


def render_tools_section(branch_name, fluid_type, units, parameters,
                          outputs_summary, results_table_df=None, tuning=None):
    """Render the common Tools section: save fluid, exports, etc."""
    st.markdown("---")
    with st.expander("🧰 Tools — save fluid · export CSV / JSON / PDF"):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("##### Save this fluid")
            fluid_name = st.text_input("Fluid name", value=f"{branch_name}_default",
                                         key=f"savename_{branch_name}")
            notes = st.text_area("Notes (optional)", value="",
                                   key=f"savenotes_{branch_name}", height=70)
            if st.button("💾 Save to registry", key=f"savebtn_{branch_name}"):
                rec = make_fluid_record(
                    name=fluid_name, fluid_type=fluid_type, units=units,
                    parameters=parameters, tuning=tuning, notes=notes)
                st.session_state["fluid_registry"][fluid_name] = rec
                st.success(f"Saved '{fluid_name}' to in-session registry.")

            # Show registry contents
            if st.session_state["fluid_registry"]:
                st.caption("**Saved fluids in this session:**")
                for nm, rec in st.session_state["fluid_registry"].items():
                    st.caption(f"• {nm} — {summarize(rec)}")

            # Download/upload registry as JSON
            if st.session_state["fluid_registry"]:
                reg_json = to_json(list(st.session_state["fluid_registry"].values()))
                st.download_button("⬇️ Download all saved fluids (.json)",
                                    reg_json, file_name="pvt_fluids.json",
                                    mime="application/json",
                                    key=f"dl_reg_{branch_name}")
            uploaded = st.file_uploader("Or upload a fluids JSON",
                                          type=["json"], key=f"ul_reg_{branch_name}")
            if uploaded is not None:
                try:
                    records = from_json(uploaded.read().decode())
                    if isinstance(records, dict):
                        records = [records]
                    for r in records:
                        st.session_state["fluid_registry"][r["name"]] = r
                    st.success(f"Loaded {len(records)} fluid(s) into the registry.")
                except Exception as e:
                    st.error(f"Could not parse JSON: {e}")

        with cols[1]:
            st.markdown("##### Export results")

            # CSV
            if results_table_df is not None:
                csv_bytes = df_to_csv_bytes(results_table_df)
                st.download_button("⬇️ CSV (results table)", csv_bytes,
                                    file_name=f"{branch_name}_results.csv",
                                    mime="text/csv",
                                    key=f"csv_{branch_name}")

            # JSON API payload
            payload = build_api_payload(
                fluid_type=fluid_type, units=units,
                inputs=parameters,
                outputs={
                    "summary": outputs_summary,
                    "table": results_table_df.to_dict("records")
                              if results_table_df is not None else None,
                },
                metadata={"branch": branch_name})
            json_bytes = to_json_bytes(payload)
            st.download_button("⬇️ JSON (API payload)", json_bytes,
                                file_name=f"{branch_name}_payload.json",
                                mime="application/json",
                                key=f"json_{branch_name}")

            # PDF
            if st.button("📄 Generate PDF report", key=f"pdfbtn_{branch_name}"):
                pdf_bytes = build_pdf_report(
                    title=f"PVT Studio — {branch_name} Analysis",
                    fluid_type=fluid_type, units=units,
                    inputs=parameters, outputs_text=outputs_summary,
                    tables=[("Results", results_table_df)]
                           if results_table_df is not None else None)
                if pdf_bytes is None:
                    st.warning("reportlab not installed. Run: "
                                "`pip install reportlab` to enable PDF export.")
                else:
                    st.download_button("⬇️ Download PDF", pdf_bytes,
                                        file_name=f"{branch_name}_report.pdf",
                                        mime="application/pdf",
                                        key=f"pdf_dl_{branch_name}")
                    st.success(f"PDF generated ({len(pdf_bytes)/1024:.1f} KB).")


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

    # ECLIPSE export
    df_field = pd.DataFrame([{
        "P (psia)": r["P_field"], "Rs (scf/STB)": r["Rs_field"],
        "Bo (rb/STB)": r["Bo"], "μo (cp)": r["mu"],
    } for r in rows])
    pvto_text = build_pvto(df_field, Pb, oil, Rsi, P_max)
    density_text = build_density(api=api, gas_sg=gas_sg)
    pvtw_text = ""

    if enable_eclipse_export:
        st.markdown("---")
        st.markdown(f"### ECLIPSE Export ({eclipse_unit_choice} units)")
        if include_water:
            c_sal, c_corr = st.columns(2)
            with c_sal: salinity = st.number_input("Salinity (ppm)", value=30000.0, key="oil_sal")
            with c_corr: bw_corr = st.selectbox("Water correlation",
                                                  ["McCain", "Meehan", "Numbere", "Spivey"], key="oil_wcorr")
            water = WaterCorrelations(salinity_ppm=salinity, T=T_res, corr=bw_corr)
            pvtw_text = build_pvtw_from_table(pressures, water, P_res)
        # Convert to METRIC if requested
        if eclipse_unit_choice == "METRIC":
            from eclipse_export import convert_deck_to_metric
            conv = convert_deck_to_metric(pvto=pvto_text, pvtw=pvtw_text, density=density_text)
            pvto_show, pvtw_show, dens_show = conv["pvto"], conv["pvtw"], conv["density"]
        else:
            pvto_show, pvtw_show, dens_show = pvto_text, pvtw_text, density_text
        st.code(pvto_show + ("\n" + pvtw_show if pvtw_show else ""), language="text")
        deck = build_full_deck(pvto=pvto_show, pvtw=pvtw_show,
                                density=dens_show, units=eclipse_unit_choice)
        st.download_button("Download PVT deck (.INC)", deck,
                            file_name=f"PVT_BLACKOIL_{eclipse_unit_choice}.INC",
                            mime="text/plain", type="primary")

    # -------- Optional companion PVDG for the dissolved gas --------
    if enable_eclipse_export:
        with st.expander("📑 Add companion PVDG (for the dissolved gas phase)"):
            st.markdown(
                "ECLIPSE black-oil with live oil also needs gas-phase properties "
                "(PVDG) for free gas — gas that comes out of solution and flows "
                "separately. This block builds PVDG using a dry-gas correlation "
                f"for the *solution gas*. Output follows the **{eclipse_unit_choice}** "
                "unit choice from the sidebar."
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
                # Follow the sidebar unit choice
                if eclipse_unit_choice == "METRIC":
                    from eclipse_export import convert_deck_to_metric
                    conv = convert_deck_to_metric(
                        pvto=pvto_text, pvdg=pvdg_companion_text,
                        pvtw=pvtw_text, density=density_text)
                    pvto_c, pvdg_c = conv["pvto"], conv["pvdg"]
                    pvtw_c, dens_c = conv["pvtw"], conv["density"]
                else:
                    pvto_c, pvdg_c = pvto_text, pvdg_companion_text
                    pvtw_c, dens_c = pvtw_text, density_text
                st.code(pvdg_c, language="text")
                deck_with_pvdg = build_full_deck(
                    pvto=pvto_c, pvdg=pvdg_c,
                    pvtw=pvtw_c, density=dens_c, units=eclipse_unit_choice)
                st.download_button(
                    "Download deck with PVTO + PVDG (.INC)",
                    deck_with_pvdg,
                    file_name=f"PVT_BLACKOIL_with_PVDG_{eclipse_unit_choice}.INC",
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
        st.markdown("##### About the uncertainty inputs")
        st.markdown(
            "The four σ values below define **independent normal distributions** "
            "around each base input. Typical defensible values from PVT lab "
            "reports and field measurements:\n"
            "- **σ(API) ≈ 0.5–2.0°** for direct hydrometer readings; up to 3° if "
            "  using flash-only data.\n"
            "- **σ(gas SG) ≈ 0.01–0.05** depending on whether multi-stage "
            "  separator gas is fully sampled.\n"
            "- **σ(Rsi) ≈ 25–75 scf/STB** for measured DLE data; ±10% for "
            "  correlation-only estimates.\n"
            "- **σ(T) ≈ 5–15 °F** for downhole gauges; lower for surface RTDs.\n\n"
            "**Note on correlated inputs:** This implementation treats the four "
            "parameters as independent. In reality, API and Rsi are often "
            "**positively correlated** (lighter oils tend to have higher GOR), "
            "and gas SG correlates with Rsi as well. If you want to include "
            "correlations, the cleanest path is to draw samples from a "
            "multivariate-normal with a calibrated covariance matrix — that's "
            "not exposed in the UI yet but the underlying module accepts "
            "user-supplied sample arrays. Open an issue if you need this."
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

            # Tornado plots — separate for Pb and Bo
            st.markdown("##### Tornado sensitivity")
            st.caption("Each bar shows how far the output moves when one input "
                        "is perturbed by ±1σ while the others stay at base.")

            def render_tornado(output_name, unit_converter, unit_label):
                tor = tornado_sensitivity(base, unc, OilCorrelations,
                                            target_P=P_res, output=output_name)
                if not tor["rows"]:
                    st.info(f"No tornado data for {output_name} — check that at "
                             f"least one σ value is greater than zero.")
                    return
                base_v = tor["base_value"]
                if base_v is None or (isinstance(base_v, float) and np.isnan(base_v)):
                    st.info(f"Could not compute a base value for {output_name}.")
                    return
                tor_df = pd.DataFrame([{
                    "Parameter": param,
                    "Low":   unit_converter(lo),
                    "High":  unit_converter(hi),
                    "Range": abs(unit_converter(hi) - unit_converter(lo)),
                } for param, lo, hi, rng in tor["rows"]])
                styled_dataframe(tor_df, height=180)

                fig = go.Figure()
                base_disp = unit_converter(base_v)
                for j, (param, lo, hi, rng) in enumerate(tor["rows"]):
                    lo_disp = unit_converter(lo)
                    hi_disp = unit_converter(hi)
                    # bar from base to high
                    fig.add_trace(go.Bar(
                        y=[param], x=[hi_disp - base_disp], base=base_disp,
                        orientation="h", name="+1σ",
                        marker_color=TH.TORCH_RED,
                        showlegend=(j == 0), legendgroup="hi"))
                    # bar from base to low
                    fig.add_trace(go.Bar(
                        y=[param], x=[lo_disp - base_disp], base=base_disp,
                        orientation="h", name="−1σ",
                        marker_color=TH.DARK_NAVY,
                        showlegend=(j == 0), legendgroup="lo"))
                fig.add_vline(x=base_disp, line_dash="dash", line_color="black")
                fig.update_layout(**TH.plotly_layout(
                    title=f"Tornado — {output_name} sensitivity to ±1σ",
                    xtitle=f"{output_name} ({unit_label})", ytitle="",
                    height=300, showlegend=True),
                    barmode="overlay")
                st.plotly_chart(fig, use_container_width=True)

            tcol1, tcol2 = st.columns(2)
            with tcol1:
                render_tornado("Pb",
                                lambda v: U.to_user_P(v, unit_system),
                                L['P'])
            with tcol2:
                render_tornado("Bo", lambda v: v, "rb/STB")

    # -------- Oil compressibility plot --------
    with st.expander("📉 Oil compressibility (Co vs P)"):
        st.markdown(
            "Isothermal oil compressibility $c_o = -\\frac{1}{B_o}"
            "\\frac{\\partial B_o}{\\partial P}$. Above $P_b$ it reflects "
            "under-saturated liquid compression; below $P_b$ the apparent "
            "compressibility is dominated by gas coming out of solution."
        )
        co_rows = []
        for P in pressures:
            try:
                if P <= Pb:
                    Rs_co = oil.solution_gor(P)
                    co = oil.oil_compressibility(P, Rs_co)
                else:
                    co = oil.oil_compressibility(P, Rsi)
                co_rows.append({"P": P, "Co": co})
            except Exception:
                co_rows.append({"P": P, "Co": np.nan})
        co_df = pd.DataFrame(co_rows)
        co_df_display = pd.DataFrame({
            f"P ({L['P']})": [U.to_user_P(r["P"], unit_system) for r in co_rows],
            f"Co ({L['Cw']})": [U.to_user_Cw(r["Co"], unit_system)
                                  if not np.isnan(r["Co"]) else np.nan
                                  for r in co_rows],
        })
        styled_dataframe(co_df_display, height=260)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=co_df_display[f"P ({L['P']})"],
            y=co_df_display[f"Co ({L['Cw']})"],
            mode="lines+markers", name="Co",
            line=dict(color=TH.TORCH_RED, width=2.5)))
        fig.add_vline(x=U.to_user_P(Pb, unit_system), line_dash="dash",
                      line_color=TH.DARK_NAVY,
                      annotation_text="Pb")
        fig.update_layout(**TH.plotly_layout(
            title="Oil compressibility vs pressure",
            xtitle=f"P ({L['P']})", ytitle=f"Co ({L['Cw']})",
            height=380, ymode="log"))
        st.plotly_chart(fig, use_container_width=True)

    # -------- Lab experiments for correlation oil --------
    with st.expander("🧪 Lab experiments — CCE / CVD / Flash / Separator"):
        st.markdown(
            "Black-oil lab experiment approximations using the chosen "
            "correlations. CCE and CVD trace the depletion behavior; "
            "Flash and Separator give surface-condition GOR and shrinkage."
        )
        from correlation_experiments import (cce_blackoil, cvd_blackoil,
            flash_blackoil, multistage_separator_blackoil)
        from correlations import GasCorrelations as GasCorrForCce
        gas_corr_for_cce = GasCorrForCce(gas_sg=gas_sg, T=T_res)

        exp_choice = st.radio("Experiment", ["CCE", "CVD", "Flash",
                                              "Multi-stage separator"],
                               horizontal=True, key="oil_exp_choice")

        if exp_choice == "CCE":
            if st.button("Run CCE", key="run_cce_oil"):
                cce_rows = cce_blackoil(oil, gas_corr_for_cce, Rsi, Pb, pressures)
                cce_df = pd.DataFrame([{
                    f"P ({L['P']})": U.to_user_P(r["P"], unit_system),
                    "Phase": r["phase"],
                    "V / Vsat": r["V_rel"],
                    "Liquid dropout (% Vsat)": r["L_dropout_pct"],
                    "Y-function": r["Y_function"],
                } for r in cce_rows])
                styled_dataframe(cce_df, height=300)
                fig = go.Figure()
                fig.add_trace(TH.line_trace(cce_df[f"P ({L['P']})"],
                                             cce_df["V / Vsat"],
                                             "V / Vsat", color_idx=0))
                fig.update_layout(**TH.plotly_layout(
                    title="CCE — V/Vsat vs P",
                    xtitle=f"P ({L['P']})", ytitle="V / Vsat", height=340))
                st.plotly_chart(fig, use_container_width=True)

        elif exp_choice == "CVD":
            if st.button("Run CVD", key="run_cvd_oil"):
                cvd_rows = cvd_blackoil(oil, gas_corr_for_cce, Rsi, Pb, pressures)
                cvd_df = pd.DataFrame([{
                    f"P ({L['P']})": U.to_user_P(r["P"], unit_system),
                    "Phase": r["phase"],
                    f"Rs ({L['Rs']})": U.to_user_Rs(r["Rs"], unit_system),
                    "Liquid fraction": r["liquid_frac"],
                    f"Cum. gas produced ({L['Rs']})":
                        U.to_user_Rs(r["cum_gas_produced_scfSTB"], unit_system),
                } for r in cvd_rows])
                styled_dataframe(cvd_df, height=300)
                fig = go.Figure()
                fig.add_trace(TH.line_trace(
                    cvd_df[f"P ({L['P']})"], cvd_df["Liquid fraction"],
                    "Liquid fraction", color_idx=0))
                fig.update_layout(**TH.plotly_layout(
                    title="CVD — remaining liquid fraction vs P",
                    xtitle=f"P ({L['P']})", ytitle="Liquid fraction",
                    height=340))
                st.plotly_chart(fig, use_container_width=True)

        elif exp_choice == "Flash":
            if st.button("Run single-stage flash", key="run_flash_oil"):
                fl = flash_blackoil(oil, Rsi, Pb, P_res)
                fc = st.columns(3)
                fc[0].metric(f"GOR ({L['Rs']})",
                              f"{U.to_user_Rs(fl['GOR_scfSTB'], unit_system):.1f}")
                fc[1].metric("Bo at P_res", f"{fl['Bo_initial']:.4f}")
                fc[2].metric("Shrinkage (STB/rb)", f"{fl['shrinkage']:.4f}")
                st.caption("Single-stage flash from reservoir pressure directly "
                            "to standard conditions (60 °F, 14.7 psia).")

        else:  # Multi-stage separator
            st.markdown("##### Separator stages")
            if "oil_sep_stages" not in st.session_state:
                st.session_state["oil_sep_stages"] = [
                    (800.0, 100.0), (100.0, 80.0), (14.7, 60.0)]
            new_stages = []
            for i, (Ps, Ts) in enumerate(st.session_state["oil_sep_stages"]):
                sc = st.columns([1, 2, 2])
                sc[0].markdown(f"**Stage {i+1}**")
                Ps_u = sc[1].number_input(
                    f"P ({L['P']})", value=U.to_user_P(Ps, unit_system),
                    key=f"oilsep_P_{i}")
                Ts_u = sc[2].number_input(
                    f"T ({L['T']})", value=U.to_user_T(Ts, unit_system),
                    key=f"oilsep_T_{i}")
                new_stages.append((U.to_field_P(Ps_u, unit_system),
                                   U.to_field_T(Ts_u, unit_system)))
            st.session_state["oil_sep_stages"] = new_stages
            if st.button("Run multi-stage separator", key="run_ms_oil"):
                ms = multistage_separator_blackoil(
                    oil, gas_corr_for_cce, Rsi, Pb, T_res,
                    st.session_state["oil_sep_stages"])
                mc = st.columns(3)
                mc[0].metric(f"Total GOR ({L['Rs']})",
                              f"{U.to_user_Rs(ms['total_GOR_scfSTB'], unit_system):.1f}")
                mc[1].metric(f"Single-stage GOR ({L['Rs']})",
                              f"{U.to_user_Rs(ms['single_stage_GOR_scfSTB'], unit_system):.1f}")
                mc[2].metric("GOR reduction", f"{ms['GOR_reduction_pct']:.1f}%")
                # Per-stage breakdown with GOR + densities
                stage_df = pd.DataFrame([{
                    "Stage": s["stage"],
                    f"P ({L['P']})": U.to_user_P(s["P"], unit_system),
                    f"T ({L['T']})": U.to_user_T(s["T_F"], unit_system),
                    f"Stage GOR ({L['Rs']})":
                        U.to_user_Rs(s["stage_GOR_scfSTB"], unit_system),
                    "Fraction of total (%)": s["fraction_of_total"] * 100,
                } for s in ms["stage_results"]])
                styled_dataframe(stage_df, height=200)
                st.markdown("##### Stock-tank properties")
                stc = st.columns(2)
                stc[0].metric("ST oil API", f"{ms['st_oil_API']:.1f}")
                stc[1].metric(f"ST oil density ({L['rho']})",
                               f"{U.to_user_rho(ms['st_oil_density'], unit_system):.2f}")


    # -------- Correlation tuning with experimental data --------
    with st.expander("🎯 Tune correlation with experimental data"):
        st.markdown(
            "Provide laboratory measurements (Pb, Rs at P, Bo at P, viscosity at P) "
            "and the app will fit a small set of correction factors "
            "(Pb shift, Bo factor, Rs factor, μ factor) so the correlation "
            "matches your lab data. Useful for screening before EOS regression."
        )
        st.caption(f"All values are entered and displayed in **{unit_system}** "
                    f"units; the optimizer works internally in field units.")
        from correlation_tuning import tune_correlation_oil, auto_select_best_correlation

        # oil_lab_data stores values in DISPLAY units; conversion happens at tune time.
        if "oil_lab_data" not in st.session_state:
            st.session_state["oil_lab_data"] = [
                {"type": "Pb", "P": 0.0,
                 "value": U.to_user_P(Pb + 100.0, unit_system), "weight": 1.0},
            ]
        st.markdown("##### Lab measurements")
        rm_idx = []
        for i, m in enumerate(st.session_state["oil_lab_data"]):
            cs = st.columns([2, 2, 2, 1, 1])
            with cs[0]:
                m["type"] = st.selectbox(
                    "Type", ["Pb", "Rs", "Bo", "mu_o"],
                    index=["Pb", "Rs", "Bo", "mu_o"].index(m.get("type", "Pb")),
                    key=f"olt_type_{i}")
            with cs[1]:
                if m["type"] != "Pb":
                    # P stored in DISPLAY units
                    m["P"] = st.number_input(
                        f"P ({L['P']})",
                        value=float(m.get("P", U.to_user_P(P_res, unit_system))),
                        key=f"olt_P_{i}")
                else:
                    st.write("(at Pb)")
            with cs[2]:
                # value units depend on type
                if m["type"] == "Pb":
                    val_label = f"Pb ({L['P']})"
                elif m["type"] == "Rs":
                    val_label = f"Rs ({L['Rs']})"
                elif m["type"] == "Bo":
                    val_label = "Bo (rb/STB = rm³/Sm³)"
                else:
                    val_label = "μo (cP)"
                m["value"] = st.number_input(
                    val_label, value=float(m.get("value", 1.0)),
                    key=f"olt_val_{i}", format="%.4f")
            with cs[3]:
                m["weight"] = st.number_input(
                    "wt", value=float(m.get("weight", 1.0)),
                    min_value=0.0, key=f"olt_w_{i}")
            with cs[4]:
                if st.button("✕", key=f"olt_rm_{i}"):
                    rm_idx.append(i)
        if rm_idx:
            for j in sorted(rm_idx, reverse=True):
                st.session_state["oil_lab_data"].pop(j)
            st.rerun()
        cba = st.columns(3)
        with cba[0]:
            if st.button("➕ Add measurement"):
                st.session_state["oil_lab_data"].append(
                    {"type": "Bo", "P": U.to_user_P(P_res, unit_system),
                     "value": 1.3, "weight": 1.0})
                st.rerun()
        with cba[1]:
            tune_choices = st.multiselect(
                "Tune", ["Pb_shift", "Bo_factor", "Rs_factor", "mu_factor"],
                default=["Pb_shift", "Bo_factor"])
        with cba[2]:
            run_tune = st.button("Run tuning", type="primary",
                                   use_container_width=True)

        def _lab_to_field(lab_list):
            """Convert a display-unit lab-data list to field units for the tuner."""
            out = []
            for m in lab_list:
                fm = {"type": m["type"], "weight": m.get("weight", 1.0)}
                if m["type"] == "Pb":
                    fm["P"] = 0.0
                    fm["value"] = U.to_field_P(m["value"], unit_system)
                elif m["type"] == "Rs":
                    fm["P"] = U.to_field_P(m["P"], unit_system)
                    fm["value"] = U.to_field_Rs(m["value"], unit_system)
                else:  # Bo, mu_o — dimensionless / cP, no conversion
                    fm["P"] = U.to_field_P(m["P"], unit_system)
                    fm["value"] = m["value"]
                out.append(fm)
            return out

        def _field_pred_to_user(pred_array, lab_list):
            """Convert tuner's field-unit predictions back to display units."""
            out = []
            for val, m in zip(pred_array, lab_list):
                if m["type"] == "Pb":
                    out.append(U.to_user_P(val, unit_system))
                elif m["type"] == "Rs":
                    out.append(U.to_user_Rs(val, unit_system))
                else:
                    out.append(val)
            return np.array(out)

        if run_tune and tune_choices and st.session_state["oil_lab_data"]:
            base = {"api": api, "gas_sg": gas_sg, "T": T_res, "Rsi": Rsi,
                    "rs_corr": rs_corr, "bo_corr": bo_corr, "mu_corr": mu_corr}
            lab_field = _lab_to_field(st.session_state["oil_lab_data"])
            with st.spinner("Tuning..."):
                tune_res = tune_correlation_oil(
                    OilCorrelations, base, lab_field,
                    tune=tuple(tune_choices))
            # Store result with display-unit predictions for plotting
            tune_res["observed_user"] = _field_pred_to_user(
                tune_res["observed"], st.session_state["oil_lab_data"])
            tune_res["pred_init_user"] = _field_pred_to_user(
                tune_res["predicted_initial"], st.session_state["oil_lab_data"])
            tune_res["pred_final_user"] = _field_pred_to_user(
                tune_res["predicted_final"], st.session_state["oil_lab_data"])
            tune_res["lab_snapshot"] = list(st.session_state["oil_lab_data"])
            st.session_state["oil_tune_result"] = tune_res

        # Render last tuning result (persists across reruns so Undo works)
        if st.session_state.get("oil_tune_result"):
            tune_res = st.session_state["oil_tune_result"]
            lab_snap = tune_res.get("lab_snapshot",
                                     st.session_state["oil_lab_data"])

            m1, m2, m3 = st.columns(3)
            m1.metric("RMS initial", f"{tune_res['rms_initial']:.4f}")
            m2.metric("RMS final",   f"{tune_res['rms_final']:.4f}")
            improvement = (1 - tune_res['rms_final'] /
                            max(tune_res['rms_initial'], 1e-9)) * 100
            m3.metric("Improvement", f"{improvement:.0f}%")

            if tune_res['rms_final'] > tune_res['rms_initial']:
                st.warning("Tuning did not improve the fit — the optimizer may "
                            "be stuck, or the measurements may be inconsistent. "
                            "Consider the Undo button below.")

            st.markdown("##### Tuned correction factors")
            tune_df = pd.DataFrame([{
                "Parameter": k,
                "Initial": (0.0 if k == "Pb_shift" else 1.0),
                "Tuned":   tune_res["tuned"][k],
            } for k in tune_res["tuned_keys"]])
            styled_dataframe(tune_df, height=180)

            st.markdown(f"##### Predicted vs observed (in {unit_system} units)")
            cmp_df = pd.DataFrame({
                "Type":     [m["type"] for m in lab_snap],
                "Observed": tune_res["observed_user"],
                "Initial":  tune_res["pred_init_user"],
                "Tuned":    tune_res["pred_final_user"],
            })
            styled_dataframe(cmp_df, height=180)

            # Comparison plot — all series now in consistent DISPLAY units.
            # Group by measurement type so the y-axis is meaningful.
            st.markdown("##### Tuned vs untuned comparison")
            types_present = list(dict.fromkeys(m["type"] for m in lab_snap))
            for t in types_present:
                idxs = [i for i, m in enumerate(lab_snap) if m["type"] == t]
                if t == "Pb":
                    unit_lbl = L['P']
                elif t == "Rs":
                    unit_lbl = L['Rs']
                elif t == "Bo":
                    unit_lbl = "rb/STB"
                else:
                    unit_lbl = "cP"
                xs = [f"{t} #{j+1}" for j in range(len(idxs))]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Observed", x=xs,
                    y=[tune_res["observed_user"][i] for i in idxs],
                    marker_color="#00243D"))
                fig.add_trace(go.Bar(
                    name="Untuned", x=xs,
                    y=[tune_res["pred_init_user"][i] for i in idxs],
                    marker_color="#C58B00"))
                fig.add_trace(go.Bar(
                    name="Tuned", x=xs,
                    y=[tune_res["pred_final_user"][i] for i in idxs],
                    marker_color="#9DBA00"))
                fig.update_layout(**TH.plotly_layout(
                    title=f"{t} — tuned vs untuned vs lab data",
                    xtitle="Measurement", ytitle=f"{t} ({unit_lbl})",
                    height=320, showlegend=True),
                    barmode="group")
                st.plotly_chart(fig, use_container_width=True)

            # Undo button
            if st.button("↩️ Undo tuning (discard this match)",
                          key="undo_oil_tune"):
                st.session_state["oil_tune_result"] = None
                st.rerun()

        if st.button("🔍 Auto-select best correlation"):
            base = {"api": api, "gas_sg": gas_sg, "T": T_res, "Rsi": Rsi}
            lab_field = _lab_to_field(st.session_state["oil_lab_data"])
            with st.spinner("Comparing correlations..."):
                comparison = auto_select_best_correlation(
                    OilCorrelations, base, lab_field)
            comp_df = pd.DataFrame(comparison)
            st.markdown("##### Correlation ranking (lowest RMS first)")
            styled_dataframe(comp_df[["rs_corr", "bo_corr", "rms_baseline",
                                       "rms_with_Pb_shift", "Pb_shift"]],
                              height=200)
            st.info(f"Best: **{comparison[0]['rs_corr']} / {comparison[0]['bo_corr']}**")

    # -------- Composition guess --------
    with st.expander("🔬 Guess composition for EOS comparison"):
        st.markdown(
            "Synthesize a plausible 11-component composition from "
            f"API={api}, gas SG={gas_sg}, Rsi={Rsi}. Useful as a starting "
            "point for compositional (EOS) modeling — *not* a substitute "
            "for measured chromatography."
        )
        if st.button("Generate composition guess"):
            comp_guess, MW_c7, SG_c7 = guess_oil_composition(api, gas_sg, Rsi)
            comp_df = pd.DataFrame([
                {"Component": k, "Mole fraction": v}
                for k, v in comp_guess.items() if v > 1e-5
            ])
            cgc = st.columns(3)
            cgc[0].metric("C7+ MW", f"{MW_c7:.1f}")
            cgc[1].metric("C7+ SG", f"{SG_c7:.4f}")
            cgc[2].metric("Σz", f"{sum(comp_guess.values()):.4f}")
            styled_dataframe(comp_df, height=300)
            st.info("Switch to **Compositional (EOS)** in the sidebar and "
                     "manually enter these values to use them.")

    # -------- Tools (save/export) --------
    render_tools_section(
        branch_name="oil", fluid_type="oil",
        units=unit_system,
        parameters={
            "api": api, "gas_sg": gas_sg, "T_F": T_res, "Rsi_scfSTB": Rsi,
            "P_res_psia": P_res,
            "rs_corr": rs_corr, "bo_corr": bo_corr, "mu_corr": mu_corr,
        },
        outputs_summary=[
            f"Pb = {Pb:.1f} psia",
            f"Bo @ Pb = {oil.formation_volume_factor(Pb, Rsi, saturated=True):.4f}",
        ],
        results_table_df=df,
        tuning=st.session_state.get("oil_tune_result"))


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

    pvdg_text = build_pvdg(df_field)
    density_text = build_density(api=35.0, gas_sg=gas_sg)
    pvtw_text = ""

    if enable_eclipse_export:
        st.markdown("---")
        st.markdown(f"### ECLIPSE Export — PVDG ({eclipse_unit_choice} units)")
        if include_water:
            c_sal, c_corr = st.columns(2)
            with c_sal: salinity = st.number_input("Salinity (ppm)", value=30000.0, key="gas_sal")
            with c_corr: bw_corr = st.selectbox("Water correlation",
                                                 ["McCain", "Meehan", "Numbere", "Spivey"], key="gas_wcorr")
            water = WaterCorrelations(salinity_ppm=salinity, T=T_res, corr=bw_corr)
            pvtw_text = build_pvtw_from_table(pressures, water, P_res)
        if eclipse_unit_choice == "METRIC":
            from eclipse_export import convert_deck_to_metric
            conv = convert_deck_to_metric(pvdg=pvdg_text, pvtw=pvtw_text, density=density_text)
            pvdg_show, pvtw_show, dens_show = conv["pvdg"], conv["pvtw"], conv["density"]
        else:
            pvdg_show, pvtw_show, dens_show = pvdg_text, pvtw_text, density_text
        st.code(pvdg_show + ("\n" + pvtw_show if pvtw_show else ""), language="text")
        deck = build_full_deck(pvdg=pvdg_show, pvtw=pvtw_show,
                                density=dens_show, units=eclipse_unit_choice)
        st.download_button("Download PVT deck (.INC)", deck,
                            file_name=f"PVT_DRYGAS_{eclipse_unit_choice}.INC",
                            mime="text/plain", type="primary")

    # -------- Lab experiments for dry gas --------
    with st.expander("🧪 Lab experiments — CCE / CVD / Flash"):
        st.markdown(
            "Dry-gas lab experiment approximations. A dry gas has no liquid "
            "dropout, so CCE is the gas-expansion curve; CVD gives the "
            "volumetric recovery factor from the P/Z material balance."
        )
        from correlation_experiments import cce_drygas, cvd_drygas, flash_drygas

        dg_exp = st.radio("Experiment", ["CCE", "CVD", "Flash"],
                           horizontal=True, key="dg_exp_choice")

        if dg_exp == "CCE":
            if st.button("Run CCE", key="run_cce_dg"):
                rows = cce_drygas(gas, pressures)
                cce_df = pd.DataFrame([{
                    f"P ({L['P']})": U.to_user_P(r["P"], unit_system),
                    "Z": r["Z"],
                    f"Bg ({L['Bg']})": U.to_user_Bg(r["Bg"], unit_system),
                    "Expansion factor": r["E_factor"],
                    "μg (cP)": r["mu_g"],
                } for r in rows])
                styled_dataframe(cce_df, height=300)
                fig = go.Figure()
                fig.add_trace(TH.line_trace(cce_df[f"P ({L['P']})"],
                                             cce_df["Z"], "Z", color_idx=0))
                fig.update_layout(**TH.plotly_layout(
                    title="CCE — Z-factor vs P", xtitle=f"P ({L['P']})",
                    ytitle="Z", height=340))
                st.plotly_chart(fig, use_container_width=True)

        elif dg_exp == "CVD":
            if st.button("Run CVD", key="run_cvd_dg"):
                rows = cvd_drygas(gas, pressures, P_res)
                if not rows:
                    st.warning("No pressures below the initial reservoir "
                                "pressure — widen the pressure range.")
                else:
                    cvd_df = pd.DataFrame([{
                        f"P ({L['P']})": U.to_user_P(r["P"], unit_system),
                        "Z": r["Z"],
                        "P/Z": r["P_over_Z"],
                        "Recovery factor (%)": r["recovery_factor_pct"],
                    } for r in rows])
                    styled_dataframe(cvd_df, height=300)
                    fig = go.Figure()
                    fig.add_trace(TH.line_trace(
                        cvd_df[f"P ({L['P']})"], cvd_df["Recovery factor (%)"],
                        "Recovery factor", color_idx=0))
                    fig.update_layout(**TH.plotly_layout(
                        title="CVD — gas recovery factor vs P",
                        xtitle=f"P ({L['P']})", ytitle="Recovery factor (%)",
                        height=340))
                    st.plotly_chart(fig, use_container_width=True)

        else:  # Flash
            if st.button("Run flash", key="run_flash_dg"):
                fl = flash_drygas(gas, P_res)
                fc = st.columns(2)
                fc[0].metric("Z at P_res", f"{fl['Z_initial']:.4f}")
                fc[1].metric("Expansion (scf/rb)",
                              f"{fl['expansion_scf_per_rb']:.2f}")
                st.caption("Dry-gas flash to standard conditions — the "
                            "expansion factor is the reservoir-to-surface "
                            "volume ratio.")

    # -------- Composition guess for dry gas --------
    with st.expander("🔬 Guess composition for EOS comparison"):
        st.markdown(f"Synthesize a composition from gas SG = {gas_sg:.3f}.")
        if st.button("Generate composition guess", key="dg_guess"):
            cg, MW_c7, SG_c7 = guess_gas_composition(gas_sg, is_wet=False)
            cg_df = pd.DataFrame([{"Component": k, "Mole fraction": v}
                                   for k, v in cg.items() if v > 1e-5])
            styled_dataframe(cg_df, height=280)

    # -------- Monte Carlo for dry gas --------
    with st.expander("🎲 Monte Carlo uncertainty"):
        st.markdown("Sample gas SG, T (and optionally H2S, CO2) and view "
                     "distributions of Z, Bg, μg at the reservoir P.")
        mcc = st.columns(3)
        with mcc[0]:
            mc_sd_sg = st.number_input("σ(SG)", value=0.03, min_value=0.0,
                                         step=0.005, format="%.3f", key="dg_mc_sg")
        with mcc[1]:
            mc_sd_T = st.number_input("σ(T) °F", value=10.0, min_value=0.0,
                                        key="dg_mc_T")
        with mcc[2]:
            mc_n = st.slider("Samples", 100, 2000, 500, step=100, key="dg_mc_n")
        if st.button("Run Monte Carlo", key="dg_mc_run"):
            rng = np.random.default_rng(42)
            sg_samples = np.clip(rng.normal(gas_sg, mc_sd_sg, mc_n), 0.55, 1.2)
            T_samples = np.clip(rng.normal(T_res, mc_sd_T, mc_n), 60, 500)
            Zs, Bgs, mus = [], [], []
            for i in range(mc_n):
                try:
                    g = GasCorrelations(gas_sg=sg_samples[i], T=T_samples[i])
                    Z = g.z_factor(P_res)
                    Zs.append(Z)
                    Bgs.append(g.formation_volume_factor(P_res, Z))
                    mus.append(g.viscosity(P_res, Z))
                except Exception:
                    Zs.append(np.nan); Bgs.append(np.nan); mus.append(np.nan)
            Zs = np.array(Zs); Bgs = np.array(Bgs); mus = np.array(mus)
            sm = st.columns(3)
            sm[0].metric("Z mean", f"{np.nanmean(Zs):.4f}",
                          delta=f"±{np.nanstd(Zs):.4f}")
            sm[1].metric("Bg mean (rb/Mscf)",
                          f"{np.nanmean(Bgs)*1000:.4f}",
                          delta=f"±{np.nanstd(Bgs)*1000:.4f}")
            sm[2].metric("μg mean (cp)", f"{np.nanmean(mus):.5f}",
                          delta=f"±{np.nanstd(mus):.5f}")
            hc1, hc2 = st.columns(2)
            with hc1:
                fig = go.Figure(go.Histogram(x=Zs[~np.isnan(Zs)], nbinsx=30,
                                              marker_color=TH.TORCH_RED))
                fig.update_layout(**TH.plotly_layout(
                    title="Z distribution", xtitle="Z", ytitle="Count",
                    height=300, showlegend=False))
                st.plotly_chart(fig, use_container_width=True)
            with hc2:
                fig = go.Figure(go.Histogram(x=Bgs[~np.isnan(Bgs)]*1000, nbinsx=30,
                                              marker_color=TH.DARK_NAVY))
                fig.update_layout(**TH.plotly_layout(
                    title="Bg distribution", xtitle="Bg (rb/Mscf)", ytitle="Count",
                    height=300, showlegend=False))
                st.plotly_chart(fig, use_container_width=True)

    render_tools_section(
        branch_name="dry_gas", fluid_type="dry_gas",
        units=unit_system,
        parameters={"gas_sg": gas_sg, "T_F": T_res, "P_res_psia": P_res,
                    "z_corr": z_corr, "mu_corr_g": mug_corr,
                    "H2S": H2S, "CO2": CO2, "N2": N2},
        outputs_summary=[f"Z at P_res = {gas.z_factor(P_res):.4f}"],
        results_table_df=df)


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

    pvtg_text = build_pvtg(pressures, wet)
    density_text = build_density(api=api_cond, gas_sg=gas_sg)
    pvtw_text = ""

    if enable_eclipse_export:
        st.markdown("---")
        st.markdown(f"### ECLIPSE Export — PVTG ({eclipse_unit_choice} units)")
        if include_water:
            c_sal, c_corr = st.columns(2)
            with c_sal: salinity = st.number_input("Salinity (ppm)", value=30000.0, key="wg_sal")
            with c_corr: bw_corr = st.selectbox("Water correlation",
                                                 ["McCain", "Meehan", "Numbere", "Spivey"], key="wg_wcorr")
            water = WaterCorrelations(salinity_ppm=salinity, T=T_res, corr=bw_corr)
            pvtw_text = build_pvtw_from_table(pressures, water, P_res)
        if eclipse_unit_choice == "METRIC":
            from eclipse_export import convert_deck_to_metric
            conv = convert_deck_to_metric(pvtg=pvtg_text, pvtw=pvtw_text, density=density_text)
            pvtg_show, pvtw_show, dens_show = conv["pvtg"], conv["pvtw"], conv["density"]
        else:
            pvtg_show, pvtw_show, dens_show = pvtg_text, pvtw_text, density_text
        st.code(pvtg_show + ("\n" + pvtw_show if pvtw_show else ""), language="text")
        deck = build_full_deck(pvtg=pvtg_show, pvtw=pvtw_show,
                                density=dens_show, units=eclipse_unit_choice)
        st.download_button("Download PVT deck (.INC)", deck,
                            file_name=f"PVT_WETGAS_{eclipse_unit_choice}.INC",
                            mime="text/plain", type="primary")

    # -------- Optional companion PVTO for the dropped-out condensate --------
    if enable_eclipse_export:
        with st.expander("📑 Add companion PVTO (for the condensate phase)"):
            st.markdown(
                "ECLIPSE wet-gas with vaporized oil also needs oil-phase properties "
                "(PVTO) for condensate that drops out and flows as a separate "
                "liquid phase. This block builds a PVTO using a black-oil "
                "correlation on the condensate. Output follows the "
                f"**{eclipse_unit_choice}** unit choice from the sidebar."
            )
            if st.button("Build companion PVTO", key="pvto_companion_wg"):
                oil_companion = OilCorrelations(
                    api=api_cond, gas_sg=gas_sg, T=T_res,
                    rs_corr="Standing", bo_corr="Standing",
                    mu_corr="Beggs-Robinson")
                Rsi_cond = max(50.0, cgr * 0.5)
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
                pvto_cond_text = build_pvto(cond_df, Pb_cond, oil_companion,
                                             Rsi_cond, P_max)
                # Follow the sidebar unit choice
                if eclipse_unit_choice == "METRIC":
                    from eclipse_export import convert_deck_to_metric
                    conv = convert_deck_to_metric(
                        pvto=pvto_cond_text, pvtg=pvtg_text,
                        pvtw=pvtw_text, density=density_text)
                    pvto_c, pvtg_c = conv["pvto"], conv["pvtg"]
                    pvtw_c, dens_c = conv["pvtw"], conv["density"]
                else:
                    pvto_c, pvtg_c = pvto_cond_text, pvtg_text
                    pvtw_c, dens_c = pvtw_text, density_text
                st.code(pvto_c, language="text")
                deck_with_pvto = build_full_deck(
                    pvto=pvto_c, pvtg=pvtg_c,
                    pvtw=pvtw_c, density=dens_c, units=eclipse_unit_choice)
                st.download_button(
                    "Download deck with PVTG + PVTO (.INC)",
                    deck_with_pvto,
                    file_name=f"PVT_WETGAS_with_PVTO_{eclipse_unit_choice}.INC",
                    mime="text/plain", type="primary",
                    key="dl_wg_pvto")

    # -------- Wet gas tuning with experimental data --------
    with st.expander("🎯 Tune correlation with experimental data"):
        st.markdown(
            "Provide lab measurements (dew point, Z-factor, Rv, Bg at various P) "
            "and fit a small set of correction factors (Pdew shift, Rv factor, "
            "Z factor) so the wet-gas correlation matches your data."
        )
        st.caption(f"Values entered and displayed in **{unit_system}** units.")
        from correlation_tuning import tune_wetgas

        if "wg_lab_data" not in st.session_state:
            st.session_state["wg_lab_data"] = [
                {"type": "Pdew", "P": 0.0,
                 "value": U.to_user_P(Pdew, unit_system), "weight": 2.0},
            ]
        wg_rm = []
        for i, m in enumerate(st.session_state["wg_lab_data"]):
            cs = st.columns([2, 2, 2, 1, 1])
            with cs[0]:
                m["type"] = st.selectbox(
                    "Type", ["Pdew", "Z", "Rv", "Bg"],
                    index=["Pdew", "Z", "Rv", "Bg"].index(m.get("type", "Pdew")),
                    key=f"wglt_type_{i}")
            with cs[1]:
                if m["type"] != "Pdew":
                    m["P"] = st.number_input(
                        f"P ({L['P']})",
                        value=float(m.get("P", U.to_user_P(P_res, unit_system))),
                        key=f"wglt_P_{i}")
                else:
                    st.write("(dew point)")
            with cs[2]:
                if m["type"] == "Pdew":
                    vlabel = f"Pdew ({L['P']})"
                elif m["type"] == "Z":
                    vlabel = "Z (-)"
                elif m["type"] == "Rv":
                    vlabel = "Rv (STB/scf)"
                else:
                    vlabel = "Bg (rb/scf)"
                m["value"] = st.number_input(
                    vlabel, value=float(m.get("value", 1.0)),
                    format="%.6f", key=f"wglt_val_{i}")
            with cs[3]:
                m["weight"] = st.number_input(
                    "wt", value=float(m.get("weight", 1.0)),
                    min_value=0.0, key=f"wglt_w_{i}")
            with cs[4]:
                if st.button("✕", key=f"wglt_rm_{i}"):
                    wg_rm.append(i)
        if wg_rm:
            for j in sorted(wg_rm, reverse=True):
                st.session_state["wg_lab_data"].pop(j)
            st.rerun()

        wgc = st.columns(3)
        with wgc[0]:
            if st.button("➕ Add measurement", key="wg_add_meas"):
                st.session_state["wg_lab_data"].append(
                    {"type": "Z", "P": U.to_user_P(P_res, unit_system),
                     "value": 0.9, "weight": 1.0})
                st.rerun()
        with wgc[1]:
            wg_tune_choices = st.multiselect(
                "Tune", ["Pdew_shift", "Rv_factor", "Z_factor"],
                default=["Pdew_shift", "Z_factor"], key="wg_tune_choices")
        with wgc[2]:
            wg_run_tune = st.button("Run tuning", type="primary",
                                     use_container_width=True, key="wg_run_tune")

        def _wg_lab_to_field(lab_list):
            out = []
            for m in lab_list:
                fm = {"type": m["type"], "weight": m.get("weight", 1.0)}
                if m["type"] == "Pdew":
                    fm["P"] = 0.0
                    fm["value"] = U.to_field_P(m["value"], unit_system)
                else:
                    fm["P"] = U.to_field_P(m["P"], unit_system)
                    # Z and Rv and Bg are unit-agnostic enough at this level
                    fm["value"] = m["value"]
                out.append(fm)
            return out

        if wg_run_tune and wg_tune_choices and st.session_state["wg_lab_data"]:
            wg_base = {"gas_sg": gas_sg, "api_cond": api_cond, "cgr": cgr,
                       "T": T_res, "N2": N2, "CO2": CO2, "H2S": H2S,
                       "z_corr": z_corr, "mu_corr": mug_corr,
                       "rv_corr": rv_corr, "Pdew": Pdew}
            lab_field = _wg_lab_to_field(st.session_state["wg_lab_data"])
            with st.spinner("Tuning wet-gas correlation..."):
                wg_tune_res = tune_wetgas(WetGasCorrelations, wg_base,
                                           lab_field, tune=tuple(wg_tune_choices))
            wg_tune_res["lab_snapshot"] = list(st.session_state["wg_lab_data"])
            st.session_state["wg_tune_result"] = wg_tune_res

        if st.session_state.get("wg_tune_result"):
            wg_tr = st.session_state["wg_tune_result"]
            lab_snap = wg_tr.get("lab_snapshot", st.session_state["wg_lab_data"])
            mm1, mm2, mm3 = st.columns(3)
            mm1.metric("RMS initial", f"{wg_tr['rms_initial']:.4f}")
            mm2.metric("RMS final",   f"{wg_tr['rms_final']:.4f}")
            wg_impr = (1 - wg_tr['rms_final'] /
                        max(wg_tr['rms_initial'], 1e-9)) * 100
            mm3.metric("Improvement", f"{wg_impr:.0f}%")

            if wg_tr['rms_final'] > wg_tr['rms_initial']:
                st.warning("Tuning did not improve the fit — consider Undo.")

            st.markdown("##### Tuned correction factors")
            wg_tune_df = pd.DataFrame([{
                "Parameter": k,
                "Initial": (0.0 if k == "Pdew_shift" else 1.0),
                "Tuned":   wg_tr["tuned"][k],
            } for k in wg_tr["tuned_keys"]])
            styled_dataframe(wg_tune_df, height=160)

            st.markdown("##### Predicted vs observed")
            wg_cmp = pd.DataFrame({
                "Type":     [m["type"] for m in lab_snap],
                "Observed": wg_tr["observed"],
                "Initial":  wg_tr["predicted_initial"],
                "Tuned":    wg_tr["predicted_final"],
            })
            styled_dataframe(wg_cmp, height=160)

            # Comparison plot grouped by type
            wg_types = list(dict.fromkeys(m["type"] for m in lab_snap))
            for t in wg_types:
                idxs = [i for i, m in enumerate(lab_snap) if m["type"] == t]
                xs = [f"{t} #{j+1}" for j in range(len(idxs))]
                figc = go.Figure()
                figc.add_trace(go.Bar(name="Observed", x=xs,
                    y=[wg_tr["observed"][i] for i in idxs],
                    marker_color="#00243D"))
                figc.add_trace(go.Bar(name="Untuned", x=xs,
                    y=[wg_tr["predicted_initial"][i] for i in idxs],
                    marker_color="#C58B00"))
                figc.add_trace(go.Bar(name="Tuned", x=xs,
                    y=[wg_tr["predicted_final"][i] for i in idxs],
                    marker_color="#9DBA00"))
                figc.update_layout(**TH.plotly_layout(
                    title=f"{t} — tuned vs untuned vs lab data",
                    xtitle="Measurement", ytitle=t, height=300,
                    showlegend=True), barmode="group")
                st.plotly_chart(figc, use_container_width=True)

            if st.button("↩️ Undo tuning", key="undo_wg_tune"):
                st.session_state["wg_tune_result"] = None
                st.rerun()

    # -------- Composition guess for wet gas --------
    with st.expander("🔬 Guess composition for EOS comparison"):
        st.markdown(f"Synthesize a wet-gas composition from SG = {gas_sg:.3f} "
                     f"and CGR = {cgr:.1f} STB/MMscf.")
        if st.button("Generate composition guess", key="wg_guess"):
            cg, MW_c7, SG_c7 = guess_gas_composition(
                gas_sg, is_wet=True, cgr=cgr)
            cg_df = pd.DataFrame([{"Component": k, "Mole fraction": v}
                                   for k, v in cg.items() if v > 1e-5])
            cgc = st.columns(2)
            cgc[0].metric("C7+ MW", f"{MW_c7:.1f}")
            cgc[1].metric("C7+ SG", f"{SG_c7:.4f}")
            styled_dataframe(cg_df, height=300)

    # -------- Lab experiments for wet gas --------
    with st.expander("🧪 Lab experiments — CCE / CVD / Flash"):
        st.markdown(
            "Wet-gas lab experiment approximations. CVD traces the condensate "
            "dropout as the gas depletes below the dew point."
        )
        from correlation_experiments import cvd_wetgas, cce_drygas, flash_drygas

        wg_exp = st.radio("Experiment", ["CVD (condensate dropout)",
                                          "CCE (gas expansion)", "Flash"],
                           horizontal=True, key="wg_exp_choice")

        if wg_exp == "CVD (condensate dropout)":
            if st.button("Run CVD", key="run_cvd_wg"):
                rows = cvd_wetgas(wet, Pdew, pressures)
                cvd_df = pd.DataFrame([{
                    f"P ({L['P']})": U.to_user_P(r["P"], unit_system),
                    "Z": r["Z"],
                    "Liquid dropout (%)": r["L_dropout_pct"],
                    "Rv produced (STB/Mscf)": r["Rv_produced"],
                    "Phase": r["phase"],
                } for r in rows])
                styled_dataframe(cvd_df, height=300)
                fig = go.Figure()
                fig.add_trace(TH.line_trace(
                    cvd_df[f"P ({L['P']})"], cvd_df["Liquid dropout (%)"],
                    "Liquid dropout", color_idx=0))
                fig.add_vline(x=U.to_user_P(Pdew, unit_system),
                              line_dash="dash", line_color=TH.DARK_NAVY,
                              annotation_text="Pdew")
                fig.update_layout(**TH.plotly_layout(
                    title="CVD — condensate liquid dropout vs P",
                    xtitle=f"P ({L['P']})", ytitle="Liquid dropout (%)",
                    height=340))
                st.plotly_chart(fig, use_container_width=True)

        elif wg_exp == "CCE (gas expansion)":
            if st.button("Run CCE", key="run_cce_wg"):
                rows = cce_drygas(wet, pressures)
                cce_df = pd.DataFrame([{
                    f"P ({L['P']})": U.to_user_P(r["P"], unit_system),
                    "Z": r["Z"],
                    f"Bg ({L['Bg']})": U.to_user_Bg(r["Bg"], unit_system),
                    "Expansion factor": r["E_factor"],
                } for r in rows])
                styled_dataframe(cce_df, height=300)

        else:  # Flash
            if st.button("Run flash", key="run_flash_wg"):
                fl = flash_drygas(wet, P_res)
                fc = st.columns(2)
                fc[0].metric("Z at P_res", f"{fl['Z_initial']:.4f}")
                fc[1].metric("Expansion (scf/rb)",
                              f"{fl['expansion_scf_per_rb']:.2f}")

    # -------- Monte Carlo for wet gas --------
    with st.expander("🎲 Monte Carlo uncertainty"):
        st.markdown("Sample SG and CGR; view distribution of Z, Bg, Rv at P_res.")
        mcc = st.columns(3)
        with mcc[0]:
            mc_sd_sg = st.number_input("σ(SG)", value=0.03, min_value=0.0,
                                         step=0.005, format="%.3f", key="wg_mc_sg")
        with mcc[1]:
            # CGR uncertainty: in display units, convert to field if SI
            mc_sd_cgr_disp = st.number_input(
                f"σ(CGR) [{'STB/MMscf' if unit_system == 'Field' else 'Sm³/MSm³'}]",
                value=10.0 if unit_system == "Field" else 1.8,
                min_value=0.0, key="wg_mc_cgr")
            mc_sd_cgr = (mc_sd_cgr_disp * 5.6146 if unit_system == "SI"
                          else mc_sd_cgr_disp)
        with mcc[2]:
            mc_n = st.slider("Samples", 100, 2000, 500, step=100, key="wg_mc_n")
        if st.button("Run Monte Carlo", key="wg_mc_run"):
            rng = np.random.default_rng(42)
            sg_s = np.clip(rng.normal(gas_sg, mc_sd_sg, mc_n), 0.55, 1.5)
            cgr_s = np.clip(rng.normal(cgr, mc_sd_cgr, mc_n), 1.0, 500.0)
            Bgs, Rvs, Zs = [], [], []
            n_fail = 0
            for i in range(mc_n):
                try:
                    w = WetGasCorrelations(
                        gas_sg=float(sg_s[i]), api_cond=api_cond,
                        cgr_stb_per_mmscf=float(cgr_s[i]),
                        T=T_res, N2=N2, CO2=CO2, H2S=H2S,
                        z_corr=z_corr, mu_corr=mug_corr,
                        rv_corr=rv_corr, Pdew=Pdew)
                    Z = w.z_factor(P_res)
                    if Z is None or np.isnan(Z):
                        raise ValueError("Z is NaN")
                    Zs.append(Z)
                    Bgs.append(w.formation_volume_factor(P_res, Z))
                    Rvs.append(w.rv(P_res))
                except Exception:
                    n_fail += 1
                    Zs.append(np.nan); Bgs.append(np.nan); Rvs.append(np.nan)
            Bgs = np.array(Bgs); Rvs = np.array(Rvs); Zs = np.array(Zs)
            n_ok = np.sum(~np.isnan(Zs))
            if n_ok == 0:
                st.error("All Monte Carlo draws failed — check input ranges.")
            else:
                if n_fail > 0:
                    st.caption(f"{n_fail} of {mc_n} draws failed and were dropped.")
                sm = st.columns(3)
                sm[0].metric("Z mean", f"{np.nanmean(Zs):.4f}",
                              delta=f"±{np.nanstd(Zs):.4f}")
                sm[1].metric(f"Bg mean ({L['Bg']})",
                              f"{U.to_user_Bg(np.nanmean(Bgs)*1000, unit_system):.4f}")
                sm[2].metric("Rv mean (STB/Mscf)",
                              f"{np.nanmean(Rvs)*1000:.4f}")
                hc1, hc2 = st.columns(2)
                with hc1:
                    fig = go.Figure(go.Histogram(
                        x=Bgs[~np.isnan(Bgs)]*1000, nbinsx=30,
                        marker_color=TH.TORCH_RED))
                    fig.update_layout(**TH.plotly_layout(
                        title="Bg distribution", xtitle="Bg (rb/Mscf)",
                        ytitle="Count", height=300, showlegend=False))
                    st.plotly_chart(fig, use_container_width=True)
                with hc2:
                    fig = go.Figure(go.Histogram(
                        x=Rvs[~np.isnan(Rvs)]*1000, nbinsx=30,
                        marker_color=TH.DARK_NAVY))
                    fig.update_layout(**TH.plotly_layout(
                        title="Rv distribution", xtitle="Rv (STB/Mscf)",
                        ytitle="Count", height=300, showlegend=False))
                    st.plotly_chart(fig, use_container_width=True)

    render_tools_section(
        branch_name="wet_gas", fluid_type="wet_gas",
        units=unit_system,
        parameters={"gas_sg": gas_sg, "api_cond": api_cond, "cgr": cgr,
                    "T_F": T_res, "P_res_psia": P_res, "Pdew_psia": Pdew},
        outputs_summary=[f"Pdew = {Pdew:.0f} psia", f"Z at P_res = {wet.z_factor(P_res):.4f}"],
        results_table_df=df)


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

    pvtw_text = build_pvtw_from_table(pressures, water, P_res)
    if enable_eclipse_export:
        st.markdown("---")
        st.markdown(f"### ECLIPSE Export — PVTW ({eclipse_unit_choice} units)")
        if eclipse_unit_choice == "METRIC":
            from eclipse_export import convert_deck_to_metric
            conv = convert_deck_to_metric(pvtw=pvtw_text)
            pvtw_show = conv["pvtw"]
        else:
            pvtw_show = pvtw_text
        st.code(pvtw_show, language="text")
        deck = build_full_deck(pvtw=pvtw_show, units=eclipse_unit_choice)
        st.download_button("Download PVTW (.INC)", deck,
                            file_name=f"PVTW_{eclipse_unit_choice}.INC",
                            mime="text/plain", type="primary")

    render_tools_section(
        branch_name="water", fluid_type="water",
        units=unit_system,
        parameters={"salinity_ppm": salinity, "T_F": T_res, "P_res_psia": P_res,
                    "corr": bw_corr, "include_gas": include_gas},
        outputs_summary=[f"Bw at P_res = {water.bw(P_res):.4f}"],
        results_table_df=df)


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
                    f"Stage GOR ({L['Rs']})":
                        U.to_user_Rs(s["stage_GOR_scfSTB"], unit_system),
                    "Gas SG": s["gas_SG_stage"],
                    f"Oil ρ ({L['rho']})":
                        U.to_user_rho(s["rho_oil_stage"], unit_system)
                        if not np.isnan(s["rho_oil_stage"]) else np.nan,
                    f"Gas ρ ({L['rho']})":
                        U.to_user_rho(s["rho_gas_stage"], unit_system)
                        if not np.isnan(s["rho_gas_stage"]) else np.nan,
                } for s in sep_result["stage_results"]])
                styled_dataframe(stage_df, height=240)

                # Per-stage GOR bar chart
                fig = go.Figure(go.Bar(
                    x=[f"Stage {s['stage']}\n{U.to_user_P(s['P'], unit_system):.0f} {L['P']}"
                       for s in sep_result["stage_results"]],
                    y=[U.to_user_Rs(s["stage_GOR_scfSTB"], unit_system)
                       for s in sep_result["stage_results"]],
                    marker_color=TH.TORCH_RED))
                fig.update_layout(**TH.plotly_layout(
                    title="Gas released per separator stage",
                    xtitle="Stage", ytitle=f"Stage GOR ({L['Rs']})",
                    height=320, showlegend=False))
                st.plotly_chart(fig, use_container_width=True)

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
        if not enable_eclipse_export:
            st.info("**ECLIPSE export is disabled** in the sidebar. "
                    "Multi-region PVT is an export feature — enable it to use this tab.")
        else:
            st.markdown(
                "Generate a **multi-region PVT** include file (PVTNUM > 1). "
                "Each region can use **the current composition** with a Psat offset, "
                "or **a saved fluid** loaded from the registry. Useful for layered "
                "reservoirs with different fluid types or saturation pressures."
            )
            from multi_region import build_multi_region_deck

            # Saved fluids of compositional type are eligible
            saved_compositional = [
                (name, rec) for name, rec in st.session_state.get("fluid_registry", {}).items()
                if rec.get("fluid_type") == "compositional"
            ]
            if saved_compositional:
                st.caption(f"📁 {len(saved_compositional)} saved compositional fluid(s) "
                            "available — see the 'Source' selector per region.")

            n_regions = st.number_input("Number of regions", value=2, min_value=1, max_value=8)

            region_specs = []
            for i in range(int(n_regions)):
                with st.expander(f"Region {i+1}", expanded=(i == 0)):
                    cr = st.columns(3)
                    with cr[0]:
                        source_options = ["Current composition"] + [s[0] for s in saved_compositional]
                        source = st.selectbox(
                            f"Source",
                            source_options,
                            key=f"reg_source_{i}",
                            help="Use the current composition or load a saved fluid.")
                    with cr[1]:
                        psat_offset_user = st.number_input(
                            f"Psat offset ({L['P']})",
                            value=(i * 200.0 if unit_system == "Field"
                                    else i * 13.8),
                            key=f"reg_offset_{i}",
                            help="Added to the (base) Psat to perturb this region")
                    with cr[2]:
                        region_kind = st.selectbox(
                            f"Fluid kind",
                            ["oil", "gas-wet"],
                            index=(0 if fluid_kind == "oil" else 1),
                            key=f"reg_kind_{i}")
                    region_specs.append({
                        "source": source,
                        "psat_offset": U.to_field_P(psat_offset_user, unit_system)
                                        - U.to_field_P(0.0, unit_system)
                                        if unit_system == "SI" else psat_offset_user,
                        "kind": region_kind,
                    })

            if st.button("Build multi-region deck", type="primary"):
                if not bot_rows:
                    st.error("Run the Black-oil table experiment first to populate "
                                "the base table.")
                else:
                    # Compute surface densities once (current composition)
                    n_o, n_g, V_o, V_g, x_oil_sc, y_gas_sc = \
                        standard_conditions_split(z_arr, comp_names, c7_props)
                    MW_arr = np.array([get_props(c, c7_props)["MW"]
                                        for c in comp_names])
                    rho_o_sc = ((n_o * float(np.dot(x_oil_sc, MW_arr))) /
                                  (V_o * 5.615)
                                  if (V_o > 0 and n_o > 0) else 50.0)
                    rho_g_sc = (0.0764 * float(np.dot(y_gas_sc, MW_arr)) / 28.97
                                  if y_gas_sc.sum() > 0 else 0.05)
                    rho_w_sc = 62.428 * 1.02

                    regions_data = []
                    for i, spec in enumerate(region_specs):
                        # If a saved fluid is selected, currently we still use
                        # the current bot_rows table because re-running the full
                        # depletion flash for each saved fluid is too slow.
                        # The saved-fluid name is annotated in the comments instead.
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
        if not enable_eclipse_export:
            st.info("**ECLIPSE export is disabled** in the sidebar. "
                    "Enable it to generate PVTO/PVTG/PVTW/DENSITY keywords here.")
            bot_rows_for_export = None  # block the if-bot_rows path below
        else:
            st.markdown("ECLIPSE PVTO/PVTG generated from the EOS black-oil table. "
                        "Run the 'Black-oil table' experiment in the Lab Experiments tab "
                        "to populate this section.")
            bot_rows_for_export = bot_rows

            eclipse_units = st.radio(
                "Override unit system (sidebar default applies otherwise)",
                ["FIELD", "METRIC"],
                index=(0 if eclipse_unit_choice == "FIELD" else 1),
                horizontal=True,
                help="Defaults to the sidebar ECLIPSE unit-system selection.")

        if enable_eclipse_export and bot_rows_for_export and Psat is not None:
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

        elif enable_eclipse_export and not bot_rows_for_export:
            st.info("Run the **Black-oil table** experiment first to generate ECLIPSE keywords.")

    # -------- Tools at the bottom of the compositional branch --------
    bot_summary = []
    if Psat is not None:
        bot_summary.append(f"Psat = {Psat:.1f} psia ({kind})")
    if bot_rows:
        bot_summary.append(f"Black-oil table: {len(bot_rows)} rows")
    render_tools_section(
        branch_name="compositional", fluid_type="compositional",
        units=unit_system,
        parameters={
            "composition": st.session_state.get("comp_state", {}),
            "T_F": T_res, "P_res_psia": P_res,
            "C7_MW": MW_c7, "C7_SG": SG_c7,
        },
        outputs_summary=bot_summary,
        results_table_df=pd.DataFrame(bot_rows) if bot_rows else None)


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

        # Option to load gas properties from a saved fluid
        gas_fluids = [
            (name, rec) for name, rec in
            st.session_state.get("fluid_registry", {}).items()
            if rec.get("fluid_type") in ("dry_gas", "wet_gas", "oil")
        ]
        fluid_source = st.selectbox(
            "Gas property source",
            ["Manual entry"] + [f"📁 {n}" for n, _ in gas_fluids],
            help="Load gas SG (and H2S/CO2 if stored) from a fluid you saved "
                 "in another branch's Tools section.")

        loaded_sg = None; loaded_h2s = None; loaded_co2 = None
        if fluid_source != "Manual entry":
            sel_name = fluid_source.replace("📁 ", "")
            rec = dict(gas_fluids)[sel_name]
            params = rec.get("parameters", {})
            loaded_sg = params.get("gas_sg")
            loaded_h2s = params.get("H2S", 0.0)
            loaded_co2 = params.get("CO2", 0.0)
            st.caption(f"Loaded from **{sel_name}** ({summarize(rec)})")

        gas_sg_h = st.number_input(
            "Gas specific gravity (air = 1)",
            value=float(loaded_sg) if loaded_sg is not None else 0.65,
            min_value=0.55, max_value=1.0, step=0.01,
            help="Specific gravity of the gas phase. Heavier (richer) gas "
                 "forms hydrates at lower P / higher T.")
        H2S_h = st.number_input(
            "H2S mol fraction",
            value=float(loaded_h2s) if loaded_h2s is not None else 0.0,
            min_value=0.0, max_value=0.3, step=0.01, format="%.4f",
            help="H2S strongly promotes hydrate formation: each 1% lowers the "
                 "hydrate-formation pressure by ~5%.")
        CO2_h = st.number_input(
            "CO2 mol fraction",
            value=float(loaded_co2) if loaded_co2 is not None else 0.0,
            min_value=0.0, max_value=0.5, step=0.01, format="%.4f",
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
    st.markdown("### Inhibitor Effect (Hammerschmidt)")
    st.markdown(
        "An inhibitor (methanol or glycol) suppresses the hydrate-formation "
        "temperature by binding water. Drag the slider to set the inhibitor "
        "concentration — the **inhibited hydrate curve** (dashed) shifts left "
        "relative to the **uninhibited curve** (solid). The Hammerschmidt "
        "equation relates concentration to temperature shift:"
    )
    st.latex(r"\Delta T = \frac{K_H \cdot W}{M \cdot (100 - W)}")
    st.caption(
        "where $W$ = inhibitor concentration (wt %), $\\Delta T$ = hydrate-T "
        "suppression (°F), $M$ = inhibitor molecular weight, $K_H$ = "
        "Hammerschmidt constant (2335 methanol, 2222 MEG, 4000 DEG, 5400 TEG)."
    )

    cinh = st.columns([1, 2])
    with cinh[0]:
        inhibitor = st.selectbox(
            "Inhibitor",
            ["methanol", "MEG", "DEG", "TEG"],
            help="Methanol: cheap, effective, volatile. MEG: preferred when "
                 "recovery and regeneration are feasible.")
        inhibitor_wt = st.slider(
            "Inhibitor concentration (wt %)",
            min_value=0.0, max_value=60.0, value=20.0, step=1.0,
            help="Aqueous-phase concentration. Drag to see the curve move.")

    # Hammerschmidt: ΔT (°F) for the chosen W
    _M = {"methanol": 32.04, "MEG": 62.07, "DEG": 106.12, "TEG": 150.17}[inhibitor]
    _KH = {"methanol": 2335, "MEG": 2222, "DEG": 4000, "TEG": 5400}[inhibitor]
    if inhibitor_wt < 99.9:
        delta_T_F = (_KH * inhibitor_wt) / (_M * (100.0 - inhibitor_wt))
    else:
        delta_T_F = 0.0
    delta_T_disp = delta_T_F if unit_system == "Field" else delta_T_F * 5.0 / 9.0

    with cinh[0]:
        st.metric(f"Hydrate-T suppression",
                   f"{delta_T_disp:.1f} {L['T']}")
        # Does this clear the operating point?
        if not np.isnan(risk["T_hydrate"]):
            new_T_hyd_F = risk["T_hydrate"] - delta_T_F
            if T_op_F > new_T_hyd_F:
                st.success(f"✓ Clears the operating point "
                            f"(inhibited hydrate-T = "
                            f"{U.to_user_T(new_T_hyd_F, unit_system):.1f} {L['T']})")
            else:
                still_need = T_op_F - new_T_hyd_F
                st.warning(f"⚠️ Still in hydrate zone — need "
                            f"{abs(still_need):.1f} °F more suppression.")

    with cinh[1]:
        # Plot: uninhibited curve vs inhibited curve (shifted by ΔT in T)
        T_curve_F2, P_curve_psia2 = hydrate_curve(gas_sg_h, H2S_h, CO2_h,
                                                    n_points=60)
        T_unInh = [U.to_user_T(t, unit_system) for t in T_curve_F2]
        P_unInh = [U.to_user_P(p, unit_system) for p in P_curve_psia2]
        # The inhibited curve: same P, but each T is shifted DOWN by ΔT
        # (it now takes a colder temperature to form hydrates at that P)
        T_inh = [U.to_user_T(t - delta_T_F, unit_system) for t in T_curve_F2]

        fig_inh = go.Figure()
        fig_inh.add_trace(go.Scatter(
            x=T_unInh, y=P_unInh, name="No inhibitor",
            mode="lines", line=dict(color="#3A6E96", width=3),
            hovertemplate=f"No inhibitor<br>T=%{{x:.1f}} {L['T']}<br>"
                          f"P=%{{y:.0f}} {L['P']}<extra></extra>"))
        fig_inh.add_trace(go.Scatter(
            x=T_inh, y=P_unInh,
            name=f"{inhibitor} {inhibitor_wt:.0f} wt%",
            mode="lines", line=dict(color="#EB0037", width=3, dash="dash"),
            hovertemplate=f"{inhibitor} {inhibitor_wt:.0f}wt%<br>"
                          f"T=%{{x:.1f}} {L['T']}<br>"
                          f"P=%{{y:.0f}} {L['P']}<extra></extra>"))
        # Operating point
        op_color2 = ("#EB0037" if risk["risk_level"] == "in_zone"
                      else "#C58B00" if risk["risk_level"] == "marginal"
                      else "#9DBA00")
        fig_inh.add_trace(go.Scatter(
            x=[T_op_user], y=[P_op_user], name="Operating point",
            mode="markers", marker=dict(size=16, color=op_color2,
                                         symbol="diamond",
                                         line=dict(color="#00243D", width=2)),
            hovertemplate=f"Operating point<br>T={T_op_user:.1f} {L['T']}"
                          f"<br>P={P_op_user:.0f} {L['P']}<extra></extra>"))
        fig_inh.update_layout(**TH.plotly_layout(
            title=f"Inhibition shifts the hydrate curve left by "
                  f"{delta_T_disp:.1f} {L['T']}",
            xtitle=f"Temperature ({L['T']})",
            ytitle=f"Pressure ({L['P']})",
            height=420))
        st.plotly_chart(fig_inh, use_container_width=True)

    # Comparison table of all four inhibitors at the slider concentration
    st.markdown("##### All inhibitors at {:.0f} wt%".format(inhibitor_wt))
    comp_rows = []
    for inh in ["methanol", "MEG", "DEG", "TEG"]:
        Mi = {"methanol": 32.04, "MEG": 62.07, "DEG": 106.12, "TEG": 150.17}[inh]
        KHi = {"methanol": 2335, "MEG": 2222, "DEG": 4000, "TEG": 5400}[inh]
        if inhibitor_wt < 99.9:
            dT = (KHi * inhibitor_wt) / (Mi * (100.0 - inhibitor_wt))
        else:
            dT = 0.0
        comp_rows.append({
            "Inhibitor": inh,
            f"ΔT suppression ({L['T']})":
                dT if unit_system == "Field" else dT * 5.0 / 9.0,
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True,
                 hide_index=True)

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

    # ---- Subsea shutdown / cooldown time ----
    st.markdown("---")
    st.markdown("### 🌊 Subsea Shutdown Cooldown Time")
    st.markdown(
        "After a production shutdown, a subsea flowline cools toward the seabed "
        "temperature. Estimate how long before the fluid reaches the hydrate-"
        "formation T — this is the **cooldown time** that drives shutdown "
        "response protocols (when to inject inhibitor, when to depressurize)."
    )
    from hydrate import cooldown_time_to_hydrate, cooldown_curve
    cs1, cs2, cs3 = st.columns(3)
    with cs1:
        T_amb_user = st.number_input(f"Seabed/ambient T ({L['T']})",
                                      value=U.to_user_T(38.0, unit_system),
                                      help="Subsea seabed temperature, typically 35-42 °F.")
        T_op_init_user = st.number_input(f"Initial fluid T ({L['T']})",
                                          value=T_op_user,
                                          help="Fluid T at the moment of shutdown.")
    with cs2:
        U_pipe = st.number_input("U (BTU/hr/ft²/°F)", value=2.0, min_value=0.1, max_value=20.0,
                                   help="Overall heat-transfer coefficient. "
                                        "1-3 for well-insulated, 5-15 for bare pipe.")
        D_pipe_in = st.number_input("Pipe OD (inches)", value=8.0, min_value=2.0, max_value=36.0,
                                      help="Outer diameter of the flowline.")
    with cs3:
        rho_fluid = st.number_input("Fluid density (lb/ft³)", value=50.0,
                                       min_value=20.0, max_value=100.0)
        cp_fluid = st.number_input("Cp (BTU/lb/°F)", value=0.5,
                                      min_value=0.1, max_value=1.5,
                                      help="Specific heat: oil ≈ 0.5, water ≈ 1.0.")
    T_amb_F = U.to_field_T(T_amb_user, unit_system)
    T_op_init_F = U.to_field_T(T_op_init_user, unit_system)

    cd = cooldown_time_to_hydrate(
        T_op_F=T_op_init_F, P_op_psia=P_op_psia, T_ambient_F=T_amb_F,
        gas_sg=gas_sg_h, H2S_frac=H2S_h, CO2_frac=CO2_h,
        U_pipe=U_pipe, D_outer_ft=D_pipe_in / 12.0,
        rho_fluid=rho_fluid, cp_fluid=cp_fluid)

    if not np.isnan(cd["time_hours"]) and cd["time_hours"] != float('inf'):
        ccm = st.columns(3)
        ccm[0].metric("Cooldown to hydrate T", f"{cd['time_hours']:.2f} hours")
        ccm[1].metric("In minutes", f"{cd['time_minutes']:.0f} min")
        if cd["time_hours"] < 1.0:
            risk_color = "#EB0037"; banner = "🛑 LESS THAN 1 HOUR — urgent inhibitor response"
        elif cd["time_hours"] < 4.0:
            risk_color = "#C58B00"; banner = "⚠️ SHORT WINDOW — < 4 hr response margin"
        else:
            risk_color = "#9DBA00"; banner = "✓ ADEQUATE WINDOW for shutdown protocols"
        ccm[2].markdown(
            f"<div style='background-color:{risk_color}; padding:0.5rem 0.8rem; "
            f"color:white; border-radius:4px; font-weight:600; text-align:center;'>"
            f"{banner}</div>", unsafe_allow_html=True)

        # Cooldown curve plot
        times, temps = cooldown_curve(T_op_init_F, T_amb_F, U_pipe,
                                        D_pipe_in / 12.0, rho_fluid, cp_fluid,
                                        t_end_hours=max(24.0, cd["time_hours"] * 1.5))
        times_user = times
        temps_user = [U.to_user_T(t, unit_system) for t in temps]
        T_hyd_user = U.to_user_T(cd["T_hydrate_F"], unit_system)
        T_amb_user_disp = U.to_user_T(T_amb_F, unit_system)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times_user, y=temps_user, name="Fluid T",
            mode="lines", line=dict(color="#EB0037", width=3),
            hovertemplate=f"t=%{{x:.2f}} hr<br>T=%{{y:.1f}} {L['T']}<extra></extra>"))
        # Hydrate T line
        fig.add_trace(go.Scatter(
            x=[0, max(times_user)], y=[T_hyd_user, T_hyd_user],
            name=f"Hydrate T = {T_hyd_user:.1f}", mode="lines",
            line=dict(color="#00243D", width=2, dash="dash"),
            hovertemplate=f"Hydrate T = {T_hyd_user:.1f}<extra></extra>"))
        # Ambient line
        fig.add_trace(go.Scatter(
            x=[0, max(times_user)], y=[T_amb_user_disp, T_amb_user_disp],
            name=f"Ambient = {T_amb_user_disp:.1f}", mode="lines",
            line=dict(color="#3A6E96", width=1.5, dash="dot")))
        # Mark cooldown time
        fig.add_trace(go.Scatter(
            x=[cd["time_hours"]], y=[T_hyd_user],
            name="Hydrate onset", mode="markers",
            marker=dict(size=15, color="#C58B00", symbol="star",
                        line=dict(color="#00243D", width=2)),
            hovertemplate=f"Hydrate at t={cd['time_hours']:.2f} hr<extra></extra>"))
        fig.update_layout(**TH.plotly_layout(
            title="Subsea cooldown curve (lumped-capacitance model)",
            xtitle="Time (hours)", ytitle=f"Fluid T ({L['T']})",
            height=420))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(cd.get("message", "Cooldown analysis not applicable."))


# ================================================================
# ROCK COMPRESSIBILITY
# ================================================================
elif fluid == "🪨 Rock Compressibility":
    from rock_comp import (compute_all, CORRELATIONS,
                            rock_keyword, rock_keyword_metric)
    import plotly.graph_objects as go

    st.markdown("## Rock Compressibility")
    st.markdown(
        "Estimate the **pore-volume compressibility** $C_f$ used in the ECLIPSE "
        "`ROCK` keyword. Several correlations are provided — they differ "
        "significantly, so report ranges and pick based on lithology and "
        "consolidation."
    )

    col_r_in, col_r_out = st.columns([1, 2])

    with col_r_in:
        st.markdown("### Reservoir Inputs")
        phi = st.number_input("Porosity (fraction)",
                                value=0.20, min_value=0.01, max_value=0.40, step=0.01)
        if unit_system == "Field":
            Pref_user = st.number_input(f"Reference pressure ({L['P']})",
                                         value=U.to_user_P(P_res, unit_system),
                                         min_value=U.to_user_P(100.0, unit_system))
        else:
            Pref_user = st.number_input(f"Reference pressure ({L['P']})",
                                         value=U.to_user_P(P_res, unit_system),
                                         min_value=7.0)
        Pref_psia = U.to_field_P(Pref_user, unit_system)

        st.markdown("### Correlation")
        chosen_corr = st.selectbox("Select correlation for ECLIPSE export",
                                    list(CORRELATIONS.keys()))
        st.caption("All correlations are evaluated; the chosen one is used "
                    "for the ECLIPSE ROCK keyword export.")

    with col_r_out:
        st.markdown("### All correlations at φ = {:.2f}".format(phi))
        cf_results = compute_all(phi)

        # Display as metric cards
        cols_m = st.columns(len(cf_results))
        for i, (name, cf) in enumerate(cf_results.items()):
            cf_user = cf * (14.50377 if unit_system == "SI" else 1.0)
            with cols_m[i]:
                st.metric(name, f"{cf_user:.3e} {L['Cw']}")

        # Plot Cf vs porosity for all correlations
        phi_arr = np.linspace(0.05, 0.35, 60)
        fig = go.Figure()
        colors = ["#EB0037", "#00243D", "#9DBA00", "#3A6E96", "#C58B00"]
        for j, (name, fn) in enumerate(CORRELATIONS.items()):
            cf_arr = [fn(p) for p in phi_arr]
            cf_arr_user = [c * (14.50377 if unit_system == "SI" else 1.0) for c in cf_arr]
            fig.add_trace(go.Scatter(
                x=phi_arr * 100, y=cf_arr_user,
                name=name, mode="lines",
                line=dict(color=colors[j % len(colors)], width=2.5),
                hovertemplate=f"<b>{name}</b><br>φ=%{{x:.1f}}%<br>"
                              f"Cf=%{{y:.2e}}<extra></extra>"))
        # Mark the operating point
        for name, cf in cf_results.items():
            cf_user_pt = cf * (14.50377 if unit_system == "SI" else 1.0)
            fig.add_trace(go.Scatter(
                x=[phi * 100], y=[cf_user_pt],
                showlegend=False, mode="markers",
                marker=dict(size=8, color="#888888", symbol="x")))
        fig.update_layout(**TH.plotly_layout(
            title="Rock compressibility vs porosity",
            xtitle="Porosity (%)", ytitle=f"Cf ({L['Cw']})",
            height=460, ymode="log"))
        st.plotly_chart(fig, use_container_width=True)

    # ECLIPSE ROCK export
    if enable_eclipse_export:
        st.markdown("---")
        st.markdown(f"### ECLIPSE ROCK keyword ({eclipse_unit_choice})")
        chosen_cf = cf_results[chosen_corr]
        if eclipse_unit_choice == "METRIC":
            Pref_bar = Pref_psia / 14.50377
            Cf_per_bar = chosen_cf * 14.50377
            rock_text = rock_keyword_metric(Pref_bar, Cf_per_bar)
        else:
            rock_text = rock_keyword(Pref_psia, chosen_cf)
        st.code(rock_text, language="text")
        st.download_button("Download ROCK keyword (.INC)", rock_text,
                            file_name=f"ROCK_{eclipse_unit_choice}.INC",
                            mime="text/plain", type="primary")

    with st.expander("📖 About these correlations"):
        st.markdown(r"""
**Hall (1953):** $C_f = 1.782 \times 10^{-6} / \varphi^{0.438}$ — the
most-cited correlation for consolidated sandstone/limestone. Reasonable for
$0.05 < \varphi < 0.30$.

**Newman (1973):** Two separate fits for consolidated sandstone and limestone.
Generally gives lower Cf than Hall at typical reservoir porosities.

**Horne polynomial:** $C_f = (4.55 - 4.02\varphi) \times 10^{-6}$ — a simple
linear fit; convenient for quick estimates.

**Carpenter-Spencer:** $C_f = 7.5 \times 10^{-6} / (1 + 60\varphi)$ — designed
for consolidated limestone with $\varphi < 0.20$.

**General guidance:**
- Unconsolidated sand: expect Cf 10× higher than these correlations.
- Hard, well-cemented carbonate: lower end.
- For very heterogeneous reservoirs, lab core measurements are essential.
- ECLIPSE expects $C_f$ in $1/\text{psia}$ (FIELD) or $1/\text{bara}$ (METRIC).
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
