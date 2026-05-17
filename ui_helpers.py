"""
PVT Studio — Shared UI Helper Functions
========================================

Pure, self-contained Streamlit rendering helpers extracted from the main
app for maintainability. Every function here depends only on its
arguments and the module imports below — none read application global
state — so they can be unit-reasoned about and reused across branches.

Contents:
  - line_chart_plotly      : themed line chart with optional tuned overlay
  - styled_dataframe       : safe dataframe display with column formatting
  - render_eclipse_qc      : ECLIPSE monotonicity QC panel + export plot
  - render_depth_profile   : Rs/Rv-vs-depth builder (RSVD/RVVD)
  - render_property_plots  : multi-select property plotter
  - render_input_correlation : Monte Carlo input correlation heatmap
  - render_tornado_chart   : tornado chart (add_shape rectangles)
  - fluid_fingerprint      : tuning-staleness fingerprint
  - tuning_is_stale        : tuning-staleness check
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import theme as TH
import eclipse_qc as EQC


def line_chart_plotly(df, x_col, y_cols, title="", height=320, ymode="linear",
                       overlay_df=None, overlay_label="Tuned",
                       base_label="Untuned"):
    """Plotly line chart with Equinor styling.

    If overlay_df is provided, its series are drawn as dashed lines on the
    same axes so the user can compare (e.g. tuned vs untuned fluid).
    """
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    fig = go.Figure()
    for i, c in enumerate(y_cols):
        name = c if overlay_df is None else f"{base_label}: {c}"
        fig.add_trace(TH.line_trace(df[x_col].values, df[c].values,
                                     name, color_idx=i))
    if overlay_df is not None:
        for i, c in enumerate(y_cols):
            if c in overlay_df.columns:
                fig.add_trace(go.Scatter(
                    x=overlay_df[x_col].values, y=overlay_df[c].values,
                    mode="lines+markers", name=f"{overlay_label}: {c}",
                    line=dict(color="#EB0037", width=2.5, dash="dash"),
                    marker=dict(size=5)))
    show_leg = (len(y_cols) > 1) or (overlay_df is not None)
    fig.update_layout(**TH.plotly_layout(
        title=title, xtitle=x_col, ytitle=(y_cols[0] if len(y_cols) == 1 else "Value"),
        height=height, ymode=ymode, showlegend=show_leg))
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


def render_eclipse_qc(df_field, kind, label="PVT table", pb=None):
    """Render a monotonicity QC panel + a plot of the export table.

    df_field : the field-unit property DataFrame about to be exported.
    kind     : 'pvto', 'pvdg', or 'pvtg' — selects the QC ruleset.
    pb       : bubble point (for PVTO, to split saturated/under-saturated).
    Returns the QC result dict so callers can block export on failure.
    """
    if kind == "pvto":
        qc = EQC.qc_pvto_table(df_field, pb=pb)
    elif kind == "pvdg":
        qc = EQC.qc_pvdg_table(df_field)
    elif kind == "pvtg":
        qc = EQC.qc_pvtg_table(df_field)
    else:
        qc = EQC.qc_pvto_table(df_field, pb=pb)
    if qc["ok"]:
        st.success(f"✓ {label} is monotonic — ECLIPSE should accept it.")
    else:
        st.error(f"⛔ {label} has monotonicity problems that will make "
                 f"ECLIPSE reject the deck:")
        for p in qc["problems"]:
            st.markdown(f"- {p}")
        st.caption("Fix: widen the pressure range, reduce the number of "
                    "points, or check the input parameters — non-monotonic "
                    "rows usually come from extrapolation artefacts.")

    # Plot the table columns the user will export
    with st.expander(f"📈 Plot the {label} that will be exported"):
        num_cols = [c for c in df_field.columns
                     if pd.api.types.is_numeric_dtype(df_field[c])]
        x_default = num_cols[0] if num_cols else None
        if x_default and len(num_cols) > 1:
            ycols = st.multiselect(
                "Columns to plot (vs pressure)",
                [c for c in num_cols if c != x_default],
                default=[c for c in num_cols if c != x_default][:1],
                key=f"eqc_plot_{kind}_{label}")
            if ycols:
                line_chart_plotly(df_field, x_default, ycols,
                                   title=f"{label} — export preview")
    return qc


def render_depth_profile(fluid_kind, ref_value, ref_depth_default,
                           value_label, value_unit, key_prefix):
    """Render an Rs-vs-depth (oil) or Rv-vs-depth (gas) profile builder.

    Produces an RSVD or RVVD keyword block from a linear compositional
    gradient. Returns the keyword text (or None if not generated).
    """
    st.markdown(f"##### {value_label} vs depth "
                 f"({'RSVD' if fluid_kind == 'oil' else 'RVVD'})")
    st.caption(
        "Build a compositional-gradient keyword: the dissolved-gas "
        "(or vaporized-oil) content varies linearly with depth. ECLIPSE "
        "uses this to initialize a reservoir that is not uniformly mixed.")
    dc = st.columns(4)
    with dc[0]:
        d_ref = st.number_input("Reference depth", value=float(ref_depth_default),
                                 key=f"{key_prefix}_dref")
    with dc[1]:
        d_top = st.number_input("Top depth", value=float(ref_depth_default) - 200.0,
                                 key=f"{key_prefix}_dtop")
    with dc[2]:
        d_bot = st.number_input("Bottom depth", value=float(ref_depth_default) + 200.0,
                                 key=f"{key_prefix}_dbot")
    with dc[3]:
        grad = st.number_input(f"Gradient ({value_unit}/depth)",
                                value=-0.30, format="%.4f",
                                key=f"{key_prefix}_grad",
                                help="Negative means the value decreases "
                                     "downward (typical for Rs).")
    n_lvl = st.slider("Depth levels", 3, 20, 6, key=f"{key_prefix}_nlvl")
    if d_bot <= d_top:
        st.warning("Bottom depth must be greater than top depth.")
        return None
    depths = list(np.linspace(d_top, d_bot, n_lvl))
    values = EQC.linear_depth_profile(ref_value, grad, d_ref, depths)

    prof_df = pd.DataFrame({"Depth": depths, value_label: values})
    styled_dataframe(prof_df, height=240)
    line_chart_plotly(prof_df, "Depth", value_label,
                       title=f"{value_label} vs depth")

    if fluid_kind == "oil":
        kw = EQC.build_rsvd(depths, values,
                             comment=f"{value_label} vs depth (linear gradient)")
    else:
        kw = EQC.build_rvvd(depths, values,
                             comment=f"{value_label} vs depth (linear gradient)")
    st.code(kw, language="text")
    st.download_button(
        f"Download {'RSVD' if fluid_kind == 'oil' else 'RVVD'} keyword",
        kw, file_name=f"{'RSVD' if fluid_kind == 'oil' else 'RVVD'}.INC",
        mime="text/plain", key=f"{key_prefix}_dl")
    return kw


def render_property_plots(df, x_col, prop_cols, key_prefix,
                            overlay_df=None, overlay_label="Tuned",
                            default_props=None):
    """Reusable property-plot panel with a multi-select dropdown.

    df         : display-unit property table.
    x_col      : the x-axis column (pressure).
    prop_cols  : list of plottable property column names.
    overlay_df : optional second table (e.g. tuned fluid) drawn dashed.

    The user picks which properties to show; each is rendered as its own
    chart. A 'combine' toggle overlays them all on one normalized chart so
    trends can be compared at a glance — mirroring the rock-compressibility
    plot's multi-line style.
    """
    avail = [c for c in prop_cols if c in df.columns]
    if not avail:
        return
    if default_props is None:
        default_props = avail
    sel = st.multiselect(
        "Properties to plot", avail,
        default=[p for p in default_props if p in avail],
        key=f"{key_prefix}_propsel")
    if not sel:
        st.caption("Select one or more properties above to plot them.")
        return
    combine = False
    if len(sel) > 1:
        combine = st.checkbox(
            "Overlay all on one chart (normalized 0–1)",
            value=False, key=f"{key_prefix}_combine",
            help="Plots every selected property on a single chart, each "
                 "scaled to its own 0–1 range, so you can compare the "
                 "shape of the trends regardless of units.")
    if combine:
        fig = go.Figure()
        for i, c in enumerate(sel):
            y = df[c].astype(float).values
            rng = (y.max() - y.min()) or 1.0
            y_norm = (y - y.min()) / rng
            fig.add_trace(go.Scatter(
                x=df[x_col].values, y=y_norm, mode="lines+markers",
                name=c, line=dict(width=2.5)))
            if overlay_df is not None and c in overlay_df.columns:
                yo = overlay_df[c].astype(float).values
                yo_norm = (yo - y.min()) / rng   # same scale as base
                fig.add_trace(go.Scatter(
                    x=overlay_df[x_col].values, y=yo_norm,
                    mode="lines", name=f"{overlay_label}: {c}",
                    line=dict(width=2, dash="dash")))
        fig.update_layout(**TH.plotly_layout(
            title="Properties (each normalized 0–1)",
            xtitle=x_col, ytitle="Normalized value",
            height=380, showlegend=True))
        st.plotly_chart(fig, use_container_width=True)
    else:
        cols = st.columns(min(3, len(sel)))
        for i, c in enumerate(sel):
            with cols[i % len(cols)]:
                line_chart_plotly(df, x_col, c,
                                   title=c.split("(")[0].strip(),
                                   overlay_df=overlay_df,
                                   overlay_label=overlay_label)


def render_input_correlation(samples_dict, key_prefix, title="Input correlation"):
    """Render a correlation-matrix heatmap between sampled input arrays.

    samples_dict : {parameter_name: 1-D array of sampled values}.
    Used after a Monte Carlo run to show how the sampled inputs co-vary
    (useful to confirm they were sampled independently, or to visualize an
    imposed correlation). Pearson correlation is used.
    """
    names = [k for k, v in samples_dict.items()
             if v is not None and len(np.asarray(v)) > 1]
    if len(names) < 2:
        return
    mat = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            va = np.asarray(samples_dict[a], dtype=float)
            vb = np.asarray(samples_dict[b], dtype=float)
            m = np.isfinite(va) & np.isfinite(vb)
            if m.sum() > 1 and np.std(va[m]) > 0 and np.std(vb[m]) > 0:
                mat[i, j] = np.corrcoef(va[m], vb[m])[0, 1]
    fig = go.Figure(go.Heatmap(
        z=mat, x=names, y=names, zmin=-1, zmax=1,
        colorscale=[[0, "#00243D"], [0.5, "#FFFFFF"], [1, "#EB0037"]],
        text=[[f"{mat[i,j]:.2f}" if np.isfinite(mat[i,j]) else ""
                for j in range(len(names))] for i in range(len(names))],
        texttemplate="%{text}", colorbar=dict(title="r")))
    fig.update_layout(**TH.plotly_layout(
        title=title, xtitle="", ytitle="", height=340,
        showlegend=False))
    st.plotly_chart(fig, use_container_width=True)


def render_tornado_chart(rows, base_value, output_name, unit_label,
                          unit_converter=None, show_table=True):
    """Reusable tornado chart used by every branch.

    rows         : list of (param_name, low, high, range) tuples in field units.
    base_value   : the base output value in field units.
    unit_converter : optional fn(value)->display value. Defaults to identity.
    The chart is drawn with add_shape rectangles at explicit numeric
    y-positions — shapes honour absolute coordinates, so the bars cannot
    collapse the way data-driven Plotly bars can.
    """
    if unit_converter is None:
        unit_converter = lambda v: v
    if not rows:
        st.info(f"No tornado data for {output_name} — check that at least "
                 f"one σ value is greater than zero.")
        return
    if base_value is None or (isinstance(base_value, float)
                               and np.isnan(base_value)):
        st.info(f"Could not compute a base value for {output_name}.")
        return

    base_disp = unit_converter(base_value)
    sorted_rows = sorted(rows, key=lambda r: r[3])  # smallest impact first
    params  = [r[0] for r in sorted_rows]
    lo_disp = [unit_converter(r[1]) for r in sorted_rows]
    hi_disp = [unit_converter(r[2]) for r in sorted_rows]

    if show_table:
        styled_dataframe(pd.DataFrame({
            "Parameter": params, "Low": lo_disp, "High": hi_disp,
            "Range": [abs(h - l) for h, l in zip(hi_disp, lo_disp)],
        }), height=180)

    fig = go.Figure()
    n = len(params)
    bar_h = 0.36
    for i in range(n):
        x0_lo, x1_lo = sorted([base_disp, lo_disp[i]])
        fig.add_shape(type="rect", y0=i - bar_h, y1=i + bar_h,
                       x0=x0_lo, x1=x1_lo, fillcolor=TH.DARK_NAVY,
                       line=dict(width=0), layer="above")
        x0_hi, x1_hi = sorted([base_disp, hi_disp[i]])
        fig.add_shape(type="rect", y0=i - bar_h, y1=i + bar_h,
                       x0=x0_hi, x1=x1_hi, fillcolor=TH.TORCH_RED,
                       line=dict(width=0), layer="above")
    fig.add_trace(go.Scatter(
        x=lo_disp, y=list(range(n)), mode="markers",
        marker=dict(size=1, color=TH.DARK_NAVY), name="−1σ",
        hovertemplate="%{text}: low=%{x:.4g}<extra></extra>", text=params))
    fig.add_trace(go.Scatter(
        x=hi_disp, y=list(range(n)), mode="markers",
        marker=dict(size=1, color=TH.TORCH_RED), name="+1σ",
        hovertemplate="%{text}: high=%{x:.4g}<extra></extra>", text=params))
    fig.add_vline(x=base_disp, line_dash="dash", line_color="black",
                   annotation_text=f"base = {base_disp:.4g}",
                   annotation_position="top")
    all_vals = lo_disp + hi_disp + [base_disp]
    span = max(all_vals) - min(all_vals)
    pad = span * 0.15 if span > 1e-9 else max(abs(base_disp) * 0.1, 1.0)
    fig.update_layout(**TH.plotly_layout(
        title=f"Tornado — {output_name} sensitivity (±1σ)",
        xtitle=f"{output_name} ({unit_label})", ytitle="Parameter",
        height=320, showlegend=True))
    fig.update_xaxes(range=[min(all_vals) - pad, max(all_vals) + pad])
    fig.update_yaxes(tickmode="array", tickvals=list(range(n)),
                      ticktext=params, range=[-0.6, n - 0.4])
    st.plotly_chart(fig, use_container_width=True)


def fluid_fingerprint(**params):
    """Build a stable fingerprint of the fluid parameters a tuning run was
    performed against. Used to detect when a stored tuning result no longer
    matches the current inputs (the user changed an input after tuning)."""
    return tuple(sorted((k, round(float(v), 6))
                         for k, v in params.items() if v is not None))


def tuning_is_stale(tune_result, current_fp):
    """True when a stored tuning result was tuned against different inputs
    than are currently entered."""
    if not tune_result:
        return False
    stored = tune_result.get("fluid_fp")
    if stored is None:
        return False
    return tuple(stored) != tuple(current_fp)
