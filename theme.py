"""
Equinor-themed colors and Plotly helpers for the PVT app.

Brand reference: Equinor 2018+ rebrand uses Torch Red (#EB0037) as primary,
Karry (#FFE7D6) and Pistachio (#9DBA00) as secondary, on white / dark-navy base.
"""

# -----------------------------
# Color palette
# -----------------------------
TORCH_RED   = "#EB0037"
KARRY       = "#FFE7D6"
PISTACHIO   = "#9DBA00"
DARK_NAVY   = "#00243D"
SLATE       = "#3A4A5C"
LIGHT_GRAY  = "#F4F4F4"
MID_GRAY    = "#B8B8B8"
WHITE       = "#FFFFFF"
INK         = "#1A1A1A"

# Chart sequence: ordered for distinguishability on white
CHART_COLORS = [
    TORCH_RED,     # primary series
    DARK_NAVY,     # secondary series
    PISTACHIO,     # accent / third
    "#FF8FA8",     # light red (red tint)
    "#3A6E96",     # mid blue
    "#C58B00",     # amber for fourth+
    "#5A7A00",     # dark pistachio
]

# Phase-specific colors (for phase envelope: bubble = red, dew = blue)
COLOR_BUBBLE   = TORCH_RED
COLOR_DEW      = DARK_NAVY
COLOR_CRITICAL = PISTACHIO
COLOR_OIL      = TORCH_RED
COLOR_GAS      = DARK_NAVY
COLOR_WATER    = "#3A6E96"

# -----------------------------
# Custom CSS for Streamlit
# -----------------------------
CUSTOM_CSS = """
<style>
/* Header band */
.equinor-header {
    background: linear-gradient(90deg, #00243D 0%, #1B3A5B 100%);
    padding: 1.2rem 1.5rem;
    border-radius: 4px;
    margin-bottom: 1rem;
    border-left: 6px solid #EB0037;
}
.equinor-header h1 {
    color: #FFFFFF !important;
    margin: 0 !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}
.equinor-header p {
    color: #FFE7D6 !important;
    margin: 0.3rem 0 0 0 !important;
    font-size: 0.9rem !important;
}

/* Section dividers and headers */
h2, h3 {
    color: #00243D !important;
    font-weight: 600 !important;
}

/* Metrics styling */
[data-testid="stMetric"] {
    background-color: #F4F4F4;
    padding: 0.6rem;
    border-radius: 4px;
    border-left: 3px solid #EB0037;
}
[data-testid="stMetricValue"] {
    color: #00243D !important;
    font-weight: 600 !important;
}

/* Primary buttons get Equinor red */
.stButton > button[kind="primary"] {
    background-color: #EB0037 !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #C50030 !important;
}

/* Sidebar accent */
section[data-testid="stSidebar"] {
    background-color: #F4F4F4;
    border-right: 1px solid #DDD;
}

/* Tables - subtle borders */
[data-testid="stDataFrame"] {
    border: 1px solid #DDD;
    border-radius: 4px;
}

/* Success / info / warning boxes */
.stSuccess { border-left: 4px solid #9DBA00; }
.stInfo    { border-left: 4px solid #3A6E96; }
.stWarning { border-left: 4px solid #C58B00; }
.stError   { border-left: 4px solid #EB0037; }

/* Code blocks - cleaner */
.stCodeBlock pre {
    background-color: #F8F8F8 !important;
    border: 1px solid #E0E0E0 !important;
}
</style>
"""


# -----------------------------
# Plotly helpers
# -----------------------------
def plotly_layout(title="", xtitle="", ytitle="", height=320,
                  showlegend=True, ymode="linear"):
    """Common Equinor-styled Plotly layout dict."""
    return dict(
        title=dict(text=title, font=dict(color=DARK_NAVY, size=14, family="Inter, sans-serif")),
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(color=INK, family="Inter, sans-serif", size=11),
        height=height,
        margin=dict(l=55, r=20, t=40 if title else 15, b=45),
        xaxis=dict(
            title=dict(text=xtitle, font=dict(size=11, color=SLATE)),
            gridcolor="#EAEAEA",
            zerolinecolor="#DDDDDD",
            showline=True, linecolor=MID_GRAY, linewidth=1,
            ticks="outside", tickcolor=MID_GRAY,
        ),
        yaxis=dict(
            title=dict(text=ytitle, font=dict(size=11, color=SLATE)),
            gridcolor="#EAEAEA",
            zerolinecolor="#DDDDDD",
            showline=True, linecolor=MID_GRAY, linewidth=1,
            ticks="outside", tickcolor=MID_GRAY,
            type=ymode,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=MID_GRAY, borderwidth=1,
            font=dict(size=10),
        ),
        hoverlabel=dict(bgcolor=DARK_NAVY, font_color=WHITE,
                        font_family="Inter, sans-serif"),
        showlegend=showlegend,
    )


def line_trace(x, y, name, color_idx=0, dash=None, width=2.5, mode="lines"):
    """Build a Plotly line trace using the Equinor palette."""
    import plotly.graph_objects as go
    color = CHART_COLORS[color_idx % len(CHART_COLORS)]
    return go.Scatter(
        x=list(x), y=list(y), name=name,
        mode=mode,
        line=dict(color=color, width=width, dash=dash),
        marker=dict(size=6, color=color),
        hovertemplate=f"<b>{name}</b><br>%{{x:.2f}} → %{{y:.4g}}<extra></extra>",
    )


def header_banner(title, subtitle=""):
    return (f'<div class="equinor-header">'
            f'<h1>{title}</h1>'
            f'<p>{subtitle}</p>'
            f'</div>')
