"""
PVT Studio — Example Fluid Presets
===================================

A new user faces a wall of empty/default number inputs and no sense of
what a realistic case looks like. These presets are well-known, published
fluid types: loading one fills the branch with sensible values so the user
sees a working result immediately and can learn by modifying it.

Each preset is a plain dict of input values in FIELD units. The branches
apply a preset by copying its values into st.session_state under the
input widget keys, then rerunning.

Values are representative literature fluids, not any specific reservoir.
"""

# ----------------------------------------------------------------------
# Oil presets — black-oil branch
# ----------------------------------------------------------------------
OIL_PRESETS = {
    "Light North Sea oil": {
        "api": 38.0, "gas_sg": 0.78, "Rsi": 750.0, "T_F": 220.0,
        "rs_corr": "Glaso", "bo_corr": "Glaso", "mu_corr": "Beggs-Robinson",
        "_note": "A typical light North Sea crude — Glaso correlations "
                 "were fit to exactly this kind of fluid.",
    },
    "Medium black oil": {
        "api": 32.0, "gas_sg": 0.72, "Rsi": 500.0, "T_F": 180.0,
        "rs_corr": "Standing", "bo_corr": "Standing",
        "mu_corr": "Beggs-Robinson",
        "_note": "A conventional medium-gravity black oil — squarely "
                 "inside the Standing correlation's data envelope.",
    },
    "Heavy oil": {
        "api": 19.0, "gas_sg": 0.68, "Rsi": 180.0, "T_F": 140.0,
        "rs_corr": "Vasquez-Beggs", "bo_corr": "Vasquez-Beggs",
        "mu_corr": "Beggs-Robinson",
        "_note": "A low-GOR heavy oil. Vasquez-Beggs extends to lower "
                 "API than Standing or Glaso.",
    },
    "Volatile oil": {
        "api": 44.0, "gas_sg": 0.85, "Rsi": 1450.0, "T_F": 250.0,
        "rs_corr": "Standing", "bo_corr": "Standing",
        "mu_corr": "Beggs-Robinson",
        "_note": "A high-GOR volatile oil near the upper edge of "
                 "black-oil applicability — consider the EOS branch too.",
    },
}

# ----------------------------------------------------------------------
# Dry gas presets
# ----------------------------------------------------------------------
DRY_GAS_PRESETS = {
    "Lean sweet gas": {
        "gas_sg": 0.60, "T_F": 200.0, "N2": 0.01, "CO2": 0.01, "H2S": 0.0,
        "z_corr": "Dranchuk-Abou-Kassem", "mu_corr": "Lee-Gonzalez-Eakin",
        "_note": "A clean, lean dry gas — close to pure methane.",
    },
    "Typical natural gas": {
        "gas_sg": 0.70, "T_F": 220.0, "N2": 0.02, "CO2": 0.02, "H2S": 0.0,
        "z_corr": "Hall-Yarborough", "mu_corr": "Lee-Gonzalez-Eakin",
        "_note": "A representative pipeline-quality natural gas.",
    },
    "Sour gas": {
        "gas_sg": 0.80, "T_F": 240.0, "N2": 0.03, "CO2": 0.08, "H2S": 0.05,
        "z_corr": "Dranchuk-Abou-Kassem", "mu_corr": "Carr-Kobayashi-Burrows",
        "_note": "A sour gas — the Wichert-Aziz correction is applied for "
                 "the H2S and CO2 content.",
    },
    "High-CO2 gas": {
        "gas_sg": 0.78, "T_F": 210.0, "N2": 0.02, "CO2": 0.20, "H2S": 0.0,
        "z_corr": "Dranchuk-Abou-Kassem", "mu_corr": "Lee-Gonzalez-Eakin",
        "_note": "A CO2-rich gas — note the Z-factor accuracy warning for "
                 "high inert content.",
    },
}

# ----------------------------------------------------------------------
# Wet gas / condensate presets
# ----------------------------------------------------------------------
WET_GAS_PRESETS = {
    "Lean gas condensate": {
        "gas_sg": 0.70, "api_cond": 58.0, "cgr": 40.0, "T_F": 240.0,
        "Pdew": 4200.0, "z_corr": "Dranchuk-Abou-Kassem",
        "mu_corr": "Lee-Gonzalez-Eakin", "rv_corr": "Linear-Pdew",
        "_note": "A lean gas condensate — low condensate yield, high "
                 "dew point.",
    },
    "Rich gas condensate": {
        "gas_sg": 0.78, "api_cond": 52.0, "cgr": 150.0, "T_F": 250.0,
        "Pdew": 5200.0, "z_corr": "Dranchuk-Abou-Kassem",
        "mu_corr": "Lee-Gonzalez-Eakin", "rv_corr": "Linear-Pdew",
        "_note": "A rich gas condensate — high condensate yield, "
                 "significant liquid dropout below the dew point.",
    },
    "Near-critical condensate": {
        "gas_sg": 0.82, "api_cond": 48.0, "cgr": 250.0, "T_F": 245.0,
        "Pdew": 6000.0, "z_corr": "Dranchuk-Abou-Kassem",
        "mu_corr": "Lee-Gonzalez-Eakin", "rv_corr": "Linear-Pdew",
        "_note": "A near-critical fluid — behaviour is sensitive; the "
                 "compositional EOS branch is recommended for these.",
    },
}

# ----------------------------------------------------------------------
# Water / brine presets
# ----------------------------------------------------------------------
WATER_PRESETS = {
    "Low-salinity brine": {
        "salinity_ppm": 10000.0, "T_F": 180.0, "corr": "McCain",
        "_note": "A low-salinity formation water.",
    },
    "Typical formation brine": {
        "salinity_ppm": 35000.0, "T_F": 210.0, "corr": "McCain",
        "_note": "A seawater-salinity formation brine.",
    },
    "High-salinity brine": {
        "salinity_ppm": 150000.0, "T_F": 240.0, "corr": "Spivey",
        "_note": "A concentrated brine — the Spivey-McCain correlation "
                 "handles high salinity well.",
    },
}

# ----------------------------------------------------------------------
# Compositional (EOS) presets — composition dicts (mole fractions)
# ----------------------------------------------------------------------
COMPOSITIONAL_PRESETS = {
    "Black oil (compositional)": {
        "composition": {
            "N2": 0.002, "CO2": 0.009, "H2S": 0.000,
            "C1": 0.365, "C2": 0.097, "C3": 0.070,
            "iC4": 0.014, "nC4": 0.039, "iC5": 0.014, "nC5": 0.014,
            "C6": 0.043, "C7+": 0.333,
        },
        "MW_c7": 218.0, "SG_c7": 0.852, "T_F": 200.0,
        "fluid_kind": "Oil (bubble point)",
        "_note": "A conventional black oil — moderate C1, heavy C7+ "
                 "fraction.",
    },
    "Gas condensate (compositional)": {
        "composition": {
            "N2": 0.004, "CO2": 0.020, "H2S": 0.000,
            "C1": 0.730, "C2": 0.082, "C3": 0.040,
            "iC4": 0.008, "nC4": 0.016, "iC5": 0.006, "nC5": 0.006,
            "C6": 0.010, "C7+": 0.078,
        },
        "MW_c7": 145.0, "SG_c7": 0.790, "T_F": 250.0,
        "fluid_kind": "Gas / Condensate (dew point)",
        "_note": "A gas condensate — methane-rich with a light C7+ "
                 "fraction.",
    },
    "Volatile oil (compositional)": {
        "composition": {
            "N2": 0.003, "CO2": 0.015, "H2S": 0.000,
            "C1": 0.557, "C2": 0.114, "C3": 0.087,
            "iC4": 0.013, "nC4": 0.034, "iC5": 0.013, "nC5": 0.016,
            "C6": 0.022, "C7+": 0.126,
        },
        "MW_c7": 175.0, "SG_c7": 0.815, "T_F": 235.0,
        "fluid_kind": "Oil (bubble point)",
        "_note": "A volatile oil — high C1 with a still-significant C7+ "
                 "fraction; sits between black oil and condensate.",
    },
}


def get_presets(branch):
    """Return the preset dict for a branch key:
    'oil', 'dry_gas', 'wet_gas', 'water', 'compositional'."""
    return {
        "oil":           OIL_PRESETS,
        "dry_gas":       DRY_GAS_PRESETS,
        "wet_gas":       WET_GAS_PRESETS,
        "water":         WATER_PRESETS,
        "compositional": COMPOSITIONAL_PRESETS,
    }.get(branch, {})
