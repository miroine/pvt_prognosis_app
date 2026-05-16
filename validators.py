"""
PVT Studio — Input Validation & Correlation Validity Ranges
============================================================

Two jobs:

1. Hard guards — catch physically impossible inputs (negative pressure,
   zero/negative GOR, empty composition) before they reach a correlation
   and produce silently-wrong numbers (a negative bubble point, a NaN
   Z-factor). These return a blocking error.

2. Soft validity warnings — every empirical correlation was fit to a
   finite database. Using it outside that envelope is extrapolation, not
   a bug, but the user should be told. These return a non-blocking
   advisory.

The branches call `check_*` helpers and render the messages with
`render_messages()`.

Published validity envelopes are taken from the original correlation
papers and from McCain, *The Properties of Petroleum Fluids* (1990).
"""

import streamlit as st


# ----------------------------------------------------------------------
# Message container
# ----------------------------------------------------------------------
class ValidationResult:
    """Collects errors (blocking) and warnings (advisory) from checks."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    @property
    def ok(self):
        """True when there are no blocking errors."""
        return len(self.errors) == 0

    def merge(self, other):
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        return self


def render_messages(result, stop_on_error=False):
    """Render a ValidationResult in the Streamlit UI.

    If stop_on_error is True and there are blocking errors, the script
    halts via st.stop() so downstream code never runs on bad inputs.
    """
    for msg in result.errors:
        st.error(f"⛔ {msg}")
    for msg in result.warnings:
        st.warning(f"⚠️ {msg}")
    if stop_on_error and not result.ok:
        st.stop()


# ----------------------------------------------------------------------
# Correlation validity envelopes (published ranges)
# ----------------------------------------------------------------------
# Each entry: parameter -> (low, high, unit, source)
OIL_RANGES = {
    "Standing": {
        "api":    (16.5, 63.8, "°API", "Standing 1947"),
        "T":      (100.0, 258.0, "°F",  "Standing 1947"),
        "gas_sg": (0.59, 0.95, "",      "Standing 1947"),
        "Rsi":    (20.0, 1425.0, "scf/STB", "Standing 1947"),
    },
    "Vasquez-Beggs": {
        "api":    (15.3, 59.5, "°API", "Vasquez & Beggs 1980"),
        "T":      (75.0, 294.0, "°F",  "Vasquez & Beggs 1980"),
        "gas_sg": (0.51, 1.35, "",      "Vasquez & Beggs 1980"),
        "Rsi":    (0.0, 2199.0, "scf/STB", "Vasquez & Beggs 1980"),
    },
    "Glaso": {
        "api":    (22.3, 48.1, "°API", "Glaso 1980"),
        "T":      (80.0, 280.0, "°F",  "Glaso 1980"),
        "gas_sg": (0.65, 1.28, "",      "Glaso 1980"),
        "Rsi":    (90.0, 2637.0, "scf/STB", "Glaso 1980"),
    },
    "Lasater": {
        "api":    (17.9, 51.1, "°API", "Lasater 1958"),
        "T":      (82.0, 272.0, "°F",  "Lasater 1958"),
        "gas_sg": (0.57, 1.22, "",      "Lasater 1958"),
        "Rsi":    (3.0, 2905.0, "scf/STB", "Lasater 1958"),
    },
}

# Gas Z-factor correlations work on pseudo-reduced coordinates
GAS_Z_RANGES = {
    "Hall-Yarborough": {
        "Tpr": (1.0, 3.0, "", "Hall & Yarborough 1973"),
        "Ppr": (0.2, 30.0, "", "Hall & Yarborough 1973"),
    },
    "Dranchuk-Abou-Kassem": {
        "Tpr": (1.0, 3.0, "", "Dranchuk & Abou-Kassem 1975"),
        "Ppr": (0.2, 30.0, "", "Dranchuk & Abou-Kassem 1975"),
    },
}


# ----------------------------------------------------------------------
# Hard guards — physical impossibilities
# ----------------------------------------------------------------------
def check_pressure_range(p_min, p_max, label="pressure range"):
    """Validate a pressure range used to build a property table."""
    r = ValidationResult()
    if p_min is None or p_max is None:
        r.error(f"The {label} is not set.")
        return r
    if p_min < 0 or p_max < 0:
        r.error(f"The {label} cannot be negative "
                 f"(got {p_min:g} to {p_max:g}).")
    if p_max <= p_min:
        r.error(f"The {label} maximum must be greater than the minimum "
                 f"(got {p_min:g} to {p_max:g}).")
    if p_min < 14.7 and p_min >= 0:
        r.warn("Minimum pressure is below atmospheric (14.7 psia). "
               "Property correlations are not reliable below ~14.7 psia; "
               "the table will start there.")
    return r


def check_temperature(T_F, label="reservoir temperature"):
    """Validate a temperature value (expects °F)."""
    r = ValidationResult()
    if T_F is None:
        r.error(f"The {label} is not set.")
        return r
    # Absolute zero is -459.67 F; reservoirs are far above that.
    if T_F <= -200.0:
        r.error(f"The {label} of {T_F:g} °F is physically implausible.")
    elif T_F < 32.0:
        r.warn(f"The {label} of {T_F:g} °F is below freezing — unusual "
               "for a reservoir. Check the unit setting.")
    elif T_F > 400.0:
        r.warn(f"The {label} of {T_F:g} °F is very high. Most PVT "
               "correlations are calibrated below ~300 °F.")
    return r


def check_positive(value, name, allow_zero=False):
    """Generic guard: a quantity that must be positive (or non-negative)."""
    r = ValidationResult()
    if value is None:
        r.error(f"{name} is not set.")
        return r
    if allow_zero and value < 0:
        r.error(f"{name} cannot be negative (got {value:g}).")
    elif not allow_zero and value <= 0:
        r.error(f"{name} must be greater than zero (got {value:g}).")
    return r


# ----------------------------------------------------------------------
# Oil branch validation
# ----------------------------------------------------------------------
def check_oil_inputs(api, gas_sg, T_F, Rsi, rs_corr, p_min, p_max):
    """Full validation for the black-oil branch inputs."""
    r = ValidationResult()
    r.merge(check_positive(Rsi, "Solution GOR (Rsi)", allow_zero=False))
    r.merge(check_temperature(T_F))
    r.merge(check_pressure_range(p_min, p_max))

    if api is not None and (api <= 0 or api > 100):
        r.error(f"Oil API gravity of {api:g} is outside any physical "
                "range (expected ~5–60 °API).")
    if gas_sg is not None and gas_sg <= 0:
        r.error(f"Gas specific gravity must be positive (got {gas_sg:g}).")

    # Soft validity-envelope check against the chosen correlation
    rng = OIL_RANGES.get(rs_corr)
    if rng and r.ok:
        _envelope_warn(r, rng, "api", api, rs_corr)
        _envelope_warn(r, rng, "gas_sg", gas_sg, rs_corr)
        _envelope_warn(r, rng, "T", T_F, rs_corr)
        _envelope_warn(r, rng, "Rsi", Rsi, rs_corr)
    return r


# ----------------------------------------------------------------------
# Gas branch validation
# ----------------------------------------------------------------------
def check_gas_inputs(gas_sg, T_F, p_min, p_max, N2=0.0, CO2=0.0, H2S=0.0):
    """Full validation for the dry-gas branch inputs."""
    r = ValidationResult()
    r.merge(check_temperature(T_F))
    r.merge(check_pressure_range(p_min, p_max))

    if gas_sg is not None and gas_sg <= 0:
        r.error(f"Gas specific gravity must be positive (got {gas_sg:g}).")
    elif gas_sg is not None and (gas_sg < 0.55 or gas_sg > 1.5):
        r.warn(f"Gas specific gravity of {gas_sg:g} is unusual "
               "(typical natural gas is 0.55–1.0). Check the value.")

    inert_total = (N2 or 0) + (CO2 or 0) + (H2S or 0)
    if inert_total > 1.0:
        r.error(f"Non-hydrocarbon mole fractions sum to {inert_total:.2f} "
                "— they cannot exceed 1.0.")
    elif inert_total > 0.5:
        r.warn(f"Non-hydrocarbon content is high ({inert_total*100:.0f} "
               "mol%). Z-factor correlations lose accuracy for very sour "
               "or inert-rich gas.")
    return r


def check_z_validity(Tpc, Ppc, T_R, p_min, p_max, z_corr):
    """Soft check that the pseudo-reduced range stays within the
    Z-factor correlation's published envelope."""
    r = ValidationResult()
    rng = GAS_Z_RANGES.get(z_corr)
    if not rng or Tpc is None or Ppc is None or Tpc <= 0 or Ppc <= 0:
        return r
    Tpr = T_R / Tpc
    Ppr_lo = max(p_min, 14.7) / Ppc
    Ppr_hi = p_max / Ppc
    tlo, thi, _, src = rng["Tpr"]
    plo, phi, _, _ = rng["Ppr"]
    if Tpr < tlo or Tpr > thi:
        r.warn(f"Pseudo-reduced temperature Tpr = {Tpr:.2f} is outside "
               f"the {z_corr} validity range ({tlo}–{thi}). "
               f"Z-factor results are extrapolated. [{src}]")
    if Ppr_hi > phi:
        r.warn(f"Pseudo-reduced pressure reaches Ppr = {Ppr_hi:.1f}, "
               f"above the {z_corr} range (≤ {phi}). High-pressure "
               "Z-factor values are extrapolated.")
    return r


# ----------------------------------------------------------------------
# Wet gas branch validation
# ----------------------------------------------------------------------
def check_wetgas_inputs(gas_sg, api_cond, cgr, T_F, Pdew, p_min, p_max):
    """Full validation for the wet-gas / condensate branch."""
    r = ValidationResult()
    r.merge(check_temperature(T_F))
    r.merge(check_pressure_range(p_min, p_max))
    r.merge(check_positive(cgr, "Condensate–gas ratio (CGR)",
                            allow_zero=True))

    if gas_sg is not None and gas_sg <= 0:
        r.error(f"Gas specific gravity must be positive (got {gas_sg:g}).")
    if api_cond is not None and (api_cond < 30 or api_cond > 75):
        r.warn(f"Condensate API of {api_cond:g} is unusual — condensates "
               "are typically 45–65 °API.")
    if Pdew is not None and Pdew <= 0:
        r.error(f"Dew-point pressure must be positive (got {Pdew:g}).")
    if (Pdew is not None and p_max is not None and Pdew > p_max):
        r.warn(f"Dew-point pressure ({Pdew:g}) is above the maximum table "
               "pressure — the table will not show the single-phase gas "
               "region above the dew point.")
    return r


# ----------------------------------------------------------------------
# Composition validation (compositional EOS branch)
# ----------------------------------------------------------------------
def check_composition(z_dict):
    """Validate a composition dictionary {component: mole_fraction}."""
    r = ValidationResult()
    if not z_dict:
        r.error("The composition is empty — enter at least one component.")
        return r
    total = sum(v for v in z_dict.values() if v is not None)
    if total <= 0:
        r.error("All component mole fractions are zero. Enter a "
                "composition that sums to a positive value.")
        return r
    if any((v or 0) < 0 for v in z_dict.values()):
        r.error("One or more component mole fractions are negative.")
    # Composition is renormalized internally, but warn if far from 1.
    if abs(total - 1.0) > 0.02:
        r.warn(f"Component mole fractions sum to {total:.3f}, not 1.0. "
               "The composition will be renormalized automatically.")
    return r


def check_c7plus(MW_c7, SG_c7, has_c7plus):
    """Validate the C7+ characterization inputs."""
    r = ValidationResult()
    if not has_c7plus:
        return r
    if MW_c7 is not None and (MW_c7 < 90 or MW_c7 > 600):
        r.warn(f"C7+ molecular weight of {MW_c7:g} is unusual — most "
               "reservoir C7+ fractions fall in 100–350 g/mol.")
    if SG_c7 is not None and (SG_c7 < 0.7 or SG_c7 > 1.0):
        r.warn(f"C7+ specific gravity of {SG_c7:g} is unusual — typical "
               "values are 0.75–0.92.")
    if MW_c7 is not None and MW_c7 <= 0:
        r.error("C7+ molecular weight must be positive.")
    if SG_c7 is not None and SG_c7 <= 0:
        r.error("C7+ specific gravity must be positive.")
    return r


# ----------------------------------------------------------------------
# Rock branch validation
# ----------------------------------------------------------------------
def check_porosity(phi):
    """Validate a porosity fraction."""
    r = ValidationResult()
    if phi is None:
        r.error("Porosity is not set.")
        return r
    if phi <= 0 or phi >= 1:
        r.error(f"Porosity must be a fraction between 0 and 1 "
                f"(got {phi:g}). Enter 0.18 for 18%, not 18.")
    elif phi < 0.03:
        r.warn(f"Porosity of {phi:g} ({phi*100:.0f}%) is very low — "
               "compressibility correlations are less reliable here.")
    elif phi > 0.4:
        r.warn(f"Porosity of {phi:g} ({phi*100:.0f}%) is very high — "
               "check the value (unconsolidated sands rarely exceed 0.4).")
    return r


# ----------------------------------------------------------------------
# Internal helper
# ----------------------------------------------------------------------
def _envelope_warn(result, ranges, key, value, corr_name):
    """Add a soft warning if `value` is outside the published envelope."""
    if key not in ranges or value is None:
        return
    lo, hi, unit, src = ranges[key]
    if value < lo or value > hi:
        unit_str = f" {unit}" if unit else ""
        result.warn(
            f"{_pretty(key)} = {value:g}{unit_str} is outside the "
            f"{corr_name} validity range ({lo:g}–{hi:g}{unit_str}). "
            f"Results are extrapolated beyond the correlation's data. "
            f"[{src}]")


def _pretty(key):
    return {"api": "Oil API gravity", "gas_sg": "Gas specific gravity",
            "T": "Temperature", "Rsi": "Solution GOR"}.get(key, key)
