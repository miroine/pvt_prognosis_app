"""
PVT Studio — Wax & Asphaltene Solid-Deposition Risk
=====================================================

Screening-level models for two flow-assurance solids:

  * Wax  — paraffin crystals drop out below the Wax Appearance
           Temperature (WAT, also called the cloud point). Deposition
           is driven by the cold side of the flow path.

  * Asphaltene — heavy polar molecules that flocculate when the oil's
           solvency drops. Unlike wax (temperature-driven), asphaltene
           onset is largely pressure-driven and is usually worst NEAR
           THE BUBBLE POINT, where the light ends have expanded the oil
           and lowered its solvency the most.

These are correlation/index screening tools, not thermodynamic
deposition models. They classify risk and locate the likely envelope so
an engineer knows whether a rigorous study (e.g. a measured WAT, an
asphaltene titration / SARA-based model) is warranted. All correlations
here are deliberately simple and transparent.

References (methods of the type used here):
  - Won (1986), Pedersen (1991) — wax precipitation thermodynamics
  - Coutinho (1998) — predictive wax models
  - de Boer et al. (1995) — asphaltene onset screening from density/
    undersaturation
  - Asphaltene colloidal-instability / SARA indices (various)
"""

import math


# ----------------------------------------------------------------------
# WAX
# ----------------------------------------------------------------------
def estimate_wat(api, wax_content_pct, gor_scf_stb=0.0):
    """Estimate the Wax Appearance Temperature (cloud point), °F.

    A screening correlation: heavier oils and higher wax content raise
    the WAT; high dissolved gas (GOR) depresses it slightly because the
    light ends improve solvency. This is NOT a substitute for a measured
    WAT — lab measurement (cross-polarized microscopy, DSC) is the
    reference method.

    api             : stock-tank oil API gravity
    wax_content_pct : paraffin wax content of the stock-tank oil (wt %)
    gor_scf_stb     : solution gas-oil ratio (scf/STB)
    """
    # Base WAT rises with wax content and falls with API.
    # Anchored so a 5 wt% wax, 35 API oil gives a WAT near ~95 °F.
    wat = 60.0 + 6.0 * wax_content_pct + 0.5 * (40.0 - api)
    # Dissolved gas depresses the WAT.
    wat -= 0.010 * gor_scf_stb
    return wat


def wax_risk(operating_T_F, wat_F):
    """Classify wax-deposition risk from the margin between the operating
    temperature and the WAT.

    Returns a dict: {level, margin_F, message}.
    margin = operating_T - WAT. A negative margin means the fluid is
    already below the cloud point — wax is precipitating.
    """
    margin = operating_T_F - wat_F
    if margin <= 0:
        level = "High"
        msg = ("The operating temperature is at or below the WAT — wax is "
               "actively precipitating. Expect deposition on cold "
               "surfaces; plan for insulation, heating, or regular "
               "pigging.")
    elif margin <= 10.0:
        level = "Elevated"
        msg = ("The operating temperature is within 10 °F of the WAT. "
               "Small temperature excursions (a cold restart, a subsea "
               "section) can trigger wax. Monitor closely.")
    elif margin <= 25.0:
        level = "Moderate"
        msg = ("There is a modest margin above the WAT. Wax is unlikely "
               "at steady state but a measured WAT is advisable before "
               "ruling it out.")
    else:
        level = "Low"
        msg = ("The operating temperature is comfortably above the WAT. "
               "Wax deposition is unlikely under normal operation.")
    return {"level": level, "margin_F": margin, "message": msg}


def wax_cooldown_to_wat(T_initial_F, wat_F, ambient_F, cooldown_const_per_hr):
    """Time for a shut-in fluid to cool from T_initial to the WAT.

    Uses Newton-style exponential cooling:
        T(t) = T_ambient + (T_initial - T_ambient) * exp(-k t)
    Solving T(t) = WAT gives the time available before wax onset — the
    'no-touch' or safe shut-in time for wax.

    Returns hours, or None if the fluid never reaches the WAT (WAT below
    ambient — it will not cool that far).
    """
    if wat_F <= ambient_F:
        return None  # never cools to the WAT
    if T_initial_F <= wat_F:
        return 0.0
    if cooldown_const_per_hr <= 0:
        return None
    ratio = (wat_F - ambient_F) / (T_initial_F - ambient_F)
    if ratio <= 0:
        return None
    return -math.log(ratio) / cooldown_const_per_hr


# ----------------------------------------------------------------------
# ASPHALTENE
# ----------------------------------------------------------------------
def colloidal_instability_index(saturates, aromatics, resins, asphaltenes):
    """Colloidal Instability Index (CII) from a SARA assay.

        CII = (saturates + asphaltenes) / (aromatics + resins)

    Saturates and asphaltenes destabilize; aromatics and resins keep
    asphaltenes in solution. Common screening interpretation:
        CII < 0.7        : stable
        0.7 <= CII < 0.9 : uncertain / metastable
        CII >= 0.9       : unstable — asphaltene problems likely

    All four arguments are SARA fractions (wt %, or any consistent unit;
    only the ratio matters).
    """
    denom = aromatics + resins
    if denom <= 0:
        return float("inf")
    return (saturates + asphaltenes) / denom


def cii_risk(cii):
    """Classify asphaltene risk from the colloidal instability index."""
    if cii >= 0.9:
        return {"level": "High",
                "message": ("CII ≥ 0.9 — the crude is colloidally "
                            "unstable. Asphaltene precipitation is "
                            "likely; a detailed onset study is "
                            "warranted.")}
    elif cii >= 0.7:
        return {"level": "Moderate",
                "message": ("CII between 0.7 and 0.9 — metastable. "
                            "Asphaltene behaviour is sensitive to "
                            "pressure and composition changes; treat "
                            "with caution.")}
    else:
        return {"level": "Low",
                "message": ("CII < 0.7 — the resin/aromatic content "
                            "keeps asphaltenes in solution. Problems "
                            "are unlikely but not impossible near the "
                            "bubble point.")}


def asphaltene_onset_pressure(p_bubble, p_reservoir, instability_factor):
    """Estimate the upper asphaltene-onset pressure (AOP), psia.

    Asphaltene solvency is lowest near the bubble point, so the onset
    envelope sits ABOVE Pb and widens for more unstable crudes. This
    screening estimate places the AOP a fraction of the way from Pb up
    toward the reservoir pressure, scaled by an instability factor in
    [0, 1] (e.g. derived from CII or de-Boer supersaturation).

    Returns the estimated AOP. If the crude is very stable the AOP can
    fall at or below Pb, meaning there is effectively no onset region
    above the bubble point.
    """
    if p_reservoir <= p_bubble:
        span = 0.0
    else:
        span = p_reservoir - p_bubble
    f = max(0.0, min(1.0, instability_factor))
    return p_bubble + f * span


def de_boer_risk(reservoir_density_gcc, undersaturation_psia):
    """De Boer-style asphaltene screening from in-situ oil density and
    the degree of undersaturation (P_reservoir - P_bubble).

    Light, highly undersaturated oils are the classic asphaltene
    problem: the oil is a poor solvent and depressurization moves it a
    long way before the bubble point stabilizes it. This returns a
    qualitative risk band.

    reservoir_density_gcc : in-situ oil density, g/cm3
    undersaturation_psia  : P_reservoir - P_bubble, psia
    """
    light = reservoir_density_gcc < 0.85
    very_light = reservoir_density_gcc < 0.80
    high_under = undersaturation_psia > 3000.0
    mod_under = undersaturation_psia > 1000.0

    if very_light and high_under:
        return {"level": "High",
                "message": ("A light oil with large undersaturation — "
                            "the classic de Boer high-risk region. A "
                            "measured asphaltene-onset study is strongly "
                            "advised.")}
    if light and (high_under or mod_under):
        return {"level": "Moderate",
                "message": ("A fairly light, undersaturated oil — "
                            "asphaltene onset is plausible somewhere "
                            "between reservoir and bubble-point "
                            "pressure.")}
    return {"level": "Low",
            "message": ("A denser or near-saturated oil — de Boer "
                        "screening places this outside the typical "
                        "asphaltene problem region.")}


# ----------------------------------------------------------------------
# Shared
# ----------------------------------------------------------------------
RISK_COLORS = {
    "Low":      "#9DBA00",   # pistachio
    "Moderate": "#E0A800",   # amber
    "Elevated": "#EB6E1F",   # orange
    "High":     "#EB0037",   # torch red
}


def overall_solids_risk(*levels):
    """Combine several risk levels into the single worst-case level."""
    order = ["Low", "Moderate", "Elevated", "High"]
    worst = 0
    for lv in levels:
        if lv in order:
            worst = max(worst, order.index(lv))
    return order[worst]
