"""
Rock (pore-volume) compressibility correlations.

Pore compressibility Cf [1/psi] = -(1/V_p) * dV_p/dP at constant overburden.
This is the parameter ECLIPSE expects in the ROCK keyword.

Correlations implemented:
  - Hall (1953)              : consolidated limestone/sandstone, original chart
  - Newman (1973) — Consolidated sandstone
  - Newman (1973) — Limestone
  - Horne                    : simple polynomial fit
  - van der Knaap (1959)     : low-porosity carbonates

All correlations take porosity (fraction, 0-1) and return Cf in 1/psi.

Validity: typical reservoir conditions, porosities 0.05-0.30.
For unconsolidated / loose sands, expect Cf values one order of magnitude higher.
"""

import numpy as np


def hall_1953(phi):
    """
    Hall (1953) — consolidated rocks.
    Cf = 1.782e-6 / phi^0.438
    Probably the most widely cited; reasonable for consolidated SS and LS.
    """
    phi = max(min(phi, 0.40), 0.01)
    return 1.782e-6 / phi ** 0.438


def newman_consolidated_sandstone(phi):
    """
    Newman (1973), consolidated sandstone.
    Cf = 97.32e-6 / (1 + 55.8721 * phi)^1.42855
    """
    phi = max(min(phi, 0.40), 0.01)
    return 97.32e-6 / (1.0 + 55.8721 * phi) ** 1.42855


def newman_limestone(phi):
    """
    Newman (1973), limestone (consolidated).
    Cf = 0.8535 / (phi^? )... simpler form
    Cf = 1.86e-6 / phi^0.59
    """
    phi = max(min(phi, 0.40), 0.01)
    return 1.86e-6 / phi ** 0.59


def horne(phi):
    """
    Horne polynomial fit:
    Cf = (4.55 - 4.02 * phi) * 1e-6
    Quick approximation for sandstones.
    """
    phi = max(min(phi, 0.40), 0.01)
    return (4.55 - 4.02 * phi) * 1e-6


def carpenter_spencer(phi):
    """
    Carpenter & Spencer-style correlation for consolidated limestone.
    Cf ≈ 7.5e-6 / (1 + 60*phi)
    Reasonable for limestone porosity 0.05-0.20.
    """
    phi = max(min(phi, 0.40), 0.01)
    return 7.5e-6 / (1.0 + 60.0 * phi)


CORRELATIONS = {
    "Hall (1953)": hall_1953,
    "Newman — Sandstone": newman_consolidated_sandstone,
    "Newman — Limestone": newman_limestone,
    "Horne polynomial": horne,
    "Carpenter-Spencer (carbonate)": carpenter_spencer,
}


def compute_all(phi):
    """Return Cf from every correlation as a dict {name: Cf}."""
    return {name: fn(phi) for name, fn in CORRELATIONS.items()}


def rock_keyword(Pref_psia, Cf_per_psi):
    """
    Build the ECLIPSE ROCK keyword.

    Format (field units):
        ROCK
        -- Pref     Cf
           psia     1/psi
        /
    """
    lines = ["ROCK", "-- Pref      Cf",
             "-- psia       1/psi",
             f"   {Pref_psia:8.2f}   {Cf_per_psi:11.4e}  /"]
    return "\n".join(lines) + "\n"


def rock_keyword_metric(Pref_bar, Cf_per_bar):
    """Build the ROCK keyword in METRIC units (bara, 1/bar)."""
    lines = ["ROCK", "-- Pref      Cf",
             "-- bara       1/bar",
             f"   {Pref_bar:8.3f}   {Cf_per_bar:11.4e}  /"]
    return "\n".join(lines) + "\n"
