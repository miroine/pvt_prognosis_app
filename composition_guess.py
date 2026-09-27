"""
Heuristic composition synthesis from black-oil parameters.

Given API, gas SG, and Rsi, produces a plausible 11-component composition
(N2, CO2, C1...C6, C7+) using the Whitson-Brule lumped-composition method.
This is a *guess* for initial EOS work — not a substitute for a measured
chromatograph.
"""

import numpy as np


def guess_oil_composition(api, gas_sg, Rsi_scfSTB,
                          N2=0.002, CO2=0.005, H2S=0.000):
    """
    Guess oil composition from black-oil inputs.

    Approach (Whitson-style):
    1. The dissolved gas composition is set by gas SG (lighter gas → more C1).
    2. The C7+ MW is correlated from API.
    3. The oil composition is back-computed from a typical molar GOR ratio.

    Returns:
        comp     : dict mapping component -> mole fraction
        MW_c7    : C7+ molecular weight (lb/lbmol)
        SG_c7    : C7+ specific gravity
    """
    # Surface gas composition (mol fractions, normalized after non-HC)
    # Empirical correlation: lighter gas SG → more C1
    if gas_sg < 0.65:
        gas_c1, gas_c2, gas_c3 = 0.88, 0.06, 0.02
    elif gas_sg < 0.75:
        gas_c1, gas_c2, gas_c3 = 0.80, 0.09, 0.05
    elif gas_sg < 0.85:
        gas_c1, gas_c2, gas_c3 = 0.72, 0.10, 0.08
    else:
        gas_c1, gas_c2, gas_c3 = 0.60, 0.12, 0.12
    gas_iC4 = (1 - gas_c1 - gas_c2 - gas_c3) * 0.30
    gas_nC4 = (1 - gas_c1 - gas_c2 - gas_c3) * 0.45
    gas_iC5 = (1 - gas_c1 - gas_c2 - gas_c3) * 0.10
    gas_nC5 = (1 - gas_c1 - gas_c2 - gas_c3) * 0.10
    gas_c6  = (1 - gas_c1 - gas_c2 - gas_c3) * 0.05

    # C7+ molecular weight from API (Cragoe-like correlation)
    # Lighter oil (higher API) -> lower C7+ MW (more volatile)
    MW_c7 = max(120.0, min(350.0, 6084.0 / (api - 5.9)))

    # C7+ specific gravity (Watson-Soreide approximation)
    SG_c7 = max(0.78, min(0.94, 0.85 - 0.005 * (api - 35.0) + 0.001 * 35))
    # Actually simpler: SG_c7 ≈ Kw-based estimate, here approximate as
    SG_c7 = max(0.78, min(0.92, 1.0 - 0.0042 * api))

    # Estimate molar GOR (lbmol gas per lbmol oil at standard conditions)
    # scf/STB to lbmol/lbmol: scf -> lbmol via 379.5 scf/lbmol;
    # STB of oil -> lbmol via oil_MW / (rho_o_lb_per_ft3 * 5.615)
    # rho_o = 62.428 * gamma_o, gamma_o = 141.5/(131.5+API)
    gamma_o = 141.5 / (131.5 + api)
    rho_o = 62.428 * gamma_o
    Mo = MW_c7 * 0.7 + 50.0     # heuristic average oil MW
    lbmol_oil_per_STB = (rho_o * 5.615) / Mo
    lbmol_gas_per_STB = Rsi_scfSTB / 379.5
    if lbmol_oil_per_STB <= 0:
        molar_GOR = 1.0
    else:
        molar_GOR = lbmol_gas_per_STB / lbmol_oil_per_STB

    # Stock-tank oil composition (approximate): mostly C7+, some C5-C6
    oil_c7  = 0.93
    oil_nC5 = 0.02
    oil_nC4 = 0.02
    oil_c6  = 0.03

    # Recombine: total mole fraction = (oil_mol * x_i + gas_mol * y_i) / (oil_mol + gas_mol)
    n_total = 1.0 + molar_GOR
    oil_frac = 1.0 / n_total
    gas_frac = molar_GOR / n_total

    # Non-HC mostly in the gas phase (assumption)
    yN2  = N2;  yCO2 = CO2; yH2S = H2S
    # Reduce gas HC fractions so non-HC + HC sum to 1 in the gas
    hc_total = 1.0 - (yN2 + yCO2 + yH2S)
    gas_c1  *= hc_total
    gas_c2  *= hc_total
    gas_c3  *= hc_total
    gas_iC4 *= hc_total; gas_nC4 *= hc_total
    gas_iC5 *= hc_total; gas_nC5 *= hc_total
    gas_c6  *= hc_total

    comp = {
        "N2":  gas_frac * yN2,
        "CO2": gas_frac * yCO2,
        "H2S": gas_frac * yH2S,
        "C1":  gas_frac * gas_c1,
        "C2":  gas_frac * gas_c2,
        "C3":  gas_frac * gas_c3,
        "iC4": gas_frac * gas_iC4,
        "nC4": gas_frac * gas_nC4 + oil_frac * oil_nC4,
        "iC5": gas_frac * gas_iC5,
        "nC5": gas_frac * gas_nC5 + oil_frac * oil_nC5,
        "C6":  gas_frac * gas_c6 + oil_frac * oil_c6,
        "C7+": oil_frac * oil_c7,
    }

    # Normalize
    total = sum(comp.values())
    if total > 0:
        comp = {k: v / total for k, v in comp.items()}

    return comp, MW_c7, SG_c7


def guess_gas_composition(gas_sg, N2=0.005, CO2=0.01, H2S=0.0, is_wet=False, cgr=None):
    """
    Guess dry-gas or wet-gas composition from gas SG.

    For wet gas, also adds a C7+ representing condensate based on CGR.
    """
    # HC fractions from SG (light to heavy)
    if gas_sg < 0.62:
        c1, c2, c3 = 0.93, 0.04, 0.015
    elif gas_sg < 0.70:
        c1, c2, c3 = 0.85, 0.07, 0.04
    elif gas_sg < 0.80:
        c1, c2, c3 = 0.76, 0.09, 0.07
    else:
        c1, c2, c3 = 0.65, 0.11, 0.10

    hc_residual = 1.0 - (c1 + c2 + c3)
    iC4 = hc_residual * 0.20
    nC4 = hc_residual * 0.25
    iC5 = hc_residual * 0.15
    nC5 = hc_residual * 0.15
    c6  = hc_residual * 0.25

    comp = {
        "N2":  N2,
        "CO2": CO2,
        "H2S": H2S,
        "C1":  c1, "C2":  c2, "C3":  c3,
        "iC4": iC4, "nC4": nC4,
        "iC5": iC5, "nC5": nC5,
        "C6":  c6,
        "C7+": 0.0,
    }
    # Scale HC down by non-HC fraction
    nonhc = N2 + CO2 + H2S
    scale = 1.0 - nonhc
    for k in ["C1", "C2", "C3", "iC4", "nC4", "iC5", "nC5", "C6"]:
        comp[k] *= scale

    MW_c7 = 110.0; SG_c7 = 0.78  # defaults

    if is_wet and cgr and cgr > 0:
        # CGR in STB/MMscf — convert to mole fraction of C7+
        # CGR -> approximate Rv (STB/scf) -> mole-frac via partial molar volume
        # Quick heuristic: each 50 STB/MMscf adds ~0.005 mole frac of C7+
        c7_frac = min(cgr * 0.0001, 0.10)
        # Take this from the HCs proportionally
        scale_down = 1.0 - c7_frac
        for k in ["C1", "C2", "C3", "iC4", "nC4", "iC5", "nC5", "C6"]:
            comp[k] *= scale_down
        comp["C7+"] = c7_frac
        MW_c7 = 130.0; SG_c7 = 0.79

    # Normalize
    total = sum(comp.values())
    comp = {k: v / total for k, v in comp.items()}

    return comp, MW_c7, SG_c7
