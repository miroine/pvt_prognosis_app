"""
ECLIPSE black oil PVT keyword formatters.

Reference: Schlumberger ECLIPSE Reference Manual — keywords PVTO, PVDG, PVTG,
PVTW, DENSITY.

PVTO format:
  Rs   P_sat   Bo   muo
            P>Psat Bo  muo     <- under-saturated branch for that Rs
            P>Psat Bo  muo  /
  next Rs ...
  /

PVTG format (the dual of PVTO):
  P    Rv_sat   Bg   mug
            Rv<Rv_sat   Bg  mug   /
  next P ...
  /
"""

import numpy as np


def build_pvto(df, Pb, oil, Rsi, P_max):
    """
    Build PVTO keyword. df contains saturated branch for P <= Pb plus
    one extra row at the highest tabulated pressure.
    We construct nodes at several Rs values (Rs from 0 to Rsi) and add
    under-saturated extension at the maximum Rs node up to P_max.
    """
    lines = ["PVTO", "-- Rs       Psat       Bo        Muo",
             "-- Mscf/STB  psia       rb/STB    cP"]

    # Choose Rs node values (e.g. 8 nodes from low to Rsi)
    n_nodes = 8
    rs_nodes = np.linspace(max(Rsi * 0.05, 1.0), Rsi, n_nodes)

    for i, Rs in enumerate(rs_nodes):
        # Saturation pressure for this Rs
        Psat = oil.bubble_point(Rs)
        Bo_sat = oil.formation_volume_factor(Psat, Rs, saturated=True)
        mu_sat = oil.viscosity(Psat, Rs, Psat, saturated=True)

        Rs_Mscf = Rs / 1000.0  # ECLIPSE wants Mscf/STB in METRIC, but METRIC differs.
        # In FIELD units PVTO uses Mscf/STB for Rs.

        line = f"  {Rs_Mscf:8.4f}   {Psat:9.2f}   {Bo_sat:7.4f}   {mu_sat:7.4f}"

        # Under-saturated extension only on the LAST (highest Rs) node
        if i == len(rs_nodes) - 1:
            lines.append(line)
            # Add P > Psat points up to P_max
            P_under = np.linspace(Psat * 1.1, P_max, 5)
            for Pu in P_under:
                if Pu <= Psat:
                    continue
                Bo_u = oil.formation_volume_factor(Pu, Rs, saturated=False, Pb=Psat)
                mu_u = oil.viscosity(Pu, Rs, Psat, saturated=False)
                lines.append(f"            {Pu:9.2f}   {Bo_u:7.4f}   {mu_u:7.4f}")
            lines[-1] += "  /"
        else:
            lines.append(line + "  /")

    lines.append("/")
    return "\n".join(lines) + "\n"


def build_pvdg(df):
    """PVDG: dry gas table — P, Bg, mug."""
    lines = ["PVDG", "-- P         Bg          Mug",
             "-- psia      rb/Mscf     cP"]
    for _, row in df.iterrows():
        # ECLIPSE field PVDG uses rb/Mscf for Bg
        Bg_Mscf = row["Bg (rb/scf)"] * 1000.0
        lines.append(f"  {row['P (psia)']:9.2f}   {Bg_Mscf:9.5f}   {row['μg (cp)']:8.5f}")
    lines.append("/")
    return "\n".join(lines) + "\n"


def build_pvtw(Pref, Bw, Cw, muw, viscosibility=0.0):
    """PVTW: single-line water properties."""
    lines = ["PVTW",
             "-- Pref     Bwref     Cw           Muw       Viscosibility",
             f"   {Pref:8.2f}  {Bw:8.4f}  {Cw:11.4e}  {muw:7.4f}  {viscosibility:.4e}  /",
             ""]
    return "\n".join(lines)


def build_pvtg(pressures, wetgas):
    """
    PVTG: live (wet) gas table.
    Outer loop = pressure node, inner branch = decreasing Rv values.
    Bg here is in rb/Mscf (FIELD units), Rv in STB/Mscf.

    For each pressure node:
        line 1 :  P   Rv_sat   Bg(P,Rv_sat)   mu_g(P,Rv_sat)
        lines  :       Rv_low  Bg_dry         mu_g_dry        /
    All terminated by /.
    """
    lines = ["PVTG", "-- P        Rv         Bg         Mug",
             "-- psia     STB/Mscf   rb/Mscf    cP"]

    for P in pressures:
        if P < 14.7:
            continue
        Z = wetgas.z_factor(P)
        Bg = wetgas.formation_volume_factor(P, Z) * 1000.0   # rb/scf -> rb/Mscf
        mu = wetgas.viscosity(P, Z)
        Rv_sat = wetgas.rv(P) * 1000.0                       # STB/scf -> STB/Mscf

        # Saturated (highest Rv) line
        lines.append(f"  {P:8.2f}  {Rv_sat:9.5f}  {Bg:9.5f}  {mu:8.5f}")

        # Lower-Rv branch — Bg decreases slightly, mu increases slightly
        # Use a small linear correction with Rv (typical lab behaviour)
        Rv_branch = np.linspace(Rv_sat, 0.0, 4)[1:]
        for i, Rv in enumerate(Rv_branch):
            # Approximate corrections (~0.5–1 % in Bg, ~1–2 % in mu)
            frac = (Rv_sat - Rv) / max(Rv_sat, 1e-9)
            Bg_b = Bg * (1 - 0.005 * frac)
            mu_b = mu * (1 + 0.015 * frac)
            terminator = "  /" if i == len(Rv_branch) - 1 else ""
            lines.append(f"            {Rv:9.5f}  {Bg_b:9.5f}  {mu_b:8.5f}{terminator}")

    lines.append("/")
    return "\n".join(lines) + "\n"


def build_pvto_from_compositional(rows, Pb, P_max):
    """
    Build PVTO from compositional black-oil table rows.
    `rows` is the list returned by black_oil_table_from_composition (oil).
    """
    sat_rows = [r for r in rows if r["P"] <= Pb + 1.0]
    und_rows = [r for r in rows if r["P"] >  Pb + 1.0]

    lines = ["PVTO", "-- Rs       Psat       Bo        Muo",
             "-- Mscf/STB  psia       rb/STB    cP"]

    for i, r in enumerate(sat_rows):
        Rs = r["Rs"] / 1000.0
        is_last_sat = (i == len(sat_rows) - 1)
        line = f"  {Rs:8.4f}   {r['P']:9.2f}   {r['Bo']:7.4f}   {r['mu_o']:7.4f}"
        if is_last_sat and und_rows:
            lines.append(line)
            for j, ru in enumerate(und_rows):
                terminator = "  /" if j == len(und_rows) - 1 else ""
                lines.append(f"            {ru['P']:9.2f}   {ru['Bo']:7.4f}   {ru['mu_o']:7.4f}{terminator}")
        else:
            lines.append(line + "  /")

    lines.append("/")
    return "\n".join(lines) + "\n"


def build_pvtg_from_compositional(rows, Pdew):
    """Build PVTG from compositional gas-condensate table rows."""
    lines = ["PVTG", "-- P        Rv         Bg         Mug",
             "-- psia     STB/Mscf   rb/Mscf    cP"]
    for r in rows:
        P = r["P"]
        Rv = r["Rv"]   # already in STB/Mscf in the table
        Bg = r["Bg"]
        mu = r["mu_g"]
        lines.append(f"  {P:8.2f}  {Rv:9.5f}  {Bg:9.5f}  {mu:8.5f}")
        Bg_dry = Bg * 0.995
        mu_dry = mu * 1.02
        lines.append(f"            {0.0:9.5f}  {Bg_dry:9.5f}  {mu_dry:8.5f}  /")
    lines.append("/")
    return "\n".join(lines) + "\n"


def build_pvtw_from_table(pressures, water, Pref):
    """
    Single-line PVTW evaluated at Pref, but using best-fit Cw and viscosibility
    derived from the table (so the linear PVTW model matches the correlation
    over the tabulated range).
    """
    Bw = np.array([water.bw(p) for p in pressures])
    mu = np.array([water.viscosity(p) for p in pressures])
    Cw_table = np.array([water.compressibility(p) for p in pressures])

    Bwref = water.bw(Pref)
    muref = water.viscosity(Pref)
    # Average Cw and effective viscosibility (d ln mu / dP)
    Cw_avg = float(np.mean(Cw_table))
    if len(pressures) > 1 and mu[0] > 0 and mu[-1] > 0:
        viscosibility = float((np.log(mu[-1]) - np.log(mu[0])) / (pressures[-1] - pressures[0]))
        viscosibility = max(viscosibility, 0.0)
    else:
        viscosibility = 0.0
    return build_pvtw(Pref, Bwref, Cw_avg, muref, viscosibility)


def build_density(api, gas_sg, water_sg=1.02):
    """DENSITY keyword — surface densities (oil, water, gas) in lb/ft3."""
    gamma_o = 141.5 / (131.5 + api)
    rho_o = 62.428 * gamma_o
    rho_w = 62.428 * water_sg
    rho_g = 0.0764 * gas_sg
    lines = ["DENSITY",
             "-- Oil       Water     Gas      (lb/ft3)",
             f"   {rho_o:7.3f}   {rho_w:7.3f}   {rho_g:7.4f}  /",
             ""]
    return "\n".join(lines)


def build_full_deck(pvto="", pvdg="", pvtg="", pvtw="", density=""):
    """Concatenate sections into a stand-alone INC file."""
    header = ("-- =====================================================\n"
              "-- Black-oil / Wet-gas PVT include file\n"
              "-- Generated by Streamlit PVT App\n"
              "-- Units: FIELD\n"
              "-- =====================================================\n\n")
    parts = [header]
    if density: parts.append(density + "\n")
    if pvto:    parts.append(pvto + "\n")
    if pvdg:    parts.append(pvdg + "\n")
    if pvtg:    parts.append(pvtg + "\n")
    if pvtw:    parts.append(pvtw + "\n")
    return "".join(parts)
