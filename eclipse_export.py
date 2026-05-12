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


def build_full_deck(pvto="", pvdg="", pvtg="", pvtw="", density="", units="FIELD"):
    """Concatenate sections into a stand-alone INC file.

    units: 'FIELD' (default — psia/°F/scf/STB) or 'METRIC' (bara/°C/Sm3/Sm3).
            This only changes the comment header; the keyword bodies must be
            converted by the caller because ECLIPSE expects consistent units
            throughout the deck (set via RUNSPEC).
    """
    units_line = ("Field: psia, °F, scf/STB, rb/STB, rb/Mscf, cP, lb/ft3"
                  if units == "FIELD" else
                  "Metric: bara, °C, Sm3/Sm3, rm3/Sm3, cP, kg/m3")
    header = (f"-- =====================================================\n"
              f"-- Multi-fluid PVT include file\n"
              f"-- Generated by Streamlit PVT App\n"
              f"-- Units: {units} ({units_line})\n"
              f"-- RUNSPEC must contain: {units}\n"
              f"-- =====================================================\n\n")
    parts = [header]
    if density: parts.append(density + "\n")
    if pvto:    parts.append(pvto + "\n")
    if pvdg:    parts.append(pvdg + "\n")
    if pvtg:    parts.append(pvtg + "\n")
    if pvtw:    parts.append(pvtw + "\n")
    return "".join(parts)


# ---------- METRIC conversion of pre-built FIELD keyword bodies ----------
# These walk through the lines and convert numerical columns in-place.

def _convert_pvto_to_metric(field_text):
    """Convert a PVTO block from FIELD (psia, scf/STB, rb/STB, cP)
    to METRIC (bara, Sm3/Sm3, rm3/Sm3, cP)."""
    PSI_PER_BAR = 14.50377
    SCF_PER_SM3 = 5.6146
    out_lines = []
    header_done = False
    for line in field_text.splitlines():
        s = line.strip()
        if s.startswith("PVTO"):
            out_lines.append(line)
            out_lines.append("-- Rs        Psat       Bo        Muo")
            out_lines.append("-- Sm3/Sm3   bara       rm3/Sm3   cP")
            header_done = True
            continue
        if not header_done and s.startswith("--"):
            continue   # drop original FIELD header comments
        if s == "/" or not s:
            out_lines.append(line); continue
        if s.startswith("--"):
            out_lines.append(line); continue
        has_slash = s.endswith("/")
        core = s.rstrip("/").strip()
        toks = core.split()
        try: vals = [float(t) for t in toks]
        except ValueError:
            out_lines.append(line); continue
        if len(vals) == 4:
            Rs, P, Bo, Mu = vals
            Rs_si = Rs * 1000.0 / SCF_PER_SM3
            new = f"  {Rs_si:8.4f}   {P/PSI_PER_BAR:9.3f}   {Bo:7.4f}   {Mu:7.4f}"
        elif len(vals) == 3:
            P, Bo, Mu = vals
            new = f"            {P/PSI_PER_BAR:9.3f}   {Bo:7.4f}   {Mu:7.4f}"
        else:
            out_lines.append(line); continue
        if has_slash: new += "  /"
        out_lines.append(new)
    return "\n".join(out_lines) + ("\n" if not field_text.endswith("\n") else "")


def _convert_pvdg_to_metric(field_text):
    """Convert PVDG: P [psia]->bara, Bg [rb/Mscf]->rm3/Sm3."""
    PSI_PER_BAR = 14.50377
    SCF_PER_SM3 = 5.6146
    out = []
    for line in field_text.splitlines():
        s = line.strip()
        if not s or s.startswith("--") or s.startswith("PVDG") or s == "/":
            out.append(line); continue
        toks = s.rstrip("/").split()
        try: P, Bg, Mu = [float(t) for t in toks]
        except ValueError: out.append(line); continue
        # Bg rb/Mscf → rm3/Sm3:  rb/Mscf * 5.6146/1000 = rm3/Mscf, /5.6146 again = rm3/Sm3... no
        # Cleaner: 1 rb = 5.6146 cuft = 0.158987 m3; 1 Mscf = 28.3168 m3 at SC; so Bg(rm3/Sm3) = Bg(rb/Mscf)*0.158987/28.3168
        Bg_si = Bg * 0.158987 / 28.3168 * 1000.0  # rm3/Mscf normalized... wait
        # Field PVDG Bg is rb/Mscf. Metric PVDG Bg is rm3/Sm3.
        # rb -> rm3: × 0.158987;  Mscf -> Sm3: × 28.3168
        # so Bg(rm3/Sm3) = Bg(rb/Mscf) × 0.158987 / 28.3168
        Bg_si = Bg * 0.158987 / 28.3168
        out.append(f"  {P/PSI_PER_BAR:9.3f}   {Bg_si:11.7f}   {Mu:8.5f}")
    return "\n".join(out) + "\n"


def _convert_pvtg_to_metric(field_text):
    """Convert PVTG: P, Rv [STB/Mscf]->Sm3/Sm3, Bg [rb/Mscf]->rm3/Sm3."""
    PSI_PER_BAR = 14.50377
    SCF_PER_SM3 = 5.6146  # 1 Sm3 = 35.3147 scf? Actually 1 Sm3 (at 15°C, 1 atm) = 5.6146 scf? No.
    # STB/Mscf: barrels per thousand scf.  Sm3/Sm3 dimensionless. Conversion:
    # 1 STB = 0.158987 Sm3,  1 Mscf = 28.3168 Sm3.  So STB/Mscf × 0.158987/28.3168 = Sm3/Sm3
    out = []
    for line in field_text.splitlines():
        s = line.strip()
        if not s or s.startswith("--") or s.startswith("PVTG") or s == "/":
            out.append(line); continue
        has_slash = s.endswith("/")
        core = s.rstrip("/").strip()
        toks = core.split()
        try: vals = [float(t) for t in toks]
        except ValueError: out.append(line); continue
        if len(vals) == 4:
            P, Rv, Bg, Mu = vals
            Rv_si = Rv * 0.158987 / 28.3168
            Bg_si = Bg * 0.158987 / 28.3168
            new = f"  {P/PSI_PER_BAR:9.3f}  {Rv_si:11.7f}  {Bg_si:11.7f}  {Mu:8.5f}"
        elif len(vals) == 3:
            Rv, Bg, Mu = vals
            Rv_si = Rv * 0.158987 / 28.3168
            Bg_si = Bg * 0.158987 / 28.3168
            new = f"             {Rv_si:11.7f}  {Bg_si:11.7f}  {Mu:8.5f}"
        else:
            out.append(line); continue
        if has_slash: new += "  /"
        out.append(new)
    return "\n".join(out) + "\n"


def _convert_pvtw_to_metric(field_text):
    """Convert PVTW: Pref [psia]->bara, Cw [1/psi]->1/bar."""
    PSI_PER_BAR = 14.50377
    out = []
    for line in field_text.splitlines():
        s = line.strip()
        if not s or s.startswith("--") or s.startswith("PVTW"):
            out.append(line); continue
        has_slash = s.endswith("/")
        core = s.rstrip("/").strip()
        toks = core.split()
        try: vals = [float(t) for t in toks]
        except ValueError: out.append(line); continue
        if len(vals) >= 4:
            P, Bw, Cw, Mu = vals[:4]
            visc = vals[4] if len(vals) > 4 else 0.0
            new = (f"   {P/PSI_PER_BAR:8.3f}  {Bw:8.4f}  "
                   f"{Cw*PSI_PER_BAR:11.4e}  {Mu:7.4f}  {visc*PSI_PER_BAR:.4e}")
            if has_slash: new += "  /"
            out.append(new)
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def _convert_density_to_metric(field_text):
    """Convert DENSITY: lb/ft3 -> kg/m3."""
    F = 16.01846
    out = []
    for line in field_text.splitlines():
        s = line.strip()
        if not s or s.startswith("--") or s.startswith("DENSITY"):
            out.append(line); continue
        has_slash = s.endswith("/")
        core = s.rstrip("/").strip()
        toks = core.split()
        try: vals = [float(t) for t in toks]
        except ValueError: out.append(line); continue
        if len(vals) >= 3:
            ro, rw, rg = vals[:3]
            new = f"   {ro*F:7.2f}   {rw*F:7.2f}   {rg*F:7.3f}"
            if has_slash: new += "  /"
            out.append(new)
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def convert_deck_to_metric(pvto="", pvdg="", pvtg="", pvtw="", density=""):
    """Convenience: convert each pre-built FIELD section to METRIC."""
    return {
        "pvto":    _convert_pvto_to_metric(pvto) if pvto else "",
        "pvdg":    _convert_pvdg_to_metric(pvdg) if pvdg else "",
        "pvtg":    _convert_pvtg_to_metric(pvtg) if pvtg else "",
        "pvtw":    _convert_pvtw_to_metric(pvtw) if pvtw else "",
        "density": _convert_density_to_metric(density) if density else "",
    }


# -------------------- RSVD / RVVD: composition vs depth --------------------
def build_rsvd(depth_rs_pairs, units="FIELD"):
    """
    Build RSVD keyword: solution GOR vs depth.

    depth_rs_pairs: list of (depth, Rs) tuples.
       FIELD: depth in ft, Rs in scf/STB (will be output as Mscf/STB)
       METRIC: depth in m, Rs in Sm3/Sm3
    """
    label = "ft, Mscf/STB" if units == "FIELD" else "m, Sm3/Sm3"
    lines = ["RSVD", f"-- Depth   Rs    ({label})"]
    for d, rs in depth_rs_pairs:
        rs_out = rs / 1000.0 if units == "FIELD" else rs   # Mscf/STB for field
        lines.append(f"  {d:9.2f}   {rs_out:8.4f}")
    lines.append("/")
    return "\n".join(lines) + "\n"


def build_rvvd(depth_rv_pairs, units="FIELD"):
    """
    Build RVVD keyword: vaporized-oil ratio vs depth.

    depth_rv_pairs: list of (depth, Rv) tuples.
       FIELD: depth in ft, Rv in STB/Mscf (output as STB/Mscf)
       METRIC: depth in m, Rv in Sm3/Sm3
    """
    label = "ft, STB/Mscf" if units == "FIELD" else "m, Sm3/Sm3"
    lines = ["RVVD", f"-- Depth   Rv    ({label})"]
    for d, rv in depth_rv_pairs:
        lines.append(f"  {d:9.2f}   {rv:8.5f}")
    lines.append("/")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------
# Multi-region ECLIPSE export
# ----------------------------------------------------------------
def build_multi_region_deck(regions, header_extra=""):
    """
    Build a multi-region ECLIPSE PVT deck (PVTNUM > 1).

    `regions` is a list of dicts, each with optional keys:
        name, pvto, pvdg, pvtg, pvtw, density

    Each keyword is written as its own block with header comments
    indicating the region index. ECLIPSE associates region 1 with
    the first PVTNUM, region 2 with the second, etc.

    Each KEYWORD-then-`/` block within a single PVTO/PVDG/PVTG keyword
    is one region's table; ECLIPSE expects the regions to appear in
    PVTNUM order under the same keyword.
    """
    header = ("-- =====================================================\n"
              f"-- Multi-region black-oil PVT include file\n"
              f"-- {len(regions)} PVT regions\n"
              "-- Generated by PVT Studio (Equinor-themed Streamlit app)\n"
              "-- Units: FIELD\n"
              "-- =====================================================\n\n"
              + (header_extra + "\n" if header_extra else ""))
    out = [header]

    # DENSITY: stack one line per region inside one DENSITY keyword
    densities = [r.get("_density_line") for r in regions if r.get("_density_line")]
    if densities:
        out.append("DENSITY\n-- Oil       Water     Gas      (lb/ft3)\n")
        for i, line in enumerate(densities):
            out.append(f"   {line}  -- region {i+1}\n")
        out.append("/\n\n")

    # PVTO: concatenate each region's saturated-block(s) inside one PVTO keyword
    pvto_regions = [r.get("_pvto_body") for r in regions if r.get("_pvto_body")]
    if pvto_regions:
        out.append("PVTO\n-- Rs        Psat       Bo        Muo\n"
                   "-- Mscf/STB  psia       rb/STB    cP\n")
        for i, body in enumerate(pvto_regions):
            out.append(f"-- region {i+1}\n{body}/\n")
        out.append("/\n\n")

    # PVDG: same structure
    pvdg_regions = [r.get("_pvdg_body") for r in regions if r.get("_pvdg_body")]
    if pvdg_regions:
        out.append("PVDG\n-- P         Bg          Mug\n"
                   "-- psia      rb/Mscf     cP\n")
        for i, body in enumerate(pvdg_regions):
            out.append(f"-- region {i+1}\n{body}/\n")
        out.append("/\n\n")

    # PVTG: similar
    pvtg_regions = [r.get("_pvtg_body") for r in regions if r.get("_pvtg_body")]
    if pvtg_regions:
        out.append("PVTG\n-- P        Rv         Bg         Mug\n"
                   "-- psia     STB/Mscf   rb/Mscf    cP\n")
        for i, body in enumerate(pvtg_regions):
            out.append(f"-- region {i+1}\n{body}/\n")
        out.append("/\n\n")

    # PVTW: one line per region
    pvtw_regions = [r.get("_pvtw_line") for r in regions if r.get("_pvtw_line")]
    if pvtw_regions:
        out.append("PVTW\n-- Pref     Bwref     Cw           Muw       Viscosibility\n")
        for i, line in enumerate(pvtw_regions):
            out.append(f"   {line}  -- region {i+1}\n")
        out.append("/\n\n")

    return "".join(out)


# ----------------------------------------------------------------
# Helpers: extract body of a PVTO/PVDG/PVTG/PVTW/DENSITY string
# (the part between the keyword line and the final `/`, useful for
# stacking into multi-region keywords above)
# ----------------------------------------------------------------
def extract_keyword_body(text, keyword):
    """
    Given a full keyword block string (e.g. as produced by build_pvto),
    return everything between the keyword header (and its column comments)
    and the final closing `/`.
    """
    lines = text.strip().split("\n")
    # Skip the keyword line and any leading `--` comments
    body_start = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith(keyword):
            body_start = i + 1
            # Skip immediately following comment lines
            while body_start < len(lines) and lines[body_start].strip().startswith("--"):
                body_start += 1
            break
    # Drop trailing `/` line
    body_end = len(lines)
    while body_end > body_start and lines[body_end - 1].strip() in ("/", ""):
        body_end -= 1
    return "\n".join(lines[body_start:body_end]) + "\n"


def extract_density_line(text):
    """Extract just the numeric line from a DENSITY keyword block."""
    for ln in text.strip().split("\n"):
        if not ln.strip().startswith(("--", "DENSITY")) and "/" in ln:
            # Strip trailing /
            return ln.split("/")[0].strip()
    return None


def extract_pvtw_line(text):
    """Extract just the numeric line from a PVTW keyword block."""
    for ln in text.strip().split("\n"):
        if not ln.strip().startswith(("--", "PVTW")) and "/" in ln:
            return ln.split("/")[0].strip()
    return None
