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

APP_VERSION = "1.4.2"   # must match pvt_app.py (deployment check)

import numpy as np


def pvto_nodes(oil, Rsi, Pb, P_max, P_min=14.7, n_sat=20, n_under=6,
               extend_above_pb=True):
    """Compute the PVTO node structure (FIELD units).

    Returns a list of dicts, one per saturated Rs node, in increasing Rs:
        {"Rs": scf/STB, "Psat": psia, "Bo_sat", "mu_sat",
         "P_u": [...], "Bo_u": [...], "mu_u": [...]}   (undersaturated)

    Span (the ECLIPSE-recommended layout):
      * saturated nodes from ~1 atm up to Pb — and, if extend_above_pb,
        on up to P_max using the correlation Rs(P) beyond Rsi, so cells
        that re-dissolve gas (gas injection, gas-cap contact, pressure
        maintenance) never fall off the table;
      * Rsi at exactly Pb is always a node (the initial oil);
      * EVERY node carries an undersaturated branch from its Psat up to
        ~1.25 x the table top, so ECLIPSE never has to borrow a branch
        from a neighbouring node.
    """
    P_lo = max(P_min, 14.7)
    P_top_sat = max(P_max, Pb) if extend_above_pb else Pb
    grid = set(np.round(np.geomspace(P_lo, P_top_sat, int(n_sat)), 3))
    grid.add(round(Pb, 3))
    P_grid = sorted(p for p in grid if P_lo <= p <= P_top_sat + 1e-6)
    P_branch_top = max(P_max, Pb) * 1.25

    # Reference undersaturated behaviour, taken at the REAL bubble point
    # (Rsi, Pb) where the Vasquez-Beggs undersaturated correlations are
    # valid, expressed as ratios vs (P - Psat). Every node's branch is
    # Bo_sat*rBo(dP), mu_sat*rMu(dP) — the same shifting ECLIPSE applies
    # when a branch is omitted. Evaluating V-B directly at a low Psat
    # (e.g. 15 psia) is outside its range and blows up (mu x18, Bo < 1).
    Bo_pb = oil.formation_volume_factor(Pb, Rsi, saturated=True)
    mu_pb = oil.viscosity(Pb, Rsi, Pb, saturated=True)

    def _ratios(dP):
        P = Pb + dP
        return (oil.formation_volume_factor(P, Rsi, saturated=False, Pb=Pb)
                / Bo_pb,
                oil.viscosity(P, Rsi, Pb, saturated=False) / mu_pb)

    nodes, last_rs = [], -1.0
    for P in P_grid:
        if abs(P - Pb) < 1e-3:
            Rs = Rsi
        else:
            Rs = oil.solution_gor(P)
            if P < Pb:
                Rs = min(Rs, Rsi * P / Pb if Rs <= 0 else Rs)
        if not np.isfinite(Rs) or Rs <= last_rs + 1e-6:
            continue                      # Rs must strictly increase
        Psat = P
        Bo_sat = oil.formation_volume_factor(Psat, Rs, saturated=True)
        mu_sat = oil.viscosity(Psat, Rs, Psat, saturated=True)
        P_u = list(np.linspace(Psat, P_branch_top, int(n_under) + 1)[1:])
        rat = [_ratios(p - Psat) for p in P_u]
        Bo_u = [Bo_sat * r[0] for r in rat]
        mu_u = [mu_sat * r[1] for r in rat]
        nodes.append({"Rs": Rs, "Psat": Psat, "Bo_sat": Bo_sat,
                      "mu_sat": mu_sat, "P_u": P_u, "Bo_u": Bo_u,
                      "mu_u": mu_u})
        last_rs = Rs
    return nodes


def build_pvto(df, Pb, oil, Rsi, P_max, P_min=14.7, n_sat=20, n_under=6,
               extend_above_pb=True):
    """PVTO keyword (FIELD units: Rs Mscf/STB, P psia, Bo rb/STB, cP).

    Layout per record (one saturated Rs node):
        Rs  Psat  Bo_sat  mu_sat
              P1    Bo1     mu1       <- undersaturated, same Rs
              P2    Bo2     mu2  /
    and a final '/' closes the table. See pvto_nodes() for the span.
    `df` is kept for call-site compatibility and not used.
    """
    nodes = pvto_nodes(oil, Rsi, Pb, P_max, P_min=P_min, n_sat=n_sat,
                       n_under=n_under, extend_above_pb=extend_above_pb)
    lines = ["PVTO",
             f"-- {len(nodes)} saturated Rs nodes, each with an "
             f"undersaturated branch",
             "-- Rs         Psat        Bo          Muo",
             "-- Mscf/STB   psia        rb/STB      cP"]
    for nd in nodes:
        lines.append(f"  {nd['Rs'] / 1000.0:9.5f}   {nd['Psat']:9.2f}   "
                     f"{nd['Bo_sat']:9.5f}   {nd['mu_sat']:9.5f}")
        for j, (p, bo, mu) in enumerate(zip(nd["P_u"], nd["Bo_u"],
                                             nd["mu_u"])):
            end = "  /" if j == len(nd["P_u"]) - 1 else ""
            lines.append(f"              {p:9.2f}   {bo:9.5f}   {mu:9.5f}{end}")
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


def _gas_props_at_rv(wetgas, P, Rv):
    """Bg (rb/Mscf of SURFACE dry gas) and mu_g (cP) for gas carrying
    Rv STB/scf of vaporised oil, at pressure P (psia).

    The in-situ gas is separator gas + Rv of vaporised condensate. Its
    gravity per scf of separator gas (McCain recombination):
        gamma = (gamma_sep + 4584*gamma_o*Rv) / (1 + Veq*Rv)
    Z and mu come from that mixture; Bg is converted to the surface-gas
    basis ECLIPSE expects: Bg_dry = Bg_mix * (1 + Veq*Rv).
    """
    from correlations import GasCorrelations
    Veq = getattr(wetgas, "Veq", 0.0)
    g_sep = getattr(wetgas, "gamma_g_sep", 0.7)
    g_o = getattr(wetgas, "gamma_cond", 0.75)
    gamma = (g_sep + 4584.0 * g_o * Rv) / (1.0 + Veq * Rv)
    gas = GasCorrelations(gamma, wetgas.T, N2=wetgas.N2, CO2=wetgas.CO2,
                          H2S=wetgas.H2S, z_corr=wetgas.z_corr,
                          mu_corr=wetgas.mu_corr)
    Z = gas.z_factor(P)
    Bg = gas.formation_volume_factor(P, Z) * (1.0 + Veq * Rv) * 1000.0
    return Bg, gas.viscosity(P, Z)


def pvtg_nodes(pressures, wetgas, n_under=4):
    """PVTG node structure (FIELD units), one dict per pressure node:
        {"P": psia, "Rv": [STB/Mscf, decreasing, first = saturated, last 0],
         "Bg": [rb/Mscf], "mu": [cP]}

    Saturated Rv: below the dew point it follows the wet-gas model. Above
    it the saturated line keeps rising, so the reservoir gas (Rv = CGR)
    sits on an UNDERSATURATED branch and the dew point falls exactly where
    Rv_sat = CGR. The extension slope is the largest (<= the slope below
    Pdew) that keeps saturated Bg strictly decreasing with pressure —
    rich gas extrapolated too steeply would imply negative compressibility.
    Rv = CGR is a branch point on every node above the dew point.
    """
    Rv_max = wetgas.Rv_max
    Pdew = wetgas.Pdew
    const = getattr(wetgas, "rv_corr", "") == "Constant"
    P_nodes = sorted(set(float(p) for p in pressures) | (
        {float(Pdew)} if min(pressures) < Pdew < max(pressures) else set()))
    P_nodes = [p for p in P_nodes if p >= 14.7]   # dew point is a node

    def rv_sat_at(P, slope):
        if const:
            return Rv_max
        if P < Pdew:
            return wetgas.rv(P)
        return Rv_max * (1.0 + slope * (P - Pdew) / Pdew)

    def sat_bg(slope):
        return [_gas_props_at_rv(wetgas, P, rv_sat_at(P, slope))[0]
                for P in P_nodes]

    def decreasing(v):
        return all(x > y for x, y in zip(v, v[1:]))

    slope = 0.95
    if not const and any(P >= Pdew for P in P_nodes) \
            and not decreasing(sat_bg(slope)):
        lo, hi = 0.0, slope              # lo = flat, always feasible above Pdew
        for _ in range(25):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if decreasing(sat_bg(mid)) else (lo, mid)
        slope = lo

    nodes = []
    for P in P_nodes:
        rv_sat = rv_sat_at(P, slope)
        grid = list(np.linspace(rv_sat, 0.0, int(n_under) + 1))
        if 0.0 < Rv_max < rv_sat:
            # snap the nearest interior grid point to the reservoir-gas Rv
            k = min(range(1, len(grid) - 1),
                    key=lambda i: abs(grid[i] - Rv_max), default=None)
            if k is not None:
                grid[k] = Rv_max
            else:
                grid.insert(1, Rv_max)
        rvs = sorted(set(round(r, 12) for r in grid), reverse=True)
        props = [_gas_props_at_rv(wetgas, P, rv) for rv in rvs]
        nodes.append({"P": P, "Rv": [rv * 1000.0 for rv in rvs],
                      "Bg": [p[0] for p in props],
                      "mu": [p[1] for p in props]})
    return nodes


def build_pvtg(pressures, wetgas, n_under=4):
    """PVTG keyword (FIELD units: P psia, Rv STB/Mscf, Bg rb/Mscf, cP).

    One record per pressure node:
        P   Rv_sat  Bg  mu        <- saturated gas at this pressure
            Rv_2    Bg  mu        <- undersaturated (leaner) gas
            0.0     Bg  mu  /     <- dry gas
    and a final '/' closes the table.
    """
    nodes = pvtg_nodes(pressures, wetgas, n_under=n_under)
    lines = ["PVTG",
             f"-- {len(nodes)} pressure nodes, each with an undersaturated "
             f"branch down to dry gas (Rv = 0)",
             "-- P          Rv            Bg           Mug",
             "-- psia       STB/Mscf      rb/Mscf      cP"]
    for nd in nodes:
        for j, (rv, bg, mu) in enumerate(zip(nd["Rv"], nd["Bg"], nd["mu"])):
            end = "  /" if j == len(nd["Rv"]) - 1 else ""
            lead = f"  {nd['P']:9.2f}" if j == 0 else " " * 11
            lines.append(f"{lead}   {rv:11.6f}   {bg:10.5f}   {mu:9.6f}{end}")
    lines.append("/")
    return "\n".join(lines) + "\n"


def build_pvto_from_compositional(rows, Pb, P_max):
    """PVTO from EOS black-oil rows (FIELD units).

    `rows` is the list returned by black_oil_table_from_composition (oil):
    saturated rows at P <= Pb (Rs, Bo, mu_o from the DLE) and rows above
    Pb at constant Rsi. Every saturated Rs node gets an undersaturated
    branch: the EOS rows above Pb define the reference behaviour as
    ratios Bo/Bo_pb and mu/mu_pb vs (P - Pb), applied from each node's
    own Psat (the same shifting ECLIPSE uses for omitted branches).
    """
    rows = sorted(rows, key=lambda r: r["P"])
    sat = [r for r in rows if r["P"] <= Pb + 1.0]
    und = [r for r in rows if r["P"] > Pb + 1.0]
    # Rs must strictly increase node to node
    nodes, last = [], -1.0
    for r in sat:
        if r["Rs"] > last + 1e-6:
            nodes.append(r)
            last = r["Rs"]
    lines = ["PVTO",
             f"-- EOS-derived; {len(nodes)} saturated Rs nodes, each with an "
             f"undersaturated branch",
             "-- Rs         Psat        Bo          Muo",
             "-- Mscf/STB   psia        rb/STB      cP"]
    if not nodes:
        lines.append("/")
        return "\n".join(lines) + "\n"
    ref = nodes[-1]
    if und:
        dP = [r["P"] - ref["P"] for r in und]
        rBo = [r["Bo"] / ref["Bo"] for r in und]
        rMu = [r["mu_o"] / ref["mu_o"] for r in und]
    else:
        # No EOS data above Pb (table top <= Pb): a minimal branch with a
        # typical oil compressibility (1e-5 /psi) so ECLIPSE has the
        # mandatory undersaturated data. Widen the pressure range to
        # replace it with EOS values.
        lines.insert(2, "-- NOTE: no EOS points above Pb - undersaturated "
                        "branch uses co = 1e-5 /psi")
        dP = [500.0, 1000.0, 2000.0]
        rBo = [float(np.exp(-1e-5 * d)) for d in dP]
        rMu = [1.0 + 5e-5 * d for d in dP]
    for nd in nodes:
        lines.append(f"  {nd['Rs'] / 1000.0:9.5f}   {nd['P']:9.2f}   "
                     f"{nd['Bo']:9.5f}   {nd['mu_o']:9.5f}")
        for j, (d, rb, rm) in enumerate(zip(dP, rBo, rMu)):
            end = "  /" if j == len(dP) - 1 else ""
            lines.append(f"              {nd['P'] + d:9.2f}   "
                         f"{nd['Bo'] * rb:9.5f}   {nd['mu_o'] * rm:9.5f}{end}")
    lines.append("/")
    return "\n".join(lines) + "\n"


def build_pvtg_from_compositional(rows, Pdew, n_under=4):
    """PVTG from EOS gas-condensate rows (FIELD units).

    Each pressure node: the saturated gas (Rv, Bg, mu from the EOS vapour,
    Bg per Mscf of surface dry gas) and an undersaturated branch down to
    dry gas (Rv = 0: the surface gas re-flashed at reservoir P, T).
    Interior branch points interpolate linearly in Rv between the two
    computed end points.
    """
    lines = ["PVTG",
             "-- EOS-derived; each node has a branch to dry gas (Rv = 0)",
             "-- P          Rv            Bg           Mug",
             "-- psia       STB/Mscf      rb/Mscf      cP"]
    for r in sorted(rows, key=lambda q: q["P"]):
        P, rv, bg, mu = r["P"], r["Rv"], r["Bg"], r["mu_g"]
        bg0, mu0 = r.get("Bg_dry", bg), r.get("mu_g_dry", mu)
        if rv <= 0:
            continue
        for j, f in enumerate(np.linspace(1.0, 0.0, int(n_under) + 1)):
            end = "  /" if j == int(n_under) else ""
            lead = f"  {P:9.2f}" if j == 0 else " " * 11
            lines.append(f"{lead}   {rv * f:11.6f}   "
                         f"{bg0 + f * (bg - bg0):10.5f}   "
                         f"{mu0 + f * (mu - mu0):9.6f}{end}")
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
    from units import to_ascii
    return to_ascii("".join(parts))


# ---------- METRIC conversion of pre-built FIELD keyword bodies ----------
# These walk through the lines and convert numerical columns in-place.

def _convert_pvto_to_metric(field_text):
    """Convert a PVTO block from FIELD (psia, scf/STB, rb/STB, cP)
    to METRIC (bara, Sm3/Sm3, rm3/Sm3, cP)."""
    PSI_PER_BAR = 14.50377
    SCF_PER_SM3 = 5.6146
    out_lines = []
    seen_data = False
    for line in field_text.splitlines():
        s = line.strip()
        if s.startswith("PVTO"):
            out_lines.append(line)
            out_lines.append("-- Rs        Psat       Bo        Muo")
            out_lines.append("-- Sm3/Sm3   bara       rm3/Sm3   cP")
            continue
        if s == "/" or not s:
            out_lines.append(line); continue
        # comment lines: drop the original FIELD headers (before any data),
        # keep inline comments that come after data has started
        if s.startswith("--"):
            if seen_data:
                out_lines.append(line)
            continue
        has_slash = s.endswith("/")
        core = s.rstrip("/").strip()
        toks = core.split()
        try: vals = [float(t) for t in toks]
        except ValueError:
            out_lines.append(line); continue
        seen_data = True
        if len(vals) == 4:
            Rs, P, Bo, Mu = vals
            Rs_si = Rs * 1000.0 / SCF_PER_SM3
            new = f"  {Rs_si:9.4f}   {P/PSI_PER_BAR:9.3f}   {Bo:9.5f}   {Mu:9.5f}"
        elif len(vals) == 3:
            P, Bo, Mu = vals
            new = f"             {P/PSI_PER_BAR:9.3f}   {Bo:9.5f}   {Mu:9.5f}"
        else:
            out_lines.append(line); continue
        if has_slash: new += "  /"
        out_lines.append(new)
    return "\n".join(out_lines) + ("\n" if not field_text.endswith("\n") else "")


def _convert_pvdg_to_metric(field_text):
    """Convert PVDG: P [psia]->bara, Bg [rb/Mscf]->rm3/Sm3."""
    PSI_PER_BAR = 14.50377
    out = []
    seen_data = False
    for line in field_text.splitlines():
        s = line.strip()
        if s.startswith("PVDG"):
            out.append(line)
            out.append("-- P          Bg            Mug")
            out.append("-- bara       rm3/Sm3       cP")
            continue
        if not s or s == "/":
            out.append(line); continue
        if s.startswith("--"):
            if seen_data:
                out.append(line)
            continue
        toks = s.rstrip("/").split()
        try: P, Bg, Mu = [float(t) for t in toks]
        except ValueError: out.append(line); continue
        seen_data = True
        # Field PVDG Bg is rb/Mscf -> Metric rm3/Sm3:
        # rb -> rm3: × 0.158987;  Mscf -> Sm3: × 28.3168
        Bg_si = Bg * 0.158987 / 28.3168
        out.append(f"  {P/PSI_PER_BAR:9.3f}   {Bg_si:11.7f}   {Mu:8.5f}")
    return "\n".join(out) + "\n"


def _convert_pvtg_to_metric(field_text):
    """Convert PVTG: P, Rv [STB/Mscf]->Sm3/Sm3, Bg [rb/Mscf]->rm3/Sm3."""
    PSI_PER_BAR = 14.50377
    out = []
    seen_data = False
    for line in field_text.splitlines():
        s = line.strip()
        if s.startswith("PVTG"):
            out.append(line)
            out.append("-- P         Rv           Bg           Mug")
            out.append("-- bara      Sm3/Sm3      rm3/Sm3      cP")
            continue
        if not s or s == "/":
            out.append(line); continue
        if s.startswith("--"):
            if seen_data:
                out.append(line)
            continue
        has_slash = s.endswith("/")
        core = s.rstrip("/").strip()
        toks = core.split()
        try: vals = [float(t) for t in toks]
        except ValueError: out.append(line); continue
        seen_data = True
        if len(vals) == 4:
            P, Rv, Bg, Mu = vals
            Rv_si = Rv * 0.158987 / 28.3168
            Bg_si = Bg * 0.158987 / 28.3168
            new = f"  {P/PSI_PER_BAR:9.3f}  {Rv_si:13.6e}  {Bg_si:13.6e}  {Mu:9.6f}"
        elif len(vals) == 3:
            Rv, Bg, Mu = vals
            Rv_si = Rv * 0.158987 / 28.3168
            Bg_si = Bg * 0.158987 / 28.3168
            new = f"             {Rv_si:13.6e}  {Bg_si:13.6e}  {Mu:9.6f}"
        else:
            out.append(line); continue
        if has_slash: new += "  /"
        out.append(new)
    return "\n".join(out) + "\n"


def _convert_pvtw_to_metric(field_text):
    """Convert PVTW: Pref [psia]->bara, Cw [1/psi]->1/bar."""
    PSI_PER_BAR = 14.50377
    out = []
    seen_data = False
    for line in field_text.splitlines():
        s = line.strip()
        if s.startswith("PVTW"):
            out.append(line)
            out.append("-- Pref     Bwref     Cw           Muw       Viscosibility")
            out.append("-- bara     rm3/Sm3   1/bar        cP        1/bar")
            continue
        if not s:
            out.append(line); continue
        if s.startswith("--"):
            if seen_data:
                out.append(line)
            continue
        has_slash = s.endswith("/")
        core = s.rstrip("/").strip()
        toks = core.split()
        try: vals = [float(t) for t in toks]
        except ValueError: out.append(line); continue
        seen_data = True
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
    seen_data = False
    for line in field_text.splitlines():
        s = line.strip()
        if s.startswith("DENSITY"):
            out.append(line)
            out.append("-- Oil       Water     Gas      (kg/m3)")
            continue
        if not s:
            out.append(line); continue
        if s.startswith("--"):
            if seen_data:
                out.append(line)
            continue
        has_slash = s.endswith("/")
        core = s.rstrip("/").strip()
        toks = core.split()
        try: vals = [float(t) for t in toks]
        except ValueError: out.append(line); continue
        seen_data = True
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
        lines.append(f"  {d:9.2f}   {rv:13.6e}")
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
