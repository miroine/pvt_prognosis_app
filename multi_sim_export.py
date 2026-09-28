"""
PVT Studio — Multi-simulator PVT export
========================================

Generate PVT tables in the keyword syntax of several reservoir
simulators from the same field-unit DataFrame that feeds the ECLIPSE
builder. Each generator returns a ready-to-include text block (or a
CSV / JSON string) with a header comment carrying the case metadata.

Supported outputs
-----------------
* **ECLIPSE** — delegated to eclipse_export.py (this module doesn't
  duplicate that work; it wraps a bundle download).
* **tNavigator (RFD)** — the Eclipse-compatible dialect; the same
  keywords work, we add a compatibility header.
* **CMG IMEX** — *PVT / *DENSITY / *ROCKFLUID keywords with an
  asterisk prefix and CMG's column ordering.
* **CMG GEM** — compositional-oriented, but IMEX-style PVT tables
  also load into GEM's black-oil approximation.
* **OPM Flow (Free ECLIPSE clone)** — same keywords as ECLIPSE 100;
  we add a comment stating OPM compatibility.
* **CSV** — a plain 6-column dump (P, Rs, Bo, μo, Bg, μg) that
  Excel, pandas, or R will read directly. Also a companion
  metadata.json.
* **JSON** — machine-readable full case (inputs + tables) — useful
  for scripts, PowerBI, or handing to another PVT tool.

All builders take the same `case` dict:

    case = {
        "kind"       : "oil" | "drygas" | "wetgas" | "water",
        "fluid_name" : "Case-A",
        "notes"      : "One-line comment for the header",
        "units"      : "FIELD" | "METRIC",
        "df_field"   : pandas.DataFrame in FIELD units (P psia, Rs
                       scf/STB, Bo rb/STB, μo cP, Bg rb/Mscf, μg cP,
                       Rv STB/Mscf, etc. — whichever columns apply),
        "api"        : float, "gas_sg": float, "water_sg": float,
        "Pb_psia"    : float,   # for oil
        "Pdew_psia"  : float,   # for wet gas
        "Pref_psia"  : float,   # water reference
        "Bw"         : float, "Cw": float, "muw": float,
        "T_res_F"    : float,
    }

Only the fields relevant to `kind` need to be populated.
"""

APP_VERSION = "1.4.1"   # must match pvt_app.py (deployment check)

import io
import json
import datetime as _dt

# Local conversion constants (kept here so this module is
# self-contained and doesn't add a new import cycle).
_PSIA_TO_BARA = 1.0 / 14.5038
_SCF_STB_TO_SM3_SM3 = 1.0 / 5.6146
_RB_STB_TO_RM3_SM3 = 1.0        # ratio unchanged (same-name/different-basis cancels)
_RB_MSCF_TO_RM3_SM3 = 0.158987 / 28.3168   # 1 bbl/Mscf -> m3/Sm3
_FT3_TO_M3 = 0.028316846592
_LB_TO_KG = 0.453592


# ----------------------------------------------------------------------
# Header block — used by every text-format builder
# ----------------------------------------------------------------------
def _header(case, tag, comment_prefix="--"):
    """Return a multi-line header block using the given comment prefix."""
    now = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"{comment_prefix} =====================================================",
        f"{comment_prefix} {tag} PVT export — PVT Studio",
        f"{comment_prefix} Case          : {case.get('fluid_name', 'unnamed')}",
        f"{comment_prefix} Kind          : {case.get('kind', '?')}",
        f"{comment_prefix} Units         : {case.get('units', 'FIELD')}",
        f"{comment_prefix} Reservoir T   : {case.get('T_res_F', '?')} °F",
        f"{comment_prefix} Oil API       : {case.get('api', '?')}",
        f"{comment_prefix} Gas SG (air=1): {case.get('gas_sg', '?')}",
        f"{comment_prefix} Generated at  : {now}",
    ]
    if case.get("notes"):
        lines.append(f"{comment_prefix} Notes         : {case['notes']}")
    lines.append(
        f"{comment_prefix} =====================================================")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Normalisation — one place that understands every branch's column names
# ----------------------------------------------------------------------
# Branches build df_field with slightly different headers ("μo (cp)" vs
# "μo (cP)", dry gas stores "Bg (rb/scf)" while wet gas uses rb/Mscf) and
# different kind tags ("dry_gas", "gas-dry", "wetgas" ...). Every writer
# goes through _normalise() so a naming difference can never silently
# drop a column or crash a writer again.
_BBL_PER_MSCF_TO_M3_PER_SM3 = 0.158987 / 28.3168   # Bg, Rv
_LBFT3_TO_KGM3 = _LB_TO_KG / _FT3_TO_M3
_PSIA_TO_KPA = 6.894757


def _kind(case):
    k = str(case.get("kind", "")).lower().replace("-", "_")
    if "oil" in k:
        return "oil"
    if "wet" in k:
        return "wetgas"
    if "gas" in k:
        return "drygas"
    return k


def _find(df, *names):
    """Return the first column whose name matches (case-insensitive)."""
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _normalise(case):
    """Return dict of field-unit lists: P, Rs, Bo, muo, Bg (rb/Mscf),
    mug, Z, Rv (STB/Mscf) — only keys present in the source table."""
    df = case["df_field"]
    out = {}
    c = _find(df, "P (psia)", "P")
    if c is not None:
        out["P"] = [float(v) for v in df[c]]
    for key, names in (("Rs", ("Rs (scf/STB)",)),
                       ("Bo", ("Bo (rb/STB)",)),
                       ("muo", ("μo (cp)", "mu_o (cp)", "μo")),
                       ("mug", ("μg (cp)", "mu_g (cp)", "μg")),
                       ("Z", ("Z",)),
                       ("Rv", ("Rv (STB/Mscf)",))):
        c = _find(df, *names)
        if c is not None:
            out[key] = [float(v) for v in df[c]]
    c = _find(df, "Bg (rb/Mscf)")
    if c is not None:
        out["Bg"] = [float(v) for v in df[c]]
    else:
        c = _find(df, "Bg (rb/scf)")
        if c is not None:
            out["Bg"] = [float(v) * 1000.0 for v in df[c]]
    return out


def _conv(key, v, units, p_unit="bar"):
    """Field value -> target unit system for one normalised quantity."""
    if units != "METRIC":
        return v
    if key == "P":
        return v * (_PSIA_TO_KPA if p_unit == "kPa" else _PSIA_TO_BARA)
    if key == "Rs":
        return v * _SCF_STB_TO_SM3_SM3
    if key in ("Bg", "Rv"):
        return v * _BBL_PER_MSCF_TO_M3_PER_SM3
    return v          # Bo, viscosities, Z are unit-free here


def _densities(case, units):
    api = case.get("api", 35.0)
    rho_o = 141.5 / (api + 131.5) * 62.428
    rho_g = 0.0764 * case.get("gas_sg", 0.75)
    rho_w = 62.4 * case.get("water_sg", 1.02)
    f = _LBFT3_TO_KGM3 if units == "METRIC" else 1.0
    return rho_o * f, rho_g * f, rho_w * f


# ----------------------------------------------------------------------
# CMG IMEX
# ----------------------------------------------------------------------
def build_cmg_imex(case):
    """CMG IMEX PVT keywords.

    FIELD -> *INUNIT *FIELD (psia, scf/STB, rb/Mscf).
    METRIC -> *INUNIT *SI — CMG's SI system uses kPa for pressure (there
    is no native bar system in CMG), m3/m3 for Rs and Bg, kg/m3.
    """
    units = case.get("units", "FIELD").upper()
    kind = _kind(case)
    d = _normalise(case)
    out = [_header(case, "CMG IMEX", comment_prefix="**")]
    out.append("*INUNIT *SI" if units == "METRIC" else "*INUNIT *FIELD")
    p_lbl = "kPa" if units == "METRIC" else "psia"

    if kind == "oil":
        Pb = case.get("Pb_psia")
        if Pb:
            out.append(f"** Bubble point : {_conv('P', Pb, units, 'kPa'):.2f} {p_lbl}")
        cols = [k for k in ("P", "Rs", "Bo", "muo") if k in d]
        out.append("*PVT *BG 1")
    else:
        cols = [k for k in ("P", "Bg", "mug", "Z", "Rv") if k in d]
        out.append("*PVT *GAS 1")
    names = {"P": "p", "Rs": "rs", "Bo": "bo", "muo": "viso",
             "Bg": "bg", "mug": "visg", "Z": "z", "Rv": "rv"}
    out.append("**   " + "".join(f"{names[k]:>14s}" for k in cols))
    for i in range(len(d.get("P", []))):
        out.append("     " + "".join(
            f"{_conv(k, d[k][i], units, 'kPa'):14.6g}" for k in cols))
    out.append("")

    rho_o, rho_g, rho_w = _densities(case, units)
    out.append(f"*DENSITY *OIL   {rho_o:.4f}")
    out.append(f"*DENSITY *GAS   {rho_g:.4f}")
    out.append(f"*DENSITY *WATER {rho_w:.4f}")

    if case.get("Bw"):
        Pref = case.get("Pref_psia", 3000.0)
        out.append("")
        out.append(f"*BWI    {case['Bw']:.5f}   ** Bw at reference")
        out.append(f"*PBW    {_conv('P', Pref, units, 'kPa'):.2f}   ** Pref ({p_lbl})")
        cw = case["Cw"] if units != "METRIC" else case["Cw"] / _PSIA_TO_KPA
        out.append(f"*CW     {cw:.5e}  ** Cw (1/{'kPa' if units == 'METRIC' else 'psi'})")
        out.append(f"*VWI    {case.get('muw', 0.5):.5f}   ** μw (cP)")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------
# CMG GEM (compositional; the black-oil PVT still loads via *MODEL *BLACKOIL)
# ----------------------------------------------------------------------
def build_cmg_gem(case):
    """CMG GEM header block for a black-oil case.

    GEM is a compositional simulator; when running a black-oil deck it
    reads the same PVT table but wrapped in *MODEL *BLACKOIL. We emit
    that wrapper around the IMEX body so the same file loads in either
    engine.
    """
    body = build_cmg_imex(case)
    header = _header(case, "CMG GEM (black-oil mode)",
                      comment_prefix="**")
    return (header
             + "*MODEL *BLACKOIL\n\n"
             + body)


# ----------------------------------------------------------------------
# tNavigator (RFD) — Eclipse-compatible dialect
# ----------------------------------------------------------------------
def build_tnavigator(case, eclipse_body):
    """tNavigator reads ECLIPSE keywords natively.

    We wrap the ECLIPSE body in a compatibility header noting the
    dialect. This mirrors what commercial vendors ship as
    'tNavigator-ready' decks.
    """
    header = _header(case, "tNavigator (RFD, Eclipse-compatible)",
                      comment_prefix="--")
    footer = ("-- Loads directly in tNavigator. If you use OPM Flow "
               "the same body works.\n")
    return header + eclipse_body + "\n" + footer


# ----------------------------------------------------------------------
# OPM Flow — Eclipse-compatible open source
# ----------------------------------------------------------------------
def build_opm(case, eclipse_body):
    """OPM Flow — reads Eclipse E100 keywords."""
    header = _header(case, "OPM Flow (open-source Eclipse-compatible)",
                      comment_prefix="--")
    return header + eclipse_body


# ----------------------------------------------------------------------
# Halliburton Nexus — PVT_TABLE keyword
# ----------------------------------------------------------------------
def build_nexus(case):
    """Halliburton Nexus black-oil / dry-gas PVT_TABLE.

    FIELD  -> UNITS ENGLISH (psia, scf/STB, rb/Mscf).
    METRIC -> UNITS METBAR (bar, Sm3/Sm3, rm3/Sm3). If your deck uses
    Nexus METRIC (kPa), multiply the pressure column by 100.
    Comments start with `!`.
    """
    units = case.get("units", "FIELD").upper()
    kind = _kind(case)
    d = _normalise(case)
    lines = [_header(case, "Halliburton Nexus", comment_prefix="!")]
    lines.append("UNITS " + ("METBAR" if units == "METRIC" else "ENGLISH"))
    lines.append("")
    n = len(d.get("P", []))
    if kind == "oil":
        Pb = case.get("Pb_psia", 0.0) or 0.0
        lines.append("PVT_TABLE 1  TYPE BLACKOIL")
        lines.append("PROPS  RS  BO  VISCO")
        lines.append(f"PSATURATION {_conv('P', Pb, units):.3f}")
        lines.append("TABLE")
        lines.append("!     PRES          RS             BO           MU_O")
        for i in range(n):
            lines.append(
                f"  {_conv('P', d['P'][i], units):12.4f}"
                f"  {_conv('Rs', d.get('Rs', [0]*n)[i], units):12.5g}"
                f"  {d.get('Bo', [0]*n)[i]:10.6f}"
                f"  {d.get('muo', [0]*n)[i]:10.6f}")
        lines.append("ENDTABLE")
    else:
        lines.append("PVT_TABLE 1  TYPE DRYGAS")
        lines.append("PROPS  BG  VISCG")
        lines.append("TABLE")
        lines.append("!     PRES          BG            MU_G")
        for i in range(n):
            lines.append(
                f"  {_conv('P', d['P'][i], units):12.4f}"
                f"  {_conv('Bg', d.get('Bg', [0]*n)[i], units):14.6e}"
                f"  {d.get('mug', [0]*n)[i]:12.6f}")
        lines.append("ENDTABLE")

    rho_o, rho_g, rho_w = _densities(case, units)
    lines.append("")
    lines.append("!  Surface densities (lb/ft3 ENGLISH, kg/m3 METBAR)")
    lines.append(f"DENSITY_OIL   {rho_o:.4f}")
    lines.append(f"DENSITY_GAS   {rho_g:.4f}")
    lines.append(f"DENSITY_WATER {rho_w:.4f}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Schlumberger INTERSECT (IX) — AFI-style scoped block
# ----------------------------------------------------------------------
def build_intersect(case):
    """SLB INTERSECT (IX) black-oil PVT block (AFI style).
    METRIC writes bar, Sm3/Sm3, rm3/Sm3, kg/m3. Comments start `//`."""
    units = case.get("units", "FIELD").upper()
    kind = _kind(case)
    d = _normalise(case)
    p_unit = "bara" if units == "METRIC" else "psia"
    lines = [_header(case, "Schlumberger INTERSECT (IX)",
                      comment_prefix="//"), ""]
    lines.append("BLACK_OIL_MODEL PVT1 {")
    lines.append(f'    units = "{units}"')
    rho_o, rho_g, rho_w = _densities(case, units)
    lines.append(f"    oil_stock_tank_density = {rho_o:.4f}")
    lines.append(f"    gas_standard_density   = {rho_g:.4f}")
    lines.append(f"    water_standard_density = {rho_w:.4f}")

    def _arr(name, key, fmt):
        lines.append(f"        {name} = [")
        for v in d.get(key, []):
            lines.append("            " + fmt % _conv(key, v, units))
        lines.append("        ]")

    lines.append("")
    if kind == "oil":
        lines.append("    saturated_pvt_table = {")
        lines.append(f"        // pressure ({p_unit}), Rs, Bo, oil viscosity (cP)")
        _arr("pressure", "P", "%.4f")
        _arr("Rs", "Rs", "%.6g")
        _arr("Bo", "Bo", "%.6f")
        _arr("oil_viscosity", "muo", "%.6f")
    else:
        lines.append("    dry_gas_pvt_table = {")
        lines.append(f"        // pressure ({p_unit}), Bg, gas viscosity (cP)")
        _arr("pressure", "P", "%.4f")
        _arr("Bg", "Bg", "%.6e")
        _arr("gas_viscosity", "mug", "%.6f")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# CSV — plain tabular dump
# ----------------------------------------------------------------------
def build_csv(case):
    """One CSV per PVT table. Columns match the DataFrame's field-unit
    columns; a small `_meta` block is prepended as commented lines."""
    df = case["df_field"]
    buf = io.StringIO()
    buf.write(f"# PVT Studio CSV export — {case.get('fluid_name', 'unnamed')}\n")
    buf.write(f"# kind={case.get('kind')}, units=FIELD, "
              f"T_res={case.get('T_res_F')} F, "
              f"API={case.get('api')}, SG_g={case.get('gas_sg')}\n")
    if case.get("notes"):
        buf.write(f"# notes: {case['notes']}\n")
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ----------------------------------------------------------------------
# JSON — full-case machine-readable
# ----------------------------------------------------------------------
def build_json(case):
    """Full case in JSON. The DataFrame is emitted as a list of records
    so the file loads directly into pandas.read_json."""
    payload = {
        "_format": "pvt_studio_pvt_case",
        "_version": 1,
        "case": {k: v for k, v in case.items()
                 if k not in ("df_field",)
                 and not hasattr(v, "to_dict")},
        "table_field_units": (case["df_field"].to_dict(orient="records")
                              if "df_field" in case else []),
    }
    return json.dumps(payload, indent=2, default=str)


# ----------------------------------------------------------------------
# ZIP bundle — one file per format, downloadable together
# ----------------------------------------------------------------------
def build_bundle(case, eclipse_body):
    """Zip bundle with every generator's output. Returns bytes."""
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        name = case.get("fluid_name", "case").replace(" ", "_")
        z.writestr(f"{name}_ECLIPSE.INC", eclipse_body)
        z.writestr(f"{name}_CMG_IMEX.dat", build_cmg_imex(case))
        z.writestr(f"{name}_CMG_GEM.dat", build_cmg_gem(case))
        z.writestr(f"{name}_tNavigator.INC",
                    build_tnavigator(case, eclipse_body))
        z.writestr(f"{name}_OPM.INC", build_opm(case, eclipse_body))
        z.writestr(f"{name}_Nexus.dat", build_nexus(case))
        z.writestr(f"{name}_INTERSECT.afi", build_intersect(case))
        z.writestr(f"{name}_table.csv", build_csv(case))
        z.writestr(f"{name}_case.json", build_json(case))
        # Simple README
        readme = (f"PVT Studio — export bundle for '{name}'\n"
                   f"Generated {_dt.datetime.utcnow().isoformat(timespec='seconds')} UTC\n\n"
                   f"Files:\n"
                   f"  {name}_ECLIPSE.INC     — ECLIPSE E100 keywords\n"
                   f"  {name}_CMG_IMEX.dat    — CMG IMEX (*PVT, *DENSITY)\n"
                   f"  {name}_CMG_GEM.dat     — CMG GEM (black-oil mode)\n"
                   f"  {name}_tNavigator.INC  — tNavigator (Eclipse-compatible)\n"
                   f"  {name}_OPM.INC         — OPM Flow (open-source)\n"
                   f"  {name}_Nexus.dat       — Halliburton Nexus PVT_TABLE\n"
                   f"  {name}_INTERSECT.afi   — Schlumberger INTERSECT (IX)\n"
                   f"  {name}_table.csv       — plain CSV of the PVT table\n"
                   f"  {name}_case.json       — full machine-readable case\n"
                   f"\nAll simulator files are field-unit unless the case "
                   f"specifies METRIC.\n")
        z.writestr("README.txt", readme)
    return buf.getvalue()
