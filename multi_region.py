"""
Multi-region PVT export for ECLIPSE.

ECLIPSE supports multiple PVT regions via the PVTNUM keyword. Each region
gets its own table under the same PVT keyword, separated by `/`. Order matters:
region 1 is the first table, region 2 the second, etc.

Example PVTO with 2 regions:
    PVTO
    -- Region 1
    ...table 1 rows...
    /
    -- Region 2
    ...table 2 rows...
    /
    /

Note the double trailing `/`: one closes the last region, one closes the keyword.
"""


def build_multi_region_pvto(region_texts):
    """
    Stack multiple PVTO table bodies into one PVTO block.

    Args:
        region_texts: list of strings, each a complete PVTO block produced by
                       build_pvto() or build_pvto_from_compositional(). The full
                       block including the keyword line and trailing `/` per region.

    Returns single PVTO block containing all regions.
    """
    # Strip the 'PVTO' header and the final '/' from each region body,
    # keep only the rows + per-region terminator.
    out = ["PVTO", "-- Multi-region PVT (PVTNUM)",
           "-- Rs       Psat       Bo        Muo",
           "-- Mscf/STB  psia       rb/STB    cP"]
    for i, txt in enumerate(region_texts):
        out.append(f"-- =========== Region {i+1} ===========")
        body = _strip_keyword_and_outer_terminator(txt, "PVTO")
        out.append(body.rstrip())
    out.append("/")  # closes the keyword (already-stripped regions end in their own `/`)
    return "\n".join(out) + "\n"


def build_multi_region_pvdg(region_texts):
    """Stack multiple PVDG region tables."""
    out = ["PVDG", "-- Multi-region dry-gas PVT (PVTNUM)",
           "-- P         Bg          Mug",
           "-- psia      rb/Mscf     cP"]
    for i, txt in enumerate(region_texts):
        out.append(f"-- =========== Region {i+1} ===========")
        body = _strip_keyword_and_outer_terminator(txt, "PVDG")
        out.append(body.rstrip())
    out.append("/")
    return "\n".join(out) + "\n"


def build_multi_region_pvtg(region_texts):
    """Stack multiple PVTG region tables."""
    out = ["PVTG", "-- Multi-region wet-gas PVT (PVTNUM)",
           "-- P        Rv         Bg         Mug",
           "-- psia     STB/Mscf   rb/Mscf    cP"]
    for i, txt in enumerate(region_texts):
        out.append(f"-- =========== Region {i+1} ===========")
        body = _strip_keyword_and_outer_terminator(txt, "PVTG")
        out.append(body.rstrip())
    out.append("/")
    return "\n".join(out) + "\n"


def build_multi_region_pvtw(region_pvtw_texts):
    """
    Stack multiple PVTW lines. PVTW is simpler: one line per region, all under
    a single PVTW keyword.
    """
    out = ["PVTW",
           "-- Pref     Bwref     Cw           Muw       Viscosibility"]
    for i, txt in enumerate(region_pvtw_texts):
        # Each region txt contains the keyword + comment + the line + `/`
        # Pull out the data line(s)
        for line in txt.splitlines():
            ls = line.strip()
            if (not ls) or ls.startswith("PVTW") or ls.startswith("--"):
                continue
            out.append(f"   {ls}  -- Region {i+1}" if "/" in ls else line)
    return "\n".join(out) + "\n"


def build_multi_region_density(region_densities):
    """
    Stack DENSITY entries for each PVT region.
    region_densities: list of (rho_o, rho_w, rho_g) tuples in lb/ft3.
    """
    out = ["DENSITY", "-- Oil       Water     Gas      (lb/ft3)"]
    for i, (ro, rw, rg) in enumerate(region_densities):
        out.append(f"   {ro:7.3f}   {rw:7.3f}   {rg:7.4f}  /  -- Region {i+1}")
    return "\n".join(out) + "\n"


def _strip_keyword_and_outer_terminator(text, keyword):
    """
    Strip the leading keyword line, header comments, and the *outer* trailing
    `/` line from a single-region keyword block, keeping the per-region `/`.

    Input (typical):
        PVTO
        -- comment
        -- comment
          0.5  500   1.1   0.5  /
          0.6  800   1.2   0.4
                    1500   1.18  0.6  /
        /

    Output:
          0.5  500   1.1   0.5  /
          0.6  800   1.2   0.4
                    1500   1.18  0.6  /
    """
    lines = text.splitlines()
    # Remove the keyword line
    out = []
    seen_keyword = False
    in_header = True
    for line in lines:
        s = line.strip()
        if not seen_keyword and s.startswith(keyword):
            seen_keyword = True
            continue
        if in_header and s.startswith("--"):
            continue
        in_header = False
        out.append(line)
    # Strip trailing empty lines and a *final* line that is just `/`
    while out and not out[-1].strip():
        out.pop()
    if out and out[-1].strip() == "/":
        out.pop()
    return "\n".join(out)


def build_multi_region_deck(regions_data, water_train=None):
    """
    Assemble a full multi-region PVT include file.

    regions_data: list of dicts, one per region, each with:
        kind        : 'oil' or 'gas'
        pvt_text    : the PVTO or PVTG block as a string
        density     : (rho_o, rho_w, rho_g) in lb/ft3
        pvtw_text   : optional PVTW single-line text

    Returns the assembled deck string.
    """
    if not regions_data:
        return ""

    header = ("-- =====================================================\n"
              "-- Multi-region PVT include file (PVTNUM > 1)\n"
              "-- Generated by Streamlit PVT App\n"
              "-- Units: FIELD\n"
              "-- =====================================================\n\n")

    # Group by kind
    oil_regions = [r for r in regions_data if r["kind"] == "oil"]
    gas_regions = [r for r in regions_data if r["kind"] == "gas-dry"]
    wet_regions = [r for r in regions_data if r["kind"] == "gas-wet"]

    parts = [header]

    # DENSITY (always included)
    densities = [r["density"] for r in regions_data]
    parts.append(build_multi_region_density(densities) + "\n")

    if oil_regions:
        pvto_combined = build_multi_region_pvto([r["pvt_text"] for r in oil_regions])
        parts.append(pvto_combined + "\n")
    if gas_regions:
        pvdg_combined = build_multi_region_pvdg([r["pvt_text"] for r in gas_regions])
        parts.append(pvdg_combined + "\n")
    if wet_regions:
        pvtg_combined = build_multi_region_pvtg([r["pvt_text"] for r in wet_regions])
        parts.append(pvtg_combined + "\n")

    # PVTW
    pvtw_texts = [r.get("pvtw_text", "") for r in regions_data if r.get("pvtw_text")]
    if pvtw_texts:
        parts.append(build_multi_region_pvtw(pvtw_texts) + "\n")

    return "".join(parts)
