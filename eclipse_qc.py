"""
PVT Studio — ECLIPSE Export Quality-Control Utilities
======================================================

Helpers that make the generated ECLIPSE decks safer and more useful:

1. Monotonicity checking — ECLIPSE rejects PVT tables whose columns are not
   monotonic in the way it expects (Bo decreasing with P in the
   under-saturated region, Bg strictly decreasing with P, viscosity
   monotonic, etc.). A non-monotonic table causes the simulator to abort
   with a cryptic error. These functions catch the problem *before* export
   and report exactly which rows offend.

2. Property extraction for plotting — pulls the numeric (P, Bo, Rs, ...)
   columns back out of a generated keyword block so the UI can plot what
   was actually written to the deck.

3. Rs / Rv versus depth — builds the RSVD / RVVD compositional-gradient
   tables, which let ECLIPSE initialize a reservoir whose dissolved-gas or
   vaporized-oil content varies with depth.
"""

import re


# ----------------------------------------------------------------------
# 1. Monotonicity checking
# ----------------------------------------------------------------------
def check_monotonic(values, name, expect="any", tol=1e-9):
    """Check a sequence for monotonic behaviour.

    expect : 'increasing', 'decreasing', or 'any' (just flags reversals).
    Returns a list of human-readable problem strings (empty if all good).
    """
    problems = []
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return problems
    for i in range(1, len(vals)):
        d = vals[i] - vals[i - 1]
        if expect == "increasing" and d < -tol:
            problems.append(
                f"{name}: row {i+1} decreases ({vals[i-1]:.5g} → "
                f"{vals[i]:.5g}) — expected non-decreasing.")
        elif expect == "decreasing" and d > tol:
            problems.append(
                f"{name}: row {i+1} increases ({vals[i-1]:.5g} → "
                f"{vals[i]:.5g}) — expected non-increasing.")
    return problems


def qc_pvto_table(df, pb=None):
    """QC a black-oil PVTO property table (a DataFrame with P, Rs, Bo, mu).

    A black-oil PVTO table has TWO regimes:
      - Saturated (P <= Pb): Rs increases, Bo increases, viscosity decreases.
      - Under-saturated (P > Pb): Rs is constant, Bo DECREASES (liquid
        compression), viscosity INCREASES. This is physically correct.
    So monotonicity must be checked within each branch, not across the
    whole table. If `pb` is given the split uses it; otherwise the branch
    split is inferred from where Rs stops increasing.
    """
    problems = []
    p_col  = _find_col(df, ["P (", "P_"])
    rs_col = _find_col(df, ["Rs"])
    bo_col = _find_col(df, ["Bo"])
    mu_col = _find_col(df, ["μ", "mu"])
    if p_col is None:
        return {"ok": True, "problems": []}

    pvals = list(df[p_col])
    # Pressure itself must always increase.
    problems += check_monotonic(pvals, "Pressure", "increasing")

    # Determine the saturated/under-saturated split index.
    # split = index of the first row in the under-saturated branch.
    split = len(df)
    if pb is not None:
        for i, p in enumerate(pvals):
            if p > pb + 1e-6:
                split = i
                break
    elif rs_col is not None:
        rs = list(df[rs_col])
        for i in range(1, len(rs)):
            if rs[i] <= rs[i - 1] + 1e-9:   # Rs stopped rising -> Pb reached
                split = i
                break

    # Saturated branch: rows [0, split). Under-saturated: rows [split, end).
    # The two branches are checked independently — no overlap row, because
    # the saturated/under-saturated transition is a genuine kink in Bo and
    # viscosity, not a monotonicity violation.
    sat = slice(0, split)
    uns = slice(split, len(df))

    if rs_col is not None:
        problems += check_monotonic(list(df[rs_col])[sat],
                                     "Solution GOR (saturated)", "increasing")
    if bo_col is not None:
        problems += check_monotonic(list(df[bo_col])[sat],
                                     "Oil FVF (saturated)", "increasing")
        if split < len(df):
            problems += check_monotonic(list(df[bo_col])[uns],
                                         "Oil FVF (under-saturated)",
                                         "decreasing")
    if mu_col is not None:
        problems += check_monotonic(list(df[mu_col])[sat],
                                     "Oil viscosity (saturated)", "decreasing")
        if split < len(df):
            problems += check_monotonic(list(df[mu_col])[uns],
                                         "Oil viscosity (under-saturated)",
                                         "increasing")
    return {"ok": len(problems) == 0, "problems": problems}


def qc_pvdg_table(df):
    """QC a dry-gas PVDG table (P, Bg, mu).

    ECLIPSE expects pressure increasing, Bg strictly decreasing,
    viscosity increasing.
    """
    problems = []
    p_col  = _find_col(df, ["P (", "P_"])
    bg_col = _find_col(df, ["Bg"])
    mu_col = _find_col(df, ["μ", "mu"])
    if p_col:
        problems += check_monotonic(list(df[p_col]), "Pressure",
                                     "increasing")
    if bg_col:
        problems += check_monotonic(list(df[bg_col]), "Gas FVF (Bg)",
                                     "decreasing")
    if mu_col:
        problems += check_monotonic(list(df[mu_col]), "Gas viscosity",
                                     "increasing")
    return {"ok": len(problems) == 0, "problems": problems}


def qc_pvtg_table(df):
    """QC a wet-gas PVTG table (P, Rv, Bg, mu)."""
    problems = []
    p_col  = _find_col(df, ["P (", "P_"])
    bg_col = _find_col(df, ["Bg"])
    if p_col:
        problems += check_monotonic(list(df[p_col]), "Pressure",
                                     "increasing")
    if bg_col:
        problems += check_monotonic(list(df[bg_col]), "Gas FVF (Bg)",
                                     "decreasing")
    return {"ok": len(problems) == 0, "problems": problems}


# ----------------------------------------------------------------------
# 2. Numeric extraction from a generated keyword block (for plotting)
# ----------------------------------------------------------------------
def extract_numeric_rows(keyword_text):
    """Pull numeric rows out of a generated ECLIPSE keyword block.

    Returns a list of float-tuples, one per data line. Comment lines
    (starting with --), the keyword name, and terminators (/) are skipped.
    Useful for plotting exactly what was written to the deck.
    """
    rows = []
    for line in keyword_text.splitlines():
        s = line.strip()
        if not s or s.startswith("--"):
            continue
        # Drop a trailing slash terminator
        s = s.rstrip("/").strip()
        if not s:
            continue
        # Skip a bare keyword name (all letters)
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", s):
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if nums:
            try:
                rows.append(tuple(float(n) for n in nums))
            except ValueError:
                continue
    return rows


# ----------------------------------------------------------------------
# 3. Rs / Rv versus depth (RSVD / RVVD compositional gradient)
# ----------------------------------------------------------------------
def build_rsvd(depths, rs_values, comment="Rs vs depth"):
    """Build an ECLIPSE RSVD keyword block.

    depths     : list of depths (ft for FIELD, m for METRIC).
    rs_values  : matching solution-GOR values.
    """
    lines = [f"-- {comment}", "RSVD"]
    for d, rs in zip(depths, rs_values):
        lines.append(f"   {d:10.2f}  {rs:12.4f}")
    lines.append("/")
    return "\n".join(lines) + "\n"


def build_rvvd(depths, rv_values, comment="Rv vs depth"):
    """Build an ECLIPSE RVVD keyword block (vaporized OGR vs depth)."""
    lines = [f"-- {comment}", "RVVD"]
    for d, rv in zip(depths, rv_values):
        lines.append(f"   {d:10.2f}  {rv:12.6f}")
    lines.append("/")
    return "\n".join(lines) + "\n"


def linear_depth_profile(value_at_ref, gradient, ref_depth, depths):
    """A simple linear compositional gradient.

    value(d) = value_at_ref + gradient * (d - ref_depth)

    gradient is 'units per unit depth' (e.g. scf/STB per ft). Values are
    floored at zero — a negative Rs or Rv is unphysical.
    """
    return [max(0.0, value_at_ref + gradient * (d - ref_depth))
            for d in depths]


# ----------------------------------------------------------------------
# Internal helper
# ----------------------------------------------------------------------
def _find_col(df, needles):
    """Return the first DataFrame column whose name contains any needle."""
    for c in df.columns:
        cs = str(c)
        for nd in needles:
            if nd in cs:
                return c
    return None
