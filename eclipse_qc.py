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


# ----------------------------------------------------------------------
# PVTO Rs-branch parser (for plotting the true PVTO structure)
# ----------------------------------------------------------------------
def parse_pvto_branches(pvto_text):
    """Parse a generated PVTO keyword block into per-Rs branches.

    A PVTO table is organized as a set of saturated Rs nodes; each node
    may carry an under-saturated extension (rows with a blank Rs column).
    Returns a list of branch dicts:
        {"Rs": <float>, "P": [...], "Bo": [...], "mu": [...]}
    one per saturated Rs node, each including its saturated point plus any
    under-saturated rows. Units are whatever the deck is written in
    (FIELD or METRIC) — the caller is responsible for labelling.
    """
    branches = []
    current = None
    for line in pvto_text.splitlines():
        s = line.strip()
        if not s or s.startswith("--"):
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", s):
            continue  # the 'PVTO' keyword line
        s_clean = s.rstrip("/").strip()
        if not s_clean:
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s_clean)
        try:
            vals = [float(n) for n in nums]
        except ValueError:
            continue
        # Count leading whitespace to tell a saturated node (has Rs in the
        # first column) from an under-saturated continuation row.
        indent = len(line) - len(line.lstrip())
        is_saturated_node = (len(vals) >= 4)
        if is_saturated_node:
            # New Rs node: Rs, Psat, Bo, mu
            if current is not None:
                branches.append(current)
            current = {"Rs": vals[0], "P": [vals[1]],
                        "Bo": [vals[2]], "mu": [vals[3]]}
        elif current is not None and len(vals) >= 3:
            # Under-saturated continuation: P, Bo, mu (same Rs)
            current["P"].append(vals[0])
            current["Bo"].append(vals[1])
            current["mu"].append(vals[2])
    if current is not None:
        branches.append(current)
    return branches


# ----------------------------------------------------------------------
# Auto-fix: enforce monotonicity on a PVT table
# ----------------------------------------------------------------------
def _enforce_monotonic(values, expect, tol=1e-12):
    """Return a copy of `values` made monotonic in the expected direction.

    Each value that would violate monotonicity is clamped to its
    predecessor. This is a minimal, local repair — it never moves a value
    further than necessary — and the caller should make clear to the user
    that the table has been adjusted.
    """
    out = list(values)
    for i in range(1, len(out)):
        if out[i] is None or out[i - 1] is None:
            continue
        if expect == "increasing" and out[i] < out[i - 1] - tol:
            out[i] = out[i - 1]
        elif expect == "decreasing" and out[i] > out[i - 1] + tol:
            out[i] = out[i - 1]
    return out


def autofix_pvto_table(df, pb=None):
    """Repair a PVTO DataFrame so every column is monotonic in the
    direction ECLIPSE expects, branch by branch (saturated /
    under-saturated). Returns (fixed_df, list_of_changes)."""
    fixed = df.copy()
    changes = []
    p_col  = _find_col(fixed, ["P (", "P_"])
    rs_col = _find_col(fixed, ["Rs"])
    bo_col = _find_col(fixed, ["Bo"])
    mu_col = _find_col(fixed, ["μ", "mu"])
    if p_col is None:
        return fixed, changes

    pvals = list(fixed[p_col])
    # Pressure must increase across the whole table.
    new_p = _enforce_monotonic(pvals, "increasing")
    if new_p != pvals:
        changes.append("Pressure column clamped to be non-decreasing.")
        fixed[p_col] = new_p

    # Saturated / under-saturated split.
    split = len(fixed)
    if pb is not None:
        for i, p in enumerate(new_p):
            if p > pb + 1e-6:
                split = i
                break
    elif rs_col is not None:
        rs = list(fixed[rs_col])
        for i in range(1, len(rs)):
            if rs[i] <= rs[i - 1] + 1e-9:
                split = i
                break

    def _fix_segment(col, expect, lo, hi, name):
        if col is None:
            return
        seg = list(fixed[col])[lo:hi]
        new_seg = _enforce_monotonic(seg, expect)
        if new_seg != seg:
            changes.append(f"{name}: {sum(1 for a, b in zip(seg, new_seg) if a != b)} "
                            f"row(s) adjusted.")
            full = list(fixed[col])
            full[lo:hi] = new_seg
            fixed[col] = full

    _fix_segment(rs_col, "increasing", 0, split, "Solution GOR (saturated)")
    _fix_segment(bo_col, "increasing", 0, split, "Oil FVF (saturated)")
    _fix_segment(mu_col, "decreasing", 0, split, "Oil viscosity (saturated)")
    if split < len(fixed):
        _fix_segment(bo_col, "decreasing", split, len(fixed),
                      "Oil FVF (under-saturated)")
        _fix_segment(mu_col, "increasing", split, len(fixed),
                      "Oil viscosity (under-saturated)")
    return fixed, changes


def autofix_pvdg_table(df):
    """Repair a PVDG DataFrame: P increasing, Bg decreasing, mu
    increasing. Returns (fixed_df, list_of_changes)."""
    fixed = df.copy()
    changes = []
    p_col  = _find_col(fixed, ["P (", "P_"])
    bg_col = _find_col(fixed, ["Bg"])
    mu_col = _find_col(fixed, ["μ", "mu"])
    for col, expect, name in [(p_col, "increasing", "Pressure"),
                               (bg_col, "decreasing", "Gas FVF (Bg)"),
                               (mu_col, "increasing", "Gas viscosity")]:
        if col is None:
            continue
        old = list(fixed[col])
        new = _enforce_monotonic(old, expect)
        if new != old:
            changes.append(f"{name}: "
                            f"{sum(1 for a, b in zip(old, new) if a != b)} "
                            f"row(s) adjusted.")
            fixed[col] = new
    return fixed, changes


def autofix_pvtg_table(df):
    """Repair a PVTG DataFrame: P increasing, Bg decreasing.
    Returns (fixed_df, list_of_changes)."""
    fixed = df.copy()
    changes = []
    p_col  = _find_col(fixed, ["P (", "P_"])
    bg_col = _find_col(fixed, ["Bg"])
    for col, expect, name in [(p_col, "increasing", "Pressure"),
                               (bg_col, "decreasing", "Gas FVF (Bg)")]:
        if col is None:
            continue
        old = list(fixed[col])
        new = _enforce_monotonic(old, expect)
        if new != old:
            changes.append(f"{name}: "
                            f"{sum(1 for a, b in zip(old, new) if a != b)} "
                            f"row(s) adjusted.")
            fixed[col] = new
    return fixed, changes
