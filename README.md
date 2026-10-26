# PVT Studio — Equinor-themed PVT Application

By **Merouane Hamdani** · MIT License · Early-phase screening tool

A Streamlit application for petroleum-fluid PVT modeling: black-oil and
gas correlations, Peng-Robinson EOS, lab-experiment simulation, tuning,
ECLIPSE deck export, and flow-assurance screening.

## Modes (sidebar selector)

1. **Oil (Black Oil)** — correlations, tuning, Monte Carlo, ECLIPSE export
2. **Dry Gas** — Z-factor / Bg / viscosity, tuning, multi-region
3. **Wet Gas / Condensate** — recombination, Rv, dew point
4. **Water** — brine properties, PVTW
5. **Compositional (EOS)** — Peng-Robinson: flash, phase envelope, lab
   experiments (CCE/CVD/DLE), separator train, EOS tuning, multi-region,
   Monte Carlo, ECLIPSE export
6. **Hydrate Likelihood** — Makogon screening, inhibitor, cooldown
7. **Rock Compressibility** — correlations + recommendation by rock type
8. **Wax & Asphaltene Risk** — WAT / cloud-point + asphaltene-onset
   screening (CII, de Boer), cooldown to wax onset
9. **Documentation** — full LaTeX equation reference

## How to run

```
pip install -r requirements.txt
streamlit run pvt_app.py
```

Validation suite (51 checks against published references):

```
python test_validation.py
```

## Module layout

`pvt_app.py` is the Streamlit orchestrator. `ui_helpers.py` holds the pure
rendering helpers. Physics is in dedicated modules: `correlations.py`,
`eos_pr.py`, `experiments.py`, `phase_envelope.py`, `separator.py`,
`hydrate.py`, `rock_comp.py`, `solids_risk.py`, `lbc.py`. Support:
`units.py` (all unit conversions), `validators.py` (input validation),
`presets.py` (16 example fluids), `eclipse_export.py` / `eclipse_qc.py`
(deck generation + monotonicity QC), `documentation.py` (equation
reference).

---

## Change history

## v1.4.7 — Gas lift and ALQ in the lift curves / VFP

- Lift curves: "Artificial lift" section — natural flow or continuous gas
  lift: operating-valve depth (MD), lift-gas rate (the ALQ), lift-gas
  gravity, valve dP, available casing-head injection pressure.
- March: above the valve the free gas is formation + lift gas (mixed
  gravity); the segment is split exactly at the valve.
- Gas-lift results: tubing pressure at the valve, required casing-head
  injection pressure (static annulus gas column + valve dP), check against
  the available pressure, deepest feasible injection point (point of
  balance), and the gas-lift performance curve (IPR/VLP liquid rate vs
  lift-gas rate, maximum marked, CSV). Traverse shows the valve and the
  annulus gas-pressure line; the schematic draws the valve, the lift-gas
  inlet and annulus gas.
- Well tests take an optional lift-gas rate per test (calibration with
  gas lift on).
- VFPPROD: ALQ axis — none, GRAT (lift-gas rate, Mscf/d | Sm3/d) or IGLR
  (injection GLR, Mscf/STB | Sm3/Sm3); the table becomes
  THP x WFR x GFR x ALQ x rate.
- VFPPROD: FLO sweep — the liquid-rate axis can now be set independently
  of the lift curve ("Custom range": min, max, number of points, STB/d |
  Sm3/d) instead of always reusing the nodal rate points, e.g. to cover
  gas-lifted rates beyond the natural-flow curve.
- Fix: FIELD VFPPROD gas ratios (GFR) are now written in Mscf/STB (they
  were written in scf/STB, 1000x too high). METRIC was correct.

## v1.4.6 — Imported ECLIPSE PVT for the nodal analysis

- New `eclipse_pvt_import.py`: reads PVTO, PVDO, PVTG, PVDG, PVTW,
  DENSITY and GRAVITY (FIELD or METRIC, auto-detected; any number of
  PVTNUM regions, repeat counts / defaults such as `1*`).
- Lift curves: "PVT model for the VLP / IPR" — built-in correlations or
  imported ECLIPSE tables (upload, paste, or load an example deck).
  The march uses a black-oil split with dissolved gas (Rs) and vaporised
  oil (Rv): undersaturated oil below the producing GOR (undersaturated
  branches interpolated ECLIPSE-style), condensate drop-out from PVTG,
  table FVFs, viscosities and in-situ densities from DENSITY; PVTW for
  water. Tables are isothermal; gas FVF optionally corrected to the local
  tubing temperature. Missing tables fall back to the correlations.
- IPR: Darcy oil Bo / mu_o at Pr from the tables, Darcy gas mu*z from the
  gas table, and Pb at the producing GOR for the composite / Darcy-oil IPR.
- Applies to the lift curve, comparison curves, traverse, schematic hover,
  correlation ranking / VLP tuning and the VFP export.
- QC expander: table Rs, Bo (saturated and at the producing GOR), mu_o,
  Bg, mu_g, Rv vs pressure against the built-in correlations, plus a
  summary of what was read.

## v1.4.5 — More flow correlations, Darcy IPR, calibration

**Multiphase-flow correlations (7)**
- New: Mukherjee & Brill (1985, all inclinations incl. downhill),
  Hasan & Kabir (1988 drift-flux, deviation-corrected), Orkiszewski (1967,
  Griffith bubble / slug with liquid-distribution coefficient / Duns & Ros
  mist), No-slip homogeneous (lower bound). Existing: Beggs & Brill,
  Hagedorn & Brown, Gray.
- Hagedorn & Brown: chart fits replaced by Trick's rational fits for CNL,
  HL/psi and psi - the old fit clamped H_L to the no-slip value.
- Correlation recommendation updated (deviated oil -> Mukherjee & Brill),
  reference expander with the models and equations.
- Overlay any other correlations on the lift-curve plot.

**IPR**
- Darcy radial inflow, pseudo-steady state: oil (k, h, mu, Bo, re, rw,
  skin; Vogel below Pb when Pb > 0) and gas (pressure-squared form with
  mu*z at the average pressure and non-Darcy coefficient D).
- Optional skin calibration of the Darcy models to the test point.

**Calibration**
- Well-test table (rate, THP, Pwf at the node, optional GLR / WC).
- IPR fit to all tests: J, qmax, C and n, skin, or skin and D.
- VLP: rank all correlations against the tests (MAPE, MAE, bias), one
  click to use the best; tune the chosen correlation with gravity and
  friction multipliers (gravity only for one test), predicted-vs-measured
  plot, applied to the lift curve, traverse, schematic hover and the VFP
  export.
- Tests plotted on the IPR / VLP chart.

## v1.4.4 — Interactive (JavaScript) well schematic

- New `well_schematic.py`: the Nodal Analysis well schematic is now a
  JavaScript / SVG drawing rendered in the page (no extra package):
  formation layers with lithology patterns, cap rock and reservoir, sea and
  seabed or ground level, platform deck and Christmas tree, every casing
  string with cement, shoes and liner hanger, tubing with animated flow,
  production packer, perforation jets, solution node (reference depth) and
  the Pr datum, with callouts for every component.
- Hover the well for MD, TVD, temperature, the casing strings and tubing at
  that depth and, after a run, the pressure from the traverse at the
  operating point. Toggles for flow animation, geology and labels.
- "Completion details (schematic)" expander: offshore platform / onshore,
  water depth, editable casing programme (defaults scale with the well),
  packer depth, perforated interval, rathole, reservoir thickness. Drawing
  only - the VLP still uses the tubing ID to the solution node.
- Deviated wells keep true scale when they fit; otherwise the horizontal
  scale is compressed (stated on the drawing). Flowlines keep the Plotly
  schematic.

## v1.4.3 — Nodal analysis: well schematic, reference depth, IPR / VLP QC

**New (Nodal Analysis / Lift Curves)**
- **Well schematic**: trajectory (TVD vs displacement) from the segments,
  casing/tubing, outlet node (THP, T) and the **solution node** (MD/TVD,
  inlet T) with perforations, plus the Pr datum when it differs.
- **Assumptions table** next to the schematic: reference depth, outlet
  node, temperature profile, pipe, correlation, PVT basis, rate basis,
  Pr at the node and the fitted IPR.
- **IPR on the same plot as the VLP**, both at the solution node:
  straight-line PI, Vogel, composite PI + Vogel, gas back-pressure (C, n),
  each calibrated from one well test. Optional Pr datum shifted to the
  node with a static gradient.
- Operating point(s) with stability (stable when the VLP rises faster
  than the IPR), metrics (rate, gas rate, Pwf, drawdown, % AOF), a QC
  checklist (no intersection, unstable crossing, THP > Pr ...) and a CSV
  of the IPR / VLP table.

**Fixes (nodal.py)**
- Beggs & Brill two-phase friction factor: the S(y) branches were swapped
  and a sign guard made exp(S) overflow -> NaN points on the lift curve.
- Beggs & Brill flow-pattern map: very dry flow (lambda < 0.01) is now
  distributed when Fr >= L1 (was always segregated).
- Pressure marching uses adaptive sub-steps (<= 5 % of P per step); the
  fixed explicit step could jump hundreds of psi on steep gradients.
- "Stable point" on the VLP renamed to "VLP minimum (stability limit)" -
  the operating point is the IPR / VLP intersection.

## v1.4.2 — SI by default, unit-consistent messages, E300 + multi-export everywhere

**Bugs fixed**
- `build_pvto() got an unexpected keyword argument 'P_min'` / `parse_pvtg_branches`
  AttributeError on Streamlit Cloud: the app now purges stale cached modules
  when any module's APP_VERSION differs from the main script (self-heals
  after a push; one "Reboot app" is still recommended after this upload).
- Compressibility plot Pb at the wrong pressure: the share-URL restore could
  inject field-unit values into SI widgets after a unit switch. The restore
  now runs once per session, before any widget, together with the unit
  system. A warning is shown when Pb exceeds the table's P_max.
- Tuned EOS: Psat, the tuned export table and the E300 deck now use the
  tuned C1/N2-C7+ kij (previously only the C7+ multipliers were applied).

**Units**
- SI (bara, degC, Sm3/Sm3) is the default everywhere; ECLIPSE and VFP exports
  default to METRIC. SI P_min default = 1.01325 bara.
- All validator, hydrate, wax, Monte-Carlo, CGR and saturation messages are
  formatted in the selected unit system (`units.fmt`).
- RSVD/RVVD grading points are stored in field units, edited in display
  units and written in the deck unit set. Unit switch clears every
  unit-dependent widget, including depth-grading rows.
- Multi-simulator headers print reservoir T in the export's unit set.
- Every deck / simulator file is plain ASCII.

**New**
- Compositional (EOS) → **ECLIPSE 300 PROPS deck**: CNAMES, EOS PR + PRCORR,
  TCRIT, PCRIT, VCRIT, ZCRIT, ACF, MW, PARACHOR, BIC, RTEMP, STCOND, ZI
  (FIELD or METRIC), with a component table, structural QC, the RUNSPEC
  lines and Psat as a QC target. Also a **CMG GEM EOS** block.
- Multi-simulator export (CMG IMEX/GEM, tNavigator, OPM, Nexus, IX, CSV,
  JSON, zip bundle) in **all fluid types**: oil, dry gas, wet gas (with Rv),
  water (Bw/Cw/mu_w/Cvw) and compositional (black-oil tables + E300 + GEM).
  Shared helper: `ui_helpers.render_multi_sim_export`.
- test_audit.py: 77 checks (E300 criticals/units, ASCII decks, wet-gas and
  water writers, bundle extras).

## v1.4.1 — deployment check

Every module now carries `APP_VERSION`. On startup the app checks that all
files come from the same release. If any are older or missing, it stops
with a list of the files to re-upload, instead of failing inside a page
(e.g. `AttributeError: ... parse_pvtg_branches`). **Always upload every file
from the release zip together.**

## v1.4 — PVTO/PVTG rebuild + second formula audit (Sep 2026)

**PVTO (correlation and EOS).** The table now spans from ~1 atm to Pb, and
optionally on up to the table top (gas re-solution). Rsi sits exactly at Pb,
and **every** Rs node carries an undersaturated branch up to 1.25 × the table
top. Branches use the bubble-point reference behaviour (the Petrel/PVTi and
ECLIPSE-default method): Vasquez-Beggs evaluated at a 15-psia Psat had inflated
viscosity 18×. There are span controls (saturated nodes, undersaturated points,
extend above Pb). The EOS export re-runs the depletion on a dense grid, where it
previously had about 6 Rs nodes.

**PVTG.** Each (P, Rv) point is computed from the gas that actually carries that
Rv, with Bg on a surface-dry-gas basis. This fixes branch viscosity (it
increased as gas got leaner) and the saturated gravity below the dew point. The
saturated Rv line continues above the dew point, so reservoir gas sits on an
undersaturated branch. The dew point is a node, and the extension is limited so
saturated Bg keeps decreasing. The EOS PVTG uses the surface-gas basis and a
computed dry-gas branch end.

**QC and plots.** QC parses the real keyword structure. The old QC flattened
PVTG into one table, which shifted every branch row into the wrong columns, so
it always failed and the plot was garbled. Checks are split into ECLIPSE errors
and physical warnings. PVTO plots show the saturated curve with branches, μ, and
Rs vs Psat. PVTG plots show Rv_sat vs P with the CGR line, Bg and μg (saturated
and dry gas with branch segments), and Bg vs Rv.

**Other fixes.** LBC viscosity used °R/psia in K/atm coefficients, so every EOS
viscosity was about 5.4× high; there is now a Vc(C7+) lever to match measured oil
viscosity. Multi-region PVTO/PVDG/PVTG had one terminator for all regions and
now has one per PVTNUM table. The wet-gas CVD cumulative production was a
placeholder (+0.1 %/step) and now uses the material balance. METRIC PVTO/PVTG
output has more precision, and PVTG Rv is written in scientific notation.

Tests: `test_audit.py` (56), `test_validation.py` (51), `validate_nodal.py` (16).

## v1.3 — formula & bug audit (Sep 2026)

Run `python test_audit.py` (33 regression checks), plus `python test_validation.py`
(51) and `python validate_nodal.py` (16).

**SI display (critical).** Bg, Rv and CGR used the scf/STB factor (5.6146) for
per-Mscf / per-MMscf quantities, so SI Bg and Rv were ~32x too large and CGR
~32x too small. Fixed (ECLIPSE METRIC decks were already correct). Lab data now
converts when you switch Field/SI, and all unit-dependent inputs reset.

**Formulas.** Z-factor solvers (HY, DAK) replaced with a bracketed root search.
DAK returned non-physical Z (up to 5.3) at Tpr 1.05-1.40. Carr-Kobayashi-Burrows
viscosity now uses Dempsey's fit (was up to 5x low). Lasater uses the published
form (Pb was ~3x high). Undersaturated Bo uses the integrated Vasquez-Beggs form.
Newman limestone uses the published equation, and two unsourced rock formulas are
relabelled as screening fits. PVTG Bg is now on a surface-dry-gas basis
(+ ~6% for a typical condensate).

**Nodal / VFP.** The march counted dissolved gas as free gas and applied oil Bo
to water, so BHP was understated. Both are fixed, and live-oil viscosity is used.
The VFP FLO = OIL/GAS/WAT options previously only changed the header label; the
axis is now in the correct units. Fixed a crash in the traverse slider, and
results now persist after Run.

**Multi-simulator export.** Dry-gas CMG export was empty, dry-gas IX crashed,
Nexus wrote zero Bg, and oil CMG dropped viscosity. All fixed through one
column/kind normaliser. METRIC now converts Nexus/IX tables, and CMG SI uses kPa.
Removed the unused `simulator_exports.py`.


## Latest round — robustness: validation, presets, unit audit

- **Input validation** (`validators.py`) — hard guards reject physically
  impossible inputs (negative pressure, zero GOR, empty composition,
  porosity entered as a percent); soft warnings flag inputs outside a
  correlation's published validity envelope, with the range and source.
- **Example fluid presets** (`presets.py`) — 16 representative literature
  fluids across all branches. "Load an example fluid" fills every input
  so a new user sees a complete worked result immediately.
- **Tuning staleness detection** — each tuning result stores a fingerprint
  of the fluid it was tuned against; the tuned overlay/export is hidden
  with a warning if the current inputs no longer match.
- **Unit-conversion audit** — every conversion now routes through
  `units.py`; all inline conversion factors removed. New CGR converters
  added. The validation suite checks round-trip identity for all 8
  conversion families (pressure, T, GOR, Bg, Rv, Cw, density, CGR).
- **Validation suite is now 51 checks** (was 38) — `python test_validation.py`.

## Maintainability — ui_helpers.py

The pure, self-contained Streamlit rendering helpers (`render_tornado_chart`,
`render_property_plots`, `render_eclipse_qc`, `render_depth_profile`,
`render_input_correlation`, `line_chart_plotly`, `styled_dataframe`, plus the
tuning-staleness helpers) have been extracted into `ui_helpers.py`. These
functions depend only on their arguments — no application global state — so
they can be reasoned about and tested in isolation. `pvt_app.py` imports them
by name, so all call sites are unchanged.

## How to run

```
pip install -r requirements.txt
streamlit run pvt_app.py
```

Validation suite: `python test_validation.py`

---

## Earlier rounds

# PVT Studio — Equinor-themed PVT Application

By **Merouane Hamdani** · MIT License · Early-phase tool

## What's new this round

### Branding & legal
- **SVG mascot** in the header (cute oil-drop scientist with goggles + beaker)
- **MIT License + disclaimer expander** at the top
- Owner attribution: Merouane Hamdani
- App version: PVT Studio v1.0

### New top-level mode: 🪨 Rock Compressibility
Five correlations compared side-by-side (Hall 1953, Newman SS/LS, Horne polynomial,
Carpenter-Spencer carbonate). Cf-vs-φ log-scale plot with operating point marked.
Exports `ROCK` keyword in FIELD or METRIC.

### Hydrate tab additions
- **Subsea shutdown cooldown time** using lumped-capacitance heat transfer.
- Traffic-light risk banner (urgent <1hr / short <4hr / adequate ≥4hr).
- Cooldown curve plot with hydrate-T line and ambient line.
- Inputs: U-factor, pipe OD, fluid density, Cp.

### Oil branch additions
- **Correlation tuning** with experimental data (Pb shift + Rs/Bo/μ factors,
  L-BFGS-B optimizer).
- **Tuned vs untuned comparison plot** (grouped bar chart vs lab data).
- **Auto-select best correlation** — compares Standing/Vasquez-Beggs/Glaso/Lasater.
- **Composition guess** from API + gas SG + Rsi (Whitson-style 11-component).
- **Monte Carlo documentation** — explanation of uncertainty inputs and
  notes on parameter correlations.

### Dry & Wet Gas branches additions
- **ECLIPSE METRIC option** (via global sidebar toggle).
- **Monte Carlo uncertainty** with histograms for Z/Bg/Rv.
- **Composition guess** from SG (and CGR for wet gas).

### Sidebar
- **ECLIPSE export master toggle** — when OFF, all ECLIPSE panels are hidden
  across all fluid types.
- **Global ECLIPSE unit system** (FIELD/METRIC) applies to all branches.
- License/owner footer.

### Common across all branches: Tools section
At the bottom of every branch, an expander provides:
- **Save fluid to in-session registry** (name + notes + JSON-able payload).
- **List of saved fluids** with one-line summaries.
- **Download all saved fluids as JSON** (and upload to restore).
- **CSV export** of the results table.
- **JSON API payload** (structured input + output for downstream tools).
- **PDF report** (reportlab; lazy import — graceful fallback if missing).

### Multi-region (Compositional)
- Per-region **Source selector**: use current composition or a saved fluid.
- Real DENSITY values computed from EOS standard-conditions split.

## File map
- `pvt_app.py`              — Main UI (~2800 lines, 7 fluid/analysis modes)
- `mascot.py`               — Inline SVG header mascot
- `hydrate.py`              — Makogon hydrate + Hammerschmidt + subsea cooldown
- `rock_comp.py`            — Five Cf correlations + ROCK keyword
- `correlation_tuning.py`   — L-BFGS-B fit of correlation correction factors
- `composition_guess.py`    — API/SG → composition synthesis
- `fluid_registry.py`       — JSON save/load fluid records
- `export_utils.py`         — CSV/JSON/PDF exports (reportlab optional)
- `theme.py`                — Equinor colors, CSS, Plotly layout helpers
- `eos_pr.py`               — Peng-Robinson EOS
- `eos_tuning.py`           — EOS L-M tuning to lab data
- `lbc.py`                  — Lohrenz-Bray-Clark viscosity
- `components.py`           — Component library + Kesler-Lee C7+
- `composition_pvt.py`      — Compositional black-oil table generation
- `experiments.py`          — EOS-based Flash, CCE, CVD, DLE
- `correlation_experiments.py` — CCE/CVD using correlations
- `separator.py`            — Multi-stage surface separator
- `multi_region.py`         — Multi-region ECLIPSE export
- `monte_carlo.py`          — MC sampling + tornado
- `phase_envelope.py`       — Bubble/dew locus tracer
- `correlations.py`         — Oil/gas/wet-gas/water correlations
- `eclipse_export.py`       — ECLIPSE keyword formatters (FIELD + METRIC)
- `units.py`                — Field ↔ SI display conversion
- `.streamlit/config.toml`  — Streamlit theme config

## Run
```bash
pip install -r requirements.txt
streamlit run pvt_app.py
```

Dependencies: `streamlit, numpy, pandas, plotly, scipy`, optional `reportlab` for PDF.

## License
MIT. © 2026 Merouane Hamdani.

## Disclaimer
Early-phase tool for screening only. Validate against rigorous PVT software
before use in field design.
