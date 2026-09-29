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
