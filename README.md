# PVT Studio — Equinor-themed PVT Application

By **Merouane Hamdani** · MIT License · Early-phase screening tool

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
