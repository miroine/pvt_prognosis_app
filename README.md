# PVT Studio — Equinor-themed PVT Application

Streamlit app for petroleum-engineering PVT analysis. Five fluid types,
correlation and EOS modes, four lab experiments, phase envelopes, standalone
flash calculator, and ECLIPSE keyword export.

## Run
```
pip install -r requirements.txt
streamlit run pvt_app.py
```

## What's in this iteration

### Visual / UX
- Equinor color palette (Torch Red `#EB0037`, dark navy `#00243D`, Karry, Pistachio)
- Dark-navy header band with red accent stripe
- Plotly-based interactive charts (zoom, hover tooltips, equinor color sequence)
- Metric cards with red left borders
- Custom CSS for tables, code blocks, buttons, alerts
- `.streamlit/config.toml` sets primary color so widgets pick up the theme

### Fluids
- **Oil (Black Oil)** — Standing, Vasquez-Beggs, Glaso, Lasater correlations
- **Dry Gas** — HY / DAK Z, LGE / CKB viscosity, with Wichert-Aziz sour-gas
- **Wet Gas / Condensate** — McCain recombination + linear-Pdew Rv model
- **Water** — McCain, Meehan, Numbere, Spivey-Valko-McCain
- **Compositional (EOS)** — full Peng-Robinson with C7+ characterization

### Compositional mode — four tabs
1. **Lab Experiments** — Black-oil table, Flash, CCE, CVD, DLE
2. **Phase Envelope** — bubble locus, dew locus, estimated critical point,
   reservoir (P,T) marker, cricondenbar / cricondentherm metrics
3. **Flash Calculator** — single-stage flash at any (P,T): phase compositions,
   K-values, Z-factors, densities, viscosities for each phase
4. **ECLIPSE Export** — PVTO or PVTG with optional PVTW

### Units
Field ↔ SI toggle in sidebar. Inputs, tables, and charts all flip; ECLIPSE
output stays in field (the spec).

### Composition input
- Normalize button (rescale Σz to 1.0)
- Reset button (restore default black-oil composition)
- Live Σz indicator (green if within 1e-3 of 1.0)
- C7+ MW and SG → Kesler-Lee Tc/Pc/ω characterization

## File map
- `pvt_app.py`         — Streamlit UI (tabbed for Compositional mode)
- `theme.py`           — Equinor colors, CSS, Plotly layout helpers
- `phase_envelope.py`  — Bubble/dew locus tracer + critical estimation
- `experiments.py`     — Flash, CCE, CVD, DLE simulations
- `eos_pr.py`          — Peng-Robinson EOS, stability test, flash, saturation P
- `lbc.py`             — Lohrenz-Bray-Clark viscosity
- `composition_pvt.py` — Compositional black-oil table generation
- `components.py`      — Component library + C7+ Kesler-Lee characterization
- `correlations.py`    — Oil/gas/wet-gas/water correlations
- `eclipse_export.py`  — PVTO/PVDG/PVTG/PVTW/DENSITY keyword formatters
- `units.py`           — Field ↔ SI conversions
- `.streamlit/config.toml` — Streamlit theme (Torch Red primary)

## Notes
The phase envelope uses single-T saturation searches at each sample point.
This is robust but slower than a true predictor-corrector continuation method.
For ~25 sampling points it takes 10–30 seconds. The critical point estimate
is a midpoint between the two branch endpoints; it's a reasonable indicator
but not a rigorously located critical.

The flash calculator can be used independently of the experiments — change
T or P inside the tab and re-flash without recomputing tables.
