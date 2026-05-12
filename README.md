# PVT Studio — Equinor-themed PVT Application

Full-featured petroleum-engineering PVT analysis tool. Six analysis modes,
correlation and EOS approaches, lab experiments, phase envelopes, EOS tuning,
multi-region export, multi-stage separators, Monte Carlo uncertainty,
hydrate likelihood screening, and ECLIPSE keyword export in FIELD or METRIC units.

## Run
```bash
pip install -r requirements.txt
streamlit run pvt_app.py
```

## Modes (sidebar selector)
1. **Oil (Black Oil)** — Standing / Vasquez-Beggs / Glaso / Lasater correlations
   plus optional companion PVDG export, CCE experiment, and Monte Carlo
   uncertainty analysis with tornado plot.
2. **Dry Gas** — Hall-Yarborough / Dranchuk-Abou-Kassem Z-factor with
   Wichert-Aziz sour-gas correction.
3. **Wet Gas / Condensate** — McCain recombination + linear-Pdew Rv model,
   plus optional companion PVTO export for dropped-out condensate.
4. **Water** — McCain, Meehan, Numbere, Spivey-Valko-McCain correlations.
5. **Compositional (EOS)** — full Peng-Robinson with 9 sub-tabs:
   *Lab Experiments • Phase Envelope • Flash Calculator • Separator Train •
   EOS Tuning • Multi-Region • Monte Carlo • Docs • ECLIPSE Export*.
6. **❄️ Hydrate Likelihood** — Makogon hydrate envelope, P-T diagram with
   operating-point indicator, traffic-light risk banner, and Hammerschmidt
   inhibitor concentration recommendation.

## Featured this revision

### Bug fix
- **styled_dataframe crash** on string columns (newer pandas reports
  dtype as `str`, not `object`). Switched to `pd.api.types.is_numeric_dtype`
  and Streamlit `column_config` formatting.

### ECLIPSE export
- Separate **FIELD / METRIC** unit toggle inside the Export tab,
  independent of the display-unit toggle in the sidebar.
- **DENSITY keyword** correctly generated and included in all exports
  (single-region and multi-region), with real surface densities computed
  from the EOS standard-conditions split.
- **RSVD / RVVD** vs depth keywords for compositional grading.

### Multi-region PVT (fixed)
- Each PVTNUM region gets its own PVTO/PVTG body stacked under a single
  keyword.
- DENSITY keyword now emits one entry per region using EOS-computed surface
  densities (no more placeholder values).

### Multi-stage separator train
- Configure 1–3 stages with P, T per stage, or load presets.
- Reports per-stage breakdown (vapor mole fraction, oil/gas out),
  total field GOR, ST oil API, combined gas SG.

### ❄️ Hydrate likelihood (new)
- **Makogon (1981) correlation** for hydrate formation P-T envelope.
- **Sour-gas correction**: H2S lowers P_hyd by ~5% per mol%; CO2 by ~1.5% per mol%.
- **Traffic-light risk banner** (red/amber/green) at operating (P, T).
- **P-T diagram** with hydrate locus shaded and operating point marked.
- **Hammerschmidt inhibitor calculator**: methanol, MEG, DEG, TEG wt%
  required to suppress hydrate formation by a chosen ΔT.
- Validity range: 32–75 °F, 0.55 ≤ gas SG ≤ 1.0.

## File map
- `pvt_app.py`                  — Streamlit UI (~2000 lines)
- `hydrate.py`                  — Makogon hydrate prediction + Hammerschmidt
- `theme.py`                    — Equinor colors, CSS, Plotly layout helpers
- `phase_envelope.py`           — Bubble/dew locus tracer
- `experiments.py`              — EOS-based Flash, CCE, CVD, DLE
- `correlation_experiments.py`  — CCE/CVD from black-oil correlations
- `separator.py`                — Multi-stage surface separator flash
- `eos_tuning.py`               — Levenberg-Marquardt EOS regression
- `multi_region.py`             — Multi-region ECLIPSE PVT export
- `monte_carlo.py`              — MC sampling + tornado sensitivity
- `eos_pr.py`                   — Peng-Robinson EOS, stability test, flash, sat-P
- `lbc.py`                      — Lohrenz-Bray-Clark viscosity
- `composition_pvt.py`          — Compositional black-oil table generation
- `components.py`               — Component library + Kesler-Lee C7+
- `correlations.py`             — Oil/gas/wet-gas/water correlations
- `eclipse_export.py`           — ECLIPSE keyword formatters (FIELD + METRIC)
- `units.py`                    — Field ↔ SI display conversion
- `.streamlit/config.toml`      — Streamlit Equinor theme

## Notes
- The DENSITY keyword is required by ECLIPSE when oil and gas phases are
  present in the deck. The app now always emits it with EOS-computed
  surface densities (rather than placeholders).
- Hydrate analysis is a first-pass screening; for sour systems with
  > 15% H2S or temperatures outside 32–75 °F, use a rigorous flash-based
  hydrate model (CSMHyd, PVTsim, Multiflash).
- Phase envelope tracing: 25 points takes 10–30 s; 60 points 1–2 min.
- EOS tuning: 30 iterations × 5 fn-evals each typically 30–90 s.
- Monte Carlo (Oil branch): 1000 samples ≈ 3 s.
