"""
PVT Studio — Documentation & Equation Reference
================================================

Central store of help text and LaTeX equations for every part of the app.
Each function renders a Streamlit help block (markdown + st.latex) for one
topic. Branches call render_help(topic) to drop an expandable reference in.

Keeping this in one module means the equations live in exactly one place and
every branch shows a consistent, citable reference.
"""

import streamlit as st


# ----------------------------------------------------------------------
# Topic registry — maps a short key to a (title, renderer) pair
# ----------------------------------------------------------------------
def render_help(topic, expanded=False):
    """Render an expandable help/equation block for the given topic key."""
    entry = _TOPICS.get(topic)
    if entry is None:
        st.info(f"No documentation found for '{topic}'.")
        return
    title, renderer = entry
    with st.expander(f"📖 {title}", expanded=expanded):
        renderer()


# ----------------------------------------------------------------------
# 1. OIL CORRELATIONS
# ----------------------------------------------------------------------
def _doc_oil():
    st.markdown(
        "Black-oil correlations estimate the solution gas–oil ratio "
        "$R_s$, oil formation volume factor $B_o$, bubble-point pressure "
        "$P_b$, and viscosity $\\mu_o$ from readily available field data "
        "(API gravity, gas specific gravity, temperature, separator GOR). "
        "They are empirical fits to large databases of laboratory PVT "
        "studies and are intended for **screening**, not for final "
        "reservoir simulation input."
    )

    st.markdown("##### Standing (1947)")
    st.markdown(
        "Standing's correlation was fit to 105 California crude samples. "
        "The bubble-point pressure is:")
    st.latex(r"P_b = 18.2\left[\left(\frac{R_s}{\gamma_g}\right)^{0.83}"
             r"\cdot 10^{\,(0.00091\,T - 0.0125\,\text{API})} - 1.4\right]")
    st.markdown(
        "where $R_s$ is in scf/STB, $\\gamma_g$ is gas specific gravity "
        "(air = 1), $T$ is temperature in °F, and $P_b$ is in psia. "
        "The oil formation volume factor at or below $P_b$ is:")
    st.latex(r"B_o = 0.9759 + 12\times10^{-5}\,F^{1.2},\quad "
             r"F = R_s\left(\frac{\gamma_g}{\gamma_o}\right)^{0.5} + 1.25\,T")
    st.markdown("with $\\gamma_o$ the oil specific gravity from API gravity:")
    st.latex(r"\gamma_o = \frac{141.5}{131.5 + \text{API}}")

    st.markdown("##### Vasquez & Beggs (1980)")
    st.markdown(
        "Fit to over 6 000 data points worldwide, with separate "
        "coefficient sets for API ≤ 30 and API > 30. The solution GOR is:")
    st.latex(r"R_s = C_1\,\gamma_{gs}\,P^{C_2}\,"
             r"\exp\!\left(C_3\frac{\text{API}}{T + 460}\right)")
    st.markdown(
        "where $\\gamma_{gs}$ is the gas gravity corrected to a 100 psig "
        "separator reference and $C_1, C_2, C_3$ are API-dependent "
        "constants.")

    st.markdown("##### Glaso (1980)")
    st.markdown(
        "Developed from North Sea crude data — generally the best choice "
        "for North Sea / volatile oils. The bubble point uses a "
        "correlating function $P_b^*$:")
    st.latex(r"\log_{10} P_b^{*} = 1.7669 + 1.7447\log_{10}F"
             r" - 0.30218\,(\log_{10}F)^2")

    st.markdown("##### Oil viscosity")
    st.markdown(
        "Dead-oil viscosity $\\mu_{od}$ (gas-free) is estimated first, "
        "then corrected for dissolved gas. The Beggs–Robinson dead-oil "
        "correlation is:")
    st.latex(r"\mu_{od} = 10^{x} - 1,\quad "
             r"x = 10^{\,3.0324 - 0.02023\,\text{API}}\cdot T^{-1.163}")
    st.markdown(
        "The live-oil (saturated) viscosity applies a Beggs–Robinson "
        "gas-correction factor that always reduces viscosity, since "
        "dissolved gas lubricates the oil.")

    st.caption(
        "Sources: Standing (1947, 1981); Vasquez & Beggs, JPT 1980; "
        "Glaso, JPT 1980; Beggs & Robinson, JPT 1975. "
        "See McCain, *The Properties of Petroleum Fluids*, 2nd ed. (1990).")


# ----------------------------------------------------------------------
# 2. GAS CORRELATIONS
# ----------------------------------------------------------------------
def _doc_gas():
    st.markdown(
        "Dry-gas properties follow from the **real-gas law** "
        "$PV = ZnRT$, where the compressibility factor $Z$ captures the "
        "departure from ideal behaviour. $Z$ is read from the "
        "Standing–Katz chart as a function of pseudo-reduced pressure and "
        "temperature."
    )

    st.markdown("##### Pseudo-reduced properties")
    st.latex(r"P_{pr} = \frac{P}{P_{pc}}, \qquad "
             r"T_{pr} = \frac{T}{T_{pc}}")
    st.markdown(
        "The pseudo-critical $P_{pc}$ and $T_{pc}$ are estimated from gas "
        "gravity via Sutton's correlation, then corrected for "
        "non-hydrocarbons.")

    st.markdown("##### Z-factor — Hall & Yarborough (1973)")
    st.markdown(
        "An equation-of-state fit to the Standing–Katz chart. $Z$ is "
        "obtained from the reduced density $y$, found by solving:")
    st.latex(r"-\,\alpha P_{pr} + \frac{y + y^2 + y^3 - y^4}{(1-y)^3}"
             r" - \beta y^2 + \gamma\, y^{\delta} = 0")
    st.latex(r"Z = \frac{\alpha\, P_{pr}}{y}")
    st.markdown(
        "where $\\alpha,\\beta,\\gamma,\\delta$ are functions of the "
        "reciprocal reduced temperature $t = 1/T_{pr}$. The equation is "
        "solved by Newton iteration.")

    st.markdown("##### Z-factor — Dranchuk & Abou-Kassem (1975)")
    st.markdown(
        "An 11-constant equation of state, also fit to Standing–Katz. "
        "It is valid over $0.2 \\le P_{pr} < 30$ and "
        "$1.0 < T_{pr} \\le 3.0$ and is the more widely used modern "
        "correlation.")

    st.markdown("##### Wichert–Aziz sour-gas correction")
    st.markdown(
        "$\\mathrm{H_2S}$ and $\\mathrm{CO_2}$ shift the pseudo-criticals. "
        "The correction factor $\\varepsilon$ is applied as:")
    st.latex(r"T_{pc}' = T_{pc} - \varepsilon, \qquad "
             r"P_{pc}' = \frac{P_{pc}\,T_{pc}'}"
             r"{T_{pc} + B(1-B)\,\varepsilon}")
    st.markdown(
        "where $B$ is the $\\mathrm{H_2S}$ mole fraction. Note that "
        "lowering $T_{pc}$ *raises* $T_{pr}$, which can move $Z$ either "
        "way depending on where the gas sits relative to its critical "
        "point.")

    st.markdown("##### Gas formation volume factor")
    st.latex(r"B_g = 0.02827\,\frac{Z\,T}{P} \quad [\text{rcf/scf}]")
    st.markdown(
        "with $T$ in °R and $P$ in psia. Divide by 5.615 cuft/bbl to "
        "express $B_g$ in rb/scf.")

    st.markdown("##### Gas viscosity — Lee–Gonzalez–Eakin (1966)")
    st.latex(r"\mu_g = 10^{-4}\,K\,\exp\!\left(X\,\rho_g^{\,Y}\right)")
    st.markdown(
        "where $\\rho_g$ is the gas density and $K, X, Y$ depend on gas "
        "molecular weight and temperature.")

    st.caption(
        "Sources: Hall & Yarborough, OGJ 1973; Dranchuk & Abou-Kassem, "
        "JCPT 1975; Wichert & Aziz, 1972; Lee, Gonzalez & Eakin, "
        "JPT 1966; Sutton, SPE 1985.")


# ----------------------------------------------------------------------
# 3. WET GAS / CONDENSATE
# ----------------------------------------------------------------------
def _doc_wetgas():
    st.markdown(
        "A wet gas (or gas condensate) produces liquid at surface "
        "conditions. The key extra property is the **vaporized oil–gas "
        "ratio** $R_v$ — the volume of condensate carried per volume of "
        "produced gas — and the **dew-point pressure** $P_{dew}$ below "
        "which liquid drops out in the reservoir."
    )

    st.markdown("##### Recombined reservoir gas gravity")
    st.markdown(
        "Surface gas and condensate are recombined into a single "
        "reservoir-gas gravity:")
    st.latex(r"\gamma_{g,res} = "
             r"\frac{R\,\gamma_g + 4584\,\gamma_{cond}}{R + V_{eq}}")
    st.markdown(
        "where $R$ is the producing gas–condensate ratio, "
        "$\\gamma_{cond}$ the condensate gravity, and $V_{eq}$ the "
        "condensate vapour equivalent (scf/STB).")

    st.markdown("##### Condensate yield model")
    st.markdown(
        "Above the dew point all condensate is vaporized, so "
        "$R_v = R_{v,max}$ (set by the CGR). Below $P_{dew}$, $R_v$ "
        "declines roughly linearly toward a small residual as liquid "
        "drops out:")
    st.latex(r"R_v(P) = R_{v,max}\left(0.05 + 0.95\,"
             r"\frac{P}{P_{dew}}\right), \quad P < P_{dew}")
    st.markdown(
        "The CGR (condensate–gas ratio, STB/MMscf) sets "
        "$R_{v,max} = \\text{CGR} / 10^6$ in STB/scf.")

    st.caption(
        "Wet-gas / condensate behaviour: see McCain ch. 6, and "
        "Whitson & Brulé, *Phase Behavior* (SPE Monograph 20, 2000).")


# ----------------------------------------------------------------------
# 4. WATER / BRINE
# ----------------------------------------------------------------------
def _doc_water():
    st.markdown(
        "Brine PVT properties depend on pressure, temperature, salinity, "
        "and dissolved gas. The formation volume factor $B_w$ stays close "
        "to 1.0 because water is nearly incompressible."
    )

    st.markdown("##### Water formation volume factor — McCain")
    st.latex(r"B_w = (1 + \Delta V_{wP})(1 + \Delta V_{wT})")
    st.markdown(
        "where $\\Delta V_{wT}$ is the thermal expansion term and "
        "$\\Delta V_{wP}$ the (negative) pressure-compression term, each "
        "a polynomial in $T$ and $P$.")

    st.markdown("##### Water compressibility")
    st.latex(r"c_w = \frac{1}{A_1 + A_2 T + A_3 T^2}")
    st.markdown(
        "Dissolved gas increases $c_w$; salinity decreases it slightly. "
        "Typical reservoir values are $3$–$6\\times10^{-6}$ psi⁻¹.")

    st.markdown("##### Solution gas in brine — and viscosity")
    st.markdown(
        "Gas solubility $R_{sw}$ rises with pressure and falls with "
        "salinity. Brine viscosity is the pure-water viscosity scaled by "
        "a salinity factor and a pressure factor:")
    st.latex(r"\mu_w = \mu_{w1}\left(A + B\,T^{C}\right)"
             r"\cdot f_P(P)")

    st.caption(
        "Sources: McCain (1990); Meehan (1980); Numbere et al. (1977); "
        "Spivey, McCain & North (2004).")


# ----------------------------------------------------------------------
# 5. PENG-ROBINSON EOS
# ----------------------------------------------------------------------
def _doc_eos():
    st.markdown(
        "The **Peng–Robinson (1976) equation of state** is a cubic EOS "
        "that models both liquid and vapour phases from a single "
        "equation. Compositional PVT uses it for flash calculations, "
        "saturation pressures, and phase envelopes."
    )

    st.markdown("##### The cubic equation")
    st.latex(r"P = \frac{RT}{V - b} - "
             r"\frac{a\,\alpha(T)}{V(V+b) + b(V-b)}")
    st.markdown(
        "In terms of the compressibility factor $Z = PV/RT$ it becomes a "
        "cubic:")
    st.latex(r"Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z"
             r" - (AB - B^2 - B^3) = 0")
    st.markdown(
        "where $A = aP/(RT)^2$ and $B = bP/RT$. The largest root is the "
        "vapour $Z$, the smallest the liquid $Z$.")

    st.markdown("##### Component parameters")
    st.latex(r"a_i = 0.45724\,\frac{R^2 T_{ci}^2}{P_{ci}}, \qquad "
             r"b_i = 0.07780\,\frac{R\,T_{ci}}{P_{ci}}")
    st.latex(r"\alpha_i = \left[1 + \kappa_i"
             r"\left(1 - \sqrt{T_{ri}}\right)\right]^2")
    st.markdown(
        "with $\\kappa_i = 0.37464 + 1.54226\\,\\omega_i "
        "- 0.26992\\,\\omega_i^2$ a function of the acentric factor "
        "$\\omega_i$.")

    st.markdown("##### Mixing rules")
    st.latex(r"a\alpha = \sum_i\sum_j x_i x_j (1 - k_{ij})"
             r"\sqrt{a_i\alpha_i\,a_j\alpha_j}, \qquad "
             r"b = \sum_i x_i b_i")
    st.markdown(
        "The binary interaction coefficients $k_{ij}$ are the primary "
        "tuning handles, especially $k_{ij}$ between methane and the "
        "C7+ fraction.")

    st.markdown("##### Vapour–liquid equilibrium")
    st.markdown(
        "Equilibrium requires equal fugacity for each component in both "
        "phases, $f_i^L = f_i^V$, expressed through equilibrium ratios "
        "$K_i = y_i / x_i$. The phase split $V$ solves the "
        "Rachford–Rice objective:")
    st.latex(r"\sum_i \frac{z_i (K_i - 1)}{1 + V(K_i - 1)} = 0")
    st.markdown(
        "Phase stability is checked with the Michelsen tangent-plane "
        "distance test before the flash is accepted.")

    st.caption(
        "Sources: Peng & Robinson, I&EC Fund. 1976; Michelsen, "
        "Fluid Phase Equilibria 1982; Whitson & Brulé, *Phase Behavior* "
        "(2000).")


# ----------------------------------------------------------------------
# 6. C7+ CHARACTERIZATION
# ----------------------------------------------------------------------
def _doc_c7plus():
    st.markdown(
        "The **C7+ (heptanes-plus) fraction** lumps all heavy "
        "components. Its critical properties are not measured directly "
        "but estimated from molecular weight and specific gravity using "
        "the Kesler–Lee correlations."
    )
    st.markdown("##### Kesler–Lee critical properties")
    st.markdown(
        "The normal boiling point $T_b$ is first estimated from MW and "
        "SG, then the pseudo-critical temperature and pressure are:")
    st.latex(r"T_c = 341.7 + 811\,\gamma + (0.4244 + 0.1174\,\gamma)T_b"
             r" + \frac{(0.4669 - 3.2623\,\gamma)\times10^5}{T_b}")
    st.latex(r"\ln P_c = 8.3634 - \frac{0.0566}{\gamma} - \dots")
    st.markdown(
        "The acentric factor $\\omega$ follows from the Lee–Kesler "
        "vapour-pressure correlation. These C7+ properties — $P_c$, "
        "$T_c$, $\\omega$ — are exactly what EOS tuning adjusts.")
    st.caption("Sources: Kesler & Lee, Hydrocarbon Processing 1976; "
                "Lee & Kesler, AIChE J. 1975.")


# ----------------------------------------------------------------------
# 7. LAB EXPERIMENTS
# ----------------------------------------------------------------------
def _doc_experiments():
    st.markdown(
        "Standard laboratory PVT experiments characterize how a fluid "
        "behaves as pressure declines. The app reproduces them either "
        "from correlations or from the EOS."
    )
    st.markdown("##### CCE — Constant Composition Expansion")
    st.markdown(
        "The cell composition is **fixed**; pressure is lowered and the "
        "total volume recorded. Reports the relative volume "
        "$V/V_{sat}$ and, for oils, the Y-function:")
    st.latex(r"Y = \frac{P_{sat} - P}{P\,(V/V_{sat} - 1)}")

    st.markdown("##### CVD — Constant Volume Depletion")
    st.markdown(
        "Gas is **removed** at each pressure step to hold the cell "
        "volume constant — this mimics depletion of a real reservoir. "
        "For a dry gas the recovery factor follows the P/Z material "
        "balance:")
    st.latex(r"RF = 1 - \frac{P/Z}{P_i/Z_i}")

    st.markdown("##### DLE — Differential Liberation Experiment")
    st.markdown(
        "Below the bubble point, liberated gas is removed at each step "
        "(differential process). Tracks $R_s$, $B_o$ and oil density as "
        "the oil is progressively stripped of gas.")

    st.markdown("##### Separator test")
    st.markdown(
        "A multi-stage separator flash determines surface GOR and oil "
        "shrinkage. A staged train recovers more liquid than a single "
        "flash, so the total GOR is lower than a single-stage flash.")

    st.caption("See McCain ch. 5, and the API RP-44 sampling guidelines.")


# ----------------------------------------------------------------------
# 8. CORRELATION & EOS TUNING
# ----------------------------------------------------------------------
def _doc_tuning():
    st.markdown(
        "Tuning adjusts a small set of parameters so model predictions "
        "match laboratory measurements. The objective is a weighted sum "
        "of normalized squared residuals:"
    )
    st.latex(r"\Phi = \sum_k w_k\left(\frac{y_k^{pred} - y_k^{obs}}"
             r"{y_k^{obs}}\right)^2")
    st.markdown(
        "minimized over the chosen tuning parameters. Normalizing by "
        "$y_k^{obs}$ lets measurements of different magnitude (a 3000 "
        "psia $P_b$ and a 1.3 $B_o$) carry comparable weight.")

    st.markdown("##### Correlation tuning")
    st.markdown(
        "For correlations the adjustable parameters are simple "
        "multipliers and shifts — for oil: a bubble-point shift "
        "$\\Delta P_b$ and factors on $R_s$, $B_o$, $\\mu_o$. These keep "
        "the correlation's shape but slide it onto the lab data.")

    st.markdown("##### EOS tuning")
    st.markdown(
        "For the EOS the regression variables are the C7+ critical "
        "properties ($P_c$, $T_c$, $\\omega$ multipliers) and the "
        "binary interaction coefficients $k_{ij}$. A "
        "Levenberg–Marquardt least-squares solver is used. EOS tuning is "
        "more powerful but can over-fit — change as few parameters as "
        "possible.")

    st.markdown("##### Reading the result")
    st.markdown(
        "Compare the **RMS before and after**: a large reduction means "
        "the fit improved. If RMS does not drop, the measurements may be "
        "inconsistent, or more parameters / iterations are needed. The "
        "Undo button discards a tuning that didn't help.")

    st.caption(
        "Tuning workflow: Coats & Smart, SPERE 1986; Whitson & Brulé "
        "(2000) ch. 9.")


# ----------------------------------------------------------------------
# 9. MONTE CARLO
# ----------------------------------------------------------------------
def _doc_monte_carlo():
    st.markdown(
        "Monte Carlo analysis propagates **input uncertainty** to the "
        "outputs. Each uncertain input is sampled from a normal "
        "distribution $\\mathcal{N}(\\mu, \\sigma)$, the model is run for "
        "every draw, and the spread of results is summarized."
    )
    st.markdown(
        "Percentiles **P10 / P50 / P90** describe the output "
        "distribution: P90 is the value exceeded 10% of the time on the "
        "low side (a conservative estimate), P10 the optimistic end.")
    st.markdown("##### Tornado sensitivity")
    st.markdown(
        "A tornado chart isolates one input at a time: each input is set "
        "to its $\\mu \\pm 1\\sigma$ values while the others stay at "
        "their mean. The bar length shows how much that single input "
        "moves the output — the widest bar is the dominant uncertainty.")
    st.caption("A standard probabilistic screening method — see SPE "
                "guidelines on uncertainty quantification.")


# ----------------------------------------------------------------------
# 10. HYDRATES
# ----------------------------------------------------------------------
def _doc_hydrate():
    st.markdown(
        "Gas hydrates are ice-like solids that form when light "
        "hydrocarbon gas and water combine at **high pressure and low "
        "temperature** — a flow-assurance hazard in subsea pipelines."
    )
    st.markdown("##### Hydrate formation pressure — Makogon")
    st.markdown(
        "The hydrate-formation pressure depends on gas gravity and "
        "temperature. Makogon's correlation has the log-linear form:")
    st.latex(r"\log_{10} P = \beta + 0.0497\,(T + k\,T^2) - 1")
    st.markdown(
        "where $\\beta$ and $k$ are functions of gas specific gravity "
        "and $T$ is in °C. Sour components ($\\mathrm{H_2S}$, "
        "$\\mathrm{CO_2}$) shift the curve toward easier hydrate "
        "formation.")

    st.markdown("##### Inhibitor effect — Hammerschmidt")
    st.markdown(
        "Thermodynamic inhibitors (methanol, MEG, etc.) depress the "
        "hydrate-formation temperature. The Hammerschmidt equation "
        "relates the temperature depression $\\Delta T$ to the inhibitor "
        "weight-percent $W$ in the aqueous phase:")
    st.latex(r"\Delta T = \frac{K_H\, W}{M\,(100 - W)}")
    st.markdown(
        "where $M$ is the inhibitor molecular weight and $K_H$ the "
        "Hammerschmidt constant ($\\approx 2335$ for methanol, "
        "$2222$ for MEG). A higher inhibitor concentration shifts the "
        "whole hydrate curve to lower temperature.")

    st.markdown("##### Subsea cooldown")
    st.markdown(
        "After a shut-in, an insulated pipeline cools toward the seabed "
        "temperature. The cooldown time is how long until the fluid "
        "reaches the hydrate-formation temperature — the window "
        "available to act before remediation is needed.")

    st.caption(
        "Sources: Makogon (1981); Hammerschmidt, I&EC 1934; "
        "Sloan & Koh, *Clathrate Hydrates of Natural Gases*, 3rd ed.")


# ----------------------------------------------------------------------
# 11. ROCK COMPRESSIBILITY
# ----------------------------------------------------------------------
def _doc_rock():
    st.markdown(
        "Pore-volume (formation) compressibility $c_f$ measures how much "
        "the rock pore space shrinks as reservoir pressure declines. It "
        "contributes to the drive energy of the reservoir."
    )
    st.markdown("##### Definition")
    st.latex(r"c_f = \frac{1}{\phi}"
             r"\left(\frac{\partial \phi}{\partial P}\right)_T")
    st.markdown(
        "with $\\phi$ the porosity. Typical consolidated-sandstone "
        "values are $3$–$6\\times10^{-6}$ psi⁻¹; unconsolidated rock "
        "can be an order of magnitude higher.")

    st.markdown("##### Correlations")
    st.markdown(
        "Hall (1953) gives $c_f$ as a function of porosity alone. Newman "
        "(1973) provides separate fits for consolidated sandstone and "
        "limestone. These are empirical envelopes — laboratory core "
        "measurement is always preferred when available.")

    st.markdown("##### Compaction model")
    st.markdown(
        "ECLIPSE applies pressure-dependent pore-volume compaction "
        "through a multiplier table. The linear model is:")
    st.latex(r"\text{PV mult}(P) = 1 + c_f\,(P - P_{ref})")
    st.markdown("and the exponential model:")
    st.latex(r"\text{PV mult}(P) = \exp\!\left[c_f\,(P - P_{ref})\right]")
    st.markdown(
        "exported via the ROCK or ROCKTAB keyword.")

    st.caption("Sources: Hall, JPT 1953; Newman, JPT 1973.")


# ----------------------------------------------------------------------
# 12. ECLIPSE EXPORT
# ----------------------------------------------------------------------
def _doc_eclipse():
    st.markdown(
        "The app generates **ECLIPSE-format PVT include files** "
        "(`.INC`) — the keyword tables a reservoir simulator reads to "
        "model fluid behaviour."
    )
    st.markdown("##### Keywords")
    st.markdown(
        "- **PVTO** — live-oil table: $R_s$, $P_{sat}$, $B_o$, $\\mu_o$\n"
        "- **PVDG** — dry-gas table: $P$, $B_g$, $\\mu_g$\n"
        "- **PVTG** — wet-gas table: $P$, $R_v$, $B_g$, $\\mu_g$\n"
        "- **PVTW** — water properties at a reference pressure\n"
        "- **DENSITY** — surface densities of oil, water, gas\n"
        "- **ROCK / ROCKTAB** — rock compressibility / compaction")
    st.markdown("##### Units")
    st.markdown(
        "ECLIPSE supports FIELD (psia, scf/STB, rb/STB) and METRIC "
        "(bara, Sm³/Sm³, rm³/Sm³) unit sets. The app converts the whole "
        "deck consistently and writes the matching unit comments.")
    st.markdown("##### Multi-region (PVTNUM)")
    st.markdown(
        "When a reservoir has layers with different fluids, each PVT "
        "region gets its own table stacked under one keyword, ordered by "
        "PVTNUM. The app builds these from per-region correlation inputs "
        "or saved fluids.")
    st.caption(
        "This is a screening tool — always validate generated decks "
        "against rigorous PVT software before simulation use.")


# ----------------------------------------------------------------------
# 13. UNITS
# ----------------------------------------------------------------------
def _doc_units():
    st.markdown(
        "The app works internally in **field units** and converts for "
        "display and export. Key conversions:"
    )
    st.latex(r"P\,[\text{bara}] = P\,[\text{psia}] / 14.50377")
    st.latex(r"T\,[^\circ\text{C}] = (T\,[^\circ\text{F}] - 32)/1.8")
    st.latex(r"R_s\,[\text{Sm}^3/\text{Sm}^3] = "
             r"R_s\,[\text{scf/STB}] / 5.6146")
    st.markdown(
        "Temperature **differences** (e.g. hydrate suppression $\\Delta "
        "T$) convert with the 1.8 scale factor only — no 32° offset. "
        "$B_o$ and $B_g$ ratios and viscosity in cP are the same in both "
        "systems.")
    st.caption("Round-trip conversion is verified in the app's "
                "validation suite.")


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
_TOPICS = {
    "oil":         ("Oil correlations — equations & references", _doc_oil),
    "gas":         ("Gas correlations — equations & references", _doc_gas),
    "wetgas":      ("Wet gas / condensate — equations", _doc_wetgas),
    "water":       ("Water / brine correlations — equations", _doc_water),
    "eos":         ("Peng-Robinson EOS — equations", _doc_eos),
    "c7plus":      ("C7+ characterization — equations", _doc_c7plus),
    "experiments": ("Lab experiments (CCE/CVD/DLE) — explained", _doc_experiments),
    "tuning":      ("Correlation & EOS tuning — method", _doc_tuning),
    "montecarlo":  ("Monte Carlo & tornado — method", _doc_monte_carlo),
    "hydrate":     ("Hydrates & inhibitors — equations", _doc_hydrate),
    "rock":        ("Rock compressibility & compaction — equations", _doc_rock),
    "eclipse":     ("ECLIPSE export — keywords & units", _doc_eclipse),
    "units":       ("Unit system & conversions", _doc_units),
}


def render_full_reference():
    """Render the complete equation reference — used by a dedicated
    documentation page / section."""
    st.markdown("## 📚 PVT Studio — Equation Reference")
    st.markdown(
        "Complete reference for every model in the app. Each correlation "
        "and method below is also available as an inline help block in "
        "its branch."
    )
    order = ["units", "oil", "gas", "wetgas", "water", "eos", "c7plus",
             "experiments", "tuning", "montecarlo", "hydrate", "rock",
             "eclipse"]
    for key in order:
        title, renderer = _TOPICS[key]
        st.markdown(f"### {title}")
        renderer()
        st.markdown("---")
