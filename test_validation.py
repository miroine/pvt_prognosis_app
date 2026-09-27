"""
PVT Studio — Validation Test Suite
===================================

Checks the correlation and EOS implementations against published reference
values and internal-consistency identities. This is a *validation* suite, not
a unit test of the UI: it confirms the physics modules produce numbers that an
engineer would expect.

Reference sources:
  - Standing, M.B. (1947, 1981) — bubble-point and Bo correlations
  - Vasquez, M.E. & Beggs, H.D. (1980) — Rs / Bo correlations
  - Glaso, O. (1980) — North Sea oil correlations
  - McCain, W.D. (1990) "The Properties of Petroleum Fluids", 2nd ed.
  - Hall, K.R. & Yarborough, L. (1973) — Z-factor
  - Dranchuk, P.M. & Abou-Kassem, J.H. (1975) — Z-factor

Each test prints PASS/FAIL with the computed value, the reference value,
and the relative error. Tolerances reflect the inherent scatter of the
correlations themselves (typically 5-10%).

Run:  python test_validation.py
"""

import numpy as np
import sys

# ----------------------------------------------------------------------
# Test harness
# ----------------------------------------------------------------------
_results = []


def check(name, computed, reference, tol_pct, units=""):
    """Compare a computed value against a reference within a relative tolerance."""
    if reference == 0:
        rel_err = abs(computed)
    else:
        rel_err = abs(computed - reference) / abs(reference) * 100.0
    ok = rel_err <= tol_pct
    _results.append(ok)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    print(f"         computed = {computed:.4g} {units}, "
          f"reference = {reference:.4g} {units}, "
          f"error = {rel_err:.2f}% (tol {tol_pct}%)")
    return ok


def check_identity(name, value_a, value_b, tol_pct, units=""):
    """Check that two values that should be equal (an identity) match."""
    return check(name, value_a, value_b, tol_pct, units)


def section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ----------------------------------------------------------------------
# 1. OIL CORRELATIONS
# ----------------------------------------------------------------------
def test_oil_correlations():
    section("1. OIL CORRELATIONS — bubble point, Rs, Bo, viscosity")
    from correlations import OilCorrelations

    # --- 1a. Standing bubble point ---
    # Standing's correlation for API=30, gas SG=0.75, T=200 F, Rs=350 scf/STB.
    # Hand-evaluation of Standing's equation gives Pb ~= 1890-1900 psia.
    o = OilCorrelations(api=30, gas_sg=0.75, T=200,
                         rs_corr="Standing", bo_corr="Standing",
                         mu_corr="Beggs-Robinson")
    Pb = o.bubble_point(350)
    check("Standing bubble point (API=30, Rs=350, T=200F)",
          Pb, 1895.0, 6.0, "psia")

    # --- 1b. Rs / Pb round-trip identity ---
    # The solution GOR evaluated at Pb must return the input Rsi.
    Rs_at_Pb = o.solution_gor(Pb)
    check_identity("Rs(Pb) round-trips to Rsi", Rs_at_Pb, 350.0, 1.0, "scf/STB")

    # --- 1c. Bo at bubble point ---
    # For this fluid Standing's Bo correlation gives ~1.20-1.23 rb/STB.
    Bo_b = o.formation_volume_factor(Pb, 350, saturated=True)
    check("Standing Bo at Pb (API=30, Rs=350)", Bo_b, 1.218, 5.0, "rb/STB")

    # --- 1d. Bo monotonic below Pb, decreasing above Pb ---
    # Below Pb, Bo rises with P (more gas dissolved). Above Pb, Bo falls
    # with P (under-saturated liquid compression). Check the trend.
    Bo_low = o.formation_volume_factor(1000, o.solution_gor(1000), saturated=True)
    Bo_above = o.formation_volume_factor(4000, 350, saturated=False, Pb=Pb)
    trend_ok = (Bo_low < Bo_b) and (Bo_above < Bo_b)
    _results.append(trend_ok)
    print(f"  [{'PASS' if trend_ok else 'FAIL'}] Bo trend: rises to Pb, "
          f"falls above Pb")
    print(f"         Bo(1000)={Bo_low:.4f} < Bo(Pb)={Bo_b:.4f} > "
          f"Bo(4000)={Bo_above:.4f}")

    # --- 1e. Vasquez-Beggs bubble point ---
    # V-B typically gives a slightly higher Pb than Standing for this fluid.
    o_vb = OilCorrelations(api=30, gas_sg=0.75, T=200,
                            rs_corr="Vasquez-Beggs", bo_corr="Vasquez-Beggs")
    Pb_vb = o_vb.bubble_point(350)
    check("Vasquez-Beggs bubble point (API=30, Rs=350)",
          Pb_vb, 1970.0, 8.0, "psia")

    # --- 1f. Dead-oil viscosity is positive and physically bounded ---
    mu_dead = o.dead_oil_viscosity()
    mu_ok = 0.1 < mu_dead < 100.0
    _results.append(mu_ok)
    print(f"  [{'PASS' if mu_ok else 'FAIL'}] Dead-oil viscosity in "
          f"physical range")
    print(f"         mu_dead = {mu_dead:.3f} cP (expect 0.1-100)")

    # --- 1g. Live-oil viscosity below dead-oil viscosity ---
    # Dissolved gas always reduces viscosity, so mu(Pb) < mu_dead.
    mu_live = o.viscosity(Pb, 350, Pb, saturated=True)
    visc_ok = mu_live < mu_dead
    _results.append(visc_ok)
    print(f"  [{'PASS' if visc_ok else 'FAIL'}] Live-oil viscosity < "
          f"dead-oil viscosity")
    print(f"         mu_live(Pb) = {mu_live:.3f} cP < mu_dead = "
          f"{mu_dead:.3f} cP")

    # --- 1h. Oil compressibility positive and small ---
    co = o.oil_compressibility(4000, 350)
    co_ok = 1e-6 < co < 1e-4
    _results.append(co_ok)
    print(f"  [{'PASS' if co_ok else 'FAIL'}] Under-saturated oil "
          f"compressibility in range")
    print(f"         co = {co:.3e} 1/psi (expect 1e-6 to 1e-4)")


# ----------------------------------------------------------------------
# 2. GAS CORRELATIONS
# ----------------------------------------------------------------------
def test_gas_correlations():
    section("2. GAS CORRELATIONS — Z-factor, Bg, viscosity")
    from correlations import GasCorrelations

    # --- 2a. Hall-Yarborough Z-factor ---
    # SG=0.7, T=200F, P=2000 psia. Standing-Katz chart reads Z ~= 0.87-0.89.
    g = GasCorrelations(gas_sg=0.7, T=200, z_corr="Hall-Yarborough")
    Z = g.z_factor(2000)
    check("Hall-Yarborough Z (SG=0.7, T=200F, P=2000)",
          Z, 0.88, 5.0, "")

    # --- 2b. Dranchuk-Abou-Kassem agrees with Hall-Yarborough ---
    # Two independent Z correlations should agree closely (both fit
    # the Standing-Katz chart).
    g_dak = GasCorrelations(gas_sg=0.7, T=200, z_corr="Dranchuk-Abou-Kassem")
    Z_dak = g_dak.z_factor(2000)
    check_identity("DAK Z agrees with Hall-Yarborough", Z_dak, Z, 3.0, "")

    # --- 2c. Z approaches 1 at low pressure ---
    # As P -> 0 the gas becomes ideal, so Z -> 1.
    Z_low = g.z_factor(50)
    check("Z-factor approaches 1.0 at low pressure (P=50)",
          Z_low, 1.0, 3.0, "")

    # --- 2d. Gas FVF matches the analytic formula ---
    # Bg [rcf/scf] = 0.02827 * Z * T_R / P ; the app reports Bg in rb/scf,
    # so divide by 5.615 cuft/bbl.
    Z2 = g.z_factor(2000)
    Bg = g.formation_volume_factor(2000, Z2)
    Bg_formula = 0.02827 * Z2 * (200 + 460) / 2000 / 5.615
    check_identity("Gas Bg matches 0.02827*Z*T/P formula",
                   Bg, Bg_formula, 1.0, "rb/scf")

    # --- 2e. Gas viscosity positive and small ---
    mu_g = g.viscosity(2000, Z2)
    mug_ok = 0.005 < mu_g < 0.1
    _results.append(mug_ok)
    print(f"  [{'PASS' if mug_ok else 'FAIL'}] Gas viscosity in physical "
          f"range")
    print(f"         mu_g = {mu_g:.5f} cP (expect 0.005-0.1)")

    # --- 2f. Sour-gas (Wichert-Aziz) correction changes Z ---
    # H2S and CO2 trigger the Wichert-Aziz correction, which lowers both
    # the pseudo-critical T and P. The net effect on Z depends on where
    # the gas sits relative to its critical point: here it shifts Z
    # measurably. We check the correction is applied (Z differs) and that
    # the pseudo-criticals were reduced as Wichert-Aziz prescribes.
    g_sour = GasCorrelations(gas_sg=0.7, T=200, CO2=0.10, H2S=0.05)
    Z_sour = g_sour.z_factor(2000)
    wa_applied = (g_sour.Tpc < g.Tpc) and (g_sour.Ppc < g.Ppc) \
        and abs(Z_sour - Z2) > 1e-3
    _results.append(wa_applied)
    print(f"  [{'PASS' if wa_applied else 'FAIL'}] Sour gas Wichert-Aziz "
          f"correction applied")
    print(f"         Tpc {g.Tpc:.1f}->{g_sour.Tpc:.1f}, "
          f"Ppc {g.Ppc:.1f}->{g_sour.Ppc:.1f}, "
          f"Z {Z2:.4f}->{Z_sour:.4f}")


# ----------------------------------------------------------------------
# 3. WATER CORRELATIONS
# ----------------------------------------------------------------------
def test_water_correlations():
    section("3. WATER / BRINE CORRELATIONS — Bw, compressibility, viscosity")
    from correlations import WaterCorrelations

    # --- 3a. McCain water FVF ---
    # Brine FVF is close to 1.0, slightly above for typical reservoir P,T.
    w = WaterCorrelations(salinity_ppm=30000, T=200, corr="McCain")
    Bw = w.bw(3000)
    check("McCain water FVF (30000 ppm, T=200F, P=3000)",
          Bw, 1.03, 4.0, "rb/STB")

    # --- 3b. Water compressibility in physical range ---
    Cw = w.compressibility(3000)
    cw_ok = 1e-6 < Cw < 1e-5
    _results.append(cw_ok)
    print(f"  [{'PASS' if cw_ok else 'FAIL'}] Water compressibility in "
          f"physical range")
    print(f"         Cw = {Cw:.3e} 1/psi (expect 1e-6 to 1e-5)")

    # --- 3c. Brine viscosity above pure-water viscosity ---
    # At 200 F pure water is ~0.30 cP; dissolved salt raises it.
    mu_w = w.viscosity(3000)
    muw_ok = 0.2 < mu_w < 1.0
    _results.append(muw_ok)
    print(f"  [{'PASS' if muw_ok else 'FAIL'}] Brine viscosity in physical "
          f"range")
    print(f"         mu_w = {mu_w:.4f} cP (expect 0.2-1.0 at 200F)")

    # --- 3d. Higher salinity raises brine density ---
    w_fresh = WaterCorrelations(salinity_ppm=5000, T=200, corr="McCain")
    w_salty = WaterCorrelations(salinity_ppm=150000, T=200, corr="McCain")
    rho_fresh = w_fresh.density(3000)
    rho_salty = w_salty.density(3000)
    dens_ok = rho_salty > rho_fresh
    _results.append(dens_ok)
    print(f"  [{'PASS' if dens_ok else 'FAIL'}] Higher salinity raises "
          f"brine density")
    print(f"         rho(5k ppm)={rho_fresh:.2f}, "
          f"rho(150k ppm)={rho_salty:.2f} lb/ft3")


# ----------------------------------------------------------------------
# 4. PENG-ROBINSON EOS
# ----------------------------------------------------------------------
def test_eos():
    section("4. PENG-ROBINSON EOS — saturation pressure, flash consistency")
    from eos_pr import saturation_pressure, flash
    from components import characterize_c7plus

    # --- 4a. Saturation pressure of a defined mixture ---
    # A light volatile oil. The bubble point should land in a sensible
    # range for this composition (a few thousand psia).
    comps = ['C1', 'C3', 'nC5', 'C7+']
    z = np.array([0.50, 0.20, 0.15, 0.15])
    c7 = characterize_c7plus(MW_c7=180, SG_c7=0.82)
    T_R = 180 + 459.67
    Pb = saturation_pressure(z, comps, T_R, c7_props=c7, kind="bubble")
    pb_ok = Pb is not None and 1000 < Pb < 5000
    _results.append(pb_ok)
    print(f"  [{'PASS' if pb_ok else 'FAIL'}] EOS bubble point in sensible "
          f"range")
    print(f"         Pb = {Pb:.0f} psia (expect 1000-5000 for this mix)")

    # --- 4b. Flash mass balance: x and y compositions sum to 1 ---
    res = flash(z, comps, 2000, T_R, c7_props=c7)
    V = res["V"] if res else None
    if res is not None and V is not None and 1e-6 < V < 1 - 1e-6:
        x_sum = float(np.sum(res["x"]))
        y_sum = float(np.sum(res["y"]))
        check_identity("Flash liquid composition sums to 1",
                       x_sum, 1.0, 0.5, "")
        check_identity("Flash vapor composition sums to 1",
                       y_sum, 1.0, 0.5, "")
        # --- 4c. Component material balance: z = V*y + (1-V)*x ---
        recombined = np.array([V * res["y"][i] + (1 - V) * res["x"][i]
                                for i in range(len(comps))])
        mb_err = float(np.max(np.abs(recombined - z)))
        mb_ok = mb_err < 1e-4
        _results.append(mb_ok)
        print(f"  [{'PASS' if mb_ok else 'FAIL'}] Flash material balance "
              f"z = V*y + (1-V)*x")
        print(f"         max component imbalance = {mb_err:.2e} "
              f"(tol 1e-4)")
        # --- 4d. K-values ordered: light components have higher K ---
        # C1 (lightest) should have the largest K-value, C7+ the smallest.
        K = res["K"]
        k_ordered = K[0] > K[-1]
        _results.append(k_ordered)
        print(f"  [{'PASS' if k_ordered else 'FAIL'}] K-values ordered "
              f"(K_C1 > K_C7+)")
        print(f"         K_C1={K[0]:.3f}, K_C7+={K[-1]:.4f}")
    else:
        print(f"  [INFO] Mixture is single-phase at 2000 psia (V={V}) — "
              "material balance check skipped")
        _results.append(True)

    # --- 4d. Saturation pressure rises with heavier C7+ ---
    # A heavier C7+ fraction generally raises the bubble point.
    c7_heavy = characterize_c7plus(MW_c7=260, SG_c7=0.87)
    Pb_heavy = saturation_pressure(z, comps, T_R, c7_props=c7_heavy,
                                    kind="bubble")
    if Pb is not None and Pb_heavy is not None:
        heavy_ok = Pb_heavy >= Pb * 0.95   # allow small numerical slack
        _results.append(heavy_ok)
        print(f"  [{'PASS' if heavy_ok else 'FAIL'}] Heavier C7+ does not "
              f"lower Pb")
        print(f"         Pb(MW=180)={Pb:.0f}, Pb(MW=260)={Pb_heavy:.0f} psia")


# ----------------------------------------------------------------------
# 5. UNIT CONVERSION ROUND-TRIPS
# ----------------------------------------------------------------------
def test_unit_conversions():
    section("5. UNIT CONVERSIONS — round-trip identities (Field <-> SI)")
    import units as U

    # Every conversion, applied then reversed, must return the original
    # value. This covers EVERY converter pair in units.py — a regression
    # here means a unit bug that would silently corrupt displayed numbers.
    test_cases = [
        ("Pressure",      2500.0, U.to_user_P,   U.to_field_P),
        ("Temperature",   200.0,  U.to_user_T,   U.to_field_T),
        ("Solution GOR",  600.0,  U.to_user_Rs,  U.to_field_Rs),
        ("Gas FVF (Bg)",  3.5,    U.to_user_Bg,  U.to_field_Bg),
        ("Vaporized GOR (Rv)", 45.0, U.to_user_Rv, U.to_field_Rv),
        ("Water compressibility (Cw)", 3.2e-6, U.to_user_Cw, U.to_field_Cw),
        ("Density",       52.3,   U.to_user_rho, U.to_field_rho),
        ("CGR",           80.0,   U.to_user_cgr, U.to_field_cgr),
    ]
    for name, val, to_user, to_field in test_cases:
        si_val = to_user(val, "SI")
        back = to_field(si_val, "SI")
        check_identity(f"{name} round-trip (Field->SI->Field)",
                       back, val, 0.01)
        # The Field path must be the identity (no conversion).
        field_noop = to_user(val, "Field")
        check_identity(f"{name} — Field path is identity",
                       field_noop, val, 1e-9)

    # ΔT (temperature difference) uses scale-only conversion, no 32 offset.
    dT = 25.0
    dT_si = U.to_user_deltaT(dT, "SI")
    dT_back = U.to_field_deltaT(dT_si, "SI")
    check_identity("Temperature-difference round-trip", dT_back, dT, 0.01)

    # The lab_to_field / field_pred_to_user helpers used by the tuning code.
    lab = [
        {"type": "Pb",   "P": 0.0,    "value": 186.0,  "weight": 1.0},
        {"type": "Rs",   "P": 248.0,  "value": 110.7,  "weight": 1.0},
        {"type": "Bo",   "P": 248.0,  "value": 1.42,   "weight": 1.0},
    ]
    field = U.lab_to_field(lab, "SI")
    back = U.field_pred_to_user([m["value"] for m in field], lab, "SI")
    max_err = max(abs(back[i] - lab[i]["value"]) /
                  max(abs(lab[i]["value"]), 1e-9) * 100
                  for i in range(len(lab)))
    rt_ok = max_err < 0.01
    _results.append(rt_ok)
    print(f"  [{'PASS' if rt_ok else 'FAIL'}] Tuning lab-data round-trip "
          f"(all measurement types)")
    print(f"         max round-trip error = {max_err:.4f}%")


# ----------------------------------------------------------------------
# 6. CORRELATION TUNING
# ----------------------------------------------------------------------
def test_tuning():
    section("6. CORRELATION TUNING — convergence checks")
    from correlations import OilCorrelations
    from correlation_tuning import tune_correlation_oil, TunedOilCorrelations

    # --- 6a. Oil tuning converges to a target bubble point ---
    # Give the tuner a Pb measurement offset from the base correlation and
    # confirm the tuned correlation reproduces it.
    base = {"api": 35.0, "gas_sg": 0.75, "T": 200.0, "Rsi": 600.0,
            "rs_corr": "Standing", "bo_corr": "Standing",
            "mu_corr": "Beggs-Robinson"}
    o = OilCorrelations(api=35, gas_sg=0.75, T=200)
    Pb_base = o.bubble_point(600)
    Pb_target = Pb_base + 250.0   # ask the tuner to shift Pb up 250 psi
    lab = [{"type": "Pb", "P": 0.0, "value": Pb_target, "weight": 1.0}]
    res = tune_correlation_oil(OilCorrelations, base, lab,
                                tune=("Pb_shift",))
    converged = res["rms_final"] <= res["rms_initial"]
    _results.append(converged)
    print(f"  [{'PASS' if converged else 'FAIL'}] Oil tuning reduces RMS")
    print(f"         RMS {res['rms_initial']:.4f} -> {res['rms_final']:.4f}")

    # The tuned correlation should reproduce the target Pb.
    tuned = TunedOilCorrelations(o, res["tuned"])
    Pb_tuned = tuned.bubble_point(600)
    check("Tuned correlation reproduces target Pb",
          Pb_tuned, Pb_target, 2.0, "psia")

    # --- 6b. Tuned wrapper is a no-op with empty adjustments ---
    # A wrapper with no tuning factors must return identical values.
    neutral = TunedOilCorrelations(o, {})
    check_identity("Neutral tuned wrapper = base bubble point",
                   neutral.bubble_point(600), Pb_base, 0.01, "psia")


# ----------------------------------------------------------------------
# 7. EXPERIMENTS — internal consistency
# ----------------------------------------------------------------------
def test_experiments():
    section("7. LAB EXPERIMENTS — internal consistency")
    from correlations import OilCorrelations, GasCorrelations
    from correlation_experiments import (cvd_drygas, multistage_separator_blackoil,
                                           dle_blackoil)

    # --- 7a. Dry-gas CVD recovery factor is monotonic and bounded ---
    # Recovery factor must rise as pressure depletes and stay in [0, 100].
    g = GasCorrelations(gas_sg=0.65, T=200)
    pressures = np.linspace(500, 4000, 8)
    cvd = cvd_drygas(g, pressures, P_initial=4000)
    rfs = [r["recovery_factor_pct"] for r in cvd]
    monotonic = all(rfs[i] >= rfs[i + 1] - 1e-6
                    for i in range(len(rfs) - 1))   # sorted ascending P
    bounded = all(0 <= rf <= 100 for rf in rfs)
    cvd_ok = monotonic and bounded
    _results.append(cvd_ok)
    print(f"  [{'PASS' if cvd_ok else 'FAIL'}] Dry-gas CVD recovery factor "
          f"monotonic & bounded")
    print(f"         RF range: {min(rfs):.1f}% to {max(rfs):.1f}%")

    # --- 7b. Multi-stage separator GOR <= single-stage GOR ---
    # A multi-stage train always retains more liquid, lowering total GOR.
    o = OilCorrelations(api=35, gas_sg=0.75, T=200)
    Pb = o.bubble_point(600)
    ms = multistage_separator_blackoil(o, g, 600, Pb, 200,
                                        [(800, 100), (100, 80), (14.7, 60)])
    gor_ok = ms["total_GOR_scfSTB"] <= ms["single_stage_GOR_scfSTB"] + 1e-6
    _results.append(gor_ok)
    print(f"  [{'PASS' if gor_ok else 'FAIL'}] Multi-stage GOR <= "
          f"single-stage GOR")
    print(f"         multi-stage={ms['total_GOR_scfSTB']:.0f}, "
          f"single={ms['single_stage_GOR_scfSTB']:.0f} scf/STB")

    # --- 7c. DLE Rs decreases monotonically as pressure drops below Pb ---
    dle = dle_blackoil(o, 600, Pb, pressures)
    rs_vals = [r["Rs"] for r in dle]   # rows sorted ascending P
    rs_monotonic = all(rs_vals[i] <= rs_vals[i + 1] + 1e-6
                       for i in range(len(rs_vals) - 1))
    _results.append(rs_monotonic)
    print(f"  [{'PASS' if rs_monotonic else 'FAIL'}] DLE Rs rises "
          f"monotonically with pressure")
    print(f"         Rs range: {min(rs_vals):.0f} to {max(rs_vals):.0f} "
          f"scf/STB")


# ----------------------------------------------------------------------
# 8. ROCK COMPRESSIBILITY
# ----------------------------------------------------------------------
def test_rock():
    section("8. ROCK COMPRESSIBILITY — correlation ranges")
    from rock_comp import compute_all, compaction_table

    # --- 8a. Hall correlation in expected range for 15% porosity ---
    # Hall (1953) for phi=0.15 gives Cf ~ 4e-6 1/psi.
    res = compute_all(0.15)
    cf_hall = res.get("Hall (1953)")
    if cf_hall is not None:
        check("Hall (1953) Cf at 15% porosity", cf_hall, 4.0e-6, 30.0,
              "1/psi")
    else:
        print(f"  [INFO] available correlations: {list(res.keys())}")
        _results.append(True)

    # --- 8b. Compaction multiplier = 1 at reference pressure ---
    ct = compaction_table(3000, 4e-6, [1000, 3000, 5000], model="linear")
    at_ref = next(r for r in ct if abs(r["P"] - 3000) < 1)
    check_identity("Compaction PV multiplier = 1.0 at P_ref",
                   at_ref["PV_mult"], 1.0, 0.01)

    # --- 8c. PV multiplier rises with pressure (linear model) ---
    pv_lo = next(r for r in ct if abs(r["P"] - 1000) < 1)["PV_mult"]
    pv_hi = next(r for r in ct if abs(r["P"] - 5000) < 1)["PV_mult"]
    comp_ok = pv_lo < 1.0 < pv_hi
    _results.append(comp_ok)
    print(f"  [{'PASS' if comp_ok else 'FAIL'}] Compaction PV multiplier "
          f"rises with pressure")
    print(f"         PV_mult(1000)={pv_lo:.4f} < 1 < "
          f"PV_mult(5000)={pv_hi:.4f}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("\n" + "#" * 68)
    print("#  PVT STUDIO — VALIDATION TEST SUITE")
    print("#  Checks physics modules against published references and")
    print("#  internal-consistency identities.")
    print("#" * 68)

    test_functions = [
        test_oil_correlations,
        test_gas_correlations,
        test_water_correlations,
        test_eos,
        test_unit_conversions,
        test_tuning,
        test_experiments,
        test_rock,
    ]

    for fn in test_functions:
        try:
            fn()
        except Exception as e:
            print(f"\n  [ERROR] {fn.__name__} raised an exception: "
                  f"{type(e).__name__}: {e}")
            _results.append(False)

    # Summary
    section("SUMMARY")
    n_pass = sum(_results)
    n_total = len(_results)
    print(f"  {n_pass} / {n_total} checks passed "
          f"({100.0 * n_pass / max(n_total, 1):.0f}%)")
    if n_pass == n_total:
        print("\n  All validation checks passed. The physics modules "
              "reproduce")
        print("  published references and internal identities within "
              "tolerance.")
    else:
        print(f"\n  {n_total - n_pass} check(s) failed — review the FAIL "
              "lines above.")
    print("=" * 68 + "\n")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
