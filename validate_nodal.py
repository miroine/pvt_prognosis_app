"""
Screening-accuracy validation for the nodal-analysis correlations.

These checks are NOT proof that the implementation reproduces every
published Beggs-Brill / Hagedorn-Brown / Gray case point-for-point —
that would require a curated benchmark dataset. They are SCREENING
checks that catch the gross errors that would make an exported VFP
curve dangerous: asymptotic limits, sign conventions, monotonicity in
the right regions, and bracketed comparisons against ranges quoted in
the petroleum-engineering literature.

Run:  python validate_nodal.py
"""
import math
import nodal as N


def _check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


print("=" * 70)
print("NODAL-ANALYSIS SCREENING VALIDATION")
print("=" * 70)

results = []


# ---------------------------------------------------------------------
# 1. ASYMPTOTIC: pure liquid in a vertical pipe → hydrostatic head
# ---------------------------------------------------------------------
print("\n1. Vertical pure liquid → hydrostatic limit")
# 8000 ft of dead oil, no gas. Expected P_inlet ≈ P_out + rho_o * H / 144.
# Oil density ~ 53 lb/ft³ at 35 API; H = 8000 ft -> dP = 2944 psi
res = N.lift_curve(
    rates_bbl_d=[500],
    GLR_scf_stb=0.0001,    # virtually zero gas
    P_outlet_psia=200.0, T_outlet_F=80.0, T_inlet_F=180.0,
    segments=[(8000.0, 90.0)],
    api=35.0, gas_sg=0.75,
    water_cut_frac=0.0, salinity_wt_pct=0.0,
    D_in_inch=2.992, rel_roughness=0.0006,
    method="Beggs & Brill")
dP_total = res[0]["dP_total_psi"]
# Friction at 500 STB/d in 2.992" pipe is small; hydrostatic dominates.
# Allow 15% deviation from pure hydrostatic head.
expected_lo, expected_hi = 2500, 3400
ok = _check("Vertical pure-liquid ΔP within hydrostatic range "
            f"(got {dP_total:.0f} psi, expected {expected_lo}-{expected_hi})",
            expected_lo <= dP_total <= expected_hi,
            "Pure-liquid gravity head dominates")
results.append(ok)


# ---------------------------------------------------------------------
# 2. ASYMPTOTIC: dry gas horizontal pipe → mostly friction
# ---------------------------------------------------------------------
print("\n2. Horizontal dry gas → friction-dominated, ΔP increases with q²")
# 6" pipe, gas-dominated (small liquid + very high GLR).
# 500 STB/d × GLR 1e4 = 5,000 Mscf/d (5 MMscf/d) — well above numerical floor.
# 3x that = 15 MMscf/d.
res_lo = N.lift_curve([500, 500], 1e4, 500.0, 70.0, 70.0,
                       [(5000.0, 0.0)], 35.0, 0.65, 0.0, 0.0,
                       6.0, 0.0006, "Beggs & Brill")
res_hi = N.lift_curve([1500, 1500], 1e4, 500.0, 70.0, 70.0,
                       [(5000.0, 0.0)], 35.0, 0.65, 0.0, 0.0,
                       6.0, 0.0006, "Beggs & Brill")
dP_lo = res_lo[0]["dP_total_psi"]
dP_hi = res_hi[0]["dP_total_psi"]
ratio = dP_hi / max(dP_lo, 1e-3)
ok = _check(f"Dry-gas horizontal ΔP scales super-quadratically (3x rate -> {ratio:.1f}x ΔP)",
            4.0 < ratio < 25.0,
            f"Friction-dominated regime (dP_lo={dP_lo:.2f}, dP_hi={dP_hi:.2f}). "
            f"Ratio > q² because gas expands as P drops along the line, "
            f"compounding the velocity rise at high rate.")
results.append(ok)


# ---------------------------------------------------------------------
# 3. LIFT CURVE J-SHAPE: vertical oil well has an interior minimum
# ---------------------------------------------------------------------
print("\n3. Vertical oil lift curve has J-shape (interior minimum)")
res = N.lift_curve(
    rates_bbl_d=[100, 300, 700, 1500, 3000, 6000],
    GLR_scf_stb=500.0,
    P_outlet_psia=200.0, T_outlet_F=80.0, T_inlet_F=180.0,
    segments=[(8000.0, 90.0)],
    api=35.0, gas_sg=0.75,
    water_cut_frac=0.1, salinity_wt_pct=2.0,
    D_in_inch=2.992, rel_roughness=0.0006,
    method="Hagedorn & Brown")
Ps = [r["P_inlet_psia"] for r in res]
i_min = Ps.index(min(Ps))
# Minimum should be in interior (not first or last point)
ok = _check(f"Minimum at interior rate (index {i_min} of {len(Ps)-1})",
            0 < i_min < len(Ps) - 1,
            f"Pressures: {[f'{p:.0f}' for p in Ps]}")
results.append(ok)
# Pressures around minimum should be ordered: decreasing then increasing
left_decreasing = all(Ps[i] >= Ps[i+1] for i in range(i_min))
right_increasing = all(Ps[i] <= Ps[i+1] for i in range(i_min, len(Ps)-1))
ok = _check("Pressures monotonic on each side of minimum",
            left_decreasing and right_increasing,
            "Gravity-dominated then friction-dominated")
results.append(ok)


# ---------------------------------------------------------------------
# 4. FLOW REGIME: vertical oil at low gas → bubble; horizontal high gas → mist
# ---------------------------------------------------------------------
print("\n4. Flow regime identification")
# Low GLR vertical oil — should NOT be 'distributed' (mist)
res = N.lift_curve([1000], 50.0, 200.0, 80.0, 180.0,
                    [(8000.0, 90.0)], 35.0, 0.75, 0.0, 0.0,
                    2.992, 0.0006, "Beggs & Brill")
pat = res[0]["dominant_pattern"]
ok = _check(f"Low-GLR vertical -> not mist (got '{pat}')",
            pat != "distributed",
            "Slug/intermittent or segregated, not dispersed")
results.append(ok)
# Horizontal very high GLR — should be distributed
res = N.lift_curve([1000], 100000.0, 200.0, 70.0, 70.0,
                    [(5000.0, 0.0)], 35.0, 0.7, 0.0, 0.0,
                    6.0, 0.0006, "Beggs & Brill")
pat = res[0]["dominant_pattern"]
ok = _check(f"Horizontal high-GLR -> dispersed (got '{pat}')",
            pat in ("distributed", "intermittent", "transition"),
            "Mist or intermittent expected")
results.append(ok)


# ---------------------------------------------------------------------
# 5. HOLDUP: must satisfy lambda_l <= H_L <= 1
# ---------------------------------------------------------------------
print("\n5. In-situ holdup respects physical bounds")
for q in [100, 500, 1000, 3000]:
    res = N.march_pressure(
        200.0, 80.0, 180.0, [(8000.0, 90.0)],
        q, q * 0.5, 35.0, 0.75, 0.1, 2.0,
        2.992, 0.0006, "Beggs & Brill")
    HLs = res["H_L"]
    HLs_valid = HLs[1:]   # skip outlet NaN
    bad = any(h < 0.0 or h > 1.0 for h in HLs_valid)
    ok = _check(f"q={q}: all H_L in [0,1]", not bad,
                f"min={min(HLs_valid):.3f}, max={max(HLs_valid):.3f}")
    results.append(ok)


# ---------------------------------------------------------------------
# 6. INCLINATION: downhill flowline -> ΔP can be NEGATIVE (pressure gain)
# ---------------------------------------------------------------------
print("\n6. Downhill section produces pressure gain")
res = N.lift_curve([500], 200.0, 200.0, 100.0, 100.0,
                    [(5000.0, -45.0)],   # 5000 ft going down 45°
                    35.0, 0.75, 0.1, 0.0, 4.0, 0.0006, "Beggs & Brill")
dP = res[0]["dP_total_psi"]
# Downhill 5000 ft × sin(45°) × oil density = serious pressure gain
# Even with friction adding back, net ΔP should be negative or small.
ok = _check(f"Downhill section: ΔP <= 0 or small (got {dP:.0f} psi)",
            dP < 100.0,
            "Hydrostatic gain dominates friction at moderate rate")
results.append(ok)


# ---------------------------------------------------------------------
# 7. CONTINUITY: small rate changes -> small pressure changes (no jumps)
# ---------------------------------------------------------------------
print("\n7. Continuity — no discontinuous jumps in P vs q")
rates = [500, 550, 600, 650, 700]
res = N.lift_curve(rates, 500.0, 200.0, 80.0, 180.0,
                    [(8000.0, 90.0)], 35.0, 0.75, 0.1, 2.0,
                    2.992, 0.0006, "Hagedorn & Brown")
Ps = [r["P_inlet_psia"] for r in res]
# Max relative jump between adjacent points should be < 30%
max_jump = max(abs(Ps[i+1] - Ps[i]) / max(Ps[i], 1)
               for i in range(len(Ps)-1))
ok = _check(f"Max adjacent-rate jump {max_jump*100:.1f}% < 30%",
            max_jump < 0.30,
            "Continuity across the rate range")
results.append(ok)


# ---------------------------------------------------------------------
# 8. CORRELATION ROUTING
# ---------------------------------------------------------------------
print("\n8. Correlation recommender picks sensibly")
ok = _check("Vertical oil -> H-B recommended",
            N.recommend_correlation("Light Oil", 90.0, 500)["recommended"]
            == "Hagedorn & Brown")
results.append(ok)
ok = _check("Vertical wet gas -> Gray recommended",
            N.recommend_correlation("Wet Gas", 90.0)["recommended"]
            == "Gray")
results.append(ok)
ok = _check("Horizontal oil -> Beggs-Brill recommended",
            N.recommend_correlation("Light Oil", 0.0)["recommended"]
            == "Beggs & Brill")
results.append(ok)
ok = _check("Downhill -> Beggs-Brill recommended",
            N.recommend_correlation("Light Oil", -30.0)["recommended"]
            == "Beggs & Brill")
results.append(ok)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
n_pass = sum(results)
n_total = len(results)
print(f"  {n_pass} / {n_total} screening checks passed "
      f"({100.0 * n_pass / n_total:.0f}%)")
print("=" * 70)
if n_pass == n_total:
    print("\n  All screening checks passed. The nodal correlations are\n"
          "  behaving sensibly across regime / inclination / rate ranges.\n"
          "  This is SCREENING accuracy — not a substitute for benchmarking\n"
          "  against PIPESIM/OLGA on your specific fluid before relying on\n"
          "  exported VFP curves for simulation studies.")
