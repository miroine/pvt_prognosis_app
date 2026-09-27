"""
PVT Studio — audit regression tests (Sep 2026 audit).

Each check pins a bug that was found and fixed, against an independent
reference (hand calculation from the published equation, or an exact
unit identity), so it cannot silently come back.

Run:  python test_audit.py
"""
import io
import zipfile
import numpy as np
import pandas as pd

import units as U
import correlations as C
import eclipse_export as E
import multi_sim_export as MSE
import nodal as N
import rock_comp as R
import vfp_export as V

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def close(a, b, rtol=2e-3):
    return abs(a - b) <= rtol * max(abs(b), 1e-12)


print("=" * 70 + "\nPVT STUDIO — AUDIT REGRESSION TESTS\n" + "=" * 70)

# 1. SI unit conversions (exact identities) --------------------------
print("\n1. SI conversions")
bbl, mscf_m3 = 0.158987, 28.3168
check("Bg 1 rb/Mscf -> rm3/Sm3", close(U.to_user_Bg(1.0, "SI"), bbl / mscf_m3))
check("Rv 1 STB/Mscf -> Sm3/Sm3", close(U.to_user_Rv(1.0, "SI"), bbl / mscf_m3))
check("CGR 1 STB/MMscf -> Sm3/MSm3 (MSm3 = 1e6 Sm3)",
      close(U.to_user_cgr(1.0, "SI"), bbl / (mscf_m3 / 1000.0)))
check("Rs 1 scf/STB -> Sm3/Sm3", close(U.to_user_Rs(1.0, "SI"), 0.0283168 / bbl))
for nm, fw, bw in (("Bg", U.to_user_Bg, U.to_field_Bg),
                   ("Rv", U.to_user_Rv, U.to_field_Rv),
                   ("CGR", U.to_user_cgr, U.to_field_cgr)):
    check(f"{nm} round-trip", close(bw(fw(1.2345, "SI"), "SI"), 1.2345))

# 2. Z-factor robustness over the Standing-Katz range ----------------
print("\n2. Z-factor solvers")
bad = 0
for Tpr in np.arange(1.05, 3.01, 0.05):
    for Ppr in np.arange(0.2, 15.01, 0.4):
        for f in (C._z_hall_yarborough, C._z_dak):
            z = f(Tpr, Ppr)
            bad += int(not (np.isfinite(z) and 0.25 < z < 2.2))
check("No NaN / non-physical Z on Tpr 1.05-3, Ppr 0.2-15", bad == 0, f"{bad} bad")
check("Z(1.5, 2.0) HY vs DAK agree", close(C._z_hall_yarborough(1.5, 2.0),
                                          C._z_dak(1.5, 2.0), rtol=5e-3))

# 3. Gas viscosity (Carr-Kobayashi-Burrows with Dempsey) --------------
print("\n3. Gas viscosity")
g_ckb = C.GasCorrelations(0.7, 200, mu_corr="Carr-Kobayashi-Burrows")
g_lge = C.GasCorrelations(0.7, 200)
for P in (1000, 5000, 10000):
    z = g_ckb.z_factor(P)
    r = g_ckb.viscosity(P, z) / g_lge.viscosity(P, z)
    check(f"CKB within 15% of LGE at {P} psia", 0.85 < r < 1.15, f"ratio {r:.3f}")

# 4. Oil correlations ---------------------------------------------------
print("\n4. Oil correlations")
pbs = {}
for c in ("Standing", "Vasquez-Beggs", "Glaso", "Lasater",
          "Al-Marhoun", "Petrosky-Farshad"):
    o = C.OilCorrelations(35, 0.75, 180, rs_corr=c)
    pbs[c] = o.bubble_point(600)
    check(f"{c}: Rs(Pb) round-trips to Rsi", close(o.solution_gor(pbs[c]), 600, 1e-3))
check("Lasater Pb = 2791 psia (hand calc, published form)",
      close(pbs["Lasater"], 2791, 5e-3), f"{pbs['Lasater']:.0f}")
o = C.OilCorrelations(35, 0.75, 180)
pb = o.bubble_point(600)
A = 1e-5 * (-1433 + 5 * 600 + 17.2 * 180 - 1180 * 0.75 + 12.61 * 35)
ref = o.formation_volume_factor(pb, 600) * np.exp(-A * np.log(4500 / pb))
check("Undersaturated Bo = integrated Vasquez-Beggs",
      close(o.formation_volume_factor(4500, 600, saturated=False, Pb=pb), ref, 1e-6))

# 5. Water (McCain, hand calc at 200 F / 5000 psia, fresh water) ---------
print("\n5. Water")
w = C.WaterCorrelations(0.0, 200.0)
check("McCain Bw", close(w.bw(5000), 1.02806, 5e-4))
check("McCain mu_w", close(w.viscosity(5000), 0.3676, 2e-3))

# 6. Rock ---------------------------------------------------------------
print("\n6. Rock compressibility")
check("Newman limestone published form (phi=0.1)",
      close(R.newman_limestone(0.1), 0.853531 / (1 + 2.47664e6 * 0.1) ** 0.9299))
check("Dolomite -> Newman limestone recommended",
      R.recommend_correlation("Dolomite", "Consolidated", 8000)["recommended"]
      == "Newman — Limestone")

# 7. PVTG Bg on surface-dry-gas basis ------------------------------------
print("\n7. ECLIPSE PVTG")
wg = C.WetGasCorrelations(0.7, 55.0, 80.0, 220.0, Pdew=4500.0)
line = E.build_pvtg([3000.0], wg).splitlines()[3].split()
Z = wg.z_factor(3000.0)
Rv = wg.rv(3000.0)
bg_ref = wg.formation_volume_factor(3000.0, Z) * 1000.0 * (1 + wg.Veq * Rv)
check("PVTG Bg includes (1 + Veq*Rv) dry-gas basis", close(float(line[2]), bg_ref, 2e-4),
      f"{line[2]} vs {bg_ref:.5f}")

# 8. Multi-simulator exporters (all kinds x units) ------------------------
print("\n8. Multi-simulator export")
oil = pd.DataFrame({"P (psia)": [500, 2500], "Rs (scf/STB)": [80, 550],
                    "Bo (rb/STB)": [1.10, 1.32], "μo (cp)": [1.5, 0.62]})
gas = pd.DataFrame({"P (psia)": [500, 2500], "Z": [0.95, 0.87],
                    "Bg (rb/scf)": [0.0058, 0.00108], "μg (cp)": [0.013, 0.019]})
base = dict(api=35, gas_sg=0.7, water_sg=1.02, Pb_psia=3500, Pref_psia=3000,
            T_res_F=200, Bw=1.02, Cw=3.5e-6, muw=0.4, fluid_name="T")
crash = 0
for kind, df in (("oil", oil), ("dry_gas", gas), ("gas-dry", gas)):
    for u in ("FIELD", "METRIC"):
        case = dict(base, kind=kind, units=u, df_field=df)
        for fn in (MSE.build_cmg_imex, MSE.build_cmg_gem, MSE.build_nexus,
                   MSE.build_intersect, MSE.build_csv, MSE.build_json):
            try:
                fn(case)
            except Exception:
                crash += 1
check("No exporter crashes (3 kind tags x 2 unit systems x 6 formats)", crash == 0)
t = MSE.build_nexus(dict(base, kind="oil", units="FIELD", df_field=oil))
check("Nexus oil table keeps viscosity column", "0.620000" in t)
t = MSE.build_cmg_imex(dict(base, kind="dry_gas", units="FIELD", df_field=gas))
check("CMG dry-gas table written (Bg converted to rb/Mscf)", "1.08" in t)
t = MSE.build_cmg_imex(dict(base, kind="dry_gas", units="METRIC", df_field=gas))
check("CMG SI: pressure in kPa", "17236.9" in t and "*INUNIT *SI" in t)
z = zipfile.ZipFile(io.BytesIO(MSE.build_bundle(
    dict(base, kind="oil", units="FIELD", df_field=oil), "PVTO\n/\n")))
check("Bundle contains all formats + README", len(z.namelist()) == 10)

# 9. Nodal phase split ----------------------------------------------------
print("\n9. Nodal / VFP")
r = N.lift_curve([100], 500.0, 200.0, 80.0, 180.0, [(8000.0, 90.0)],
                 35.0, 0.75, 0.0, 0.0, 2.992, 6e-4, "Hagedorn & Brown")[0]
check("Low-rate low-GLR oil column recognised as bubble flow",
      "bubble" in r["dominant_pattern"], r["dominant_pattern"])
kw = dict(thp_values_psia=[200], wct_values=[0.5], glr_values_scf_stb=[500],
          depth_ft=8000, T_outlet_F=80, T_inlet_F=180, segments=[(8000, 90)],
          api=35, gas_sg=0.75, salinity_wt_pct=2, D_in_inch=2.992,
          rel_roughness=6e-4, method="Hagedorn & Brown")
liq = V.build_vfp_table(flow_rates_bbl_d=[1000], flo_kind="LIQ", **kw)
oil_ = V.build_vfp_table(flow_rates_bbl_d=[500], flo_kind="OIL", **kw)
gas_ = V.build_vfp_table(flow_rates_bbl_d=[500], flo_kind="GAS", **kw)
b = liq["bhp_table_disp"][0, 0, 0, 0]
check("VFP FLO=OIL axis consistent with LIQ", close(oil_["bhp_table_disp"][0, 0, 0, 0], b, 1e-6))
check("VFP FLO=GAS axis consistent with LIQ", close(gas_["bhp_table_disp"][0, 0, 0, 0], b, 1e-6))

print("\n" + "=" * 70)
print(f"  {sum(results)} / {len(results)} audit checks passed")
print("=" * 70)
