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

# 7. PVTO / PVTG structure, span and QC ----------------------------------
print("\n7. ECLIPSE PVTO / PVTG")
import eclipse_qc as Q
import multi_region as MR
o = C.OilCorrelations(35, 0.75, 180)
pb = o.bubble_point(600)
nodes = E.pvto_nodes(o, 600, pb, 5000)
check("PVTO span starts at ~1 atm", nodes[0]["Psat"] <= 15.0)
check("PVTO extends above Pb to P_max", nodes[-1]["Psat"] >= 4999.0)
check("PVTO has Rsi exactly at Pb", any(abs(n["Rs"] - 600) < 1e-6
                                        and abs(n["Psat"] - pb) < 1e-2 for n in nodes))
check("Every PVTO node carries an undersaturated branch",
      all(len(n["P_u"]) >= 2 for n in nodes))
check("Low-Rs branch viscosity bounded (< 3x over the branch)",
      nodes[0]["mu_u"][-1] / nodes[0]["mu_sat"] < 3.0,
      f"{nodes[0]['mu_u'][-1] / nodes[0]['mu_sat']:.2f}x")
pvto = E.build_pvto(None, pb, o, 600, 5000)
for u in ("FIELD", "METRIC"):
    t = pvto if u == "FIELD" else E.convert_deck_to_metric(pvto=pvto)["pvto"]
    r = Q.qc_pvto_branches(Q.parse_pvto_branches(t))
    check(f"PVTO {u} passes structural QC", r["ok"], "; ".join(r["problems"][:2]))
for cgr, api in ((20, 60), (80, 55), (200, 50)):
    wg = C.WetGasCorrelations(0.7, api, cgr, 220.0, Pdew=4500.0)
    pvtg = E.build_pvtg(list(np.linspace(500, 6000, 12)), wg)
    nd = Q.parse_pvtg_branches(pvtg)
    check(f"PVTG CGR {cgr}: dew point is a node", any(abs(n["P"] - 4500) < 0.01 for n in nd))
    for u in ("FIELD", "METRIC"):
        t = pvtg if u == "FIELD" else E.convert_deck_to_metric(pvtg=pvtg)["pvtg"]
        r = Q.qc_pvtg_branches(Q.parse_pvtg_branches(t))
        check(f"PVTG CGR {cgr} {u} passes structural QC", r["ok"],
              "; ".join(r["problems"][:2]))
    hi = nd[-1]
    check(f"PVTG CGR {cgr}: leaner gas less viscous at high P",
          hi["mu"][-1] < hi["mu"][0])
wg = C.WetGasCorrelations(0.7, 55.0, 80.0, 220.0, Pdew=4500.0)
bg, _ = E._gas_props_at_rv(wg, 3000.0, wg.rv(3000.0))
Z = C.GasCorrelations((0.7 + 4584 * wg.gamma_cond * wg.rv(3000.0))
                      / (1 + wg.Veq * wg.rv(3000.0)), 220.0).z_factor(3000.0)
ref = 0.00504 * Z * (220 + 460) / 3000.0 * 1000 * (1 + wg.Veq * wg.rv(3000.0))
check("PVTG Bg on surface-dry-gas basis (1 + Veq*Rv)", close(bg, ref, 1e-6))
bad = "PVTG\n  1000 0.02 3.0 0.015\n       0.03 2.9 0.016 /\n/\n"
check("PVTG QC catches Rv increasing along a branch",
      not Q.qc_pvtg_branches(Q.parse_pvtg_branches(bad))["ok"])
st = MR.build_multi_region_pvto([pvto, pvto, pvto])
check("Multi-region PVTO: one terminator per region",
      sum(1 for l in st.splitlines() if l.strip() == "/") == 3)

# 7b. LBC viscosity units (Stiel-Thodos needs K and atm) --------------------
import lbc
check("LBC methane 100 F, 1 atm = 0.0114 cP (+-5 %)",
      close(lbc.lbc_viscosity(["C1"], [1.0], 0.0392, 559.67), 0.0114, 0.05))

# 7c. Wet-gas CVD material balance ------------------------------------------
import correlation_experiments as CE
cv = sorted(CE.cvd_wetgas(wg, 4500.0, [4500, 3500, 2500, 1500, 500]),
            key=lambda r: -r["P"])
cp = [r["cum_produced_pct"] for r in cv]
check("CVD cumulative production 0 at Pdew, rising with depletion",
      cp[0] == 0 and all(a < b for a, b in zip(cp, cp[1:])))

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

# 8b. Multi-sim for every fluid type + ASCII decks ---------------------------
print("\n8b. Multi-simulator export: wet gas, water, compositional")
wgc = C.WetGasCorrelations(0.7, 55.0, 80.0, 220.0, Pdew=4500.0)
pvtg_f = E.build_pvtg(list(np.linspace(500, 6000, 12)), wgc)
wdf = MSE.df_from_deck("wetgas", pvtg_f)
check("Wet-gas df from PVTG: one row per pressure node, Rv at dew-point > 0",
      len(wdf) == len(Q.parse_pvtg_branches(pvtg_f)) and wdf["Rv (STB/Mscf)"].max() > 0)
odf = MSE.df_from_deck("oil", pvto)
check("Oil df from PVTO: Rs back in scf/STB (Rsi = 600)",
      (abs(odf["Rs (scf/STB)"] - 600) < 0.1).any())
wat = pd.DataFrame({"P (psia)": [1000, 3000], "Bw (rb/STB)": [1.03, 1.02]})
crash, nonascii = 0, 0
for kind, df in (("wetgas", wdf), ("water", wat), ("oil", odf), ("drygas", gas)):
    for u in ("FIELD", "METRIC"):
        case = dict(base, kind=kind, units=u, df_field=df, Cvw=1e-5)
        for fn in (MSE.build_cmg_imex, MSE.build_cmg_gem, MSE.build_nexus,
                   MSE.build_intersect, MSE.build_csv):
            try:
                t = fn(case)
                nonascii += any(ord(ch) > 127 for ch in t)
            except Exception:
                crash += 1
check("No crash: wetgas/water/oil/drygas x 2 units x 5 formats", crash == 0)
check("Every simulator text export is plain ASCII", nonascii == 0)
t = MSE.build_intersect(dict(base, kind="wetgas", units="METRIC", df_field=wdf))
check("IX wet gas carries the Rv array", "Rv = [" in t)
t = MSE.build_cmg_imex(dict(base, kind="water", units="METRIC", df_field=wat))
check("CMG water case: *BWI/*CW/*PBW written, no oil/gas table",
      "*BWI" in t and "*PBW" in t and "*PVT" not in t)
t = MSE.build_nexus(dict(base, kind="oil", units="METRIC", df_field=odf))
check("Header reservoir T in degC for METRIC", "93.33 degC" in t)
z = zipfile.ZipFile(io.BytesIO(MSE.build_bundle(
    dict(base, kind="oil", units="FIELD", df_field=oil,
         extra_files={"E300_PROPS.INC": "CNAMES\n/\n"}), "PVTO\n/\n")))
check("Bundle includes extra (E300) files", any(n.endswith("E300_PROPS.INC")
                                                for n in z.namelist()))

# 8c. ECLIPSE 300 / GEM compositional export ------------------------------
print("\n8c. Compositional (E300 / GEM) export")
import e300_export as E3
from components import characterize_c7plus
names = ["N2", "CO2", "C1", "C2", "C3", "iC4", "nC4", "iC5", "nC5", "C6", "C7+"]
zz = [0.005, 0.02, 0.4, 0.08, 0.06, 0.01, 0.03, 0.01, 0.015, 0.03, 0.34]
c7 = characterize_c7plus(220, 0.85)
for u in ("FIELD", "METRIC"):
    t = E3.build_e300_props(names, zz, 200.0, c7, units=u, psat_psia=3000,
                            sat_kind="Bubble point")
    r = E3.qc_e300_props(t, len(names))
    check(f"E300 {u}: structural QC", r["ok"], "; ".join(r["problems"]))
    check(f"E300 {u}: ASCII only", all(ord(ch) < 128 for ch in t))
kw = E3.parse_e300_props(E3.build_e300_props(names, zz, 200.0, c7, units="METRIC"))
i1 = names.index("C1")
check("E300 METRIC: C1 Tc = 190.56 K", close(float(kw["TCRIT"][i1]), 190.56, 1e-3))
check("E300 METRIC: C1 Pc = 46.0 bara", close(float(kw["PCRIT"][i1]), 46.04, 2e-3))
check("E300 METRIC: C1 Vc = 0.0993 m3/kmol", close(float(kw["VCRIT"][i1]), 0.09926, 2e-3))
check("E300 METRIC: RTEMP in degC", close(float(kw["RTEMP"][0]), 93.333, 1e-4))
kwf = E3.parse_e300_props(E3.build_e300_props(
    names, zz, 200.0, c7, units="FIELD",
    kij_overrides={("C1", "C7+"): 0.0777}))
check("E300: tuned C1-C7+ kij written into BIC", "0.07770" in kwf["BIC"])
g = E3.build_gem_eos(names, zz, 200.0, c7, units="METRIC")
check("GEM EOS: Pc in atm (C1 = 45.44)", "45.4397" in g and "*BIN" in g)

# 8d. SI messages carry SI units --------------------------------------------
print("\n8d. SI units in messages")
_txt = " ".join(str(U.fmt(k, 1000.0, "SI")) for k in ("P", "dP", "T", "dT", "Rs"))
check("U.fmt SI never prints field units",
      not any(x in _txt for x in ("psi", "°F", "scf")), _txt)

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

# 10. Deployment stamp ------------------------------------------------------
print("\n10. Deployment")
import os, re
_app = open("pvt_app.py").read()
_ver = re.search(r'^APP_VERSION = "([^"]+)"', _app, re.M).group(1)
_req = eval(re.search(r"^_REQUIRED = (\[.*?\])", _app, re.M | re.S).group(1))
_bad = [m for m in _req if not re.search(r'^APP_VERSION = "%s"' % re.escape(_ver),
                                          open(m + ".py").read(), re.M)]
check(f"All {len(_req)} runtime modules stamped {_ver}", not _bad, ", ".join(_bad))
_mods = {f[:-3] for f in os.listdir(".") if f.endswith(".py")} - {"pvt_app", "validate_nodal"}
_mods = {m for m in _mods if not m.startswith("test_")}
check("Every runtime module is in the deployment check list",
      _mods == set(_req), str(sorted(_mods ^ set(_req))))

print("\n" + "=" * 70)
print(f"  {sum(results)} / {len(results)} audit checks passed")
print("=" * 70)
