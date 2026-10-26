"""
PVT Studio — import an ECLIPSE black-oil PVT deck and use it as the PVT
model of the nodal analysis (VLP and IPR).

Supported keywords: PVTO, PVDO, PVTG, PVDG, PVTW, DENSITY, GRAVITY
(any number of PVTNUM regions; one region is selected), FIELD or METRIC
units (the keyword is detected when present in the text).

Internally everything is held in FIELD units:
    P psia, Rs scf/STB, Bo rb/STB, Rv STB/Mscf, Bg rb/Mscf, mu cP,
    densities lb/ft3, Cw / Cvw 1/psi.

The model answers the questions the multiphase-flow march asks at every
step: given P (and T) and the surface oil / gas rates, how much free oil
and free gas is there in the pipe, and what are their FVFs, viscosities
and densities? `flash()` does the black-oil split with dissolved gas
(Rs) and vaporised oil (Rv):

    q_g = G_f + O_f Rs ,   q_o = O_f + G_f Rv

with undersaturated oil (all gas dissolved) or undersaturated gas (all
oil vaporised) handled explicitly. ECLIPSE tables are isothermal (at the
reservoir temperature); the gas FVF is optionally corrected along the
tubing with the ideal-gas temperature ratio, everything else is taken at
table conditions.
"""

APP_VERSION = "1.4.7"   # must match pvt_app.py (deployment check)

import re
import numpy as np

# METRIC -> FIELD factors
_BAR = 14.50377                      # bar -> psi
_SM3_SM3_TO_SCF_STB = 5.614583       # Rs  : sm3/sm3 -> scf/STB
_RM3_SM3_TO_RB_MSCF = 178.1076       # Bg  : rm3/sm3 -> rb/Mscf (6.28981 x 28.31685)
_SM3_SM3_TO_STB_MSCF = 178.1076      # Rv  : sm3/sm3 -> STB/Mscf
_KGM3_TO_LBFT3 = 1.0 / 16.018463


KEYWORDS = ("PVTO", "PVDO", "PVTG", "PVDG", "PVTW", "DENSITY", "GRAVITY")


# ----------------------------------------------------------------------
# Tokenising
# ----------------------------------------------------------------------
def _strip_comments(text):
    out = []
    for line in text.replace("\t", " ").splitlines():
        i = line.find("--")
        out.append(line if i < 0 else line[:i])
    return out


def _expand(tok):
    """'3*' -> [None]*3 ; '2*0.5' -> [0.5, 0.5] ; '1.2' -> [1.2]."""
    m = re.fullmatch(r"(\d+)\*(.*)", tok)
    if m:
        n = int(m.group(1))
        v = m.group(2)
        return [float(v) if v else None] * n
    return [float(tok)]


def split_keywords(text):
    """{KEYWORD: [list of records per occurrence]} where a record is a list
    of numbers (None = defaulted). An empty record marks a table end."""
    lines = _strip_comments(text)
    blocks, cur = {}, None
    units = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        first = s.split()[0]
        if re.fullmatch(r"[A-Z][A-Z0-9_]{1,7}", first) and not re.match(r"^\d", first):
            if first in ("FIELD", "METRIC"):
                units = first
            cur = first if first in KEYWORDS else None
            if cur is not None:
                blocks.setdefault(cur, []).append("")
            continue
        if cur is not None:
            blocks[cur][-1] += " " + s
    parsed = {}
    for kw, occurrences in blocks.items():
        recs_all = []
        for body in occurrences:
            body = body.replace("/", " / ")
            recs, rec = [], []
            for tok in body.split():
                if tok == "/":
                    recs.append(rec)
                    rec = []
                    continue
                try:
                    rec.extend(_expand(tok))
                except ValueError:
                    continue
            if rec:
                recs.append(rec)
            recs_all.extend(recs)
        parsed[kw] = recs_all
    return parsed, units


def _tables_from_records(records, kw):
    """Group records into tables (regions)."""
    if kw in ("PVTO", "PVTG"):
        tables, cur = [], []
        for r in records:
            if not r:
                if cur:
                    tables.append(cur)
                cur = []
            else:
                cur.append(r)
        if cur:
            tables.append(cur)
        return tables
    # one record per region
    return [r for r in records if r]


def detect_units(parsed, units_kw=None):
    """FIELD / METRIC: keyword if present, else a heuristic on PVTO / PVTG
    / PVDG pressure and Rs magnitudes. Returns (units, reason)."""
    if units_kw:
        return units_kw, f"{units_kw} keyword found in the text"
    pmax, rsmax = 0.0, None
    for kw in ("PVTO", "PVTG", "PVDG", "PVDO"):
        for r in parsed.get(kw, []):
            if not r:
                continue
            if kw == "PVTO":
                rsmax = max(rsmax or 0.0, r[0] or 0.0)
                pmax = max(pmax, max(v for v in r[1::3] if v is not None))
            elif kw == "PVTG":
                pmax = max(pmax, r[0] or 0.0)
            else:
                pmax = max(pmax, max(v for v in r[0::3] if v is not None))
    if rsmax is not None and rsmax < 5.0:
        return "FIELD", "PVTO Rs < 5 (Mscf/STB)"
    if pmax > 1500.0:
        return "FIELD", f"max pressure {pmax:.0f} > 1500 (psia)"
    if pmax > 0:
        return "METRIC", f"max pressure {pmax:.0f} <= 1500 (bara)"
    return "FIELD", "no table pressures found (default)"


# ----------------------------------------------------------------------
# The PVT model
# ----------------------------------------------------------------------
class EclipsePVT:
    """Black-oil PVT from ECLIPSE tables (one PVTNUM region)."""

    def __init__(self, parsed, units="FIELD", region=1, T_table_F=200.0,
                 temperature_correct_gas=True):
        self.units = units
        self.region = int(region)
        self.T_table_F = float(T_table_F)
        self.tcorr = bool(temperature_correct_gas)
        m = units == "METRIC"
        fP = _BAR if m else 1.0
        self.notes = []
        i = self.region - 1

        def pick(kw):
            t = _tables_from_records(parsed.get(kw, []), kw)
            self.n_regions = max(getattr(self, "n_regions", 1), len(t))
            if not t:
                return None
            if i >= len(t):
                self.notes.append(f"{kw}: region {self.region} missing, "
                                  f"using region {len(t)}.")
                return t[-1]
            return t[i]

        # ---- oil ----
        self.oil = None
        t = pick("PVTO")
        if t:
            nodes = []
            fRs = _SM3_SM3_TO_SCF_STB if m else 1000.0   # FIELD Rs is Mscf/STB
            for r in t:
                vals = [v for v in r if v is not None]
                Rs = vals[0] * fRs
                rest = vals[1:]
                n = len(rest) // 3
                P = np.array(rest[0:3 * n:3]) * fP
                Bo = np.array(rest[1:3 * n:3])
                mu = np.array(rest[2:3 * n:3])
                if n:
                    nodes.append({"Rs": Rs, "P": P, "Bo": Bo, "mu": mu})
            nodes.sort(key=lambda d: d["Rs"])
            self.oil = {"kind": "live", "nodes": nodes}
        else:
            t = pick("PVDO")
            if t:
                vals = [v for v in t if v is not None]
                n = len(vals) // 3
                self.oil = {"kind": "dead",
                            "P": np.array(vals[0:3 * n:3]) * fP,
                            "Bo": np.array(vals[1:3 * n:3]),
                            "mu": np.array(vals[2:3 * n:3])}

        # ---- gas ----
        self.gas = None
        fBg = _RM3_SM3_TO_RB_MSCF if m else 1.0
        fRv = _SM3_SM3_TO_STB_MSCF if m else 1.0
        t = pick("PVTG")
        if t:
            nodes = []
            for r in t:
                vals = [v for v in r if v is not None]
                P = vals[0] * fP
                rest = vals[1:]
                n = len(rest) // 3
                nodes.append({"P": P,
                              "Rv": np.array(rest[0:3 * n:3]) * fRv,
                              "Bg": np.array(rest[1:3 * n:3]) * fBg,
                              "mu": np.array(rest[2:3 * n:3])})
            nodes.sort(key=lambda d: d["P"])
            self.gas = {"kind": "wet", "nodes": nodes}
        else:
            t = pick("PVDG")
            if t:
                vals = [v for v in t if v is not None]
                n = len(vals) // 3
                self.gas = {"kind": "dry",
                            "P": np.array(vals[0:3 * n:3]) * fP,
                            "Bg": np.array(vals[1:3 * n:3]) * fBg,
                            "mu": np.array(vals[2:3 * n:3])}

        # ---- water ----
        self.water = None
        t = pick("PVTW")
        if t:
            v = list(t) + [None] * 5
            fC = (1.0 / _BAR) if m else 1.0
            self.water = {"Pref": (v[0] * fP if v[0] is not None else 14.696),
                          "Bw": v[1] if v[1] is not None else 1.0,
                          "Cw": (v[2] or 0.0) * fC,
                          "mu": v[3] if v[3] is not None else 0.5,
                          "Cv": (v[4] or 0.0) * fC}

        # ---- surface densities ----
        self.rho_sc = None
        t = pick("DENSITY")
        if t:
            v = (list(t) + [None] * 3)[:3]
            f = _KGM3_TO_LBFT3 if m else 1.0
            dflt = (53.66, 64.0, 0.0458)          # ECLIPSE FIELD defaults
            self.rho_sc = tuple((x * f if x is not None else d)
                                for x, d in zip(v, dflt))
        else:
            t = pick("GRAVITY")
            if t:
                v = (list(t) + [None] * 3)[:3]
                api = v[0] if v[0] is not None else 45.5
                self.rho_sc = (141.5 / (131.5 + api) * 62.428,
                               (v[1] if v[1] is not None else 1.0) * 62.428,
                               (v[2] if v[2] is not None else 0.7773) * 0.0764)
        if self.oil is None and self.gas is None:
            raise ValueError("No PVTO / PVDO / PVTG / PVDG table found.")

    # ---------------------------------------------------------------- info
    @property
    def api(self):
        return (141.5 / (self.rho_sc[0] / 62.428) - 131.5) if self.rho_sc else None

    @property
    def gas_sg(self):
        return self.rho_sc[2] / 0.0764 if self.rho_sc else None

    @property
    def water_sg(self):
        return self.rho_sc[1] / 62.428 if self.rho_sc else None

    def summary(self):
        s = {"units": self.units, "region": self.region,
             "regions in file": getattr(self, "n_regions", 1)}
        if self.oil and self.oil["kind"] == "live":
            n = self.oil["nodes"]
            s["PVTO Rs nodes"] = len(n)
            s["PVTO Psat range (psia)"] = (float(n[0]["P"][0]), float(n[-1]["P"][0]))
            s["PVTO Rs range (scf/STB)"] = (n[0]["Rs"], n[-1]["Rs"])
            s["undersaturated branches"] = sum(len(x["P"]) > 1 for x in n)
        elif self.oil:
            s["PVDO points"] = len(self.oil["P"])
        if self.gas and self.gas["kind"] == "wet":
            n = self.gas["nodes"]
            s["PVTG P nodes"] = len(n)
            s["PVTG Rv sat range (STB/Mscf)"] = (float(n[0]["Rv"][0]), float(n[-1]["Rv"][0]))
        elif self.gas:
            s["PVDG points"] = len(self.gas["P"])
        s["PVTW"] = self.water is not None
        s["DENSITY/GRAVITY"] = self.rho_sc is not None
        return s

    # ----------------------------------------------------------------- oil
    def _sat_curve(self):
        n = self.oil["nodes"]
        return (np.array([d["P"][0] for d in n]), np.array([d["Rs"] for d in n]),
                np.array([d["Bo"][0] for d in n]), np.array([d["mu"][0] for d in n]))

    def rs_sat(self, P):
        if not self.oil:
            return 0.0
        if self.oil["kind"] == "dead":
            return 0.0
        Ps, Rs, _, _ = self._sat_curve()
        return float(np.interp(P, Ps, Rs))

    def pb_of_rs(self, Rs):
        """Bubble-point pressure of oil with solution GOR Rs (scf/STB)."""
        if not self.oil or self.oil["kind"] == "dead":
            return None
        Ps, Rss, _, _ = self._sat_curve()
        if Rs > Rss[-1]:
            # extrapolate with the last segment
            if len(Ps) > 1 and Rss[-1] > Rss[-2]:
                return float(Ps[-1] + (Rs - Rss[-1]) * (Ps[-1] - Ps[-2])
                             / (Rss[-1] - Rss[-2]))
            return float(Ps[-1])
        return float(np.interp(Rs, Rss, Ps))

    def _branch_ratio(self, idx, dp, key):
        """Undersaturated ratio X(Psat + dp)/X(Psat) on node idx (or the
        nearest node that has a branch)."""
        n = self.oil["nodes"]
        order = sorted(range(len(n)), key=lambda j: (abs(j - idx), -j))
        for j in order:
            d = n[j]
            if len(d["P"]) > 1:
                x = d["P"] - d["P"][0]
                y = d[key] / d[key][0]
                if dp <= x[-1]:
                    return float(np.interp(dp, x, y))
                slope = (y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-9)
                return float(y[-1] + slope * (dp - x[-1]))
        return 1.0

    def oil_props(self, P, Rs=None):
        """(Bo rb/STB, mu_o cP, Rs used) at P for oil with solution GOR Rs
        (None = saturated)."""
        o = self.oil
        if o["kind"] == "dead":
            Bo = float(np.interp(P, o["P"], o["Bo"]))
            mu = float(np.interp(P, o["P"], o["mu"]))
            return Bo, mu, 0.0
        Ps, Rss, Bos, mus = self._sat_curve()
        rsat = float(np.interp(P, Ps, Rss))
        if Rs is None or Rs >= rsat - 1e-9:
            return (float(np.interp(P, Ps, Bos)), float(np.interp(P, Ps, mus)),
                    rsat)
        Rs = max(Rs, 0.0)
        Pb = self.pb_of_rs(Rs)
        Bob = float(np.interp(Pb, Ps, Bos))
        mub = float(np.interp(Pb, Ps, mus))
        dp = max(P - Pb, 0.0)
        k = int(np.clip(np.searchsorted(Rss, Rs), 1, len(Rss) - 1))
        w = (Rs - Rss[k - 1]) / max(Rss[k] - Rss[k - 1], 1e-12)
        rB = (1 - w) * self._branch_ratio(k - 1, dp, "Bo") + w * self._branch_ratio(k, dp, "Bo")
        rM = (1 - w) * self._branch_ratio(k - 1, dp, "mu") + w * self._branch_ratio(k, dp, "mu")
        return Bob * rB, mub * rM, Rs

    # ----------------------------------------------------------------- gas
    def rv_sat(self, P):
        if not self.gas or self.gas["kind"] == "dry":
            return 0.0
        n = self.gas["nodes"]
        return float(np.interp(P, [d["P"] for d in n], [d["Rv"][0] for d in n]))

    @staticmethod
    def _bg_interp(P, Ps, Bgs):
        """1/Bg linear in P inside the table; Bg*P constant outside."""
        if P <= Ps[0]:
            return float(Bgs[0] * Ps[0] / max(P, 1.0))
        if P >= Ps[-1]:
            return float(Bgs[-1] * Ps[-1] / P)
        return float(1.0 / np.interp(P, Ps, 1.0 / np.asarray(Bgs)))

    def gas_props(self, P, Rv=None):
        """(Bg rb/Mscf, mu_g cP, Rv used) at table temperature."""
        g = self.gas
        if g["kind"] == "dry":
            Bg = self._bg_interp(P, g["P"], g["Bg"])
            mu = float(np.interp(P, g["P"], g["mu"]))
            return Bg, mu, 0.0
        n = g["nodes"]
        Ps = np.array([d["P"] for d in n])

        def at_node(d):
            rv = d["Rv"][0] if Rv is None else min(Rv, d["Rv"][0])
            if len(d["Rv"]) > 1:
                xs = d["Rv"][::-1]
                return (float(np.interp(rv, xs, d["Bg"][::-1])),
                        float(np.interp(rv, xs, d["mu"][::-1])), rv)
            return float(d["Bg"][0]), float(d["mu"][0]), d["Rv"][0]

        if P <= Ps[0] or P >= Ps[-1]:
            j = 0 if P <= Ps[0] else -1
            Bg, mu, rv = at_node(n[j])
            return Bg * Ps[j] / max(P, 1.0), mu, rv
        k = int(np.searchsorted(Ps, P))
        w = (P - Ps[k - 1]) / (Ps[k] - Ps[k - 1])
        b0, m0, r0 = at_node(n[k - 1])
        b1, m1, r1 = at_node(n[k])
        Bg = 1.0 / ((1 - w) / b0 + w / b1)
        return Bg, (1 - w) * m0 + w * m1, (1 - w) * r0 + w * r1

    def mu_z(self, P, T_F=None):
        """mu_g * z from the gas table (for the Darcy gas IPR):
        Bg [rb/Mscf] = 5.0435 z T_R / P."""
        Bg, mu, _ = self.gas_props(P)
        T_R = (T_F if T_F is not None else self.T_table_F) + 459.67
        z = Bg * P / (5.0435 * T_R)
        return mu * z

    # --------------------------------------------------------------- water
    def water_props(self, P):
        w = self.water
        X = w["Cw"] * (P - w["Pref"])
        Y = -w["Cv"] * (P - w["Pref"])
        Bw = w["Bw"] / (1 + X + X * X / 2)
        mu = w["mu"] / (1 + Y + Y * Y / 2)
        return Bw, mu

    # --------------------------------------------------------------- flash
    def flash(self, P, T_F, q_o, q_g_mscf):
        """Black-oil split of surface rates at (P, T).

        Returns dict: O_f (STB/d of free oil), G_f (Mscf/d of free gas),
        Rs (scf/STB), Rv (STB/Mscf), Bo, mu_o, Bg (rb/Mscf, temperature
        corrected), mu_g, rho_o, rho_g (lb/ft3) — oil / gas entries are
        None when that table is missing (the caller falls back to its own
        correlations)."""
        rs = self.rs_sat(P) if self.oil else 0.0
        rv = self.rv_sat(P) if self.gas else 0.0
        q_o, q_g = max(q_o, 0.0), max(q_g_mscf, 0.0)
        den = 1.0 - rs * rv / 1000.0
        Of = (q_o - q_g * rv) / den if den > 1e-9 else q_o
        Gf = (q_g - q_o * rs / 1000.0) / den if den > 1e-9 else q_g
        if Gf < 0:                          # undersaturated oil
            Of, Gf = q_o, 0.0
            rs = q_g * 1000.0 / q_o if q_o > 0 else 0.0
            rv = 0.0
        elif Of < 0:                        # undersaturated gas
            Of, Gf = 0.0, q_g
            rv = q_o / q_g if q_g > 0 else 0.0
            rs = 0.0
        out = {"O_f": Of, "G_f": Gf, "Rs": rs, "Rv": rv,
               "Bo": None, "mu_o": None, "rho_o": None,
               "Bg": None, "mu_g": None, "rho_g": None}
        rho_o_sc, rho_w_sc, rho_g_sc = self.rho_sc or (None, None, None)
        if self.oil:
            Bo, muo, rs_used = self.oil_props(P, rs)
            out.update(Bo=Bo, mu_o=muo)
            if rho_o_sc:
                out["rho_o"] = (rho_o_sc * 5.614583 + rho_g_sc * rs_used) \
                    / (5.614583 * Bo)
        if self.gas:
            Bg, mug, rv_used = self.gas_props(P, rv)
            if self.tcorr:
                Bg *= (T_F + 459.67) / (self.T_table_F + 459.67)
            out.update(Bg=Bg, mu_g=mug)
            if rho_g_sc:
                out["rho_g"] = (rho_g_sc * 1000.0 + rv_used * 5.614583
                                * rho_o_sc) / (Bg * 5.614583)
        return out


# ----------------------------------------------------------------------
# Convenience
# ----------------------------------------------------------------------
def load_pvt(text, units=None, region=1, T_table_F=200.0,
             temperature_correct_gas=True):
    """Parse a deck and build the model. units=None -> auto-detect.
    Returns (EclipsePVT, info dict)."""
    parsed, units_kw = split_keywords(text)
    found = [k for k in KEYWORDS if parsed.get(k)]
    det, why = detect_units(parsed, units_kw)
    u = units or det
    pvt = EclipsePVT(parsed, u, region, T_table_F, temperature_correct_gas)
    return pvt, {"keywords": found, "units_detected": det,
                 "units_reason": why, "units_used": u,
                 "n_regions": getattr(pvt, "n_regions", 1)}


def example_deck(units="FIELD"):
    """An example deck generated with the app's own correlations
    (35 API oil, gas SG 0.75, 180 degF, Rsi 600 scf/STB): PVTO with
    undersaturated branches, PVDG, PVTW and DENSITY."""
    import pandas as pd
    from correlations import OilCorrelations, GasCorrelations
    import eclipse_export as E
    oil = OilCorrelations(35.0, 0.75, 180.0)
    pb = oil.bubble_point(600.0)
    pvto = E.build_pvto(None, pb, oil, 600.0, 5000.0)
    gas = GasCorrelations(0.75, 180.0)
    rows = []
    for P in np.geomspace(14.7, 6000.0, 18):
        Z = gas.z_factor(P)
        rows.append({"P (psia)": P, "Bg (rb/scf)": gas.formation_volume_factor(P, Z),
                     "μg (cp)": float(gas.viscosity(P, Z))})
    pvdg = E.build_pvdg(pd.DataFrame(rows))
    pvtw = E.build_pvtw(3000.0, 1.02, 3.0e-6, 0.40, 0.0)
    dens = E.build_density(35.0, 0.75, 1.02)
    if units == "METRIC":
        c = E.convert_deck_to_metric(pvto=pvto, pvdg=pvdg, pvtw=pvtw,
                                     density=dens)
        pvto, pvdg, pvtw, dens = c["pvto"], c["pvdg"], c["pvtw"], c["density"]
    head = ("-- Example black-oil deck generated by PVT Studio\n"
            "-- 35 API oil, gas SG 0.75, 180 degF, Rsi 600 scf/STB\n"
            f"{units}\n\n")
    return head + "\n".join([dens, pvto, pvdg, pvtw])
