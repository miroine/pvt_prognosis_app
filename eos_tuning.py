"""
EOS tuning to lab measurements via Levenberg-Marquardt least-squares.

The user provides a baseline composition + lab measurements (saturation pressure,
CCE V/Vsat or liquid dropout, DLE Rs/Bo, GOR, ST oil density).
We perturb a subset of EOS parameters (per-component Pc, Tc, omega multipliers,
plus selected kij values) and minimize the weighted RMS error between
EOS predictions and measurements.

Pattern: Whitson (1989), Coats (1985). Only C7+ properties are perturbed by
default — the standard practice when tuning a black-oil composition.
"""

import numpy as np
from scipy.optimize import least_squares

from eos_pr import saturation_pressure
from experiments import run_cce, run_dle
from separator import run_separator_train
import components as _comp_mod


def apply_c7_multipliers(c7_props_base, Pc_mult, Tc_mult, omega_mult):
    """Return a copy of C7+ props with Pc/Tc/omega multiplied."""
    if c7_props_base is None:
        return None
    new = dict(c7_props_base)
    new["Pc"]    = c7_props_base["Pc"]    * Pc_mult
    new["Tc"]    = c7_props_base["Tc"]    * Tc_mult
    new["omega"] = c7_props_base["omega"] * omega_mult
    return new


class KijOverride:
    """Context manager to temporarily set kij values in the global KIJ_TABLE."""
    def __init__(self, overrides):
        self.overrides = overrides or {}
        self._saved = {}
    def __enter__(self):
        for k, v in self.overrides.items():
            key = tuple(k) if (k in _comp_mod.KIJ_TABLE) else \
                  (tuple(reversed(k)) if tuple(reversed(k)) in _comp_mod.KIJ_TABLE else tuple(k))
            self._saved[key] = _comp_mod.KIJ_TABLE.get(key, None)
            _comp_mod.KIJ_TABLE[key] = float(v)
        return self
    def __exit__(self, *a):
        for key, old in self._saved.items():
            if old is None:
                _comp_mod.KIJ_TABLE.pop(key, None)
            else:
                _comp_mod.KIJ_TABLE[key] = old


def predict_all(z, comp_names, T_R, params, c7_base, measurements):
    """Predict measurement values given a parameter vector."""
    Pc_mult, Tc_mult, om_mult, kij_C1_C7, kij_N2_C7 = params
    c7_new = apply_c7_multipliers(c7_base, Pc_mult, Tc_mult, om_mult)

    overrides = {}
    if "C7+" in comp_names and "C1" in comp_names:
        overrides[("C1", "C7+")] = float(kij_C1_C7)
    if "C7+" in comp_names and "N2" in comp_names:
        overrides[("N2", "C7+")] = float(kij_N2_C7)

    predictions = []
    with KijOverride(overrides):
        psat_cache = None
        cce_cache = None
        dle_cache = None
        sep_cache = None

        # First pass: compute Psat (many measurements depend on it)
        try:
            psat_cache = saturation_pressure(
                z, comp_names, T_R, c7_props=c7_new,
                kind=measurements[0].get("kind", "auto")
                if measurements and "kind" in measurements[0] else "auto")
        except Exception:
            psat_cache = None

        # Pre-collect lists of P's for CCE and DLE so we run them once
        cce_Ps = [m["P"] for m in measurements if m["type"] in ("V_rel", "L_dropout")]
        dle_Ps = [m["P"] for m in measurements if m["type"] in ("Rs", "Bo")]
        if cce_Ps and psat_cache:
            try:
                cce_rows = run_cce(z, comp_names, T_R, cce_Ps, c7_new, P_sat=psat_cache)
                cce_cache = {row["P"]: row for row in cce_rows}
            except Exception:
                cce_cache = {}
        if dle_Ps and psat_cache:
            try:
                dle_rows = run_dle(z, comp_names, T_R, dle_Ps, c7_new, P_b=psat_cache)
                dle_cache = {row["P"]: row for row in dle_rows}
            except Exception:
                dle_cache = {}

        for m in measurements:
            t = m["type"]
            if t == "Psat":
                predictions.append(psat_cache if psat_cache else 0.0)
            elif t == "V_rel":
                row = (cce_cache or {}).get(m["P"])
                predictions.append(row["V_rel"] if row else np.nan)
            elif t == "L_dropout":
                row = (cce_cache or {}).get(m["P"])
                predictions.append(row["L_dropout_pct"] if row else np.nan)
            elif t == "Rs":
                row = (dle_cache or {}).get(m["P"])
                predictions.append(row["Rs"] if row else np.nan)
            elif t == "Bo":
                row = (dle_cache or {}).get(m["P"])
                predictions.append(row["Bo"] if row else np.nan)
            elif t in ("GOR", "rho_st_oil"):
                if sep_cache is None:
                    train = m.get("train", [(14.7, 60.0)])
                    try:
                        sep_cache = run_separator_train(z, comp_names, train, c7_new)
                    except Exception:
                        sep_cache = {"GOR_scfSTB": 0.0, "st_oil_density": 0.0}
                predictions.append(sep_cache["GOR_scfSTB"] if t == "GOR"
                                    else sep_cache["st_oil_density"])
            else:
                predictions.append(np.nan)

    return np.array(predictions, dtype=float)


def tune_eos(z, comp_names, T_R, c7_base, measurements,
              free_params=None, max_iter=30):
    """Run LM least-squares regression on EOS parameters."""
    if free_params is None:
        free_params = ["Pc_C7+", "Tc_C7+", "omega_C7+", "kij_C1_C7+"]
    free_map = {"Pc_C7+": 0, "Tc_C7+": 1, "omega_C7+": 2,
                "kij_C1_C7+": 3, "kij_N2_C7+": 4}
    free_idx = sorted({free_map[k] for k in free_params if k in free_map})
    all_param_names = ["Pc_C7+ mult", "Tc_C7+ mult", "omega_C7+ mult",
                       "kij(C1,C7+)", "kij(N2,C7+)"]

    x0_full = np.array([1.0, 1.0, 1.0, 0.115, 0.115])
    lo_full = np.array([0.7, 0.7, 0.3, -0.1, -0.1])
    hi_full = np.array([1.3, 1.3, 2.0,  0.3,  0.3])

    x0 = x0_full[free_idx]; lo = lo_full[free_idx]; hi = hi_full[free_idx]

    y_obs = np.array([m["value"] for m in measurements])
    weights = np.array([m.get("weight", 1.0) for m in measurements])
    scales = np.where(np.abs(y_obs) > 1e-3, np.abs(y_obs), 1.0)

    pred_init = predict_all(z, comp_names, T_R, x0_full, c7_base, measurements)

    def residual_fn(x_free):
        full = x0_full.copy()
        for i, idx in enumerate(free_idx):
            full[idx] = x_free[i]
        try:
            pred = predict_all(z, comp_names, T_R, full, c7_base, measurements)
        except Exception:
            return 1e6 * np.ones_like(y_obs)
        # Replace NaN predictions with a large finite penalty
        bad = ~np.isfinite(pred)
        if np.any(bad):
            pred = np.where(bad, y_obs * 10.0, pred)
        return weights * (pred - y_obs) / scales

    result = least_squares(residual_fn, x0=x0, bounds=(lo, hi),
                            method="trf", max_nfev=max_iter * 5,
                            xtol=1e-6, ftol=1e-6)

    full_opt = x0_full.copy()
    for i, idx in enumerate(free_idx):
        full_opt[idx] = result.x[i]
    pred_final = predict_all(z, comp_names, T_R, full_opt, c7_base, measurements)

    return {
        "param_names": all_param_names,
        "free_params": [all_param_names[i] for i in free_idx],
        "x_full_init":  x0_full,
        "x_full_final": full_opt,
        "predicted_initial": pred_init,
        "predicted_final":   pred_final,
        "observed":          y_obs,
        "rms_initial": float(np.sqrt(np.mean(((pred_init - y_obs) / scales) ** 2))),
        "rms_final":   float(np.sqrt(np.mean(((pred_final - y_obs) / scales) ** 2))),
        "n_iter":   int(result.nfev),
        "success":  bool(result.success),
        "message":  str(result.message),
    }
