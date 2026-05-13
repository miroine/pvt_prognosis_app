"""
Correlation tuning for black-oil PVT.

Strategy:
    The correlations (Standing, Vasquez-Beggs, Glaso, Lasater) have fixed
    coefficients. We can't easily refit those; what we *can* do is apply
    a multiplicative or additive shift to the predictions to match lab data.

    Tunable adjustments:
        Pb_shift     : additive shift on bubble point (psia)
        Rs_factor    : multiplicative scale on Rs at all P (default 1.0)
        Bo_factor    : multiplicative scale on Bo (default 1.0)
        mu_factor    : multiplicative scale on viscosity (default 1.0)

    Plus we let the user pick the best of multiple correlation choices via
    a separate "auto-select" workflow.

This is intentionally simple — for proper coefficient regression, fit the
EOS instead via eos_tuning.py.
"""

import numpy as np
from scipy.optimize import minimize


def apply_corrections(rows, Pb_base, Pb_shift=0.0,
                       Rs_factor=1.0, Bo_factor=1.0, mu_factor=1.0):
    """
    Apply tuning corrections to a list of {P, Rs, Bo, mu_o} rows.
    Pb_base is the unshifted bubble point; we shift to Pb_base + Pb_shift.

    Below the new Pb: Rs/Bo apply as multiplicative.
    Above the new Pb: Rs is held at Rsi*Rs_factor; Bo is scaled by Bo_factor.
    """
    Pb_new = Pb_base + Pb_shift
    out = []
    for r in rows:
        new = dict(r)
        new["Bo"] = r["Bo"] * Bo_factor
        new["Rs"] = r["Rs"] * Rs_factor
        new["mu_o"] = r["mu_o"] * mu_factor
        new["P_relative_to_Pb"] = r["P"] - Pb_new
        out.append(new)
    return out, Pb_new


def tune_correlation_oil(corr_class, base_params, lab_data,
                          tune=("Pb_shift", "Bo_factor"),
                          n_grid_points=50):
    """
    Optimize correlation tuning parameters against lab measurements.

    Args:
        corr_class   : OilCorrelations class
        base_params  : dict with api, gas_sg, T, Rsi, rs_corr, bo_corr, mu_corr
        lab_data     : list of dicts, each with 'type' ('Pb', 'Rs', 'Bo', 'mu_o')
                       'P' (psia, except for Pb where it's the *value* itself),
                       'value' (measurement)
        tune         : tuple of which adjustments to free
        n_grid_points: pressure grid for evaluation

    Returns dict with tuned parameters, RMS before/after, and predicted vs
    observed arrays for plotting.
    """
    api = base_params["api"]
    gas_sg = base_params["gas_sg"]
    T = base_params["T"]
    Rsi = base_params["Rsi"]
    rs_corr = base_params.get("rs_corr", "Standing")
    bo_corr = base_params.get("bo_corr", "Standing")
    mu_corr = base_params.get("mu_corr", "Beggs-Robinson")

    corr = corr_class(api=api, gas_sg=gas_sg, T=T,
                       rs_corr=rs_corr, bo_corr=bo_corr, mu_corr=mu_corr)
    Pb_base = corr.bubble_point(Rsi)

    # Baseline predictions for each measurement
    def predict_one(adj, m):
        Pb_shift = adj.get("Pb_shift", 0.0)
        Rs_factor = adj.get("Rs_factor", 1.0)
        Bo_factor = adj.get("Bo_factor", 1.0)
        mu_factor = adj.get("mu_factor", 1.0)
        Pb_tuned = Pb_base + Pb_shift
        if m["type"] == "Pb":
            return Pb_tuned
        P = m["P"]
        if P <= Pb_tuned:
            Rs = corr.solution_gor(P) * Rs_factor
            Bo = corr.formation_volume_factor(P, Rs / Rs_factor, saturated=True) * Bo_factor
            mu = corr.viscosity(P, Rs / Rs_factor, Pb_tuned, saturated=True) * mu_factor
        else:
            Rs = Rsi * Rs_factor
            Bo = corr.formation_volume_factor(P, Rsi, saturated=False, Pb=Pb_tuned) * Bo_factor
            mu = corr.viscosity(P, Rsi, Pb_tuned, saturated=False) * mu_factor
        return {"Rs": Rs, "Bo": Bo, "mu_o": mu}[m["type"]]

    y_obs = np.array([m["value"] for m in lab_data])
    scales = np.where(np.abs(y_obs) > 1e-3, np.abs(y_obs), 1.0)

    # Initial predictions (no tuning)
    pred_init = np.array([predict_one({}, m) for m in lab_data])

    # Tuning vars
    tune_init = {"Pb_shift": 0.0, "Rs_factor": 1.0, "Bo_factor": 1.0, "mu_factor": 1.0}
    bounds_map = {
        "Pb_shift":  (-1000.0, 1000.0),
        "Rs_factor": (0.7, 1.3),
        "Bo_factor": (0.85, 1.15),
        "mu_factor": (0.5, 2.0),
    }
    tune_keys = list(tune)
    x0 = np.array([tune_init[k] for k in tune_keys])
    bounds = [bounds_map[k] for k in tune_keys]

    def objective(x):
        adj = dict(tune_init)
        for i, k in enumerate(tune_keys):
            adj[k] = x[i]
        pred = np.array([predict_one(adj, m) for m in lab_data])
        return np.sum(((pred - y_obs) / scales) ** 2)

    res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
    adj_final = dict(tune_init)
    for i, k in enumerate(tune_keys):
        adj_final[k] = res.x[i]

    pred_final = np.array([predict_one(adj_final, m) for m in lab_data])

    return {
        "tuned":             adj_final,
        "Pb_base":           Pb_base,
        "Pb_tuned":          Pb_base + adj_final["Pb_shift"],
        "predicted_initial": pred_init,
        "predicted_final":   pred_final,
        "observed":          y_obs,
        "rms_initial": float(np.sqrt(np.mean(((pred_init - y_obs) / scales) ** 2))),
        "rms_final":   float(np.sqrt(np.mean(((pred_final - y_obs) / scales) ** 2))),
        "tuned_keys":  tune_keys,
        "success":     bool(res.success),
        "message":     str(res.message),
    }


def auto_select_best_correlation(corr_class, base_params, lab_data,
                                   correlations_to_try=None):
    """
    Run the same lab data through several correlation choices and report
    which combination gives the lowest RMS (with Pb_shift tuning enabled
    so each correlation gets a fair chance to match the Pb).
    """
    if correlations_to_try is None:
        correlations_to_try = [
            ("Standing", "Standing"),
            ("Vasquez-Beggs", "Vasquez-Beggs"),
            ("Glaso", "Glaso"),
            ("Lasater", "Standing"),
        ]
    results = []
    for rs_c, bo_c in correlations_to_try:
        try:
            params = dict(base_params)
            params["rs_corr"] = rs_c
            params["bo_corr"] = bo_c
            tune = tune_correlation_oil(corr_class, params, lab_data,
                                          tune=("Pb_shift",))
            results.append({
                "rs_corr": rs_c, "bo_corr": bo_c,
                "rms_baseline": tune["rms_initial"],
                "rms_with_Pb_shift": tune["rms_final"],
                "Pb_shift": tune["tuned"]["Pb_shift"],
            })
        except Exception as e:
            results.append({
                "rs_corr": rs_c, "bo_corr": bo_c,
                "rms_baseline": float("inf"), "rms_with_Pb_shift": float("inf"),
                "Pb_shift": 0.0, "error": str(e),
            })
    results.sort(key=lambda r: r["rms_with_Pb_shift"])
    return results
