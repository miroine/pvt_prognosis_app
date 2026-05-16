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
                          n_grid_points=50, max_iter=100, tol=1e-6):
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

    res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B",
                    options={"maxiter": max_iter, "ftol": tol, "gtol": tol})
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


# ============================================================
# Wet-gas correlation tuning
# ============================================================
def tune_wetgas(wet_class, base_params, lab_data,
                 tune=("Pdew_shift", "Rv_factor"), max_iter=40, tol=1e-6):
    """
    Tune a wet-gas correlation against lab data.

    Tunable adjustments:
        Pdew_shift : additive shift on dew point (psia)
        Rv_factor  : multiplicative scale on Rv (vaporized oil ratio)
        Z_factor   : multiplicative scale on Z-factor

    base_params: dict with gas_sg, api_cond, cgr, T, N2, CO2, H2S,
                 z_corr, mu_corr, rv_corr, Pdew
    lab_data: list of dicts with 'type' ('Pdew','Z','Rv','Bg'),
              'P' (psia, except Pdew), 'value', optional 'weight'
    tol: L-BFGS-B function-tolerance.

    Returns dict similar to tune_correlation_oil.
    """
    from scipy.optimize import minimize

    def build(adj):
        """Build a wet-gas instance with the given adjustments."""
        Pdew_shift = adj.get("Pdew_shift", 0.0)
        w = wet_class(
            gas_sg=base_params["gas_sg"],
            api_cond=base_params["api_cond"],
            cgr_stb_per_mmscf=base_params["cgr"],
            T=base_params["T"],
            N2=base_params.get("N2", 0.0),
            CO2=base_params.get("CO2", 0.0),
            H2S=base_params.get("H2S", 0.0),
            z_corr=base_params.get("z_corr", "Hall-Yarborough"),
            mu_corr=base_params.get("mu_corr", "Lee-Gonzalez-Eakin"),
            rv_corr=base_params.get("rv_corr", "Linear-Pdew"),
            Pdew=base_params["Pdew"] + Pdew_shift)
        return w

    def predict_one(adj, m):
        Rv_factor = adj.get("Rv_factor", 1.0)
        Z_factor = adj.get("Z_factor", 1.0)
        w = build(adj)
        if m["type"] == "Pdew":
            return base_params["Pdew"] + adj.get("Pdew_shift", 0.0)
        P = m["P"]
        Z = w.z_factor(P) * Z_factor
        if m["type"] == "Z":
            return Z
        if m["type"] == "Rv":
            return w.rv(P) * Rv_factor
        if m["type"] == "Bg":
            return w.formation_volume_factor(P, Z)
        return np.nan

    y_obs = np.array([m["value"] for m in lab_data])
    scales = np.where(np.abs(y_obs) > 1e-9, np.abs(y_obs), 1.0)
    pred_init = np.array([predict_one({}, m) for m in lab_data])

    tune_init = {"Pdew_shift": 0.0, "Rv_factor": 1.0, "Z_factor": 1.0}
    bounds_map = {
        "Pdew_shift": (-2000.0, 2000.0),
        "Rv_factor":  (0.5, 2.0),
        "Z_factor":   (0.85, 1.15),
    }
    tune_keys = list(tune)
    x0 = np.array([tune_init[k] for k in tune_keys])
    bounds = [bounds_map[k] for k in tune_keys]

    def objective(x):
        adj = dict(tune_init)
        for i, k in enumerate(tune_keys):
            adj[k] = x[i]
        try:
            pred = np.array([predict_one(adj, m) for m in lab_data])
        except Exception:
            return 1e9
        bad = ~np.isfinite(pred)
        if np.any(bad):
            pred = np.where(bad, y_obs * 10.0, pred)
        return float(np.sum(((pred - y_obs) / scales) ** 2))

    res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B",
                    options={"maxiter": max_iter, "ftol": tol, "gtol": tol})
    adj_final = dict(tune_init)
    for i, k in enumerate(tune_keys):
        adj_final[k] = res.x[i]
    pred_final = np.array([predict_one(adj_final, m) for m in lab_data])

    return {
        "tuned":             adj_final,
        "tuned_keys":        tune_keys,
        "predicted_initial": pred_init,
        "predicted_final":   pred_final,
        "observed":          y_obs,
        "rms_initial": float(np.sqrt(np.mean(((pred_init - y_obs) / scales) ** 2))),
        "rms_final":   float(np.sqrt(np.mean(((pred_final - y_obs) / scales) ** 2))),
        "success":     bool(res.success),
    }


# ============================================================
# Dry-gas correlation tuning
# ============================================================
def tune_drygas(gas_class, base_params, lab_data,
                 tune=("Z_factor",), max_iter=40, tol=1e-6):
    """
    Tune a dry-gas correlation against lab data.

    Tunable adjustments:
        Z_factor   : multiplicative scale on Z-factor
        mu_factor  : multiplicative scale on viscosity

    base_params: dict with gas_sg, T, N2, CO2, H2S, z_corr, mu_corr
    lab_data: list of dicts with 'type' ('Z','Bg','mu_g'),
              'P' (psia), 'value', optional 'weight'

    Returns dict similar to other tuning functions.
    """
    def build():
        return gas_class(
            gas_sg=base_params["gas_sg"], T=base_params["T"],
            N2=base_params.get("N2", 0.0),
            CO2=base_params.get("CO2", 0.0),
            H2S=base_params.get("H2S", 0.0),
            z_corr=base_params.get("z_corr", "Hall-Yarborough"),
            mu_corr=base_params.get("mu_corr", "Lee-Gonzalez-Eakin"))

    def predict_one(adj, m):
        Z_factor = adj.get("Z_factor", 1.0)
        mu_factor = adj.get("mu_factor", 1.0)
        g = build()
        P = m["P"]
        Z = g.z_factor(P) * Z_factor
        if m["type"] == "Z":
            return Z
        if m["type"] == "Bg":
            return g.formation_volume_factor(P, Z)
        if m["type"] == "mu_g":
            return g.viscosity(P, Z) * mu_factor
        return np.nan

    y_obs = np.array([m["value"] for m in lab_data])
    scales = np.where(np.abs(y_obs) > 1e-9, np.abs(y_obs), 1.0)
    pred_init = np.array([predict_one({}, m) for m in lab_data])

    tune_init = {"Z_factor": 1.0, "mu_factor": 1.0}
    bounds_map = {"Z_factor": (0.85, 1.15), "mu_factor": (0.5, 2.0)}
    tune_keys = list(tune)
    x0 = np.array([tune_init[k] for k in tune_keys])
    bounds = [bounds_map[k] for k in tune_keys]

    def objective(x):
        adj = dict(tune_init)
        for i, k in enumerate(tune_keys):
            adj[k] = x[i]
        try:
            pred = np.array([predict_one(adj, m) for m in lab_data])
        except Exception:
            return 1e9
        bad = ~np.isfinite(pred)
        if np.any(bad):
            pred = np.where(bad, y_obs * 10.0, pred)
        return float(np.sum(((pred - y_obs) / scales) ** 2))

    res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B",
                    options={"maxiter": max_iter, "ftol": tol, "gtol": tol})
    adj_final = dict(tune_init)
    for i, k in enumerate(tune_keys):
        adj_final[k] = res.x[i]
    pred_final = np.array([predict_one(adj_final, m) for m in lab_data])

    return {
        "tuned":             adj_final,
        "tuned_keys":        tune_keys,
        "predicted_initial": pred_init,
        "predicted_final":   pred_final,
        "observed":          y_obs,
        "rms_initial": float(np.sqrt(np.mean(((pred_init - y_obs) / scales) ** 2))),
        "rms_final":   float(np.sqrt(np.mean(((pred_final - y_obs) / scales) ** 2))),
        "success":     bool(res.success),
    }


# ============================================================
# Tuned correlation wrappers — apply tuning factors transparently
# ============================================================
class TunedOilCorrelations:
    """Wraps an OilCorrelations instance and applies tuning adjustments.

    The wrapper exposes the same interface as OilCorrelations (bubble_point,
    solution_gor, formation_volume_factor, viscosity, oil_compressibility)
    but applies Pb_shift / Rs_factor / Bo_factor / mu_factor so downstream
    code (property plots, experiments, ECLIPSE export) can use a tuned
    fluid without any special-casing.
    """
    def __init__(self, base_corr, tuned_adjustments):
        self._base = base_corr
        self._adj = dict(tuned_adjustments or {})
        # Expose common attributes
        self.api = base_corr.api
        self.gamma_o = base_corr.gamma_o
        self.gamma_g = base_corr.gamma_g
        self.T = base_corr.T
        self.T_R = base_corr.T_R

    @property
    def Pb_shift(self):
        return self._adj.get("Pb_shift", 0.0)

    @property
    def Rs_factor(self):
        return self._adj.get("Rs_factor", 1.0)

    @property
    def Bo_factor(self):
        return self._adj.get("Bo_factor", 1.0)

    @property
    def mu_factor(self):
        return self._adj.get("mu_factor", 1.0)

    def bubble_point(self, Rsi):
        return self._base.bubble_point(Rsi) + self.Pb_shift

    def solution_gor(self, P):
        return self._base.solution_gor(P) * self.Rs_factor

    def formation_volume_factor(self, P, Rs, saturated=True, Pb=None):
        # Rs passed in may already be tuned; un-scale before passing to base
        Rs_base = Rs / self.Rs_factor if self.Rs_factor else Rs
        return self._base.formation_volume_factor(
            P, Rs_base, saturated=saturated, Pb=Pb) * self.Bo_factor

    def viscosity(self, P, Rs, Pb, saturated=True):
        Rs_base = Rs / self.Rs_factor if self.Rs_factor else Rs
        return self._base.viscosity(
            P, Rs_base, Pb, saturated=saturated) * self.mu_factor

    def oil_compressibility(self, P, Rs):
        Rs_base = Rs / self.Rs_factor if self.Rs_factor else Rs
        return self._base.oil_compressibility(P, Rs_base)


class TunedWetGasCorrelations:
    """Wraps a WetGasCorrelations instance with tuning adjustments
    (Pdew_shift / Rv_factor / Z_factor)."""
    def __init__(self, base_corr, tuned_adjustments):
        self._base = base_corr
        self._adj = dict(tuned_adjustments or {})
        self.gamma_g = getattr(base_corr, "gamma_g", None)
        self.gamma_g_res = getattr(base_corr, "gamma_g_res", None)
        self.T = base_corr.T
        self.T_R = base_corr.T_R
        # Rv_max scales with the Rv tuning factor
        _rvmax = getattr(base_corr, "Rv_max", None)
        if _rvmax is not None:
            self.Rv_max = _rvmax * self._adj.get("Rv_factor", 1.0)
        # Apply Pdew shift directly to a stored attribute
        self.Pdew = getattr(base_corr, "Pdew", None)
        if self.Pdew is not None:
            self.Pdew = self.Pdew + self._adj.get("Pdew_shift", 0.0)

    @property
    def Z_factor_mult(self):
        return self._adj.get("Z_factor", 1.0)

    @property
    def Rv_factor(self):
        return self._adj.get("Rv_factor", 1.0)

    def z_factor(self, P):
        return self._base.z_factor(P) * self.Z_factor_mult

    def formation_volume_factor(self, P, Z=None):
        if Z is None:
            Z = self.z_factor(P)
        # Bg scales with Z, so recompute via base with the tuned Z
        return self._base.formation_volume_factor(P, Z)

    def rv(self, P):
        return self._base.rv(P) * self.Rv_factor

    def viscosity(self, P, Z=None):
        if Z is None:
            Z = self.z_factor(P)
        return self._base.viscosity(P, Z)


class TunedGasCorrelations:
    """Wraps a GasCorrelations (dry gas) instance with Z_factor / mu_factor."""
    def __init__(self, base_corr, tuned_adjustments):
        self._base = base_corr
        self._adj = dict(tuned_adjustments or {})
        self.gamma_g = base_corr.gamma_g
        self.T = base_corr.T
        self.T_R = base_corr.T_R

    @property
    def Z_factor_mult(self):
        return self._adj.get("Z_factor", 1.0)

    @property
    def mu_factor(self):
        return self._adj.get("mu_factor", 1.0)

    def z_factor(self, P):
        return self._base.z_factor(P) * self.Z_factor_mult

    def formation_volume_factor(self, P, Z=None):
        if Z is None:
            Z = self.z_factor(P)
        return self._base.formation_volume_factor(P, Z)

    def viscosity(self, P, Z=None):
        if Z is None:
            Z = self.z_factor(P)
        return self._base.viscosity(P, Z) * self.mu_factor
