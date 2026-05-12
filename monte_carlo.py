"""
Monte Carlo uncertainty quantification for black-oil correlations.

The user provides a base case (API, gas SG, Rsi, T) and uncertainty ranges
(typically ±5-10% or a standard deviation). We draw N samples from normal
distributions, run the correlations at each draw, and produce distributions
of Pb, Bo at reservoir P, Rs at reservoir P, and viscosity.

Distributions are summarized by mean, P10, P50, P90 and rendered as histograms
plus a tornado plot showing parameter sensitivity.
"""

import numpy as np


def sample_normal(mean, std, n, lower=None, upper=None):
    """Sample from a truncated normal distribution."""
    samples = np.random.normal(mean, std, n)
    if lower is not None:
        samples = np.maximum(samples, lower)
    if upper is not None:
        samples = np.minimum(samples, upper)
    return samples


def sample_uniform(low, high, n):
    """Uniform between low and high."""
    return np.random.uniform(low, high, n)


def run_monte_carlo_oil(base_params, uncertainties, n_samples=500,
                          correlation_class=None, dist_type="normal",
                          target_P=None, seed=42):
    """
    Monte Carlo for oil correlations.

    base_params: dict with keys 'api', 'gas_sg', 'Rsi', 'T', plus correlation choices
    uncertainties: dict mapping param name to (std_or_range, dist_type)
        For normal: std = uncertainties[param] (in absolute units)
        For uniform: (low_offset, high_offset) relative to base
    correlation_class: OilCorrelations class
    target_P: pressure at which to evaluate Bo, Rs, mu (usually reservoir P)
    seed: RNG seed for reproducibility

    Returns dict with arrays of sampled parameters and resulting Pb, Bo, mu.
    """
    if seed is not None:
        np.random.seed(seed)

    n = n_samples
    # Sample each uncertain parameter
    sampled = {}
    for key in ['api', 'gas_sg', 'Rsi', 'T']:
        base = base_params[key]
        unc = uncertainties.get(key, 0.0)
        if unc <= 0:
            sampled[key] = np.full(n, base)
        else:
            if dist_type == "normal":
                # Lower bounds: API > 5, gas_sg > 0.55, Rsi > 0, T > 60F
                bounds = {"api": (5, 70), "gas_sg": (0.55, 2.0),
                          "Rsi": (0, 5000), "T": (60, 500)}
                lo, hi = bounds.get(key, (-np.inf, np.inf))
                sampled[key] = sample_normal(base, unc, n, lo, hi)
            else:  # uniform
                lo, hi = base - unc, base + unc
                sampled[key] = sample_uniform(lo, hi, n)

    # Run correlation for each sample
    Pb_arr = np.zeros(n)
    Bo_arr = np.zeros(n)
    Rs_arr = np.zeros(n)
    mu_arr = np.zeros(n)

    for i in range(n):
        try:
            corr = correlation_class(
                api=sampled['api'][i],
                gas_sg=sampled['gas_sg'][i],
                T=sampled['T'][i],
                rs_corr=base_params.get('rs_corr', 'Standing'),
                bo_corr=base_params.get('bo_corr', 'Standing'),
                mu_corr=base_params.get('mu_corr', 'Beggs-Robinson'),
            )
            Pb_i = corr.bubble_point(sampled['Rsi'][i])
            Pb_arr[i] = Pb_i
            if target_P is not None and target_P > 0:
                if target_P <= Pb_i:
                    Rs_arr[i] = corr.solution_gor(target_P)
                    Bo_arr[i] = corr.formation_volume_factor(target_P, Rs_arr[i], saturated=True)
                    mu_arr[i] = corr.viscosity(target_P, Rs_arr[i], Pb_i, saturated=True)
                else:
                    Rs_arr[i] = sampled['Rsi'][i]
                    Bo_arr[i] = corr.formation_volume_factor(target_P, sampled['Rsi'][i],
                                                              saturated=False, Pb=Pb_i)
                    mu_arr[i] = corr.viscosity(target_P, sampled['Rsi'][i], Pb_i, saturated=False)
            else:
                Bo_arr[i] = corr.formation_volume_factor(Pb_i, sampled['Rsi'][i], saturated=True)
                Rs_arr[i] = sampled['Rsi'][i]
                mu_arr[i] = corr.viscosity(Pb_i, sampled['Rsi'][i], Pb_i, saturated=True)
        except Exception:
            Pb_arr[i] = Bo_arr[i] = Rs_arr[i] = mu_arr[i] = np.nan

    return {
        "samples":    sampled,
        "Pb":         Pb_arr,
        "Bo":         Bo_arr,
        "Rs":         Rs_arr,
        "mu":         mu_arr,
        "n_samples":  n,
        "target_P":   target_P,
    }


def percentiles(arr, ps=(10, 50, 90)):
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {f"P{p}": np.nan for p in ps}
    return {f"P{p}": float(np.percentile(arr, p)) for p in ps}


def summary_stats(arr):
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": np.nan, "std": np.nan, "P10": np.nan, "P50": np.nan, "P90": np.nan}
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
        "P10":  float(np.percentile(arr, 10)),
        "P50":  float(np.percentile(arr, 50)),
        "P90":  float(np.percentile(arr, 90)),
    }


def tornado_sensitivity(base_params, uncertainties, correlation_class,
                         target_P, output="Pb"):
    """
    Tornado sensitivity: for each parameter, compute the range of the output
    when that parameter sweeps ±1σ while others are held at base.

    Returns list of (param_name, low_val, high_val, range) tuples, sorted
    by range descending.
    """
    rows = []
    base = dict(base_params)

    # Baseline output
    def evaluate(params):
        try:
            corr = correlation_class(
                api=params['api'], gas_sg=params['gas_sg'], T=params['T'],
                rs_corr=params.get('rs_corr', 'Standing'),
                bo_corr=params.get('bo_corr', 'Standing'),
                mu_corr=params.get('mu_corr', 'Beggs-Robinson'),
            )
            Pb = corr.bubble_point(params['Rsi'])
            if output == "Pb":
                return Pb
            if target_P <= Pb:
                Rs = corr.solution_gor(target_P)
                Bo = corr.formation_volume_factor(target_P, Rs, saturated=True)
                mu = corr.viscosity(target_P, Rs, Pb, saturated=True)
            else:
                Rs = params['Rsi']
                Bo = corr.formation_volume_factor(target_P, Rs, saturated=False, Pb=Pb)
                mu = corr.viscosity(target_P, Rs, Pb, saturated=False)
            return {"Bo": Bo, "Rs": Rs, "mu": mu, "Pb": Pb}.get(output, Pb)
        except Exception:
            return np.nan

    base_val = evaluate(base)

    for param in ['api', 'gas_sg', 'Rsi', 'T']:
        unc = uncertainties.get(param, 0)
        if unc <= 0:
            continue
        lo_params = dict(base); lo_params[param] = base[param] - unc
        hi_params = dict(base); hi_params[param] = base[param] + unc
        lo_val = evaluate(lo_params)
        hi_val = evaluate(hi_params)
        if not (np.isnan(lo_val) or np.isnan(hi_val)):
            rng = abs(hi_val - lo_val)
            rows.append((param, lo_val, hi_val, rng))

    rows.sort(key=lambda r: r[3], reverse=True)
    return {"rows": rows, "base_value": base_val}
