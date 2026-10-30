"""S-curve (logistic / Gompertz) fits on cumulative publication and patent counts, with parametric-bootstrap uncertainty.

Method (documented so the report can cite it):
  * Fit on CUMULATIVE counts C(t) of the last complete year and earlier (2026 is partial and excluded from fits).
  * Logistic  C(t) = K / (1 + exp(-r (t - t0)))           -> inflection at t0, C(t0) = K/2, 10-90% takeover time = ln(81)/r
  * Gompertz  C(t) = K * exp(-exp(-r (t - t0)))          -> inflection at t0, C(t0) = K/e (asymmetric, slower late phase)
  * Exponential benchmark on ANNUAL counts  n(t) = a * exp(b t)   (no saturation)
  * Bootstrap: annual counts resampled as Poisson(n_t) 1000x, refit; report median and 5-95% range of K, t0, and
    the share of fits where K hit the upper bound (= saturation level NOT identifiable from the data).
  * K is bounded to [C_last, 30 * C_last]. If many bootstrap fits sit at the upper bound the curve is still in its
    exponential phase and any 'ceiling' is an artefact of the bound; the report must say so.
"""
import json, warnings
import numpy as np
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore')

def logistic(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))

def gompertz(t, K, r, t0):
    return K * np.exp(-np.exp(-r * (t - t0)))

MODELS = {'logistic': logistic, 'gompertz': gompertz}

def _fit(model, years, cum, kmax_factor=30.0):
    c_last = cum[-1]
    p0 = [c_last * 2.0, 0.3, years[-1]]
    lo = [c_last, 0.01, years[0] - 5]
    hi = [c_last * kmax_factor, 3.0, years[-1] + 40]
    p, _ = curve_fit(model, years, cum, p0=p0, bounds=(lo, hi), maxfev=20000)
    return p, hi[0]

def aic_from_sse(sse, n, k):
    return n * np.log(sse / n) + 2 * k

def fit_series(name, years, annual, first_year=None, last_full_year=2025, n_boot=1000, seed=0):
    years = np.asarray(years, dtype=float); annual = np.asarray(annual, dtype=float)
    m = years <= last_full_year
    if first_year is not None:
        m &= years >= first_year
    # cumulative includes everything before first_year as an offset so C(t) is a true cumulative
    offset = annual[years < first_year].sum() if first_year is not None else 0.0
    y, n = years[m], annual[m]
    cum = np.cumsum(n) + offset
    res = {'series': name, 'years_fitted': [int(y[0]), int(y[-1])], 'cumulative_at_end': float(cum[-1]), 'annual_last': float(n[-1])}
    for mname, model in MODELS.items():
        try:
            p, kmax = _fit(model, y, cum)
        except Exception as e:
            res[mname] = {'error': str(e)}; continue
        pred = model(y, *p); sse = float(np.sum((cum - pred) ** 2))
        # annual-count residuals are a fairer goodness-of-fit than cumulative ones
        pred_ann = np.diff(np.concatenate([[model(y[0] - 1, *p)], pred]))
        sse_ann = float(np.sum((n - pred_ann) ** 2))
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            nb = rng.poisson(np.maximum(n, 0))
            cb = np.cumsum(nb) + offset
            try:
                pb, kb = _fit(model, y, cb); boots.append(list(pb) + [pb[0] >= 0.98 * kb])
            except Exception:
                pass
        B = np.array(boots, dtype=float)
        at_bound = float(np.mean(B[:, 3])) if len(B) else None
        q = lambda a: [float(np.percentile(a, 5)), float(np.percentile(a, 50)), float(np.percentile(a, 95))]
        takeover = (np.log(81) / B[:, 1]) if mname == 'logistic' else (np.log(np.log(10) / np.log(10 / 9)) / B[:, 1])
        res[mname] = {
            'K': float(p[0]), 'r': float(p[1]), 't0_inflection': float(p[2]),
            'share_of_K_reached': float(cum[-1] / p[0]),
            'K_at_upper_bound': bool(p[0] >= 0.98 * kmax),
            'sse_cum': sse, 'aic_annual': float(aic_from_sse(max(sse_ann, 1e-9), len(n), 3)),
            'boot_K_5_50_95': q(B[:, 0]), 'boot_t0_5_50_95': q(B[:, 2]), 'boot_takeover_10_90_years_5_50_95': q(takeover),
            'boot_share_K_at_upper_bound': at_bound, 'n_boot_ok': int(len(B)),
            'boot_params_sample': B[:200, :3].round(4).tolist(),
        }
    # exponential benchmark on annual counts (log-linear least squares on n>0, then AIC on raw scale)
    pos = n > 0
    if pos.sum() >= 3:
        b, loga = np.polyfit(y[pos], np.log(n[pos]), 1)
        pred_e = np.exp(loga + b * y)
        res['exponential'] = {'growth_rate_per_year': float(np.exp(b) - 1), 'doubling_time_years': float(np.log(2) / b) if b > 0 else None,
                              'aic_annual': float(aic_from_sse(max(float(np.sum((n - pred_e) ** 2)), 1e-9), len(n), 2))}
        # recent growth (last 5 complete years)
        y5 = y[-5:]; n5 = n[-5:]
        if (n5 > 0).all():
            b5, _ = np.polyfit(y5, np.log(n5), 1)
            res['exponential']['growth_rate_last5'] = float(np.exp(b5) - 1)
    return res

def fisher_pry(years, share, last_full_year=2025, first_year=2010):
    """Fisher-Pry substitution: ln(f/(1-f)) linear in t. Returns slope, implied f=50% year and the fit quality.
    The 50% year is an extrapolation of the CURRENT substitution rate, not a forecast."""
    years = np.asarray(years, float); share = np.asarray(share, float)
    m = (years <= last_full_year) & (years >= first_year) & (share > 0) & (share < 1)
    y, f = years[m], share[m]
    L = np.log(f / (1 - f))
    b, a = np.polyfit(y, L, 1)
    pred = a + b * y; ss_res = np.sum((L - pred) ** 2); ss_tot = np.sum((L - L.mean()) ** 2)
    return {'years': [int(y[0]), int(y[-1])], 'slope_per_year': float(b), 'r2': float(1 - ss_res / ss_tot),
            'year_f_10pct': float((np.log(0.1 / 0.9) - a) / b) if b > 0 else None,
            'year_f_50pct': float((0 - a) / b) if b > 0 else None, 'share_last': float(f[-1])}

def scenario_fits(years, annual, first_year, multiples=(2, 3, 5, 10, 20), last_full_year=2025):
    """Logistic fits with the saturation level K FIXED at multiples of today's cumulative count.
    Because K is not identifiable while growth is still accelerating, this shows what the data imply
    *conditional on* an assumed ultimate field size: the inflection year and the AIC of each scenario."""
    years = np.asarray(years, float); annual = np.asarray(annual, float)
    offset = annual[years < first_year].sum()
    m = (years >= first_year) & (years <= last_full_year)
    y, n = years[m], annual[m]; cum = np.cumsum(n) + offset
    out = []
    for mult in multiples:
        K = mult * cum[-1]
        f = lambda t, r, t0: logistic(t, K, r, t0)
        try:
            (r, t0), _ = curve_fit(f, y, cum, p0=[0.3, y[-1] + 3], bounds=([0.01, y[0]], [3.0, y[-1] + 60]), maxfev=20000)
        except Exception as e:
            out.append({'K_multiple': mult, 'error': str(e)}); continue
        pred = f(y, r, t0); pred_ann = np.diff(np.concatenate([[f(y[0] - 1, r, t0)], pred]))
        out.append({'K_multiple': mult, 'K': float(K), 'r': float(r), 't0_inflection': float(t0), 'takeover_10_90_years': float(np.log(81) / r),
                    'year_reach_90pct': float(t0 + np.log(9) / r), 'aic_annual': float(aic_from_sse(max(float(np.sum((n - pred_ann) ** 2)), 1e-9), len(n), 2))})
    return out

def yoy_growth(years, annual, start=2010, end=2025):
    years = np.asarray(years); annual = np.asarray(annual, float)
    return {int(y): (float(annual[i] / annual[i - 1] - 1) if annual[i - 1] > 0 else None) for i, y in enumerate(years) if start <= y <= end and i > 0}
