"""v2 step 19 - uncertainty ranges for the tier-1 trends (bootstrap) + sensitivity checks.
Reads  data/processed/v2/tier1/{perf_points,durability}.csv (v2_08), tier1/trend_summary.csv (v2_09, for a consistency check),
       data/processed/v2/tier2/{perf_points,durability}.csv (v2_11).
Writes data/processed/v2/uncertainty/ci_by_year.csv       95% confidence interval per metric and year bin
       data/processed/v2/uncertainty/trend_slopes.csv     yearly change 2022-2026 with 95% CI
       data/processed/v2/uncertainty/sensitivity_2024_26.csv  same metrics pooled 2024-26 under different data choices
       data/processed/v2/uncertainty/ci_by_year_fulltext_exact.csv, trend_slopes_fulltext_exact.csv  the same per year for case D
       figures/v2/u_trends_with_ci.png
Method: paper-level (cluster) bootstrap. One replicate = draw the papers of a year bin with replacement (same number of papers),
keep ALL values of each drawn paper, recompute the statistic. 4,000 replicates; the 2.5th-97.5th percentile of the replicates is the
95% CI. Papers, not single values, are resampled because values from one paper share the same cell, lab and conditions (resampling
values would understate the uncertainty). A share of 0 (bootstrap CI [0, 0]) gets the rule-of-three upper bound 3/n_papers instead. The CI covers sampling uncertainty only - not publication bias or extraction errors
(see the sensitivity table and FINDINGS_UNCERTAINTY_FEASIBILITY.md).
Metric definitions are identical to v2_09 (j at 2.00 +- 0.03 V; V at 1.00 +- 0.05 A cm-2; EU KPI conversion with V_start or 1.9 V);
the script stops if a point estimate differs from trend_summary.csv.
Run: python analysis/v2/v2_19_uncertainty.py   (numpy, pandas, matplotlib; ~20 s)"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.ticker
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
P = lambda *a: os.path.join(ROOT, *a)
OUT = P('data/processed/v2/uncertainty'); os.makedirs(OUT, exist_ok=True)
B, SEED = 4000, 2026
BINS = ['<=2021', '2022', '2023', '2024', '2025', '2026']
SLOPE_YEARS = ['2022', '2023', '2024', '2025', '2026']
def ybin(y): return '<=2021' if int(y) <= 2021 else str(int(y))

def load(tier, name):
    d = pd.read_csv(P(f'data/processed/v2/{tier}/{name}.csv'))
    if 'match_level' in d: d = d[d.match_level != 'numbers_nearby']      # same rule as v2_12 figures
    d['bin'] = d.year.map(ybin); return d
def pct_kh(d):  # degradation as % per 1000 h, as in v2_09
    V = d.V_start.fillna(1.9); return d.rate_uV_h / 1e6 * 1000 / V * 100

def metric_frames(pp, du):
    """metric -> (DataFrame with uid, bin, x, stat function). x is the value the statistic is computed on."""
    j = pp[(pp.V_cell - 2.0).abs() <= 0.03]
    v = pp[(pp.j_A_cm2 - 1.0).abs() <= 0.05]
    dd = du[du.duration_h.notna()]
    r = du[du.rate_uV_h.notna() & (du.rate_uV_h > 0)].copy(); r['kpi'] = (pct_kh(r) <= 0.9).astype(float)
    med = np.median
    return {
        'j_at_2V median (A/cm2)':      (j.assign(x=j.j_A_cm2), med),
        'j_at_2V p90 (A/cm2)':         (j.assign(x=j.j_A_cm2), lambda a: np.percentile(a, 90)),
        'V_at_1A median (V)':          (v.assign(x=v.V_cell), med),
        'V_at_1A best decile (V)':     (v.assign(x=v.V_cell), lambda a: np.percentile(a, 10)),
        'test length median (h)':      (dd.assign(x=dd.duration_h), med),
        'share tests >=1000 h':        (dd.assign(x=(dd.duration_h >= 1000).astype(float)), np.mean),
        'degradation rate median (uV/h)': (r.assign(x=r.rate_uV_h), med),
        'share rates meeting EU 2024 KPI': (r.assign(x=r.kpi), np.mean),
    }

def boot(df, fn, rng, b=B):
    """Point estimate + bootstrap replicates, resampling papers (uid) with replacement."""
    groups = [g.x.to_numpy() for _, g in df.groupby('uid')]
    if not groups: return None, np.array([])
    est = fn(np.concatenate(groups)); n = len(groups); reps = np.empty(b)
    for i in range(b):
        reps[i] = fn(np.concatenate([groups[k] for k in rng.integers(0, n, n)]))
    return est, reps

rng = np.random.default_rng(SEED)
pp1, du1 = load('tier1', 'perf_points'), load('tier1', 'durability')
pp2, du2 = load('tier2', 'perf_points'), load('tier2', 'durability')
M1 = metric_frames(pp1, du1)

# ---- 1) CI per metric and year bin
rows, reps_store = [], {}
for m, (df, fn) in M1.items():
    for k in BINS:
        sub = df[df.bin == k]
        est, reps = boot(sub, fn, rng)
        if est is None: continue
        reps_store[(m, k)] = reps
        lo, hi = np.percentile(reps, 2.5), np.percentile(reps, 97.5)
        if m.startswith('share') and est == 0: hi = min(1.0, 3 / sub.uid.nunique())   # 0 of n: rule-of-three upper bound
        rows.append(dict(metric=m, year_bin=k, estimate=est, ci_low=lo, ci_high=hi,
                         n_values=len(sub), n_papers=sub.uid.nunique(),
                         unstable='yes' if sub.uid.nunique() < 10 or len(sub) < 10 else ''))
ci = pd.DataFrame(rows)

# consistency check against v2_09 trend_summary.csv
ts = pd.read_csv(P('data/processed/v2/tier1/trend_summary.csv'), dtype={'year_bin': str})
chk = {'j_at_2V median (A/cm2)': ('j_at_2V_A_cm2', 'median'), 'j_at_2V p90 (A/cm2)': ('j_at_2V_A_cm2', 'p90'),
       'V_at_1A median (V)': ('V_at_1A_V', 'median'), 'V_at_1A best decile (V)': ('V_at_1A_V', 'p10_best_decile'),
       'test length median (h)': ('durability_test_h', 'median'), 'share tests >=1000 h': ('durability_test_h', 'share_ge_1000h'),
       'degradation rate median (uV/h)': ('degradation_uV_h', 'median'), 'share rates meeting EU 2024 KPI': ('degradation_uV_h', 'share_meets_EU2024')}
bad = []
for _, r in ci.iterrows():
    tm, col = chk[r.metric]; ref = ts[(ts.metric == tm) & (ts.year_bin == r.year_bin)][col]
    if len(ref) and pd.notna(ref.iloc[0]) and abs(ref.iloc[0] - r.estimate) > 0.006 * max(1, abs(ref.iloc[0])):
        bad.append((r.metric, r.year_bin, ref.iloc[0], r.estimate))
if bad: raise SystemExit(f'Point estimates differ from trend_summary.csv (rerun v2_09?): {bad}')
ci.round(4).to_csv(os.path.join(OUT, 'ci_by_year.csv'), index=False)

# ---- 2) yearly change 2022-2026: straight line through the yearly estimates, refitted on every replicate
#      relative metrics (j, test length): fit on log -> % per year; V and shares: absolute change per year
SLOPES = {'j_at_2V median (A/cm2)': 'log', 'V_at_1A median (V)': 'abs', 'test length median (h)': 'log',
          'share tests >=1000 h': 'abs'}
xs = np.arange(len(SLOPE_YEARS)); srows = []
for m, kind in SLOPES.items():
    est = np.array([ci[(ci.metric == m) & (ci.year_bin == y)].estimate.iloc[0] for y in SLOPE_YEARS])
    R = np.vstack([reps_store[(m, y)] for y in SLOPE_YEARS])          # years x B
    f = (lambda a: np.log(a)) if kind == 'log' else (lambda a: a)
    s_est = np.polyfit(xs, f(est), 1)[0]; s_rep = np.polyfit(xs, f(R), 1)[0]
    conv = (lambda s: (np.exp(s) - 1) * 100) if kind == 'log' else (lambda s: s)
    unit = '% per year' if kind == 'log' else ('V per year' if 'V_at' in m else 'share points per year')
    srows.append(dict(metric=m, years='2022-2026 (2026 = Jan-mid Oct)', change=conv(s_est), ci_low=conv(np.percentile(s_rep, 2.5)),
                      ci_high=conv(np.percentile(s_rep, 97.5)), unit=unit, method='OLS line through yearly estimates; papers resampled within each year'))
pd.DataFrame(srows).round(4).to_csv(os.path.join(OUT, 'trend_slopes.csv'), index=False)

# ---- 3) sensitivity: 2024-26 pooled under different data choices (each with its own bootstrap CI)
recent = lambda d: d[d.bin.isin(['2024', '2025', '2026'])]
t2_uids = set(pp2.uid) | set(du2.uid)
KOH = ('KOH', 'NaOH')
cases = [
    ('A abstracts, all selected papers (main result)', recent(pp1), recent(du1)),
    ('B abstracts, only papers that also have a full text', recent(pp1[pp1.uid.isin(t2_uids)]), recent(du1[du1.uid.isin(t2_uids)])),
    ('C full texts, all kept values', recent(pp2), recent(du2)),
    ('D full texts, exact quote matches only', recent(pp2[pp2.match_level == 'exact']), recent(du2[du2.match_level == 'exact'])),
    ('E full texts, KOH/NaOH tests only', recent(pp2[pp2.electrolyte.isin(KOH)]), recent(du2[du2.electrolyte.isin(KOH)])),
    ('F full texts, pure-water tests only', recent(pp2[pp2.electrolyte == 'pure_water']), recent(du2[du2.electrolyte == 'pure_water'])),
    ('G full texts, rates stated by the authors only (no computed rates)', recent(pp2), recent(du2[du2.rate_source != 'computed'])),
]
SENS = ['j_at_2V median (A/cm2)', 'V_at_1A median (V)', 'test length median (h)', 'share tests >=1000 h', 'degradation rate median (uV/h)']
srows = []
for name, pp, du in cases:
    M = metric_frames(pp, du)
    for m in SENS:
        if name.startswith('G') and 'rate' not in m: continue
        df, fn = M[m]; est, reps = boot(df, fn, rng, 2000)
        if est is None: continue
        srows.append(dict(case=name, metric=m, estimate=est, ci_low=np.percentile(reps, 2.5), ci_high=np.percentile(reps, 97.5),
                          n_values=len(df), n_papers=df.uid.nunique(), unstable='yes' if df.uid.nunique() < 10 or len(df) < 10 else ''))
pd.DataFrame(srows).round(4).to_csv(os.path.join(OUT, 'sensitivity_2024_26.csv'), index=False)

# ---- 4) figure: yearly estimates with 95% CI bands
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'; C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 9, 'axes.grid': True, 'grid.color': GRID,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
panels = [('Current density at 2.0 V (A cm⁻²)', ['j_at_2V median (A/cm2)', 'j_at_2V p90 (A/cm2)'], None),
          ('Cell voltage at 1 A cm⁻² (V)', ['V_at_1A median (V)', 'V_at_1A best decile (V)'], None),
          ('Durability test length, median (h)', ['test length median (h)'], 'log'),
          ('Share of durability tests ≥ 1,000 h', ['share tests >=1000 h'], None),
          ('Stated degradation rate, median (µV h⁻¹)', ['degradation rate median (uV/h)'], 'log'),
          ('Share of stated rates meeting EU 2024 KPI', ['share rates meeting EU 2024 KPI'], None)]
fig, axs = plt.subplots(2, 3, figsize=(13, 7.6)); X = np.arange(len(BINS))
for ax, (title, ms, scale) in zip(axs.flat, panels):
    for m, col in zip(ms, [C1, C2]):
        d = ci[ci.metric == m].set_index('year_bin').reindex(BINS)
        lab = 'median' if 'median' in m else ('90th percentile' if 'p90' in m else ('best decile' if 'decile' in m else 'share'))
        ax.fill_between(X, d.ci_low, d.ci_high, color=col, alpha=0.18, lw=0)
        ax.plot(X, d.estimate, color=col, marker='o', ms=4, lw=1.8, label=f'{lab} (band = 95% CI)')
        for xi, (_, r) in zip(X, d.iterrows()):
            if r.unstable == 'yes': ax.plot(xi, r.estimate, 'o', mfc='white', mec=col, ms=6, zorder=4)
    if scale:
        ax.set_yscale(scale); ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(X); ax.set_xticklabels([*BINS[:-1], '2026*']); ax.set_title(title, loc='left', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5, loc='best')
fig.suptitle('AEM lab performance trends with 95% confidence intervals', x=0.01, ha='left', fontsize=13, fontweight='bold')
n_perf, n_dur = pp1.uid.nunique(), du1.uid.nunique()
fig.text(0.01, 0.01, f'Source: abstracts of the 2,465 selected AEM water-electrolysis papers (OpenAlex+Scopus); values from {n_perf:,} papers (performance) and {n_dur:,} papers (durability), extracted by Claude Haiku 5.5 and quote-verified '
         '(data/processed/v2/tier1/). CI = paper-level bootstrap, 4,000 replicates; covers sampling uncertainty only. Open marker = fewer than 10 values or papers '
         '(unstable). *2026 to mid-Oct.', fontsize=7, color=INK2, wrap=True)
fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(P('figures/v2/u_trends_with_ci.png'), dpi=200); plt.close()

# ---- 5) yearly CIs and 2022-26 slopes for sensitivity case D (full texts, exact quote matches only).
#      Own random generator (SEED + 1) so sections 1-4 keep exactly the same draws and outputs.
rng_d = np.random.default_rng(SEED + 1)
MD = metric_frames(pp2[pp2.match_level == 'exact'], du2[du2.match_level == 'exact'])
drows, dreps = [], {}
for m, (df, fn) in MD.items():
    for k in BINS:
        sub = df[df.bin == k]
        est, reps = boot(sub, fn, rng_d)
        if est is None: continue
        dreps[(m, k)] = reps
        lo, hi = np.percentile(reps, 2.5), np.percentile(reps, 97.5)
        if m.startswith('share') and est == 0: hi = min(1.0, 3 / sub.uid.nunique())
        drows.append(dict(metric=m, year_bin=k, estimate=est, ci_low=lo, ci_high=hi, n_values=len(sub), n_papers=sub.uid.nunique(),
                          unstable='yes' if sub.uid.nunique() < 10 or len(sub) < 10 else ''))
cid = pd.DataFrame(drows); cid.round(4).to_csv(os.path.join(OUT, 'ci_by_year_fulltext_exact.csv'), index=False)
srows_d = []
for m, kind in SLOPES.items():
    if not all((m, y) in dreps for y in SLOPE_YEARS): continue
    est = np.array([cid[(cid.metric == m) & (cid.year_bin == y)].estimate.iloc[0] for y in SLOPE_YEARS])
    R = np.vstack([dreps[(m, y)] for y in SLOPE_YEARS])
    f = (lambda a: np.log(a)) if kind == 'log' else (lambda a: a)
    s_est = np.polyfit(xs, f(est), 1)[0]; s_rep = np.polyfit(xs, f(R), 1)[0]
    conv = (lambda s: (np.exp(s) - 1) * 100) if kind == 'log' else (lambda s: s)
    unit = '% per year' if kind == 'log' else ('V per year' if 'V_at' in m else 'share points per year')
    srows_d.append(dict(metric=m, years='2022-2026 (2026 = Jan-mid Oct)', change=conv(s_est), ci_low=conv(np.percentile(s_rep, 2.5)),
                        ci_high=conv(np.percentile(s_rep, 97.5)), unit=unit, method='as trend_slopes.csv; pool = full texts, exact quotes'))
pd.DataFrame(srows_d).round(4).to_csv(os.path.join(OUT, 'trend_slopes_fulltext_exact.csv'), index=False)

pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
print(ci.round(3).to_string(index=False)); print(pd.read_csv(os.path.join(OUT, 'trend_slopes.csv')).to_string(index=False))
print(pd.read_csv(os.path.join(OUT, 'sensitivity_2024_26.csv')).to_string(index=False))
print(cid.round(3).to_string(index=False)); print(pd.read_csv(os.path.join(OUT, 'trend_slopes_fulltext_exact.csv')).to_string(index=False))
