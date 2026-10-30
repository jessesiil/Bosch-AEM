"""v2 step 9 - tier-1 trends (abstract-level extraction, all selected papers) + figures.
Reads data/processed/v2/tier1/{perf_points,tests,durability}.csv (made by v2_08_clean_tier1.py).
Writes data/processed/v2/tier1/trend_summary.csv (one row per year bin and metric) and figures/v2/t1_*.png.
Year bins: <=2021 merged (few papers), then 2022..2026; 2026 is partial (data up to mid-Oct 2026).
Metrics:
  j_at_2V      current density at a cell voltage of 2.00 +- 0.03 V   (median, 90th percentile, max, n)
  V_at_1A      cell voltage at 1.00 +- 0.05 A cm-2                   (median, 10th percentile = best decile, min, n)
  dur_h        durability test length in hours                     (median, 90th percentile, share >= 1000 h, n)
  rate         stated or computed voltage degradation rate, uV/h   (median, share meeting EU Clean Hydrogen JU AEMEL KPI
               0.9 %/1000 h (2024) and 0.5 %/1000 h (2030), converted with the test's start voltage or 1.9 V)
  qual_only    share of durability results with no voltage or rate at all ("stable for X h")
Reference values: EU Clean Hydrogen JU SRIA AEMEL KPIs (degradation 0.9 -> 0.5 %/1000 h; current density 0.6 -> 1.5 A cm-2
at nominal system operation, 2024 -> 2030); DOE liquid-alkaline stack lifetime target 80,000 h (no DOE AEM table found).
Inputs must be free of cover records and duplicate papers (removed in v2_08 since 2026-10-22); the script stops if they are not."""
import csv, os, collections, statistics as st
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
csv.field_size_limit(10**9)
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
T1 = os.path.join(ROOT, 'data/processed/v2/tier1'); FIG = os.path.join(ROOT, 'figures/v2')
BINS = ['<=2021', '2022', '2023', '2024', '2025', '2026']
def b(y): return '<=2021' if int(y or 0) <= 2021 else str(y)
def pct(xs, p): return float(np.percentile(xs, p)) if xs else None
pp = list(csv.DictReader(open(os.path.join(T1, 'perf_points.csv'), encoding='utf-8')))
du = list(csv.DictReader(open(os.path.join(T1, 'durability.csv'), encoding='utf-8')))
tests = list(csv.DictReader(open(os.path.join(T1, 'tests.csv'), encoding='utf-8')))
from v2_dedupe import assert_no_duplicate_papers, corpus_texts
_texts = corpus_texts(ROOT)
for _rows, _n in [(pp, 'perf_points'), (du, 'durability'), (tests, 'tests')]: assert_no_duplicate_papers(_rows, f'tier1/{_n}.csv', _texts)

# --- colours / style (reference palette, light surface)
SURF, INK, INK2, GRID, PT = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#a8a69f'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'font.size': 10, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False})
X = {k: i for i, k in enumerate(BINS)}
rng = np.random.default_rng(0)
def scatter(ax, rows, key):
    xs = [X[b(r['year'])] + rng.uniform(-0.18, 0.18) for r in rows]; ys = [float(r[key]) for r in rows]
    ax.scatter(xs, ys, s=10, color=PT, alpha=0.45, linewidths=0, label='individual reported values', zorder=1)
def line(ax, vals, color, label, ls='-'):
    xs = [X[k] for k in BINS if vals.get(k) is not None]; ys = [vals[k] for k in BINS if vals.get(k) is not None]
    ax.plot(xs, ys, color=color, lw=2, ls=ls, marker='o', ms=5, label=label, zorder=3)
def finish(ax, title, sub, ylabel, path):
    import textwrap
    fig = ax.figure
    ax.set_xticks(range(len(BINS))); ax.set_xticklabels([*BINS[:-1], '2026*'])
    ax.set_ylabel(ylabel); ax.set_title(title, loc='left', fontsize=12, fontweight='bold', color=INK, pad=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8.5)
    fig.text(0.02, 0.015, textwrap.fill(sub, 135), fontsize=7.2, color=INK2, va='bottom')
    fig.subplots_adjust(left=0.11, right=0.97, top=0.91, bottom=0.27)
    plt.savefig(path, dpi=200); plt.close()
SRC = 'Source: abstracts of {n} AEM water-electrolysis papers (OpenAlex+Scopus), values extracted by Claude Haiku 5.5, quote-verified. *2026 to mid-Oct.'

rows_out = []
def rec(metric, k, **v): rows_out.append(dict(metric=metric, year_bin=k, **{a: (round(x, 4) if isinstance(x, float) else x) for a, x in v.items()}))

# 1) current density at 2.0 V
sel = [p for p in pp if abs(float(p['V_cell']) - 2.0) <= 0.03]
g = collections.defaultdict(list); [g[b(p['year'])].append(float(p['j_A_cm2'])) for p in sel]
med = {k: st.median(g[k]) for k in BINS if len(g[k]) >= 3}; p90 = {k: pct(g[k], 90) for k in BINS if len(g[k]) >= 3}
for k in BINS: rec('j_at_2V_A_cm2', k, n=len(g[k]), median=med.get(k), p90=p90.get(k), max=max(g[k]) if g[k] else None)
fig, ax = plt.subplots(figsize=(7.6, 5.2)); scatter(ax, sel, 'j_A_cm2')
line(ax, med, C1, 'median'); line(ax, p90, C2, 'top 10% (90th percentile)')
ax.axhline(1.5, color=C3, lw=1.2, ls='--'); ax.text(5.35, 1.6, 'EU 2030 KPI 1.5 A cm⁻²\n(system, nominal point)', fontsize=7.5, color=INK2, ha='right')
finish(ax, 'Current density at 2.0 V cell voltage', SRC.format(n=len({p["uid"] for p in sel})) + ' n per bin: ' +
       ', '.join(str(len(g[k])) for k in BINS), 'A cm⁻² at 2.0 V', os.path.join(FIG, 't1_current_density_at_2V.png'))

# 2) cell voltage at 1 A cm-2
sel = [p for p in pp if abs(float(p['j_A_cm2']) - 1.0) <= 0.05]
g = collections.defaultdict(list); [g[b(p['year'])].append(float(p['V_cell'])) for p in sel]
med = {k: st.median(g[k]) for k in BINS if len(g[k]) >= 3}; p10 = {k: pct(g[k], 10) for k in BINS if len(g[k]) >= 3}
for k in BINS: rec('V_at_1A_V', k, n=len(g[k]), median=med.get(k), p10_best_decile=p10.get(k), min=min(g[k]) if g[k] else None,
                   median_eff_HHV=1.481 / med[k] if k in med else None)
fig, ax = plt.subplots(figsize=(7.6, 5.2)); scatter(ax, sel, 'V_cell')
line(ax, med, C1, 'median'); line(ax, p10, C2, 'best 10% (10th percentile)')
finish(ax, 'Cell voltage at 1 A cm⁻² (lower = more efficient)', SRC.format(n=len({p["uid"] for p in sel})) + ' n per bin: ' +
       ', '.join(str(len(g[k])) for k in BINS), 'cell voltage (V) at 1 A cm⁻²', os.path.join(FIG, 't1_voltage_at_1A.png'))

# 3) durability test length
sel = [d for d in du if d['duration_h']]
g = collections.defaultdict(list); [g[b(d['year'])].append(float(d['duration_h'])) for d in sel]
qual = collections.defaultdict(list); [qual[b(d['year'])].append(int(d['qualitative_only'])) for d in du]
med = {k: st.median(g[k]) for k in BINS if g[k]}; p90 = {k: pct(g[k], 90) for k in BINS if g[k]}
for k in BINS: rec('durability_test_h', k, n=len(g[k]), median=med.get(k), p90=p90.get(k), max=max(g[k]) if g[k] else None,
                   share_ge_1000h=sum(x >= 1000 for x in g[k]) / len(g[k]) if g[k] else None,
                   share_qualitative_only=sum(qual[k]) / len(qual[k]) if qual[k] else None)
fig, ax = plt.subplots(figsize=(7.6, 5.2)); scatter(ax, sel, 'duration_h'); ax.set_yscale('log')
line(ax, med, C1, 'median'); line(ax, p90, C2, 'longest 10% (90th percentile)')
ax.axhline(80000, color=C3, lw=1.2, ls='--'); ax.text(0, 52000, 'stack lifetime target ~80,000 h (DOE, liquid alkaline)', fontsize=7.5, color=INK2)
ax.set_ylim(1, 2e5)
finish(ax, 'Length of reported durability tests', SRC.format(n=len({d["uid"] for d in sel})) + ' n per bin: ' +
       ', '.join(str(len(g[k])) for k in BINS), 'test duration (h, log scale)', os.path.join(FIG, 't1_durability_test_length.png'))

# 4) degradation rates vs EU KPI
sel = [d for d in du if d['rate_uV_h'] and float(d['rate_uV_h']) > 0]
def pct_kh(d): V = float(d['V_start']) if d['V_start'] else 1.9; return float(d['rate_uV_h']) / 1e6 * 1000 / V * 100
g = collections.defaultdict(list); [g[b(d['year'])].append(d) for d in sel]
for k in BINS:
    xs = [float(d['rate_uV_h']) for d in g[k]]
    rec('degradation_uV_h', k, n=len(xs), median=st.median(xs) if xs else None,
        share_meets_EU2024=sum(pct_kh(d) <= 0.9 for d in g[k]) / len(xs) if xs else None,
        share_meets_EU2030=sum(pct_kh(d) <= 0.5 for d in g[k]) / len(xs) if xs else None,
        median_test_h_of_these=st.median([float(d['duration_h']) for d in g[k] if d['duration_h']]) if any(d['duration_h'] for d in g[k]) else None)
fig, ax = plt.subplots(figsize=(7.6, 5.2)); scatter(ax, sel, 'rate_uV_h'); ax.set_yscale('log')
med = {k: st.median([float(d['rate_uV_h']) for d in g[k]]) for k in BINS if len(g[k]) >= 3}; line(ax, med, C1, 'median')
for y, lab in [(17.1, 'EU KPI 2024: 0.9 %/1000 h (≈17 µV h⁻¹ at 1.9 V)'), (9.5, 'EU KPI 2030: 0.5 %/1000 h (≈9.5 µV h⁻¹)')]:
    ax.axhline(y, color=C3, lw=1.2, ls='--'); ax.text(5.35, y * 1.12, lab, fontsize=7.5, color=INK2, ha='right')
finish(ax, 'Reported voltage degradation rates', SRC.format(n=len({d["uid"] for d in sel})) + ' Only results that state a rate or start/end voltage. n per bin: ' +
       ', '.join(str(len(g[k])) for k in BINS), 'degradation (µV h⁻¹, log scale)', os.path.join(FIG, 't1_degradation_rate.png'))

# 5) conditions reporting
g = collections.defaultdict(collections.Counter)
for t in tests: k = b(t['year']); g[k]['n'] += 1; g[k][t['electrolyte']] += 1; g[k]['stack'] += t['test_level'] == 'stack'
for k in BINS: rec('test_conditions', k, n=g[k]['n'], share_pure_water=g[k]['pure_water'] / g[k]['n'], share_KOH=g[k]['KOH'] / g[k]['n'],
                   share_not_stated=g[k]['not_stated'] / g[k]['n'], stack_tests=g[k]['stack'])
keys = sorted({k for r in rows_out for k in r})
with open(os.path.join(T1, 'trend_summary.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['metric', 'year_bin'] + [k for k in keys if k not in ('metric', 'year_bin')]); w.writeheader(); w.writerows(rows_out)
for r in rows_out: print({k: v for k, v in r.items() if v is not None})
