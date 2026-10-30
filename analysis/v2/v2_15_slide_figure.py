"""
v2_15_slide_figure.py - one 16:9 overview figure of the three hurdle metrics for slides.

Panels
  A  current density: median j at 2.0 V (headline performance) vs median current density
     held during durability tests, by publication year; EU 2030 nominal-point KPI for reference
  B  efficiency: median and best-decile cell voltage at 1 A/cm2 by year, with HHV efficiency
  C  durability test length: median and 90th percentile by year (log), vs ~80,000 h stack target
  D  degradation rate by test length (log), vs EU 2024/2030 KPIs, with the total voltage drift
     per bin (rate x duration) to show why long tests report low rates

Inputs  data/processed/v2/tier1/perf_points.csv (abstract values, one record per paper, v2_dedupe.py)
        data/processed/v2/tables/durability_tests.csv (from v2_14; full text where available)
Output  figures/v2/slide_hurdles_overview.png (+ prints the plotted numbers)
Run     python analysis/v2/v2_15_slide_figure.py   (after v2_14)
"""
import csv, os, re, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
csv.field_size_limit(10**9)
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
P = os.path.join(ROOT, 'data', 'processed', 'v2'); FIG = os.path.join(ROOT, 'figures', 'v2')

SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
C1, C2 = '#2a78d6', '#eb6834'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'font.size': 10.5, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'ytick.left': False, 'xtick.major.size': 0, 'axes.axisbelow': True})
BINS = ['<=2021', '2022', '2023', '2024', '2025', '2026']; XL = ['≤2021', '2022', '2023', '2024', '2025', '2026*']
def yb(y): return '<=2021' if int(float(y)) <= 2021 else str(int(float(y)))
def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None

pp = list(csv.DictReader(open(os.path.join(P, 'tier1', 'perf_points.csv'), encoding='utf-8')))
from v2_dedupe import collapse_duplicate_papers, corpus_texts  # one uid per paper (safety net; v2_08 already collapses duplicates)
_texts = corpus_texts(ROOT)
keep, _ = collapse_duplicate_papers((u, _texts[u]['title'], _texts[u]['abstract']) for u in {r['uid'] for r in pp})
pp = [r for r in pp if r['uid'] in keep]
du = [r for r in csv.DictReader(open(os.path.join(P, 'tables', 'durability_tests.csv'), encoding='utf-8-sig'))
      if r['Quote check'] != 'numbers_nearby' and r['Year']]

def by_year(rows, ycol, val, cond=lambda r: True):
    g = defaultdict(list)
    for r in rows:
        v = val(r)
        if v is not None and cond(r): g[yb(r[ycol])].append(v)
    return g
jg = by_year(pp, 'year', lambda r: f(r['j_A_cm2']), lambda r: abs(f(r['V_cell']) - 2.0) <= 0.03)
vg = by_year(pp, 'year', lambda r: f(r['V_cell']), lambda r: abs(f(r['j_A_cm2']) - 1.0) <= 0.05)
hg = by_year(du, 'Year', lambda r: f(r['j held (A/cm²)']))
dg = by_year(du, 'Year', lambda r: f(r['Duration (h)']))

DB = [(0, 100, '<100 h'), (100, 200, '100–200'), (200, 500, '200–500'), (500, 1000, '500–1000'), (1000, 3000, '1000–3000'), (3000, 1e9, '≥3000 h')]
rate_b, drift_b = defaultdict(list), defaultdict(list)
for r in du:
    d, rt = f(r['Duration (h)']), f(r['Rate (µV/h)'])
    if d and rt and rt > 0:
        lab = next(l for lo, hi, l in DB if lo <= d < hi); rate_b[lab].append(rt); drift_b[lab].append(rt * d / 1000)

fig, axs = plt.subplots(2, 2, figsize=(13.33, 7.5))
fig.subplots_adjust(left=0.075, right=0.97, top=0.83, bottom=0.1, hspace=0.62, wspace=0.2)
fig.text(0.075, 0.955, 'AEM electrolysis in the literature: performance has jumped, durability evidence has not',
         fontsize=17, fontweight='bold', color=INK)
fig.text(0.075, 0.915, 'Medians of values reported in AEM water-electrolysis papers, by publication year (*2026 to mid-October)',
         fontsize=10.5, color=INK2)
X = np.arange(len(BINS))

def series(ax, g, fn, color, label, label_dy=0):
    xs = [i for i, k in enumerate(BINS) if len(g[k]) >= 5]; ys = [fn(g[BINS[i]]) for i in xs]
    ax.plot(xs, ys, color=color, lw=2.2, zorder=3, solid_capstyle='round')
    ax.scatter(xs[:-1], ys[:-1], s=42, color=color, zorder=4, edgecolors=SURF, linewidths=1.5)
    ax.scatter(xs[-1:], ys[-1:], s=42, facecolors=SURF, edgecolors=color, linewidths=2, zorder=4)  # partial year
    ax.annotate(label, (xs[-1], ys[-1]), xytext=(10, label_dy), textcoords='offset points', va='center', fontsize=9.5, color=INK2)
    return dict(zip([BINS[i] for i in xs], ys))
def head(ax, title, sub):
    ax.set_title(title, loc='left', fontsize=12.5, fontweight='bold', color=INK, pad=24)
    ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=9.5, color=INK2)
def ref(ax, y, text, xpos=0.0, va='bottom'):
    ax.axhline(y, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.text(xpos, y, ' ' + text, transform=ax.get_yaxis_transform(), fontsize=8.5, color=INK2, va=va, ha='left')
def years(ax):
    ax.set_xticks(X); ax.set_xticklabels(XL); ax.set_xlim(-0.3, len(BINS) - 1 + 1.6)

# A current density
ax = axs[0, 0]
a1 = series(ax, jg, st.median, C1, 'peak performance\n(j at 2.0 V)')
a2 = series(ax, hg, st.median, C2, 'durability tests\n(j held during test)')
ref(ax, 1.5, 'EU 2030 target 1.5 A/cm² (system)', 0.565)
years(ax); ax.set_ylim(0, 3.8); ax.set_ylabel('A/cm²')
head(ax, 'Current density: up ~6×, but durability is shown at ~1 A/cm²', 'Median current density')

# B efficiency
ax = axs[0, 1]
b1 = series(ax, vg, st.median, C1, 'median')
b2 = series(ax, vg, lambda v: float(np.percentile(v, 10)), C2, 'best 10%')
years(ax); ax.set_ylim(1.55, 2.0); ax.set_ylabel('cell voltage (V)')
ax2y = [1.6, 1.7, 1.8, 1.9]
for v in ax2y: ax.text(len(BINS) - 1 + 1.55, v, f'{1.481 / v:.0%}', fontsize=8.5, color=MUTED, ha='right', va='center')
ax.text(len(BINS) - 1 + 1.55, 1.975, 'HHV eff.', fontsize=8.5, color=MUTED, ha='right', va='center')
head(ax, 'Efficiency: flat since 2022 at ~1.80 V (≈82% HHV)', 'Cell voltage at 1 A/cm² (lower is better)')

# C test length
ax = axs[1, 0]
c1 = series(ax, dg, lambda v: float(np.percentile(v, 90)), C2, 'longest 10%', 0)
c2 = series(ax, dg, st.median, C1, 'median test', 0)
ax.set_yscale('log'); ax.set_ylim(40, 150000)
ax.set_yticks([100, 1000, 10000, 100000]); ax.set_yticklabels(['100 h', '1,000 h', '10,000 h', '100,000 h'])
ref(ax, 80000, '~80,000 h commercial stack lifetime (DOE, liquid alkaline)', 0.0, 'top')
years(ax)
head(ax, 'Test length: ~3× longer, still ~200× short of stack lifetime', 'Durability test length (log scale)')

# D rate vs length
ax = axs[1, 1]
labs = [l for _, _, l in DB]; xs = np.arange(len(labs))
ys = [st.median(rate_b[l]) for l in labs]; dr = [st.median(drift_b[l]) for l in labs]
ax.plot(xs, ys, color=C1, lw=2.2, zorder=3)
ax.scatter(xs, ys, s=42, color=C1, zorder=4, edgecolors=SURF, linewidths=1.5)
for x, y, d, l in zip(xs, ys, dr, labs):
    ax.annotate(f'{y:,.0f}', (x, y), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9, color=INK)
    ax.text(x, 1.25, f'Δ{d:.0f} mV', ha='center', fontsize=8.5, color=MUTED)
ax.set_yscale('log'); ax.set_ylim(1, 6000)
ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(['1', '10', '100', '1,000'])
ref(ax, 17.1, 'EU 2024 target ≈17 µV/h', 0.62); ref(ax, 9.5, 'EU 2030 ≈9.5 µV/h', 0.62, 'top')
ax.set_xticks(xs); ax.set_xticklabels([l if len(rate_b[l]) >= 10 else f'{l}\n(n={len(rate_b[l])})' for l in labs]); ax.set_xlim(-0.4, len(labs) - 0.6); ax.set_ylabel('µV/h')
head(ax, 'Degradation: long tests look better, partly by selection', 'Median rate by test length (log)  ·  Δ = median total voltage rise in the test')

fig.text(0.075, 0.018, 'Source: OpenAlex + Scopus AEM water-electrolysis papers; values extracted by Claude Haiku 5.5 from abstracts (A, B) and full texts '
         'where available (C, D), each verified against a verbatim quote. Hollow marker = partial year. Single-cell lab tests dominate; '
         'targets are stack/system level. EU targets: Clean Hydrogen JU SRIA (AEM); rates converted at 1.9 V.', fontsize=7.5, color=MUTED, wrap=True)
out = os.path.join(FIG, 'slide_hurdles_overview.png'); fig.savefig(out, dpi=200); plt.close()
print(out)
for name, d, g in [('j@2V', a1, jg), ('j held', a2, hg), ('V@1A med', b1, vg), ('V@1A p10', b2, vg), ('dur p90', c1, dg), ('dur med', c2, dg)]:
    print(name, {k: round(v, 2) for k, v in d.items()}, 'n', {k: len(g[k]) for k in BINS})
print('rate by length', dict(zip(labs, [round(y) for y in ys])), 'drift mV', dict(zip(labs, [round(d) for d in dr])), 'n', {l: len(rate_b[l]) for l in labs})
