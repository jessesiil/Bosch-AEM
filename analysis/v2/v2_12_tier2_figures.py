"""v2 step 12 - tier-2 (full-text) figures. Reads data/processed/v2/tier2/*.csv (step 11); values with match_level
'numbers_nearby' are excluded. Writes figures/v2/t2_electrolyte_comparison.png and figures/v2/t2_rate_vs_duration.png.
Electrolyte groups: pure water | KOH or NaOH (≈90% of KOH tests stating a concentration are at 0.9-1.1 M) | other/not stated.
Inputs must be free of cover records and duplicate papers (removed in v2_11 since 2026-10-22); the script stops if they are not."""
import csv, os, statistics as st
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, textwrap
csv.field_size_limit(10**9)
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
T2 = os.path.join(ROOT, 'data/processed/v2/tier2'); FIG = os.path.join(ROOT, 'figures/v2')
L = lambda n: [r for r in csv.DictReader(open(os.path.join(T2, n), encoding='utf-8')) if r.get('match_level', 'exact') != 'numbers_nearby']
pp, du = L('perf_points.csv'), L('durability.csv'); f = float
from v2_dedupe import assert_no_duplicate_papers, corpus_texts
_texts = corpus_texts(ROOT); assert_no_duplicate_papers(pp, 'tier2/perf_points.csv', _texts); assert_no_duplicate_papers(du, 'tier2/durability.csv', _texts)
SURF, INK, INK2, GRID, PT = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#a8a69f'; C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'font.size': 9.5, 'axes.grid': True, 'grid.color': GRID, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})
G = [('pure water', lambda e: e == 'pure_water', C3), ('KOH / NaOH', lambda e: e in ('KOH', 'NaOH'), C1)]
n_papers = len({r['uid'] for r in pp} | {r['uid'] for r in du})
SRC = (f'Source: full texts of {n_papers} AEM water-electrolysis papers (641 obtained of 2,552 selected; ~80% Wiley journals, Elsevier '
       f'almost absent), values extracted by Claude Haiku 5.5 and verified against the text. Lines = medians.')
rng = np.random.default_rng(1)
fig, axes = plt.subplots(1, 3, figsize=(10, 4.4))
panels = [('Cell voltage at 1 A cm⁻² (V)', [f(p['V_cell']) for p in pp if abs(f(p['j_A_cm2']) - 1) <= .05], [p['electrolyte'] for p in pp if abs(f(p['j_A_cm2']) - 1) <= .05], False),
          ('Current density at 2.0 V (A cm⁻²)', [f(p['j_A_cm2']) for p in pp if abs(f(p['V_cell']) - 2) <= .03], [p['electrolyte'] for p in pp if abs(f(p['V_cell']) - 2) <= .03], False),
          ('Degradation rate (µV h⁻¹, log)', [f(d['rate_uV_h']) for d in du if d['rate_uV_h'] and f(d['rate_uV_h']) > 0], [d['electrolyte'] for d in du if d['rate_uV_h'] and f(d['rate_uV_h']) > 0], True)]
for ax, (title, vals, els, log) in zip(axes, panels):
    for i, (name, test, col) in enumerate(G):
        xs = [v for v, e in zip(vals, els) if test(e)]
        ax.scatter(i + rng.uniform(-.2, .2, len(xs)), xs, s=11, color=col, alpha=.35, linewidths=0)
        if xs:
            m = st.median(xs); ax.plot([i - .3, i + .3], [m, m], color=INK, lw=2)
            ax.text(i + .33, m, f'{m:.3g}\nn={len(xs)}', fontsize=8, color=INK2, va='center')
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in G]); ax.set_xlim(-.6, 1.9); ax.set_title(title, fontsize=10, loc='left', color=INK)
    if log: ax.set_yscale('log'); ax.axhline(17.1, color=INK2, lw=1, ls='--'); ax.text(-.55, 19, 'EU KPI 2024', fontsize=7.5, color=INK2)
fig.suptitle('Pure-water vs KOH-supported AEM cells (full-text data)', x=0.01, ha='left', fontsize=12, fontweight='bold')
fig.text(0.01, 0.01, textwrap.fill(SRC, 170), fontsize=7, color=INK2); fig.subplots_adjust(left=.06, right=.97, top=.84, bottom=.16, wspace=.35)
plt.savefig(os.path.join(FIG, 't2_electrolyte_comparison.png'), dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(7.6, 5.2))
for name, test, col in G + [('other / not stated', lambda e: e not in ('pure_water', 'KOH', 'NaOH'), PT)]:
    ds = [d for d in du if d['rate_uV_h'] and f(d['rate_uV_h']) > 0 and d['duration_h'] and test(d['electrolyte'])]
    ax.scatter([f(d['duration_h']) for d in ds], [f(d['rate_uV_h']) for d in ds], s=16, color=col, alpha=.6, linewidths=0, label=f'{name} (n={len(ds)})')
ax.set_xscale('log'); ax.set_yscale('log')
for y, lab in [(17.1, 'EU KPI 2024 (0.9 %/1000 h)'), (9.5, 'EU KPI 2030 (0.5 %/1000 h)')]:
    ax.axhline(y, color=INK2, lw=1, ls='--'); ax.text(ax.get_xlim()[1] * .9 if False else 3000, y * 1.12, lab, fontsize=7.5, color=INK2)
allr = [d for d in du if d['rate_uV_h'] and f(d['rate_uV_h']) > 0 and d['duration_h']]
for lo, hi in [(0, 200), (200, 1000), (1000, 1e6)]:
    r = [f(d['rate_uV_h']) for d in allr if lo <= f(d['duration_h']) < hi]
    if r:
        xm = np.sqrt(max(lo, 10) * min(hi, 20000)); ax.plot([max(lo, 10), min(hi, 20000)], [st.median(r)] * 2, color=C2, lw=2.5)
ax.plot([], [], color=C2, lw=2.5, label='median per test-length band (<200 h, 200–1000 h, ≥1000 h)')
ax.set_xlabel('test duration (h, log)'); ax.set_ylabel('reported degradation (µV h⁻¹, log)')
ax.set_title('Longer tests report lower degradation rates', loc='left', fontsize=12, fontweight='bold', color=INK)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)
fig.text(0.01, 0.01, textwrap.fill(SRC, 140), fontsize=7, color=INK2); fig.subplots_adjust(left=.11, right=.97, top=.92, bottom=.3)
plt.savefig(os.path.join(FIG, 't2_rate_vs_duration.png'), dpi=200); plt.close(); print('ok')
