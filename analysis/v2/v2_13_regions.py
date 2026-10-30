"""v2 step 13 - regional view: where the reported AEM performance and durability come from.
Region of a paper = countries of its authors' institutions (OpenAlex country codes; for Scopus-only papers the last element of
each affiliation in the Scopus export). Whole counting: a China-Germany paper counts for both China and Europe.
Regions: China (CN + HK), Europe (EU-27 + UK, CH, NO, IS), United States, South Korea (2nd-largest producer of AEM papers,
added for context); everything else = other (not plotted).
Reads data/processed/v2/tier1/{perf_points,durability}.csv (abstract-level, all selected papers).
Writes data/processed/v2/tier1/paper_regions.csv, data/processed/v2/tier1/region_trends.csv,
figures/v2/t1_region_current_density.png, figures/v2/t1_region_durability.png.
Inputs must be free of cover records and duplicate papers (removed in v2_08 since 2026-10-22); the script stops if they are not.
2026-10-22 (T-058): Scopus country-name map completed for all EU-27 members (Slovakia etc. were mapped to '?');
paper_regions.csv rows sorted by uid (was set order, different on every run)."""
import csv, json, os, re, collections, statistics as st, textwrap
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
csv.field_size_limit(10**9)
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
T1 = os.path.join(ROOT, 'data/processed/v2/tier1'); FIG = os.path.join(ROOT, 'figures/v2')
EU = set('AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE GB UK CH NO IS'.split())
NAME = {'china': 'CN', 'hong kong': 'HK', 'united states': 'US', 'usa': 'US', 'south korea': 'KR', 'korea': 'KR', 'republic of korea': 'KR',
        'germany': 'DE', 'united kingdom': 'GB', 'france': 'FR', 'italy': 'IT', 'spain': 'ES', 'netherlands': 'NL', 'switzerland': 'CH',
        'denmark': 'DK', 'norway': 'NO', 'sweden': 'SE', 'finland': 'FI', 'belgium': 'BE', 'austria': 'AT', 'poland': 'PL', 'czech republic': 'CZ',
        'czechia': 'CZ', 'portugal': 'PT', 'ireland': 'IE', 'greece': 'GR', 'slovenia': 'SI', 'hungary': 'HU', 'romania': 'RO', 'estonia': 'EE',
        'japan': 'JP', 'india': 'IN', 'taiwan': 'TW', 'australia': 'AU', 'canada': 'CA', 'singapore': 'SG',
        'slovakia': 'SK', 'slovak republic': 'SK', 'croatia': 'HR', 'bulgaria': 'BG', 'lithuania': 'LT', 'latvia': 'LV',
        'luxembourg': 'LU', 'cyprus': 'CY', 'malta': 'MT', 'iceland': 'IS'}
def regions(codes):
    out = set()
    for k in codes:
        if k in ('CN', 'HK'): out.add('China')
        elif k == 'US': out.add('United States')
        elif k == 'KR': out.add('South Korea')
        elif k in EU: out.add('Europe')
    return out

corpus = {r['uid']: r for r in csv.DictReader(open(os.path.join(ROOT, 'data/processed/v2/corpus.csv'), encoding='utf-8'))}
aff = {r['EID']: r['Affiliations'] for r in csv.DictReader(open(os.path.join(ROOT, 'data/raw/scopus/scopus_export_2026-10-14.csv'), encoding='utf-8-sig'))}
pp = list(csv.DictReader(open(os.path.join(T1, 'perf_points.csv'), encoding='utf-8')))
du = list(csv.DictReader(open(os.path.join(T1, 'durability.csv'), encoding='utf-8')))
from v2_dedupe import assert_no_duplicate_papers, corpus_texts
_texts = corpus_texts(ROOT); assert_no_duplicate_papers(pp, 'tier1/perf_points.csv', _texts); assert_no_duplicate_papers(du, 'tier1/durability.csv', _texts)
reg, src = {}, collections.Counter()
for u in sorted({r['uid'] for r in pp} | {r['uid'] for r in du}):
    c = corpus[u]; codes = [x.strip() for x in c['countries'].split(';') if x.strip()]
    if codes: src['openalex'] += 1
    elif c['scopus_eid'] and aff.get(c['scopus_eid']):
        codes = [NAME.get(a.split(',')[-1].strip().lower(), '?') for a in aff[c['scopus_eid']].split(';')]; src['scopus_affiliations'] += 1
    else: src['none'] += 1
    reg[u] = regions(codes)
with open(os.path.join(T1, 'paper_regions.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['uid', 'regions']); [w.writerow([u, '; '.join(sorted(r))]) for u, r in reg.items()]
print('country source:', dict(src), '| papers per region:', collections.Counter(x for r in reg.values() for x in r))

BINS = ['<=2022', '2023', '2024', '2025', '2026']
def b(y): return '<=2022' if int(y or 0) <= 2022 else str(y)
REG = [('China', '#2a78d6'), ('Europe', '#eb6834'), ('United States', '#1baf7a'), ('South Korea', '#6250d6')]
MIN_N = 8
N_SEL = json.load(open(os.path.join(T1, 'cleaning_log.json')))['papers']   # distinct selected papers (v2_08)
rows = []
def series(values, agg=st.median):   # values: list of (uid, year, value)
    out = {}
    for name, _ in REG:
        g = collections.defaultdict(list)
        for u, y, v in values:
            if name in reg.get(u, ()): g[b(y)].append(v)
        out[name] = {k: (agg(g[k]), len(g[k])) for k in BINS if g[k]}
    return out
j2 = series([(p['uid'], p['year'], float(p['j_A_cm2'])) for p in pp if abs(float(p['V_cell']) - 2.0) <= 0.03])
dh = series([(d['uid'], d['year'], float(d['duration_h'])) for d in du if d['duration_h']])
dk = series([(d['uid'], d['year'], float(d['duration_h'])) for d in du if d['duration_h']], agg=lambda xs: 100 * sum(x >= 1000 for x in xs) / len(xs))
for metric, s in [('j_at_2V_median_A_cm2', j2), ('durability_test_h_median', dh), ('durability_share_ge_1000h_pct', dk)]:
    for name, _ in REG:
        for k in BINS:
            if k in s[name]: rows.append(dict(metric=metric, region=name, year_bin=k, value=round(s[name][k][0], 3), n=s[name][k][1]))
with open(os.path.join(T1, 'region_trends.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['metric', 'region', 'year_bin', 'value', 'n']); w.writeheader(); w.writerows(rows)
for r in rows: print(r)

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'font.size': 10, 'axes.grid': True, 'grid.color': GRID, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})
def plot(s, title, ylabel, fname, log=False):
    fig, ax = plt.subplots(figsize=(7.6, 5.2)); X = {k: i for i, k in enumerate(BINS)}
    for name, col in REG:
        pts = [(X[k], m, n) for k, (m, n) in s[name].items()]
        ok = [(x, m, n) for x, m, n in pts if n >= MIN_N]
        ax.plot([], [], color=col, lw=2, label=name)
        seg = []   # draw lines only between consecutive year bins that both have enough values
        for x, m, n in sorted(pts) + [(99, None, 0)]:
            if n >= MIN_N and (not seg or x == seg[-1][0] + 1): seg.append((x, m)); continue
            if len(seg) > 1: ax.plot([a for a, _ in seg], [c for _, c in seg], color=col, lw=2, zorder=2)
            seg = [(x, m)] if n >= MIN_N else []
        if ok:
            ax.scatter([x for x, _, _ in ok], [m for _, m, _ in ok], s=[18 + 1.6 * n for _, _, n in ok], color=col, edgecolors='#fcfcfb', linewidths=1.5, zorder=3)
        thin = [(x, m) for x, m, n in pts if n < MIN_N]
        if thin: ax.scatter([x for x, _ in thin], [m for _, m in thin], s=30, facecolors='none', edgecolors=col, linewidths=1.5, zorder=3)
    if log: ax.set_yscale('log')
    ax.set_xticks(range(len(BINS))); ax.set_xticklabels([*BINS[:-1], '2026*']); ax.set_xlim(-0.3, len(BINS) - 0.7)
    ax.set_ylabel(ylabel); ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=10)
    ax.scatter([], [], s=36, facecolors='none', edgecolors=INK2, label=f'< {MIN_N} values')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=5, fontsize=8.5)
    fig.text(0.02, 0.015, textwrap.fill(
        f'Marker size = number of reported values (exact n in data/processed/v2/tier1/region_trends.csv). Hollow markers = fewer than 8 values, not joined. Source: abstracts of the {N_SEL:,} selected AEM '
        'water-electrolysis papers (OpenAlex+Scopus), values extracted by Claude Haiku 5.5 and quote-verified. Region = authors\' institutions, '
        'whole counting (a joint China-Germany paper counts for both). Europe = EU-27 + UK, CH, NO, IS; China includes Hong Kong. *2026 to mid-Oct.', 135),
        fontsize=7.2, color=INK2, va='bottom')
    fig.subplots_adjust(left=0.11, right=0.97, top=0.91, bottom=0.27); plt.savefig(os.path.join(FIG, fname), dpi=200); plt.close()
plot(j2, 'Current density at 2.0 V by region (median)', 'A cm⁻² at 2.0 V', 't1_region_current_density.png')
plot(dk, 'Share of durability tests lasting 1,000 h or more, by region', '% of reported durability tests', 't1_region_durability.png')
