"""v2 step 20 - feasibility rating: how close is each reported test to commercially relevant operation?
Reads  data/processed/v2/tier2/{tests,perf_points,durability}.csv (v2_11, full texts - the only tier where conditions are mostly stated).
Writes data/processed/v2/feasibility/test_scores.csv      one row per test: 6 criteria (met / not_met / unknown), score_low, score_high
       data/processed/v2/feasibility/criteria_summary.csv share met / not met / unknown per criterion
       data/processed/v2/feasibility/score_by_year.csv    score distribution + share of high-feasibility tests per year (95% CI)
       data/processed/v2/feasibility/metrics_by_conditions.csv  performance/durability medians per conditions class (95% CI)
       data/processed/v2/feasibility/top_feasible_tests.csv     tests meeting >= 4 of 6 criteria (score_low), with values and DOI
       figures/v2/f_feasibility.png
Criteria (1 point each; reference = a commercial AEM unit, Enapter FAQ: 1% KOH ~ 0.18 M, rated 55 C, end of life at 2.0 V avg cell):
  C1 scale        stack test, or active area >= 25 cm2 (lab button cells are 1-5 cm2; median stated area in the full texts is 5 cm2)
  C2 electrolyte  pure water, or KOH/NaOH/carbonate <= 0.2 M (dilute, like the 1% KOH of commercial units)
  C3 PGM-free     anode AND cathode stated, no Pt/Ir/Ru/Pd/Rh/Os element symbol (anywhere in a formula, e.g. 'CoRu-MoS2') or
                  element name in either, and no PGM loading > 0 stated
  C4 temperature  45-70 C (commercial rated ~55 C; 80 C lab tests raise j but shorten membrane life; room-temperature tests are a
                  lab convenience, not an operating point)
  C5 duration     a durability test of >= 1,000 h on this test
  C6 load         a durability test held at >= 1.0 A cm-2 (commercial nominal range; EU 2030 KPI 1.5 A cm-2 at system level)
Unknown (condition not stated) is scored as not met in score_low and as met in score_high, so score_high - score_low = missing info.
Tests without any durability result get not_met on C5/C6 (nothing demonstrated). conditions_score = C1-C4 only; it is used to compare
performance across classes, because C5/C6 would make the durability comparison circular (long tests have lower rates by construction).
CIs: paper-level bootstrap as in v2_19 (2,000 replicates); a share of 0 gets the rule-of-three upper bound 3/n_papers. Values with match_level 'numbers_nearby' are excluded (as in v2_12).
Run: python analysis/v2/v2_20_feasibility.py"""
import os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
P = lambda *a: os.path.join(ROOT, *a)
OUT = P('data/processed/v2/feasibility'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026); B = 2000
HIGH = 4                                      # score_low >= HIGH = "high feasibility"
T2 = lambda n: pd.read_csv(P(f'data/processed/v2/tier2/{n}.csv'))
tests, pp, du = T2('tests'), T2('perf_points'), T2('durability')
pp = pp[pp.match_level != 'numbers_nearby']; du = du[du.match_level != 'numbers_nearby']
PGM_SET = {'Pt', 'Ir', 'Ru', 'Pd', 'Rh', 'Os'}
ELEM = re.compile(r'[A-Z][a-z]?(?![a-z])')   # element-symbol tokens: 'FeNiPt@C' -> Fe, Ni, Pt; 'Iron'/'Raney' -> no token

def tri(cond_known, cond_met): return np.where(~cond_known, 'unknown', np.where(cond_met, 'met', 'not_met'))
t = tests.copy()
# C1 scale
t['C1_scale'] = np.where(t.test_level == 'stack', 'met', tri(t.area_cm2.notna(), t.area_cm2 >= 25))
# C2 electrolyte
dil = t.electrolyte.isin(['KOH', 'NaOH', 'K2CO3_or_KHCO3'])
t['C2_electrolyte'] = np.select([t.electrolyte == 'pure_water', dil & t.conc_M.notna() & (t.conc_M <= 0.2), dil & t.conc_M.notna(),
                                 dil | t.electrolyte.isin(['not_stated']), t.electrolyte == 'other'],
                                ['met', 'met', 'not_met', 'unknown', 'not_met'], 'unknown')
# C3 PGM-free
def has_pgm(s): return isinstance(s, str) and bool(PGM_SET & set(ELEM.findall(s)) or re.search(r'platinum|iridium|ruthen|pallad|rhodium|osmium', s, re.I))
named = t.anode.notna() & t.cathode.notna()
pgm_named = t.anode.map(has_pgm) | t.cathode.map(has_pgm) | (t.pgm_mg_cm2.fillna(0) > 0)
t['C3_pgm_free'] = np.where(pgm_named, 'not_met', np.where(named, 'met', 'unknown'))
# C4 temperature
t['C4_temperature'] = tri(t.temperature_C.notna(), t.temperature_C.between(45, 70))
# C5 / C6 from durability results of the same test
du = du.assign(has_h=du.duration_h.notna(), has_j=du.j_held_A_cm2.notna())
dg = du.groupby('test_uid').agg(max_h=('duration_h', 'max'), any_h=('has_h', 'any'),
                                max_j=('j_held_A_cm2', 'max'), any_j=('has_j', 'any'))
t = t.join(dg, on='test_uid'); has_d = t.any_h.notna()
t['C5_duration'] = np.where(~has_d, 'not_met', tri(t.any_h.eq(True), t.max_h >= 1000))
t['C6_load'] = np.where(~has_d, 'not_met', tri(t.any_j.eq(True), t.max_j >= 1.0))
C = ['C1_scale', 'C2_electrolyte', 'C3_pgm_free', 'C4_temperature', 'C5_duration', 'C6_load']
t['score_low'] = (t[C] == 'met').sum(1); t['score_high'] = (t[C] != 'not_met').sum(1)
t['conditions_score'] = (t[C[:4]] == 'met').sum(1)
t['n_unknown'] = (t[C] == 'unknown').sum(1)
keep = ['uid', 'doi', 'year', 'title', 'test_uid', 'test_level', 'electrolyte', 'conc_M', 'temperature_C', 'area_cm2', 'anode', 'cathode',
        'pgm_mg_cm2', 'max_h', 'max_j', *C, 'score_low', 'score_high', 'n_unknown', 'conditions_score']
t[keep].to_csv(os.path.join(OUT, 'test_scores.csv'), index=False)

# criteria summary
cs = pd.DataFrame({c: t[c].value_counts(normalize=True) for c in C}).T.reindex(columns=['met', 'not_met', 'unknown']).fillna(0)
cs['n_tests'] = len(t); cs.round(4).to_csv(os.path.join(OUT, 'criteria_summary.csv'))

def boot(df, col, fn):
    g = [x[col].to_numpy() for _, x in df.groupby('uid')]
    if not g: return (np.nan,) * 3
    est = fn(np.concatenate(g)); n = len(g)
    reps = [fn(np.concatenate([g[k] for k in rng.integers(0, n, n)])) for _ in range(B)]
    return est, np.percentile(reps, 2.5), np.percentile(reps, 97.5)

# score by year (years with full texts; <=2021 merged)
t['bin'] = t.year.map(lambda y: '<=2021' if y <= 2021 else str(int(y)))
rows = []
for k, g in t.groupby('bin'):
    e, lo, hi = boot(g.assign(h=(g.score_low >= HIGH).astype(float)), 'h', np.mean)
    eh, loh, hih = boot(g.assign(h=(g.score_high >= HIGH).astype(float)), 'h', np.mean)
    if e == 0: hi = min(1.0, 3 / g.uid.nunique())       # 0 of n: bootstrap CI is [0, 0]; use rule-of-three upper bound
    rows.append(dict(year_bin=k, n_tests=len(g), n_papers=g.uid.nunique(), mean_score_low=g.score_low.mean(), mean_score_high=g.score_high.mean(),
                     share_high_low=e, ci_low=lo, ci_high=hi, share_high_if_unknown_met=eh, ci_low_u=loh, ci_high_u=hih,
                     **{f'score_low_{s}': (g.score_low == s).sum() for s in range(7)}))
ORDER = ['<=2021', '2022', '2023', '2024', '2025', '2026']
sy = pd.DataFrame(rows).set_index('year_bin').reindex(ORDER).dropna(how='all').reset_index(); sy.round(4).to_csv(os.path.join(OUT, 'score_by_year.csv'), index=False)

# metrics per conditions class (C1-C4 met: 0-1 / 2 / 3-4)
cls = lambda s: '0-1 of 4' if s <= 1 else ('2 of 4' if s == 2 else '3-4 of 4')
t['cls'] = t.conditions_score.map(cls)
m = t.set_index('test_uid')['cls']
ppc = pp.assign(cls=pp.test_uid.map(m)); duc = du.assign(cls=du.test_uid.map(m))
mets = {'j at 2.0 V, median (A/cm2)': (ppc[(ppc.V_cell - 2).abs() <= .03], 'j_A_cm2'),
        'V at 1 A/cm2, median (V)': (ppc[(ppc.j_A_cm2 - 1).abs() <= .05], 'V_cell'),
        'durability test length, median (h)': (duc[duc.duration_h.notna()], 'duration_h'),
        'degradation rate, median (uV/h)': (duc[duc.rate_uV_h.notna() & (duc.rate_uV_h > 0)], 'rate_uV_h')}
rows = []
for name, (df, col) in mets.items():
    for c in ['0-1 of 4', '2 of 4', '3-4 of 4', 'all']:
        sub = df if c == 'all' else df[df.cls == c]
        e, lo, hi = boot(sub, col, np.median)
        rows.append(dict(metric=name, conditions_class=c, estimate=e, ci_low=lo, ci_high=hi, n_values=len(sub), n_papers=sub.uid.nunique()))
mc = pd.DataFrame(rows); mc.round(4).to_csv(os.path.join(OUT, 'metrics_by_conditions.csv'), index=False)

# reading list: high-feasibility tests with their best values
top = t[t.score_low >= HIGH].copy()
bestV = pp[(pp.j_A_cm2 - 1).abs() <= .05].groupby('test_uid').V_cell.min()
bestJ = pp[(pp.V_cell - 2).abs() <= .03].groupby('test_uid').j_A_cm2.max()
rate = du.groupby('test_uid').rate_uV_h.min()
top = top.assign(V_at_1A=top.test_uid.map(bestV), j_at_2V=top.test_uid.map(bestJ), min_rate_uV_h=top.test_uid.map(rate),
                 doi_link='https://doi.org/' + top.doi.astype(str))
top.sort_values(['score_low', 'max_h'], ascending=False)[['score_low', 'score_high', 'year', 'title', 'doi_link', 'test_level', 'electrolyte',
    'conc_M', 'temperature_C', 'area_cm2', 'anode', 'cathode', 'max_h', 'max_j', 'V_at_1A', 'j_at_2V', 'min_rate_uV_h', *C]].to_csv(
    os.path.join(OUT, 'top_feasible_tests.csv'), index=False)

# figure
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'; C1c, C2c, C3c, GR = '#2a78d6', '#eb6834', '#1baf7a', '#c9c7c0'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
fig = plt.figure(figsize=(13, 8)); gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, :2]); lab = ['C1 stack or ≥25 cm²', 'C2 pure water / ≤0.2 M', 'C3 PGM-free', 'C4 45–70 °C', 'C5 test ≥1,000 h', 'C6 held ≥1 A cm⁻²']
y = np.arange(len(C))[::-1]; left = np.zeros(len(C))
for col, c, nm in [('met', C3c, 'met'), ('not_met', C2c, 'not met'), ('unknown', GR, 'not stated')]:
    ax.barh(y, cs[col].values, left=left, color=c, label=nm); left += cs[col].values
ax.set_yticks(y); ax.set_yticklabels(lab); ax.set_xlim(0, 1); ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.13), fontsize=8)
ax.set_title(f'A. Feasibility criteria met ({len(t):,} tests, {t.uid.nunique()} papers, full texts)', loc='left', fontweight='bold')
ax = fig.add_subplot(gs[0, 2]); cnt = t.score_low.value_counts().reindex(range(7), fill_value=0)
ax.bar(cnt.index, cnt.values, color=[C2c if s < HIGH else C3c for s in cnt.index])
for s, v in cnt.items(): ax.text(s, v, f'{v}', ha='center', va='bottom', fontsize=8)
ax.set_xlabel('criteria met (unknown = not met)'); ax.set_ylabel('tests'); ax.set_title(f'B. Score distribution (≥{HIGH} = high)', loc='left', fontweight='bold')
ax = fig.add_subplot(gs[1, 0]); X = np.arange(len(sy))
ax.fill_between(X, sy.ci_low, sy.ci_high, color=C3c, alpha=.2, lw=0); ax.plot(X, sy.share_high_low, 'o-', color=C3c, label='unknown = not met')
ax.plot(X, sy.share_high_if_unknown_met, 'o--', color=GR, label='unknown = met (upper bound)')
ax.set_xticks(X); ax.set_xticklabels([*sy.year_bin[:-1], sy.year_bin.iloc[-1] + '*']); ax.legend(fontsize=7.5); ax.set_ylabel('share of tests with score ≥ 4')
ax.set_title(f'C. Share of tests with score ≥{HIGH} per year', loc='left', fontweight='bold')
for i, (name, unit) in enumerate([('V at 1 A/cm2, median (V)', 'V'), ('degradation rate, median (uV/h)', 'µV h⁻¹')]):
    ax = fig.add_subplot(gs[1, 1 + i]); d = mc[mc.metric == name].set_index('conditions_class').reindex(['0-1 of 4', '2 of 4', '3-4 of 4'])
    ax.errorbar(range(3), d.estimate, yerr=[d.estimate - d.ci_low, d.ci_high - d.estimate], fmt='none', ecolor=C1c, capsize=4)
    for j, (_, r) in enumerate(d.iterrows()):
        few = r.n_values < 10
        ax.plot(j, r.estimate, 'o', ms=7, mfc='white' if few else C1c, mec=C1c)
        ax.text(j + .08, r.estimate, f'n={int(r.n_values)}' + (' (too few)' if few else ''), fontsize=7.5, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels(d.index); ax.set_xlabel('conditions met (C1–C4); median with 95% CI'); ax.set_ylabel(unit)
    if 'rate' in name: ax.set_yscale('log')
    ax.set_title(('D. Cell voltage at 1 A cm⁻² by conditions met' if i == 0 else 'E. Degradation rate by conditions met'), loc='left', fontweight='bold', fontsize=9.5)
fig.suptitle('How commercially relevant are reported AEM tests? (feasibility rating)', x=.01, ha='left', fontsize=13, fontweight='bold')
fig.text(.01, .01, f'Source: full texts ({t.uid.nunique()} papers with extracted tests; Wiley-heavy), conditions and values extracted by Claude Haiku 5.5, quote-verified '
         '(data/processed/v2/tier2/). Criteria C1–C6 = commercial-relevance checks per test (scale, dilute electrolyte, platinum-group-metal (PGM)-free catalysts, '
         '45–70 °C, ≥1,000 h test, ≥1 A cm⁻² held); details in analysis/v2/v2_20_feasibility.py. CI = 95% confidence interval, paper-level bootstrap. '
         'Open marker = fewer than 10 values. *2026 to mid-Oct.',
         fontsize=7, color=INK2, wrap=True)
fig.tight_layout(rect=(0, .06, 1, .95)); fig.savefig(P('figures/v2/f_feasibility.png'), dpi=200); plt.close()
pd.set_option('display.width', 220)
print(cs.round(3)); print(t.score_low.value_counts().sort_index().to_dict(), 'high (low):', (t.score_low >= HIGH).sum(), 'high (upper):', (t.score_high >= HIGH).sum())
print(sy.round(3).to_string(index=False)); print(mc.round(3).to_string(index=False)); print('top tests', len(top), 'papers', top.uid.nunique())
