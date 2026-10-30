"""Publication trend series + S-curve fits (OpenAlex). Run after classify_aem.py and classify_gh.py.
Outputs: data/processed/pub_series.csv, pub_scurves.json, pub_sample_estimates.csv, figures/*.png"""
import csv, json, os
import numpy as np
import pandas as pd
from common import load_openalex, PROC, LABELS, FIG
from scurves import fit_series, fisher_pry, scenario_fits, yoy_growth
import plotting as P

os.makedirs(FIG, exist_ok=True)
recs, metas = load_openalex()
counts_meta = [m for m in metas if 'counts_by_year' in m][0]['counts_by_year']
aem = pd.read_csv(os.path.join(PROC, 'aem_classified.csv'))
gh = pd.read_csv(os.path.join(PROC, 'gh_classified.csv'))
YEARS = list(range(2000, 2027))

def by_year(df, mask=None):
    d = df if mask is None else df[mask]
    return d.groupby('year').size().reindex(YEARS, fill_value=0)

S = pd.DataFrame(index=YEARS)
S['AEM_raw_query'] = by_year(aem)
S['AEM_keyword_strict'] = by_year(aem, aem.in_AEM_strict == 1)
S['AEM_keyword_rule_core'] = by_year(aem, aem.rule_core == 1)
S['AEM_ML_core'] = by_year(aem, aem.ml_core == 1)
S['AEM_ML_core_plus_peripheral'] = by_year(aem, aem.ml_relevant == 1)
S['ELY_raw_union'] = by_year(gh)
S['ELY_ML_electrolysis'] = by_year(gh, gh.ml_elec == 1)
S['GH_ML_green_h2_broad'] = by_year(gh, gh.ml_gh == 1)
E = gh[gh.ml_elec == 1]
for t in ['PEM', 'ALK', 'SOEC']:
    S[f'ELY_tag_{t}'] = by_year(E, E[f'tag_{t}'] == 1)
S['ELY_tag_AEM_regex'] = by_year(E, E['tag_AEM_regex'] == 1)
# AEM share of electrolysis research: AEM ML-core papers over (electrolysis papers U AEM core papers)
aem_core_ids = set(aem.loc[aem.ml_core == 1, 'id'])
ely_ids_by_year = E.groupby('year')['id'].apply(set)
aem_ids_by_year = aem[aem.ml_core == 1].groupby('year')['id'].apply(set)
S['ELY_union_AEMcore'] = [len(ely_ids_by_year.get(y, set()) | aem_ids_by_year.get(y, set())) for y in YEARS]
S['AEM_share_of_electrolysis'] = (S['AEM_ML_core'] / S['ELY_union_AEMcore']).round(4)
for t in ['PEM', 'ALK', 'SOEC']:
    S[f'{t}_share_of_electrolysis'] = (S[f'ELY_tag_{t}'] / S['ELY_union_AEMcore']).round(4)
allw = counts_meta['ALL_WORKS']['by_year']
S['OpenAlex_all_articles'] = [int(allw.get(str(y), 0)) for y in YEARS]
S['AEM_core_per_million_articles'] = (S['AEM_ML_core'] / S['OpenAlex_all_articles'] * 1e6).round(2)
S['ELY_per_million_articles'] = (S['ELY_ML_electrolysis'] / S['OpenAlex_all_articles'] * 1e6).round(2)
# industry involvement
for name, d, m in [('AEM_core', aem, aem.ml_core == 1), ('ELY', gh, gh.ml_elec == 1)]:
    dd = d[m]
    S[f'{name}_share_with_company_author'] = (dd[dd.n_company_affil > 0].groupby('year').size().reindex(YEARS, fill_value=0) / by_year(dd).replace(0, np.nan)).round(3)
S.index.name = 'year'
S.to_csv(os.path.join(PROC, 'pub_series.csv'))

# ---- country mix of AEM core papers (whole counting: a paper counts once for each country group present) ----
def cgroup(cc):
    EU = set('AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE'.split())
    s = set(str(cc).split(';')) if isinstance(cc, str) else set()
    g = set()
    for c in s:
        if c == 'CN': g.add('China')
        elif c == 'KR': g.add('South Korea')
        elif c in EU or c in ('GB', 'CH', 'NO'): g.add('Europe (EU+UK/CH/NO)')
        elif c == 'US': g.add('USA')
        elif c == 'JP': g.add('Japan')
        elif c == 'IN': g.add('India')
        elif c: g.add('Other')
    return g
rows = []
core = aem[aem.ml_core == 1]
for p0, p1 in [(2000, 2015), (2016, 2020), (2021, 2023), (2024, 2025)]:
    d = core[(core.year >= p0) & (core.year <= p1)]
    tot = len(d); cnt = {}
    for cc in d.countries:
        for g in cgroup(cc): cnt[g] = cnt.get(g, 0) + 1
    rows.append({'period': f'{p0}-{p1}', 'n_papers': tot, **{k: round(v / tot, 3) for k, v in sorted(cnt.items())}})
pd.DataFrame(rows).to_csv(os.path.join(PROC, 'aem_country_mix.csv'), index=False)

# ---- sample-based (design-unbiased) estimates per stratum, as an independent check on the classifier counts ----
def wilson(k, n, z=1.645):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
est = []
for fname, pos, clf_df, clf_col in [('aem_labels.csv', {'C'}, aem, 'ml_core'), ('gh_labels.csv', {'E'}, gh, 'ml_elec')]:
    L = pd.read_csv(os.path.join(LABELS, fname))
    for s, g in L.groupby('stratum'):
        a, b = map(int, s.split('-')); N = int(g.stratum_size.iloc[0]); k = int(g.label.isin(pos).sum()); n = len(g)
        lo, hi = wilson(k, n)
        clf_n = int(clf_df[(clf_df.year >= a) & (clf_df.year <= b)][clf_col].sum())
        est.append({'corpus': fname.split('_')[0], 'stratum': s, 'retrieved_N': N, 'sample_n': n, 'precision_sample': round(k / n, 3),
                    'est_relevant': round(N * k / n), 'est_low_90': round(N * lo), 'est_high_90': round(N * hi), 'classifier_count': clf_n})
EST = pd.DataFrame(est); EST.to_csv(os.path.join(PROC, 'pub_sample_estimates.csv'), index=False)

# ---- S-curve fits ----
fits = {}
for col, fy in [('AEM_ML_core', 2005), ('AEM_ML_core_plus_peripheral', 2005), ('AEM_keyword_strict', 2005), ('AEM_raw_query', 2005),
                ('ELY_ML_electrolysis', 2000), ('ELY_raw_union', 2000), ('GH_ML_green_h2_broad', 2000),
                ('ELY_tag_PEM', 2000), ('ELY_tag_ALK', 2000), ('ELY_tag_SOEC', 2000)]:
    fits[col] = fit_series(col, YEARS, S[col].values, first_year=fy)
    # sensitivity: does the picture change if the fit ends one year earlier?
    fits[col + '__to2024'] = fit_series(col, YEARS, S[col].values, first_year=fy, last_full_year=2024, n_boot=300)
for col, fy in [('AEM_ML_core', 2005), ('ELY_ML_electrolysis', 2000)]:
    fits[col + '__scenarios'] = scenario_fits(YEARS, S[col].values, fy)
    fits[col + '__yoy'] = yoy_growth(YEARS, S[col].values)
fits['FisherPry_AEM_share'] = fisher_pry(YEARS, S['AEM_share_of_electrolysis'].values, first_year=2010)
fits['FisherPry_AEM_share_from2016'] = fisher_pry(YEARS, S['AEM_share_of_electrolysis'].values, first_year=2016)
json.dump(fits, open(os.path.join(PROC, 'pub_scurves.json'), 'w'), indent=1)

# ---- figures ----
P.fig_filter_comparison(S, os.path.join(FIG, 'pub_aem_filter_comparison.png'))
P.fig_scurve(S, fits, 'AEM_ML_core', 'AEM water electrolysis publications (ML-filtered core)', os.path.join(FIG, 'pub_aem_scurve.png'))
P.fig_scurve(S, fits, 'ELY_ML_electrolysis', 'Water-electrolysis (green H2 production) publications (ML-filtered)', os.path.join(FIG, 'pub_electrolysis_scurve.png'))
P.fig_tech_shares(S, os.path.join(FIG, 'pub_tech_shares.png'))
P.fig_fisher_pry(S, fits['FisherPry_AEM_share_from2016'], os.path.join(FIG, 'pub_aem_fisher_pry.png'))
P.fig_indexed_growth(S, os.path.join(FIG, 'pub_indexed_growth.png'))
P.fig_scenarios(S, fits, os.path.join(FIG, 'pub_scurve_scenarios.png'))

def brief(f):
    o = {'C_end': f['cumulative_at_end']}
    for m in ['logistic', 'gompertz']:
        if m in f and 'K' in f[m]:
            o[m] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f[m].items() if k in ('K', 't0_inflection', 'share_of_K_reached', 'K_at_upper_bound', 'boot_share_K_at_upper_bound', 'aic_annual', 'boot_K_5_50_95', 'boot_t0_5_50_95')}
    o['exp'] = f.get('exponential')
    return o
print(S[['AEM_raw_query', 'AEM_keyword_strict', 'AEM_ML_core', 'AEM_ML_core_plus_peripheral', 'ELY_raw_union', 'ELY_ML_electrolysis', 'AEM_share_of_electrolysis', 'PEM_share_of_electrolysis', 'ALK_share_of_electrolysis', 'SOEC_share_of_electrolysis', 'AEM_core_share_with_company_author']].loc[2005:].to_string())
print(EST.to_string())
for k in ['AEM_ML_core', 'AEM_ML_core__to2024', 'ELY_ML_electrolysis', 'ELY_ML_electrolysis__to2024', 'ELY_tag_PEM', 'ELY_tag_SOEC', 'ELY_tag_ALK']:
    print(k, json.dumps(brief(fits[k]), default=str))
for k in ['AEM_ML_core__scenarios','ELY_ML_electrolysis__scenarios']:
    print(k); [print('  ', {a: (round(b,3) if isinstance(b,float) else b) for a,b in d.items()}) for d in fits[k]]
print('yoy AEM', {k: round(v,2) for k,v in fits['AEM_ML_core__yoy'].items() if v is not None})
print('yoy ELY', {k: round(v,2) for k,v in fits['ELY_ML_electrolysis__yoy'].items() if v is not None})
print('FP', fits['FisherPry_AEM_share'], fits['FisherPry_AEM_share_from2016'])
