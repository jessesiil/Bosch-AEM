"""EU (CORDIS) funding trend analysis for water electrolysis / AEM / wider green hydrogen.

Inputs : data/processed/cordis_h2_candidates.csv (from cordis_build.py)
         data/processed/cordis_h2_candidate_orgs.csv
         data/labels/cordis_labels.csv (manual A/a/E/V/I label for every candidate project)
Outputs: data/processed/cordis_labelled_projects.csv, cordis_funding_by_year.csv, cordis_summary.json

Money is the EU contribution (ecMaxContribution, nominal EUR). Two time attributions are reported:
  * by start year (lumpy: a 5-year project is booked in its start year), and
  * pro-rata (contribution spread evenly over the project's months) = EU money "flowing" per calendar year.
Horizon Europe (2021-2027) is still running, so the most recent start years are incomplete (calls still being signed).
"""
import json, os
import numpy as np
import pandas as pd
from common import PROC, LABELS

c = pd.read_csv(os.path.join(PROC, 'cordis_h2_candidates.csv'), dtype={'id': str})
lab = pd.read_csv(os.path.join(LABELS, 'cordis_labels.csv'), dtype={'id': str})
c = c.merge(lab[['id', 'label']], on='id', how='left')
assert c['label'].notna().all()

# Category sets used in the report
c['is_AEM_core'] = c['label'].eq('A')
c['is_AEM_any'] = c['label'].isin(['A', 'a'])
c['is_ELY'] = c['label'].isin(['A', 'a', 'E'])          # all water-electrolysis technology projects
c['is_GH'] = c['label'].isin(['A', 'a', 'E', 'V'])       # electrolysis + other green-H2 value chain

def scheme_class(s):
    s = str(s).upper()
    if 'ERC' in s: return 'ERC (frontier research)'
    if 'MSCA' in s or s.startswith('MC-'): return 'MSCA (fellowships/training)'
    if 'EIC' in s or 'SME' in s: return 'EIC / SME instrument (start-ups)'
    if s.endswith('-IA') or s == 'IA' or 'JU-IA' in s or s.startswith('CP-') and 'IP' in s: return 'Innovation Action (demo / close to market)'
    if 'CSA' in s or s == 'CSA': return 'CSA (coordination)'
    return 'RIA / collaborative research'
c['scheme_class'] = c['fundingScheme'].map(scheme_class)

for col in ['ec_PRC', 'ec_HES', 'ec_REC', 'ec_PUB', 'ec_OTH']:
    if col not in c: c[col] = 0.0
    c[col] = c[col].fillna(0.0)
c['ec_sum_orgs'] = c[['ec_PRC', 'ec_HES', 'ec_REC', 'ec_PUB', 'ec_OTH']].sum(axis=1)
c['private_share'] = np.where(c['ec_sum_orgs'] > 0, c['ec_PRC'] / c['ec_sum_orgs'], np.nan)
c['start'] = pd.to_datetime(c['startDate'], errors='coerce'); c['end'] = pd.to_datetime(c['endDate'], errors='coerce')
c['ec'] = c['ecMaxContribution'].fillna(0.0)
c.to_csv(os.path.join(PROC, 'cordis_labelled_projects.csv'), index=False)

def prorata(df):
    """Spread each project's EU contribution evenly over its active months; return EUR per calendar year."""
    out = {}
    for _, r in df.iterrows():
        if pd.isna(r['start']) or pd.isna(r['end']) or r['end'] <= r['start']:
            out[r['start'].year if not pd.isna(r['start']) else None] = out.get(r['start'].year if not pd.isna(r['start']) else None, 0) + r['ec']; continue
        months = pd.period_range(r['start'], r['end'], freq='M')
        per = r['ec'] / len(months)
        for p in months: out[p.year] = out.get(p.year, 0.0) + per
    return pd.Series(out).sort_index()

years = list(range(2007, 2027))
rows = []
groups = {'AEM_core': c[c.is_AEM_core], 'AEM_any': c[c.is_AEM_any], 'Electrolysis_all': c[c.is_ELY], 'GreenH2_all': c[c.is_GH], 'Electrolysis_non_AEM': c[c.is_ELY & ~c.is_AEM_any]}
pr = {k: prorata(g) for k, g in groups.items()}
for y in years:
    row = {'year': y}
    for k, g in groups.items():
        gy = g[g.start_year == y]
        row[f'{k}_n_projects_started'] = len(gy)
        row[f'{k}_EC_by_start_MEUR'] = round(gy['ec'].sum() / 1e6, 3)
        row[f'{k}_EC_prorata_MEUR'] = round(float(pr[k].get(y, 0.0)) / 1e6, 3)
    rows.append(row)
F = pd.DataFrame(rows)
F['AEM_share_of_electrolysis_EC_by_start'] = (F['AEM_any_EC_by_start_MEUR'] / F['Electrolysis_all_EC_by_start_MEUR']).round(3)
F['AEM_share_of_electrolysis_EC_prorata'] = (F['AEM_any_EC_prorata_MEUR'] / F['Electrolysis_all_EC_prorata_MEUR']).round(3)
F.to_csv(os.path.join(PROC, 'cordis_funding_by_year.csv'), index=False)

# Programme-period view (removes the framework-programme step artefacts)
def period(y):
    if y <= 2013: return 'FP7 (2007-2013)'
    if y <= 2020: return 'H2020 (2014-2020)'
    return 'Horizon Europe (2021-2026*)'
per_rows = []
for name, g in groups.items():
    for p, gp in g.groupby(g.start_year.map(lambda y: period(int(y)) if not pd.isna(y) else 'unknown')):
        nyears = {'FP7 (2007-2013)': 7, 'H2020 (2014-2020)': 7, 'Horizon Europe (2021-2026*)': 6}.get(p, np.nan)
        per_rows.append({'group': name, 'period': p, 'n_projects': len(gp), 'EC_MEUR': round(gp['ec'].sum() / 1e6, 2),
                         'EC_MEUR_per_year': round(gp['ec'].sum() / 1e6 / nyears, 2), 'median_project_EC_MEUR': round(gp['ec'].median() / 1e6, 2),
                         'private_for_profit_share_of_EC': round(gp['ec_PRC'].sum() / max(gp['ec_sum_orgs'].sum(), 1), 3)})
P = pd.DataFrame(per_rows)
P.to_csv(os.path.join(PROC, 'cordis_by_programme_period.csv'), index=False)

# Instrument mix for electrolysis projects by period (shift from research to innovation/demo actions = maturation signal)
mix = c[c.is_ELY].assign(period=c.start_year.map(lambda y: period(int(y)))).pivot_table(index='period', columns='scheme_class', values='ec', aggfunc='sum', fill_value=0) / 1e6
mix_share = mix.div(mix.sum(axis=1), axis=0).round(3)
mix_share.to_csv(os.path.join(PROC, 'cordis_electrolysis_instrument_mix.csv'))
mix_aem = c[c.is_AEM_any].assign(period=c.start_year.map(lambda y: period(int(y)))).pivot_table(index='period', columns='scheme_class', values='ec', aggfunc='sum', fill_value=0) / 1e6

# CAGR of pro-rata electrolysis funding 2014->2025 (both full calendar years inside H2020/HE)
def cagr(s, a, b):
    va, vb = s.get(a, 0), s.get(b, 0)
    return round((vb / va) ** (1 / (b - a)) - 1, 3) if va > 0 and vb > 0 else None

summary = {
    'n_candidates': int(len(c)), 'label_counts': c['label'].value_counts().to_dict(),
    'totals_MEUR': {k: round(g['ec'].sum() / 1e6, 1) for k, g in groups.items()},
    'n_projects': {k: int(len(g)) for k, g in groups.items()},
    'cagr_prorata_2014_2025': {k: cagr(pr[k], 2014, 2025) for k in groups},
    'aem_projects': c[c.is_AEM_any][['acronym', 'programme', 'start_year', 'ec', 'fundingScheme', 'label', 'coord_name', 'coord_country', 'private_share']].sort_values('start_year').assign(ec=lambda d: (d.ec / 1e6).round(2)).to_dict('records'),
    'electrolysis_instrument_mix_share': mix_share.to_dict(),
    'aem_instrument_mix_MEUR': mix_aem.round(2).to_dict(),
    'by_programme_period': per_rows,
    'clean_h2_partnership_share_of_electrolysis_EC': round(c[c.is_ELY & c.clean_h2_partnership]['ec'].sum() / c[c.is_ELY]['ec'].sum(), 3),
}
json.dump(summary, open(os.path.join(PROC, 'cordis_summary.json'), 'w'), indent=1, default=str)
print(json.dumps({k: summary[k] for k in ['label_counts', 'totals_MEUR', 'n_projects', 'cagr_prorata_2014_2025', 'clean_h2_partnership_share_of_electrolysis_EC']}, indent=1))
print(P.to_string())
print(mix_share.to_string())
print(F[['year', 'AEM_any_n_projects_started', 'AEM_any_EC_by_start_MEUR', 'Electrolysis_all_n_projects_started', 'Electrolysis_all_EC_by_start_MEUR', 'Electrolysis_all_EC_prorata_MEUR', 'AEM_any_EC_prorata_MEUR', 'GreenH2_all_EC_prorata_MEUR']].to_string())
