"""Patent trend analysis (Google Patents, CPC C25B1/04 = hydrogen/oxygen production by water electrolysis).
Families are counted by earliest priority year. Last ~2 priority years are incomplete (18-month publication lag).
AEM series is precision-corrected using a hand-labelled stratified sample of the retrieved AEM families."""
import json, os, re
import numpy as np, pandas as pd
from common import RAW, PROC, LABELS, FIG
from scurves import fit_series, scenario_fits
import plotting as P
import matplotlib.pyplot as plt

d = json.load(open(os.path.join(RAW, 'patents', 'bosch_patents_googlepatents.json')))
meta, R = d['meta'], pd.DataFrame(d['aem_records'])
YEARS = list(range(2000, 2027))
T = pd.DataFrame({k: pd.Series({int(y): v for y, v in s.items()}) for k, s in meta['counts_total'].items()}).reindex(YEARS)
for k, s in meta['counts_by_country'].items():
    T[k] = pd.Series({int(y): v for y, v in s.items()}).reindex(YEARS)
T.index.name = 'priority_year'

# --- precision of the AEM keyword set from the labelled sample ---
S = json.load(open(os.path.join(LABELS, 'patent_aem_sample.json')))
pos = set(int(x) for x in open(os.path.join(LABELS, 'patent_aem_labels.txt')).read().split('R:')[1].split(','))
L = pd.DataFrame([{'stratum': s['stratum'], 'stratum_size': s['stratum_size'], 'rel': int(i in pos), 'id': s['id']} for i, s in enumerate(S)])
prec = L.groupby('stratum').agg(n=('rel', 'size'), precision=('rel', 'mean'), N=('stratum_size', 'first')).reset_index()
def stratum_of(y):
    for s in prec.stratum:
        a, b = map(int, s.split('-'))
        if a <= y <= b: return s
pmap = dict(zip(prec.stratum, prec.precision))
T['AEM_precision_used'] = [pmap.get(stratum_of(y), np.nan) for y in YEARS]
T['AEM_corrected'] = (T['AEM'] * T['AEM_precision_used']).round(1)
T['AEM_share_of_tech_tagged'] = (T['AEM_corrected'] / (T['AEM_corrected'] + T['PEM'] + T['ALK'] + T['SOEC'])).round(3)
T['AEM_CN_share'] = (T['AEM_CN'] / T['AEM']).round(3)
T['WE_CN_share'] = (T['WE_all_CN'] / T['WE_all']).round(3)
T.to_csv(os.path.join(PROC, 'patent_series.csv'))

# --- applicant analysis on retrieved AEM families (labelled-relevant weighting not possible per record; use keyword-strict titles) ---
STRICT = re.compile(r'anion[- ]exchange membrane (water )?electroly|AEM ?WE|AEM electroly|anion exchange membrane.*(water electroly|hydrogen production|electrolyzer|electrolyser|electrolytic)|for (alkaline )?water electrolysis', re.I)
R['strict'] = (R['title'].fillna('') + ' ' + R['snippet'].fillna('')).str.contains(STRICT)
def atype(a):
    a = str(a)
    if re.search(r'univ|大学|대학|institute|研究院|研究所|academy|科学院|research|fraunhofer|CNRS|CEA|研究|연구', a, re.I): return 'University / research org'
    if a in ('None', 'nan', ''): return 'unknown'
    if re.search(r'gmbh|inc|ltd|llc|corp|co\.|s\.r\.l|s\.p\.a|b\.v|\bag\b|\bsa\b|company|有限公司|株式会社|주식회사|se$|plc|holding|group|enapter', a, re.I): return 'Company'
    return 'Individual / other'
R['applicant_type'] = R['assignee'].map(atype)
R['office'] = R['pn'].str[:2]
mix = R[R.strict].assign(period=pd.cut(R.y, [1999, 2015, 2020, 2022, 2024, 2026], labels=['2000-2015', '2016-2020', '2021-2022', '2023-2024', '2025-2026*'])).pivot_table(index='period', columns='applicant_type', values='id', aggfunc='count', fill_value=0)
mix_share = mix.div(mix.sum(axis=1), axis=0).round(3)
top = R[R.strict & (R.y >= 2018)].groupby('assignee').size().sort_values(ascending=False).head(25)
eu_offices = {'EP', 'DE', 'FR', 'IT', 'GB', 'NL', 'AT', 'ES', 'DK', 'SE', 'FI', 'BE', 'CH', 'NO'}
top_eu = R[R.strict & (R.y >= 2018) & R.office.isin(eu_offices | {'WO'})].groupby('assignee').size().sort_values(ascending=False).head(25)
R.to_csv(os.path.join(PROC, 'patent_aem_records.csv'), index=False)

# --- S-curve fits (through 2023 = last priority year essentially complete; sensitivity through 2024) ---
fits = {}
for col, fy in [('AEM_corrected', 2005), ('AEM', 2005), ('PEM', 2000), ('ALK', 2000), ('SOEC', 2000), ('WE_all', 2000)]:
    v = T[col].fillna(0).values
    fits[col] = fit_series(col, YEARS, v, first_year=fy, last_full_year=2023)
    fits[col + '__to2024'] = fit_series(col, YEARS, v, first_year=fy, last_full_year=2024, n_boot=300)
fits['AEM_corrected__scenarios'] = scenario_fits(YEARS, T['AEM_corrected'].fillna(0).values, 2005, last_full_year=2023)
fits['WE_all__scenarios'] = scenario_fits(YEARS, T['WE_all'].fillna(0).values, 2000, last_full_year=2023)
json.dump({'precision_by_stratum': prec.to_dict('records'), 'fits': fits, 'applicant_mix_share_strict': mix_share.to_dict(),
           'top_applicants_2018plus_strict': top.to_dict(), 'top_applicants_EU_WO_offices_2018plus': top_eu.to_dict(),
           'n_records': int(len(R)), 'n_strict': int(R.strict.sum())}, open(os.path.join(PROC, 'patent_summary.json'), 'w'), indent=1, default=str)

# --- figures ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]; yrs = np.array(YEARS)
for i, (col, lab) in enumerate([('AEM_corrected', 'AEM (precision-corrected)'), ('PEM', 'PEM'), ('ALK', 'Alkaline'), ('SOEC', 'SOEC')]):
    v = T[col].values.astype(float); m = (yrs >= 2005) & (yrs <= 2023)
    ax.plot(yrs[m], v[m], color=P.C[i], label=lab)
    m2 = (yrs >= 2023) & (yrs <= 2024); ax.plot(yrs[m2], v[m2], color=P.C[i], linestyle=(0, (2, 2)), linewidth=1.5)
ax.axvspan(2023.5, 2024.5, color='#f0efec', zorder=0); ax.text(2024, ax.get_ylim()[1] * 0.97, 'likely\nincomplete', ha='center', va='top', fontsize=7, color=P.INK2)
ax.set_title('Water-electrolysis patent families by technology keyword', fontsize=10); ax.set_ylabel('families per priority year'); ax.legend(fontsize=8, loc='upper left'); ax.set_xlim(2005, 2024.8)
ax = axes[1]
v = T['WE_all'].values.astype(float); cn = T['WE_all_CN'].values.astype(float); intl = T['WE_all_INTL'].values.astype(float)
m = (yrs >= 2005) & (yrs <= 2024)
ax.plot(yrs[m], v[m], color=P.C[0], label='all CPC C25B1/04 families'); ax.plot(yrs[m], cn[m], color=P.C[1], label='with a Chinese (CN) member')
ax.plot(yrs[m], intl[m], color=P.C[2], label='with an EP / WO / US member')
ax.axvspan(2023.5, 2024.5, color='#f0efec', zorder=0)
ax.set_title('All water-electrolysis patent families (CPC C25B1/04)', fontsize=10); ax.set_ylabel('families per priority year'); ax.legend(fontsize=8, loc='upper left')
fig.suptitle('Patent activity (Google Patents, earliest priority year)', fontweight='bold', color=P.INK)
P.finalize(fig); fig.savefig(os.path.join(FIG, 'patents_trends.png')); plt.close(fig)
P.fig_scurve(T.assign(AEM_corrected=T.AEM_corrected.fillna(0)), {'AEM_corrected': fits['AEM_corrected']}, 'AEM_corrected', 'AEM water-electrolysis patent families (precision-corrected), fit to 2023', os.path.join(FIG, 'patents_aem_scurve.png'), unit='patent families')
print(prec.to_string()); print(T[['WE_all', 'WE_all_CN', 'WE_all_INTL', 'AEM', 'AEM_corrected', 'AEM_CN', 'AEM_INTL', 'PEM', 'ALK', 'SOEC', 'AEM_share_of_tech_tagged']].loc[2010:].to_string())
print(mix_share.to_string()); print(top.head(15).to_string()); print(top_eu.head(15).to_string())
for k in ['AEM_corrected', 'AEM_corrected__to2024', 'PEM', 'ALK', 'SOEC', 'WE_all']:
    f = fits[k]; print(k, {m: {a: (round(b, 2) if isinstance(b, float) else b) for a, b in f[m].items() if a in ('K', 't0_inflection', 'share_of_K_reached', 'boot_share_K_at_upper_bound', 'boot_t0_5_50_95', 'aic_annual')} for m in ('logistic', 'gompertz') if m in f}, f.get('exponential'))
for sc in fits['AEM_corrected__scenarios']: print(' AEM pat scen', {a: (round(b, 2) if isinstance(b, float) else b) for a, b in sc.items()})
for sc in fits['WE_all__scenarios']: print(' WE pat scen', {a: (round(b, 2) if isinstance(b, float) else b) for a, b in sc.items()})
