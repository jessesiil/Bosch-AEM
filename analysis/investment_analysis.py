"""Economic trend analysis: (1) EU funding from CORDIS (see cordis_analysis.py for the data build), figures here;
(2) disclosed AEM company financing events (Researcher dataset, sourced per row); (3) macro hydrogen investment context.
All EUR amounts are nominal; private-round FX conversions are approximate (see aem_financing_events.csv header notes)."""
import json, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import RAW, PROC, FIG
import plotting as P

# ---------- CORDIS figures ----------
F = pd.read_csv(os.path.join(PROC, 'cordis_funding_by_year.csv')).set_index('year')
c = pd.read_csv(os.path.join(PROC, 'cordis_labelled_projects.csv'), dtype={'id': str})
# Panels A (funding by year) and B (AEM share) are functions so the 3-panel and 2-panel figures stay identical.
def panel_funding(ax):
    yrs = F.loc[2008:2026].index.values
    non = F.loc[2008:2026, 'Electrolysis_non_AEM_EC_prorata_MEUR'].values; aem = F.loc[2008:2026, 'AEM_any_EC_prorata_MEUR'].values
    ax.bar(yrs, non, color=P.C[0], width=0.8, label='other water electrolysis (PEM, alkaline, SOEC, generic)')
    ax.bar(yrs, aem, bottom=non, color=P.C[1], width=0.8, label='AEM electrolysis (core + multi-tech incl. AEM)', edgecolor='white', linewidth=0.5)
    ax.set_title('EU contribution flowing to electrolysis projects\n(pro-rata over project duration)', fontsize=10); ax.set_ylabel('€ million per year (nominal)')
    ax.legend(fontsize=7.5, loc='upper left'); ax.set_xlabel('2026 = grants still running only (partial)', fontsize=7.5, color=P.INK2)
def panel_share(ax):
    yy = F.loc[2014:2026].index.values
    sh_any = F.loc[2014:2026, 'AEM_share_of_electrolysis_EC_prorata'].values * 100
    sh_core = (F.loc[2014:2026, 'AEM_core_EC_prorata_MEUR'] / F.loc[2014:2026, 'Electrolysis_all_EC_prorata_MEUR']).values * 100
    ax.plot(yy, sh_any, color=P.C[1], marker='o', markersize=4, label='AEM core + multi-technology projects incl. AEM')
    ax.plot(yy, sh_core, color=P.C[0], marker='o', markersize=4, label='AEM-core projects only')
    for y, v in zip(yy, sh_core):
        if y in (2020, 2024, 2026): ax.annotate(f'{v:.0f}%', (y, v), xytext=(8 if y == 2020 else 0, -12), textcoords='offset points', ha='left' if y == 2020 else 'center', fontsize=8, color=P.INK2)
    for y, v in zip(yy, sh_any):
        if y in (2026,): ax.annotate(f'{v:.0f}%', (y, v), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8, color=P.INK2)
    ax.set_title('AEM share of EU electrolysis funding\n(pro-rata € basis)', fontsize=10); ax.set_ylabel('% of electrolysis EU contribution'); ax.set_ylim(0, max(sh_any) * 1.3); ax.legend(fontsize=7, loc='upper left')

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={'width_ratios': [1.25, 1, 1]})
panel_funding(axes[0]); panel_share(axes[1])
ax = axes[2]
mix = pd.read_csv(os.path.join(PROC, 'cordis_electrolysis_instrument_mix.csv')).set_index('period')
order = ['RIA / collaborative research', 'Innovation Action (demo / close to market)', 'EIC / SME instrument (start-ups)', 'ERC (frontier research)', 'MSCA (fellowships/training)', 'CSA (coordination)']
bottom = np.zeros(len(mix))
for i, col in enumerate([o for o in order if o in mix.columns]):
    v = mix[col].values * 100
    ax.bar(range(len(mix)), v, bottom=bottom, color=P.C[i], width=0.6, label=col, edgecolor='white', linewidth=1)
    bottom += v
ax.set_xticks(range(len(mix))); ax.set_xticklabels([p.replace(' (', '\n(') for p in mix.index], fontsize=8)
ax.set_title('Instrument mix of EU electrolysis funding', fontsize=10); ax.set_ylabel('% of EU contribution'); ax.legend(fontsize=6.8, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
fig.suptitle('EU research & innovation funding (CORDIS: FP7, H2020, Horizon Europe; all 620 candidate projects hand-labelled)', fontweight='bold', color=P.INK)
P.finalize(fig); fig.savefig(os.path.join(FIG, 'cordis_funding.png')); plt.close(fig)

# 2-panel variant (no instrument-mix panel), requested 2026-10-09 for use in the course report
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.25, 1]})
panel_funding(axes[0]); panel_share(axes[1])
fig.suptitle('EU research & innovation funding for water electrolysis (CORDIS: FP7, H2020, Horizon Europe; 620 candidate projects hand-labelled)', fontweight='bold', color=P.INK, fontsize=10)
P.finalize(fig); fig.savefig(os.path.join(FIG, 'cordis_funding_2panel.png')); plt.close(fig)

# ---------- private / public AEM company financing ----------
E = pd.read_csv(os.path.join(RAW, 'investment', 'aem_financing_events.csv'))
E['year'] = E['date'].astype(str).str[:4].astype(int)
def cls(r):
    t = str(r['event_type']).lower(); flag = str(r['flag']).lower()
    if 'extension to' in t or 'licence' in t or 'cancelled' in flag: return 'excluded (no new money / cancelled)'
    if 'state-aid' in t or 'grant' in t: return 'public grant / state aid'
    if 'capex' in t: return 'corporate capex'
    if 'debt' in t or 'bond' in t: return 'debt'
    return 'equity (venture, strategic, listed)'
E['class'] = E.apply(cls, axis=1)
E['amount_eur_m'] = pd.to_numeric(E['amount_eur_approx'], errors='coerce')
E.loc[E['company'].eq('Evonik') & E['amount_eur_m'].isna(), 'amount_eur_m'] = 15.0   # 'low double-digit million' -> midpoint assumption, flagged
E.to_csv(os.path.join(PROC, 'aem_financing_events_classified.csv'), index=False)
inc = E[~E['class'].str.startswith('excluded')]
pv = inc.pivot_table(index='year', columns='class', values='amount_eur_m', aggfunc='sum', fill_value=0).reindex(range(2018, 2027), fill_value=0)
ev = inc.groupby('year').size().reindex(range(2018, 2027), fill_value=0)
venture_only = inc[inc['class'].eq('equity (venture, strategic, listed)') & ~inc['company'].eq('Enapter')]
vy = venture_only.groupby('year')['amount_eur_m'].sum().reindex(range(2018, 2027), fill_value=0)
# Linear scale on purpose: stacked segments on a log axis are not proportional (2026-10-09 fix -- the earlier
# symlog version made 2025's EUR 418M of state aid look smaller than its EUR 32M of equity).
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]; bottom = np.zeros(len(pv))
for i, col in enumerate(['equity (venture, strategic, listed)', 'debt', 'corporate capex', 'public grant / state aid']):
    if col in pv:
        v = pv[col].values
        ax.bar(pv.index, v, bottom=bottom, color=P.C[i], width=0.7, label=col, edgecolor='white', linewidth=0.8,
               hatch=['///' if y == P.PARTIAL else '' for y in pv.index]); bottom += v
for y, n in ev.items():
    tot = bottom[list(pv.index).index(y)]
    ax.annotate(f'€{tot:.0f}M\n{n} ev.' if tot else f'{n} ev.', (y, tot), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7, color=P.INK2)
ax.set_ylabel('€ million (nominal, approx. FX)'); ax.set_ylim(0, bottom.max() * 1.18)
ax.set_title('Disclosed financing of AEM-focused companies by year', fontsize=10); ax.legend(fontsize=7.5, loc='upper left')
ax.set_xlabel('hatched 2026 = partial year (events found up to 8 Oct 2026)', fontsize=7.5, color=P.INK2)
ax = axes[1]
ax.bar(vy.index, vy.values, color=P.C[0], width=0.7, hatch=['///' if y == P.PARTIAL else '' for y in vy.index], edgecolor='white')
ax.set_title('Venture / strategic equity into AEM start-ups\n(excl. listed Enapter)', fontsize=10); ax.set_ylabel('€ million (nominal, approx. FX)')
ax.set_xlabel('hatched 2026 = partial year', fontsize=7.5, color=P.INK2)
for y, v in vy.items():
    if v > 0: ax.annotate(f'{v:.0f}', (y, v), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, color=P.INK2)
fig.suptitle('AEM company financing (hand-collected, sourced event list; undisclosed amounts counted as events only)', fontweight='bold', color=P.INK, fontsize=10.5)
P.finalize(fig); fig.savefig(os.path.join(FIG, 'aem_private_financing.png')); plt.close(fig)

# ---------- macro context ----------
M = pd.read_csv(os.path.join(RAW, 'investment', 'macro_hydrogen_investment.csv'))
inst = M[(M.metric == 'installed electrolysis capacity') & (M.scope == 'global')].copy()
inst['GW'] = inst.apply(lambda r: float(str(r['value']).split('-')[0]) / (1000 if r['unit'] == 'MW' else 1), axis=1)
comm = M[M.metric.str.startswith('committed')].copy(); comm['bn'] = comm['value'].astype(float)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]; ax.plot(inst.year, inst.GW, color=P.C[0], marker='o', markersize=4)
for y, g in zip(inst.year, inst.GW): ax.annotate(f'{g:g}', (y, g), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8, color=P.INK2)
ax.set_title('Global installed electrolyser capacity (IEA, via press)', fontsize=10); ax.set_ylabel('GW (cumulative, end of year)')
ax = axes[1]; ax.plot(comm.year, comm.bn, color=P.C[1], marker='o', markersize=4)
for y, g in zip(comm.year, comm.bn): ax.annotate(f'${g:g}bn', (y, g), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8, color=P.INK2)
ax.set_title('Committed (FID) clean-H2 investment, cumulative\n(Hydrogen Council)', fontsize=10); ax.set_ylabel('USD billion')
fig.suptitle('Macro context: green-hydrogen deployment and committed capital', fontweight='bold', color=P.INK)
P.finalize(fig); fig.savefig(os.path.join(FIG, 'macro_context.png')); plt.close(fig)

summary = {'events_total': int(len(E)), 'events_included': int(len(inc)), 'by_class_MEUR': inc.groupby('class')['amount_eur_m'].sum().round(1).to_dict(),
           'by_year_class_MEUR': pv.round(1).to_dict(), 'events_by_year': ev.to_dict(), 'venture_ex_enapter_by_year_MEUR': vy.round(1).to_dict(),
           'largest_venture_round_MEUR': float(venture_only['amount_eur_m'].max()), 'n_companies': int(inc['company'].nunique())}
json.dump(summary, open(os.path.join(PROC, 'investment_summary.json'), 'w'), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
