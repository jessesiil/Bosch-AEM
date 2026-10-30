"""Shared figure styling (static PNGs for the report). Palette = validated categorical order from the dataviz reference
(blue, orange, aqua, yellow, magenta); thin 2px lines, recessive grid, direct labels, partial years marked."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scurves import logistic, gompertz

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
                     'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'legend.frameon': False, 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'lines.linewidth': 2})
PARTIAL = 2026
from matplotlib.ticker import MaxNLocator

def finalize(fig):
    for ax in fig.axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    has_sup = fig._suptitle is not None
    fig.tight_layout(rect=[0, 0, 1, 0.95 if has_sup else 1], h_pad=0.5)

def _label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8.5, color=INK2)

def _plot_series(ax, years, vals, color, label, partial_year=PARTIAL, marker=None):
    years = np.asarray(years); vals = np.asarray(vals, float)
    full = years < partial_year
    ax.plot(years[full], vals[full], color=color, label=label, marker=marker, markersize=4)
    if (~full).any():
        idx = np.where(years == partial_year)[0]
        if len(idx):
            i = idx[0]
            ax.plot(years[i - 1:i + 1], vals[i - 1:i + 1], color=color, linestyle=(0, (2, 2)), linewidth=1.5)
            ax.plot(years[i], vals[i], marker='o', mfc='white', mec=color, markersize=5)

def fig_filter_comparison(S, path):
    fig, ax = plt.subplots(figsize=(7.2, 4))
    yrs = S.loc[2005:].index.values
    # All four lines are subsets of the SAME pool: the 4,091 papers returned by the BROAD AEM query (aem_classified.csv).
    for i, (col, lab) in enumerate([('AEM_raw_query', 'Broad AEM query, unfiltered (4,091 papers)'), ('AEM_keyword_strict', 'Broad pool ∩ strict query (explicit AEMWE phrases)'),
                                    ('AEM_ML_core_plus_peripheral', 'ML: core + peripheral'), ('AEM_ML_core', 'ML: core AEM water electrolysis')]):
        _plot_series(ax, yrs, S.loc[2005:, col].values, C[i], lab)
    ax.set_title('AEM electrolysis publications per year under different filters\n(all lines = subsets of the broad AEM query pool)', fontsize=10.5)
    ax.set_ylabel('papers per year (OpenAlex, articles + reviews)')
    ax.legend(loc='upper left', fontsize=8.5)
    ax.text(0.99, 0.02, 'dashed / open marker = 2026, partial year (to Oct)', transform=ax.transAxes, ha='right', fontsize=7.5, color=INK2)
    finalize(fig); fig.savefig(path); plt.close(fig)

def fig_scurve(S, fits, col, title, path, horizon=2045, unit='papers'):
    f = fits[col]
    yrs = S.index.values; ann = S[col].values.astype(float)
    fy = f['years_fitted'][0]; ly = f['years_fitted'][1]
    off = ann[yrs < fy].sum()
    cum = np.cumsum(ann[yrs >= fy]) + off; cy = yrs[yrs >= fy]
    t = np.linspace(fy, horizon, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    m = cy <= ly
    ax.plot(cy[m], cum[m], 'o', color=INK, markersize=3.5, label='observed cumulative (fitted years)')
    ax.plot(cy[~m], cum[~m], 'o', mfc='white', mec=INK, markersize=4, label='not fitted (incomplete year(s))')
    for i, (mname, model) in enumerate([('logistic', logistic), ('gompertz', gompertz)]):
        if mname not in f or 'K' not in f[mname]: continue
        p = f[mname]; col_ = C[i]
        bp = np.array(p['boot_params_sample'])
        if len(bp):
            curves = np.array([model(t, *row) for row in bp])
            lo, hi = np.percentile(curves, 5, axis=0), np.percentile(curves, 95, axis=0)
            ax.fill_between(t, lo, hi, color=col_, alpha=0.12, linewidth=0)
        flag = ' (K at bound: ceiling not identifiable)' if p['boot_share_K_at_upper_bound'] and p['boot_share_K_at_upper_bound'] > 0.2 else ''
        ax.plot(t, model(t, p['K'], p['r'], p['t0_inflection']), color=col_, label=f"{mname}: inflection {p['t0_inflection']:.0f}, K≈{p['K']:,.0f}{flag}")
    ax.set_xlim(fy, horizon); ax.set_title('Cumulative count with S-curve fits (90% bootstrap band)', fontsize=10)
    ax.set_ylabel(f'cumulative {unit}'); ax.legend(fontsize=7.5, loc='upper left')
    ax = axes[1]
    ax.bar(yrs[(yrs >= fy) & (yrs <= ly)], ann[(yrs >= fy) & (yrs <= ly)], color='#86b6ef', width=0.8, label='observed per year')
    ax.bar([PARTIAL], [ann[yrs == PARTIAL][0]], color='white', edgecolor='#86b6ef', hatch='///', width=0.8, label='2026 partial')
    for i, (mname, model) in enumerate([('logistic', logistic), ('gompertz', gompertz)]):
        if mname not in f or 'K' not in f[mname]: continue
        p = f[mname]; tt = np.arange(fy, horizon + 1)
        ax.plot(tt, np.diff(np.concatenate([[model(fy - 1, p['K'], p['r'], p['t0_inflection'])], model(tt, p['K'], p['r'], p['t0_inflection'])])), color=C[i], label=f'{mname} implied annual')
    ax.set_xlim(fy, horizon); ax.set_title('Annual count vs model-implied annual output', fontsize=10); ax.set_ylabel(f'{unit} per year'); ax.legend(fontsize=7.5, loc='upper left')
    fig.suptitle(title, fontweight='bold', color=INK)
    finalize(fig); fig.savefig(path); plt.close(fig)

def fig_tech_shares(S, path):
    fig, ax = plt.subplots(figsize=(7.2, 4))
    yrs = S.loc[2005:2025].index.values; ends = []
    for i, (col, lab) in enumerate([('AEM_share_of_electrolysis', 'AEM'), ('PEM_share_of_electrolysis', 'PEM'), ('ALK_share_of_electrolysis', 'Alkaline'), ('SOEC_share_of_electrolysis', 'SOEC')]):
        v = S.loc[2005:2025, col].values * 100
        ax.plot(yrs, v, color=C[i], label=lab)
        ends.append((v[-1], lab))
    ends.sort(); last = -1e9
    for val, lab in ends:
        y = max(val, last + 1.6); last = y
        ax.annotate(lab, (yrs[-1], val), xytext=(yrs[-1] + 0.4, y), textcoords='data', va='center', fontsize=8.5, color=INK2)
    ax.set_title('Share of water-electrolysis papers mentioning each technology')
    ax.set_ylabel('% of electrolysis papers (multi-label)'); ax.legend(fontsize=8.5, loc='upper left'); ax.set_xlim(2005, 2027.5)
    finalize(fig); fig.savefig(path); plt.close(fig)

def fig_fisher_pry(S, fp, path):
    yrs = S.loc[2010:2025].index.values; f = S.loc[2010:2025, 'AEM_share_of_electrolysis'].values
    ok = (f > 0) & (f < 1)
    L = np.log(f[ok] / (1 - f[ok]))
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(yrs[ok], L, 'o', color=C[0], markersize=5, label='observed ln(f/(1-f))')
    t = np.arange(fp['years'][0], 2041)
    a = -fp['slope_per_year'] * fp['year_f_50pct'] if fp['year_f_50pct'] else 0
    ax.plot(t, a + fp['slope_per_year'] * t, color=C[1], label=f"linear fit {fp['years'][0]}-{fp['years'][1]} (R²={fp['r2']:.2f})")
    for share, ls in [(0.1, ':'), (0.5, '--')]:
        ax.axhline(np.log(share / (1 - share)), color=INK2, linewidth=0.8, linestyle=ls)
        ax.text(2040.5, np.log(share / (1 - share)), f'{int(share*100)}% share', fontsize=8, color=INK2, va='center')
    ax.set_title('Fisher–Pry view: AEM share of electrolysis research (logit scale)')
    ax.set_ylabel('ln( f / (1−f) )'); ax.legend(fontsize=8.5, loc='upper left'); ax.set_xlim(2010, 2044)
    finalize(fig); fig.savefig(path); plt.close(fig)

def fig_indexed_growth(S, path, base=2015):
    fig, ax = plt.subplots(figsize=(7.2, 4))
    yrs = S.loc[2010:2025].index.values
    for i, (col, lab) in enumerate([('AEM_ML_core', 'AEM water electrolysis'), ('ELY_ML_electrolysis', 'All water electrolysis'), ('OpenAlex_all_articles', 'All OpenAlex articles')]):
        v = S.loc[2010:2025, col].values / S.loc[base, col] * 100
        ax.plot(yrs, v, color=C[i], label=lab); _label_end(ax, yrs[-1], v[-1], lab, C[i])
    ax.set_yscale('log'); ax.set_title(f'Growth indexed to {base} = 100 (log scale)')
    ax.set_ylabel(f'index ({base} = 100)'); ax.legend(fontsize=8.5, loc='upper left'); ax.set_xlim(2010, 2029)
    finalize(fig); fig.savefig(path); plt.close(fig)

def fig_scenarios(S, fits, path, horizon=2050):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, fy, title in [(axes[0], 'AEM_ML_core', 2005, 'AEM water electrolysis papers'), (axes[1], 'ELY_ML_electrolysis', 2000, 'All water-electrolysis papers')]:
        yrs = S.index.values; ann = S[col].values.astype(float)
        off = ann[yrs < fy].sum(); cy = yrs[(yrs >= fy) & (yrs <= 2025)]; cum = np.cumsum(ann[(yrs >= fy) & (yrs <= 2025)]) + off
        ax.plot(cy, cum, 'o', color=INK, markersize=3.5, label='observed cumulative')
        t = np.linspace(fy, horizon, 400)
        for i, sc in enumerate([s for s in fits[col + '__scenarios'] if 'K' in s]):
            ax.plot(t, logistic(t, sc['K'], sc['r'], sc['t0_inflection']), color=C[i % len(C)], linewidth=1.6,
                    label=f"K = {sc['K_multiple']}× today → inflection {sc['t0_inflection']:.0f}")
        ax.set_yscale('log'); ax.set_xlim(fy, horizon); ax.set_title(title, fontsize=10); ax.set_ylabel('cumulative papers (log)')
        ax.legend(fontsize=7.5, loc='lower right')
    fig.suptitle('Scenario S-curves: logistic fits with the ceiling K fixed at multiples of the 2025 cumulative count', fontweight='bold', color=INK)
    finalize(fig); fig.savefig(path); plt.close(fig)
