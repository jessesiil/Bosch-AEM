"""Raw (unfiltered) OpenAlex hits per publication year for each of the 7 search queries, in one chart.

Reads : data/raw/openalex/bosch_openalex_part2.jsonl, line 1 -> metadata 'counts_by_year' (exact per-year totals
        returned by OpenAlex's group_by for each query: articles + reviews, retracted excluded).
Writes: data/processed/pub_query_counts_by_year.csv  (year x query table, plus all OpenAlex articles for scale)
        figures/pub_query_counts.png                  (log-scale line chart, 2000-2026; 2026 = partial year)
These are retrieval counts BEFORE relevance filtering; see pub_series.csv for the filtered (ML) series."""
import json, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import RAW, PROC, FIG
import plotting as P

meta = json.loads(open(os.path.join(RAW, 'openalex', 'bosch_openalex_part2.jsonl'), encoding='utf-8').readline())
C = meta['counts_by_year']
YEARS = list(range(2000, 2027))
LABELS = {  # query key -> plain-language name used in the chart
    'WE_core': 'Water electrolysis (generic phrases)',
    'GREEN_H2': 'Green / renewable hydrogen',
    'PEM': 'PEM electrolysis',
    'ALK': 'Alkaline electrolysis',
    'AEM_broad': 'AEM (broad query)',
    'SOEC': 'SOEC (solid oxide)',
    'AEM_strict': 'AEM (strict query)',
}
T = pd.DataFrame({k: [int(C[k]['by_year'].get(str(y), 0)) for y in YEARS] for k in list(LABELS) + ['ALL_WORKS']}, index=YEARS)
T.index.name = 'year'
T = T.rename(columns={'ALL_WORKS': 'ALL_OpenAlex_articles'})
T.to_csv(os.path.join(PROC, 'pub_query_counts_by_year.csv'))

fig, ax = plt.subplots(figsize=(9, 5.2))
yrs = np.array(YEARS); full = yrs <= 2025
for i, (k, lab) in enumerate(LABELS.items()):
    v = T[k].values.astype(float); v[v == 0] = np.nan
    ls = '--' if k == 'AEM_strict' else '-'
    ax.plot(yrs[full], v[full], color=P.C[i], linestyle=ls, label=f'{lab}  (N = {C[k]["N"]:,})')
    ax.plot(yrs[-2:], v[-2:], color=P.C[i], linestyle=(0, (1.5, 2)), linewidth=1.5)
    ax.plot(yrs[-1], v[-1], marker='o', mfc='white', mec=P.C[i], markersize=5)
ax.set_yscale('log'); ax.set_xlim(1999.5, 2026.6)
ax.set_ylabel('papers per year (log scale)')
ax.set_title('OpenAlex hits per year for each search query (before relevance filtering)')
ax.legend(fontsize=7.8, loc='upper left')
ax.text(0.99, 0.01, 'Articles + reviews, retracted excluded. Dotted segment / open marker = 2026, partial year (to 8 Oct).\n'
        'Queries overlap (one paper can match several). N = total hits 1900-2026.', transform=ax.transAxes, fontsize=7, color=P.INK2, va='bottom', ha='right')
P.finalize(fig); fig.savefig(os.path.join(FIG, 'pub_query_counts.png')); plt.close(fig)
print(T.loc[[2000, 2010, 2015, 2020, 2025, 2026]].to_string())
