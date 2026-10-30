"""
v2_14_durability_table.py - one browsable table of every extracted durability test.

What it does
  Merges the tier-2 (full text) and tier-1 (abstract) durability results into a
  single table with one row per durability test, and adds summary tables.
  Per paper: if the full-text extraction found durability results, those rows are
  used (they are more complete); otherwise the abstract rows are used. A paper
  therefore never appears twice.

Inputs  (data/processed/v2/)
  tier2/durability.csv, tier1/durability.csv, tier1/paper_regions.csv
Outputs (data/processed/v2/tables/)
  durability_tests.xlsx  sheets: README, tests, by_year, by_duration, by_electrolyte,
                         longest_tests
  durability_tests.csv   the 'tests' sheet as CSV (for pandas / R)

Run (anywhere, no API keys):  python analysis/v2/v2_14_durability_table.py
Needs: openpyxl
"""
import csv, os, statistics as st
from collections import defaultdict
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
P = os.path.join(ROOT, 'data', 'processed', 'v2')
OUT = os.path.join(P, 'tables'); os.makedirs(OUT, exist_ok=True)
csv.field_size_limit(10**9)

KPI_2024 = 0.009 * 1.9 / 1000 * 1e6   # 0.9 %/1000 h at 1.9 V = 17.1 uV/h (EU CHP JU SRIA, AEMEL)
KPI_2030 = 0.005 * 1.9 / 1000 * 1e6   # 0.5 %/1000 h at 1.9 V = 9.5 uV/h

def load(p):
    with open(os.path.join(P, p), encoding='utf-8') as f:
        return list(csv.DictReader(f))

def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None

t1, t2 = load('tier1/durability.csv'), load('tier2/durability.csv')
regions = {r['uid']: r['regions'] for r in load('tier1/paper_regions.csv')}
# Same paper under two DOIs (Angewandte German edition 10.1002/ange.* vs International
# edition 10.1002/anie.*, repository copies, ...): v2_08 / v2_11 already keep one uid per title
# within each tier; here the two tiers may still hold different uids of one paper. Collapse by
# normalised title or abstract start (shared rule, v2_dedupe.py); keep the version with full-text
# results, then the publisher record, then the International edition, then the one with most rows.
from v2_dedupe import collapse_duplicate_papers, corpus_texts, is_repository_copy
texts = corpus_texts(ROOT)
n_rows = defaultdict(int)
for r in t1 + t2: n_rows[r['uid']] += 1
t2_uids = {r['uid'] for r in t2}
keep, _dropped = collapse_duplicate_papers(((u, texts[u]['title'], texts[u]['abstract']) for u in n_rows),
                                           rank=lambda x: (x in t2_uids, not is_repository_copy(x, texts[x]['source']), '/anie.' in x, n_rows[x]))
n_dup_papers = len(_dropped)
t1 = [r for r in t1 if r['uid'] in keep]; t2 = [r for r in t2 if r['uid'] in keep]
full_uids = {r['uid'] for r in t2}
src = [(r, 'full text') for r in t2] + [(r, 'abstract') for r in t1 if r['uid'] not in full_uids]

COLS = [  # (key, header, width)
    ('year', 'Year', 6), ('region', 'Region(s)', 14), ('title', 'Title', 50), ('journal', 'Journal', 24),
    ('doi', 'DOI', 26), ('source', 'Source', 9), ('test_id', 'Test', 5), ('test_level', 'Test level', 10),
    ('electrolyte', 'Electrolyte', 11), ('conc_M', 'Conc. (M)', 8), ('temperature_C', 'T (°C)', 7),
    ('mode', 'Operation mode', 15), ('j_held', 'j held (A/cm²)', 9), ('duration_h', 'Duration (h)', 9),
    ('V_start', 'V start (V)', 8), ('V_end', 'V end (V)', 8), ('rate', 'Rate (µV/h)', 9),
    ('rate_source', 'Rate source', 9), ('qual_only', 'Qualitative only', 9),
    ('kpi24', f'≤ EU 2024 KPI ({KPI_2024:.0f} µV/h)', 10), ('kpi30', f'≤ EU 2030 KPI ({KPI_2030:.1f} µV/h)', 10),
    ('location', 'Location in paper', 14), ('quote_check', 'Quote check', 11),
    ('claim', 'Claim (paraphrase)', 40), ('quote', 'Verbatim quote', 60),
]

rows = []
for r, s in src:
    rate = num(r['rate_uV_h'])
    rows.append(dict(
        year=int(r['year']) if r['year'] else None, region=regions.get(r['uid'], ''),
        title=r['title'], journal=r['journal'], doi=r['doi'], source=s,
        test_id=r['test_uid'].split('#')[-1], test_level=r['test_level'],
        electrolyte=r['electrolyte'], conc_M=num(r.get('conc_M')), temperature_C=num(r['temperature_C']),
        mode=r['mode'], j_held=num(r['j_held_A_cm2']), duration_h=num(r['duration_h']),
        V_start=num(r['V_start']), V_end=num(r['V_end']), rate=rate, rate_source=r['rate_source'],
        qual_only='yes' if r['qualitative_only'] == '1' else 'no',
        kpi24='' if rate is None else ('yes' if rate <= KPI_2024 else 'no'),
        kpi30='' if rate is None else ('yes' if rate <= KPI_2030 else 'no'),
        location=r.get('location', ''),
        quote_check=r.get('match_level', '') or 'abstract (exact)',
        claim=r['qualitative_claim'], quote=r['quote']))
# The extractor sometimes lists one test twice (e.g. from the abstract and the results
# section). Rows of the same paper with identical duration, rate, held current density and
# electrolyte are counted once. Cost: two configurations reported with identical numbers
# collapse into one row (rare).
seen, dedup = set(), []
for d in rows:
    k = (d['doi'], d['duration_h'], d['rate'], d['j_held'], d['electrolyte'])
    if d['duration_h'] is None and d['rate'] is None: k += (d['quote'][:80],)
    if k not in seen: seen.add(k); dedup.append(d)
n_dup = len(rows) - len(dedup); rows = dedup
rows.sort(key=lambda d: (-(d['year'] or 0), -(d['duration_h'] or 0)))

# ---- summaries (exclude 'numbers_nearby' quote matches: weakest verification) ----
S = [d for d in rows if d['quote_check'] != 'numbers_nearby']
def q(v, p):
    v = sorted(v)
    if not v: return None
    k = (len(v) - 1) * p; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)
def med(v): return st.median(v) if v else None
def share(v, f): return round(sum(map(f, v)) / len(v), 3) if v else None

def summarise(groups):
    out = []
    for k, g in groups:
        dur = [d['duration_h'] for d in g if d['duration_h']]
        rate = [d['rate'] for d in g if d['rate'] and d['rate'] > 0]
        rate_long = [d['rate'] for d in g if d['rate'] and d['rate'] > 0 and (d['duration_h'] or 0) >= 1000]
        j = [d['j_held'] for d in g if d['j_held']]
        out.append([k, len(g), len(dur), med(dur), q(dur, .9), max(dur) if dur else None,
                    share(dur, lambda x: x >= 1000), share(g, lambda d: d['qual_only'] == 'yes'),
                    len(rate), med(rate), len(rate_long), med(rate_long),
                    share(rate, lambda x: x <= KPI_2024), share(rate, lambda x: x <= KPI_2030), med(j)])
    return out
SUMH = ['Group', 'Tests', 'Tests with duration', 'Median duration (h)', 'P90 duration (h)', 'Max duration (h)',
        'Share ≥1000 h', 'Share qualitative only', 'Tests with rate', 'Median rate (µV/h)',
        'Tests with rate & ≥1000 h', 'Median rate, ≥1000 h tests (µV/h)', 'Share of rates ≤ 2024 KPI',
        'Share of rates ≤ 2030 KPI', 'Median j held (A/cm²)']

def yb(y): return '≤2021' if y and y <= 2021 else str(y)
by_year = defaultdict(list)
for d in S: by_year[yb(d['year'])].append(d)
year_rows = summarise(sorted(by_year.items(), key=lambda kv: kv[0].replace('≤', '0')))

BINS = [(0, 100, '<100 h'), (100, 200, '100–200 h'), (200, 500, '200–500 h'), (500, 1000, '500–1000 h'),
        (1000, 3000, '1000–3000 h'), (3000, 1e9, '≥3000 h')]
by_dur = [(lab, [d for d in S if d['duration_h'] and lo <= d['duration_h'] < hi]) for lo, hi, lab in BINS]
by_dur.append(('duration not stated', [d for d in S if not d['duration_h']]))
dur_rows = summarise(by_dur)

def el(e): return {'KOH': 'KOH', 'pure_water': 'pure water', 'not_stated': 'not stated'}.get(e, 'other (NaOH, carbonate, …)')
by_el = defaultdict(list)
for d in S: by_el[el(d['electrolyte'])].append(d)
el_rows = summarise(sorted(by_el.items()))
lvl = defaultdict(list)
for d in S: lvl['stack' if d['test_level'] == 'stack' else 'single cell'].append(d)
el_rows += [[''] * len(SUMH)] + summarise(sorted(lvl.items()))

longest = sorted([d for d in S if d['duration_h']], key=lambda d: -d['duration_h'])[:40]

# ---- write ----
with open(os.path.join(OUT, 'durability_tests.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow([h for _, h, _ in COLS])
    for d in rows: w.writerow(['' if d[k] is None else d[k] for k, _, _ in COLS])

wb = Workbook()
HF, HFILL = Font(bold=True, color='FFFFFF'), PatternFill('solid', fgColor='2A78D6')
def header(ws, hs, widths=None):
    ws.append(hs)
    for i, c in enumerate(ws[1], 1):
        c.font, c.fill = HF, HFILL; c.alignment = Alignment(wrap_text=True, vertical='top')
        ws.column_dimensions[get_column_letter(i)].width = (widths[i - 1] if widths else 14)
    ws.row_dimensions[1].height = 45; ws.freeze_panes = 'B2'

ws = wb.active; ws.title = 'README'
notes = [
    'AEM durability tests extracted from the literature (generated by analysis/v2/v2_14_durability_table.py)',
    '',
    f'Rows: {len(rows)} durability tests from {len({d["doi"] for d in rows})} papers '
    f'({sum(d["source"]=="full text" for d in rows)} from full text, {sum(d["source"]=="abstract" for d in rows)} from abstracts).',
    'Source = full text when the paper\'s full text was extracted and contained durability results, otherwise abstract. A paper appears under one source only.',
    'Duplicates: rows of the same paper with identical duration, rate, held current density and electrolyte are counted once '
    '(the extractor sometimes lists one test from both abstract and results section).',
    'Duplicate papers: the same article under two DOIs (mainly Angewandte Chemie German/International editions) is counted once, '
    'matched on normalised title.',
    'Every value is backed by a verbatim quote (column "Verbatim quote") checked against the source text in code.',
    'Quote check: exact = quote found verbatim; loose = found after removing spaces/symbols (PDF unit mangling); '
    'numbers_nearby = the numbers occur close together in the text but the full quote did not match (weakest; excluded from all summary sheets).',
    'Rate source: stated = rate given in the paper; computed = (V end − V start) / duration by the script; empty = no rate.',
    'Qualitative only = the paper only says "stable", "negligible degradation" etc. without a number for the rate.',
    f'KPI columns compare the rate with the EU Clean Hydrogen JU SRIA AEM targets converted at 1.9 V: 2024 0.9 %/1000 h ≈ {KPI_2024:.1f} µV/h, 2030 0.5 %/1000 h ≈ {KPI_2030:.1f} µV/h. '
    'These are stack/system targets; single-cell rates are not directly comparable.',
    'Region(s): from author affiliations (whole counting; a paper with Chinese and European authors counts for both).',
    'Year 2026 is partial (papers published up to the retrieval date).',
    '',
    'Caveats (see the limitations doc): rates from short tests are not comparable with rates from long tests (break-in, '
    'selection of best samples); publication bias toward good results; electrolyte often not stated in abstracts.',
    'The tables use LLM extraction (Claude Haiku 5.5) verified against quotes; extraction recall vs Sonnet was ~76–78%, so some tests are missing.',
    '',
    'Sheets: tests (one row per test, filterable) · by_year · by_duration · by_electrolyte (incl. stack vs single cell) · longest_tests (top 40).',
]
for n in notes: ws.append([n])
ws['A1'].font = Font(bold=True, size=13); ws.column_dimensions['A'].width = 160
for r in ws.iter_rows(min_row=2): r[0].alignment = Alignment(wrap_text=True, vertical='top')

def write_rows(ws, data):
    for d in data:
        ws.append(['' if d[k] is None else d[k] for k, _, _ in COLS])
        c = ws.cell(ws.max_row, 5)
        if d['doi']: c.hyperlink = 'https://doi.org/' + d['doi']; c.font = Font(color='2A78D6', underline='single')
    ws.auto_filter.ref = ws.dimensions

ws = wb.create_sheet('tests'); header(ws, [h for _, h, _ in COLS], [w for _, _, w in COLS]); write_rows(ws, rows)
ws = wb.create_sheet('longest_tests'); header(ws, [h for _, h, _ in COLS], [w for _, _, w in COLS]); write_rows(ws, longest)

PCT = {'Share ≥1000 h', 'Share qualitative only', 'Share of rates ≤ 2024 KPI', 'Share of rates ≤ 2030 KPI'}
for name, data, first in [('by_year', year_rows, 'Publication year'), ('by_duration', dur_rows, 'Test duration'),
                          ('by_electrolyte', el_rows, 'Electrolyte / test level')]:
    ws = wb.create_sheet(name); header(ws, [first] + SUMH[1:], [22] + [13] * (len(SUMH) - 1))
    for r in data: ws.append([('' if v is None else (round(v, 3 if SUMH[i] in PCT else 1) if isinstance(v, float) else v))
                              for i, v in enumerate(r)])
    for i, h in enumerate(SUMH, 1):
        if h in PCT:
            for c in ws.iter_cols(min_col=i, max_col=i, min_row=2):
                for cell in c: cell.number_format = '0%'
    ws.append([]); ws.append(['Summaries exclude numbers_nearby quote matches. Rates ≤0 (voltage improved) excluded from rate medians. 2026 partial.'])

wb.save(os.path.join(OUT, 'durability_tests.xlsx'))
print(f'collapsed {n_dup_papers} duplicate papers (same paper in tier 1 and tier 2 under different uids); removed {n_dup} duplicate test rows'); print(f'{len(rows)} tests from {len({d["doi"] for d in rows})} papers -> {OUT}')
for r in year_rows: print(r)
