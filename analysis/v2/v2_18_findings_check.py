"""v2 step 18 - recompute every number quoted in analysis/v2/FINDINGS_TIER1.md and FINDINGS_TIER2.md and report differences.

  python analysis/v2/v2_18_findings_check.py            # check: OK / DIFF per number, exit code 1 if any DIFF
  python analysis/v2/v2_18_findings_check.py --print    # also print every expected phrase (use when rewriting FINDINGS)

How it works: every FINDINGS number is computed from saved outputs only (no API calls) and turned into the exact phrase or
table row FINDINGS must contain. A phrase that is not found is reported as DIFF together with the closest FINDINGS line.
Inputs (data/processed/v2/ unless stated): tier1/*.csv + cleaning_log.json + trend_summary.csv + paper_regions.csv +
region_trends.csv (steps 08/09/13), tier2/*.csv + cleaning_log.json (step 11), tier2_sonnet_sample40/*.csv (step 11 on the Sonnet
sample run), llm/fulltext_haiku_results.jsonl, llm/fulltext_sonnet_sample40_requests_map.csv, llm/gold_haiku_v2_* and
data/labels/v2/gold_labels_v2.csv (step 04), fulltext_index.csv (step 07), data/raw/fulltext/fetch_log.csv (steps 06/06b/06c),
corpus.csv (step 01).
Definitions FINDINGS relies on (written down here once):
  'same paper' across tier 1 / tier 2 / Sonnet sample = same normalised title or abstract start (v2_dedupe.py), because the
      tiers may keep different uids of one paper (e.g. ange vs anie edition).
  Abstract-vs-full-text agreement: share of tier-1 values whose exact numbers (j and V; durability hours) also occur among the
      tier-2 values of the same paper (all match levels); durability results without hours are left out.
  Haiku-vs-Sonnet: unique (j, V) pairs and unique durations (incl. 'no duration') per paper; Haiku side = tier2/ rows of the
      same paper; electrolyte = most frequent electrolyte of the paper's tests.
  Electrolyte stated = electrolyte not 'not_stated'; KOH ~1 M = 0.9-1.1 M among KOH tests that state a concentration.
  Conditions 'abstracts (same papers)' = tier-1 tests of the papers that are also in tier 2.
Not computed (external references, cited in FINDINGS): EU Clean Hydrogen JU KPIs, DOE 80,000 h lifetime target.
Added 2026-10-22 (T-058)."""
import csv, json, os, sys, difflib, statistics as st, collections
import numpy as np
csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v2_dedupe import title_key, abstract_key
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
P = os.path.join(ROOT, 'data', 'processed', 'v2')


def L(*p):
    with open(os.path.join(P, *p), encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def f(x): return float(x) if x not in ('', None) else None
def pc(x): return f'{100 * x:.0f}%'


C = []   # (file, label, expected phrase)
def chk(file, label, phrase): C.append((file, label, phrase))
T1F, T2F = 'FINDINGS_TIER1.md', 'FINDINGS_TIER2.md'

# ------------------------------------------------------------------ tier 1
lg1 = json.load(open(os.path.join(P, 'tier1', 'cleaning_log.json')))
t1t, t1p, t1d = L('tier1', 'tests.csv'), L('tier1', 'perf_points.csv'), L('tier1', 'durability.csv')
chk(T1F, 'papers', f"Data: {lg1['papers']:,} distinct papers ({lg1['papers_selected_incl_duplicates']:,} records before removing "
                   f"{lg1['cover_records_dropped']} cover records and {lg1['duplicate_papers_dropped']} duplicates)")
chk(T1F, 'kept values', f"Kept: {lg1['pp_kept']:,} of {lg1['pp_raw']:,} current-density/voltage pairs, "
                        f"{lg1['dur_kept']:,} of {lg1['dur_raw']:,} durability results")
gold = L('..', '..', 'labels', 'v2', 'gold_labels_v2.csv')
gm = {r['custom_id']: r['gid'] for r in L('llm', 'gold_haiku_v2_requests_map.csv')}
go = {}
for line in open(os.path.join(P, 'llm', 'gold_haiku_v2_results.jsonl'), encoding='utf-8'):
    r = json.loads(line); go[gm[r['custom_id']]] = r.get('output') or {}
y = [g['durability_data'] == 'Y' for g in gold]
p = [any(t.get('durability') for t in go.get(g['gid'], {}).get('tests') or []) for g in gold]
tp = sum(a and b for a, b in zip(y, p)); fp = sum(b and not a for a, b in zip(y, p)); fn = sum(a and not b for a, b in zip(y, p))
chk(T1F, 'gold durability', f"Durability extraction vs the {len(gold)} gold papers: precision {tp / (tp + fp):.2f}, recall {tp / (tp + fn):.2f}")

TS = {(r['metric'], r['year_bin']): r for r in L('tier1', 'trend_summary.csv')}
BINS = ['<=2021', '2022', '2023', '2024', '2025', '2026']
def row(label, metric, key, fmt):
    cells = [fmt(TS[(metric, b)]) if TS[(metric, b)].get(key) not in ('', None) else '—' for b in BINS]
    chk(T1F, 'table ' + label, f"| {label} | " + ' | '.join(cells) + ' |')
num = lambda k, d: (lambda r: f"{float(r[k]) + 1e-9:,.{d}f}")
row('Current density at 2.0 V, median (A cm-2)', 'j_at_2V_A_cm2', 'median', num('median', 2))
row('— 90th percentile (unstable when n < 40)', 'j_at_2V_A_cm2', 'p90', num('p90', 1))
row('— n values (j at 2.0 V)', 'j_at_2V_A_cm2', 'n', lambda r: r['n'])
row('Cell voltage at 1 A cm-2, median (V)', 'V_at_1A_V', 'median', num('median', 2))
row('— best decile (V)', 'V_at_1A_V', 'p10_best_decile', num('p10_best_decile', 2))
row('— n values (V at 1 A cm-2)', 'V_at_1A_V', 'n', lambda r: r['n'])
row('Durability test length, median (h)', 'durability_test_h', 'median', num('median', 0))
row('— n tests with a duration', 'durability_test_h', 'n', lambda r: r['n'])
row('— share of tests >= 1,000 h', 'durability_test_h', 'share_ge_1000h', lambda r: pc(float(r['share_ge_1000h'])))
row('— share with no voltage or rate at all ("stable for X h")', 'durability_test_h', 'share_qualitative_only',
    lambda r: pc(float(r['share_qualitative_only'])))
row('Stated degradation rate, median (µV/h)', 'degradation_uV_h', 'median', lambda r: f"{float(r['median']):,.0f} (n={r['n']})")
row('— share meeting EU 2024 KPI (0.9 %/1000 h)', 'degradation_uV_h', 'share_meets_EU2024', lambda r: pc(float(r['share_meets_EU2024'])))
row('Tests stating pure-water feed', 'test_conditions', 'share_pure_water', lambda r: pc(float(r['share_pure_water'])))
row('Tests stating the electrolyte at all', 'test_conditions', 'share_not_stated', lambda r: pc(1 - float(r['share_not_stated'])))
row('Stack tests (count)', 'test_conditions', 'stack_tests', lambda r: r['stack_tests'])
g = lambda m, b, k: float(TS[(m, b)][k])
j23 = [float(r['j_A_cm2']) for r in t1p if r['year'] == '2023' and abs(float(r['V_cell']) - 2) <= .03]
chk(T1F, '2023 p90', f"2023 p90 = {g('j_at_2V_A_cm2', '2023', 'p90'):.1f} A cm-2 from {len(j23)} values")
# readings
chk(T1F, 'R1 x', f"median j@2 V ×{g('j_at_2V_A_cm2', '2026', 'median') / g('j_at_2V_A_cm2', '2022', 'median'):.0f} since 2022")
rec = [r for r in t1p if abs(float(r['V_cell']) - 2) <= .03 and float(r['j_A_cm2']) >= 10]
yrs = sorted({r['year'] for r in rec})
chk(T1F, 'R1 records', f"{len(rec)} record claims of {min(float(r['j_A_cm2']) for r in rec):.0f}–{max(float(r['j_A_cm2']) for r in rec):.1f} "
                       f"A cm-2 at 2 V ({yrs[0]}–{yrs[-1]}); {sum(r['electrolyte'] in ('KOH', 'NaOH') for r in rec)} state KOH/NaOH, "
                       f"{sum(r['electrolyte'] == 'not_stated' for r in rec)} state no electrolyte, "
                       f"{sum((f(r['temperature_C']) or 0) >= 80 for r in rec)} state ≥ 80 °C")
v22 = [g('V_at_1A_V', b, 'median') for b in BINS[1:]]
chk(T1F, 'R2', f"median {st.median(v22) + 1e-9:.2f} V at 1 A cm-2 (≈{1.481 / st.median(v22) * 100:.0f} % HHV voltage efficiency) since 2022")
p10 = [g('V_at_1A_V', b, 'p10_best_decile') for b in BINS[1:]]
chk(T1F, 'R2 decile', f"best decile has stayed at {min(p10):.2f}–{max(p10):.2f} V since 2022")
chk(T1F, 'R3', f"median {g('durability_test_h', '2022', 'median'):.0f} h -> {g('durability_test_h', '2026', 'median'):.0f} h; ≥1,000 h share "
               f"{pc(g('durability_test_h', '2022', 'share_ge_1000h'))} -> {pc(g('durability_test_h', '2026', 'share_ge_1000h'))}")
top = sorted({f(r['duration_h']) for r in t1d if r['duration_h']}, reverse=True)[:2]
chk(T1F, 'R3 longest', f"Longest abstract claims: {top[0]:,.0f} h, {top[1]:,.0f} h")
chk(T1F, 'R4 qual', f"~{pc(sum(int(r['qualitative_only']) for r in t1d) / len(t1d))} of durability claims give no rate or voltages")
rm = [g('degradation_uV_h', b, 'median') for b in ['2024', '2025', '2026']]
ks = [g('degradation_uV_h', b, 'share_meets_EU2024') for b in ['2024', '2025', '2026']]
chk(T1F, 'R4 rate', f"the 2024–26 median (~{min(rm):.0f}–{max(rm):.0f} µV/h) is {min(rm) / 17.1:.0f}–{max(rm) / 17.1:.0f}× the EU 2024 KPI; "
                    f"only {pc(min(ks))}–{pc(max(ks))} of stated rates meet it")
chk(T1F, 'R5', f"pure-water case is a shrinking share of tests ({pc(g('test_conditions', '<=2021', 'share_pure_water'))} -> "
               f"{pc(g('test_conditions', '2026', 'share_pure_water'))})")
ns = [g('test_conditions', b, 'share_not_stated') for b in ['2023', '2024', '2025', '2026']]
chk(T1F, 'R5 not stated', f"{pc(min(ns))}–{pc(max(ns))} of abstract-level tests (2023–26) do not state the electrolyte")
sp = {r['uid']: r['year'] for r in t1t if r['paper_level'] == 'stack'}
chk(T1F, 'R6', f"{len(sp)} papers with extracted tests whose highest test level is stack, {sum(v in ('2025', '2026') for v in sp.values())} of them in 2025–26")
reg = L('tier1', 'paper_regions.csv'); rc = collections.Counter(x for r in reg for x in r['regions'].split('; ') if x)
chk(T1F, 'regions', f"China {rc['China']:,} · South Korea {rc['South Korea']:,} · Europe {rc['Europe']:,} · United States {rc['United States']:,}")
chk(T1F, 'Europe share', f"Europe contributes ~{pc(rc['Europe'] / len(reg))} of the {len(reg):,} papers with extracted values "
                         f"({pc(rc['Europe'] / lg1['papers'])} of all {lg1['papers']:,} selected papers)")
RT = {(r['metric'], r['region'], r['year_bin']): (float(r['value']), int(r['n'])) for r in L('tier1', 'region_trends.csv')}
jr = lambda rg, b: RT[('j_at_2V_median_A_cm2', rg, b)]
sr = lambda rg, b: RT.get(('durability_share_ge_1000h_pct', rg, b))
j26 = [jr('China', '2026')[0], jr('Europe', '2026')[0]]
chk(T1F, 'reg j', f"China and Europe both rise to {min(j26):.1f}–{max(j26):.1f} A cm-2 (2026); "
                  f"South Korea peaked at {jr('South Korea', '2024')[0]:.1f} in 2024 (n={jr('South Korea', '2024')[1]})")
usn = [RT[k][1] for k in RT if k[0] == 'j_at_2V_median_A_cm2' and k[1] == 'United States']
chk(T1F, 'reg US', f"the US has {min(usn)}–{max(usn)} values per year")
RB = ['<=2022', '2023', '2024', '2025', '2026']
chk(T1F, 'reg China share', f"China rises from {sr('China', '<=2022')[0]:.0f}% (≤2022; {sr('China', '2023')[0]:.0f}% in 2023) to "
                            f"{sr('China', '2026')[0]:.0f}% (2026)")
eu = [sr(r, b) for r in ('Europe', 'United States') for b in RB if sr(r, b)]
chk(T1F, 'reg EU/US', f"Europe and US swing ({min(v for v, _ in eu):.0f}–{max(v for v, _ in eu):.0f}%, n {min(n for _, n in eu)}–{max(n for _, n in eu)})")
kr = [sr('South Korea', b)[0] for b in RB]
chk(T1F, 'reg KR', f"South Korea {min(kr):.0f}–{max(kr):.0f}%")
r26 = [sr(r, '2026')[0] for r in ('China', 'Europe', 'United States', 'South Korea')]
chk(T1F, 'reg 2026', f"In 2026 all regions sit at {min(r26):.0f}–{max(r26):.0f}%")

# ------------------------------------------------------------------ tier 2
lg2 = json.load(open(os.path.join(P, 'tier2', 'cleaning_log.json')))
t2t, t2p, t2d = L('tier2', 'tests.csv'), L('tier2', 'perf_points.csv'), L('tier2', 'durability.csv')
corpus = {r['uid']: r for r in L('corpus.csv')}
par = {}
def find(x):
    while par.setdefault(x, x) != x:
        x = par[x]
    return x
def link(u):
    c = corpus[u]; par[find(u)] = find('T:' + title_key(u, c['title'])); ak = abstract_key(c['abstract'])
    if ak: par[find(u)] = find('A:' + ak)
idx = [r for r in L('fulltext_index.csv') if r['status'] == 'ok']
sonnet = {k: L('tier2_sonnet_sample40', f'{k}.csv') for k in ('tests', 'perf_points', 'durability')}
smap = L('llm', 'fulltext_sonnet_sample40_requests_map.csv')
for u in sorted({r['uid'] for r in t1t + t2t + idx + smap}): link(u)
key = find
fetch = L('..', '..', 'raw', 'fulltext', 'fetch_log.csv')
sel = {r['uid']: r['doi'] for r in fetch}
chk(T2F, 'texts', f"Data: {len(idx)} full texts (of {len(sel):,} selected papers; {len({key(r['uid']) for r in idx})} distinct papers")
route = collections.Counter(r['source'] for r in idx)
chk(T2F, 'routes', f"usable texts by route: Wiley TDM {route['wiley_tdm']}, open-access PDFs {route['openalex_oa_pdf']}, "
                   f"Semantic Scholar {route['semantic_scholar']}")
PUB = {'10.1002': 'Wiley', '10.1016': 'Elsevier', '10.1021': 'ACS', '10.1039': 'RSC'}
den = collections.Counter(PUB.get(d.split('/')[0]) for d in sel.values() if d)
nm = collections.Counter(PUB.get((sel.get(r['uid']) or '').split('/')[0]) for r in idx)
chk(T2F, 'publishers', f"Wiley {nm['Wiley']}/{den['Wiley']} (≈{pc(nm['Wiley'] / len(idx))} of the texts), Elsevier {nm['Elsevier']}/{den['Elsevier']} "
                       f"({pc(nm['Elsevier'] / den['Elsevier'])}), ACS {nm['ACS']}/{den['ACS']}, RSC {nm['RSC']}/{den['RSC']}")
# all fulltext_haiku* runs (base run + reruns of truncated papers, as combined by v2_11): which requested papers ended with output
_runs = sorted(f[:-14] for f in os.listdir(os.path.join(P, 'llm')) if f.startswith('fulltext_haiku') and f.endswith('_results.jsonl'))
_req, _out = set(), set()
for _run in _runs:
    _cmap = {r['custom_id']: r['uid'] for r in csv.DictReader(open(os.path.join(P, 'llm', f'{_run}_requests_map.csv'), encoding='utf-8'))}
    _req |= set(_cmap.values())
    for line in open(os.path.join(P, 'llm', f'{_run}_results.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        if r.get('output'): _out.add(_cmap[r['custom_id']])
chk(T2F, 'run', f"Run status: {len(_out & _req)}/{len(_req)} with output; {lg2.get('papers_out_of_scope', 0)} judged out of scope")
chk(T2F, 'papers', f"{lg2['papers_with_output']} papers with output → {lg2['papers']} distinct papers after removing "
                   f"{lg2['cover_records_dropped']} cover records and {lg2['duplicate_papers_dropped']} duplicates")
ml = collections.Counter(r['match_level'] for r in t2p); dl = collections.Counter(r['match_level'] for r in t2d)
chk(T2F, 'pp kept', f"{lg2['pp_kept']:,} of {lg2['pp_raw']:,} performance points kept ({ml['exact']:,} exact quote match, {ml['loose']} 'loose'")
chk(T2F, 'pp nearby', f"{ml['numbers_nearby']} 'numbers_nearby'")
chk(T2F, 'dur kept', f"{lg2['dur_kept']} of {lg2['dur_raw']} durability results kept ({dl['exact']} exact, {dl['loose']} loose, "
                     f"{dl['numbers_nearby']} numbers_nearby; {lg2['dur_drop_hours_not_in_quote']} dropped because the hours were not in the quote)")
hk, hd, ht, stt = (collections.defaultdict(list) for _ in range(4))
for r in t2p: hk[key(r['uid'])].append(r)
for r in t2d: hd[key(r['uid'])].append(r)
for r in t2t: ht[key(r['uid'])].append(r)
for r in sonnet['tests']: stt[key(r['uid'])].append(r)
sk = {key(m['uid']) for m in smap}
Hp = {(k, f(r['j_A_cm2']), f(r['V_cell'])) for k in sk for r in hk[k]}
Sp = {(key(r['uid']), f(r['j_A_cm2']), f(r['V_cell'])) for r in sonnet['perf_points']}
Hd = {(k, f(r['duration_h'])) for k in sk for r in hd[k]}
Sd = {(key(r['uid']), f(r['duration_h'])) for r in sonnet['durability']}
chk(T2F, 'val pp', f"{len(Hp & Sp)} of Haiku's {len(Hp)} performance points also found by Sonnet ({pc(len(Hp & Sp) / len(Hp))} agreement); "
                   f"Haiku finds {len(Hp & Sp)} of Sonnet's {len(Sp)} ({pc(len(Hp & Sp) / len(Sp))})")
chk(T2F, 'val dur', f"Durability: {len(Hd & Sd)} of {len(Hd)} shared; Haiku finds {len(Hd & Sd)} of Sonnet's {len(Sd)} ({pc(len(Hd & Sd) / len(Sd))})")
def main_el(rows):
    c = collections.Counter(r['electrolyte'] for r in rows); return c.most_common(1)[0][0] if c else None
chk(T2F, 'val el', f"Electrolyte identical in {sum(main_el(stt[k]) is not None and main_el(ht[k]) == main_el(stt[k]) for k in sk)}/{len(smap)} papers")
t2k = {key(r['uid']) for r in t2t}
ap = [r for r in t1p if key(r['uid']) in t2k]; ad = [r for r in t1d if key(r['uid']) in t2k and r['duration_h']]
mp = sum(any(f(a['j_A_cm2']) == f(b['j_A_cm2']) and f(a['V_cell']) == f(b['V_cell']) for b in hk[key(a['uid'])]) for a in ap)
md = sum(any(f(a['duration_h']) == f(b['duration_h']) for b in hd[key(a['uid'])]) for a in ad)
chk(T2F, 'abs vs full', f"{pc(mp / len(ap))} of abstract performance points ({mp}/{len(ap)}) and {pc(md / len(ad))} of abstract durability "
                        f"results with hours ({md}/{len(ad)}) are found again")
same = [r for r in t1t if key(r['uid']) in t2k]
chk(T2F, 'cond el', f"| electrolyte stated | {pc(sum(r['electrolyte'] != 'not_stated' for r in same) / len(same))} of tests | "
                    f"{pc(sum(r['electrolyte'] != 'not_stated' for r in t2t) / len(t2t))} of tests |")
chk(T2F, 'cond T', f"| temperature stated | {pc(sum(r['temperature_C'] != '' for r in same) / len(same))} | "
                   f"{pc(sum(r['temperature_C'] != '' for r in t2t) / len(t2t))} |")
em = collections.Counter(r['electrolyte'] for r in t2t)
koh = [f(r['conc_M']) for r in t2t if r['electrolyte'] == 'KOH' and r['conc_M']]
chk(T2F, 'mix', f"KOH {em['KOH']} tests (≈{pc(sum(0.9 <= c <= 1.1 for c in koh) / len(koh))} of those stating a concentration at 0.9–1.1 M), "
                f"pure water {em['pure_water']}, NaOH {em['NaOH']}, carbonate {em['K2CO3_or_KHCO3']}, other {em['other']}, not stated {em['not_stated']}")
tt = [f(r['temperature_C']) for r in t2t if r['temperature_C']]
chk(T2F, 'temp', f"Temperature: median {st.median(tt):.0f} °C (55–65 °C: {sum(55 <= x <= 65 for x in tt)} tests, 75–85 °C: {sum(75 <= x <= 85 for x in tt)})")
pp = [r for r in t2p if r['match_level'] != 'numbers_nearby']; du = [r for r in t2d if r['match_level'] != 'numbers_nearby']
G = [('pure', lambda e: e == 'pure_water'), ('koh', lambda e: e in ('KOH', 'NaOH'))]
V = {n: [f(r['V_cell']) for r in pp if abs(f(r['j_A_cm2']) - 1) <= .05 and t(r['electrolyte'])] for n, t in G}
J = {n: [f(r['j_A_cm2']) for r in pp if abs(f(r['V_cell']) - 2) <= .03 and t(r['electrolyte'])] for n, t in G}
R = {n: [f(r['rate_uV_h']) for r in du if r['rate_uV_h'] and f(r['rate_uV_h']) > 0 and t(r['electrolyte'])] for n, t in G}
H = {n: [f(r['duration_h']) for r in du if r['duration_h'] and t(r['electrolyte'])] for n, t in G}
chk(T2F, 'pw V', f"| cell voltage at 1 A cm-2, median | {st.median(V['pure']):.2f} V (n={len(V['pure'])}) | {st.median(V['koh']):.2f} V "
                 f"(n={len(V['koh'])}; best decile {np.percentile(V['koh'], 10):.2f} V) |")
chk(T2F, 'pw j', f"| current density at 2.0 V, median | {st.median(J['pure']):.1f} A cm-2 (n={len(J['pure'])}, max {max(J['pure']):.1f}) | "
                 f"{st.median(J['koh']):.1f} A cm-2 (n={len(J['koh'])}, p90 {np.percentile(J['koh'], 90):.1f}) |")
chk(T2F, 'pw rate', f"| degradation rate, median | {st.median(R['pure']):.0f} µV/h (n={len(R['pure'])}) | {st.median(R['koh']):.0f} µV/h (n={len(R['koh'])}) |")
chk(T2F, 'pw dur', f"| durability test, median length | {st.median(H['pure']):.0f} h ({sum(x >= 1000 for x in H['pure'])} tests ≥ 1,000 h) | "
                   f"{st.median(H['koh']):.0f} h ({sum(x >= 1000 for x in H['koh'])} tests ≥ 1,000 h) |")
pw = sorted(((f(r['duration_h']), f(r['j_held_A_cm2']), r['temperature_C'], r['rate_uV_h']) for r in du
             if r['electrolyte'] == 'pure_water' and r['duration_h']), key=lambda x: (-x[0], x[3] == ''))
seen, longest = set(), []
for d, jh, T, rt in pw:
    if d in seen: continue
    seen.add(d)
    longest.append(f"{d:,.0f} h" + (f" at {jh:g} A cm-2" if jh is not None else '') + (f"/{float(T):.0f} °C" if T else '') + (f" ({float(rt):g} µV/h stated)" if rt else ''))
chk(T2F, 'pw longest', 'Longest pure-water results: ' + ', '.join(longest[:3]))
allr = [r for r in du if r['rate_uV_h'] and f(r['rate_uV_h']) > 0 and r['duration_h']]
band = lambda lo, hi: [f(r['rate_uV_h']) for r in allr if lo <= f(r['duration_h']) < hi]
b1, b2, b3 = band(0, 200), band(200, 1000), band(1000, 1e9)
chk(T2F, 'rate bands', f"<200 h tests {st.median(b1):,.0f} µV/h (n={len(b1)}); 200–1,000 h {st.median(b2):,.0f} µV/h (n={len(b2)}); "
                       f"≥1,000 h {st.median(b3):,.0f} µV/h (n={len(b3)})")
rr = [f(r['rate_uV_h']) for r in du if r['rate_uV_h'] and f(r['rate_uV_h']) > 0]
chk(T2F, 'kpi', f"{sum(x <= 17.1 for x in rr)} of {len(rr)} stated rates meet it, {sum(x <= 9.5 for x in rr)} meet the 2030 level")
chk(T2F, 'qual', f"Qualitative-only durability claims: {pc(sum(int(r['qualitative_only']) for r in du) / len(du))} in full texts "
                 f"(vs ~{pc(sum(int(r['qualitative_only']) for r in t1d) / len(t1d))} in abstracts)")
chk(T2F, 'stacks', f"{len({r['uid'] for r in t2t if r['test_level'] == 'stack'})} papers with stack tests in the full-text set")

# ------------------------------------------------------------------ compare with the FINDINGS files
texts = {fn: open(os.path.join(HERE, fn), encoding='utf-8').read() for fn in (T1F, T2F)}
bad = 0
for fn, label, phrase in C:
    if phrase in texts[fn]:
        print(f'OK   {fn[9:14]} {label}' + (f'\n       {phrase}' if '--print' in sys.argv else ''))
        continue
    bad += 1
    lines = texts[fn].splitlines()
    best = max(lines, key=lambda l: difflib.SequenceMatcher(None, l, phrase).ratio())
    print(f'DIFF {fn[9:14]} {label}\n       expected: {phrase}\n       closest : {best.strip()}')
print(f'{len(C) - bad} of {len(C)} FINDINGS numbers/phrases match the recomputed values; {bad} differ.')
sys.exit(1 if bad else 0)
