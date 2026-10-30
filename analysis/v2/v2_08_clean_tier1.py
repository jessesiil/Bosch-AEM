"""v2 step 8 - clean the abstract-level (tier 1) LLM extraction for analysis.
Reads corpus_haiku_v2 results for the selected papers (in_scope AND test level single_cell/stack/unclear) and flattens them:
  tests.csv           one row per extracted cell/stack test condition (+ paper metadata)
  perf_points.csv     one row per current-density / cell-voltage pair
  durability.csv      one row per durability result
  cleaning_log.json   how many values were dropped/flagged and why
Checks (deterministic, no model involved):
  Q1 quote check   - the evidence quote must occur in the title+abstract (after normalising unicode, case and whitespace);
                     values whose quote is not found are dropped.
  Q2 number check  - each number must appear in its own quote (also accepting the mA / mV / mV h-1 form the model converted from).
  Q3 sanity ranges - cell voltage 1.30-3.00 V; current density 0.01-20 A cm-2; duration 0.1-100,000 h. Outside -> dropped and logged.
One record per paper (v2_dedupe.py; added 2026-10-22, T-058): Wiley cover/frontispiece records are dropped; the same article
under two ids (Angewandte ange/anie, repository copies, copied abstracts) is collapsed after the checks, keeping the publisher
record, then the one with most kept values. cleaning_log: cover_records_dropped, duplicate_papers_dropped; all other counts
refer to the kept papers only.
Derived:  voltage efficiency (HHV) = 1.481 V / cell voltage; degradation rate computed from start/end voltage and duration when the
          paper does not state one (flag rate_source = 'computed'); 'qualitative_only' = durability claim without any voltage or rate.
Output folder: data/processed/v2/tier1/"""
import csv, json, os, re, collections
from v2_dedupe import collapse_duplicate_papers, keeper_rank, is_cover_record
csv.field_size_limit(10**9)
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
LLM = os.path.join(ROOT, 'data/processed/v2/llm'); OUT = os.path.join(ROOT, 'data/processed/v2/tier1')
RUN = 'corpus_haiku_v2'; LEVELS = {'single_cell', 'stack', 'unclear'}

def norm(s):
    s = (s or '').lower()
    for a, b in [('−', '-'), ('–', '-'), ('—', '-'), ('μ', 'µ'), ('⁻', '-'), ('²', '2'), ('¹', '1'), ('·', ' '), (' ', ' '), ('\xa0', ' ')]:
        s = s.replace(a, b)
    return re.sub(r'\s+', '', s)

NUM = re.compile(r'\d+(?:[.,]\d+)?')
def nums(q): return [float(x.replace(',', '.')) for x in NUM.findall((q or '').replace('−', '-'))]
def num_in(v, q, factors=(1, 1000, 0.001)):
    if v is None: return True
    ns = nums(q)
    return any(abs(n - v * f) <= max(1e-6, 0.011 * abs(v * f)) for n in ns for f in factors)

def main():
    corpus = {r['uid']: r for r in csv.DictReader(open(os.path.join(ROOT, 'data/processed/v2/corpus_screened.csv'), encoding='utf-8'))}
    cmap = {r['custom_id']: r['uid'] for r in csv.DictReader(open(os.path.join(LLM, f'{RUN}_requests_map.csv'), encoding='utf-8'))}
    log = collections.Counter(); tests, pps, durs = [], [], []
    sel = []
    for line in open(os.path.join(LLM, f'{RUN}_results.jsonl'), encoding='utf-8'):
        r = json.loads(line); o = r.get('output') or {}
        if o.get('in_scope') and o.get('highest_test_level') in LEVELS: sel.append((cmap[r['custom_id']], o))
    covers = {u for u, _ in sel if is_cover_record(corpus[u]['title'], corpus[u]['abstract'])}
    plog = collections.defaultdict(collections.Counter)   # per-paper counts; summed over the kept papers below
    for uid, o in sel:
        if uid in covers: continue
        log = plog[uid]
        c = corpus[uid]; src = norm(c['title'] + ' ' + c['abstract'])
        meta = dict(uid=uid, doi=c['doi'], year=c['year'], title=c['title'], journal=c['source'], paper_level=o.get('highest_test_level'),
                    focus='; '.join(o.get('focus_components') or []))
        log['papers'] += 1
        for k, t in enumerate(o.get('tests') or []):
            tid = f'{uid}#T{k+1}'
            tests.append(dict(meta, test_uid=tid, test_level=t.get('test_level'), electrolyte=t.get('electrolyte'),
                              conc_M=t.get('electrolyte_concentration_M'), temperature_C=t.get('temperature_C'),
                              area_cm2=t.get('active_area_cm2'), membrane=t.get('membrane'), anode=t.get('anode_catalyst'),
                              cathode=t.get('cathode_catalyst'), pgm_mg_cm2=t.get('pgm_loading_mg_cm2')))
            log['tests'] += 1
            for p in t.get('performance_points') or []:
                log['pp_raw'] += 1
                j, v, q = p.get('current_density_A_cm2'), p.get('cell_voltage_V'), p.get('quote', '')
                if not q or norm(q) not in src: log['pp_drop_quote_not_found'] += 1; continue
                if not (num_in(j, q) and num_in(v, q)): log['pp_drop_number_not_in_quote'] += 1; continue
                if j is None or v is None or not (0.01 <= j <= 20) or not (1.30 <= v <= 3.0): log['pp_drop_sanity'] += 1; continue
                pps.append(dict(meta, test_uid=tid, electrolyte=t.get('electrolyte'), temperature_C=t.get('temperature_C'),
                                test_level=t.get('test_level'), j_A_cm2=j, V_cell=v, eff_HHV=round(1.481 / v, 4), quote=q))
                log['pp_kept'] += 1
            d = t.get('durability')
            if d:
                log['dur_raw'] += 1; q = d.get('quote', '') or d.get('qualitative_claim', '')
                if not q or norm(q) not in src: log['dur_drop_quote_not_found'] += 1; continue
                h = d.get('duration_h')
                if h is not None and not num_in(h, q, (1, 1 / 24, 1 / 60, 60, 24)): log['dur_drop_hours_not_in_quote'] += 1; continue
                if h is not None and not (0.1 <= h <= 100000): log['dur_drop_sanity'] += 1; continue
                v0, v1, rate = d.get('voltage_start_V'), d.get('voltage_end_V'), d.get('degradation_rate_uV_h')
                rate_src = 'stated' if rate is not None else ''
                if rate is not None and not num_in(rate, q): rate, rate_src = None, ''; log['dur_rate_not_in_quote'] += 1
                if rate is None and v0 and v1 and h and 1.3 <= v0 <= 3 and 1.3 <= v1 <= 3:
                    rate, rate_src = round((v1 - v0) / h * 1e6, 2), 'computed'
                durs.append(dict(meta, test_uid=tid, electrolyte=t.get('electrolyte'), temperature_C=t.get('temperature_C'),
                                 test_level=t.get('test_level'), duration_h=h, mode=d.get('operation_mode'),
                                 j_held_A_cm2=d.get('held_current_density_A_cm2'), V_start=v0, V_end=v1, rate_uV_h=rate,
                                 rate_source=rate_src, qualitative_only=int(rate is None and not v0 and not v1),
                                 qualitative_claim=d.get('qualitative_claim', ''), quote=q))
                log['dur_kept'] += 1
    # same paper under two ids -> keep one: publisher > more kept values > anie > in Scopus (v2_dedupe.py)
    n_kept = collections.Counter(r['uid'] for r in pps + durs)
    keep, dropped = collapse_duplicate_papers(((u, corpus[u]['title'], corpus[u]['abstract']) for u, _ in sel if u not in covers),
                                              keeper_rank(corpus, n_kept))
    tests, pps, durs = ([r for r in rows if r['uid'] in keep] for rows in (tests, pps, durs))
    log = sum((plog[u] for u in sorted(keep)), collections.Counter())
    log.update(papers_selected_incl_duplicates=len(sel), cover_records_dropped=len(covers), duplicate_papers_dropped=len(dropped))
    os.makedirs(OUT, exist_ok=True)
    for name, rows in [('tests', tests), ('perf_points', pps), ('durability', durs)]:
        with open(os.path.join(OUT, f'{name}.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(log), open(os.path.join(OUT, 'cleaning_log.json'), 'w'), indent=1); print(json.dumps(dict(log), indent=1))

if __name__ == '__main__':
    main()
