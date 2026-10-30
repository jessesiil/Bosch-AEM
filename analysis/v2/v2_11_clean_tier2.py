"""v2 step 11 - clean the full-text (tier 2) LLM extraction, same checks as step 8 but against the full text.
  python analysis/v2/v2_11_clean_tier2.py                    # combines all fulltext_haiku* runs (not the sonnet sample)
  python analysis/v2/v2_11_clean_tier2.py --runs fulltext_sonnet_sample40 --out tier2_sonnet_sample40
Quote check (match_level column): 'exact' = quote occurs in full text or abstract after normalising unicode/case/whitespace;
'loose' = occurs after reducing both to letters+digits only (PDF text often mangles units, e.g. 'A cm-2' -> 'Acm  2');
'numbers_nearby' = not found as text, but all numbers of the quote occur within a 400-character window (likely a paraphrase;
kept but flagged, excluded from frontier charts); otherwise dropped. Number-in-quote and sanity ranges as in step 8. If a paper occurs in several runs, the
latest successful run wins. One record per paper as in step 8 (v2_dedupe.py; added 2026-10-22, T-058): cover records dropped,
duplicates collapsed after the checks (publisher > most kept values); cleaning_log cover_records_dropped, duplicate_papers_dropped.
Outputs: data/processed/v2/<out>/{tests,perf_points,durability}.csv, cleaning_log.json (default out = tier2)."""
import argparse, csv, json, os, re, collections, importlib.util
from v2_dedupe import collapse_duplicate_papers, keeper_rank, is_cover_record
csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('c8', os.path.join(HERE, 'v2_08_clean_tier1.py'))
c8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c8)
ROOT = c8.ROOT; LLM = c8.LLM

ALNUM = lambda t: re.sub(r'[^a-z0-9]', '', t.lower())
NUMS = re.compile(r'\d+(?:\.\d+)?')
def match_level(q, src_norm, src_alnum, raw):
    if not q: return None
    if c8.norm(q) in src_norm: return 'exact'
    if len(ALNUM(q)) >= 12 and ALNUM(q) in src_alnum: return 'loose'
    ns = [n for n in NUMS.findall(q) if len(n) >= 2 or '.' in n]
    if not ns: return None
    anchor = max(ns, key=len)
    for m in re.finditer(re.escape(anchor), raw):
        win = raw[max(0, m.start() - 400): m.end() + 400]
        if all(n in win for n in ns): return 'numbers_nearby'
    return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', default=''); ap.add_argument('--out', default='tier2'); a = ap.parse_args()
    runs = a.runs.split(',') if a.runs else sorted(f[:-14] for f in os.listdir(LLM)
                                                    if f.startswith('fulltext_haiku') and f.endswith('_results.jsonl'))
    corpus = {r['uid']: r for r in csv.DictReader(open(os.path.join(ROOT, 'data/processed/v2/corpus_screened.csv'), encoding='utf-8'))}
    files = {r['uid']: os.path.join(ROOT, *r['file'].replace('\\', '/').split('/'))
             for r in csv.DictReader(open(os.path.join(ROOT, 'data/processed/v2/fulltext_index.csv'), encoding='utf-8')) if r['status'] == 'ok'}
    outputs = {}
    for run in runs:
        cmap = {r['custom_id']: r['uid'] for r in csv.DictReader(open(os.path.join(LLM, f'{run}_requests_map.csv'), encoding='utf-8'))}
        for line in open(os.path.join(LLM, f'{run}_results.jsonl'), encoding='utf-8'):
            r = json.loads(line)
            if r.get('output'): outputs[cmap[r['custom_id']]] = (run, r['output'])
    log = collections.Counter(); tests, pps, durs = [], [], []
    log['runs'] = len(runs); log['papers_with_output'] = len(outputs)
    covers = {u for u in outputs if is_cover_record(corpus[u]['title'], corpus[u]['abstract'])}
    glog, plog = log, collections.defaultdict(collections.Counter)   # per-paper counts; summed over the kept papers below
    for uid, (run, o) in outputs.items():
        if uid in covers: continue
        log = plog[uid]
        c = corpus[uid]; raw = c['title'] + ' ' + c['abstract'] + ' ' + open(files[uid], encoding='utf-8').read()
        src = c8.norm(raw); src_al = ALNUM(raw)
        meta = dict(uid=uid, doi=c['doi'], year=c['year'], title=c['title'], journal=c['source'], run=run,
                    in_scope=int(bool(o.get('in_scope'))), paper_level=o.get('highest_test_level'), focus='; '.join(o.get('focus_components') or []),
                    cited_values_present=int(bool(o.get('values_from_cited_work_present'))))
        log['papers'] += 1
        if not o.get('in_scope'): log['papers_out_of_scope'] += 1; continue
        for k, t in enumerate(o.get('tests') or []):
            tid = f'{uid}#T{k+1}'; ev = t.get('evidence_conditions', '')
            ev_ok = all(match_level(q.strip(), src, src_al, raw) in ('exact', 'loose') for q in ev.split('||') if q.strip()) if ev.strip() else False
            tests.append(dict(meta, test_uid=tid, test_level=t.get('test_level'), electrolyte=t.get('electrolyte'),
                              conc_M=t.get('electrolyte_concentration_M'), temperature_C=t.get('temperature_C'), area_cm2=t.get('active_area_cm2'),
                              membrane=t.get('membrane'), anode=t.get('anode_catalyst'), cathode=t.get('cathode_catalyst'),
                              pgm_mg_cm2=t.get('pgm_loading_mg_cm2'), conditions_quote_verified=int(ev_ok)))
            log['tests'] += 1; log['tests_conditions_quote_verified'] += ev_ok
            for p in t.get('performance_points') or []:
                log['pp_raw'] += 1; j, v, q = p.get('current_density_A_cm2'), p.get('cell_voltage_V'), p.get('quote', '')
                ml = match_level(q, src, src_al, raw)
                if not ml: log['pp_drop_quote_not_found'] += 1; continue
                log[f'pp_match_{ml}'] += 1
                if not (c8.num_in(j, q) and c8.num_in(v, q)): log['pp_drop_number_not_in_quote'] += 1; continue
                if j is None or v is None or not (0.01 <= j <= 20) or not (1.30 <= v <= 3.0): log['pp_drop_sanity'] += 1; continue
                pps.append(dict(meta, test_uid=tid, electrolyte=t.get('electrolyte'), conc_M=t.get('electrolyte_concentration_M'),
                                temperature_C=t.get('temperature_C'), test_level=t.get('test_level'), j_A_cm2=j, V_cell=v,
                                eff_HHV=round(1.481 / v, 4), location=p.get('location', ''), match_level=ml, quote=q))
                log['pp_kept'] += 1
            d = t.get('durability')
            if d:
                log['dur_raw'] += 1; q = d.get('quote', '') or d.get('qualitative_claim', '')
                ml = match_level(q, src, src_al, raw)
                if not ml: log['dur_drop_quote_not_found'] += 1; continue
                log[f'dur_match_{ml}'] += 1
                h = d.get('duration_h')
                if h is not None and not c8.num_in(h, q, (1, 1 / 24, 1 / 60, 60, 24)): log['dur_drop_hours_not_in_quote'] += 1; continue
                if h is not None and not (0.1 <= h <= 100000): log['dur_drop_sanity'] += 1; continue
                v0, v1, rate = d.get('voltage_start_V'), d.get('voltage_end_V'), d.get('degradation_rate_uV_h')
                rate_src = 'stated' if rate is not None else ''
                if rate is not None and not c8.num_in(rate, q): rate, rate_src = None, ''; log['dur_rate_not_in_quote'] += 1
                if rate is None and v0 and v1 and h and 1.3 <= v0 <= 3 and 1.3 <= v1 <= 3:
                    rate, rate_src = round((v1 - v0) / h * 1e6, 2), 'computed'
                durs.append(dict(meta, test_uid=tid, electrolyte=t.get('electrolyte'), conc_M=t.get('electrolyte_concentration_M'),
                                 temperature_C=t.get('temperature_C'), test_level=t.get('test_level'), duration_h=h, mode=d.get('operation_mode'),
                                 j_held_A_cm2=d.get('held_current_density_A_cm2'), V_start=v0, V_end=v1, rate_uV_h=rate, rate_source=rate_src,
                                 qualitative_only=int(rate is None and not v0 and not v1), qualitative_claim=d.get('qualitative_claim', ''),
                                 location=d.get('location', ''), match_level=ml, quote=q))
                log['dur_kept'] += 1
    # same paper under two ids -> keep one: publisher > more kept values > anie > in Scopus (v2_dedupe.py)
    n_kept = collections.Counter(r['uid'] for r in pps + durs)
    keep, dropped = collapse_duplicate_papers(((u, corpus[u]['title'], corpus[u]['abstract']) for u in outputs if u not in covers),
                                              keeper_rank(corpus, n_kept))
    tests, pps, durs = ([r for r in rows if r['uid'] in keep] for rows in (tests, pps, durs))
    log = glog + sum((plog[u] for u in sorted(keep)), collections.Counter())
    log.update(cover_records_dropped=len(covers), duplicate_papers_dropped=len(dropped))
    out = os.path.join(ROOT, 'data/processed/v2', a.out); os.makedirs(out, exist_ok=True)
    for name, rows in [('tests', tests), ('perf_points', pps), ('durability', durs)]:
        if rows:
            with open(os.path.join(out, f'{name}.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(log), open(os.path.join(out, 'cleaning_log.json'), 'w'), indent=1); print(json.dumps(dict(log), indent=1))

if __name__ == '__main__':
    main()
