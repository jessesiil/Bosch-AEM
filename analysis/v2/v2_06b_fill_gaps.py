"""v2 step 6b - try to close the full-text gap left by v2_06_fetch_fulltext.py, then rank what is still missing.

Run on YOUR laptop AFTER v2_06 has finished (TU/e network/VPN for the Wiley route):
  python analysis/v2/v2_06b_fill_gaps.py --dry-run     # how many are missing, which routes are configured
  python analysis/v2/v2_06b_fill_gaps.py               # try extra sources, then write the manual-download priority list

Extra (legal) sources, tried in order for every selected paper without a full text yet:
  1. Semantic Scholar 'openAccessPdf' (no key needed; finds author manuscripts / preprints OpenAlex sometimes misses)
  2. CORE (aggregator of university repositories; optional free key CORE_API_KEY from core.ac.uk/services/api)
  3. Wiley text-and-data-mining API for 10.1002/ DOIs (optional WILEY_TDM_TOKEN: log in at onlinelibrary.wiley.com on the
     TU/e network, then get the token at https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining)
Results are appended to data/raw/fulltext/fetch_log.csv (same format as step 6), so step 7 picks them up automatically.

Whatever is still missing goes to data/processed/v2/manual_download_priority.csv, ranked by how much the paper matters for the
analysis (from the abstract-level LLM output): stack test, long durability test, pure-water feed, citations. Download the top
ones yourself through the TU/e library (normal personal use) and save them as data/raw/fulltext/manual/<file_id>.pdf;
step 7 includes that folder.
"""
import argparse, csv, json, os, sys, time, urllib.parse, importlib.util
csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('f6', os.path.join(HERE, 'v2_06_fetch_fulltext.py'))
f6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f6)
ROOT, OUT, LOG, CORPUS, LLM = f6.ROOT, f6.OUT, f6.LOG, f6.CORPUS, f6.LLM
PRIORITY = os.path.join(ROOT, 'data', 'processed', 'v2', 'manual_download_priority.csv')

def semantic_scholar(s, dois):
    found = {}
    for i in range(0, len(dois), 400):
        chunk = dois[i:i + 400]
        for attempt in range(5):
            r = s.post('https://api.semanticscholar.org/graph/v1/paper/batch', params={'fields': 'openAccessPdf'},
                       json={'ids': ['DOI:' + d for d in chunk]}, timeout=90)
            if r.status_code == 429: time.sleep(10 * (attempt + 1)); continue
            break
        if r.status_code != 200:
            print('  Semantic Scholar HTTP', r.status_code); continue
        for d, p in zip(chunk, r.json()):
            url = ((p or {}).get('openAccessPdf') or {}).get('url')
            if url: found[d] = url
        time.sleep(3)
        print(f'  Semantic Scholar lookup {min(i + 400, len(dois))}/{len(dois)}', flush=True)
    return found

def core_url(s, doi):
    key = os.environ.get('CORE_API_KEY')
    if not key: return None
    try:
        r = s.get('https://api.core.ac.uk/v3/search/works', params={'q': f'doi:"{doi}"', 'limit': 3},
                  headers={'Authorization': f'Bearer {key}'}, timeout=60)
        if r.status_code != 200: return None
        for w in r.json().get('results', []):
            if w.get('downloadUrl'): return w['downloadUrl']
    except Exception:
        return None

def wiley(s, doi, path):
    tok = os.environ.get('WILEY_TDM_TOKEN')
    if not tok or not doi.startswith('10.1002/'): return 'skipped'
    try:
        r = s.get('https://api.wiley.com/onlinelibrary/tdm/v1/articles/' + urllib.parse.quote(doi, safe=''),
                  headers={'Wiley-TDM-Client-Token': tok}, timeout=90, allow_redirects=True)
        if r.status_code != 200: return f'http_{r.status_code}'
        if not r.content.startswith(b'%PDF'): return 'not_pdf'
        open(path, 'wb').write(r.content); return 'ok'
    except Exception as e:
        return 'error_' + type(e).__name__

def priority_rows(missing, corpus):
    m = {r['custom_id']: r['uid'] for r in csv.DictReader(open(os.path.join(LLM, 'corpus_haiku_v2_requests_map.csv'), encoding='utf-8'))}
    out = {}
    for line in open(os.path.join(LLM, 'corpus_haiku_v2_results.jsonl'), encoding='utf-8'):
        r = json.loads(line); out[m[r['custom_id']]] = r.get('output') or {}
    rows = []
    for u in missing:
        o, c = out.get(u, {}), corpus[u]; tests = o.get('tests') or []
        dur = max([(t.get('durability') or {}).get('duration_h') or 0 for t in tests] or [0])
        stack = o.get('highest_test_level') == 'stack' or any(t.get('test_level') == 'stack' for t in tests)
        pure = any(t.get('electrolyte') == 'pure_water' for t in tests)
        cites = max(int(c.get('cited_oa') or 0), int(c.get('cited_sc') or 0) if (c.get('cited_sc') or '').isdigit() else 0)
        score = 3 * stack + (3 if dur >= 1000 else 2 if dur >= 300 else 1 if dur > 0 else 0) + 2 * pure + (1 if cites >= 50 else 0)
        rows.append(dict(score=score, uid=u, file_id=f6.safe_id(u), doi=c['doi'], link=f"https://doi.org/{c['doi']}",
                         year=c['year'], source_journal=c['source'], title=c['title'], stack=int(stack),
                         max_durability_h_in_abstract=dur, pure_water=int(pure), citations=cites))
    rows.sort(key=lambda r: (-r['score'], -r['citations']))
    for i, r in enumerate(rows, 1): r['rank'] = i
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
    if not os.path.exists(LOG): sys.exit('Run v2_06_fetch_fulltext.py first.')
    corpus = {r['uid']: r for r in csv.DictReader(open(CORPUS, encoding='utf-8'))}
    sel = [u for u in f6.selection('corpus_haiku_v2') if corpus[u]['doi']]
    ok = {r['uid'] for r in csv.DictReader(open(LOG, encoding='utf-8')) if r['result'] == 'ok'}
    manual_dir = os.path.join(OUT, 'manual'); os.makedirs(manual_dir, exist_ok=True)
    ok |= {u for u in sel if os.path.exists(os.path.join(manual_dir, f6.safe_id(u) + '.pdf'))}
    missing = [u for u in sel if u not in ok]
    print(f'selected {len(sel)}, have full text {len(sel) - len(missing)}, missing {len(missing)}')
    for k in ['CORE_API_KEY', 'WILEY_TDM_TOKEN']:
        print(f'  {k}: {"set" if os.environ.get(k) else "not set (route skipped)"}')
    if not a.dry_run:
        s = f6.session()
        ss = semantic_scholar(s, [corpus[u]['doi'] for u in missing])
        print(f'  Semantic Scholar has an open PDF link for {len(ss)} of them')
        with open(LOG, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['uid', 'doi', 'oa_status', 'source', 'url', 'result', 'file', 'time'])
            for i, u in enumerate(missing, 1):
                doi = corpus[u]['doi']; pdf = os.path.join(OUT, 'pdf', f6.safe_id(u) + '.pdf'); res, src, url = 'no_source', '', ''
                for src, url in [('semantic_scholar', ss.get(doi)), ('core', core_url(s, doi))]:
                    if url:
                        res = f6.try_pdf(s, url, pdf); time.sleep(1.0)
                        if res == 'ok': break
                if res != 'ok':
                    r2 = wiley(s, doi, pdf)
                    if r2 != 'skipped': res, src, url = r2, 'wiley_tdm', 'api.wiley.com/' + doi; time.sleep(1.0)
                if src:
                    w.writerow(dict(uid=u, doi=doi, oa_status='', source=src, url=url or '', result=res,
                                    file=os.path.relpath(pdf, ROOT) if res == 'ok' else '', time=time.strftime('%H:%M:%S')))
                    f.flush()
                if res == 'ok': ok.add(u)
                if i % 50 == 0: print(f'  {i}/{len(missing)} tried, {len(ok)} total with full text', flush=True)
        missing = [u for u in sel if u not in ok]
    rows = priority_rows(missing, corpus)
    if rows:
        with open(PRIORITY, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['rank', 'score', 'file_id', 'doi', 'link', 'year', 'source_journal', 'title', 'stack',
                                              'max_durability_h_in_abstract', 'pure_water', 'citations', 'uid'])
            w.writeheader(); w.writerows(rows)
    print(f'still missing: {len(missing)}; priority list -> {os.path.relpath(PRIORITY, ROOT)} '
          f'(score >= 5: {sum(1 for r in rows if r["score"] >= 5)}, score >= 3: {sum(1 for r in rows if r["score"] >= 3)})')

if __name__ == '__main__':
    main()
