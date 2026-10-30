"""v2 step 6 - fetch full texts for the papers selected by the abstract-level LLM screen.

Run on YOUR laptop (normal internet; TU/e network or VPN needed only for the optional Elsevier route).
  python analysis/v2/v2_06_fetch_fulltext.py --dry-run          # show selection size + open-access availability, download nothing
  python analysis/v2/v2_06_fetch_fulltext.py                    # download (resumable: already fetched papers are skipped)

Selection (documented in data/processed/v2/llm/README.md): corpus_haiku_v2 output with in_scope == true AND
highest_test_level in {single_cell, stack, unclear}.
Sources, in order of preference:
  1. Open-access PDF links listed by OpenAlex (best_oa_location first, then other OA locations: publisher OA, repositories).
  2. OPTIONAL Elsevier Article Retrieval API (plain-text full text for TU/e-subscribed articles, text-and-data-mining for
     non-commercial research). Used only if ELSEVIER_API_KEY is set and the DOI is Elsevier (10.1016/...). Needs the TU/e network.
Optional env vars: OPENALEX_API_KEY (your free OpenAlex key, raises rate limits), CONTACT_EMAIL (sent in the User-Agent, polite
crawling), ELSEVIER_API_KEY. Keys are read from the environment only and never written to disk.
Outputs: data/raw/fulltext/pdf/<id>.pdf, data/raw/fulltext/elsevier_txt/<id>.txt, data/raw/fulltext/fetch_log.csv
"""
import argparse, csv, json, os, re, sys, time, urllib.parse
csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
LLM = os.path.join(ROOT, 'data', 'processed', 'v2', 'llm')
CORPUS = os.path.join(ROOT, 'data', 'processed', 'v2', 'corpus_screened.csv')
OUT = os.path.join(ROOT, 'data', 'raw', 'fulltext')
LOG = os.path.join(OUT, 'fetch_log.csv')
LEVELS = {'single_cell', 'stack', 'unclear'}

def safe_id(uid): return re.sub(r'[^A-Za-z0-9._-]+', '_', uid)[:150]

def selection(run):
    m = {r['custom_id']: r['uid'] for r in csv.DictReader(open(os.path.join(LLM, f'{run}_requests_map.csv'), encoding='utf-8'))}
    sel = []
    for line in open(os.path.join(LLM, f'{run}_results.jsonl'), encoding='utf-8'):
        r = json.loads(line); o = r.get('output') or {}
        if o.get('in_scope') and o.get('highest_test_level') in LEVELS:
            sel.append(m[r['custom_id']])
    return sel

def session():
    import requests
    s = requests.Session()
    mail = os.environ.get('CONTACT_EMAIL', '')
    s.headers['User-Agent'] = f'Mozilla/5.0 (TU/e student research, text-and-data-mining{"; mailto:" + mail if mail else ""})'
    return s

def openalex_lookup(s, dois):
    """Return {doi: [pdf urls in preference order]} using OpenAlex, 50 DOIs per call."""
    key = os.environ.get('OPENALEX_API_KEY'); urls = {}
    for i in range(0, len(dois), 50):
        chunk = dois[i:i + 50]
        flt = 'doi:' + '|'.join('https://doi.org/' + d for d in chunk)
        params = {'filter': flt, 'per_page': 50, 'select': 'doi,open_access,best_oa_location,locations'}
        if key: params['api_key'] = key
        for attempt in range(4):
            r = s.get('https://api.openalex.org/works', params=params, timeout=60)
            if r.status_code == 429: time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status(); break
        for w in r.json().get('results', []):
            d = (w.get('doi') or '').lower().replace('https://doi.org/', '')
            cand = []
            b = w.get('best_oa_location') or {}
            if b.get('pdf_url'): cand.append(b['pdf_url'])
            for loc in w.get('locations') or []:
                if loc.get('is_oa') and loc.get('pdf_url') and loc['pdf_url'] not in cand: cand.append(loc['pdf_url'])
            urls[d] = dict(urls=cand, oa_status=(w.get('open_access') or {}).get('oa_status', ''))
        time.sleep(0.2 if key else 1.0)
        print(f'  OpenAlex lookup {min(i + 50, len(dois))}/{len(dois)}', flush=True)
    return urls

def try_pdf(s, url, path):
    try:
        r = s.get(url, timeout=45, allow_redirects=True, stream=True)
        if r.status_code != 200: return f'http_{r.status_code}'
        data = r.raw.read(40 * 1024 * 1024, decode_content=True)
        if not data.startswith(b'%PDF'): return 'not_pdf'
        open(path, 'wb').write(data); return 'ok'
    except Exception as e:
        return 'error_' + type(e).__name__

def try_elsevier(s, doi, path):
    key = os.environ.get('ELSEVIER_API_KEY')
    if not key or not doi.startswith('10.1016/'): return 'skipped'
    try:
        r = s.get(f'https://api.elsevier.com/content/article/doi/{urllib.parse.quote(doi)}',
                  headers={'X-ELS-APIKey': key, 'Accept': 'text/plain'}, params={'view': 'FULL'}, timeout=60)
        if r.status_code != 200: return f'http_{r.status_code}'
        txt = r.text
        if len(txt) < 8000: return 'abstract_only_not_entitled'   # not entitled -> API returns metadata/abstract only
        open(path, 'w', encoding='utf-8').write(txt); return 'ok'
    except Exception as e:
        return 'error_' + type(e).__name__

def diagnose_elsevier():
    import requests
    key = os.environ.get('ELSEVIER_API_KEY', '')
    print('ELSEVIER_API_KEY:', f'set ({len(key)} characters)' if key else 'NOT set')
    try:
        print('Your public IP as seen from the internet:', requests.get('https://api.ipify.org', timeout=15).text)
    except Exception as e:
        print('IP lookup failed:', e)
    doi = '10.1016/j.ijhydene.2023.05.076'      # a real Elsevier article from the corpus, used only as a probe
    A = 'https://api.elsevier.com/content/article'
    for label, url, params, acc in [
            ('metadata (key check)', 'https://api.elsevier.com/content/search/scopus', {'query': 'DOI(' + doi + ')'}, 'application/json'),
            ('entitlement endpoint', f'{A}/entitlement/doi/{doi}', {}, 'application/json'),
            ('full text, plain, view=FULL', f'{A}/doi/{doi}', {'view': 'FULL'}, 'text/plain'),
            ('full text, plain, default view', f'{A}/doi/{doi}', {}, 'text/plain'),
            ('full text, XML, default view', f'{A}/doi/{doi}', {}, 'text/xml')]:
        r = requests.get(url, headers={'X-ELS-APIKey': key, 'Accept': acc}, params=params, timeout=60)
        print(f'\n{label}: HTTP {r.status_code}, {len(r.text):,} chars')
        for h in ['X-ELS-Status', 'X-ELS-APIKey-Status', 'X-RateLimit-Remaining']:
            if h in r.headers: print(f'  {h}: {r.headers[h]}')
        if r.status_code != 200 or 'entitlement' in url: print('  response:', r.text[:400].replace('\n', ' '))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='corpus_haiku_v2')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--test-elsevier', action='store_true', help='diagnose the Elsevier route and exit')
    a = ap.parse_args()
    try:
        import requests  # noqa
    except ImportError:
        sys.exit('Missing package. Run: python -m pip install requests')
    if a.test_elsevier:
        diagnose_elsevier(); return
    corpus = {r['uid']: r for r in csv.DictReader(open(CORPUS, encoding='utf-8'))}
    sel = selection(a.run)
    with_doi = [u for u in sel if corpus[u]['doi']]
    print(f'selected {len(sel)} papers from {a.run} ({len(sel) - len(with_doi)} without DOI are skipped)')
    s = session()
    lk = openalex_lookup(s, [corpus[u]['doi'] for u in with_doi])
    n_oa = sum(1 for u in with_doi if lk.get(corpus[u]['doi'], {}).get('urls'))
    n_els = sum(1 for u in with_doi if corpus[u]['doi'].startswith('10.1016/') and not lk.get(corpus[u]['doi'], {}).get('urls'))
    print(f'{n_oa} have at least one open-access PDF link; {n_els} closed Elsevier papers (reachable only via ELSEVIER_API_KEY)')
    for k in ['OPENALEX_API_KEY', 'CONTACT_EMAIL', 'ELSEVIER_API_KEY']:
        print(f'  {k}: {"set" if os.environ.get(k) else "NOT set (open a new terminal after setx)"}')
    if os.environ.get('ELSEVIER_API_KEY'):
        test = next((corpus[u]['doi'] for u in with_doi if corpus[u]['doi'].startswith('10.1016/')
                     and not lk.get(corpus[u]['doi'], {}).get('urls')), None)
        if test:
            import tempfile
            tmp = os.path.join(tempfile.gettempdir(), 'els_test.txt')
            res = try_elsevier(s, test, tmp)
            size = os.path.getsize(tmp) if res == 'ok' else 0
            print(f'  Elsevier test on {test}: {res}' + (f' ({size:,} chars of full text) -> Elsevier route works' if res == 'ok' else
                  ' -> if abstract_only_not_entitled or http_401/403: connect the TU/e VPN (or eduroam on campus) and retry'))
            if res == 'ok': os.remove(tmp)
    if a.dry_run: return
    os.makedirs(os.path.join(OUT, 'pdf'), exist_ok=True); os.makedirs(os.path.join(OUT, 'elsevier_txt'), exist_ok=True)
    done = set()
    if os.path.exists(LOG):
        done = {r['uid'] for r in csv.DictReader(open(LOG, encoding='utf-8')) if r['result'] == 'ok'}
    new = not os.path.exists(LOG)
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['uid', 'doi', 'oa_status', 'source', 'url', 'result', 'file', 'time'])
        if new: w.writeheader()
        for i, u in enumerate(with_doi, 1):
            if u in done: continue
            doi = corpus[u]['doi']; info = lk.get(doi, {'urls': [], 'oa_status': ''}); sid = safe_id(u)
            res, src, url, fpath = 'no_source', '', '', ''
            for url in info['urls']:
                fpath = os.path.join(OUT, 'pdf', sid + '.pdf'); res = try_pdf(s, url, fpath); src = 'openalex_oa_pdf'
                time.sleep(1.0)
                if res == 'ok': break
            if res != 'ok':
                p2 = os.path.join(OUT, 'elsevier_txt', sid + '.txt'); r2 = try_elsevier(s, doi, p2)
                if r2 != 'skipped':
                    res, src, url, fpath = r2, 'elsevier_api', f'api.elsevier.com/{doi}', p2; time.sleep(0.5)
            w.writerow(dict(uid=u, doi=doi, oa_status=info['oa_status'], source=src, url=url, result=res,
                            file=os.path.relpath(fpath, ROOT) if res == 'ok' else '', time=time.strftime('%H:%M:%S')))
            f.flush()
            if i % 25 == 0: print(f'  {i}/{len(with_doi)} processed', flush=True)
    rows = list(csv.DictReader(open(LOG, encoding='utf-8')))
    from collections import Counter
    print('results:', Counter((r['source'], r['result']) for r in rows).most_common(12))

if __name__ == '__main__':
    main()
