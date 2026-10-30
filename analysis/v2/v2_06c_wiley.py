"""v2 step 6c - Wiley text-and-data-mining (TDM) API only: fetch full-text PDFs for selected Wiley papers (DOI 10.1002/...)
that do not have a full text yet. Faster than 6b because it skips all other sources.

Run on YOUR laptop, on the TU/e network or VPN, with WILEY_TDM_TOKEN set (never written to disk):
  python analysis/v2/v2_06c_wiley.py --test      # try 1 paper and print the exact API response
  python analysis/v2/v2_06c_wiley.py            # all missing Wiley papers (resumable; ~1 request per second)
  python analysis/v2/v2_06c_wiley.py --limit 50 # only the first 50, highest priority first

Order: papers are processed by the priority score of v2_06b (stack, long durability, pure water, citations) so an interrupted
run still has the most useful papers. Results are appended to data/raw/fulltext/fetch_log.csv (source = wiley_tdm);
PDFs go to data/raw/fulltext/pdf/<id>.pdf. Afterwards run v2_07_extract_text.py.
"""
import argparse, csv, os, sys, time, urllib.parse, importlib.util
csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
f6 = load('v2_06_fetch_fulltext'); f6b = load('v2_06b_fill_gaps')
ROOT, OUT, LOG, CORPUS = f6.ROOT, f6.OUT, f6.LOG, f6.CORPUS
API = 'https://api.wiley.com/onlinelibrary/tdm/v1/articles/'

def fetch(s, tok, doi, path):
    r = s.get(API + urllib.parse.quote(doi, safe=''), headers={'Wiley-TDM-Client-Token': tok}, timeout=90, allow_redirects=True)
    if r.status_code == 200 and r.content.startswith(b'%PDF'):
        open(path, 'wb').write(r.content); return 'ok', r
    return ('not_pdf' if r.status_code == 200 else f'http_{r.status_code}'), r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--test', action='store_true'); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    tok = os.environ.get('WILEY_TDM_TOKEN')
    if not tok: sys.exit('WILEY_TDM_TOKEN is not set in this terminal (setx, then open a NEW terminal).')
    corpus = {r['uid']: r for r in csv.DictReader(open(CORPUS, encoding='utf-8'))}
    sel = [u for u in f6.selection('corpus_haiku_v2') if corpus[u]['doi'].startswith('10.1002/')]
    ok = set()
    if os.path.exists(LOG): ok = {r['uid'] for r in csv.DictReader(open(LOG, encoding='utf-8')) if r['result'] == 'ok'}
    todo_uids = [u for u in sel if u not in ok and not os.path.exists(os.path.join(OUT, 'pdf', f6.safe_id(u) + '.pdf'))]
    order = {r['uid']: r['rank'] for r in f6b.priority_rows(todo_uids, corpus)}
    todo = sorted(todo_uids, key=lambda u: order.get(u, 10**6))
    if a.limit: todo = todo[:a.limit]
    print(f'Wiley papers selected: {len(sel)}, already have full text: {len(sel) - len(todo_uids)}, to try now: {len(todo)}')
    s = f6.session(); os.makedirs(os.path.join(OUT, 'pdf'), exist_ok=True)
    if a.test:
        doi = corpus[todo[0]]['doi']; res, r = fetch(s, tok, doi, os.path.join(OUT, 'pdf', f6.safe_id(todo[0]) + '.pdf'))
        print(f'test {doi}: {res}; HTTP {r.status_code}; content-type {r.headers.get("content-type")}; {len(r.content):,} bytes')
        if res != 'ok': print('response start:', r.content[:300])
        print('-> works, run without --test' if res == 'ok' else
              '-> 401/403: token not accepted or not on the TU/e network/VPN; 404: DOI not served by Wiley TDM')
        return
    new = not os.path.exists(LOG); n_ok = 0; fails = 0
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['uid', 'doi', 'oa_status', 'source', 'url', 'result', 'file', 'time'])
        if new: w.writeheader()
        for i, u in enumerate(todo, 1):
            doi = corpus[u]['doi']; path = os.path.join(OUT, 'pdf', f6.safe_id(u) + '.pdf')
            try:
                res, r = fetch(s, tok, doi, path)
            except Exception as e:
                res = 'error_' + type(e).__name__
            if res == 'http_429': time.sleep(30)
            w.writerow(dict(uid=u, doi=doi, oa_status='', source='wiley_tdm', url=API + doi, result=res,
                            file=os.path.relpath(path, ROOT) if res == 'ok' else '', time=time.strftime('%H:%M:%S'))); f.flush()
            n_ok += res == 'ok'; fails = 0 if res == 'ok' else fails + 1
            if fails >= 25 and n_ok == 0: sys.exit('25 failures in a row and no success: stop. Run with --test to see the response.')
            if i % 25 == 0: print(f'  {i}/{len(todo)} tried, {n_ok} PDFs', flush=True)
            time.sleep(1.1)
    print(f'done: {n_ok} new Wiley PDFs of {len(todo)} tried. Next: python analysis\\v2\\v2_07_extract_text.py')

if __name__ == '__main__':
    main()
