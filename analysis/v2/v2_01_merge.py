"""v2 step 1 - merge OpenAlex (AEM_broad pool, fetched 2026-10-08) and Scopus (combined v1+v2 query, exported 2026-10-14)
into one record per paper. Output: data/processed/v2/corpus.csv
Dedup key: normalised DOI; records without DOI are matched on normalised title + year (+-1)."""
import csv, json, re, os, collections
csv.field_size_limit(10**9)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
OA = os.path.join(ROOT, 'data/raw/openalex/bosch_openalex_part2.jsonl')
SC = os.path.join(ROOT, 'data/raw/scopus/scopus_export_2026-10-14.csv')
OUT = os.path.join(ROOT, 'data/processed/v2/corpus.csv')

def ndoi(d): return re.sub(r'^https?://(dx\.)?doi\.org/', '', (d or '').strip().lower())
def ntitle(t): return re.sub(r'[^a-z0-9]', '', (t or '').lower())
def good_ab(a):
    a = (a or '').strip()
    return a if a and not a.startswith('[No abstract') else ''
def strip_copyright(a):
    # Scopus abstracts often end with "© 2024 Elsevier ..." - remove so it does not pollute text rules
    return re.sub(r'\s*(©|\(c\)|copyright).*$', '', a, flags=re.I | re.S).strip()

recs = {}          # key -> record
title_index = {}   # (ntitle) -> key

def add(key, r):
    recs[key] = r
    if r['title_n']: title_index.setdefault(r['title_n'], key)

# --- OpenAlex
with open(OA, encoding='utf-8') as f:
    next(f)
    for line in f:
        o = json.loads(line)
        if 'AEM_broad' not in o.get('q', []): continue
        doi = ndoi(o.get('doi'))
        r = dict(uid=doi or o['id'], doi=doi, openalex_id=o['id'], scopus_eid='',
                 title=o.get('t') or '', title_n=ntitle(o.get('t')), year_oa=o.get('y') or '', date_oa=o.get('d') or '',
                 year_sc='', source=o.get('src') or '', doctype_oa=o.get('ty') or '', doctype_sc='',
                 ab_oa=good_ab(o.get('ab')), ab_sc='', kw_oa='; '.join(o.get('kw') or []) if isinstance(o.get('kw'), list) else (o.get('kw') or ''),
                 kw_sc='', countries='; '.join(o.get('ctry') or []), cited_oa=o.get('cit') or 0, cited_sc='', open_access_sc='',
                 in_openalex=1, in_scopus=0)
        add(r['uid'], r)

# --- Scopus
matched_title = 0
with open(SC, encoding='utf-8-sig') as f:
    for s in csv.DictReader(f):
        doi = ndoi(s['DOI']); tn = ntitle(s['Title'])
        key = doi if doi in recs else None
        if key is None and tn in title_index:
            cand = recs[title_index[tn]]
            try: ok = abs(int(cand['year_oa'] or 0) - int(s['Year'] or 0)) <= 1
            except ValueError: ok = False
            if ok: key = title_index[tn]; matched_title += 1
        fields = dict(scopus_eid=s['EID'], year_sc=s['Year'], doctype_sc=s['Document Type'],
                      ab_sc=strip_copyright(good_ab(s['Abstract'])),
                      kw_sc='; '.join(x for x in [s['Author Keywords'], s['Index Keywords']] if x),
                      cited_sc=s['Cited by'], open_access_sc=s['Open Access'], in_scopus=1)
        if key:
            recs[key].update(fields)
            if not recs[key]['source']: recs[key]['source'] = s['Source title']
        else:
            r = dict(uid=doi or s['EID'], doi=doi, openalex_id='', title=s['Title'], title_n=tn, year_oa='', date_oa='',
                     source=s['Source title'], doctype_oa='', ab_oa='', kw_oa='', countries='', cited_oa='', in_openalex=0)
            r.update(fields); add(r['uid'], r)

# --- derived fields
for r in recs.values():
    # year: OpenAlex publication year (online date) preferred; Scopus 2027 = volume year of online-first 2026 papers
    y = r['year_oa'] or r['year_sc']
    r['year'] = int(y) if str(y).isdigit() else ''
    r['year_flag'] = 'scopus_volume_year_2027->2026' if (not r['year_oa'] and str(r['year_sc']) == '2027') else ''
    if r['year_flag']: r['year'] = 2026
    # abstract: prefer the longer one (OpenAlex is truncated at 3000 chars)
    a, b = r['ab_oa'], r['ab_sc']
    r['abstract'] = a if len(a) >= len(b) else b
    r['abstract_source'] = ('none' if not r['abstract'] else 'openalex' if r['abstract'] == a else 'scopus')
    r['keywords'] = '; '.join(x for x in [r['kw_oa'], r['kw_sc']] if x)
    r['is_review'] = int(r['doctype_sc'] == 'Review' or r['doctype_oa'] == 'review')
    r['db'] = 'both' if r['in_openalex'] and r['in_scopus'] else 'openalex' if r['in_openalex'] else 'scopus'

cols = ['uid','doi','openalex_id','scopus_eid','db','year','year_flag','date_oa','title','source','is_review','abstract','abstract_source',
        'keywords','countries','cited_oa','cited_sc','open_access_sc']
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader()
    for r in sorted(recs.values(), key=lambda r: (r['year'] or 0, r['title'])): w.writerow(r)

c = collections.Counter(r['db'] for r in recs.values())
print('records', len(recs), dict(c), 'title-matched', matched_title)
print('abstract source', dict(collections.Counter(r['abstract_source'] for r in recs.values())))
print('reviews', sum(r['is_review'] for r in recs.values()), 'year flags', sum(1 for r in recs.values() if r['year_flag']))
