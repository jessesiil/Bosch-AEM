"""v2 step 7 - turn fetched full texts into clean body text for the full-text LLM run.

  python analysis/v2/v2_07_extract_text.py        (needs: python -m pip install pymupdf)

PDF -> text with PyMuPDF (fitz). Then: drop everything from the last 'References'/'Bibliography' heading onward
(also 'Acknowledg(e)ments', 'Author contributions', 'Conflicts of interest', 'Supporting Information' blocks near the end),
de-hyphenate line breaks, collapse whitespace. Elsevier plain text gets the same reference stripping.
We deliberately do NOT use GROBID (needs a Java/Docker server); the cost is that section labels are not structured, which the
LLM prompt does not need. Tables survive only as flattened text; figures are lost (the prompt already forbids reading figures).
Outputs: data/processed/v2/fulltext/<id>.txt and data/processed/v2/fulltext_index.csv (uid, source, chars, pages, status)."""
import csv, os, re, sys
csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
LOG = os.path.join(ROOT, 'data', 'raw', 'fulltext', 'fetch_log.csv')  # rows with result 'ok' (steps 6 and 6b)
OUTDIR = os.path.join(ROOT, 'data', 'processed', 'v2', 'fulltext')
INDEX = os.path.join(ROOT, 'data', 'processed', 'v2', 'fulltext_index.csv')
END_HEAD = re.compile(r'\n\s*(references|reference list|bibliography|notes and references|literature cited)\s*\n', re.I)
TAIL_HEAD = re.compile(r'\n\s*(acknowledg(e)?ments?|author contributions?|credit authorship contribution statement|'
                       r'(declaration of )?(competing|conflicts?) of interests?|data availability)\s*\n', re.I)

def clean(t):
    t = t.replace('\r', '')
    t = re.sub(r'-\n(?=[a-z])', '', t)            # de-hyphenate
    ms = list(END_HEAD.finditer(t))
    if ms and ms[-1].start() > 0.4 * len(t): t = t[:ms[-1].start()]
    ms = list(TAIL_HEAD.finditer(t))
    if ms and ms[0].start() > 0.6 * len(t): t = t[:ms[0].start()]
    t = re.sub(r'[ \t]+', ' ', t); t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def main():
    try:
        import fitz
    except ImportError:
        sys.exit('Missing package. Run: python -m pip install pymupdf')
    os.makedirs(OUTDIR, exist_ok=True); rows = []
    for r in csv.DictReader(open(LOG, encoding='utf-8')):
        if r['result'] != 'ok': continue
        src = os.path.join(ROOT, r['file']); out = os.path.join(OUTDIR, os.path.splitext(os.path.basename(src))[0] + '.txt')
        try:
            if src.endswith('.pdf'):
                doc = fitz.open(src); pages = doc.page_count; txt = '\n'.join(p.get_text() for p in doc)
            else:
                pages = ''; txt = open(src, encoding='utf-8').read()
            body = clean(txt); status = 'ok' if len(body) > 5000 else 'too_short_check'
            open(out, 'w', encoding='utf-8').write(body)
        except Exception as e:
            body, pages, status = '', '', 'error_' + type(e).__name__
        rows.append(dict(uid=r['uid'], doi=r['doi'], source=r['source'], pages=pages, chars=len(body), status=status,
                         file=os.path.relpath(out, ROOT) if status != 'error' else ''))
    # PDFs downloaded by hand (see v2_06b): data/raw/fulltext/manual/<file_id>.pdf
    mdir = os.path.join(ROOT, 'data', 'raw', 'fulltext', 'manual'); seen = {r['uid'] for r in rows}
    if os.path.isdir(mdir):
        corpus = {re.sub(r'[^A-Za-z0-9._-]+', '_', r['uid'])[:150]: r for r in csv.DictReader(open(os.path.join(ROOT, 'data', 'processed', 'v2', 'corpus_screened.csv'), encoding='utf-8'))}
        for fn in os.listdir(mdir):
            fid = os.path.splitext(fn)[0]; c = corpus.get(fid)
            if not fn.lower().endswith('.pdf') or not c or c['uid'] in seen: continue
            out = os.path.join(OUTDIR, fid + '.txt')
            try:
                doc = fitz.open(os.path.join(mdir, fn)); pages = doc.page_count; body = clean('\n'.join(p.get_text() for p in doc))
                status = 'ok' if len(body) > 5000 else 'too_short_check'; open(out, 'w', encoding='utf-8').write(body)
            except Exception as e:
                body, pages, status = '', '', 'error_' + type(e).__name__
            rows.append(dict(uid=c['uid'], doi=c['doi'], source='manual', pages=pages, chars=len(body), status=status,
                             file=os.path.relpath(out, ROOT)))
    with open(INDEX, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import statistics
    ch = [r['chars'] for r in rows if r['status'] == 'ok']
    print(f'{len(rows)} texts, {len(ch)} ok; median {statistics.median(ch) if ch else 0:,.0f} chars (~{(statistics.median(ch) if ch else 0)/3.8:,.0f} tokens)')

if __name__ == '__main__':
    main()
