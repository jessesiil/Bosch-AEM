"""Shared helpers for the Bosch AEM bibliometric pipeline."""
import json, os, re, glob
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'data', 'raw')
PROC = os.path.join(BASE, 'data', 'processed')
LABELS = os.path.join(BASE, 'data', 'labels')
FIG = os.path.join(BASE, 'figures')

def load_openalex():
    """Load all OpenAlex jsonl parts; merge query-membership flags across parts."""
    recs, metas = {}, []
    for fn in sorted(glob.glob(os.path.join(RAW, 'openalex', 'bosch_openalex_part*.jsonl'))):
        with open(fn, encoding='utf-8') as f:
            metas.append(json.loads(f.readline()))
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r['id'] in recs:
                    recs[r['id']]['q'] = sorted(set(recs[r['id']]['q']) | set(r['q']))
                else:
                    recs[r['id']] = r
    return recs, metas

def text_of(r, n_abs=None):
    t = (r.get('t') or '')
    a = (r.get('ab') or '')
    if n_abs is not None:
        a = a[:n_abs]
    kw = '; '.join(r.get('kw') or [])
    return f"{t}. {a} Keywords: {kw}"
