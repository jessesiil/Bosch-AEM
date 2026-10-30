"""Stratified random sample from the union 'green hydrogen / electrolysis' corpus for manual E/V/I labelling."""
import json, os, random
from common import load_openalex, LABELS
recs, metas = load_openalex()
UNION_Q = {'WE_core', 'GREEN_H2', 'PEM', 'ALK', 'SOEC', 'AEM_strict', 'AEM_broad'}
pool = [r for r in recs.values() if set(r['q']) & UNION_Q and r.get('y')]
bins = [(0, 2010), (2011, 2017), (2018, 2021), (2022, 2023), (2024, 2030)]
rng = random.Random(11)
out = []
for b in bins:
    s = [r for r in pool if b[0] <= r['y'] <= b[1]]
    k = 80
    for r in rng.sample(s, k):
        out.append({'id': r['id'], 'y': r['y'], 'stratum': f'{b[0]}-{b[1]}', 'stratum_size': len(s), 'q': r['q'], 't': r['t'], 'ab': (r.get('ab') or '')[:300]})
rng.shuffle(out)
with open(os.path.join(LABELS, 'gh_sample.jsonl'), 'w', encoding='utf-8') as f:
    for o in out: f.write(json.dumps(o) + '\n')
print('pool', len(pool), {f'{b[0]}-{b[1]}': sum(1 for r in pool if b[0] <= r['y'] <= b[1]) for b in bins})
