"""Draw a stratified random sample from a query set for manual relevance labelling.
Usage: python make_label_sample.py QUERYKEY N SEED OUTNAME
Strata = publication-year bins, allocated ~proportional with a floor so early years are represented."""
import sys, random, json, os
from common import load_openalex, LABELS
q, n, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
recs, _ = load_openalex()
pool = [r for r in recs.values() if q in r['q'] and r.get('y')]
bins = [(0, 2015), (2016, 2020), (2021, 2023), (2024, 2030)]
rng = random.Random(seed)
strata = {b: [r for r in pool if b[0] <= r['y'] <= b[1]] for b in bins}
tot = len(pool)
alloc = {b: max(int(round(n * len(s) / tot)), min(len(s), n // 8)) for b, s in strata.items()}
sample = []
for b, s in strata.items():
    k = min(alloc[b], len(s))
    sample += [(b, r) for r in rng.sample(s, k)]
rng.shuffle(sample)
with open(os.path.join(LABELS, out + '_sample.jsonl'), 'w', encoding='utf-8') as f:
    for b, r in sample:
        f.write(json.dumps({'id': r['id'], 'y': r['y'], 'stratum': f'{b[0]}-{b[1]}', 'stratum_size': len(strata[b]), 't': r['t'], 'ab': (r.get('ab') or '')[:450]}) + '\n')
print(q, 'pool', tot, {f'{b[0]}-{b[1]}': (len(s), alloc[b]) for b, s in strata.items()}, 'sample', len(sample))
