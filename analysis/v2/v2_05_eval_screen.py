"""v2 step 5 - compare screening methods against the 150 gold labels.
Methods: (1) rules (stage 1 + stage-2 class), (2) TF-IDF + logistic regression (5-fold stratified CV, out-of-fold predictions,
trained only on the 150 gold papers), (3) LLM abstract runs (gold_haiku, gold_sonnet) if their results exist.
Tasks: in_scope (Y vs N) and device (gold test_level == device vs anything else).
Metrics are on the (stratified) sample itself, unweighted - fine for comparing methods, not for corpus-level prevalence.
Output: data/processed/v2/screen_eval.json + printed table."""
import csv, json, os, collections
import numpy as np
csv.field_size_limit(10**9)
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
gold = list(csv.DictReader(open(os.path.join(ROOT, 'data/labels/v2/gold_labels_v2.csv'), encoding='utf-8')))
corpus = {r['uid']: r for r in csv.DictReader(open(os.path.join(ROOT, 'data/processed/v2/corpus_screened.csv'), encoding='utf-8'))}
y_scope = np.array([g['in_scope'] == 'Y' for g in gold])
y_dev = np.array([g['test_level'] == 'device' for g in gold])

def metrics(y, p):
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum()); tn = int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float('nan'); rec = tp / (tp + fn) if tp + fn else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float('nan')
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3))

preds = {}
# 1) rules
c = [corpus[g['uid']] for g in gold]
preds['rules'] = dict(in_scope=np.array([r['stage1'] == 'include' for r in c]),
                      device=np.array([r['rule_class'] == 'device_likely' for r in c]))
# 2) TF-IDF + LR, out-of-fold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
texts = [f"{r['title']} {r['abstract']} {r['keywords']}" for r in c]
def oof(y):
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000),
                         LogisticRegression(C=4, class_weight='balanced', max_iter=2000))
    return cross_val_predict(pipe, texts, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).astype(bool)
preds['tfidf_lr_cv'] = dict(in_scope=oof(y_scope), device=oof(y_dev))
# 3) LLM runs
llm_dir = os.path.join(ROOT, 'data/processed/v2/llm')
for run in sorted(f[:-14] for f in os.listdir(llm_dir) if f.startswith('gold_') and f.endswith('_results.jsonl')):
    res_p = os.path.join(llm_dir, f'{run}_results.jsonl'); map_p = os.path.join(llm_dir, f'{run}_requests_map.csv')
    if not os.path.exists(res_p): continue
    gid_of = {m['custom_id']: m['gid'] for m in csv.DictReader(open(map_p, encoding='utf-8'))}
    out = {}
    for line in open(res_p, encoding='utf-8'):
        r = json.loads(line); o = r.get('output') or {}
        out[gid_of[r['custom_id']]] = o
    preds[run] = dict(in_scope=np.array([bool(out.get(g['gid'], {}).get('in_scope')) for g in gold]),
                      device=np.array([out.get(g['gid'], {}).get('highest_test_level') in ('single_cell', 'stack') for g in gold]))
    missing = [g['gid'] for g in gold if g['gid'] not in out or not out[g['gid']]]
    if missing: print(f'{run}: {len(missing)} papers without output: {missing[:10]}')

report = {m: {t: metrics(y, preds[m][t]) for t, y in [('in_scope', y_scope), ('device', y_dev)]} for m in preds}
report['_gold_counts'] = dict(n=len(gold), in_scope=int(y_scope.sum()), device=int(y_dev.sum()))
json.dump(report, open(os.path.join(ROOT, 'data/processed/v2/screen_eval.json'), 'w'), indent=1)
print(f"gold: {len(gold)} papers, {int(y_scope.sum())} in scope, {int(y_dev.sum())} device")
print(f"{'method':14s} {'task':9s} {'prec':>6s} {'recall':>6s} {'f1':>6s}   tp fp fn")
for m in preds:
    for t in ['in_scope', 'device']:
        r = report[m][t]; print(f"{m:14s} {t:9s} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f}   {r['tp']:2d} {r['fp']:2d} {r['fn']:2d}")
