"""Relevance classifier for the union 'green hydrogen / water electrolysis' OpenAlex corpus.
Tasks: elec = E vs (V+I)  ("water-electrolysis technology");  gh = (E+V) vs I ("green-hydrogen related").
Also tags electrolysis technology (AEM/PEM/ALK/SOEC) by regex on title+abstract+keywords."""
import csv, json, os, re
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from common import load_openalex, text_of, LABELS, PROC
from classify_aem import model, metrics   # same model spec as the AEM classifier (module re-runs AEM step on import)

recs, _ = load_openalex()
UNION_Q = {'WE_core', 'GREEN_H2', 'PEM', 'ALK', 'SOEC', 'AEM_strict', 'AEM_broad'}
labels = {row['id']: row for row in csv.DictReader(open(os.path.join(LABELS, 'gh_labels.csv'), encoding='utf-8'))}
ids = list(labels); X = [text_of(recs[i]) for i in ids]; lab = np.array([labels[i]['label'] for i in ids])
report = {'n_labelled': len(ids), 'label_counts': {k: int((lab == k).sum()) for k in 'EVI'}}
final = {}
for task, pos in [('elec', {'E'}), ('gh', {'E', 'V'})]:
    y = np.array([1 if l in pos else 0 for l in lab])
    sc = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(X, y):
        m = model().fit([X[i] for i in tr], y[tr]); sc.append(metrics(y[te], m.predict([X[i] for i in te])))
    report[task] = {'ml_cv_mean_sd': {k: (round(float(np.mean([s[k] for s in sc])), 3), round(float(np.std([s[k] for s in sc])), 3)) for k in sc[0]}}
    # naive baseline: everything retrieved by the queries counts as relevant
    report[task]['no_filter_precision'] = round(float(y.mean()), 3)
    oof = cross_val_predict(model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1), method='predict_proba')[:, 1]
    st = {}
    for i, idx in enumerate(ids):
        s = labels[idx]['stratum']; st.setdefault(s, ([], [])); st[s][0].append(int(y[i])); st[s][1].append(int(oof[i] >= 0.5))
    report[task]['ml_oof_by_stratum'] = {s: dict(n=len(a), true_pos_rate=round(float(np.mean(a)), 3), **metrics(np.array(a), np.array(b))) for s, (a, b) in sorted(st.items())}
    final[task] = model().fit(X, y)

TECH = {'AEM': re.compile(r"anion[- ]exchange[- ]membrane|AEM ?WE|\bAEM\b|AEMEL|hydroxide[- ]exchange[- ]membrane|alkaline (anion )?exchange membrane|alkaline membrane water", re.I),
        'PEM': re.compile(r"proton[- ]exchange[- ]membrane|polymer electrolyte membrane|PEM ?WE|\bPEM\b|PEMEL|solid polymer electrolyte|acidic (oxygen evolution|OER|water)|iridium", re.I),
        'ALK': re.compile(r"alkaline water electroly|alkaline electroly|\bAWE\b|zero[- ]gap alkaline|diaphragm", re.I),
        'SOEC': re.compile(r"solid oxide|SOEC|SOEL|steam electrolysis|high[- ]temperature electrolysis|protonic ceramic|proton[- ]conducting (ceramic|oxide)|co-electrolysis", re.I)}
aem_core = {}
with open(os.path.join(PROC, 'aem_classified.csv'), encoding='utf-8') as f:
    for row in csv.DictReader(f): aem_core[row['id']] = row
pool = [r for r in recs.values() if set(r['q']) & UNION_Q and r.get('y')]
Xall = [text_of(r) for r in pool]
pe = final['elec'].predict_proba(Xall)[:, 1]; pg = final['gh'].predict_proba(Xall)[:, 1]
with open(os.path.join(PROC, 'gh_classified.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['id', 'year', 'queries', 'p_elec', 'p_gh', 'ml_elec', 'ml_gh', 'tag_AEM_regex', 'aem_ml_core', 'tag_PEM', 'tag_ALK', 'tag_SOEC', 'n_company_affil', 'countries', 'hand_label'])
    for r, a, b in zip(pool, pe, pg):
        s = (r.get('t') or '') + ' ' + (r.get('ab') or '') + ' ' + ' '.join(r.get('kw') or [])
        ac = aem_core.get(r['id'])
        w.writerow([r['id'], r['y'], '|'.join(sorted(r['q'])), round(a, 4), round(b, 4), int(a >= 0.5), int(b >= 0.5 or a >= 0.5),
                    int(bool(TECH['AEM'].search(s))), int(ac['ml_core']) if ac else 0, int(bool(TECH['PEM'].search(s))), int(bool(TECH['ALK'].search(s))), int(bool(TECH['SOEC'].search(s))),
                    sum(1 for i in r.get('inst', []) if i[1] == 'company'), ';'.join(r.get('ctry', [])), labels.get(r['id'], {}).get('label', '')])
report['n_pool'] = len(pool); report['n_pred_elec'] = int((pe >= 0.5).sum()); report['n_pred_gh'] = int(((pg >= 0.5) | (pe >= 0.5)).sum())
json.dump(report, open(os.path.join(PROC, 'gh_classifier_report.json'), 'w'), indent=1)
print(json.dumps(report, indent=1))
