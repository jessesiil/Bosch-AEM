"""Relevance classifier for the AEM_broad OpenAlex corpus.

Trains TF-IDF + logistic regression on the hand-labelled stratified sample (data/labels/aem_labels.csv),
evaluates by repeated stratified 5-fold cross-validation, compares against keyword rules, and writes
per-record predictions for the full corpus to data/processed/aem_classified.csv.

Two binary tasks:
  core     : C vs (P+I)   -> "about AEM water electrolysis"
  relevant : (C+P) vs I   -> "water-electrolysis relevant (AEM or alkaline-catalysis adjacent)"
"""
import csv, json, os, re, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from common import load_openalex, text_of, LABELS, PROC

recs, _ = load_openalex()
labels = {row['id']: row for row in csv.DictReader(open(os.path.join(LABELS, 'aem_labels.csv'), encoding='utf-8'))}
ids = list(labels)
X = [text_of(recs[i]) for i in ids]
lab = np.array([labels[i]['label'] for i in ids])

def model():
    feats = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words='english'),
        TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=40000))
    return make_pipeline(feats, LogisticRegression(C=4.0, class_weight='balanced', max_iter=3000))

# ---- keyword baselines (transparent rules) ----
AEMWE = re.compile(r"(anion[- ]exchange[- ]membrane[s]? (water )?electroly|AEM ?WE|AEMWE|AEMEL|AEM (water )?electroly|AEM-WE|"
                   r"hydroxide[- ]exchange[- ]membrane[s]? (water )?electroly|alkaline (anion )?exchange membrane (water )?electroly|"
                   r"anion[- ]exchange[- ]membrane[s]? (based )?(water )?(electrolyz|electrolys)|solid alkaline water electroly|"
                   r"alkaline membrane water electroly|HEMEL|AEM electrolyz|AEM electrolys|AEM water)", re.I)
EXCL = re.compile(r"(CO2|CO 2|carbon dioxide|electrodialy|desalinat|redox flow|flow batter|fuel cell|chlor|uranium|zinc recovery|"
                  r"heavy metal|microbial|acidic (oxygen evolution|water|OER)|proton exchange membrane water|PEMWE|furfural|glycer|"
                  r"electrosynthes|wastewater treatment|nitrate|ammonia)", re.I)
WE = re.compile(r"(water electroly|water splitting|hydrogen evolution|oxygen evolution|HER\b|OER\b|hydrogen production|electrolyz|electrolys)", re.I)
def rule_core(r):
    return bool(AEMWE.search((r.get('t') or '') + ' ' + (r.get('ab') or '')))
def rule_rel(r):
    s = (r.get('t') or '') + ' ' + (r.get('ab') or '')
    if rule_core(r):
        return True
    return bool(WE.search(s)) and not bool(EXCL.search(r.get('t') or ''))

def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average='binary', zero_division=0)
    return dict(precision=round(pr, 3), recall=round(rc, 3), f1=round(f1, 3), accuracy=round(accuracy_score(y, p), 3))

def main():
    report = {'n_labelled': len(ids), 'label_counts': {k: int((lab == k).sum()) for k in 'CPI'}}
    final_models = {}
    for task, pos in [('core', {'C'}), ('relevant', {'C', 'P'})]:
        y = np.array([1 if l in pos else 0 for l in lab])
        # repeated CV for the ML model
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
        scores = []
        for tr, te in rskf.split(X, y):
            m = model().fit([X[i] for i in tr], y[tr])
            scores.append(metrics(y[te], m.predict([X[i] for i in te])))
        agg = {k: (round(float(np.mean([s[k] for s in scores])), 3), round(float(np.std([s[k] for s in scores])), 3)) for k in scores[0]}
        rule = rule_core if task == 'core' else rule_rel
        rule_pred = np.array([1 if rule(recs[i]) else 0 for i in ids])
        report[task] = {'ml_cv_mean_sd': agg, 'keyword_rule': metrics(y, rule_pred)}
        # out-of-fold probabilities for calibration-style per-stratum checks
        oof = cross_val_predict(model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1), method='predict_proba')[:, 1]
        strata = {}
        for i, idx in enumerate(ids):
            s = labels[idx]['stratum']
            strata.setdefault(s, {'y': [], 'p': []})
            strata[s]['y'].append(int(y[i])); strata[s]['p'].append(int(oof[i] >= 0.5))
        report[task]['ml_oof_by_stratum'] = {s: dict(n=len(v['y']), true_pos_rate=round(np.mean(v['y']), 3), **metrics(np.array(v['y']), np.array(v['p']))) for s, v in sorted(strata.items())}
        final_models[task] = model().fit(X, y)

    # ---- apply to full corpus ----
    pool = [r for r in recs.values() if 'AEM_broad' in r['q']]
    Xall = [text_of(r) for r in pool]
    p_core = final_models['core'].predict_proba(Xall)[:, 1]
    p_rel = final_models['relevant'].predict_proba(Xall)[:, 1]
    os.makedirs(PROC, exist_ok=True)
    with open(os.path.join(PROC, 'aem_classified.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'year', 'type', 'title', 'has_abstract', 'in_AEM_strict', 'p_core', 'p_relevant', 'ml_core', 'ml_relevant', 'rule_core', 'rule_relevant', 'hand_label', 'countries', 'n_company_affil', 'cited_by'])
        for r, pc, pr in zip(pool, p_core, p_rel):
            ncomp = sum(1 for i in r.get('inst', []) if i[1] == 'company')
            w.writerow([r['id'], r['y'], r['ty'], r['t'], int(bool(r.get('ab'))), int('AEM_strict' in r['q']), round(pc, 4), round(pr, 4),
                        int(pc >= 0.5), int(pr >= 0.5 or pc >= 0.5), int(rule_core(r)), int(rule_rel(r)), labels.get(r['id'], {}).get('label', ''),
                        ';'.join(r.get('ctry', [])), ncomp, r.get('cit')])
    report['n_corpus'] = len(pool)
    report['n_pred_core'] = int((p_core >= 0.5).sum()); report['n_pred_relevant'] = int(((p_rel >= 0.5) | (p_core >= 0.5)).sum())
    report['n_rule_core'] = sum(rule_core(r) for r in pool); report['n_AEM_strict'] = sum('AEM_strict' in r['q'] for r in pool)
    report['share_no_abstract'] = round(float(np.mean([not r.get('ab') for r in pool])), 3)
    json.dump(report, open(os.path.join(PROC, 'aem_classifier_report.json'), 'w'), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
