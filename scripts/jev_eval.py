"""Jev (TypeSafe's zero-shot decision model) as the cheap first stage, on the full held-out sets.

Question: does a label-free zero-shot decision model replace the label-trained SVM in front of the
frontier LLM? Methods, all on identical test items and the same 5 calib subsamples as final_eval.py:
  svm / sonnet_thr / fuse_{rho}      the paper's SVM stage, full-call Sonnet, SVM->Sonnet fused cascade
  jev_raw / jev_thr / jev_nnppi      Jev alone: P(yes) >= 0.5, calib-tuned threshold, + NN-PPI (k on calib)
  jevrep_{rho} / jevfuse_{rho}       Jev->Sonnet cascades: the rho claims Jev is least sure of (closest to its
                                     calib threshold in logit) go to Sonnet, whose decision replaces Jev's
                                     (sonnet_thr) or is fused with it by logistic regression on (logit Jev, s)
  svmjev                             no LLM call: logistic regression on (SVM margin, logit Jev)
  tri_{rho}                          svmjev routed like the paper's cascade; queried claims decided by a
                                     logistic regression on (SVM margin, logit Jev, s)
Also reports how often each cheap stage repeats Sonnet's errors (the "correlated errors" question).

Needs results/jev/nnppi/*.jsonl (scripts/score_jev.py). Writes results/jev_eval_full.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.data import DATASETS, FULL_PARTS, RESULTS, Split, best_threshold, load_frontier, read_frontier_scores  # noqa: E402
from cwcascade.nnppi import nn_ppi  # noqa: E402
from final_eval import BUDGETS, SEEDS, compare, metrics, route, select_nnppi_k  # noqa: E402

VARIANT = "nnppi"


def jev_scores(key):
    with open(os.path.join(RESULTS, "jev", VARIANT, f"{key}.jsonl"), encoding="utf-8") as f:
        return {r["id"]: r["noul"] for r in map(json.loads, f)}


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def evaluate(name):
    test, calib, test_items = load_frontier(name, full=True)
    calib_key = DATASETS[name][0]
    jt = [jev_scores(p) for p in (name,) + FULL_PARTS[name]]
    lens = [sum(r["part"] == p for r in test_items) for p in (name,) + FULL_PARTS[name]]
    j_test = np.concatenate([[d[i] for i in range(n)] for d, n in zip(jt, lens)])
    jc = jev_scores(calib_key)
    j_cal = np.array([jc[i] for i in sorted(read_frontier_scores()[calib_key])])
    y = test.y

    preds = {'jev_raw': [(j_test >= .5).astype(int)]}
    keys = ['svm', 'sonnet_thr', 'jev_thr', 'jev_nnppi', 'svmjev'] + \
           [f'{v}_{b}' for b in BUDGETS for v in ('fuse', 'jevrep', 'jevfuse', 'tri')]
    preds.update({k: [] for k in keys})
    overlap = {}
    for seed in range(SEEDS):
        idx = np.random.default_rng(seed).permutation(len(calib.y))[:int(0.8 * len(calib.y))]
        Xs, ys, ss, js = calib.X[idx], calib.y[idx], calib.s[idx], j_cal[idx]
        svm = SVC(class_weight='balanced', random_state=seed).fit(Xs, ys)
        d_test = svm.decision_function(test.X)
        d_cal = cross_val_predict(SVC(class_weight='balanced', random_state=seed), Xs, ys, cv=5, method='decision_function')
        thr_s, thr_j = best_threshold(ss, ys), best_threshold(js, ys)

        p_svm = (d_test > 0).astype(int)
        p_sonnet = (test.s >= thr_s).astype(int)
        p_jev = (j_test >= thr_j).astype(int)
        p_fuse = LogisticRegression(class_weight='balanced').fit(np.c_[d_cal, ss], ys).predict(np.c_[d_test, test.s])
        p_jevfuse = LogisticRegression(class_weight='balanced').fit(np.c_[logit(js), ss], ys).predict(
            np.c_[logit(j_test), test.s])
        cheap = LogisticRegression(class_weight='balanced').fit(np.c_[d_cal, logit(js)], ys)
        m_cheap = cheap.decision_function(np.c_[d_test, logit(j_test)])
        p_tri = LogisticRegression(class_weight='balanced').fit(np.c_[d_cal, logit(js), ss], ys).predict(
            np.c_[d_test, logit(j_test), test.s])
        k = select_nnppi_k(Split(Xs, ys, js), seed)[0]

        preds['svm'].append(p_svm)
        preds['sonnet_thr'].append(p_sonnet)
        preds['jev_thr'].append(p_jev)
        preds['jev_nnppi'].append((nn_ppi(j_test, test.X, js, Xs, ys, k) >= .5).astype(int))
        preds['svmjev'].append((m_cheap > 0).astype(int))
        order_svm = np.argsort(np.abs(d_test))
        order_jev = np.argsort(np.abs(logit(j_test) - logit(thr_j)))
        order_cheap = np.argsort(np.abs(m_cheap))
        for b in BUDGETS:
            preds[f'fuse_{b}'].append(route(p_svm, p_fuse, order_svm, b))
            preds[f'jevrep_{b}'].append(route(p_jev, p_sonnet, order_jev, b))
            preds[f'jevfuse_{b}'].append(route(p_jev, p_jevfuse, order_jev, b))
            preds[f'tri_{b}'].append(route((m_cheap > 0).astype(int), p_tri, order_cheap, b))
        if seed == 0:
            wrong_s = p_sonnet != y
            overlap = {n: dict(p_wrong_given_sonnet_wrong=float(np.mean(p[wrong_s] != y[wrong_s])),
                               p_wrong_given_sonnet_right=float(np.mean(p[~wrong_s] != y[~wrong_s])))
                       for n, p in (('svm', p_svm), ('jev_thr', p_jev))}
            overlap['n_sonnet_errors'] = int(wrong_s.sum())

    summary = {}
    for key, plist in preds.items():
        ms = [metrics(y, p) for p in plist]
        summary[key] = {m: (float(np.mean([x[m] for x in ms])), float(np.std([x[m] for x in ms]))) for m in ms[0]}
    first = {key: plist[0] for key, plist in preds.items()}
    contrasts = {'jev_thr_vs_svm': compare(y, first['jev_thr'], first['svm']),
                 'jev_thr_vs_sonnet_thr': compare(y, first['jev_thr'], first['sonnet_thr']),
                 'svmjev_vs_svm': compare(y, first['svmjev'], first['svm'])}
    for b in (.3, .5, 1.0):
        contrasts[f'fuse_{b}_vs_jevfuse_{b}'] = compare(y, first[f'fuse_{b}'], first[f'jevfuse_{b}'])
        contrasts[f'fuse_{b}_vs_jevrep_{b}'] = compare(y, first[f'fuse_{b}'], first[f'jevrep_{b}'])
        contrasts[f'tri_{b}_vs_fuse_{b}'] = compare(y, first[f'tri_{b}'], first[f'fuse_{b}'])
    parts = np.array([r['part'] for r in test_items])
    by_part = {}
    for part in dict.fromkeys(parts):
        m = parts == part
        by_part[part] = {key: float(np.mean([np.mean(p[m] == y[m]) for p in preds[key]]))
                         for key in ('svm', 'jev_thr', 'sonnet_thr', 'fuse_0.5', 'jevfuse_0.5', 'tri_0.5')}
    result = dict(n_test=len(y), auc=dict(jev=float(roc_auc_score(y, j_test)), sonnet=float(roc_auc_score(y, test.s))),
                  score_corr_jev_sonnet=float(np.corrcoef(j_test, test.s)[0, 1]), error_overlap_seed0=overlap,
                  metrics=summary, contrasts=contrasts, by_part=by_part)
    report(name, result)
    return result


def report(name, r):
    m = r['metrics']
    print(f"\n===== {name}: full test n={r['n_test']}, seeds={SEEDS}")
    print(f"  test AUC: Jev {r['auc']['jev']:.3f}  Sonnet {r['auc']['sonnet']:.3f};  corr(Jev, Sonnet scores) = {r['score_corr_jev_sonnet']:.3f}")
    for key in ('svm', 'jev_raw', 'jev_thr', 'jev_nnppi', 'sonnet_thr', 'svmjev'):
        print(f"  {key:12s} acc={m[key]['acc'][0]:.3f}±{m[key]['acc'][1]:.3f}  F1c1={m[key]['f1c1'][0]:.3f}")
    print("  budget   SVM->Sonnet fuse   Jev->Sonnet replace   Jev->Sonnet fuse   SVM+Jev->Sonnet fuse")
    for b in BUDGETS:
        print(f"  {b:>5.0%}    " + "      ".join(f"{m[f'{v}_{b}']['acc'][0]:.3f}±{m[f'{v}_{b}']['acc'][1]:.3f}"
                                          for v in ('fuse', 'jevrep', 'jevfuse', 'tri')))
    print(f"  error overlap with Sonnet (seed 0): {r['error_overlap_seed0']}")
    for c, v in r['contrasts'].items():
        print(f"  {c:28s} diff={v['diff']:+.3f} CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] McNemar p={v['mcnemar_p']:.4f}")
    for part, a in r['by_part'].items():
        print(f"  -- part {part}: " + "  ".join(f"{k}={v:.3f}" for k, v in a.items()))


if __name__ == '__main__':
    results = {name: evaluate(name) for name in DATASETS}
    with open(os.path.join(RESULTS, 'jev_eval_full.json'), 'w') as f:
        json.dump(results, f, indent=1)
