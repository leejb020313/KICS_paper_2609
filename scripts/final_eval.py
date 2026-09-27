"""Main experiment of the paper (Table 1, Figure 1).

Methods, all on identical test items:
  gemma_raw / gemma_nnppi_k{3,5,10} / gemma_nnppi_sel   Gemma 3 4B, raw and + NN-PPI (k chosen on calib)
  svm                                                   embedding RBF-SVM trained on calib labels, no LLM
  sonnet_raw / sonnet_thr                               Claude Sonnet 5 on all claims, thr 0.5 / calib-tuned thr
  replace_{rho} / fuse_{rho}                            cascades that query the LLM on the rho most uncertain
                                                        claims and either replace the SVM decision with
                                                        sonnet_thr or fuse SVM margin + LLM score (Eq. 1)

Every calib-fitted component (SVM, fuser, threshold) is refit on 5 random 80% calib subsamples
drawn without replacement; nothing is tuned on test. Headline contrasts use McNemar tests and
paired bootstrap CIs on the seed-0 predictions.

Writes results/final_results.json.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.data import DATASETS, RESULTS, best_threshold, load_frontier, load_gemma  # noqa: E402
from cwcascade.nnppi import nn_ppi  # noqa: E402

SEEDS = 5
BUDGETS = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
NNPPI_KS = (3, 5, 10)
BOOT = 2000


def metrics(y, p):
    return dict(acc=float(np.mean(p == y)), wf1=float(f1_score(y, p, average='weighted')), f1c1=float(f1_score(y, p)))


def route(base, alt, order, rho):
    """Replace the `rho` fraction of base predictions that come first in `order` by alt."""
    out = base.copy()
    sel = order[:int(round(rho * len(base)))]
    out[sel] = alt[sel]
    return out


def compare(y, a, b):
    """Accuracy difference a - b with paired-bootstrap 95% CI and exact McNemar p-value."""
    ca, cb = (a == y), (b == y)
    g = np.random.default_rng(0)
    d = [np.mean(ca[i]) - np.mean(cb[i]) for i in (g.integers(0, len(y), len(y)) for _ in range(BOOT))]
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return dict(diff=float(np.mean(ca) - np.mean(cb)), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                mcnemar_p=float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0)


def select_nnppi_k(gemma_cal):
    """Choose k on a 70/30 split of the Gemma calib set (70% neighbours, 30% scored)."""
    perm = np.random.default_rng(0).permutation(len(gemma_cal.y))
    pool, tune = perm[:int(0.7 * len(perm))], perm[int(0.7 * len(perm)):]
    c = gemma_cal
    tune_acc = {k: float(np.mean((nn_ppi(c.s[tune], c.X[tune], c.s[pool], c.X[pool], c.y[pool], k) >= .5) == c.y[tune]))
                for k in NNPPI_KS}
    return max(tune_acc, key=tune_acc.get), tune_acc


def evaluate(name):
    test, calib, test_items = load_frontier(name)
    gemma_test_s, gemma_cal = load_gemma(name, test_items)
    y = test.y

    preds = {'gemma_raw': [(gemma_test_s >= .5).astype(int)]}
    for k in NNPPI_KS:
        theta = nn_ppi(gemma_test_s, test.X, gemma_cal.s, gemma_cal.X, gemma_cal.y, k)
        preds[f'gemma_nnppi_k{k}'] = [(theta >= .5).astype(int)]
    k_sel, tune_acc = select_nnppi_k(gemma_cal)
    preds['gemma_nnppi_sel'] = preds[f'gemma_nnppi_k{k_sel}']
    preds['sonnet_raw'] = [(test.s >= .5).astype(int)]
    for key in ['fuse_sel', 'svm', 'sonnet_thr'] + [f'{v}_{b}' for b in BUDGETS for v in ('fuse', 'replace')]:
        preds[key] = []
    rho_sel, calib_curve = [], {}

    for seed in range(SEEDS):
        idx = np.random.default_rng(seed).permutation(len(calib.y))[:int(0.8 * len(calib.y))]
        Xs, ys, ss = calib.X[idx], calib.y[idx], calib.s[idx]

        svm = SVC(class_weight='balanced', random_state=seed).fit(Xs, ys)
        d_test = svm.decision_function(test.X)
        # out-of-fold SVM margins on calib, so the fuser learns how much to trust an unseen margin
        d_cal = cross_val_predict(SVC(class_weight='balanced', random_state=seed), Xs, ys, cv=5, method='decision_function')
        fuser = LogisticRegression(class_weight='balanced').fit(np.c_[d_cal, ss], ys)
        thr = best_threshold(ss, ys)

        p_svm = (d_test > 0).astype(int)
        p_fuse = fuser.predict(np.c_[d_test, test.s])
        p_sonnet = (test.s >= thr).astype(int)
        order = np.argsort(np.abs(d_test))  # most uncertain first
        preds['svm'].append(p_svm)
        preds['sonnet_thr'].append(p_sonnet)
        for b in BUDGETS:
            preds[f'fuse_{b}'].append(route(p_svm, p_fuse, order, b))
            preds[f'replace_{b}'].append(route(p_svm, p_sonnet, order, b))

        # calib-only view of the same curve: out-of-fold fused predictions routed by |d_cal|
        p_svm_cal = (d_cal > 0).astype(int)
        p_fuse_cal = cross_val_predict(LogisticRegression(class_weight='balanced'), np.c_[d_cal, ss], ys, cv=5)
        order_cal = np.argsort(np.abs(d_cal))
        curve = {b: float(np.mean(route(p_svm_cal, p_fuse_cal, order_cal, b) == ys)) for b in BUDGETS}
        sonnet_oof = np.zeros(len(ys), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, ys):
            sonnet_oof[te] = (ss[te] >= best_threshold(ss[tr], ys[tr])).astype(int)
        calib_curve.setdefault('sonnet_thr_oof', []).append(float(np.mean(sonnet_oof == ys)))
        calib_curve.setdefault('svm_oof', []).append(float(np.mean(p_svm_cal == ys)))
        for b in BUDGETS:
            calib_curve.setdefault(f'fuse_{b}', []).append(curve[b])

        # pre-declared budget rule: smallest rho whose calib accuracy reaches the frontier LLM's
        target = np.mean((ss >= thr) == ys)
        rho = next((b for b in BUDGETS if curve[b] >= target), 1.0)
        rho_sel.append(rho)
        preds['fuse_sel'].append(route(p_svm, p_fuse, order, rho))

    summary = {}
    for key, plist in preds.items():
        ms = [metrics(y, p) for p in plist]
        summary[key] = {m: (float(np.mean([x[m] for x in ms])), float(np.std([x[m] for x in ms]))) for m in ms[0]}

    first = {key: plist[0] for key, plist in preds.items()}
    contrasts = {}
    for b in (.2, .3, .4, .5, 1.0):
        contrasts[f'fuse_{b}_vs_sonnet_thr'] = compare(y, first[f'fuse_{b}'], first['sonnet_thr'])
        contrasts[f'fuse_{b}_vs_replace_{b}'] = compare(y, first[f'fuse_{b}'], first[f'replace_{b}'])
    for sd in range(SEEDS):
        contrasts[f'fuse_0.5_vs_fuse_1.0_seed{sd}'] = compare(y, preds['fuse_0.5'][sd], preds['fuse_1.0'][sd])
    contrasts['fuse_sel_vs_sonnet_thr'] = compare(y, first['fuse_sel'], first['sonnet_thr'])
    contrasts['svm_vs_gemma_nnppi_sel'] = compare(y, first['svm'], first['gemma_nnppi_sel'])
    best_nnppi = max((first[f'gemma_nnppi_k{k}'] for k in NNPPI_KS), key=lambda p: np.mean(p == y))
    contrasts['svm_vs_best_gemma_nnppi'] = compare(y, first['svm'], best_nnppi)

    result = dict(calib_curve={n: (float(np.mean(v)), float(np.std(v))) for n, v in calib_curve.items()},
                  k_sel=k_sel, nnppi_tune_acc=tune_acc, rho_sel=rho_sel, n_test=len(y),
                  n_calib_frontier=len(calib.y), n_calib_gemma=len(gemma_cal.y), metrics=summary, contrasts=contrasts)
    report(name, result)
    return result


def report(name, r):
    m = r['metrics']
    print(f"\n===== {name}: test n={r['n_test']}, frontier-scored calib n={r['n_calib_frontier']}, seeds={SEEDS}")
    print('  calib OOF curve:', {n: round(v[0], 3) for n, v in r['calib_curve'].items()})
    print(f"  NN-PPI k selected on calib: k={r['k_sel']} (tune acc {r['nnppi_tune_acc']}); rho selected per seed: {r['rho_sel']}")
    for key in ['gemma_raw', 'gemma_nnppi_k3', 'gemma_nnppi_k5', 'gemma_nnppi_k10', 'gemma_nnppi_sel', 'svm', 'sonnet_raw', 'sonnet_thr', 'fuse_sel']:
        print(f"  {key:18s} acc={m[key]['acc'][0]:.3f}±{m[key]['acc'][1]:.3f}  wF1={m[key]['wf1'][0]:.3f}  F1c1={m[key]['f1c1'][0]:.3f}")
    print("  budget   FUSE acc (±sd)     REPLACE acc (±sd)")
    for b in BUDGETS:
        print(f"  {b:>5.0%}    {m[f'fuse_{b}']['acc'][0]:.3f}±{m[f'fuse_{b}']['acc'][1]:.3f}      "
              f"{m[f'replace_{b}']['acc'][0]:.3f}±{m[f'replace_{b}']['acc'][1]:.3f}")
    for c, v in r['contrasts'].items():
        print(f"  {c:32s} diff={v['diff']:+.3f} CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] McNemar p={v['mcnemar_p']:.4f}")


if __name__ == '__main__':
    results = {name: evaluate(name) for name in DATASETS}
    with open(os.path.join(RESULTS, 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=1)
