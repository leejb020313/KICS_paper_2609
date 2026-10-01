"""Streaming router check (last paragraph of Section III).

final_eval.py routes the rho most uncertain items of the whole test batch to the LLM, which
needs the batch in advance. A deployable router decides per sentence: fix a threshold tau on
|d| so that a rho fraction of the calib out-of-fold margins fall below it, then query the LLM
for a test sentence iff |d(x)| < tau. The realised test call rate no longer equals rho exactly,
so it is reported alongside accuracy. Uses the same 5 calib subsamples as final_eval.py.

Writes results/analysis/streaming_results.json (with --full: the full held-out sets, results/paper/streaming_results_full.json).
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.data import ANALYSIS, DATASETS, PAPER, RESULTS, best_threshold, load_frontier  # noqa: E402

SEEDS = 5
BUDGETS = [.1, .2, .3, .4, .5, .6, .7, .8, .9]


def mcnemar(y, a, b):
    ca, cb = (a == y), (b == y)
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0


def evaluate(name, full=False):
    test, calib, _ = load_frontier(name, full)
    y = test.y
    acc = {f'{v}_{b}': [] for v in ('batch_global', 'stream_global') for b in BUDGETS}
    rate = {b: [] for b in BUDGETS}
    sonnet_acc, preds0 = [], {}
    for seed in range(SEEDS):
        idx = np.random.default_rng(seed).permutation(len(calib.y))[:int(0.8 * len(calib.y))]
        Xs, ys, ss = calib.X[idx], calib.y[idx], calib.s[idx]
        d_test = SVC(class_weight='balanced', random_state=seed).fit(Xs, ys).decision_function(test.X)
        d_cal = cross_val_predict(SVC(class_weight='balanced', random_state=seed), Xs, ys, cv=5, method='decision_function')
        fuser = LogisticRegression(class_weight='balanced').fit(np.c_[d_cal, ss], ys)
        p_svm, p_fuse = (d_test > 0).astype(int), fuser.predict(np.c_[d_test, test.s])
        sonnet_acc.append(float(np.mean((test.s >= best_threshold(ss, ys)) == y)))
        order = np.argsort(np.abs(d_test))
        for b in BUDGETS:
            # batch router (as in final_eval.py): the top-rho most uncertain test items
            batch = p_svm.copy()
            sel = order[:int(round(b * len(y)))]
            batch[sel] = p_fuse[sel]
            # streaming router: per-sentence threshold fixed on calib
            routed = np.abs(d_test) < np.quantile(np.abs(d_cal), b)
            stream = p_svm.copy()
            stream[routed] = p_fuse[routed]
            rate[b].append(float(routed.mean()))
            for v, p in (('batch_global', batch), ('stream_global', stream)):
                acc[f'{v}_{b}'].append(float(np.mean(p == y)))
                if seed == 0:
                    preds0[f'{v}_{b}'] = p

    result = dict(
        acc={n: (float(np.mean(v)), float(np.std(v))) for n, v in acc.items()},
        test_call_rate={str(b): (float(np.mean(v)), float(np.std(v))) for b, v in rate.items()},
        sonnet_thr=(float(np.mean(sonnet_acc)), float(np.std(sonnet_acc))),
        mcnemar_seed0={f'stream_global_vs_batch_global_{b}': mcnemar(y, preds0[f'stream_global_{b}'], preds0[f'batch_global_{b}'])
                       for b in (.3, .5)},
    )
    print(f"\n===== {name}  (Sonnet+thr all claims: {result['sonnet_thr'][0]:.3f})")
    print("  rho   test-call-rate   batch    stream")
    for b in BUDGETS:
        r, a = result['test_call_rate'][str(b)], result['acc']
        print(f"  {b:>4.0%}  {r[0]:>6.1%}±{r[1]:.1%}     {a[f'batch_global_{b}'][0]:.3f}    {a[f'stream_global_{b}'][0]:.3f}")
    print("  McNemar stream vs batch (seed 0):", {n: round(v, 3) for n, v in result['mcnemar_seed0'].items()})
    return result


if __name__ == '__main__':
    full = '--full' in sys.argv
    results = {name: evaluate(name, full) for name in DATASETS}
    with open((os.path.join(PAPER, 'streaming_results_full.json') if full else os.path.join(ANALYSIS, 'streaming_results.json')), 'w') as f:
        json.dump(results, f, indent=1)
