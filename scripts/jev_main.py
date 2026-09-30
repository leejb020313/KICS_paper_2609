"""Jev as the main (frontier) model: the main paper's cascade with Claude Sonnet 5 replaced by TypeSafe Jev.

The embedding SVM screens every sentence; the rho least certain (smallest |d|) go to Jev, whose P(yes) either
replaces the SVM decision (after a learning-set threshold) or is fused with the SVM margin by logistic regression
on (d, logit Jev) -- Eq. (1) of the main paper with s = Jev's score. Protocol as in the main paper: 5 x 80%
learning-set subsamples, every threshold/coefficient/call rate chosen on the learning set, McNemar on each run.

Methods:  svm | jev_raw, jev_thr (Jev on every sentence, 0.5 / learning-set threshold) | jev_nnppi (Jev on every
sentence + NN-PPI) | replace_{b}, fuse_{b} (SVM -> Jev cascades) | sonnet_fuse_0.5 (the main paper, for reference)

    uv run --locked python scripts/jev_main.py      -> results/jev_main_full.json
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.data import FULL_PARTS, RESULTS, Split, best_threshold, load_frontier, read_frontier_scores  # noqa: E402
from cwcascade.nnppi import nn_ppi  # noqa: E402
from final_eval import BUDGETS, SEEDS, compare, metrics, route, select_nnppi_k  # noqa: E402
from jev_eval import jev_scores, logit  # noqa: E402
from jev_strict import SATURATION, select_rate  # noqa: E402

DATASETS = {"clef": "clefcal", "cb": "cbcal"}


def lr():
    return LogisticRegression(class_weight='balanced')


def evaluate(name):
    test, calib, test_items = load_frontier(name, full=True)
    parts = (name,) + FULL_PARTS[name]
    jt = [jev_scores(p) for p in parts]
    lens = [sum(r["part"] == p for r in test_items) for p in parts]
    j_test = np.concatenate([[d[i] for i in range(n)] for d, n in zip(jt, lens)])
    jc = jev_scores(DATASETS[name])
    j_cal = np.array([jc[i] for i in sorted(read_frontier_scores()[DATASETS[name]])])
    y = test.y

    preds = {'jev_raw': [(j_test >= .5).astype(int)]}
    preds.update({k: [] for k in ['svm', 'jev_thr', 'jev_nnppi', 'sonnet_fuse_0.5'] +
                  [f'{v}_{b}' for b in BUDGETS for v in ('replace', 'fuse')]})
    curves = {v: {b: [] for b in BUDGETS} for v in ('replace', 'fuse')}
    coef = []
    for seed in range(SEEDS):
        idx = np.random.default_rng(seed).permutation(len(calib.y))[:int(0.8 * len(calib.y))]
        Xs, ys, ss, js = calib.X[idx], calib.y[idx], calib.s[idx], j_cal[idx]
        svm = SVC(class_weight='balanced', random_state=seed).fit(Xs, ys)
        d_test = svm.decision_function(test.X)
        d_cal = cross_val_predict(SVC(class_weight='balanced', random_state=seed), Xs, ys, cv=5, method='decision_function')
        thr_j = best_threshold(js, ys)
        p_svm, p_jev = (d_test > 0).astype(int), (j_test >= thr_j).astype(int)
        fuser = lr().fit(np.c_[d_cal, logit(js)], ys)
        coef.append([*fuser.coef_[0], fuser.intercept_[0]])
        p_fuse = fuser.predict(np.c_[d_test, logit(j_test)])
        order = np.argsort(np.abs(d_test))
        k = select_nnppi_k(Split(Xs, ys, js), seed)[0]
        preds['svm'].append(p_svm)
        preds['jev_thr'].append(p_jev)
        preds['jev_nnppi'].append((nn_ppi(j_test, test.X, js, Xs, ys, k) >= .5).astype(int))
        p_sfuse = lr().fit(np.c_[d_cal, ss], ys).predict(np.c_[d_test, test.s])
        preds['sonnet_fuse_0.5'].append(route(p_svm, p_sfuse, order, .5))
        for b in BUDGETS:
            preds[f'replace_{b}'].append(route(p_svm, p_jev, order, b))
            preds[f'fuse_{b}'].append(route(p_svm, p_fuse, order, b))

        # learning-set (out-of-fold) curves for choosing the call rate
        jev_oof = np.zeros(len(ys), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, ys):
            jev_oof[te] = (js[te] >= best_threshold(js[tr], ys[tr])).astype(int)
        fuse_oof = cross_val_predict(lr(), np.c_[d_cal, logit(js)], ys, cv=5)
        order_c = np.argsort(np.abs(d_cal))
        for b in BUDGETS:
            curves['replace'][b].append(float(np.mean(route((d_cal > 0).astype(int), jev_oof, order_c, b) == ys)))
            curves['fuse'][b].append(float(np.mean(route((d_cal > 0).astype(int), fuse_oof, order_c, b) == ys)))

    mean = {k: float(np.mean([np.mean(p == y) for p in v])) for k, v in preds.items()}
    summary = {k: {m: (float(np.mean([metrics(y, p)[m] for p in v])), float(np.std([metrics(y, p)[m] for p in v])))
                   for m in ('acc', 'prec1', 'rec1')} for k, v in preds.items()}
    curve = {v: {b: float(np.mean(c)) for b, c in cs.items()} for v, cs in curves.items()}
    rate = {v: select_rate(curve[v]) for v in curve}
    best = max(('jev_raw', 'jev_thr'), key=mean.get)
    reach = {v: next((b for b in BUDGETS if mean[f'{v}_{b}'] >= mean[best]), None) for v in ('replace', 'fuse')}

    def per_seed(a, b):
        rows = [compare(y, preds[a][min(sd, len(preds[a]) - 1)], preds[b][min(sd, len(preds[b]) - 1)]) for sd in range(SEEDS)]
        return dict(diff=[r['diff'] for r in rows], p=[r['mcnemar_p'] for r in rows], n_sig=sum(r['mcnemar_p'] < 0.05 for r in rows))

    fs = f"fuse_{rate['fuse']}"
    claims = {'jev_thr_vs_svm': ('jev_thr', 'svm'), 'fuse_sel_vs_best_jev': (fs, best), 'fuse_0.5_vs_best_jev': ('fuse_0.5', best),
              'fuse_sel_vs_jev_nnppi': (fs, 'jev_nnppi'), 'fuse_sel_vs_replace_same_rate': (fs, f"replace_{rate['fuse']}"),
              'fuse_sel_vs_sonnet_fuse_0.5': (fs, 'sonnet_fuse_0.5'), 'jev_nnppi_vs_best_jev': ('jev_nnppi', best)}
    for b in BUDGETS[1:]:
        claims[f'fuse_{b}_vs_replace_{b}'] = (f'fuse_{b}', f'replace_{b}')
        claims[f'fuse_{b}_vs_best_jev'] = (f'fuse_{b}', best)
    return dict(n_test=len(y), auc_jev=float(roc_auc_score(y, j_test)), auc_sonnet=float(roc_auc_score(y, test.s)),
                best_jev=best, rate=rate, reach=reach, learning_curve=curve, mean_acc=mean, metrics=summary,
                fuser_coef_seed_mean=np.mean(coef, axis=0).tolist(), tests={c: per_seed(a, b) for c, (a, b) in claims.items()})


def report(name, r):
    m = r['mean_acc']
    print(f"\n######## {name}: n={r['n_test']}  AUC Jev {r['auc_jev']:.3f} / Sonnet {r['auc_sonnet']:.3f};  "
          f"stronger all-call Jev = {r['best_jev']} ({m[r['best_jev']]:.3f})")
    print(f"  call rate chosen on the learning set: {r['rate']};  first rate reaching all-call Jev (test): {r['reach']}")
    for k in ('svm', 'jev_raw', 'jev_thr', 'jev_nnppi', 'sonnet_fuse_0.5'):
        met = r['metrics'][k]
        print(f"  {k:16s} acc {m[k]:.3f}  prec {met['prec1'][0]:.2f}  rec {met['rec1'][0]:.2f}")
    for v in ('replace', 'fuse'):
        print(f"  {v:8s} " + " ".join(f"{b:.0%}:{m[f'{v}_{b}']:.3f}" for b in BUDGETS))
        print(f"  {'(learn)':8s} " + " ".join(f"{b:.0%}:{r['learning_curve'][v][b]:.3f}" for b in BUDGETS))
    print(f"  fuser (a, b, c) mean over runs: {np.round(r['fuser_coef_seed_mean'], 2).tolist()}")
    for c, t in r['tests'].items():
        print(f"  {c:30s} diffs(pp) " + " ".join(f"{100 * d:+.1f}" for d in t['diff']) + "  p " +
              " ".join(f"{p:.3f}" for p in t['p']) + f"  sig {t['n_sig']}/5")


if __name__ == '__main__':
    out = {name: evaluate(name) for name in DATASETS}
    for name, r in out.items():
        report(name, r)
    with open(os.path.join(RESULTS, 'jev_main_full.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
