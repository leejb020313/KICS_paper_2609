"""Jev as the cheap stage, under the protocol of the main paper (the 09-28 jev_eval.py read call rates off the test curve).

Methods (same test items, same 5 x 80% learning-set subsamples, same models as scripts/jev_eval.py):
  svm, jev_thr                cheap stages alone (Jev: P(yes) with a learning-set threshold)
  sonnet_raw, sonnet_thr      the frontier LLM on every sentence; the stronger of the two is the all-call baseline
  sonnet_nnppi                all-call LLM + NN-PPI (the NN-PPI paper's strongest setting)
  fuse_{b}                    the main paper: SVM -> Sonnet, queried sentences fused on (SVM margin, s)
  jevrep_{b} / jevfuse_{b}    Jev -> Sonnet: queried sentences replaced by sonnet_thr / fused on (logit Jev, s)
  tri_{b}                     (SVM margin, logit Jev) -> Sonnet, queried sentences fused on all three

Call rate: for every cascade, chosen on the learning set only, as the smallest rate after which the out-of-fold
learning-set accuracy rises by at most 0.1 pp (mean over the 5 subsamples). The same rule is applied to the main
paper's SVM cascade as a check. Significance: exact McNemar on each of the 5 runs.

    uv run --locked python scripts/jev_strict.py      -> results/jev_strict_full.json
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.data import FULL_PARTS, RESULTS, Split, best_threshold, load_frontier, read_frontier_scores  # noqa: E402
from cwcascade.nnppi import nn_ppi  # noqa: E402
from final_eval import BUDGETS, SEEDS, compare, metrics, route, select_nnppi_k  # noqa: E402
from jev_eval import jev_scores, logit  # noqa: E402

DATASETS = {"clef": "clefcal", "cb": "cbcal"}  # the two datasets of the main paper (Jev scored only these)
CASCADES = ("fuse", "jevrep", "jevfuse", "tri")
SATURATION = 0.001  # "rises by at most 0.1 pp"


def lr():
    return LogisticRegression(class_weight='balanced')


def select_rate(curve):
    """Smallest call rate after which the learning-set accuracy rises by at most SATURATION."""
    return next(b for b in BUDGETS if max(curve[c] for c in BUDGETS if c >= b) - curve[b] <= SATURATION)


def evaluate(name):
    test, calib, test_items = load_frontier(name, full=True)
    calib_key = DATASETS[name]
    parts = (name,) + FULL_PARTS[name]
    jt = [jev_scores(p) for p in parts]
    lens = [sum(r["part"] == p for r in test_items) for p in parts]
    j_test = np.concatenate([[d[i] for i in range(n)] for d, n in zip(jt, lens)])
    jc = jev_scores(calib_key)
    j_cal = np.array([jc[i] for i in sorted(read_frontier_scores()[calib_key])])
    y = test.y

    preds = {'sonnet_raw': [(test.s >= .5).astype(int)]}
    preds.update({k: [] for k in ['svm', 'jev_thr', 'sonnet_thr', 'sonnet_nnppi'] +
                  [f'{v}_{b}' for b in BUDGETS for v in CASCADES]})
    curves = {v: {b: [] for b in BUDGETS} for v in CASCADES}
    for seed in range(SEEDS):
        idx = np.random.default_rng(seed).permutation(len(calib.y))[:int(0.8 * len(calib.y))]
        Xs, ys, ss, js = calib.X[idx], calib.y[idx], calib.s[idx], j_cal[idx]
        svm = SVC(class_weight='balanced', random_state=seed).fit(Xs, ys)
        d_test = svm.decision_function(test.X)
        d_cal = cross_val_predict(SVC(class_weight='balanced', random_state=seed), Xs, ys, cv=5, method='decision_function')
        thr_s, thr_j = best_threshold(ss, ys), best_threshold(js, ys)
        lj_cal, lj_test = logit(js), logit(j_test)

        # test predictions (identical to jev_eval.py)
        p_svm, p_jev = (d_test > 0).astype(int), (j_test >= thr_j).astype(int)
        p_sonnet = (test.s >= thr_s).astype(int)
        p_fuse = lr().fit(np.c_[d_cal, ss], ys).predict(np.c_[d_test, test.s])
        p_jevfuse = lr().fit(np.c_[lj_cal, ss], ys).predict(np.c_[lj_test, test.s])
        cheap = lr().fit(np.c_[d_cal, lj_cal], ys)
        m_cheap = cheap.decision_function(np.c_[d_test, lj_test])
        p_tri = lr().fit(np.c_[d_cal, lj_cal, ss], ys).predict(np.c_[d_test, lj_test, test.s])
        k = select_nnppi_k(Split(Xs, ys, ss), seed)[0]
        preds['svm'].append(p_svm)
        preds['jev_thr'].append(p_jev)
        preds['sonnet_thr'].append(p_sonnet)
        preds['sonnet_nnppi'].append((nn_ppi(test.s, test.X, ss, Xs, ys, k) >= .5).astype(int))
        base = {'fuse': p_svm, 'jevrep': p_jev, 'jevfuse': p_jev, 'tri': (m_cheap > 0).astype(int)}
        alt = {'fuse': p_fuse, 'jevrep': p_sonnet, 'jevfuse': p_jevfuse, 'tri': p_tri}
        order = {'fuse': np.argsort(np.abs(d_test)), 'jevrep': np.argsort(np.abs(lj_test - logit(thr_j))),
                 'jevfuse': np.argsort(np.abs(lj_test - logit(thr_j))), 'tri': np.argsort(np.abs(m_cheap))}
        for b in BUDGETS:
            for v in CASCADES:
                preds[f'{v}_{b}'].append(route(base[v], alt[v], order[v], b))

        # the same cascades out-of-fold on the learning set, for choosing the call rate
        sonnet_oof = np.zeros(len(ys), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, ys):
            sonnet_oof[te] = (ss[te] >= best_threshold(ss[tr], ys[tr])).astype(int)
        m_cheap_cal = cross_val_predict(lr(), np.c_[d_cal, lj_cal], ys, cv=5, method='decision_function')
        base_c = {'fuse': (d_cal > 0).astype(int), 'jevrep': (js >= thr_j).astype(int),
                  'jevfuse': (js >= thr_j).astype(int), 'tri': (m_cheap_cal > 0).astype(int)}
        alt_c = {'fuse': cross_val_predict(lr(), np.c_[d_cal, ss], ys, cv=5), 'jevrep': sonnet_oof,
                 'jevfuse': cross_val_predict(lr(), np.c_[lj_cal, ss], ys, cv=5),
                 'tri': cross_val_predict(lr(), np.c_[d_cal, lj_cal, ss], ys, cv=5)}
        order_c = {'fuse': np.argsort(np.abs(d_cal)), 'jevrep': np.argsort(np.abs(lj_cal - logit(thr_j))),
                   'jevfuse': np.argsort(np.abs(lj_cal - logit(thr_j))), 'tri': np.argsort(np.abs(m_cheap_cal))}
        for v in CASCADES:
            for b in BUDGETS:
                curves[v][b].append(float(np.mean(route(base_c[v], alt_c[v], order_c[v], b) == ys)))

    mean = {k: float(np.mean([np.mean(p == y) for p in v])) for k, v in preds.items()}
    summary = {k: {m: (float(np.mean([metrics(y, p)[m] for p in v])), float(np.std([metrics(y, p)[m] for p in v])))
                   for m in ('acc', 'prec1', 'rec1')} for k, v in preds.items()}
    curve = {v: {b: float(np.mean(c)) for b, c in cs.items()} for v, cs in curves.items()}
    rate = {v: select_rate(curve[v]) for v in CASCADES}
    best = max(('sonnet_raw', 'sonnet_thr'), key=mean.get)
    reach = {v: next((b for b in BUDGETS if mean[f'{v}_{b}'] >= mean[best]), None) for v in CASCADES}

    def per_seed(a, b):
        rows = [compare(y, preds[a][min(sd, len(preds[a]) - 1)], preds[b][min(sd, len(preds[b]) - 1)]) for sd in range(SEEDS)]
        return dict(diff=[r['diff'] for r in rows], p=[r['mcnemar_p'] for r in rows],
                    n_sig=sum(r['mcnemar_p'] < 0.05 for r in rows))

    sel = {v: f'{v}_{rate[v]}' for v in CASCADES}
    claims = {'jev_thr_vs_svm': ('jev_thr', 'svm'), 'jev_thr_vs_best_full': ('jev_thr', best),
              'jevfuse_sel_vs_best_full': (sel['jevfuse'], best), 'jevfuse_sel_vs_sonnet_nnppi': (sel['jevfuse'], 'sonnet_nnppi'),
              'jevfuse_sel_vs_fuse_0.5': (sel['jevfuse'], 'fuse_0.5'), 'jevfuse_sel_vs_fuse_sel': (sel['jevfuse'], sel['fuse']),
              'tri_sel_vs_jevfuse_sel': (sel['tri'], sel['jevfuse']), 'tri_sel_vs_best_full': (sel['tri'], best),
              'jevfuse_sel_vs_jevrep_sel': (sel['jevfuse'], f"jevrep_{rate['jevfuse']}")}
    for b in BUDGETS[1:]:
        claims[f'jevfuse_{b}_vs_jevrep_{b}'] = (f'jevfuse_{b}', f'jevrep_{b}')
    # Table 1 of the JEV paper: the proposed row (JEV -> LLM fused at its selected rate) against every other row
    for row in ('svm', 'jev_thr', 'sonnet_raw', 'sonnet_thr', 'sonnet_nnppi', sel['fuse'], f"jevrep_{rate['jevfuse']}"):
        claims[f'table:{row}'] = (sel['jevfuse'], row)
    tests = {c: per_seed(a, b) for c, (a, b) in claims.items()}
    return dict(n_test=len(y), best_full=best, rate=rate, reach=reach, learning_curve=curve,
                mean_acc=mean, metrics=summary, tests=tests)


def report(name, r):
    m = r['mean_acc']
    print(f"\n######## {name}: n={r['n_test']}; stronger all-call LLM = {r['best_full']} ({m[r['best_full']]:.3f})")
    print(f"  call rate chosen on the learning set: {r['rate']};  first rate reaching all-call (test): {r['reach']}")
    for k in ('svm', 'jev_thr', 'sonnet_raw', 'sonnet_thr', 'sonnet_nnppi'):
        print(f"  {k:14s} {m[k]:.3f}")
    for v in CASCADES:
        print(f"  {v:8s} " + " ".join(f"{b:.0%}:{m[f'{v}_{b}']:.3f}" for b in BUDGETS) + f"   <- selected {r['rate'][v]:.0%}")
    print("  learning-set curve: " + "; ".join(f"{v} " + " ".join(f"{r['learning_curve'][v][b]:.3f}" for b in BUDGETS) for v in CASCADES))
    for c, t in r['tests'].items():
        print(f"  {c:32s} diffs(pp) " + " ".join(f"{100 * d:+.1f}" for d in t['diff']) + "  p " +
              " ".join(f"{p:.3f}" for p in t['p']) + f"  sig {t['n_sig']}/5")


if __name__ == '__main__':
    out = {name: evaluate(name) for name in DATASETS}
    for name, r in out.items():
        report(name, r)
    with open(os.path.join(RESULTS, 'jev_strict_full.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
