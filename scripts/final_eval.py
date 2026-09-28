"""Main experiment of the paper (Table 1, Figure 1).

Methods, all on identical test items:
  gemma_raw / gemma_nnppi_k{3,5,10} / gemma_nnppi_sel   Gemma 3 4B, raw and + NN-PPI (k chosen on calib)
  svm                                                   embedding RBF-SVM trained on calib labels, no LLM
  sonnet_raw / sonnet_thr                               Claude Sonnet 5 on all claims, thr 0.5 / calib-tuned thr
  sonnet_nnppi                                          Claude Sonnet 5 + NN-PPI on all claims (k chosen on calib)
  replace_{rho} / fuse_{rho}                            cascades that query the LLM on the rho most uncertain
                                                        claims and either replace the SVM decision with
                                                        sonnet_thr or fuse SVM margin + LLM score (Eq. 1)
  nnppi_{rho}                                           same routing, queried claims decided by sonnet_nnppi

Every calib-fitted component (SVM, fuser, threshold) is refit on 5 random 80% calib subsamples
drawn without replacement; nothing is tuned on test. Headline contrasts use McNemar tests and
paired bootstrap CIs on the seed-0 predictions.

Writes results/final_results.json. With --full, evaluates on the whole held-out sets instead
(CLEF dev + dev-test + official test, all of ClaimBuster 2016; see data/build_full_sets.py) and
writes results/final_results_full.json; Gemma rows are skipped where Gemma scores are missing.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.data import DATASETS, RESULTS, Split, best_threshold, load_frontier, load_gemma  # noqa: E402
from cwcascade.nnppi import nn_ppi  # noqa: E402

SEEDS = 5
BUDGETS = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
NNPPI_KS = (3, 5, 10)
BOOT = 2000


def metrics(y, p):
    return dict(acc=float(np.mean(p == y)), wf1=float(f1_score(y, p, average='weighted')), f1c1=float(f1_score(y, p)),
                prec1=float(precision_score(y, p, zero_division=0)), rec1=float(recall_score(y, p)))


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


def select_nnppi_k(c, seed=0):
    """Choose k on a 70/30 split of a calib set (70% neighbours, 30% scored)."""
    perm = np.random.default_rng(seed).permutation(len(c.y))
    pool, tune = perm[:int(0.7 * len(perm))], perm[int(0.7 * len(perm)):]
    tune_acc = {k: float(np.mean((nn_ppi(c.s[tune], c.X[tune], c.s[pool], c.X[pool], c.y[pool], k) >= .5) == c.y[tune]))
                for k in NNPPI_KS}
    return max(tune_acc, key=tune_acc.get), tune_acc


def evaluate(name, full=False):
    test, calib, test_items = load_frontier(name, full)
    gemma_test_s, gemma_cal = load_gemma(name, test_items)
    y = test.y

    preds, k_sel, tune_acc = {}, None, None
    if gemma_test_s is not None:
        preds['gemma_raw'] = [(gemma_test_s >= .5).astype(int)]
        for k in NNPPI_KS:
            theta = nn_ppi(gemma_test_s, test.X, gemma_cal.s, gemma_cal.X, gemma_cal.y, k)
            preds[f'gemma_nnppi_k{k}'] = [(theta >= .5).astype(int)]
        k_sel, tune_acc = select_nnppi_k(gemma_cal)
        preds['gemma_nnppi_sel'] = preds[f'gemma_nnppi_k{k_sel}']
    preds['sonnet_raw'] = [(test.s >= .5).astype(int)]
    for key in ['fuse_sel', 'svm', 'sonnet_thr', 'sonnet_nnppi'] + [f'{v}_{b}' for b in BUDGETS for v in ('fuse', 'replace', 'nnppi')]:
        preds[key] = []
    rho_sel, calib_curve, sonnet_k = [], {}, []

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
        # the NN-PPI paper's strongest setting: frontier LLM + NN-PPI on all claims, decision at 0.5
        k = select_nnppi_k(Split(Xs, ys, ss), seed)[0]
        sonnet_k.append(k)
        p_sonnet_nnppi = (nn_ppi(test.s, test.X, ss, Xs, ys, k) >= .5).astype(int)
        preds['sonnet_nnppi'].append(p_sonnet_nnppi)
        for b in BUDGETS:
            preds[f'fuse_{b}'].append(route(p_svm, p_fuse, order, b))
            preds[f'replace_{b}'].append(route(p_svm, p_sonnet, order, b))
            preds[f'nnppi_{b}'].append(route(p_svm, p_sonnet_nnppi, order, b))
        if seed == 0:
            # on the half the SVM is most confident about: correct SVM decisions each cascade breaks / wrong ones it fixes
            conf = order[len(y) // 2:]
            ok = p_svm[conf] == y[conf]
            flips = {v: dict(broke=int((ok & (p[conf] != y[conf])).sum()), fixed=int((~ok & (p[conf] == y[conf])).sum()))
                     for v, p in (('replace', p_sonnet), ('fuse', p_fuse), ('nnppi', p_sonnet_nnppi))}

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
        contrasts[f'fuse_{b}_vs_nnppi_{b}'] = compare(y, first[f'fuse_{b}'], first[f'nnppi_{b}'])
    for sd in range(SEEDS):
        contrasts[f'fuse_0.5_vs_fuse_1.0_seed{sd}'] = compare(y, preds['fuse_0.5'][sd], preds['fuse_1.0'][sd])
    contrasts['fuse_0.5_vs_sonnet_nnppi'] = compare(y, first['fuse_0.5'], first['sonnet_nnppi'])
    contrasts['fuse_sel_vs_sonnet_thr'] = compare(y, first['fuse_sel'], first['sonnet_thr'])
    contrasts['nnppi_0.5_vs_sonnet_nnppi'] = compare(y, first['nnppi_0.5'], first['sonnet_nnppi'])
    if 'gemma_raw' in preds:
        contrasts['svm_vs_gemma_nnppi_sel'] = compare(y, first['svm'], first['gemma_nnppi_sel'])
        best_nnppi = max((first[f'gemma_nnppi_k{k}'] for k in NNPPI_KS), key=lambda p: np.mean(p == y))
        contrasts['svm_vs_best_gemma_nnppi'] = compare(y, first['svm'], best_nnppi)

    # the same comparisons within each part of a full held-out set (e.g. CLEF official test alone)
    parts = np.array([r['part'] for r in test_items])
    by_part = {}
    for part in dict.fromkeys(parts) if full else ():
        m = parts == part
        by_part[part] = dict(n=int(m.sum()), n_pos=int(y[m].sum()),
                             acc={key: float(np.mean([np.mean(p[m] == y[m]) for p in plist])) for key, plist in preds.items()},
                             contrasts={f'fuse_0.5_vs_{o}': compare(y[m], first['fuse_0.5'][m], first[o][m])
                                        for o in ('sonnet_thr', 'sonnet_nnppi', 'nnppi_0.5', 'replace_0.5')})

    result = dict(by_part=by_part, calib_curve={n: (float(np.mean(v)), float(np.std(v))) for n, v in calib_curve.items()},
                  k_sel=k_sel, nnppi_tune_acc=tune_acc, sonnet_nnppi_k=sonnet_k, flips_confident_half_seed0=flips, rho_sel=rho_sel, n_test=len(y),
                  n_calib_frontier=len(calib.y), n_calib_gemma=len(gemma_cal.y) if gemma_cal else None, metrics=summary, contrasts=contrasts)
    report(name, result)
    return result


def report(name, r):
    m = r['metrics']
    print(f"\n===== {name}: test n={r['n_test']}, frontier-scored calib n={r['n_calib_frontier']}, seeds={SEEDS}")
    print('  calib OOF curve:', {n: round(v[0], 3) for n, v in r['calib_curve'].items()})
    print(f"  NN-PPI k selected on calib: k={r['k_sel']} (tune acc {r['nnppi_tune_acc']}); rho selected per seed: {r['rho_sel']}")
    for key in ['gemma_raw', 'gemma_nnppi_k3', 'gemma_nnppi_k5', 'gemma_nnppi_k10', 'gemma_nnppi_sel', 'svm', 'sonnet_raw', 'sonnet_thr', 'sonnet_nnppi', 'fuse_sel']:
        if key not in m:
            continue
        print(f"  {key:18s} acc={m[key]['acc'][0]:.3f}±{m[key]['acc'][1]:.3f}  wF1={m[key]['wf1'][0]:.3f}  F1c1={m[key]['f1c1'][0]:.3f}")
    print("  budget   FUSE acc (±sd)     REPLACE acc (±sd)   NNPPI-cascade acc (±sd)")
    for b in BUDGETS:
        print(f"  {b:>5.0%}    {m[f'fuse_{b}']['acc'][0]:.3f}±{m[f'fuse_{b}']['acc'][1]:.3f}      "
              f"{m[f'replace_{b}']['acc'][0]:.3f}±{m[f'replace_{b}']['acc'][1]:.3f}      "
              f"{m[f'nnppi_{b}']['acc'][0]:.3f}±{m[f'nnppi_{b}']['acc'][1]:.3f}")
    print(f"  flips on SVM-confident half (seed 0): {r['flips_confident_half_seed0']}")
    for c, v in r['contrasts'].items():
        print(f"  {c:32s} diff={v['diff']:+.3f} CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] McNemar p={v['mcnemar_p']:.4f}")
    for part, pr in r['by_part'].items():
        a = pr['acc']
        print(f"  -- part {part}: n={pr['n']} (pos {pr['n_pos']})  " + "  ".join(
            f"{key}={a[key]:.3f}" for key in ('svm', 'sonnet_thr', 'sonnet_nnppi', 'replace_0.5', 'nnppi_0.5', 'fuse_0.5', 'fuse_1.0')))
        for c, v in pr['contrasts'].items():
            print(f"     {c:28s} diff={v['diff']:+.3f} CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] McNemar p={v['mcnemar_p']:.4f}")


if __name__ == '__main__':
    full = '--full' in sys.argv
    results = {name: evaluate(name, full) for name in DATASETS}
    with open(os.path.join(RESULTS, 'final_results_full.json' if full else 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=1)
