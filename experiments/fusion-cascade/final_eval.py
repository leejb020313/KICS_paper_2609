"""Final evaluation for the fused-cascade paper.

Methods on identical test items:
  Gemma 3 4B raw / + NN-PPI (project implementation, k in {3,5,10})
  Emb-SVM (all-MiniLM-L6-v2 + calib labels, no LLM)
  Sonnet raw (thr .5) / Sonnet + calib-tuned threshold
  Cascade at budget rho: REPLACE (Sonnet+thr decides) vs FUSE (logreg over SVM margin + Sonnet score)

Robustness: every calib-fitted component is refit under S random calib splits
(stacker/threshold never see test); test uncertainty via paired bootstrap.
Writes final_results.json for the figure script.
"""
import glob
import json
import os
import re

import numpy as np
from scipy.stats import binomtest
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

import sys; sys.path.insert(0, '../..')
from src.nnppi.calibration import NNPPIConfig, nn_ppi_apply

SEEDS = 5
BUDGETS = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
BOOT = 2000

ev = json.load(open('frontier_eval_set.json', encoding='utf-8'))
cs = json.load(open('frontier_calib_set.json', encoding='utf-8'))


def read_scores(pattern):
    out = {}
    for f in glob.glob(pattern):
        k = os.path.basename(f).split('_')[0]
        t = open(f, encoding='utf-8', errors='ignore').read()
        out.setdefault(k, {}).update({int(a): float(b) for a, b in json.loads(re.search(r'\{.*\}', t, re.S).group(0)).items()})
    return out


fs = read_scores('batches/*.out')
emb = SentenceTransformer('all-MiniLM-L6-v2')


def E(x):
    return np.asarray(emb.encode(x, normalize_embeddings=True, show_progress_bar=False))


def load(f):
    return [json.loads(l) for l in open('../../results/' + f + '.jsonl', encoding='utf-8') if l.strip()]


def best_thr(s, y):
    ts = np.linspace(0.02, 0.98, 97)
    return ts[np.argmax([np.mean((s >= t) == y) for t in ts])]


def metrics(y, p):
    return dict(acc=float(np.mean(p == y)), wf1=float(f1_score(y, p, average='weighted')), f1c1=float(f1_score(y, p)))


results = {}
for k, ck, cal, test in [('clef', 'clefcal', 'clef_calib_scores', 'clef_test_scores'),
                         ('cb', 'cbcal', 'claimbuster_calib_scores', 'claimbuster_test_scores')]:
    T = ev[k]
    y = np.array([r['label'] for r in T])
    st = np.array([fs[k][i] for i in range(len(T))])
    Xt = E([r['text'] for r in T])

    # calib: only items the frontier scored (all, once the CLEF rerun finished)
    have = sorted(fs[ck])
    C = [cs[ck][i] for i in have]
    yc = np.array([r['label'] for r in C])
    sc = np.array([fs[ck][i] for i in have])
    Xc = E([r['text'] for r in C])

    # Gemma baselines use the full Gemma-scored calib set (the paper's setting)
    GC = [r for r in load(cal) if r.get('parse_ok')]
    gsc = np.array([r['confidence_score'] for r in GC], float)
    gyc = np.array([r['label'] for r in GC])
    gXc = E([r['text'] for r in GC])
    gem = {str(r['Sentence_id']): r for r in load(test)}
    gst = np.array([gem[r['Sentence_id']]['confidence_score'] if gem[r['Sentence_id']].get('parse_ok') else 0.0 for r in T], float)

    preds = {'gemma_raw': [(gst >= .5).astype(int)]}
    for kk in (3, 5, 10):
        preds[f'gemma_nnppi_k{kk}'] = [(nn_ppi_apply(gst, Xt, gsc, gXc, gyc, NNPPIConfig(k=kk)).theta >= .5).astype(int)]
    # k chosen WITHOUT the test set: 70/30 split of the Gemma calib set (pool = neighbours, tune = scoring)
    gperm = np.random.default_rng(0).permutation(len(GC)); gcut = int(0.7 * len(GC))
    gpool, gtune = gperm[:gcut], gperm[gcut:]
    tune_acc = {kk: float(np.mean((nn_ppi_apply(gsc[gtune], gXc[gtune], gsc[gpool], gXc[gpool], gyc[gpool], NNPPIConfig(k=kk)).theta >= .5) == gyc[gtune]))
                for kk in (3, 5, 10)}
    k_sel = max(tune_acc, key=tune_acc.get)
    preds['gemma_nnppi_sel'] = [preds[f'gemma_nnppi_k{k_sel}'][0]]
    preds['sonnet_raw'] = [(st >= .5).astype(int)]
    preds['fuse_sel'] = []
    rho_sel = []
    calib_curve = {}
    preds['svm'] = []
    preds['sonnet_thr'] = []
    for b in BUDGETS:
        preds[f'fuse_{b}'] = []
        preds[f'replace_{b}'] = []

    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        # subsample 80% of calib WITHOUT replacement: bootstrap duplicates would leak
        # across the stacker's internal CV folds and inflate SVM confidence
        idx = rng.permutation(len(C))[:int(0.8 * len(C))]
        Xs, ys, ss = Xc[idx], yc[idx], sc[idx]
        svm = SVC(class_weight='balanced', random_state=seed).fit(Xs, ys)
        dt = svm.decision_function(Xt)
        dcal = cross_val_predict(SVC(class_weight='balanced', random_state=seed), Xs, ys, cv=5, method='decision_function')
        lr = LogisticRegression(class_weight='balanced').fit(np.c_[dcal, ss], ys)
        thr = best_thr(ss, ys)
        p_svm = (dt > 0).astype(int)
        p_fuse_all = lr.predict(np.c_[dt, st])
        p_son = (st >= thr).astype(int)
        preds['svm'].append(p_svm)
        preds['sonnet_thr'].append(p_son)
        order = np.argsort(np.abs(dt))
        # budget chosen WITHOUT the test set: smallest rho whose out-of-fold calib accuracy
        # reaches the calib accuracy of the frontier LLM on all claims (+threshold)
        p_fuse_cal = cross_val_predict(LogisticRegression(class_weight='balanced'), np.c_[dcal, ss], ys, cv=5)
        p_svm_cal = (dcal > 0).astype(int); order_cal = np.argsort(np.abs(dcal))
        target = np.mean((ss >= thr) == ys)
        rho = 1.0
        for b in BUDGETS:
            pc = p_svm_cal.copy(); sc_ = order_cal[:int(round(b * len(ys)))]; pc[sc_] = p_fuse_cal[sc_]
            if np.mean(pc == ys) >= target:
                rho = b
                break
        rho_sel.append(rho)
        # full out-of-fold calib curve (fused) and OOF-threshold frontier reference, for reporting
        from sklearn.model_selection import StratifiedKFold
        son_oof = np.zeros(len(ys), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, ys):
            son_oof[te] = (ss[te] >= best_thr(ss[tr], ys[tr])).astype(int)
        calib_curve.setdefault('sonnet_thr_oof', []).append(float(np.mean(son_oof == ys)))
        calib_curve.setdefault('svm_oof', []).append(float(np.mean(p_svm_cal == ys)))
        for b in BUDGETS:
            pc = p_svm_cal.copy(); sc_ = order_cal[:int(round(b * len(ys)))]; pc[sc_] = p_fuse_cal[sc_]
            calib_curve.setdefault(f'fuse_{b}', []).append(float(np.mean(pc == ys)))
        sel = order[:int(round(rho * len(y)))]
        pf = p_svm.copy(); pf[sel] = p_fuse_all[sel]
        preds['fuse_sel'].append(pf)
        for b in BUDGETS:
            sel = order[:int(round(b * len(y)))]
            pf = p_svm.copy(); pf[sel] = p_fuse_all[sel]
            pr = p_svm.copy(); pr[sel] = p_son[sel]
            preds[f'fuse_{b}'].append(pf)
            preds[f'replace_{b}'].append(pr)

    R = {}
    for name, plist in preds.items():
        ms = [metrics(y, p) for p in plist]
        R[name] = {m: (float(np.mean([x[m] for x in ms])), float(np.std([x[m] for x in ms]))) for m in ms[0]}

    # paired bootstrap on test for headline contrasts (seed 0 predictions)
    def boot(a, b):
        ca, cb_ = (a == y), (b == y)
        g = np.random.default_rng(0)
        d = [np.mean(ca[i]) - np.mean(cb_[i]) for i in (g.integers(0, len(y), len(y)) for _ in range(BOOT))]
        n01, n10 = int((ca & ~cb_).sum()), int((~ca & cb_).sum())
        return dict(diff=float(np.mean(ca) - np.mean(cb_)), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    mcnemar_p=float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0)

    contrasts = {}
    for b in (.2, .3, .4, .5, 1.0):
        contrasts[f'fuse_{b}_vs_sonnet_thr'] = boot(preds[f'fuse_{b}'][0], preds['sonnet_thr'][0])
        contrasts[f'fuse_{b}_vs_replace_{b}'] = boot(preds[f'fuse_{b}'][0], preds[f'replace_{b}'][0])
    contrasts['fuse_sel_vs_sonnet_thr'] = boot(preds['fuse_sel'][0], preds['sonnet_thr'][0])
    contrasts['svm_vs_gemma_nnppi_sel'] = boot(preds['svm'][0], preds['gemma_nnppi_sel'][0])
    contrasts['svm_vs_best_gemma_nnppi'] = boot(preds['svm'][0], max((preds[f'gemma_nnppi_k{kk}'][0] for kk in (3, 5, 10)), key=lambda p: np.mean(p == y)))
    results[k] = dict(calib_curve={n: (float(np.mean(v)), float(np.std(v))) for n, v in calib_curve.items()}, k_sel=k_sel, nnppi_tune_acc=tune_acc, rho_sel=rho_sel, n_test=len(y), n_calib_frontier=len(C), n_calib_gemma=len(GC), metrics=R, contrasts=contrasts)

    print(f"\n===== {k}: test n={len(y)}, frontier-scored calib n={len(C)}, seeds={SEEDS}")
    print('  CALIB OOF curve:', {n: round(v[0], 3) for n, v in {n: (np.mean(v), 0) for n, v in calib_curve.items()}.items()})
    print(f"  NN-PPI k selected on calib holdout: k={k_sel} (tune acc {tune_acc});  rho selected on calib per seed: {rho_sel}")
    for name in ['gemma_raw', 'gemma_nnppi_k3', 'gemma_nnppi_k5', 'gemma_nnppi_k10', 'gemma_nnppi_sel', 'svm', 'sonnet_raw', 'sonnet_thr', 'fuse_sel']:
        r = R[name]
        print(f"  {name:18s} acc={r['acc'][0]:.3f}±{r['acc'][1]:.3f}  wF1={r['wf1'][0]:.3f}  F1c1={r['f1c1'][0]:.3f}")
    print("  budget   FUSE acc (±sd)     REPLACE acc (±sd)")
    for b in BUDGETS:
        print(f"  {b:>5.0%}    {R[f'fuse_{b}']['acc'][0]:.3f}±{R[f'fuse_{b}']['acc'][1]:.3f}      {R[f'replace_{b}']['acc'][0]:.3f}±{R[f'replace_{b}']['acc'][1]:.3f}")
    for c, v in contrasts.items():
        print(f"  {c:32s} diff={v['diff']:+.3f} CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] McNemar p={v['mcnemar_p']:.4f}")

json.dump(results, open('final_results.json', 'w'), indent=1)
