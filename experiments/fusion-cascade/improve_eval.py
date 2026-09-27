"""Two cascade refinements, evaluated with the same 5 calib subsamples as final_eval.py.

(1) Streaming router: instead of taking the top-rho most uncertain items of the whole
    TEST batch (transductive), fix a threshold tau on |d| from the calib out-of-fold
    decisions so that rho of CALIB items fall below it; route a test item iff |d(x)| < tau.
    The realised test call rate is reported, since it no longer equals rho exactly.
(2) Region-trained fuser: fit the logistic fuser only on calib items inside the routed
    region (|d| < tau) instead of on all calib items.

Pre-declared selection rule (no test data): for each seed, compare the calib out-of-fold
accuracy of the cascade at rho=0.5 with the global fuser vs the region fuser; the variant
with higher calib accuracy (ties -> global, the simpler one) is the "selected" variant.
Writes improve_results.json.
"""
import glob
import json
import os
import re

import numpy as np
from scipy.stats import binomtest
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

SEEDS = 5
BUDGETS = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

ev = json.load(open('frontier_eval_set.json', encoding='utf-8'))
cs = json.load(open('frontier_calib_set.json', encoding='utf-8'))
fs = {}
for f in glob.glob('batches/*.out'):
    k = os.path.basename(f).split('_')[0]
    t = open(f, encoding='utf-8', errors='ignore').read()
    fs.setdefault(k, {}).update({int(a): float(b) for a, b in json.loads(re.search(r'\{.*\}', t, re.S).group(0)).items()})
emb = SentenceTransformer('all-MiniLM-L6-v2')


def E(x):
    return np.asarray(emb.encode(x, normalize_embeddings=True, show_progress_bar=False))


def best_thr(s, y):
    ts = np.linspace(0.02, 0.98, 97)
    return ts[np.argmax([np.mean((s >= t) == y) for t in ts])]


def fit_lr(X, y):
    # a region can be single-class for tiny rho; fall back to None -> caller uses the global fuser
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None
    return LogisticRegression(class_weight='balanced').fit(X, y)


def oof_region_pred(Xf, y, mask, seed):
    """Out-of-fold predictions of a fuser trained only on the region `mask` (calib)."""
    out = np.zeros(len(y), int)
    idx = np.where(mask)[0]
    if len(idx) < 20 or len(np.unique(y[idx])) < 2:
        return None
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xf[idx], y[idx]):
        m = LogisticRegression(class_weight='balanced').fit(Xf[idx][tr], y[idx][tr])
        out[idx[te]] = m.predict(Xf[idx][te])
    return out


results = {}
for k, ck in [('clef', 'clefcal'), ('cb', 'cbcal')]:
    T = ev[k]
    y = np.array([r['label'] for r in T])
    st = np.array([fs[k][i] for i in range(len(T))])
    Xt = E([r['text'] for r in T])
    have = sorted(fs[ck])
    C = [cs[ck][i] for i in have]
    yc = np.array([r['label'] for r in C])
    sc = np.array([fs[ck][i] for i in have])
    Xc = E([r['text'] for r in C])

    rec = {f'{v}_{b}': [] for v in ('batch_global', 'stream_global', 'stream_region', 'stream_sel') for b in BUDGETS}
    rate = {b: [] for b in BUDGETS}
    sel_choice, calib_acc50, son_acc, preds0 = [], [], [], {}
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(C))[:int(0.8 * len(C))]
        Xs, ys, ss = Xc[idx], yc[idx], sc[idx]
        svm = SVC(class_weight='balanced', random_state=seed).fit(Xs, ys)
        dt = svm.decision_function(Xt)
        dcal = cross_val_predict(SVC(class_weight='balanced', random_state=seed), Xs, ys, cv=5, method='decision_function')
        Fc, Ft = np.c_[dcal, ss], np.c_[dt, st]
        g = LogisticRegression(class_weight='balanced').fit(Fc, ys)
        p_svm, p_glob = (dt > 0).astype(int), g.predict(Ft)
        p_glob_cal = cross_val_predict(LogisticRegression(class_weight='balanced'), Fc, ys, cv=5)
        son_acc.append(float(np.mean((st >= best_thr(ss, ys)) == y)))
        order = np.argsort(np.abs(dt))
        for b in BUDGETS:
            tau = np.quantile(np.abs(dcal), b)
            m_cal, m_te = np.abs(dcal) < tau, np.abs(dt) < tau
            rate[b].append(float(m_te.mean()))
            r = fit_lr(Fc[m_cal], ys[m_cal])
            p_reg = r.predict(Ft) if r is not None else p_glob
            # (a) original transductive batch router, global fuser (reference)
            pb = p_svm.copy(); s_ = order[:int(round(b * len(y)))]; pb[s_] = p_glob[s_]
            # (1) streaming router, global fuser
            pg = p_svm.copy(); pg[m_te] = p_glob[m_te]
            # (1)+(2) streaming router, region fuser
            pr = p_svm.copy(); pr[m_te] = p_reg[m_te]
            # calib-only selection between global and region fuser (decided at rho=0.5, applied to all rho)
            if b == .5:
                base = (dcal > 0).astype(int)
                cg = base.copy(); cg[m_cal] = p_glob_cal[m_cal]
                oof_r = oof_region_pred(Fc, ys, m_cal, seed)
                cr = base.copy()
                if oof_r is not None:
                    cr[m_cal] = oof_r[m_cal]
                acc_g, acc_r = float(np.mean(cg == ys)), float(np.mean(cr == ys))
                calib_acc50.append((acc_g, acc_r))
                sel_choice.append('region' if acc_r > acc_g else 'global')
            for v, p in (('batch_global', pb), ('stream_global', pg), ('stream_region', pr)):
                rec[f'{v}_{b}'].append(float(np.mean(p == y)))
                if seed == 0:
                    preds0[f'{v}_{b}'] = p
        # apply the choice made at rho=0.5 for this seed
        for b in BUDGETS:
            key = 'stream_region' if sel_choice[-1] == 'region' else 'stream_global'
            rec[f'stream_sel_{b}'].append(rec[f'{key}_{b}'][-1])

    def mc(a, b_):
        ca, cb_ = (a == y), (b_ == y)
        n01, n10 = int((ca & ~cb_).sum()), int((~ca & cb_).sum())
        return float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0

    results[k] = dict(
        acc={n: (float(np.mean(v)), float(np.std(v))) for n, v in rec.items()},
        test_call_rate={str(b): (float(np.mean(v)), float(np.std(v))) for b, v in rate.items()},
        sonnet_thr=(float(np.mean(son_acc)), float(np.std(son_acc))),
        calib_acc50_global_vs_region=calib_acc50, selected=sel_choice,
        mcnemar_seed0={f'stream_global_vs_batch_global_{b}': mc(preds0[f'stream_global_{b}'], preds0[f'batch_global_{b}']) for b in (.3, .5)}
        | {f'stream_region_vs_stream_global_{b}': mc(preds0[f'stream_region_{b}'], preds0[f'stream_global_{b}']) for b in (.3, .5)},
    )
    R = results[k]
    print(f"\n===== {k}  (Sonnet+thr all claims: {R['sonnet_thr'][0]:.3f})")
    print(f"  calib-OOF acc @50% global vs region per seed: {[(round(a,3), round(b_,3)) for a,b_ in calib_acc50]} -> selected {sel_choice}")
    print("  rho   test-call-rate   batch/global   stream/global   stream/region   stream/selected")
    for b in BUDGETS:
        a = R['acc']
        print(f"  {b:>4.0%}  {R['test_call_rate'][str(b)][0]:>6.1%}±{R['test_call_rate'][str(b)][1]:.1%}     "
              f"{a[f'batch_global_{b}'][0]:.3f}          {a[f'stream_global_{b}'][0]:.3f}           "
              f"{a[f'stream_region_{b}'][0]:.3f}           {a[f'stream_sel_{b}'][0]:.3f}")
    print("  McNemar (seed 0):", {n: round(v, 3) for n, v in R['mcnemar_seed0'].items()})

json.dump(results, open('improve_results.json', 'w'), indent=1)
