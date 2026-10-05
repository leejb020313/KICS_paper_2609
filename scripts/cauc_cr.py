"""CAUC's own fusion gate (arXiv:2609.11446, Eq. 5) applied to our cascade.

CAUC calibrates each model separately, then fuses on deferred inputs only if the calibration-set complementarity
rate CR = (N_gain - N_loss) / N_cal > 0, where among samples on which the small model is more confident than the
large one, N_gain counts small right / large wrong and N_loss the reverse. Otherwise it uses the large model directly
(= our replacement cascade). Here: small = embedding SVM (out-of-fold margins), large = the LLM; both calibrated by a
logistic (Platt) fit with source-prior class weights, as in scripts/prior_shift.py. CR is computed on the learning
subsample, over all of it and over the half that the cascade would call (smallest |d|). 20 subsamples.
Writes results/analysis/cauc_cr.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cwcascade.data import load_frontier  # noqa: E402

PRIOR = {"clef": 5413 / 22501, "cb": 0.2666}
RUNS = 20


def cr(ps_pos, pl_pos, y):
    ys, yl = (ps_pos >= .5).astype(int), (pl_pos >= .5).astype(int)
    cs, cl = np.maximum(ps_pos, 1 - ps_pos), np.maximum(pl_pos, 1 - pl_pos)
    m = cs > cl
    gain = int((m & (ys == y) & (yl != y)).sum())
    loss = int((m & (ys != y) & (yl == y)).sum())
    return (gain - loss) / len(y), gain, loss


out = {}
for llm in ("sonnet5", "haiku45"):
    os.environ["CWC_LLM"] = llm
    for name in ("clef", "cb"):
        _, calib, _ = load_frontier(name, full=True)
        pi = PRIOR[name]
        cw = {0: (1 - pi) / 0.5, 1: pi / 0.5}
        rows = []
        for seed in range(RUNS):
            idx = np.random.default_rng(seed).permutation(len(calib.y))[:int(0.8 * len(calib.y))]
            Xs, ys, ss = calib.X[idx], calib.y[idx], calib.s[idx]
            d = cross_val_predict(SVC(class_weight="balanced", random_state=seed), Xs, ys, cv=5, method="decision_function")
            p_s = cross_val_predict(LogisticRegression(class_weight=cw), d[:, None], ys, cv=5, method="predict_proba")[:, 1]
            p_l = cross_val_predict(LogisticRegression(class_weight=cw), ss[:, None], ys, cv=5, method="predict_proba")[:, 1]
            half = np.argsort(np.abs(d))[:len(ys) // 2]
            rows.append(dict(all=cr(p_s, p_l, ys), called_half=cr(p_s[half], p_l[half], ys[half])))
        res = {k: dict(cr_mean=float(np.mean([r[k][0] for r in rows])), gain_mean=float(np.mean([r[k][1] for r in rows])),
                       loss_mean=float(np.mean([r[k][2] for r in rows])), runs_cr_pos=int(sum(r[k][0] > 0 for r in rows)))
               for k in ("all", "called_half")}
        out[f"{llm}_{name}"] = res
        print(f"{llm:8s} {name:5s} | all: CR {100 * res['all']['cr_mean']:+.2f}%p (gain {res['all']['gain_mean']:.1f}, loss {res['all']['loss_mean']:.1f}, CR>0 in {res['all']['runs_cr_pos']}/{RUNS})"
              f" | called half: CR {100 * res['called_half']['cr_mean']:+.2f}%p (gain {res['called_half']['gain_mean']:.1f}, loss {res['called_half']['loss_mean']:.1f}, CR>0 in {res['called_half']['runs_cr_pos']}/{RUNS})")
with open(os.path.join(os.path.dirname(__file__), "..", "results", "analysis", "cauc_cr.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
