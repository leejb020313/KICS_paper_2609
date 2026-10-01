"""ClaimBuster's classifier is trained on 2012 debates and tested on 2016 debates; its recall drops (0.79 -> 0.57).
Two follow-ups with the existing Claude Sonnet 5 scores (no new LLM calls):

1. Remedy: add k labelled 2016 sentences (k = 100, 300) to the learning set, retrain the SVM and the fusion, and
   evaluate on the remaining 2016 sentences. The no-adaptation cascade and the threshold-tuned all-call LLM are
   scored on the same remaining sentences.
2. Label-free warning: among the sentences the cascade sends to the LLM (50%), the share where the LLM says
   "check-worthy" (s >= its learning-set threshold) while the SVM says "not" (d < 0). Compared between the
   learning set (out-of-fold), CLEF test (same source) and ClaimBuster test (later debates).

Same protocol as scripts/recall_target.py (5 runs, 80% learning-set subsamples).

    uv run --locked python scripts/drift_remedy.py      -> results/analysis/drift_remedy.json
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recall_target import acc_threshold, metrics, route  # noqa: E402
from cwcascade.data import ANALYSIS, load_frontier  # noqa: E402

RHO = 0.5
KS = (0, 100, 300)


def cascade(Xs, ys, ss, Xt, st):
    make = lambda: SVC(class_weight="balanced", random_state=0)
    d_te = make().fit(Xs, ys).decision_function(Xt)
    d_cal = cross_val_predict(make(), Xs, ys, cv=5, method="decision_function")
    lr = LogisticRegression(class_weight="balanced").fit(np.c_[d_cal, ss], ys)
    order = np.argsort(np.abs(d_te))
    return route((d_te > 0).astype(int), lr.predict(np.c_[d_te, st]), order, RHO), d_te, d_cal, order


def disagreement(d, s, thr, order):
    """Share of the LLM-queried sentences where the LLM says check-worthy and the SVM says not."""
    q = order[: int(round(RHO * len(d)))]
    return float(np.mean((s[q] >= thr) & (d[q] < 0)))


def main():
    out = {"remedy_cb": {}, "warning": {}}
    cl_test, cl_cal, _ = load_frontier("clef", full=True)
    cb_test, cb_cal, _ = load_frontier("cb", full=True)

    # 1. remedy on ClaimBuster
    res = {k: {"fused": [], "llm_tuned": []} for k in KS}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(cb_cal.y), int(0.8 * len(cb_cal.y)), replace=False)
        perm = rng.permutation(len(cb_test.y))
        adapt_pool, held = perm[: max(KS)], perm[max(KS):]  # same held-out sentences for every k
        for k in KS:
            a = adapt_pool[:k]
            Xs = np.r_[cb_cal.X[idx], cb_test.X[a]]
            ys = np.r_[cb_cal.y[idx], cb_test.y[a]]
            ss = np.r_[cb_cal.s[idx], cb_test.s[a]]
            p, *_ = cascade(Xs, ys, ss, cb_test.X[held], cb_test.s[held])
            thr = acc_threshold(ss, ys)
            res[k]["fused"].append(metrics(cb_test.y[held], p))
            res[k]["llm_tuned"].append(metrics(cb_test.y[held], (cb_test.s[held] >= thr).astype(int)))
    for k in KS:
        out["remedy_cb"][str(k)] = {m: {q: float(np.mean([r[q] for r in v])) for q in v[0]} for m, v in res[k].items()}

    # 2. label-free warning
    w = {"clef_learning_oof": [], "clef_test": [], "cb_learning_oof": [], "cb_test": []}
    for seed in range(5):
        for name, test, cal in (("clef", cl_test, cl_cal), ("cb", cb_test, cb_cal)):
            idx = np.random.default_rng(seed).choice(len(cal.y), int(0.8 * len(cal.y)), replace=False)
            Xs, ys, ss = cal.X[idx], cal.y[idx], cal.s[idx]
            _, d_te, d_cal, order = cascade(Xs, ys, ss, test.X, test.s)
            thr = acc_threshold(ss, ys)
            w[f"{name}_learning_oof"].append(disagreement(d_cal, ss, thr, np.argsort(np.abs(d_cal))))
            w[f"{name}_test"].append(disagreement(d_te, test.s, thr, order))
    out["warning"] = {k: float(np.mean(v)) for k, v in w.items()}

    with open(os.path.join(ANALYSIS, "drift_remedy.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("## remedy (ClaimBuster, evaluated on the same held-out 2016 sentences)")
    print("  k added   fused acc/recall/precision      threshold-tuned all-call acc/recall")
    for k, r in out["remedy_cb"].items():
        fz, lt = r["fused"], r["llm_tuned"]
        print(f"  {k:>5}     {fz['acc']:.3f} / {fz['recall']:.3f} / {fz['precision']:.3f}          "
              f"{lt['acc']:.3f} / {lt['recall']:.3f}")
    print("## warning: share of LLM-queried sentences where LLM says yes and SVM says no")
    for k, v in out["warning"].items():
        print(f"  {k:20s} {100 * v:.1f}%")


if __name__ == "__main__":
    main()
