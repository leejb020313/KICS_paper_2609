"""Does the fused cascade miss more check-worthy sentences than the threshold-tuned all-call LLM, and can its
operating point be moved without looking at the test set?

The fused decision uses sigma(a d + b s + c) >= 0.5. Here the cut-off tau is instead chosen on the learning set:
the largest tau whose out-of-fold cascade (same 50% routing by |d|) reaches the learning-set recall of the
threshold-tuned all-call LLM. Recall is conditional on the positive class, so the target does not depend on the
learning set being class-balanced while the test sets are not. Same protocol as scripts/ablation_stage_fusion.py
(existing Claude Sonnet 5 scores, 5 runs on 80% subsamples, no new LLM calls).

    uv run --locked python scripts/recall_target.py      -> results/analysis/recall_target.json
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.data import ANALYSIS, load_frontier  # noqa: E402

RHO = 0.5
TAUS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def route(base, alt, order, rho):
    p = base.copy()
    k = int(round(rho * len(base)))
    p[order[:k]] = alt[order[:k]]
    return p


def metrics(y, p):
    tp = int(((p == 1) & (y == 1)).sum())
    rec = tp / max(1, int((y == 1).sum()))
    prec = tp / max(1, int((p == 1).sum()))
    return dict(acc=float(np.mean(p == y)), recall=rec, precision=prec, f1=2 * prec * rec / max(1e-9, prec + rec))


def acc_threshold(s, y):
    """LLM threshold maximising learning-set accuracy (the paper's "임계값 조정")."""
    cands = np.unique(s)
    return float(max(cands, key=lambda t: np.mean((s >= t) == y)))


def run(name):
    test, calib, _ = load_frontier(name, full=True)
    rows = {k: [] for k in ("llm_all_tuned", "replace_50", "fuse_50_tau0.5", "fuse_50_recall_matched")}
    taus = []
    for seed in range(5):
        idx = np.random.default_rng(seed).choice(len(calib.y), int(0.8 * len(calib.y)), replace=False)
        Xs, ys, ss = calib.X[idx], calib.y[idx], calib.s[idx]
        make = lambda: SVC(class_weight="balanced", random_state=0)
        d_te = make().fit(Xs, ys).decision_function(test.X)
        d_cal = cross_val_predict(make(), Xs, ys, cv=5, method="decision_function")
        order_te, order_cal = np.argsort(np.abs(d_te)), np.argsort(np.abs(d_cal))

        thr = acc_threshold(ss, ys)
        target = metrics(ys, (ss >= thr).astype(int))["recall"]  # learning-set recall of the tuned all-call LLM

        lr = LogisticRegression(class_weight="balanced").fit(np.c_[d_cal, ss], ys)
        prob_te = lr.predict_proba(np.c_[d_te, test.s])[:, 1]
        # out-of-fold fused probabilities on the learning set, for choosing tau
        prob_cal = np.zeros(len(ys))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, ys):
            m = LogisticRegression(class_weight="balanced").fit(np.c_[d_cal[tr], ss[tr]], ys[tr])
            prob_cal[te] = m.predict_proba(np.c_[d_cal[te], ss[te]])[:, 1]
        base_cal = (d_cal > 0).astype(int)
        ok = [t for t in TAUS
              if metrics(ys, route(base_cal, (prob_cal >= t).astype(int), order_cal, RHO))["recall"] >= target]
        tau = float(max(ok)) if ok else float(TAUS[0])
        taus.append(tau)

        base_te = (d_te > 0).astype(int)
        rows["llm_all_tuned"].append(metrics(test.y, (test.s >= thr).astype(int)))
        rows["replace_50"].append(metrics(test.y, route(base_te, (test.s >= thr).astype(int), order_te, RHO)))
        rows["fuse_50_tau0.5"].append(metrics(test.y, route(base_te, (prob_te >= 0.5).astype(int), order_te, RHO)))
        rows["fuse_50_recall_matched"].append(metrics(test.y, route(base_te, (prob_te >= tau).astype(int), order_te, RHO)))
    mean = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in rows.items()}
    return dict(tau=taus, mean=mean, per_run=rows)


def main():
    res = {name: run(name) for name in ("clef", "cb")}
    with open(os.path.join(ANALYSIS, "recall_target.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for name, r in res.items():
        print(f"\n## {name}   tau chosen on the learning set (5 runs): {r['tau']}")
        print("  method                     acc     recall  precision  F1(pos)")
        for k, m in r["mean"].items():
            print(f"  {k:24s}  {m['acc']:.3f}   {m['recall']:.3f}   {m['precision']:.3f}     {m['f1']:.3f}")


if __name__ == "__main__":
    main()
