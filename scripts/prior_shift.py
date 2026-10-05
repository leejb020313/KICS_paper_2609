"""Prior-shift-corrected comparison (pre-registered: results/analysis/prior_shift_preregistration.md).

The learning set is class-balanced, the source corpora are not (CLEF 2024 train 24.1%, ClaimBuster 2012 26.7%).
Here every learned threshold / intercept is fitted with the source prior pi instead of "balanced" weights:
all-call threshold, replacement threshold, fusion logistic regression; variant "svm" also re-weights the SVM.
Same 100 learning-set subsamples, routing and call rates as scripts/final_eval.py. Usage:
    CWC_LLM=sonnet5 uv run --locked python scripts/prior_shift.py clef
Writes results/analysis/prior_shift_<llm>_<dataset>.json.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cwcascade.data import load_frontier  # noqa: E402

SEEDS = int(os.environ.get("CWC_SEEDS", 100))
BUDGETS = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
PRIOR = {"clef": 5413 / 22501, "cb": 0.2666}  # positive rate of the corpus the learning set was drawn from


def route(base, alt, order, rho):
    out = base.copy()
    sel = order[:int(round(rho * len(base)))]
    out[sel] = alt[sel]
    return out


def mcnemar(y, a, b):
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return float(np.mean(ca) - np.mean(cb)), float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0


def weighted_threshold(s, y, w):
    ts = np.linspace(0.02, 0.98, 97)
    return float(ts[np.argmax([np.sum(w * ((s >= t) == y)) for t in ts])])


def main(name):
    llm = os.environ.get("CWC_LLM", "sonnet5")
    test, calib, _ = load_frontier(name, full=True)
    y, pi = test.y, PRIOR[name]
    cw = {0: (1 - pi) / 0.5, 1: pi / 0.5}  # learning set is 50/50, so these weights re-create the source prior
    keys = ["svm", "svm_prior", "allcall_prior", "allcall_bal"] + [f"{v}_{b}" for b in BUDGETS
                                                                     for v in ("fuse_prior", "replace_prior", "fuse_prior_svm", "replace_prior_svm", "fuse_bal")]
    preds = {k: [] for k in keys}
    for seed in range(SEEDS):
        idx = np.random.default_rng(seed).permutation(len(calib.y))[:int(0.8 * len(calib.y))]
        Xs, ys, ss = calib.X[idx], calib.y[idx], calib.s[idx]
        w = np.where(ys == 1, cw[1], cw[0])
        t_prior = weighted_threshold(ss, ys, w)
        t_bal = weighted_threshold(ss, ys, np.ones_like(w, float))
        p_all_prior = (test.s >= t_prior).astype(int)
        preds["allcall_prior"].append(p_all_prior)
        preds["allcall_bal"].append((test.s >= t_bal).astype(int))
        for tag, svm_cw in (("", "balanced"), ("_svm", cw)):
            svm = SVC(class_weight=svm_cw, random_state=seed).fit(Xs, ys)
            d_test = svm.decision_function(test.X)
            d_cal = cross_val_predict(SVC(class_weight=svm_cw, random_state=seed), Xs, ys, cv=5, method="decision_function")
            p_svm = (d_test > 0).astype(int)
            order = np.argsort(np.abs(d_test))
            fz_prior = LogisticRegression(class_weight=cw).fit(np.c_[d_cal, ss], ys).predict(np.c_[d_test, test.s])
            if tag == "":
                preds["svm"].append(p_svm)
                fz_bal = LogisticRegression(class_weight="balanced").fit(np.c_[d_cal, ss], ys).predict(np.c_[d_test, test.s])
            else:
                preds["svm_prior"].append(p_svm)
            for b in BUDGETS:
                preds[f"fuse_prior{tag}_{b}"].append(route(p_svm, fz_prior, order, b))
                preds[f"replace_prior{tag}_{b}"].append(route(p_svm, p_all_prior, order, b))
                if tag == "":
                    preds[f"fuse_bal_{b}"].append(route(p_svm, fz_bal, order, b))
        if seed % 20 == 0:
            print(f"[{llm} {name}] run {seed}", file=sys.stderr, flush=True)
    res = {"llm": llm, "dataset": name, "prior": pi, "runs": SEEDS,
           "mean_acc": {k: float(np.mean([np.mean(p == y) for p in v])) for k, v in preds.items()},
           "per_run": {}}
    for a, b in [("fuse_prior_0.5", "allcall_prior"), ("fuse_prior_svm_0.5", "allcall_prior"),
                 ("fuse_prior_0.5", "replace_prior_0.5"), ("fuse_prior_svm_0.5", "replace_prior_svm_0.5"),
                 ("fuse_prior_1.0", "allcall_prior"), ("fuse_prior_0.5", "fuse_prior_1.0")]:
        dp = [mcnemar(y, pa, pb) for pa, pb in zip(preds[a], preds[b])]
        res["per_run"][f"{a}_vs_{b}"] = dict(
            mean_diff=float(np.mean([d for d, _ in dp])),
            sig_higher=sum(1 for d, p in dp if p < 0.05 and d > 0), sig_lower=sum(1 for d, p in dp if p < 0.05 and d < 0))
    out = os.path.join(os.path.dirname(__file__), "..", "results", "analysis", f"prior_shift_{llm}_{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    m = res["mean_acc"]
    print(f"{llm} {name}: allcall bal {m['allcall_bal']:.3f} prior {m['allcall_prior']:.3f} | "
          f"fuse50 bal {m['fuse_bal_0.5']:.3f} prior {m['fuse_prior_0.5']:.3f} prior+svm {m['fuse_prior_svm_0.5']:.3f} | "
          f"replace50 prior {m['replace_prior_0.5']:.3f} prior+svm {m['replace_prior_svm_0.5']:.3f}")
    for k, v in res["per_run"].items():
        print(f"   {k}: mean {100 * v['mean_diff']:+.2f}%p  sig higher {v['sig_higher']}  lower {v['sig_lower']}")


if __name__ == "__main__":
    main(sys.argv[1])
