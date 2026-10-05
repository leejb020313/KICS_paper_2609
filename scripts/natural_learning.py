"""Natural-ratio learning set, no class weights (pre-registered: results/analysis/natural_learning_preregistration.md).

    CWC_LLM=sonnet5 uv run --locked python scripts/natural_learning.py clef  -> results/analysis/natural_<llm>_<ds>.json
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cwcascade.data import FULL_PARTS, RESULTS, best_threshold, load_frontier, read_frontier_scores  # noqa: E402
from jev_prior import BUDGETS, CALIB_KEY, PRIOR, jev_scores, logit, mcnemar, route  # noqa: E402

SEEDS = int(os.environ.get("CWC_SEEDS", 100))
WITHIN = 0.003


def natural_subset(y, pi, rng):
    """All negatives plus a random subset of positives so that positives are a fraction pi."""
    neg, pos = np.where(y == 0)[0], np.where(y == 1)[0]
    n_pos = int(round(pi / (1 - pi) * len(neg)))
    return np.concatenate([neg, rng.choice(pos, min(n_pos, len(pos)), replace=False)])


def main(name):
    llm = os.environ.get("CWC_LLM", "sonnet5")
    test, calib, test_items = load_frontier(name, full=True)
    use_jev = llm == "sonnet5"
    if use_jev:
        parts = (name,) + FULL_PARTS[name]
        jt = [jev_scores(p) for p in parts]
        lens = [sum(r["part"] == p for r in test_items) for p in parts]
        j_test = np.concatenate([[d[i] for i in range(n)] for d, n in zip(jt, lens)])
        jc = jev_scores(CALIB_KEY[name])
        j_cal = np.array([jc[i] for i in sorted(read_frontier_scores()[CALIB_KEY[name]])])
    y, pi = test.y, PRIOR[name]
    keys = ["allcall", "svm"] + [f"{v}_{b}" for b in BUDGETS for v in ("svmrep", "svmfuse")]
    if use_jev:
        keys += ["jev_alone", "jevrep_sel"] + [f"{v}_{b}" for b in BUDGETS for v in ("jevrep", "jevfuse")]
    preds = {k: [] for k in keys}
    rates, n_learn = [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        nat = natural_subset(calib.y, pi, rng)
        idx = rng.permutation(nat)[:int(0.8 * len(nat))]
        n_learn.append(len(idx))
        Xs, ys, ss = calib.X[idx], calib.y[idx], calib.s[idx]
        t_s = best_threshold(ss, ys)
        p_s = (test.s >= t_s).astype(int)
        preds["allcall"].append(p_s)
        d_test = SVC(random_state=seed).fit(Xs, ys).decision_function(test.X)
        d_cal = cross_val_predict(SVC(random_state=seed), Xs, ys, cv=5, method="decision_function")
        p_svm, order = (d_test > 0).astype(int), np.argsort(np.abs(d_test))
        p_fz = LogisticRegression().fit(np.c_[d_cal, ss], ys).predict(np.c_[d_test, test.s])
        preds["svm"].append(p_svm)
        for b in BUDGETS:
            preds[f"svmrep_{b}"].append(route(p_svm, p_s, order, b))
            preds[f"svmfuse_{b}"].append(route(p_svm, p_fz, order, b))
        if use_jev:
            js = j_cal[idx]
            t_j = best_threshold(js, ys)
            p_j = (j_test >= t_j).astype(int)
            o_j = np.argsort(np.abs(logit(j_test) - logit(t_j)))
            p_jf = LogisticRegression().fit(np.c_[logit(js), ss], ys).predict(np.c_[logit(j_test), test.s])
            preds["jev_alone"].append(p_j)
            for b in BUDGETS:
                preds[f"jevrep_{b}"].append(route(p_j, p_s, o_j, b))
                preds[f"jevfuse_{b}"].append(route(p_j, p_jf, o_j, b))
            base_c, alt_c, key_c = np.zeros(len(ys), int), np.zeros(len(ys), int), np.zeros(len(ys))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(js, ys):
                tj, ts_ = best_threshold(js[tr], ys[tr]), best_threshold(ss[tr], ys[tr])
                base_c[te], alt_c[te] = (js[te] >= tj).astype(int), (ss[te] >= ts_).astype(int)
                key_c[te] = np.abs(logit(js[te]) - logit(tj))
            oc = np.argsort(key_c)
            curve = {b: float(np.mean(route(base_c, alt_c, oc, b) == ys)) for b in BUDGETS}
            rate = next(b for b in BUDGETS if max(curve.values()) - curve[b] <= WITHIN)
            rates.append(rate)
            preds["jevrep_sel"].append(preds[f"jevrep_{rate}"][-1])
        if seed % 20 == 0:
            print(f"[{llm} {name}] run {seed}", file=sys.stderr, flush=True)

    def tests(a, b):
        dp = [mcnemar(y, pa, pb) for pa, pb in zip(preds[a], preds[b])]
        return dict(mean_diff=float(np.mean([d for d, _ in dp])),
                    sig_higher=sum(1 for d, p in dp if p < .05 and d > 0), sig_lower=sum(1 for d, p in dp if p < .05 and d < 0))
    pairs = [("svmfuse_0.5", "allcall"), ("svmrep_0.5", "allcall"), ("svmfuse_0.5", "svmrep_0.5")]
    if use_jev:
        pairs += [("jevrep_0.3", "allcall"), ("jevrep_0.5", "allcall"), ("jevfuse_0.5", "jevrep_0.5"), ("jevrep_sel", "allcall")]
    res = dict(llm=llm, dataset=name, prior=pi, runs=SEEDS, learn_size_mean=float(np.mean(n_learn)),
               mean_acc={k: float(np.mean([np.mean(p == y) for p in v])) for k, v in preds.items()},
               tests={f"{a}_vs_{b}": tests(a, b) for a, b in pairs})
    if use_jev:
        res["jev_rate_mean"] = float(np.mean(rates))
    with open(os.path.join(RESULTS, "analysis", f"natural_{llm}_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    m = res["mean_acc"]
    line = (f"{llm} {name} (n_learn {res['learn_size_mean']:.0f}): allcall {m['allcall']:.3f}  svm {m['svm']:.3f}  "
            f"svmrep50 {m['svmrep_0.5']:.3f}  svmfuse50 {m['svmfuse_0.5']:.3f}")
    if use_jev:
        line += (f"  | jev {m['jev_alone']:.3f}  jevrep30 {m['jevrep_0.3']:.3f}  jevrep50 {m['jevrep_0.5']:.3f}  "
                 f"jevfuse50 {m['jevfuse_0.5']:.3f}  jevrep_sel {m['jevrep_sel']:.3f} (rate {res['jev_rate_mean']:.2f})")
    print(line)
    for k, v in res["tests"].items():
        print(f"   {k}: {100 * v['mean_diff']:+.2f}%p  sig higher {v['sig_higher']}  lower {v['sig_lower']}")


if __name__ == "__main__":
    main(sys.argv[1])
