"""JEV -> Sonnet replacement with a learning-set call rate (pre-registered: results/analysis/jev_replace_preregistration.md).

    uv run --locked python scripts/jev_replace.py clef    -> results/analysis/jev_replace_sonnet5_clef.json
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cwcascade.data import FULL_PARTS, RESULTS, load_frontier, read_frontier_scores  # noqa: E402
from jev_prior import BUDGETS, CALIB_KEY, PRIOR, jev_scores, logit, mcnemar, route, wthr  # noqa: E402

SEEDS = int(os.environ.get("CWC_SEEDS", 100))
WITHIN = 0.003  # selected rate: smallest b within 0.3 pp of the learning-set curve's maximum


def main(name):
    test, calib, test_items = load_frontier(name, full=True)
    parts = (name,) + FULL_PARTS[name]
    jt = [jev_scores(p) for p in parts]
    lens = [sum(r["part"] == p for r in test_items) for p in parts]
    j_test = np.concatenate([[d[i] for i in range(n)] for d, n in zip(jt, lens)])
    jc = jev_scores(CALIB_KEY[name])
    j_cal = np.array([jc[i] for i in sorted(read_frontier_scores()[CALIB_KEY[name]])])
    y, pi = test.y, PRIOR[name]
    preds = {k: [] for k in ["sonnet_all", "jevrep_sel", "svmrep_0.5"] + [f"jevrep_{b}" for b in BUDGETS]}
    rates, curves = [], []
    for seed in range(SEEDS):
        idx = np.random.default_rng(seed).permutation(len(calib.y))[:int(0.8 * len(calib.y))]
        Xs, ys, ss, js = calib.X[idx], calib.y[idx], calib.s[idx], j_cal[idx]
        w = np.where(ys == 1, pi / ys.mean(), (1 - pi) / (1 - ys.mean()))
        t_s, t_j = wthr(ss, ys, w), wthr(js, ys, w)
        p_s, p_j = (test.s >= t_s).astype(int), (j_test >= t_j).astype(int)
        order = np.argsort(np.abs(logit(j_test) - logit(t_j)))
        preds["sonnet_all"].append(p_s)
        for b in BUDGETS:
            preds[f"jevrep_{b}"].append(route(p_j, p_s, order, b))
        # learning-set curve, out of fold
        base_c, alt_c, key_c = np.zeros(len(ys), int), np.zeros(len(ys), int), np.zeros(len(ys))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(js, ys):
            tj, ts_ = wthr(js[tr], ys[tr], w[tr]), wthr(ss[tr], ys[tr], w[tr])
            base_c[te], alt_c[te] = (js[te] >= tj).astype(int), (ss[te] >= ts_).astype(int)
            key_c[te] = np.abs(logit(js[te]) - logit(tj))
        order_c = np.argsort(key_c)
        curve = {b: float(np.sum(w * (route(base_c, alt_c, order_c, b) == ys)) / w.sum()) for b in BUDGETS}
        rate = next(b for b in BUDGETS if max(curve.values()) - curve[b] <= WITHIN)
        rates.append(rate)
        curves.append(curve)
        preds["jevrep_sel"].append(preds[f"jevrep_{rate}"][-1])
        # reference: the SVM -> Sonnet replacement at 50%, prior-corrected (scripts/prior_shift.py)
        d_test = SVC(class_weight="balanced", random_state=seed).fit(Xs, ys).decision_function(test.X)
        preds["svmrep_0.5"].append(route((d_test > 0).astype(int), p_s, np.argsort(np.abs(d_test)), 0.5))
        if seed % 20 == 0:
            print(f"[{name}] run {seed}", file=sys.stderr, flush=True)

    def tests(a, b):
        dp = [mcnemar(y, pa, pb) for pa, pb in zip(preds[a], preds[b])]
        return dict(mean_diff=float(np.mean([d for d, _ in dp])),
                    sig_higher=sum(1 for d, p in dp if p < .05 and d > 0), sig_lower=sum(1 for d, p in dp if p < .05 and d < 0))
    res = dict(dataset=name, prior=pi, runs=SEEDS,
               mean_acc={k: float(np.mean([np.mean(p == y) for p in v])) for k, v in preds.items()},
               rate_mean=float(np.mean(rates)), rate_counts={str(r): rates.count(r) for r in sorted(set(rates))},
               learning_curve={str(b): float(np.mean([c[b] for c in curves])) for b in BUDGETS},
               tests={f"{a}_vs_{b}": tests(a, b) for a, b in [
                   ("jevrep_sel", "sonnet_all"), ("jevrep_sel", "svmrep_0.5"), ("jevrep_0.3", "sonnet_all"),
                   ("jevrep_0.4", "sonnet_all"), ("jevrep_0.5", "sonnet_all")]})
    with open(os.path.join(RESULTS, "analysis", f"jev_replace_sonnet5_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    m = res["mean_acc"]
    print(f"{name}: sonnet_all {m['sonnet_all']:.3f}  jevrep_sel {m['jevrep_sel']:.3f}  rate mean {res['rate_mean']:.2f} {res['rate_counts']}  "
          f"svmrep_0.5 {m['svmrep_0.5']:.3f}")
    print("  learning curve " + "  ".join(f"{b}:{v:.3f}" for b, v in res["learning_curve"].items()))
    for k, v in res["tests"].items():
        print(f"  {k}: {100 * v['mean_diff']:+.2f}%p  sig higher {v['sig_higher']}  lower {v['sig_lower']}")


if __name__ == "__main__":
    main(sys.argv[1])
