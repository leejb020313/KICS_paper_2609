"""Call rate from a small labelled sample of the target data (pre-registered: results/analysis/target_calibration_preregistration.md).

    uv run --locked python scripts/target_calibration.py clef   -> results/analysis/target_calibration_<ds>.json
"""
import json
import os
import sys

import numpy as np
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cwcascade.data import FULL_PARTS, RESULTS, best_threshold, load_frontier  # noqa: E402
from jev_prior import BUDGETS, jev_scores, logit, mcnemar, route  # noqa: E402

SPLITS, N_CAL = 100, 800
MARGINS = (0.01, 0.02)


def select_rate(correct_casc, correct_all, margin):
    """Smallest call rate whose one-sided 95% lower bound of (cascade - all-call) accuracy is >= -margin."""
    for b in BUDGETS:
        d = correct_casc[b].astype(float) - correct_all.astype(float)
        if d.mean() - 1.645 * d.std(ddof=1) / np.sqrt(len(d)) >= -margin:
            return b
    return 1.0


def main(name):
    test, calib, test_items = load_frontier(name, full=True)
    parts = (name,) + FULL_PARTS[name]
    jt = [jev_scores(p) for p in parts]
    lens = [sum(r["part"] == p for r in test_items) for p in parts]
    j = np.concatenate([[d[i] for i in range(n)] for d, n in zip(jt, lens)])
    y, s = test.y, test.s
    d_svm = SVC(class_weight="balanced", random_state=0).fit(calib.X, calib.y).decision_function(test.X)
    res_runs = []
    for split in range(SPLITS):
        perm = np.random.default_rng(split).permutation(len(y))
        cal, ev = perm[:N_CAL], perm[N_CAL:]
        t_s, t_j = best_threshold(s[cal], y[cal]), best_threshold(j[cal], y[cal])
        p_all = (s >= t_s).astype(int)
        p_j = (j >= t_j).astype(int)
        p_svm = (d_svm > 0).astype(int)
        key_j, key_svm = np.abs(logit(j) - logit(t_j)), np.abs(d_svm)
        run = {"t_s": t_s, "t_j": t_j}
        for arm, base, key in (("jevrep", p_j, key_j), ("svmrep", p_svm, key_svm)):
            # routing order computed within each subset, so the calibration rate means the same on both
            casc = {sub: {b: route(base[idx], p_all[idx], np.argsort(key[idx]), b) for b in BUDGETS}
                    for sub, idx in (("cal", cal), ("ev", ev))}
            corr_cal = {b: casc["cal"][b] == y[cal] for b in BUDGETS}
            for m in MARGINS:
                b = select_rate(corr_cal, p_all[cal] == y[cal], m)
                diff, p = mcnemar(y[ev], casc["ev"][b], p_all[ev])
                run[f"{arm}_m{int(m * 100)}"] = dict(rate=b, acc=float(np.mean(casc["ev"][b] == y[ev])), diff=diff, p=p)
            run[f"{arm}_curve"] = {str(b): float(np.mean(casc["ev"][b] == y[ev])) for b in BUDGETS}
        run["allcall_acc"] = float(np.mean(p_all[ev] == y[ev]))
        res_runs.append(run)
    out = {"dataset": name, "splits": SPLITS, "n_cal": N_CAL, "n_eval": int(len(y) - N_CAL),
           "allcall_acc": float(np.mean([r["allcall_acc"] for r in res_runs]))}
    for arm in ("jevrep", "svmrep"):
        for m in MARGINS:
            k = f"{arm}_m{int(m * 100)}"
            rs = [r[k] for r in res_runs]
            out[k] = dict(rate_mean=float(np.mean([r["rate"] for r in rs])), acc=float(np.mean([r["acc"] for r in rs])),
                          mean_diff=float(np.mean([r["diff"] for r in rs])),
                          sig_lower=sum(r["p"] < .05 and r["diff"] < 0 for r in rs),
                          sig_higher=sum(r["p"] < .05 and r["diff"] > 0 for r in rs))
        out[f"{arm}_curve"] = {str(b): float(np.mean([r[f"{arm}_curve"][str(b)] for r in res_runs])) for b in BUDGETS}
    with open(os.path.join(RESULTS, "analysis", f"target_calibration_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"{name}: n_eval {out['n_eval']}  Sonnet all-call {out['allcall_acc']:.3f}")
    for arm in ("jevrep", "svmrep"):
        for m in MARGINS:
            v = out[f"{arm}_m{int(m * 100)}"]
            print(f"  {arm} margin {int(m * 100)}pp: rate {100 * v['rate_mean']:.0f}%  acc {v['acc']:.3f}  "
                  f"{100 * v['mean_diff']:+.2f}%p  sig lower {v['sig_lower']}  higher {v['sig_higher']}")


if __name__ == "__main__":
    main(sys.argv[1])
