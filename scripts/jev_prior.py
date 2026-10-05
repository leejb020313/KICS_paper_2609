"""JEV -> Sonnet cascade under the prior-shift correction (pre-registered: results/analysis/jev_prior_preregistration.md).

    uv run --locked python scripts/jev_prior.py clef     -> results/analysis/jev_prior_sonnet5_clef.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cwcascade.data import FULL_PARTS, RESULTS, load_frontier, read_frontier_scores  # noqa: E402

SEEDS = int(os.environ.get("CWC_SEEDS", 100))
BUDGETS = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
PRIOR = {"clef": 5413 / 22501, "cb": 0.2666}
CALIB_KEY = {"clef": "clefcal", "cb": "cbcal"}
SATURATION = 0.003  # the main paper's call-rate rule: further calls add at most 0.3 pp on the learning set


def jev_scores(key):
    with open(os.path.join(RESULTS, "jev", "nnppi", f"{key}.jsonl"), encoding="utf-8") as f:
        return {r["id"]: r["noul"] for r in map(json.loads, f)}


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def route(base, alt, order, rho):
    out = base.copy()
    sel = order[:int(round(rho * len(base)))]
    out[sel] = alt[sel]
    return out


def wthr(s, y, w):
    ts = np.linspace(0.02, 0.98, 97)
    return float(ts[np.argmax([np.sum(w * ((s >= t) == y)) for t in ts])])


def mcnemar(y, a, b):
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return float(np.mean(ca) - np.mean(cb)), float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0


def cr(ps, pl, y):
    """CAUC Eq. 5: among samples where the small model is more confident, (small right & large wrong) - reverse, / N."""
    cs, cl = np.maximum(ps, 1 - ps), np.maximum(pl, 1 - pl)
    m = cs > cl
    ys, yl = (ps >= .5).astype(int), (pl >= .5).astype(int)
    return float(((m & (ys == y) & (yl != y)).sum() - (m & (ys != y) & (yl == y)).sum()) / len(y))


def main(name):
    test, calib, test_items = load_frontier(name, full=True)
    parts = (name,) + FULL_PARTS[name]
    jt = [jev_scores(p) for p in parts]
    lens = [sum(r["part"] == p for r in test_items) for p in parts]
    j_test = np.concatenate([[d[i] for i in range(n)] for d, n in zip(jt, lens)])
    jc = jev_scores(CALIB_KEY[name])
    j_cal = np.array([jc[i] for i in sorted(read_frontier_scores()[CALIB_KEY[name]])])
    y, pi = test.y, PRIOR[name]
    cw = {0: (1 - pi) / 0.5, 1: pi / 0.5}
    keys = ["sonnet_all", "jev_alone"] + [f"{v}_{b}" for b in BUDGETS for v in ("jevrep", "jevfuse")] + ["jevfuse_sel", "jevrep_sel"]
    preds = {k: [] for k in keys}
    rates, crs, curves = [], [], []
    for seed in range(SEEDS):
        idx = np.random.default_rng(seed).permutation(len(calib.y))[:int(0.8 * len(calib.y))]
        ys, ss, js = calib.y[idx], calib.s[idx], j_cal[idx]
        w = np.where(ys == 1, cw[1], cw[0])
        t_s, t_j = wthr(ss, ys, w), wthr(js, ys, w)
        p_s, p_j = (test.s >= t_s).astype(int), (j_test >= t_j).astype(int)
        fz = LogisticRegression(class_weight=cw).fit(np.c_[logit(js), ss], ys)
        p_fz = fz.predict(np.c_[logit(j_test), test.s])
        order = np.argsort(np.abs(logit(j_test) - logit(t_j)))
        preds["sonnet_all"].append(p_s)
        preds["jev_alone"].append(p_j)
        for b in BUDGETS:
            preds[f"jevrep_{b}"].append(route(p_j, p_s, order, b))
            preds[f"jevfuse_{b}"].append(route(p_j, p_fz, order, b))
        # learning set, out of fold: call-rate curve (prior-weighted accuracy) and CAUC's complementarity rate
        base_c, s_oof = np.zeros(len(ys), int), np.zeros(len(ys), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(js, ys):
            base_c[te] = (js[te] >= wthr(js[tr], ys[tr], w[tr])).astype(int)
        fz_c = cross_val_predict(LogisticRegression(class_weight=cw), np.c_[logit(js), ss], ys, cv=5)
        order_c = np.argsort(np.abs(logit(js) - logit(t_j)))
        curve = {b: float(np.sum(w * (route(base_c, fz_c, order_c, b) == ys)) / w.sum()) for b in BUDGETS}
        rate = next(b for b in BUDGETS if max(curve[c] for c in BUDGETS if c >= b) - curve[b] <= SATURATION)
        rates.append(rate)
        curves.append(curve)
        pj_cal = cross_val_predict(LogisticRegression(class_weight=cw), logit(js)[:, None], ys, cv=5, method="predict_proba")[:, 1]
        ps_cal = cross_val_predict(LogisticRegression(class_weight=cw), ss[:, None], ys, cv=5, method="predict_proba")[:, 1]
        called = order_c[:int(round(rate * len(ys)))] if rate > 0 else order_c[:len(ys) // 2]
        crs.append(cr(pj_cal[called], ps_cal[called], ys[called]))
        preds["jevfuse_sel"].append(preds[f"jevfuse_{rate}"][-1])
        preds["jevrep_sel"].append(preds[f"jevrep_{rate}"][-1])
        if seed % 20 == 0:
            print(f"[{name}] run {seed}", file=sys.stderr, flush=True)

    def tests(a, b):
        dp = [mcnemar(y, pa, pb) for pa, pb in zip(preds[a], preds[b])]
        return dict(mean_diff=float(np.mean([d for d, _ in dp])),
                    sig_higher=sum(1 for d, p in dp if p < .05 and d > 0), sig_lower=sum(1 for d, p in dp if p < .05 and d < 0))
    res = dict(dataset=name, prior=pi, runs=SEEDS,
               mean_acc={k: float(np.mean([np.mean(p == y) for p in v])) for k, v in preds.items()},
               rate_counts={str(r): rates.count(r) for r in sorted(set(rates))},
               cr_called=dict(mean=float(np.mean(crs)), runs_pos=int(sum(c > 0 for c in crs))),
               learning_curve={str(b): float(np.mean([c[b] for c in curves])) for b in BUDGETS},
               tests={f"{a}_vs_{b}": tests(a, b) for a, b in [
                   ("jevfuse_sel", "sonnet_all"), ("jevfuse_sel", "jevrep_sel"), ("jevrep_sel", "sonnet_all"),
                   ("jev_alone", "sonnet_all"), ("jevfuse_0.3", "sonnet_all"), ("jevfuse_0.5", "sonnet_all"),
                   ("jevfuse_0.3", "jevrep_0.3"), ("jevfuse_0.5", "jevrep_0.5")]})
    out = os.path.join(RESULTS, "analysis", f"jev_prior_sonnet5_{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    m = res["mean_acc"]
    print(f"{name}: sonnet_all {m['sonnet_all']:.3f}  jev_alone {m['jev_alone']:.3f}  rate {res['rate_counts']}  "
          f"jevfuse_sel {m['jevfuse_sel']:.3f}  jevrep_sel {m['jevrep_sel']:.3f}  CR {100 * res['cr_called']['mean']:+.2f}%p "
          f"(>0 in {res['cr_called']['runs_pos']})")
    print("  by rate  " + "  ".join(f"{b}: fuse {m[f'jevfuse_{b}']:.3f} rep {m[f'jevrep_{b}']:.3f}" for b in BUDGETS))
    for k, v in res["tests"].items():
        print(f"  {k}: {100 * v['mean_diff']:+.2f}%p  sig higher {v['sig_higher']}  lower {v['sig_lower']}")


if __name__ == "__main__":
    main(sys.argv[1])
