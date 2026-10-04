"""Option 1 (results/analysis/operating_point_preregistration.md): fused and replacement cascades with the same
operating-point rule, chosen on the learning set — (A) max accuracy, (B) max positive-class F1.
With --knn also option 2 (results/analysis/knn_fusion_preregistration.md): a third fusion input, the positive-label
rate of the k=10 nearest learning-set sentences (leave-self-out on the learning set).

    uv run --locked python scripts/operating_point.py            -> results/analysis/operating_point.json
    uv run --locked python scripts/operating_point.py --knn      -> results/analysis/knn_fusion.json
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cwcascade.data import RESULTS, load_frontier  # noqa: E402

KNN = "--knn" in sys.argv
SEEDS = int(os.environ.get("CWC_SEEDS", 100))
LLMS = os.environ.get("CWC_LLMS", "sonnet5,sonnet5_cli_default,haiku45").split(",")
RATES = (0.3, 0.5)
GRID = np.round(np.arange(0.02, 0.985, 0.01), 2)
OBJ = {"acc": lambda y, p: np.mean(p == y), "f1": lambda y, p: f1_score(y, p, zero_division=0)}
K = 10


def best_t(score, y, obj):
    vals = [OBJ[obj](y, (score >= t).astype(int)) for t in GRID]
    return GRID[int(np.argmax(vals))]


def knn_rate(Xq, Xs, ys, exclude_self=False):
    sim = Xq @ Xs.T  # embeddings are L2-normalised: cosine similarity
    if exclude_self:
        np.fill_diagonal(sim, -np.inf)
    nn = np.argpartition(-sim, K, axis=1)[:, :K]
    return ys[nn].mean(axis=1)


def run(name):
    test, calib, _ = load_frontier(name, full=True)
    y, s, X = test.y, test.s, test.X
    out = {}
    for seed in range(SEEDS):
        idx = np.random.default_rng(seed).permutation(len(calib.y))[:int(0.8 * len(calib.y))]
        Xs, ys, ss = calib.X[idx], calib.y[idx], calib.s[idx]
        svm = SVC(class_weight="balanced", random_state=seed).fit(Xs, ys)
        d = svm.decision_function(X)
        d_cal = cross_val_predict(SVC(class_weight="balanced", random_state=seed), Xs, ys, cv=5, method="decision_function")
        F_cal, F_test = [d_cal, ss], [d, s]
        if KNN:
            F_cal.append(knn_rate(Xs, Xs, ys, exclude_self=True))
            F_test.append(knn_rate(X, Xs, ys))
        F_cal, F_test = np.c_[tuple(F_cal)], np.c_[tuple(F_test)]
        lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        p_test = lr.fit(F_cal, ys).predict_proba(F_test)[:, 1]
        p_oof = cross_val_predict(LogisticRegression(class_weight="balanced", max_iter=1000), F_cal, ys, cv=5, method="predict_proba")[:, 1]
        svm_pred = (d > 0).astype(int)
        order = np.argsort(np.abs(d))
        variants = {"fused_0.5": (p_test >= 0.5).astype(int)}  # the paper's current rule
        for obj in OBJ:
            t, tau = best_t(ss, ys, obj), best_t(p_oof, ys, obj)
            variants[f"llm_all_{obj}"] = (s >= t).astype(int)
            variants[f"replace_{obj}"] = (s >= t).astype(int)
            variants[f"fused_{obj}"] = (p_test >= tau).astype(int)
        for b in RATES:
            routed = order[:int(b * len(y))]
            for v, full_pred in variants.items():
                key = v if v.startswith("llm_all") else f"{v}@{int(100 * b)}"
                pred = full_pred if v.startswith("llm_all") else svm_pred.copy()
                if not v.startswith("llm_all"):
                    pred[routed] = full_pred[routed]
                m = out.setdefault(key, {"acc": [], "f1": []})
                if v.startswith("llm_all") and len(m["acc"]) > seed:
                    continue
                m["acc"].append(float(np.mean(pred == y)))
                m["f1"].append(float(f1_score(y, pred)))
    return out


def main():
    res = {}
    for llm in LLMS:
        os.environ["CWC_LLM"] = llm
        res[llm] = {d: run(d) for d in ("clef", "cb")}
        for d, out in res[llm].items():
            print(f"\n== {llm} {d}")
            for k, m in out.items():
                print(f"  {k:18s} acc {np.mean(m['acc']):.3f}  F1 {np.mean(m['f1']):.3f}")
            for b in RATES:
                for obj in OBJ:
                    for met in ("acc", "f1"):
                        f, r = np.array(out[f"fused_{obj}@{int(100 * b)}"][met]), np.array(out[f"replace_{obj}@{int(100 * b)}"][met])
                        print(f"  rule {obj:3s} @{int(100 * b)}%: fused - replace {met:3s} {100 * np.mean(f - r):+.2f}pp, "
                              f"fused higher in {np.sum(f > r)}/{len(f)} runs")
    tag = "" if "CWC_LLMS" not in os.environ else "_" + "_".join(LLMS)  # parallel runs, one file per LLM
    path = os.path.join(RESULTS, "analysis", ("knn_fusion" if KNN else "operating_point") + tag + ".json")
    json.dump(res, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
