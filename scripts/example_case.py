"""Worked example for the paper (Section 3.2): how the fused decision rule moves the LLM-score bar with d.

Refits the seed-0 CLEF models exactly as final_eval.py --full does (80% calib subsample, SVM, out-of-fold
margins, logistic fuser, calib-tuned LLM threshold), then:
  - bar(d) = (-c - a*d) / b  is the LLM score above which Eq. (1) says "check-worthy"
  - picks the test sentence among the queried half (|d| below the median) where the replacement cascade is
    wrong and the fused cascade right, with the largest bar - threshold gap
Writes results/paper/example_case.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.data import PAPER, best_threshold, load_frontier  # noqa: E402


def main():
    test, calib, items = load_frontier("clef", full=True)
    idx = np.random.default_rng(0).permutation(len(calib.y))[:int(0.8 * len(calib.y))]
    Xs, ys, ss = calib.X[idx], calib.y[idx], calib.s[idx]
    d_test = SVC(class_weight="balanced", random_state=0).fit(Xs, ys).decision_function(test.X)
    d_cal = cross_val_predict(SVC(class_weight="balanced", random_state=0), Xs, ys, cv=5, method="decision_function")
    lr = LogisticRegression(class_weight="balanced").fit(np.c_[d_cal, ss], ys)
    (a, b), c = lr.coef_[0], lr.intercept_[0]
    thr = best_threshold(ss, ys)
    bar = lambda d: (-c - a * d) / b

    p_fuse = lr.predict(np.c_[d_test, test.s])
    p_rep = (test.s >= thr).astype(int)
    queried = np.argsort(np.abs(d_test))[:len(d_test) // 2]
    cand = [i for i in queried if p_fuse[i] == test.y[i] != p_rep[i]]
    i = max(cand, key=lambda i: bar(d_test[i]) - thr)
    out = dict(a=float(a), b=float(b), c=float(c), llm_threshold=float(thr), bar_at_0=float(bar(0.0)),
               n_queried_fuse_right_replace_wrong=len(cand),
               n_queried_replace_right_fuse_wrong=int(sum(p_rep[j] == test.y[j] != p_fuse[j] for j in queried)),
               example=dict(text=items[i]["text"], label=int(test.y[i]), d=float(d_test[i]), s=float(test.s[i]),
                            bar=float(bar(d_test[i])), fused=int(p_fuse[i]), replaced=int(p_rep[i])))
    with open(os.path.join(PAPER, "example_case.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
