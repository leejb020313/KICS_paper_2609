"""Eq. (1) check: the fusion LR is fitted on the whole learning set but applied only to the queried (low-|d|) part.
Does fitting it only on the learning-set sentences that would be queried (|d| below the rho-quantile) help?
Same protocol as scripts/ablation_stage_fusion.py (existing Sonnet scores, 5 runs, 80% subsamples).

    uv run --locked python scripts/ablation_fusion_region.py
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_stage_fusion import route  # noqa: E402
from cwcascade.data import load_frontier  # noqa: E402

for name in ("clef", "cb"):
    test, calib, _ = load_frontier(name, full=True)
    for rho in (0.3, 0.5):
        res = {"all learning set (paper)": [], "queried region only": []}
        for seed in range(5):
            idx = np.random.default_rng(seed).choice(len(calib.y), int(0.8 * len(calib.y)), replace=False)
            Xs, ys, ss = calib.X[idx], calib.y[idx], calib.s[idx]
            make = lambda: SVC(class_weight="balanced", random_state=0)
            d_te = make().fit(Xs, ys).decision_function(test.X)
            d_cal = cross_val_predict(make(), Xs, ys, cv=5, method="decision_function")
            order = np.argsort(np.abs(d_te))
            region = np.abs(d_cal) <= np.quantile(np.abs(d_cal), rho)
            for key, mask in (("all learning set (paper)", np.ones(len(ys), bool)), ("queried region only", region)):
                lr = LogisticRegression(class_weight="balanced").fit(np.c_[d_cal, ss][mask], ys[mask])
                p = lr.predict(np.c_[d_te, test.s])
                res[key].append(np.mean(route((d_te > 0).astype(int), p, order, rho) == test.y))
        print(name, f"rho={rho}", {k: round(float(np.mean(v)), 4) for k, v in res.items()})
