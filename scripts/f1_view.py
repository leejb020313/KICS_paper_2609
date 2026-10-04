"""Positive-class F1 (the CheckThat! Task 1 official measure) instead of accuracy, from the cached 100-run predictions:
mean F1 / recall / precision per method, and per-run fused-vs-replacement F1 at 50% calls.

    uv run --locked python scripts/f1_view.py
"""
import os
import pickle

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "cache")
LLMS = {"Sonnet Oct-04": "sonnet5", "Sonnet Sept": "sonnet5_cli_default", "Haiku": "haiku45"}
KEYS = ("svm", "sonnet_raw", "sonnet_thr", "sonnet_nnppi", "replace_0.5", "fuse_0.5", "replace_1.0", "fuse_1.0")
for d in ("clef", "cb"):
    print(f"===== {d}")
    for name, folder in LLMS.items():
        (y, preds), _ = pickle.load(open(os.path.join(CACHE, f"{folder}_{d}_full_100.pkl"), "rb"))
        cells = []
        for k in KEYS:
            f = np.mean([f1_score(y, p) for p in preds[k]])
            r = np.mean([recall_score(y, p) for p in preds[k]])
            pr = np.mean([precision_score(y, p) for p in preds[k]])
            cells.append(f"{k} F1 {f:.3f} (R {r:.2f} P {pr:.2f})")
        print(f"  {name}")
        for c in cells:
            print("     " + c)
        for b in (0.3, 0.5):
            df = np.array([f1_score(y, f) - f1_score(y, r) for f, r in zip(preds[f"fuse_{b}"], preds[f"replace_{b}"])]) * 100
            print(f"     fuse-replace F1 at {int(100 * b)}%: mean {df.mean():+.2f}pp, higher in {np.sum(df > 0)}/100 runs")
