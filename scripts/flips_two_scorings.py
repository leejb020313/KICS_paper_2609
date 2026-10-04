"""The paper's mechanism claim ("replacement overturns decisions the classifier already got right; fusion keeps them")
under both Claude Sonnet 5 scorings (Sept plain CLI = sonnet5_cli_default, Oct-04 clean = sonnet5) and Haiku 4.5.

For each call rate and run: broke = SVM correct -> cascade wrong, fixed = SVM wrong -> cascade correct (only routed
sentences can change). Mean over the 100 runs. Needs the per-run predictions cached by final_eval.evaluate with
CWC_CACHE=results/cache for each LLM folder.

    uv run --locked python scripts/flips_two_scorings.py
"""
import os
import pickle

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "cache")
LLMS = {"Sonnet Sept": "sonnet5_cli_default", "Sonnet Oct-04": "sonnet5", "Haiku": "haiku45"}
RATES = (0.1, 0.2, 0.3, 0.5, 1.0)

for d in ("clef", "cb"):
    print(f"===== {d}")
    for name, folder in LLMS.items():
        f = os.path.join(CACHE, f"{folder}_{d}_full_100.pkl")
        if not os.path.exists(f):
            print(f"  {name}: no cache")
            continue
        (y, preds), res = pickle.load(open(f, "rb"))
        svm = preds["svm"]
        best = max(("sonnet_raw", "sonnet_thr"), key=lambda k: np.mean([np.mean(p == y) for p in preds[k]]))
        best_acc = np.mean([np.mean(p == y) for p in preds[best]])
        print(f"  {name:14s} best all-call ({best}) {best_acc:.3f}")
        for b in RATES:
            row = []
            for v in ("replace", "fuse"):
                br = np.mean([np.sum((s == y) & (p != y)) for s, p in zip(svm, preds[f"{v}_{b}"])])
                fx = np.mean([np.sum((s != y) & (p == y)) for s, p in zip(svm, preds[f"{v}_{b}"])])
                acc = np.mean([np.mean(p == y) for p in preds[f"{v}_{b}"]])
                row.append(f"{v:7s} broke {br:6.1f} fixed {fx:6.1f} net {fx - br:+6.1f} acc {acc:.3f}")
            print(f"    {int(100 * b):3d}%  " + " | ".join(row))
