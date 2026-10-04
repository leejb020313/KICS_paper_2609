"""How strong is "fused > replacement"? Per call rate, over the 100 runs: mean difference, runs where fused is
higher / lower, runs where McNemar p < 0.05 in each direction. For both Sonnet 5 scorings and Haiku 4.5, from the
per-run predictions cached by final_eval.evaluate (results/cache).

    uv run --locked python scripts/fuse_vs_replace_strength.py
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_eval as fe  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "cache")
LLMS = {"Sonnet Sept": "sonnet5_cli_default", "Sonnet Oct-04": "sonnet5", "Haiku": "haiku45"}
for d in ("clef", "cb"):
    print(f"===== {d}")
    for name, folder in LLMS.items():
        (y, preds), _ = pickle.load(open(os.path.join(CACHE, f"{folder}_{d}_full_100.pkl"), "rb"))
        print(f"  {name}")
        for b in (0.2, 0.3, 0.5, 1.0):
            diffs, up, down = [], 0, 0
            for f, r in zip(preds[f"fuse_{b}"], preds[f"replace_{b}"]):
                a, c = int(np.sum((f == y) & (r != y))), int(np.sum((f != y) & (r == y)))
                diffs.append((a - c) / len(y))
                p = fe.binomtest(min(a, c), a + c, 0.5).pvalue if a + c else 1.0
                up += p < 0.05 and a > c
                down += p < 0.05 and c > a
            diffs = np.array(diffs) * 100
            print(f"    {int(100 * b):3d}%  mean {diffs.mean():+.2f}pp  higher {np.sum(diffs > 0):3d}  lower {np.sum(diffs < 0):3d}  "
                  f"sig higher {up:3d}  sig lower {down:3d}")
