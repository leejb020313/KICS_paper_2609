import json

import numpy as np
from scipy.stats import wilcoxon

d = json.load(open("results/ece_comparison.json", encoding="utf-8"))
for r in d:
    t = np.array(r["ece_temp_seeds"])
    o = np.array(r["ece_ours_seeds"])
    diff = t - o  # positive = ours has lower ECE (better)
    stat, p = wilcoxon(diff)
    sig = "significant, ours better" if p < 0.05 and diff.mean() > 0 else (
        "significant, ours worse" if p < 0.05 and diff.mean() < 0 else "not significant")
    print(f"{r['model']:15s} {r['dataset']:12s} mean(temp-ours)={diff.mean():+.3f}  "
          f"wilcoxon p={p:.4f}  {sig}")
