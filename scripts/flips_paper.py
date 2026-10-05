"""Numbers the paper prints about the mechanism and the metrics: from the cached 100-run predictions of the paper's
LLM (Claude Sonnet 5, results/llm_scores/sonnet5), per dataset —
- correct SVM decisions broken / wrong ones fixed by replacement and fused cascades at 50% calls (mean over runs);
- positive-class F1 of fused@50%, replacement@50% and the stronger all-call setting (mean over runs).
Weighted F1 is already in final_results_full.json.

    CWC_CACHE=results/cache uv run --locked python scripts/flips_paper.py   -> results/paper/flips.json
"""
import json
import os
import pickle

import numpy as np
from sklearn.metrics import f1_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CACHE = os.environ.get("CWC_CACHE", os.path.join(ROOT, "results", "cache"))
out = {}
for d in ("clef", "cb"):
    (y, preds), _ = pickle.load(open(os.path.join(CACHE, f"sonnet5_{d}_full_100.pkl"), "rb"))
    best = max(("sonnet_raw", "sonnet_thr"), key=lambda k: np.mean([np.mean(p == y) for p in preds[k]]))
    r = {"best_full": best}
    for v in ("replace", "fuse"):
        r[v] = dict(broke=float(np.mean([np.sum((s == y) & (p != y)) for s, p in zip(preds["svm"], preds[f"{v}_0.5"])])),
                    fixed=float(np.mean([np.sum((s != y) & (p == y)) for s, p in zip(preds["svm"], preds[f"{v}_0.5"])])))
    r["f1_pos"] = {k: float(np.mean([f1_score(y, p) for p in preds[k]])) for k in ("fuse_0.5", "replace_0.5", best)}
    out[d] = r
json.dump(out, open(os.path.join(ROOT, "results", "paper", "flips.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
