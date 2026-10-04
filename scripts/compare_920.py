"""The 920 test sentences scored three times by Claude Sonnet 5: September plain CLI mode, the 2026-10-01 clean-mode
cost check (checks/sonnet5_cost), and the 2026-10-04 clean re-score (sonnet5/). Same prompts, same batches.

    uv run --locked python scripts/compare_920.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cwcascade.data import read_json, DATA, SCORES, read_frontier_scores  # noqa: E402

lab = {}
for k, v in list(read_json(os.path.join(DATA, "frontier", "eval_set.json")).items()) + \
        list(read_json(os.path.join(DATA, "frontier", "full_set.json")).items()):
    lab.update({(k, i): r["label"] for i, r in enumerate(v)})
cost, think = {}, {}
for p in glob.glob(os.path.join(SCORES, "checks", "sonnet5_cost", "*.json")):
    b = os.path.basename(p)[:-5]
    if b.startswith("overhead"):
        continue
    d = json.load(open(p, encoding="utf-8"))
    think[b] = d["usage"]["output_tokens_details"].get("thinking_tokens", 0)
    cost.update({(b.split("_")[0], int(i)): float(s) for i, s in json.loads(re.search(r"\{.*\}", d["result"], re.S).group(0)).items()})
new_think = {b: json.load(open(os.path.join(SCORES, "sonnet5", b + ".json"), encoding="utf-8"))["usage"]["output_tokens_details"].get("thinking_tokens", 0)
             for b in think}
keys = [k for k in cost if k in lab]
y = np.array([lab[k] for k in keys])
plain, clean = read_frontier_scores("sonnet5_cli_default"), read_frontier_scores("sonnet5")
S = {"Sep plain": [plain[a][i] for a, i in keys], "Oct-01 clean (cost check)": [cost[k] for k in keys],
     "Oct-04 clean (re-score)": [clean[a][i] for a, i in keys]}
print(f"n={len(keys)} sentences, labels {100 * y.mean():.0f}% positive")
for n, s in S.items():
    s = np.array(s)
    k = s >= .5
    print(f"  {n:26s} mean={s.mean():.3f} flag={100 * k.mean():4.1f}% acc@.5={np.mean(k == y):.3f} AUC={roc_auc_score(y, s):.3f}")
A = {n: np.array(s) for n, s in S.items()}
names = list(A)
for i in range(3):
    for j in range(i + 1, 3):
        print(f"  corr {names[i]} vs {names[j]}: {np.corrcoef(A[names[i]], A[names[j]])[0, 1]:.3f}")
print(f"  mean thinking tokens per batch: Oct-01 {np.mean(list(think.values())):.0f}, Oct-04 {np.mean(list(new_think.values())):.0f}")
