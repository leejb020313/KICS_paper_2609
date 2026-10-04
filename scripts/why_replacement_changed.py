"""Why did the replacement cascade improve after the clean re-score? Compares old (Sept, plain CLI) and new (clean)
Sonnet 5 scores on: the per-run LLM threshold t chosen on the learning set (80% subsamples, seeds 0-99, as in
final_eval.py), test accuracy of the LLM at that t, and replacement@50% / all-call accuracy (mean, sd) from the
100-run result files (old ones from git commit e419d03).

    uv run --locked python scripts/why_replacement_changed.py
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cwcascade.data import DATASETS, DATA, best_threshold, read_frontier_scores, read_json  # noqa: E402

ev, full = read_json(os.path.join(DATA, "frontier", "eval_set.json")), read_json(os.path.join(DATA, "frontier", "full_set.json"))
cal_sets = read_json(os.path.join(DATA, "frontier", "calib_set.json"))
parts = {"clef": ["clef", "clefdev", "clefoff"], "cb": ["cb", "cbrest"]}
R = {"old": json.loads(subprocess.check_output(["git", "show", "e419d03:results/paper/final_results_full.json"]).decode()),
     "new": json.load(open("results/paper/final_results_full.json", encoding="utf-8"))}
for v, folder in (("old", "sonnet5_cli_default"), ("new", "sonnet5")):
    fs = read_frontier_scores(folder)
    for d in ("clef", "cb"):
        ck = DATASETS[d][0]
        have = sorted(fs[ck])
        cs = np.array([fs[ck][i] for i in have])
        cy = np.array([cal_sets[ck][i]["label"] for i in have])
        ts, ys = [], []
        for p in parts[d]:
            items = ev[p] if p in ev else full[p]
            ts += [fs[p][i] for i in range(len(items))]
            ys += [r["label"] for r in items]
        ts, ys = np.array(ts), np.array(ys)
        thr = []
        for seed in range(100):
            idx = np.random.default_rng(seed).permutation(len(cy))[:int(0.8 * len(cy))]
            thr.append(best_threshold(cs[idx], cy[idx]))
        thr = np.array(thr)
        acc_at = [np.mean((ts >= t) == ys) for t in thr]
        vals, cnt = np.unique(np.round(thr, 2), return_counts=True)
        M = R[v][d]["metrics"]
        print(f"{v} {d:4s} t: mean {thr.mean():.2f} sd {thr.std():.3f} values {dict(zip(vals.tolist(), cnt.tolist()))}")
        print(f"         LLM-all-call acc at t: mean {np.mean(acc_at):.3f} sd {np.std(acc_at):.4f} | best test t (oracle) acc "
              f"{max(np.mean((ts >= t) == ys) for t in np.linspace(.02, .98, 97)):.3f}")
        print(f"         raw {M['sonnet_raw']['acc'][0]:.3f} | thr {M['sonnet_thr']['acc'][0]:.3f}±{M['sonnet_thr']['acc'][1]:.4f} | "
              f"replace50 {M['replace_0.5']['acc'][0]:.3f}±{M['replace_0.5']['acc'][1]:.4f} | fuse50 {M['fuse_0.5']['acc'][0]:.3f}±{M['fuse_0.5']['acc'][1]:.4f}")
