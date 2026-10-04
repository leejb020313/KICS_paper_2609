"""Old (plain CLI mode) vs new (clean mode) Sonnet 5 results, both over 100 runs: the items listed in
results/analysis/rescore_clean_preregistration.md. Old files are read from git commit e419d03.

    uv run --locked python scripts/compare_rescore.py
"""
import json
import os
import subprocess
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cwcascade.data import read_json, DATA, read_frontier_scores  # noqa: E402

OLD = "e419d03"
P = "results/paper/"


def old(f):
    return json.loads(subprocess.check_output(["git", "show", f"{OLD}:{P}{f}"]).decode("utf-8"))


def new(f):
    return json.load(open(P + f, encoding="utf-8"))


ver = {"old (plain)": {f: old(f) for f in ("final_results_full.json", "seed_robustness_full.json", "equivalence.json")},
       "new (clean)": {f: new(f) for f in ("final_results_full.json", "seed_robustness_full.json", "equivalence.json")}}
HK = new("seed_robustness_full_haiku45.json")

# 1. score behaviour on the full test sets
ev, full = read_json(os.path.join(DATA, "frontier", "eval_set.json")), read_json(os.path.join(DATA, "frontier", "full_set.json"))
parts = {"clef": ["clef", "clefdev", "clefoff"], "cb": ["cb", "cbrest"]}
for name, folder in (("old (plain)", "sonnet5_cli_default"), ("new (clean)", "sonnet5"), ("Haiku (clean)", "haiku45")):
    fs = read_frontier_scores(folder)
    for d, ps in parts.items():
        s, y = [], []
        for p in ps:
            items = ev[p] if p in ev else full[p]
            s += [fs[p][i] for i in range(len(items))]
            y += [r["label"] for r in items]
        s, y = np.array(s), np.array(y)
        k = s >= .5
        print(f"{name:14s} {d:4s} flag={100 * k.mean():4.1f}% (labels {100 * y.mean():.0f}%) rec={k[y == 1].mean():.2f} "
              f"prec={y[k].mean():.2f} AUC={roc_auc_score(y, s):.3f}")

print()
for v, R in ver.items():
    for d in ("clef", "cb"):
        M = R["final_results_full.json"][d]["metrics"]
        S = R["seed_robustness_full.json"][d]
        E = R["equivalence.json"]["sonnet5"][d]["rows"]
        best = S["best_full"]
        a = lambda k: M[k]["acc"][0]
        fl = S["per_seed"][f"fuse_0.5_vs_best_full({best})"]
        lower = sum(1 for x, p in zip(fl["diff"], fl["p"]) if p < .05 and x < 0)
        higher = sum(1 for x, p in zip(fl["diff"], fl["p"]) if p < .05 and x > 0)
        ni = lambda k: sum(lo >= -0.006 for lo in E[k]["ci_lo"])
        print(f"{v:12s} {d:4s} svm={a('svm'):.3f} raw={a('sonnet_raw'):.3f} thr={a('sonnet_thr'):.3f} nnppi={a('sonnet_nnppi'):.3f} "
              f"repl50={a('replace_0.5'):.3f} fuse50={a('fuse_0.5'):.3f} | best={best} "
              f"fuse-best={100 * np.mean(fl['diff']):+.2f}pp (sig lower {lower}, sig higher {higher}) "
              f"meet-0.6: fuse {ni('fuse_0.5')} repl {ni('replace_0.5')} | fuse>nnppi sig {S['per_seed']['fuse_0.5_vs_sonnet_nnppi']['n_sig']} "
              f"| reach fuse {S['first_rate_reaching_best_full']['fuse']} repl {S['first_rate_reaching_best_full']['replace']}")
        gaps = [round(100 * (M[f'fuse_{b}']['acc'][0] - M[f'replace_{b}']['acc'][0]), 2) for b in (.1, .2, .3, .4, .5, .7, 1.0)]
        print(f"{'':17s} fuse-replace pp at 10..100%: {gaps}")
print()
for d in ("clef", "cb"):
    print(f"Haiku {d}: raw={HK[d]['mean_acc']['sonnet_raw']:.3f} fuse50={HK[d]['mean_acc']['fuse_0.5']:.3f} "
          f"| Sonnet new raw={ver['new (clean)']['final_results_full.json'][d]['metrics']['sonnet_raw']['acc'][0]:.3f}")
