"""Two checks on the paper's claims, on the full held-out sets, with the exact predictions of final_eval.py --full.

1. Credit for "half the calls": is it the cascade (replacement already reaches all-call accuracy at 50%)
   or the fusion? For every call rate, mean accuracy of both cascades minus the stronger all-call LLM
   setting, and the smallest call rate at which each cascade reaches it.
2. Seed dependence of the significance tests: the paper tests on seed-0 predictions only. Every
   significance claim of the paper is recomputed on each of the 5 seeds (McNemar, exact).

    uv run --locked python scripts/seed_robustness.py      -> results/seed_robustness_full.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_eval as fe  # noqa: E402

OUT = os.path.join(fe.RESULTS, "seed_robustness_full.json")


def acc(y, p):
    return float(np.mean(p == y))


def seed_pred(preds, key, sd):
    """Per-seed prediction; methods without calib refits (Gemma, raw LLM) have a single entry."""
    plist = preds[key]
    return plist[sd] if len(plist) > 1 else plist[0]


def main():
    out = {}
    for name in fe.DATASETS:
        fe.evaluate(name, full=True)
        y, preds = fe.LAST_PREDS[name]
        mean = {k: float(np.mean([acc(y, p) for p in v])) for k, v in preds.items()}
        best = max(("sonnet_raw", "sonnet_thr"), key=mean.get)

        # 1. credit for the call reduction
        curve = {b: dict(fuse=mean[f"fuse_{b}"] - mean[best], replace=mean[f"replace_{b}"] - mean[best]) for b in fe.BUDGETS}
        reach = {v: next((b for b in fe.BUDGETS if mean[f"{v}_{b}"] >= mean[best]), None) for v in ("fuse", "replace")}

        # 2. every significance claim, per seed
        claims = {"svm_vs_gemma_nnppi_sel": ("svm", "gemma_nnppi_sel"),
                  f"fuse_0.5_vs_best_full({best})": ("fuse_0.5", best),
                  f"replace_0.5_vs_best_full({best})": ("replace_0.5", best),
                  "fuse_0.5_vs_sonnet_nnppi": ("fuse_0.5", "sonnet_nnppi"),
                  "fuse_0.5_vs_fuse_1.0": ("fuse_0.5", "fuse_1.0")}
        for b in fe.BUDGETS[1:]:
            claims[f"fuse_{b}_vs_replace_{b}"] = (f"fuse_{b}", f"replace_{b}")
        per_seed = {}
        for cname, (a, b) in claims.items():
            rows = [fe.compare(y, seed_pred(preds, a, sd), seed_pred(preds, b, sd)) for sd in range(fe.SEEDS)]
            per_seed[cname] = dict(diff=[r["diff"] for r in rows], p=[r["mcnemar_p"] for r in rows],
                                   n_sig=sum(r["mcnemar_p"] < 0.05 for r in rows),
                                   same_sign=len({np.sign(r["diff"]) for r in rows if r["diff"] != 0}) <= 1)
        out[name] = dict(best_full=best, mean_acc={k: mean[k] for k in ("svm", "gemma_nnppi_sel", best, "sonnet_nnppi")},
                         gap_to_best_full=curve, first_rate_reaching_best_full=reach, per_seed=per_seed)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for name, r in out.items():
        print(f"\n######## {name}: stronger all-call LLM = {r['best_full']} ({r['mean_acc'][r['best_full']]:.3f})")
        print("  call rate   fuse - best   replace - best")
        for b, g in r["gap_to_best_full"].items():
            print(f"  {b:>8.0%}   {100 * g['fuse']:+6.2f}pp     {100 * g['replace']:+6.2f}pp")
        print(f"  first call rate reaching all-call accuracy: {r['first_rate_reaching_best_full']}")
        print("  claim                                   p per seed (0..4)                    #sig  same sign")
        for c, v in r["per_seed"].items():
            print(f"  {c:38s} " + " ".join(f"{p:.3f}" for p in v["p"]) + f"   {v['n_sig']}/5   {v['same_sign']}"
                  + "   diffs(pp) " + " ".join(f"{100 * d:+.1f}" for d in v["diff"]))


if __name__ == "__main__":
    main()
