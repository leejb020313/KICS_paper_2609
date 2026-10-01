"""Non-inferiority of the cascades to the stronger all-call LLM setting (the paper's core claim).

"No significant difference" is absence of evidence; this checks the claim directly. For each LLM (Claude Sonnet 5,
Claude Haiku 4.5), dataset, call rate and run, the accuracy difference cascade - stronger all-call setting and its
95% paired bootstrap CI over test sentences (final_eval.compare, 2000 resamples). The cascade is non-inferior at
margin delta when the CI's lower bound is above -delta (a two-sided 95% CI, i.e. one-sided alpha = 0.025).

    uv run --locked python scripts/equivalence.py      -> results/paper/equivalence.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_eval as fe  # noqa: E402

RATES = (0.2, 0.3, 0.4, 0.5, 0.6)
MARGINS = (0.01, 0.02)


def run(llm):
    os.environ["CWC_LLM"] = llm
    out = {}
    for name in fe.DATASETS:
        fe.evaluate(name, full=True)
        y, preds = fe.LAST_PREDS[name]
        mean = {k: float(np.mean([np.mean(p == y) for p in v])) for k, v in preds.items()}
        best = max(("sonnet_raw", "sonnet_thr"), key=mean.get)
        seed = lambda k, sd: preds[k][sd] if len(preds[k]) > 1 else preds[k][0]
        rows = {}
        for v in ("fuse", "replace"):
            for b in RATES:
                cs = [fe.compare(y, seed(f"{v}_{b}", sd), seed(best, sd)) for sd in range(fe.SEEDS)]
                lo = [c["ci"][0] for c in cs]
                rows[f"{v}_{b}"] = dict(diff=[c["diff"] for c in cs], ci_lo=lo, ci_hi=[c["ci"][1] for c in cs],
                                        worst_lo=min(lo),
                                        noninferior={str(m): sum(l > -m for l in lo) for m in MARGINS})
        out[name] = dict(best_full=best, best_full_acc=mean[best], rows=rows)
    return out


def main():
    res = {"sonnet5": run("sonnet5"), "haiku45": run("haiku45")}
    with open(os.path.join(fe.PAPER, "equivalence.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for llm, r in res.items():
        for name, d in r.items():
            print(f"\n## {llm} {name}: stronger all-call = {d['best_full']} ({d['best_full_acc']:.3f})")
            print("  method       mean diff   worst 95% CI low   non-inferior runs (1pp / 2pp)")
            for k, v in d["rows"].items():
                print(f"  {k:12s} {100 * np.mean(v['diff']):+6.2f}pp   {100 * v['worst_lo']:+6.2f}pp"
                      f"          {v['noninferior']['0.01']}/5 / {v['noninferior']['0.02']}/5")


if __name__ == "__main__":
    main()
