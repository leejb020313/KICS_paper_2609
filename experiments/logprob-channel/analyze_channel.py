"""Measure whether reading logprobs actually widened the score channel.

Compares, on the same test claims:
  - the verbalized score from the main reproduction (results/*_test_scores.jsonl)
  - each logprob mode scored here

Reports, per channel: distinct values, accuracy at the best calib-fit threshold,
the ORACLE ceiling (best possible mapping of the channel's values to labels),
and mutual information with the gold label. The ceiling and MI are what matter:
they bound every post-hoc calibration method, including NN-PPI and ours.

    python analyze_channel.py --dataset clef
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("parse_ok") and r.get("confidence_score") is not None:
                    rows.append(r)
    return {r["text"]: r for r in rows}


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def quantize_for_ceiling(scores, n_bins=40):
    """A continuous channel would have a trivially perfect oracle ceiling (every
    value unique), so bin it before measuring. 40 equal-width bins is far finer
    than the ~13 values the verbalized channel uses, and coarse enough that the
    ceiling stays meaningful rather than memorizing the test set."""
    return np.clip((np.asarray(scores) * n_bins).astype(int), 0, n_bins - 1)


def channel_stats(tag, scores, labels, calib_scores=None, calib_labels=None, n_bins=40):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n = len(labels)
    n_distinct = len(set(np.round(scores, 6)))

    symbols = quantize_for_ceiling(scores, n_bins)
    correct, H_cond = 0, 0.0
    for v in set(symbols):
        m = symbols == v
        cnt = m.sum()
        p1 = labels[m].mean()
        correct += max(labels[m].sum(), cnt - labels[m].sum())
        H_cond += (cnt / n) * entropy(p1)
    ceiling = correct / n
    mi = entropy(labels.mean()) - H_cond

    # honest accuracy: threshold fitted on calib if available, else fixed 0.5
    if calib_scores is not None and len(calib_scores):
        grid = np.linspace(0.02, 0.98, 97)
        errs = [np.mean((np.asarray(calib_scores) >= t).astype(int) != np.asarray(calib_labels))
                for t in grid]
        thr = float(grid[int(np.argmin(errs))])
    else:
        thr = 0.5
    acc = np.mean((scores >= thr).astype(int) == labels)

    print(f"  {tag}")
    print(f"    distinct values   : {n_distinct}")
    print(f"    threshold (calib) : {thr:.2f}")
    print(f"    accuracy          : {acc:.3f}   (wrong {1-acc:.1%})")
    print(f"    ORACLE CEILING    : {ceiling:.3f}   (wrong {1-ceiling:.1%})   [{n_bins}-bin]")
    print(f"    mutual info       : {mi:.3f} bits ({100*mi/entropy(labels.mean()):.0f}% of label entropy)")
    return dict(acc=acc, ceiling=ceiling, mi=mi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["clef", "claimbuster"], required=True)
    ap.add_argument("--bins", type=int, default=40)
    args = ap.parse_args()
    name = args.dataset

    sources = [("verbalized (main repro)", ROOT / "results" / f"{name}_test_scores.jsonl",
                ROOT / "results" / f"{name}_calib_scores.jsonl")]
    for mode in ["yesno", "digit", "verbal"]:
        t = Path("results") / f"{name}_test_{mode}.jsonl"
        c = Path("results") / f"{name}_calib_{mode}.jsonl"
        if t.exists():
            sources.append((f"logprob:{mode}", t, c))

    if len(sources) == 1:
        print("No logprob results found yet -- run score_with_logprobs.py first.")
        return

    print(f"=== {name}: score channel comparison ===")
    for tag, test_path, calib_path in sources:
        test = load(test_path)
        texts = sorted(test)
        scores = [test[t]["confidence_score"] for t in texts]
        labels = [test[t]["label"] for t in texts]
        cs, cl = None, None
        if calib_path.exists():
            calib = load(calib_path)
            ct = sorted(calib)
            cs = [calib[t]["confidence_score"] for t in ct]
            cl = [calib[t]["label"] for t in ct]
        channel_stats(tag, scores, labels, cs, cl, args.bins)
        print()


if __name__ == "__main__":
    main()
