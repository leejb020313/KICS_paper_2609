# -*- coding: utf-8 -*-
"""How much information does the verbalized score channel actually carry?

For each distinct score value v we measure P(gold=1 | score=v). The best any
post-hoc method can do with that score alone is to map each v to its majority
label -- that's an ORACLE ceiling (fit on test, so not achievable in practice,
but it upper-bounds every calibration method we've tried).
"""
import json
import sys
import numpy as np
from collections import Counter

def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("parse_ok"):
                    rows.append(r)
    return {r["text"]: r for r in rows}

def entropy(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def analyze(tag, scores, labels, show_table=True):
    scores = np.asarray(scores); labels = np.asarray(labels)
    n = len(labels)
    vals = sorted(set(scores))
    base_rate = labels.mean()

    # oracle ceiling: map each distinct value to its majority label
    correct_ceiling = 0
    rows = []
    H_cond = 0.0
    for v in vals:
        m = scores == v
        cnt = m.sum()
        p1 = labels[m].mean()
        correct_ceiling += max(labels[m].sum(), cnt - labels[m].sum())
        H_cond += (cnt/n) * entropy(p1)
        rows.append((v, cnt, p1))
    ceiling = correct_ceiling / n

    # fixed threshold 0.5
    acc_05 = ((scores >= 0.5).astype(int) == labels).mean()

    H_y = entropy(base_rate)
    mi = H_y - H_cond

    print(f"\n--- {tag} (n={n}) ---")
    print(f"  distinct score values: {len(vals)}  |  gold positive rate: {base_rate:.3f}")
    print(f"  accuracy @ fixed threshold 0.5 : {acc_05:.3f}  (wrong {1-acc_05:.1%})")
    print(f"  ORACLE CEILING (best possible mapping of these values): {ceiling:.3f}  (wrong {1-ceiling:.1%})")
    print(f"  H(Y)={H_y:.3f} bits   H(Y|score)={H_cond:.3f} bits   mutual info={mi:.3f} bits ({100*mi/H_y:.1f}% of label entropy)")
    if show_table:
        print(f"  {'score':>6} {'n':>6} {'P(gold=1)':>10}  {'':>12}")
        for v, cnt, p1 in rows:
            bar = "#" * int(round(p1*20))
            flag = "  <-- COIN FLIP" if 0.35 <= p1 <= 0.65 and cnt >= 0.03*n else ""
            print(f"  {v:>6.2f} {cnt:>6d} {p1:>10.3f}  {bar:<20}{flag}")
    return ceiling, mi, rows

for name in ["clef", "claimbuster"]:
    orig_test = load(f"../../results/{name}_test_scores.jsonl")
    reord_test = load(f"results/{name}_test_scores_reordered.jsonl")
    texts = [t for t in reord_test if t in orig_test]
    ot = np.array([orig_test[t]["confidence_score"] for t in texts])
    rt = np.array([reord_test[t]["confidence_score"] for t in texts])
    yt = np.array([reord_test[t]["label"] for t in texts])

    print("=" * 78)
    print(f"DATASET: {name}")
    print("=" * 78)
    analyze(f"{name}: Gemma 3 4B, ORIGINAL prompt", ot, yt)
    analyze(f"{name}: Gemma 3 4B, REORDERED prompt", rt, yt, show_table=False)

    # joint 2D channel: (orig, reord) pair as a single symbol
    joint = np.array([f"{a:.2f}|{b:.2f}" for a, b in zip(ot, rt)])
    vals = set(joint)
    correct = 0
    for v in vals:
        m = joint == v
        correct += max(yt[m].sum(), m.sum() - yt[m].sum())
    print(f"\n--- {name}: JOINT (orig, reord) 2-D channel ---")
    print(f"  distinct joint symbols: {len(vals)}")
    print(f"  ORACLE CEILING with both scores: {correct/len(yt):.3f}  (wrong {1-correct/len(yt):.1%})")
