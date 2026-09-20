# -*- coding: utf-8 -*-
"""Same 50 claims: how fine-grained is each model's score channel?"""
import json
import numpy as np

with open("frontier_sample.json", encoding="utf-8") as f:
    sample = json.load(f)
with open("frontier_scores.json", encoding="utf-8") as f:
    sonnet = np.array(json.load(f))

labels = np.array([s["label"] for s in sample])
gemma = np.array([s["gemma_score"] for s in sample])

def entropy(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def channel(tag, scores, labels):
    n = len(labels)
    vals = sorted(set(scores))
    correct = 0; H_cond = 0.0
    biggest = None
    for v in vals:
        m = scores == v
        cnt = m.sum(); p1 = labels[m].mean()
        correct += max(labels[m].sum(), cnt - labels[m].sum())
        H_cond += (cnt/n) * entropy(p1)
        if biggest is None or cnt > biggest[1]:
            biggest = (v, cnt, p1)
    H_y = entropy(labels.mean())
    acc = ((scores >= 0.5).astype(int) == labels).mean()
    print(f"  {tag}")
    print(f"    distinct values used  : {len(vals)} over {n} claims  ({len(vals)/n:.2f} per claim)")
    print(f"    largest single bucket : score={biggest[0]:.2f}, n={biggest[1]} ({biggest[1]/n:.0%} of data), P(gold=1)={biggest[2]:.2f}")
    print(f"    accuracy @ 0.5        : {acc:.3f}")
    print(f"    ORACLE CEILING        : {correct/n:.3f}")
    print(f"    mutual information    : {H_y - H_cond:.3f} bits  ({100*(H_y-H_cond)/H_y:.0f}% of label entropy)")

print("=== SAME 50 CLEF CLAIMS, two models ===")
channel("Gemma 3 4B (local, 4B params)", gemma, labels)
print()
channel("Claude Sonnet 5 (frontier)", sonnet, labels)
