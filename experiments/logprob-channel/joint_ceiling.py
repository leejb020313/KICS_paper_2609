# -*- coding: utf-8 -*-
"""Oracle ceiling of the JOINT channel: (orig, reordered, digit, yesno_logit)
all together as one multi-dimensional symbol, vs each alone.

If combining everything we've measured today gives a meaningfully higher
ceiling than any single channel, that's the real question -- is there a big
number hiding in the data we already have, or have we actually hit the wall.
"""
import json
import numpy as np
from pathlib import Path

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
    p = np.clip(p, 1e-12, 1-1e-12)
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def quantize_ties(scores, n_bins):
    scores = np.asarray(scores, dtype=float)
    uniq, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    quota = max(1, len(scores)/n_bins)
    bin_of_value = np.empty(len(uniq), dtype=int)
    cur_bin, filled = 0, 0
    for i, c in enumerate(counts):
        bin_of_value[i] = cur_bin
        filled += c
        if filled >= quota and cur_bin < n_bins-1:
            cur_bin += 1; filled = 0
    return bin_of_value[inverse]

def ceiling_and_mi(symbols, labels):
    n = len(labels)
    correct, H_cond = 0, 0.0
    for v in set(symbols):
        m = symbols == v
        cnt = m.sum(); p1 = labels[m].mean()
        correct += max(labels[m].sum(), cnt - labels[m].sum())
        H_cond += (cnt/n) * entropy(p1)
    H_y = entropy(labels.mean())
    return correct/n, H_y - H_cond

for name in ["clef", "claimbuster"]:
    orig = load(f"../../results/{name}_test_scores.jsonl")
    reord = load(f"../prompt-reorder-ablation/results/{name}_test_scores_reordered.jsonl")
    digit = load(f"results/{name}_test_digit.jsonl")
    yesno = load(f"results/{name}_test_yesno.jsonl")

    texts = [t for t in orig if t in reord and t in digit and t in yesno]
    labels = np.array([orig[t]["label"] for t in texts])
    o = np.array([orig[t]["confidence_score"] for t in texts])
    r = np.array([reord[t]["confidence_score"] for t in texts])
    d = np.array([digit[t]["confidence_score"] for t in texts])
    y_logit = np.array([(yesno[t].get("detail") or {}).get("logit", 0.0) for t in texts])

    print(f"\n=== {name} (n={len(texts)}) ===")
    for tag, s in [("original", o), ("reordered", r), ("digit", d), ("yesno[logit]", y_logit)]:
        sym = quantize_ties(s, 40)
        c, mi = ceiling_and_mi(sym, labels)
        print(f"  {tag:14s} alone: ceiling={c:.3f}  MI={mi:.3f} bits")

    # joint symbol: quantize each into 8 equal-freq buckets, concatenate as tuple
    n_bins_joint = 8
    joint = np.stack([quantize_ties(s, n_bins_joint) for s in [o, r, d, y_logit]], axis=1)
    joint_ids = np.array([hash(tuple(row.tolist())) for row in joint])
    c, mi = ceiling_and_mi(joint_ids, labels)
    print(f"  JOINT (all 4, {n_bins_joint} bins each): ceiling={c:.3f}  MI={mi:.3f} bits   "
          f"(distinct symbols: {len(set(joint_ids))})")

    # also: fit an actual logistic regression on calib and measure REALIZED test accuracy
    # (not just the oracle bound) to see how much of that ceiling is reachable
    from sklearn.linear_model import LogisticRegression
    orig_c = load(f"../../results/{name}_calib_scores.jsonl")
    reord_c = load(f"../prompt-reorder-ablation/results/{name}_calib_scores_reordered.jsonl")
    digit_c = load(f"results/{name}_calib_digit.jsonl")
    yesno_c = load(f"results/{name}_calib_yesno.jsonl")
    ctexts = [t for t in orig_c if t in reord_c and t in digit_c and t in yesno_c]
    yc = np.array([orig_c[t]["label"] for t in ctexts])
    oc = np.array([orig_c[t]["confidence_score"] for t in ctexts])
    rc = np.array([reord_c[t]["confidence_score"] for t in ctexts])
    dc = np.array([digit_c[t]["confidence_score"] for t in ctexts])
    ylc = np.array([(yesno_c[t].get("detail") or {}).get("logit", 0.0) for t in ctexts])

    Xc = np.column_stack([oc, rc, dc, ylc])
    Xt = np.column_stack([o, r, d, y_logit])
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xc, yc)
    pred = clf.predict(Xt)
    acc = (pred == labels).mean()
    print(f"  REALIZED (logistic reg on all 4, calib-fit): accuracy={acc:.3f}  (wrong {1-acc:.1%})")

print("\n\n### Non-linear realization (Gradient Boosting on the same 4 features) ###")
from sklearn.ensemble import GradientBoostingClassifier
for name in ["clef", "claimbuster"]:
    orig = load(f"../../results/{name}_test_scores.jsonl")
    reord = load(f"../prompt-reorder-ablation/results/{name}_test_scores_reordered.jsonl")
    digit = load(f"results/{name}_test_digit.jsonl")
    yesno = load(f"results/{name}_test_yesno.jsonl")
    texts = [t for t in orig if t in reord and t in digit and t in yesno]
    labels = np.array([orig[t]["label"] for t in texts])
    o = np.array([orig[t]["confidence_score"] for t in texts])
    r = np.array([reord[t]["confidence_score"] for t in texts])
    d = np.array([digit[t]["confidence_score"] for t in texts])
    y_logit = np.array([(yesno[t].get("detail") or {}).get("logit", 0.0) for t in texts])

    orig_c = load(f"../../results/{name}_calib_scores.jsonl")
    reord_c = load(f"../prompt-reorder-ablation/results/{name}_calib_scores_reordered.jsonl")
    digit_c = load(f"results/{name}_calib_digit.jsonl")
    yesno_c = load(f"results/{name}_calib_yesno.jsonl")
    ctexts = [t for t in orig_c if t in reord_c and t in digit_c and t in yesno_c]
    yc = np.array([orig_c[t]["label"] for t in ctexts])
    oc = np.array([orig_c[t]["confidence_score"] for t in ctexts])
    rc = np.array([reord_c[t]["confidence_score"] for t in ctexts])
    dc = np.array([digit_c[t]["confidence_score"] for t in ctexts])
    ylc = np.array([(yesno_c[t].get("detail") or {}).get("logit", 0.0) for t in ctexts])

    Xc = np.column_stack([oc, rc, dc, ylc])
    Xt = np.column_stack([o, r, d, y_logit])

    # small grid, calib-internal split to avoid picking depth on the test set
    from sklearn.model_selection import train_test_split
    Xtr, Xval, ytr, yval = train_test_split(Xc, yc, test_size=0.3, random_state=0, stratify=yc)
    best = None
    for depth in [2, 3, 4]:
        for n_est in [50, 100, 200]:
            gb = GradientBoostingClassifier(max_depth=depth, n_estimators=n_est, random_state=0)
            gb.fit(Xtr, ytr)
            acc = (gb.predict(Xval) == yval).mean()
            if best is None or acc > best[0]:
                best = (acc, depth, n_est)
    _, bd, bn = best
    gb = GradientBoostingClassifier(max_depth=bd, n_estimators=bn, random_state=0)
    gb.fit(Xc, yc)
    pred = gb.predict(Xt)
    acc = (pred == labels).mean()
    print(f"{name}: GBM(depth={bd}, n_est={bn})  REALIZED accuracy={acc:.3f}  (wrong {1-acc:.1%})")
