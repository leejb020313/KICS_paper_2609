# -*- coding: utf-8 -*-
"""The 0.881 'joint ceiling' from joint_ceiling.py is suspicious: 141 distinct
4-way symbols over only 318 test points is ~2.3 points/symbol, and many
symbols hold exactly 1 point -- the 'oracle' majority vote is then just
memorizing that one test point's own label. That's not a ceiling, it's
overfitting the measurement to the test set itself.

Correct procedure: fit bin edges AND per-bin majority label on CALIB only,
then apply frozen to TEST. This is what any real method would have to do.
"""
import json
import numpy as np

def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("parse_ok") and r.get("confidence_score") is not None:
                    rows.append(r)
    return {r["text"]: r for r in rows}

def fit_quantile_edges(scores, n_bins):
    return np.unique(np.quantile(scores, np.linspace(0, 1, n_bins+1)[1:-1]))

def apply_edges(scores, edges):
    return np.searchsorted(edges, scores)

for name in ["clef", "claimbuster"]:
    orig = load(f"../../results/{name}_test_scores.jsonl")
    reord = load(f"../prompt-reorder-ablation/results/{name}_test_scores_reordered.jsonl")
    digit = load(f"results/{name}_test_digit.jsonl")
    yesno = load(f"results/{name}_test_yesno.jsonl")
    orig_c = load(f"../../results/{name}_calib_scores.jsonl")
    reord_c = load(f"../prompt-reorder-ablation/results/{name}_calib_scores_reordered.jsonl")
    digit_c = load(f"results/{name}_calib_digit.jsonl")
    yesno_c = load(f"results/{name}_calib_yesno.jsonl")

    texts = [t for t in orig if t in reord and t in digit and t in yesno]
    labels = np.array([orig[t]["label"] for t in texts])
    o = np.array([orig[t]["confidence_score"] for t in texts])
    r = np.array([reord[t]["confidence_score"] for t in texts])
    d = np.array([digit[t]["confidence_score"] for t in texts])
    yl = np.array([(yesno[t].get("detail") or {}).get("logit", 0.0) for t in texts])

    ctexts = [t for t in orig_c if t in reord_c and t in digit_c and t in yesno_c]
    yc = np.array([orig_c[t]["label"] for t in ctexts])
    oc = np.array([orig_c[t]["confidence_score"] for t in ctexts])
    rc = np.array([reord_c[t]["confidence_score"] for t in ctexts])
    dc = np.array([digit_c[t]["confidence_score"] for t in ctexts])
    ylc = np.array([(yesno_c[t].get("detail") or {}).get("logit", 0.0) for t in ctexts])

    print(f"\n=== {name}: HONEST joint ceiling (bins+labels fit on calib, frozen, applied to test) ===")
    for n_bins_joint in [2, 3, 4, 8]:
        edges = [fit_quantile_edges(col, n_bins_joint) for col in [oc, rc, dc, ylc]]
        calib_bins = np.stack([apply_edges(col, e) for col, e in zip([oc, rc, dc, ylc], edges)], axis=1)
        test_bins = np.stack([apply_edges(col, e) for col, e in zip([o, r, d, yl], edges)], axis=1)

        calib_sym = [tuple(row.tolist()) for row in calib_bins]
        test_sym = [tuple(row.tolist()) for row in test_bins]

        # majority label per symbol, computed ONLY from calib
        from collections import defaultdict
        votes = defaultdict(lambda: [0, 0])
        for s, y in zip(calib_sym, yc):
            votes[s][int(y)] += 1

        n_seen_in_calib = sum(1 for s in test_sym if s in votes)
        preds = np.array([np.argmax(votes[s]) if s in votes else int(yc.mean() >= 0.5)
                          for s in test_sym])
        acc = (preds == labels).mean()
        print(f"  {n_bins_joint}^4={n_bins_joint**4:4d} symbols: test accuracy={acc:.3f} (wrong {1-acc:.1%})  "
              f"  {n_seen_in_calib}/{len(texts)} test points landed in a symbol seen in calib")
