import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("parse_ok"):
                    rows.append(r)
    return {r["text"]: r for r in rows}

def build_xy(orig_d, reord_d):
    texts = [t for t in reord_d if t in orig_d]
    orig_s = np.array([orig_d[t]["confidence_score"] for t in texts])
    reord_s = np.array([reord_d[t]["confidence_score"] for t in texts])
    labels = np.array([reord_d[t]["label"] for t in texts])
    X = np.column_stack([orig_s, reord_s, reord_s - orig_s, np.abs(reord_s - orig_s)])
    return X, orig_s, reord_s, labels, texts

def report(name, scores, labels):
    pred = (scores >= 0.5).astype(int)
    wrong = pred != labels
    print(f"  {name:35s} wrong={wrong.sum():4d}/{len(labels)} ({wrong.mean():.1%})  F1={f1_score(labels,pred,average='weighted'):.3f}")

for name in ["clef", "claimbuster"]:
    orig_calib = load(f"../../results/{name}_calib_scores.jsonl")
    reord_calib = load(f"results/{name}_calib_scores_reordered.jsonl")
    orig_test = load(f"../../results/{name}_test_scores.jsonl")
    reord_test = load(f"results/{name}_test_scores_reordered.jsonl")

    Xc, oc, rc, yc, _ = build_xy(orig_calib, reord_calib)
    Xt, ot, rt, yt, _ = build_xy(orig_test, reord_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xc, yc)
    p_test = clf.predict_proba(Xt)[:, 1]

    print(f"\n=== {name} (n_test={len(yt)}) ===")
    report("original only", ot, yt)
    report("reordered only", rt, yt)
    report("simple average", (ot+rt)/2, yt)
    report("logistic ensemble (orig,reord,diff,|diff|)", p_test, yt)
    print(f"  ensemble coefficients (orig, reord, diff, |diff|): {np.round(clf.coef_[0],3)}  intercept={clf.intercept_[0]:.3f}")
