"""Two design questions, answered with the existing Claude Sonnet 5 scores (no new LLM calls):

1. First stage: was the embedding SVM the right cheap classifier? Alternatives on the same learning set:
   logistic regression on the same embeddings, an RBF-SVM with C tuned by CV, and an RBF-SVM on a larger
   sentence embedder (all-mpnet-base-v2).
2. Fusion rule: is LR(d, s) (Eq. 1) the best combination? Alternatives: LR on logit(s), LR with a d*s
   interaction, gradient boosting on (d, s), and an untrained average of the Platt-scaled d and s.

Protocol as in the paper: 5 runs on 80% subsamples of the learning set, fusion trained on 5-fold out-of-fold
d and the learning-set LLM scores, the queried rho of test sentences = smallest |d|. The rule to pick a variant
is the learning-set (out-of-fold) accuracy at 50% calls; test accuracy is reported next to it.

    uv run --locked python scripts/ablation_stage_fusion.py   -> results/ablation_stage_fusion.json
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.data import RESULTS, load_frontier  # noqa: E402

SEEDS = 5
RATES = (0.0, 0.3, 0.5, 1.0)
EPS = 1e-4


def route(base, alt, order, rho):
    p = base.copy()
    k = int(round(rho * len(base)))
    p[order[:k]] = alt[order[:k]]
    return p


def stage_models():
    return {
        "svm_rbf (paper)": lambda: SVC(class_weight="balanced", random_state=0),
        "logreg": lambda: LogisticRegression(class_weight="balanced", max_iter=2000),
        "svm_rbf_tunedC": lambda: GridSearchCV(SVC(class_weight="balanced", random_state=0),
                                               {"C": [0.3, 1, 3, 10]}, cv=5),
    }


def logit(s):
    s = np.clip(s, EPS, 1 - EPS)
    return np.log(s / (1 - s))


FUSIONS = {
    "LR(d,s) (paper)": lambda d, s: np.c_[d, s],
    "LR(d,logit s)": lambda d, s: np.c_[d, logit(s)],
    "LR(d,s,d*s)": lambda d, s: np.c_[d, s, d * s],
}


def fuse_predict(kind, d_cal, s_cal, y_cal, d_te, s_te):
    if kind in FUSIONS:
        m = LogisticRegression(class_weight="balanced").fit(FUSIONS[kind](d_cal, s_cal), y_cal)
        return m.predict(FUSIONS[kind](d_te, s_te)), m, FUSIONS[kind]
    if kind == "GBM(d,s)":
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, class_weight="balanced",
                                           random_state=0).fit(np.c_[d_cal, s_cal], y_cal)
        return m.predict(np.c_[d_te, s_te]), m, lambda d, s: np.c_[d, s]
    if kind == "avg(Platt d, s) untrained":
        platt = LogisticRegression().fit(d_cal[:, None], y_cal)
        f = lambda d, s: ((platt.predict_proba(d[:, None])[:, 1] + s) / 2 >= 0.5).astype(int)
        return f(d_te, s_te), None, None
    raise KeyError(kind)


FUSION_KINDS = list(FUSIONS) + ["GBM(d,s)", "avg(Platt d, s) untrained"]


def oof_fused_acc(kind, d_cal, s_cal, y_cal, rho, seed):
    """Learning-set accuracy of the routed cascade with out-of-fold fused predictions."""
    p_fuse = np.zeros(len(y_cal), int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(d_cal[:, None], y_cal):
        p_fuse[te] = fuse_predict(kind, d_cal[tr], s_cal[tr], y_cal[tr], d_cal[te], s_cal[te])[0]
    return float(np.mean(route((d_cal > 0).astype(int), p_fuse, np.argsort(np.abs(d_cal)), rho) == y_cal))


def embed_mpnet(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return np.asarray(m.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64))


def run(name):
    test, calib, test_items = load_frontier(name, full=True)
    from cwcascade.data import read_json, DATA, DATASETS  # noqa: F401
    out = {"stage": {}, "fusion": {}}
    # larger embedder for the stage-1 question: embed the same sentences (calib items in the same order)
    calib_texts = None
    try:
        from cwcascade.data import DATA as _D
        calib_key = DATASETS[name][0]
        items = read_json(os.path.join(_D, "frontier", "calib_set.json"))[calib_key]
        from cwcascade.data import read_frontier_scores
        have = sorted(read_frontier_scores()[calib_key])
        calib_texts = [items[i]["text"] for i in have]
    except Exception as e:  # pragma: no cover
        print("mpnet skipped:", e)
    X_alt = None
    if calib_texts is not None:
        X_alt = (embed_mpnet(calib_texts), embed_mpnet([r["text"] for r in test_items]))

    stage_variants = dict(stage_models())
    if X_alt is not None:
        stage_variants["svm_rbf on mpnet"] = stage_variants["svm_rbf (paper)"]

    for sname, make in stage_variants.items():
        Xc, Xt = (X_alt if sname.endswith("mpnet") else (calib.X, test.X))
        acc = {r: [] for r in RATES}
        cal50 = []
        for seed in range(SEEDS):
            idx = np.random.default_rng(seed).choice(len(calib.y), int(0.8 * len(calib.y)), replace=False)
            Xs, ys, ss = Xc[idx], calib.y[idx], calib.s[idx]
            clf = make().fit(Xs, ys)
            d_te = clf.decision_function(Xt)
            d_cal = cross_val_predict(make(), Xs, ys, cv=5, method="decision_function")
            p_fuse = fuse_predict("LR(d,s) (paper)", d_cal, ss, ys, d_te, test.s)[0]
            order = np.argsort(np.abs(d_te))
            for r in RATES:
                acc[r].append(float(np.mean(route((d_te > 0).astype(int), p_fuse, order, r) == test.y)))
            cal50.append(oof_fused_acc("LR(d,s) (paper)", d_cal, ss, ys, 0.5, seed))
        out["stage"][sname] = {"test_acc": {str(r): float(np.mean(v)) for r, v in acc.items()},
                               "learning_oof_acc@0.5": float(np.mean(cal50))}

    for kind in FUSION_KINDS:
        acc = {r: [] for r in RATES}
        cal50 = []
        for seed in range(SEEDS):
            idx = np.random.default_rng(seed).choice(len(calib.y), int(0.8 * len(calib.y)), replace=False)
            Xs, ys, ss = calib.X[idx], calib.y[idx], calib.s[idx]
            make = stage_models()["svm_rbf (paper)"]
            d_te = make().fit(Xs, ys).decision_function(test.X)
            d_cal = cross_val_predict(make(), Xs, ys, cv=5, method="decision_function")
            p_fuse = fuse_predict(kind, d_cal, ss, ys, d_te, test.s)[0]
            order = np.argsort(np.abs(d_te))
            for r in RATES:
                acc[r].append(float(np.mean(route((d_te > 0).astype(int), p_fuse, order, r) == test.y)))
            cal50.append(oof_fused_acc(kind, d_cal, ss, ys, 0.5, seed))
        out["fusion"][kind] = {"test_acc": {str(r): float(np.mean(v)) for r, v in acc.items()},
                               "learning_oof_acc@0.5": float(np.mean(cal50))}
    return out


def main():
    res = {name: run(name) for name in ("clef", "cb")}
    with open(os.path.join(RESULTS, "ablation_stage_fusion.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for name, r in res.items():
        for part in ("stage", "fusion"):
            print(f"\n## {name} — {part}   (test acc at call rate 0/30/50/100%, learning-set OOF acc at 50%)")
            for k, v in r[part].items():
                t = v["test_acc"]
                print(f"  {k:28s} " + " ".join(f"{t[str(x)]:.3f}" for x in RATES) + f"   | learn {v['learning_oof_acc@0.5']:.3f}")


if __name__ == "__main__":
    main()
