import json, sys
import numpy as np
sys.path.insert(0, "../../src")
from nnppi.calibration import NNPPIConfig, nn_ppi_apply
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

def theta_gated_localreg(raw_scores, emb, calib_raw, calib_emb, calib_labels,
                          k_reg=15, purity_thresh=0.7, sim_thresh=0.6, ridge=2.0):
    sims = emb @ calib_emb.T
    idx = np.argpartition(-sims, kth=k_reg-1, axis=1)[:, :k_reg]
    theta = np.empty(len(raw_scores))
    for i in range(len(raw_scores)):
        j = idx[i]
        x = calib_raw[j]; y = calib_labels[j].astype(float)
        nb_sims = sims[i, j]
        purity = max(y.mean(), 1 - y.mean())
        max_sim = nb_sims.max()
        p_trust = np.clip((purity - 0.5) / max(purity_thresh - 0.5, 1e-6), 0, 1)
        s_trust = np.clip((max_sim - 0.1) / max(sim_thresh - 0.1, 1e-6), 0, 1)
        lam = p_trust * s_trust
        if x.std() < 1e-6:
            a, b = np.mean(y - x), 1.0
        else:
            X = np.vstack([x, np.ones_like(x)]).T
            XtX = X.T @ X + ridge*np.eye(2)
            coef = np.linalg.solve(XtX, X.T @ y)
            b, a = coef[0], coef[1]
        a_shrunk = lam * a
        b_shrunk = 1.0 + lam * (b - 1.0)
        theta[i] = np.clip(a_shrunk + b_shrunk * raw_scores[i], 0, 1)
    return theta

def report(name, scores, labels):
    pred = (scores >= 0.5).astype(int)
    wrong = pred != labels
    print(f"  {name:42s} wrong={wrong.sum():4d}/{len(labels)} ({wrong.mean():.1%})  F1={f1_score(labels,pred,average='weighted'):.3f}")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
def embed(texts):
    return np.asarray(model.encode(texts, show_progress_bar=False, normalize_embeddings=True))

for name in ["clef", "claimbuster"]:
    orig_calib = load(f"../../results/{name}_calib_scores.jsonl")
    reord_calib = load(f"results/{name}_calib_scores_reordered.jsonl")
    orig_test = load(f"../../results/{name}_test_scores.jsonl")
    reord_test = load(f"results/{name}_test_scores_reordered.jsonl")

    calib_texts = [t for t in reord_calib if t in orig_calib]
    test_texts = [t for t in reord_test if t in orig_test]

    oc = np.array([orig_calib[t]["confidence_score"] for t in calib_texts])
    rc = np.array([reord_calib[t]["confidence_score"] for t in calib_texts])
    yc = np.array([reord_calib[t]["label"] for t in calib_texts])
    ot = np.array([orig_test[t]["confidence_score"] for t in test_texts])
    rt = np.array([reord_test[t]["confidence_score"] for t in test_texts])
    yt = np.array([reord_test[t]["label"] for t in test_texts])

    rng = np.random.default_rng(0)
    n = len(yc)
    perm = rng.permutation(n)
    n_pool = int(0.7*n)
    pool_idx, tune_idx = perm[:n_pool], perm[n_pool:]

    Xc_pool = np.column_stack([oc[pool_idx], rc[pool_idx], rc[pool_idx]-oc[pool_idx], np.abs(rc[pool_idx]-oc[pool_idx])])
    ens_clf = LogisticRegression(max_iter=1000)
    ens_clf.fit(Xc_pool, yc[pool_idx])

    def ens_score(o, r):
        X = np.column_stack([o, r, r-o, np.abs(r-o)])
        return ens_clf.predict_proba(X)[:, 1]

    ens_pool = ens_score(oc[pool_idx], rc[pool_idx])
    ens_tune = ens_score(oc[tune_idx], rc[tune_idx])
    ens_test = ens_score(ot, rt)

    calib_emb_pool = embed([calib_texts[i] for i in pool_idx])
    calib_emb_tune = embed([calib_texts[i] for i in tune_idx])
    test_emb = embed(test_texts)

    cfg = NNPPIConfig(k=3, clip_range=True)

    print(f"\n=== {name} (n_test={len(yt)}) ===")
    report("original raw", ot, yt)
    report("reordered raw", rt, yt)
    report("ensemble raw", ens_test, yt)

    res_orig_nnppi = nn_ppi_apply(ot, test_emb, oc[pool_idx], calib_emb_pool, yc[pool_idx], cfg)
    report("original + NN-PPI", res_orig_nnppi.theta, yt)

    res_ens_nnppi = nn_ppi_apply(ens_test, test_emb, ens_pool, calib_emb_pool, yc[pool_idx], cfg)
    report("ensemble + NN-PPI", res_ens_nnppi.theta, yt)

    # tune gate thresholds for ensemble+ours on tune_idx
    best = None
    for pt in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for st in [0.2, 0.4, 0.6, 0.99]:
            th = theta_gated_localreg(ens_tune, calib_emb_tune, ens_pool, calib_emb_pool, yc[pool_idx],
                                       purity_thresh=pt, sim_thresh=st)
            wr = np.mean((th>=0.5).astype(int) != yc[tune_idx])
            if best is None or wr < best[0]:
                best = (wr, pt, st)
    _, best_pt, best_st = best
    theta_ens_ours = theta_gated_localreg(ens_test, test_emb, ens_pool, calib_emb_pool, yc[pool_idx],
                                           purity_thresh=best_pt, sim_thresh=best_st)
    report(f"ensemble + ours (gate+localreg, pt={best_pt} st={best_st})", theta_ens_ours, yt)

    theta_orig_ours = theta_gated_localreg(ot, test_emb, oc[pool_idx], calib_emb_pool, yc[pool_idx],
                                            purity_thresh=best_pt, sim_thresh=best_st)
    report("original + ours (gate+localreg)", theta_orig_ours, yt)
