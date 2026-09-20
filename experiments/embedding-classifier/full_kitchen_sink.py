# -*- coding: utf-8 -*-
"""Combine EVERYTHING found today into one meta-ensemble:
  - embedding SVM (calibrated proba)      <- today's big win
  - original LLM verbalized score
  - reordered-prompt LLM score
  - NN-PPI (paper formula) on the original score
  - our gate+local-regression on the original score
  - logprob digit-mode score
  - logprob yesno-mode logit

All features computed honestly: NN-PPI/gate use a calib POOL as neighbor
source and a separate TUNE slice for gate-threshold selection (same protocol
as ../prompt-reorder-ablation/investigate10_final.py); the meta-learner is fit
on calib and evaluated once on the untouched test split.
"""
import json, sys
import numpy as np
sys.path.insert(0, "../../src")
from nnppi.calibration import NNPPIConfig, nn_ppi_apply
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from math import comb

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
        purity = max(y.mean(), 1-y.mean())
        max_sim = nb_sims.max()
        p_trust = np.clip((purity-0.5)/max(purity_thresh-0.5,1e-6), 0, 1)
        s_trust = np.clip((max_sim-0.1)/max(sim_thresh-0.1,1e-6), 0, 1)
        lam = p_trust * s_trust
        if x.std() < 1e-6:
            a, b = np.mean(y-x), 1.0
        else:
            X = np.vstack([x, np.ones_like(x)]).T
            XtX = X.T @ X + ridge*np.eye(2)
            coef = np.linalg.solve(XtX, X.T @ y)
            b, a = coef[0], coef[1]
        a_shrunk = lam*a; b_shrunk = 1.0 + lam*(b-1.0)
        theta[i] = np.clip(a_shrunk + b_shrunk*raw_scores[i], 0, 1)
    return theta

def mcnemar_exact(correct_a, correct_b):
    only_a = int(np.sum(correct_a & ~correct_b)); only_b = int(np.sum(~correct_a & correct_b))
    n = only_a+only_b
    if n == 0: return only_a, only_b, 1.0
    k = min(only_a, only_b)
    return only_a, only_b, min(1.0, 2*sum(comb(n,i)*0.5**n for i in range(0,k+1)))

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
def embed(texts):
    return np.asarray(model.encode(texts, show_progress_bar=False, normalize_embeddings=True))

def report(name, scores, labels):
    pred = (scores>=0.5).astype(int); wrong = pred != labels
    print(f"  {name:38s} wrong={wrong.mean():.3f} ({wrong.sum()}/{len(labels)})  F1={f1_score(labels,pred,average='weighted'):.3f}")
    return pred

for name in ["clef", "claimbuster"]:
    orig_c = load(f"../../results/{name}_calib_scores.jsonl")
    orig_t = load(f"../../results/{name}_test_scores.jsonl")
    reord_c = load(f"../prompt-reorder-ablation/results/{name}_calib_scores_reordered.jsonl")
    reord_t = load(f"../prompt-reorder-ablation/results/{name}_test_scores_reordered.jsonl")
    digit_c = load(f"../logprob-channel/results/{name}_calib_digit.jsonl")
    digit_t = load(f"../logprob-channel/results/{name}_test_digit.jsonl")
    yesno_c = load(f"../logprob-channel/results/{name}_calib_yesno.jsonl")
    yesno_t = load(f"../logprob-channel/results/{name}_test_yesno.jsonl")

    ctexts = [t for t in orig_c if t in reord_c and t in digit_c and t in yesno_c]
    ttexts = [t for t in orig_t if t in reord_t and t in digit_t and t in yesno_t]

    yc = np.array([orig_c[t]["label"] for t in ctexts])
    oc = np.array([orig_c[t]["confidence_score"] for t in ctexts])
    rc = np.array([reord_c[t]["confidence_score"] for t in ctexts])
    dc = np.array([digit_c[t]["confidence_score"] for t in ctexts])
    ylc = np.array([(yesno_c[t].get("detail") or {}).get("logit", 0.0) for t in ctexts])

    yt = np.array([orig_t[t]["label"] for t in ttexts])
    ot = np.array([orig_t[t]["confidence_score"] for t in ttexts])
    rt = np.array([reord_t[t]["confidence_score"] for t in ttexts])
    dt = np.array([digit_t[t]["confidence_score"] for t in ttexts])
    ylt = np.array([(yesno_t[t].get("detail") or {}).get("logit", 0.0) for t in ttexts])

    Xc_emb = embed(ctexts)
    Xt_emb = embed(ttexts)

    # pool/tune split of calib for NN-PPI neighbor source + gate threshold tuning
    rng = np.random.default_rng(0)
    n = len(yc); perm = rng.permutation(n); n_pool = int(0.7*n)
    pool_idx, tune_idx = perm[:n_pool], perm[n_pool:]
    POOL_RAW, POOL_EMB, POOL_LAB = oc[pool_idx], Xc_emb[pool_idx], yc[pool_idx]

    cfg = NNPPIConfig(k=3, clip_range=True)
    nnppi_tune = nn_ppi_apply(oc[tune_idx], Xc_emb[tune_idx], POOL_RAW, POOL_EMB, POOL_LAB, cfg).theta
    nnppi_calib_full = nn_ppi_apply(oc, Xc_emb, POOL_RAW, POOL_EMB, POOL_LAB, cfg).theta  # for meta-features
    nnppi_test = nn_ppi_apply(ot, Xt_emb, POOL_RAW, POOL_EMB, POOL_LAB, cfg).theta

    best = None
    for pt in [0.6,0.7,0.8,0.9,1.0]:
        for st in [0.2,0.4,0.6,0.99]:
            th = theta_gated_localreg(oc[tune_idx], Xc_emb[tune_idx], POOL_RAW, POOL_EMB, POOL_LAB, purity_thresh=pt, sim_thresh=st)
            wr = np.mean((th>=0.5).astype(int) != yc[tune_idx])
            if best is None or wr < best[0]: best = (wr, pt, st)
    _, bpt, bst = best
    gate_calib_full = theta_gated_localreg(oc, Xc_emb, POOL_RAW, POOL_EMB, POOL_LAB, purity_thresh=bpt, sim_thresh=bst)
    gate_test = theta_gated_localreg(ot, Xt_emb, POOL_RAW, POOL_EMB, POOL_LAB, purity_thresh=bpt, sim_thresh=bst)

    svm = SVC(class_weight="balanced", probability=True, random_state=0)
    svm.fit(Xc_emb, yc)
    svm_calib = svm.predict_proba(Xc_emb)[:,1]
    svm_test = svm.predict_proba(Xt_emb)[:,1]

    print(f"\n{'='*74}\n{name} (n_calib={len(yc)}, n_test={len(yt)})\n{'='*74}")
    report("LLM original", ot, yt)
    report("embedding SVM alone", svm_test, yt)

    feat_names = ["svm", "orig", "reordered", "nnppi", "gate+localreg", "digit", "yesno_logit"]
    Xc_meta = np.column_stack([svm_calib, oc, rc, nnppi_calib_full, gate_calib_full, dc, ylc])
    Xt_meta = np.column_stack([svm_test, ot, rt, nnppi_test, gate_test, dt, ylt])
    meta = LogisticRegression(max_iter=3000)
    meta.fit(Xc_meta, yc)
    p_meta = meta.predict_proba(Xt_meta)[:,1]
    pred_kitchen = report("KITCHEN SINK (all 7 signals, logistic meta)", p_meta, yt)
    print(f"    meta coefs {dict(zip(feat_names, np.round(meta.coef_[0],3)))}")

    correct_llm = (ot>=0.5).astype(int) == yt
    correct_kitchen = pred_kitchen == yt
    only_llm, only_k, p = mcnemar_exact(correct_llm, correct_kitchen)
    print(f"    McNemar vs LLM baseline: only_llm_correct={only_llm} only_kitchen_correct={only_k} p={p:.2e}")

    correct_svmalone = (svm_test>=0.5).astype(int) == yt
    only_s, only_k2, p2 = mcnemar_exact(correct_svmalone, correct_kitchen)
    print(f"    McNemar vs SVM-alone:    only_svm_correct={only_s} only_kitchen_correct={only_k2} p={p2:.2e}")
