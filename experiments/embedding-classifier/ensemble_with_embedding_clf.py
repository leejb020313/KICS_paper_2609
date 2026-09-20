# -*- coding: utf-8 -*-
"""Combine the embedding-trained classifier (trained_classifier.py) with every
signal already computed today: original LLM score, reordered-prompt score,
and (where available) our NN-PPI gate+local-regression output.

Trains SVM with probability=True (Platt-scaled) so it has a continuous score
usable in an ensemble, not just a hard label.
"""
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("parse_ok"):
                    rows.append(r)
    return {r["text"]: r for r in rows}

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
def embed(texts):
    return np.asarray(model.encode(texts, show_progress_bar=False, normalize_embeddings=True))

def report(name, scores, labels):
    pred = (scores >= 0.5).astype(int)
    wrong = pred != labels
    print(f"  {name:42s} wrong={wrong.sum():4d}/{len(labels)} ({wrong.mean():.1%})  F1={f1_score(labels,pred,average='weighted'):.3f}")

for name in ["clef", "claimbuster"]:
    orig_calib = load(f"../../results/{name}_calib_scores.jsonl")
    orig_test = load(f"../../results/{name}_test_scores.jsonl")
    reord_calib = load(f"../prompt-reorder-ablation/results/{name}_calib_scores_reordered.jsonl")
    reord_test = load(f"../prompt-reorder-ablation/results/{name}_test_scores_reordered.jsonl")

    calib_texts = [t for t in orig_calib if t in reord_calib]
    test_texts = [t for t in orig_test if t in reord_test]

    yc = np.array([orig_calib[t]["label"] for t in calib_texts])
    oc = np.array([orig_calib[t]["confidence_score"] for t in calib_texts])
    rc = np.array([reord_calib[t]["confidence_score"] for t in calib_texts])
    yt = np.array([orig_test[t]["label"] for t in test_texts])
    ot = np.array([orig_test[t]["confidence_score"] for t in test_texts])
    rt = np.array([reord_test[t]["confidence_score"] for t in test_texts])

    Xc = embed(calib_texts)
    Xt = embed(test_texts)

    print(f"\n=== {name} (n_calib={len(yc)}, n_test={len(yt)}) ===")
    report("LLM original score", ot, yt)
    report("LLM prompt-reorder ensemble (orig+reord+diff)", None, yt) if False else None

    # 1. embedding classifier, calibrated probabilities
    svm = SVC(class_weight="balanced", probability=True, random_state=0)
    svm.fit(Xc, yc)
    p_svm = svm.predict_proba(Xt)[:, 1]
    report("embedding SVM (calibrated proba)", p_svm, yt)
    print(f"    AUC: {roc_auc_score(yt, p_svm):.3f}")

    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(Xc, yc)
    p_lr = logreg.predict_proba(Xt)[:, 1]
    report("embedding LogisticRegression", p_lr, yt)

    # 2. stack: embedding-SVM proba + orig LLM score + reordered LLM score
    #    -> logistic meta-learner, calib-fit
    p_svm_calib = svm.predict_proba(Xc)[:, 1]
    Xc_meta = np.column_stack([p_svm_calib, oc, rc])
    Xt_meta = np.column_stack([p_svm, ot, rt])
    meta = LogisticRegression(max_iter=2000)
    meta.fit(Xc_meta, yc)
    p_meta = meta.predict_proba(Xt_meta)[:, 1]
    report("STACK (embedding-SVM + orig + reordered LLM score)", p_meta, yt)
    print(f"    meta coefficients (svm, orig, reordered): {np.round(meta.coef_[0], 3)}")

    # 3. embedding classifier alone vs stack: does the LLM score add anything
    #    once the embedding classifier is already there?
    only_svm_wrong = (p_svm >= 0.5).astype(int) != yt
    stack_wrong = (p_meta >= 0.5).astype(int) != yt
    print(f"    embedding-SVM alone wrong={only_svm_wrong.mean():.3f}  STACK wrong={stack_wrong.mean():.3f}")
