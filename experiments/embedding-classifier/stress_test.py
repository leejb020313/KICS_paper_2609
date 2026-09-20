# -*- coding: utf-8 -*-
"""Stress-test the embedding-classifier result before trusting it.

1. Trivial baselines (majority class, majority within a naive keyword rule)
   -- is 78% wrong-rate just a weak baseline that anything beats?
2. Multi-seed stability of the SVM result (different train subsamples of calib)
3. McNemar significance vs the LLM baseline
4. Literal leakage re-check: are calib/test texts byte-identical to anything,
   near-duplicate (e.g. same claim reworded), or is the embedding model itself
   somehow trained on this data (it isn't -- MiniLM is generic, pretrained
   long before these annotations existed, but let's at least check near-dupes)
"""
import json
import numpy as np
from math import comb
from sklearn.svm import SVC
from sklearn.metrics import f1_score

def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("parse_ok"):
                    rows.append(r)
    return rows

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
def embed(texts):
    return np.asarray(model.encode(texts, show_progress_bar=False, normalize_embeddings=True))

def mcnemar_exact(correct_a, correct_b):
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n = only_a + only_b
    if n == 0: return only_a, only_b, 1.0
    k = min(only_a, only_b)
    p = min(1.0, 2*sum(comb(n,i)*0.5**n for i in range(0,k+1)))
    return only_a, only_b, p

for name in ["clef", "claimbuster"]:
    calib = load(f"../../results/{name}_calib_scores.jsonl")
    test = load(f"../../results/{name}_test_scores.jsonl")

    Xc_txt = [r["text"] for r in calib]
    yc = np.array([r["label"] for r in calib])
    Xt_txt = [r["text"] for r in test]
    yt = np.array([r["label"] for r in test])
    ot = np.array([r["confidence_score"] for r in test])

    print(f"\n{'='*70}\n{name} (n_calib={len(yc)}, n_test={len(yt)})\n{'='*70}")

    # 1. trivial baselines
    majority = yc.mean() >= 0.5
    maj_wrong = (yt != int(majority)).mean()
    print(f"[trivial] majority-class baseline: predict all-{int(majority)}, wrong={maj_wrong:.3f}")

    llm_wrong = (ot >= 0.5).astype(int) != yt
    print(f"[baseline] LLM verbalized score:    wrong={llm_wrong.mean():.3f}")

    # 2. near-duplicate check between calib and test (exact-substring, cheap proxy)
    calib_set = set(Xc_txt)
    exact_dupe = sum(1 for t in Xt_txt if t in calib_set)
    print(f"[leak check] exact text duplicates calib<->test: {exact_dupe}/{len(Xt_txt)}")

    # 3. multi-seed stability: bootstrap-resample CALIB (not test), refit, eval on the SAME fixed test
    Xc = embed(Xc_txt)
    Xt = embed(Xt_txt)
    wrongs = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(yc), size=len(yc), replace=True)   # bootstrap resample of calib
        svm = SVC(class_weight="balanced", probability=False, random_state=seed)
        svm.fit(Xc[idx], yc[idx])
        pred = svm.predict(Xt)
        wrongs.append((pred != yt).mean())
    wrongs = np.array(wrongs)
    print(f"[stability] SVM wrong-rate over 8 bootstrap-resampled calib fits: "
          f"mean={wrongs.mean():.3f}  std={wrongs.std():.3f}  min={wrongs.min():.3f}  max={wrongs.max():.3f}")

    # 4. significance: single fit (no resampling) vs LLM baseline, McNemar on TEST
    svm = SVC(class_weight="balanced", probability=False, random_state=0)
    svm.fit(Xc, yc)
    pred = svm.predict(Xt)
    svm_wrong = pred != yt
    print(f"[point estimate] SVM (full calib, seed=0) wrong={svm_wrong.mean():.3f}  F1={f1_score(yt,pred,average='weighted'):.3f}")

    correct_llm = ~llm_wrong
    correct_svm = ~svm_wrong
    only_llm, only_svm, p = mcnemar_exact(correct_llm, correct_svm)
    print(f"[McNemar] LLM-only-correct={only_llm}  SVM-only-correct={only_svm}  exact p-value={p:.2e}")
