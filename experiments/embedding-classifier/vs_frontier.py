# -*- coding: utf-8 -*-
"""Does the embedding classifier actually close the gap to frontier (Sonnet)?
Evaluate all three -- Gemma raw, embedding SVM, Sonnet -- on the IDENTICAL
50 CLEF claims used in the earlier frontier spot-check
(../prompt-reorder-ablation/frontier_sample.json / frontier_scores.json).
"""
import json
import numpy as np
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

with open("../prompt-reorder-ablation/frontier_sample.json", encoding="utf-8") as f:
    sample = json.load(f)
with open("../prompt-reorder-ablation/frontier_scores.json", encoding="utf-8") as f:
    sonnet_scores = np.array(json.load(f))

labels = np.array([s["label"] for s in sample])
gemma = np.array([s["gemma_score"] for s in sample])
texts50 = [s["text"] for s in sample]

# train the embedding SVM on the FULL clef calib set (same as today's headline result)
calib = load("../../results/clef_calib_scores.jsonl")
Xc_txt = [r["text"] for r in calib]
yc = np.array([r["label"] for r in calib])
Xc = embed(Xc_txt)
X50 = embed(texts50)

svm = SVC(class_weight="balanced", probability=True, random_state=0)
svm.fit(Xc, yc)
p_svm = svm.predict_proba(X50)[:, 1]

def report(name, scores, labels):
    pred = (scores >= 0.5).astype(int)
    wrong = pred != labels
    print(f"  {name:30s} wrong={wrong.sum():2d}/50 ({wrong.mean():.1%})  F1={f1_score(labels,pred,average='weighted'):.3f}")
    return pred

print("=== Same 50 CLEF claims: Gemma raw vs embedding-SVM vs Sonnet (frontier proxy) ===")
pred_gemma = report("Gemma 3 4B (raw)", gemma, labels)
pred_svm = report("Gemma calib + embedding SVM", p_svm, labels)
pred_sonnet = report("Claude Sonnet 5 (raw, frontier)", sonnet_scores, labels)

print("\n=== overlap analysis ===")
correct_gemma = pred_gemma == labels
correct_svm = pred_svm == labels
correct_sonnet = pred_sonnet == labels
print(f"  SVM fixed  (gemma wrong -> svm correct): {(~correct_gemma & correct_svm).sum()}")
print(f"  SVM broke  (gemma correct -> svm wrong): {(correct_gemma & ~correct_svm).sum()}")
print(f"  SVM matches Sonnet's correctness on:     {(correct_svm == correct_sonnet).sum()}/50")
print(f"  cases where SVM right but Sonnet wrong:   {(correct_svm & ~correct_sonnet).sum()}")
print(f"  cases where Sonnet right but SVM wrong:   {(~correct_svm & correct_sonnet).sum()}")

gap_before = ((~correct_gemma).sum() - (~correct_sonnet).sum())
gap_after = ((~correct_svm).sum() - (~correct_sonnet).sum())
print(f"\n  gap to Sonnet in wrong-count: Gemma {(~correct_gemma).sum()} vs Sonnet {(~correct_sonnet).sum()} (gap={gap_before})")
print(f"  gap to Sonnet in wrong-count: SVM   {(~correct_svm).sum()} vs Sonnet {(~correct_sonnet).sum()} (gap={gap_after})")
if gap_before > 0:
    print(f"  gap closed: {100*(gap_before-gap_after)/gap_before:.0f}%")
