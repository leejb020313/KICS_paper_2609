# -*- coding: utf-8 -*-
"""Literature-grounded alternative: instead of post-hoc calibrating the
frozen LLM's verbalized score (NN-PPI's whole premise), directly train a
lightweight classifier on sentence embeddings of the calib set.

Motivation (orx-lit-review, this session):
  - "Comparing Specialised Small and General LLMs on Text Classification:
    100 Labelled Samples to Achieve Break-Even Performance" (2402.12819):
    specialized small models need ~100 labels to match/beat zero-shot large
    LLMs.
  - "Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot
    Generative AI Models in Text Classification" (2406.08660): compares
    ChatGPT/Claude Opus zero-shot directly against fine-tuned small models,
    fine-tuning wins consistently.
  - "Language Models for Text Classification: Is In-Context Learning
    Enough?" (2403.17661): same conclusion.

We have 1,314-2,406 labeled calib examples -- an order of magnitude past the
~100 break-even point in that literature. This has never been used to TRAIN
anything in this repo; it has only ever served as a kNN retrieval pool for
NN-PPI. This script trains directly on it and bypasses the LLM's quantized
verbalized score entirely.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

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

for name in ["clef", "claimbuster"]:
    calib = load(f"../../results/{name}_calib_scores.jsonl")
    test = load(f"../../results/{name}_test_scores.jsonl")

    Xc_txt = [r["text"] for r in calib]
    yc = np.array([r["label"] for r in calib])
    Xt_txt = [r["text"] for r in test]
    yt = np.array([r["label"] for r in test])
    ot = np.array([r["confidence_score"] for r in test])  # for comparison

    Xc = embed(Xc_txt)
    Xt = embed(Xt_txt)

    print(f"\n=== {name}: embedding-trained classifiers (calib n={len(yc)}, test n={len(yt)}) ===")
    baseline_wrong = ((ot >= 0.5).astype(int) != yt).mean()
    print(f"  [baseline] LLM verbalized score @ 0.5: wrong={baseline_wrong:.3f}")

    for cname, clf in [
        ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("LogisticRegression(C=0.1)", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)),
        ("SVM(rbf)", SVC(class_weight="balanced", probability=False)),
        ("MLP(64)", MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=0)),
    ]:
        cv_acc = cross_val_score(clf, Xc, yc, cv=5, scoring="accuracy").mean()
        clf.fit(Xc, yc)
        pred = clf.predict(Xt)
        wrong = (pred != yt).mean()
        f1 = f1_score(yt, pred, average="weighted")
        print(f"  {cname:28s} calib-CV acc={cv_acc:.3f}  TEST wrong={wrong:.3f} ({100*(baseline_wrong-wrong)/baseline_wrong:+.1f}% rel vs baseline)  F1={f1:.3f}")
